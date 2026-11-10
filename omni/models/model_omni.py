"""Thinker → Bridge → Talker；编码器与音频 Codec 在模型外运行。

文本与音频码本分别传入；音频/图像特征按 batch 顺序排列，None 表示无该模态。
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .config import OmniPlusConfig
from .encoders.model_encoders import Projector
from .talker.model_talker import AudioCodes, TalkerModule
from .thinker.model_thinker import AttentionMask, OmniPlusForCausalLM, PastKeyValues

type ModalFeatures = list[Float[Tensor, "tokens features"] | None]


@dataclass
class OmniOutput(MoeCausalLMOutputWithPast):
    audio_logits: Float[Tensor, "batch codebooks seq audio_vocab"] | None = None


class OmniPlusOmni(OmniPlusForCausalLM):
    def __init__(self, config: OmniPlusConfig) -> None:
        if not 0 <= config.bridge_layer < config.num_hidden_layers:
            raise ValueError("bridge_layer 必须指向有效的 Thinker Block。")
        if (
            config.num_audio_codebooks <= 0
            or config.talker_adapter_rank <= 0
            or not 0
            < config.audio_codebook_size
            <= config.audio_stop_token
            < config.audio_vocab_size
        ):
            raise ValueError("检查音频码本数、低秩维度与 audio_codebook_size/stop/vocab。")

        # 直接保留 model/lm_head 权重命名，不额外注册 thinker 别名
        super().__init__(config)
        self.audio_proj = Projector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = Projector(config.image_hidden_size, config.hidden_size)
        self.talker = TalkerModule(config)
        self.post_init()

    def _inject_features(
        self,
        input_ids: Int[Tensor, "batch seq"],
        hidden: Float[Tensor, "batch seq hidden"],
        features: ModalFeatures | None,
        marker_ids: list[int],
        projector: Projector,
    ) -> Float[Tensor, "batch seq hidden"]:
        if features is None:
            return hidden
        if len(features) != input_ids.shape[0]:
            raise ValueError("模态特征列表必须与 batch 一一对应。")

        # 按占位符顺序替换，None 保持该样本原样；全零特征仍然有效
        hidden = hidden.clone()
        for row, feature in enumerate(features):
            if feature is None:
                continue
            mask = input_ids[row] == marker_ids[0]
            if mask.sum().item() != feature.shape[0]:
                raise ValueError("模态特征长度必须与占位符数量一致，不自动截断或压缩。")

            hidden[row, mask] = projector(feature.to(hidden))
        return hidden

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        audio_ids: AudioCodes | None = None,
        attention_mask: AttentionMask | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        audio_features: ModalFeatures | None = None,
        image_features: ModalFeatures | None = None,
        spk_emb: Float[Tensor, "batch speaker_hidden"] | None = None,
    ) -> OmniOutput:
        config = self.config
        batch, seq = input_ids.shape
        n_thinker = config.num_hidden_layers
        n_layers = n_thinker + len(self.talker.layers)

        if past_key_values is not None and len(past_key_values) != n_layers:
            raise ValueError("Cache 必须包含全部 Thinker 与 Talker 层。")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep 必须 >= 0。")
        if audio_ids is not None and audio_ids.shape != (batch, config.num_audio_codebooks, seq):
            raise ValueError("audio_ids 必须是 [batch, num_audio_codebooks, seq]。")

        # 文本输入嵌入；只在 prefill 注入音频与视觉特征
        hidden = self.model.embed_tokens(input_ids)
        past_len = past_key_values[0][0].shape[-2] if past_key_values else 0
        if past_len == 0:
            hidden = self._inject_features(
                input_ids, hidden, audio_features, config.audio_ids, self.audio_proj
            )
            hidden = self._inject_features(
                input_ids, hidden, image_features, config.image_ids, self.vision_proj
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden.device)

        # Thinker 共享前向，按需返回供 Bridge 使用的单层特征
        thinker_past = None if past_key_values is None else past_key_values[:n_thinker]
        thinker = self.model.forward_embeddings(
            hidden,
            attention_mask=attention_mask,
            past_key_values=thinker_past,
            use_cache=use_cache,
            bridge_layer=config.bridge_layer,
        )
        text_logits = self.lm_head(thinker.last_hidden_state[:, -logits_to_keep:])

        # Talker 使用同位置的 Bridge 条件和历史音频码本
        if audio_ids is None:
            audio_ids = input_ids.new_full(
                (batch, config.num_audio_codebooks, seq), config.audio_pad_token
            )
        talker_past = None if past_key_values is None else past_key_values[n_thinker:]
        audio_logits, talker_cache, talker_aux = self.talker(
            thinker.bridge_hidden_state,
            audio_ids,
            thinker.position_ids,
            attention_mask,
            talker_past,
            use_cache,
            spk_emb,
        )

        return OmniOutput(
            logits=text_logits,
            audio_logits=audio_logits[:, :, -logits_to_keep:],
            aux_loss=thinker.aux_loss + talker_aux,
            past_key_values=thinker.past_key_values + talker_cache if use_cache else None,
        )

    def _sample_token(
        self, scores: Float[Tensor, "batch vocab"], temperature: float, top_k: int, top_p: float
    ) -> Int[Tensor, "batch 1"]:
        if temperature == 0:
            return scores.argmax(-1, keepdim=True)

        # 文本与音频共用温度、Top-K 与 Top-P 采样逻辑
        scores = scores.float() / temperature
        if top_k > 0:
            cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, -torch.inf)
        scores = self._filter_top_p(scores, top_p)
        return torch.multinomial(scores.softmax(-1), num_samples=1)

    def generate(
        self,
        input_ids: Int[Tensor, "batch seq"],
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        audio_temperature: float = 0.2,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        attention_mask: AttentionMask | None = None,
        audio_features: ModalFeatures | None = None,
        image_features: ModalFeatures | None = None,
        spk_emb: Float[Tensor, "batch speaker_hidden"] | None = None,
        ref_codes: AudioCodes | None = None,
        open_thinking: bool = False,
        use_cache: bool = True,
    ) -> tuple[Int[Tensor, "batch total_seq"], AudioCodes]:
        """model.eval() 后调用；返回完整文本（含 prompt）和去掉延迟的声学码本。"""
        # 分别收集文本与完整音频帧，音频尾部不会覆盖已生成的文本
        text, frames = [input_ids], []
        for token, frame in self.stream_generate(
            input_ids,
            max_new_tokens,
            temperature,
            audio_temperature,
            top_k,
            top_p,
            repetition_penalty,
            attention_mask,
            audio_features,
            image_features,
            spk_emb,
            ref_codes,
            open_thinking,
            use_cache,
        ):
            if token is not None:
                text.append(token)
            if frame is not None:
                frames.append(frame)

        codes = (
            torch.cat(frames, dim=-1)
            if frames
            else input_ids.new_empty((1, self.config.num_audio_codebooks, 0))
        )
        return torch.cat(text, dim=-1), codes

    @torch.inference_mode()
    def stream_generate(
        self,
        input_ids: Int[Tensor, "batch seq"],
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        audio_temperature: float = 0.2,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        attention_mask: AttentionMask | None = None,
        audio_features: ModalFeatures | None = None,
        image_features: ModalFeatures | None = None,
        spk_emb: Float[Tensor, "batch speaker_hidden"] | None = None,
        ref_codes: AudioCodes | None = None,
        open_thinking: bool = False,
        use_cache: bool = True,
    ) -> Iterator[tuple[Int[Tensor, "batch 1"] | None, AudioCodes | None]]:
        """单样本流式生成：每步返回新文本 token 和可选的一帧音频 codes。

        Prompt 支持左 padding，attention_mask 中至少有一个有效 token。
        第 i 路码本延迟 i 步，首路在文本生成第 1 步后启动；最多运行 max_new_tokens 步。
        这里输出离散 codes，波形解码与实时播放由外部 Codec 负责。
        """
        config = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("流式生成需要 batch=1 且非空的 prompt。")
        if (
            max_new_tokens < 0
            or temperature < 0
            or audio_temperature < 0
            or top_k < 0
            or not 0 < top_p <= 1
            or repetition_penalty <= 0
        ):
            raise ValueError("检查生成长度、温度、Top-K/Top-P 和重复惩罚。")
        if open_thinking and not config.think_end_ids:
            raise ValueError("open_thinking=True 需要非空 think_end_ids。")

        # 只接受左 padding，使文本采样和音频条件都对齐到有效 prompt 末尾
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        start = 0
        if attention_mask is not None:
            if mask.shape != input_ids.shape:
                raise ValueError("attention_mask 必须与 prompt 形状一致。")
            mask = mask.to(input_ids.device)
            valid = mask.bool()
            if not valid[0, -1] or (valid[:, :-1] & ~valid[:, 1:]).any():
                raise ValueError("流式生成只支持左 padding，且 prompt 至少有一个有效 token。")
            start = input_ids.shape[1] - int(valid.sum().item())

        # 初始化历史码本，可在 prompt 末尾放参考音频和说话人标记
        count = config.num_audio_codebooks
        audio = input_ids.new_full((1, count, input_ids.shape[1]), config.audio_pad_token)
        reserve = int(spk_emb is not None)
        slots = input_ids.shape[1] - start - reserve
        ref_len = 0 if ref_codes is None else min(ref_codes.shape[-1], slots)
        if ref_codes is not None and ref_len > 0:
            audio[:, :, -ref_len:] = ref_codes[:, :, -ref_len:]
        if spk_emb is not None:
            audio[:, :, -ref_len - 1] = config.audio_spk_token
        eos = [] if config.eos_token_id is None else config.eos_token_id
        eos_ids = torch.as_tensor(eos, device=input_ids.device, dtype=input_ids.dtype).reshape(-1)
        pad_id = config.pad_token_id if config.pad_token_id is not None else 0

        # text_done 控制文本输出，stopped 分别控制各音频码本
        text_done = False
        stopped = [False] * count
        audio_start = None if open_thinking else 1
        generated, history = [], []
        cache = None

        for step in range(max_new_tokens):
            # Cache 模式仅输入最新 token；不缓存时重新计算完整前缀
            tokens = input_ids[:, -1:] if cache is not None else input_ids
            codes = audio[:, :, -1:] if cache is not None else audio
            output = self(
                tokens,
                audio_ids=codes,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=use_cache,
                logits_to_keep=1,
                audio_features=audio_features,
                image_features=image_features,
                spk_emb=spk_emb,
            )
            cache = output.past_key_values

            # 文本采样；结束后用 padding 维持音频解码的时间轴
            token = input_ids.new_full((1, 1), pad_id)
            if not text_done:
                scores = output.logits[:, -1].float().clone()
                if repetition_penalty != 1:
                    seen_ids = input_ids[:, start:]
                    seen = scores.gather(1, seen_ids)
                    seen = torch.where(
                        seen < 0, seen * repetition_penalty, seen / repetition_penalty
                    )
                    scores.scatter_(1, seen_ids, seen)

                # 新生成的文本不再插入输入模态占位符
                markers = list(config.audio_ids) if audio_features is not None else []
                markers += config.image_ids if image_features is not None else []
                scores[:, markers] = -torch.inf
                token = self._sample_token(scores, temperature, top_k, top_p)
                generated.append(int(token.item()))
                if (
                    audio_start is None
                    and generated[-len(config.think_end_ids) :] == config.think_end_ids
                ):
                    audio_start = step + 2

            # 各码本依次启动，停止后保持 pad；不采样 speaker/pad 等非声学标记
            next_audio = input_ids.new_full((1, count), config.audio_pad_token)
            for i in range(count):
                if audio_start is None or step < audio_start + i or stopped[i]:
                    continue
                scores = output.audio_logits[:, i, -1].float().clone()
                stop_score = scores[:, config.audio_stop_token].clone()
                scores[:, config.audio_codebook_size :] = -torch.inf
                scores[:, config.audio_stop_token] = stop_score
                code = self._sample_token(scores, audio_temperature, top_k, top_p)
                next_audio[:, i] = code[:, 0]
                stopped[i] = bool(code.item() == config.audio_stop_token)
            history.append(next_audio)

            # 按对角线恢复同一时刻的各路 codes，丢弃带 special token 的不完整帧
            frame = None
            if audio_start is not None and step >= audio_start + count - 1:
                frame = torch.stack(
                    [history[step - count + 1 + i][:, i] for i in range(count)], dim=1
                )
                frame = frame.unsqueeze(-1) if (frame < config.audio_codebook_size).all() else None
            yield (None if text_done else token), frame
            text_done = text_done or bool(torch.isin(token, eos_ids).any())
            if text_done and (audio_start is None or all(stopped)):
                break

            # 追加输入、音频历史与 attention mask，继续解码音频尾部
            input_ids = torch.cat((input_ids, token), dim=-1)
            audio = torch.cat((audio, next_audio.unsqueeze(-1)), dim=-1)
            mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
