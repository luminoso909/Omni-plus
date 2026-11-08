"""Qwen Omni 组合层。媒体编码器与 Mimi 在模型外运行。"""

from collections.abc import Iterator
from dataclasses import dataclass

from torch import Tensor, nn
from transformers.cache_utils import Cache

from .bridge import FixedBridge as FixedBridge
from .bridge import bridge_spec, build_bridge
from .config import OmniPlusConfig
from .encoders.model_encoders import Projector
from .model_omni import ModalFeatures, OmniOutput
from .model_omni import OmniPlusOmni as LegacyOmni
from .talker.model_talker import TalkerModule
from .thinker.model_thinker import PastKeyValues
from .thinker.qwen_backend import QwenBackend


@dataclass
class OmniCache:
    """一轮对话的两个独立状态。Qwen 原地追加，Talker 返回新 KV。"""

    thinker: Cache
    talker: PastKeyValues


class OmniPlusOmni(nn.Module):
    def __init__(
        self, thinker: QwenBackend, config: OmniPlusConfig, *, bridge: nn.Module | None = None
    ) -> None:
        super().__init__()
        native = thinker.model.config
        if (config.hidden_size, config.num_hidden_layers, config.vocab_size) != (
            native.hidden_size,
            native.num_hidden_layers,
            native.vocab_size,
        ):
            raise ValueError("Omni 配置的 Thinker 尺寸与原生 Qwen 不一致。")
        self.bridge_spec = bridge_spec(config)
        markers = [*config.image_ids, *config.audio_ids]
        if len(config.image_ids) != 1 or len(config.audio_ids) != 1 or len(set(markers)) != 2:
            raise ValueError("图像和音频必须各有一个不同的占位 ID。")

        self.config = config
        self.thinker = thinker.requires_grad_(False).eval()
        self.vision_proj = Projector(config.image_hidden_size, native.hidden_size)
        self.audio_proj = Projector(config.audio_hidden_size, native.hidden_size)
        self.bridge = bridge if bridge is not None else build_bridge(config, self.bridge_spec)
        self.talker = TalkerModule(config, projected_condition=True)

    def train(self, mode: bool = True) -> "OmniPlusOmni":
        super().train(mode)
        self.thinker.eval()
        return self

    def _inject_features(
        self,
        tokens: Tensor,
        hidden: Tensor,
        features: ModalFeatures | None,
        marker: int,
        projector: Projector,
    ) -> Tensor:
        # 逐样本按占位顺序替换，保留每条媒体的真实长度及投影梯度。
        if features is None:
            if tokens.eq(marker).any():
                raise ValueError("模态占位缺少对应特征。")
            return hidden
        if len(features) != tokens.shape[0]:
            raise ValueError("模态特征列表必须与 batch 一一对应。")

        hidden = hidden.clone()
        for row, feature in enumerate(features):
            slots = tokens[row].eq(marker)
            count = 0 if feature is None else feature.shape[0]
            if slots.sum().item() != count:
                raise ValueError("模态特征长度与占位符数量不一致。")
            if feature is not None:
                hidden[row, slots] = projector(feature.to(next(projector.parameters()))).to(hidden)
        return hidden

    def forward(
        self,
        input_ids: Tensor,
        audio_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: OmniCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        audio_features: ModalFeatures | None = None,
        image_features: ModalFeatures | None = None,
        spk_emb: Tensor | None = None,
    ) -> OmniOutput:
        """Long input_ids [B,T]、audio_ids [B,C,T]；Bool/Long mask [B,P+T]。

        特征按 batch 排列，形状分别为 [N,512] 和 [N,768]，仅在 prefill 注入。
        文本与音频 logits 分别为 [B,K,V]、[B,C,K,A]；标签由外部按同位置计算。
        """
        batch, seq = input_ids.shape
        config = self.config
        thinker_past = None if past_key_values is None else past_key_values.thinker
        talker_past = None if past_key_values is None else past_key_values.talker
        past_len = 0 if thinker_past is None else thinker_past.get_seq_length()
        total = past_len + seq
        if seq == 0 or total > config.max_position_embeddings:
            raise ValueError("输入为空或超出 Talker 上下文长度。")
        if past_key_values is not None and not use_cache:
            raise ValueError("传入 Cache 时需要 use_cache=True。")
        if talker_past is not None and (
            len(talker_past) != len(self.talker.layers) or talker_past[0][0].shape[-2] != past_len
        ):
            raise ValueError("Talker 与 Qwen 的物理 Cache 长度必须一致。")
        audio_shape = (batch, config.num_audio_codebooks, seq)
        if audio_ids is not None and audio_ids.shape != audio_shape:
            raise ValueError("audio_ids 必须与当前输入 [B,C,T] 对齐。")
        if attention_mask is not None and attention_mask.shape != (batch, total):
            raise ValueError("attention_mask 必须覆盖 Cache 和当前输入。")

        # Qwen 使用逻辑 RoPE 位置，实际 Cache 仍按物理序列位置追加。
        mask = (
            input_ids.new_ones((batch, total))
            if attention_mask is None
            else attention_mask.to(input_ids.device)
        )
        positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask.bool(), 1)[:, -seq:]
        hidden = self.thinker.embed_tokens(input_ids)
        if past_len == 0:
            hidden = self._inject_features(
                input_ids, hidden, image_features, config.image_ids[0], self.vision_proj
            )
            hidden = self._inject_features(
                input_ids, hidden, audio_features, config.audio_ids[0], self.audio_proj
            )
        elif any(input_ids.eq(marker).any() for marker in (*config.audio_ids, *config.image_ids)):
            raise ValueError("媒体占位必须与特征一起在 prefill 输入。")

        # 冻结权重仍允许梯度从 Qwen 输出回到 Projector。
        layers = self.bridge_spec["layers"]
        selection = (
            {"hidden_layer": layers[0]} if len(layers) == 1 else {"hidden_layers": tuple(layers)}
        )
        thinker = self.thinker(
            inputs_embeds=hidden,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=thinker_past,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **selection,
        )
        if len(layers) == 1:
            bridge_input = thinker.selected_hidden_state.to(self.talker.text_scale)
        else:
            bridge_input = tuple(
                state.to(self.talker.text_scale) for state in thinker.selected_hidden_states
            )
        condition = self.bridge(bridge_input, mask[:, -seq:])
        if audio_ids is None:
            audio_ids = input_ids.new_full(audio_shape, config.audio_pad_token)

        # 旧 Collator 用零向量补齐缺失的 speaker；没有标记就不建该分支的梯度图。
        if spk_emb is not None and not audio_ids[:, 0].eq(config.audio_spk_token).any():
            spk_emb = None
        audio_logits, talker_cache, aux = self.talker(
            condition, audio_ids, positions, mask, talker_past, use_cache, spk_emb
        )
        return OmniOutput(
            logits=thinker.logits,
            audio_logits=audio_logits[:, :, -logits_to_keep:],
            aux_loss=aux,
            past_key_values=OmniCache(thinker.past_key_values, talker_cache) if use_cache else None,
        )

    # 复用已有采样和八码本状态机，使 STOP、延迟与旧路径共用同一实现。
    _sample_token = LegacyOmni._sample_token
    _filter_top_p = LegacyOmni._filter_top_p

    def stream_generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        audio_features: ModalFeatures | None = None,
        image_features: ModalFeatures | None = None,
        spk_emb: Tensor | None = None,
        ref_codes: Tensor | None = None,
        max_new_tokens: int = 256,
        use_cache: bool = True,
    ) -> Iterator[tuple[Tensor | None, Tensor | None]]:
        """单样本非 thinking 贪心流。输出文本 token [1,1] 与可选声码帧 [1,C,1]。"""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[-1] == 0:
            raise ValueError("流式生成需要单条非空 prompt。")
        valid_len = input_ids.shape[-1]
        context = self.config.max_position_embeddings
        if valid_len > context or max_new_tokens > context - valid_len + 1:
            raise ValueError("流式生成预算超出物理上下文长度。")
        if attention_mask is not None:
            valid = attention_mask.bool()
            if (
                valid.shape != input_ids.shape
                or not valid[0, -1]
                or (valid[:, :-1] & ~valid[:, 1:]).any()
            ):
                raise ValueError("流式生成仅支持左 padding。")
            valid_len = int(valid.sum())
        ref_len = 0 if ref_codes is None else ref_codes.shape[-1]
        if ref_len + int(spk_emb is not None) > valid_len:
            raise ValueError("有效 prompt 长度不足以容纳参考声码和说话人标记。")

        yield from LegacyOmni.stream_generate(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            audio_temperature=0,
            attention_mask=attention_mask,
            audio_features=[None] if audio_features is None else audio_features,
            image_features=[None] if image_features is None else image_features,
            spk_emb=spk_emb,
            ref_codes=ref_codes,
            use_cache=use_cache,
        )
