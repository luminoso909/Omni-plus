"""Qwen 原生聊天模板与 R2 音文监督。输出可直接交给旧 OmniCollator。"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from omni.models.config import OmniPlusConfig

from .audio_alignment import delay_codes

type TurnFeatures = Tensor | Sequence[Tensor | Sequence[Tensor] | None] | None


def audio_codes(value: Any, config: OmniPlusConfig) -> Tensor | None:
    """返回 CPU long [8,F]。Parquet 中的扁平数组以 frame-major 排列。"""
    if value is None:
        return None
    codes = torch.as_tensor(value, device="cpu")
    if codes.numel() == 0:
        return None
    if codes.is_floating_point() or codes.is_complex() or codes.dtype == torch.bool:
        raise ValueError("Mimi codes 必须是整数。")
    codes = codes.long()
    count = config.num_audio_codebooks
    if codes.ndim == 1 and codes.numel() % count == 0:
        codes = codes.reshape(-1, count).T
    if codes.ndim != 2 or codes.shape[0] != count:
        raise ValueError("Mimi codes 必须是 [codebooks,frames] 或完整的扁平帧。")
    if ((codes < 0) | (codes >= config.audio_codebook_size)).any():
        raise ValueError("原始 Mimi codes 不得含 PAD、STOP 或其它越界 ID。")
    return codes


def _expand_media(
    messages: list[dict[str, Any]],
    kind: str,
    token: str,
    width: int,
    features: TurnFeatures,
) -> Tensor | None:
    """仅替换 user 的占位符。每轮可传入一段或显式分开的多段特征。"""
    marker = f"<{kind}>"
    users = [message for message in messages if message["role"] == "user"]
    if features is None:
        features = [None] * len(users)
    elif isinstance(features, Tensor):
        features = [features]
    if len(features) != len(users):
        raise ValueError(f"{kind} 特征须按 user 轮次提供，缺失轮次填 None。")

    valid = []
    for message, value in zip(users, features, strict=True):
        spans = [] if value is None else [value] if isinstance(value, Tensor) else value
        parts = message["content"].split(marker)
        if len(parts) > 1 and len(parts) - 1 != len(spans):
            raise ValueError(f"{kind} 占位符数须等于该轮特征段数。")
        replacements = []
        for span in spans:
            if (
                span.ndim != 2
                or span.shape[0] == 0
                or span.shape[1] != width
                or not span.is_floating_point()
                or not span.isfinite().all()
            ):
                raise ValueError(f"{kind} 特征必须是非空有限浮点 [tokens,{width}]。")
            valid.append(span.detach().cpu())
            replacements.append(token * span.shape[0])

        if len(parts) > 1:
            message["content"] = parts[0] + "".join(
                replacement + part
                for replacement, part in zip(replacements, parts[1:], strict=True)
            )
        elif replacements:
            message["content"] = "".join(replacements) + "\n" + message["content"]
    return torch.cat(valid) if valid else None


class QwenOmniFormatter:
    """在 CPU 构建 R2 的非 thinking 训练样本和推理前缀。"""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: OmniPlusConfig, max_length: int = 1200
    ) -> None:
        self.tokenizer, self.config, self.max_length = tokenizer, config, max_length
        pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
        eos_ids = config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        if (
            pad is None
            or eos is None
            or pad != config.pad_token_id
            or eos not in eos_ids
            or max_length < 2
        ):
            raise ValueError("需要匹配的 Qwen PAD/EOS 配置及 max_length >= 2。")
        self.pad, self.eos = pad, eos
        ids = [pad, eos]
        for token, expected in (
            (config.audio_special_token, config.audio_ids),
            (config.image_special_token, config.image_ids),
        ):
            if (
                len(expected) != 1
                or tokenizer.encode(token, add_special_tokens=False) != expected
                or expected[0] not in tokenizer.all_special_ids
            ):
                raise ValueError(f"占位符 {token!r} 必须是现有 tokenizer 的单个特殊 ID。")
            ids.extend(expected)
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= config.vocab_size:
            raise ValueError("媒体占位符必须与文本 PAD/EOS 各自独立，且位于 Qwen 词表内。")

    def _prepare_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        audio_features: TurnFeatures,
        image_features: TurnFeatures,
    ) -> tuple[list[dict[str, Any]], dict[str, Tensor | None]]:
        config = self.config
        prepared = [dict(message) for message in messages]
        reserved = (
            config.audio_special_token,
            config.image_special_token,
            "<|im_start|>",
            self.tokenizer.pad_token,
            self.tokenizer.eos_token,
        )
        for message in prepared:
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("对话内容必须为字符串，媒体通过显式特征传入。")
            if any(token and token in content for token in reserved):
                raise ValueError("原始文本不得直接包含用于模板边界或模态注入的保留 Token。")
            if message["role"] != "user" and ("<audio>" in content or "<image>" in content):
                raise ValueError("媒体占位符只允许出现在 user 消息中。")
        return prepared, {
            "audio_features": _expand_media(
                prepared,
                "audio",
                config.audio_special_token,
                config.audio_hidden_size,
                audio_features,
            ),
            "image_features": _expand_media(
                prepared,
                "image",
                config.image_special_token,
                config.image_hidden_size,
                image_features,
            ),
        }

    def _tokens(self, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False
        )
        if not ids or min(ids) < 0 or max(ids) >= self.config.vocab_size:
            raise ValueError("聊天模板产生空序列或超出 Qwen 词表的 Token。")
        return ids

    def _check_slots(self, prefix: list[int], features: dict[str, Tensor | None]) -> None:
        for kind, token in (
            ("audio", self.config.audio_ids[0]),
            ("image", self.config.image_ids[0]),
        ):
            rows = features[f"{kind}_features"]
            if prefix.count(token) != (0 if rows is None else rows.shape[0]):
                raise ValueError(f"模板中 {kind} 占位符数量与有效特征长度不一致。")

    def prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        audio_features: TurnFeatures = None,
        image_features: TurnFeatures = None,
    ) -> dict[str, Tensor | None]:
        """推理前缀 input_ids/attention_mask 为 [P]，特征按占位顺序展平。"""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("推理前缀必须以 user 消息结束。")
        messages, features = self._prepare_messages(messages, audio_features, image_features)
        prefix = self._tokens(messages, generation=True)
        self._check_slots(prefix, features)
        if len(prefix) > self.max_length:
            raise ValueError("max_length 不足以容纳完整提示和输入模态。")
        ids = torch.tensor(prefix, dtype=torch.long)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), **features)

    def sample(
        self,
        record: Mapping[str, Any],
        *,
        audio_features: TurnFeatures = None,
        image_features: TurnFeatures = None,
        answer_codes: Tensor | None = None,
        reference_codes: Tensor | None = None,
        spk_emb: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """监督最后 assistant 一轮，文本 [T]，音频 [8,T]，标签只 shift 一次。

        参考音频 [8,F] 和 speaker [D] 仅接受调用方已核实的独立条件。
        原记录可含与目标同源的参考音频与说话人特征，因此不自动读取这两列。
        """
        config = self.config
        messages = record["conversations"]
        messages = json.loads(messages) if isinstance(messages, str) else messages
        if len(messages) < 2 or [msg["role"] for msg in messages[-2:]] != ["user", "assistant"]:
            raise ValueError("训练样本必须以 user、assistant 一轮结束。")
        messages, features = self._prepare_messages(messages, audio_features, image_features)
        prefix = self._tokens(messages[:-1], generation=True)
        self._check_slots(prefix, features)
        full = self._tokens(messages, generation=False)
        if full[: len(prefix)] != prefix or self.eos not in full[len(prefix) :]:
            raise ValueError("完整对话与 Qwen generation prefix 或 EOS 不一致。")

        # 训练监督到文本 EOS，与生成后继续音频时只输入 text PAD 对齐。
        start = len(prefix)
        end = full.index(self.eos, start) + 1
        if end - start < 2 or any(
            token in full[start:end]
            for token in (self.pad, config.audio_ids[0], config.image_ids[0])
        ):
            raise ValueError("最后 assistant 目标为空或包含输入侧特殊标记。")

        raw_codes = answer_codes
        if raw_codes is None:
            answers = record.get("answer_audios")
            if answers is not None and len(answers):
                if isinstance(answers, Tensor) or isinstance(answers[0], int):
                    raw_codes = answers
                else:
                    turn = sum(message["role"] == "assistant" for message in messages) - 1
                    if turn < len(answers):
                        raw_codes = answers[turn]
        codes = audio_codes(raw_codes, config)
        delayed = (
            None
            if codes is None
            else delay_codes(codes, config.audio_pad_token, config.audio_stop_token, start=1)
        )
        length = end if delayed is None else max(end, start + delayed.shape[-1])
        if length > self.max_length:
            raise ValueError("max_length 会截断文本、音频码或 STOP，请增大长度。")

        tokens = torch.tensor(full[:end] + [self.pad] * (length - end), dtype=torch.long)
        labels = torch.full_like(tokens, -100)
        labels[start:end] = tokens[start:end]
        audio = torch.full((config.num_audio_codebooks, length), config.audio_pad_token)
        audio_labels = torch.full_like(audio, -100)
        if delayed is not None:
            stop = start + delayed.shape[-1]
            audio[:, start:stop] = delayed
            audio_labels[:, start:stop] = delayed.masked_fill(
                delayed == config.audio_pad_token, -100
            )

        # 参考声码与 speaker 只作为 prompt 条件，不参与监督。
        speaker = spk_emb.detach().float().cpu() if spk_emb is not None else None
        if speaker is not None and (
            speaker.shape != (config.spk_emb_size,) or not speaker.isfinite().all()
        ):
            raise ValueError("spk_emb 必须是维度匹配的有限向量。")
        reference = audio_codes(reference_codes, config)
        ref_start = start - (0 if reference is None else reference.shape[1])
        if ref_start < int(speaker is not None):
            raise ValueError("完整提示的空间不足，无法放入参考音频或 speaker 标记。")
        if reference is not None:
            if codes is not None:
                shorter, longer = sorted((reference, codes), key=lambda x: x.shape[-1])
                windows = longer.unfold(-1, shorter.shape[-1], 1)
                if windows.eq(shorter[:, None, :]).all(dim=-1).all(dim=0).any():
                    raise ValueError("参考音频与回答码本存在完整子段重合。")
            audio[:, ref_start:start] = reference
        if speaker is not None:
            audio[:, ref_start - 1] = config.audio_spk_token

        return dict(
            input_ids=tokens[:-1],
            audio_ids=audio[:, :-1],
            labels=labels[1:],
            audio_labels=audio_labels[:, 1:],
            attention_mask=torch.ones_like(tokens[:-1]),
            spk_emb=speaker,
            **features,
        )
