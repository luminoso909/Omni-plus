"""本地 Omni 对话数据、延迟码本与批量整理。

labels/audio_labels 已在数据侧右移一次，可直接与 Omni 输出计算 CE；不要再次 shift。
原始媒体只交给外部 audio_encoder/vision_encoder 回调，返回预编码 [tokens, features]。
本模块不加载编码器、不下载资源，也不将全零特征误判为缺失模态。
"""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from omni.models.config import OmniPlusConfig


def _load_records(source: str | Path | Sequence) -> list[dict]:
    # 路径只读取本地 Parquet；内存记录不经过磁盘或网络
    if isinstance(source, (str, Path)):
        source = [Path(path.strip()) for path in str(source).split(",")]
    if len(source) and isinstance(source[0], (str, Path)):
        return [record for path in source for record in pq.read_table(path).to_pylist()]
    return [dict(record) for record in source]


class OmniDataset(Dataset):
    """单样本：文本/标签/mask [T]，音频码本/标签 [C,T]，模态特征 [N,D]。

    max_length 是 shift 前的时间轴上限；training 才会抽取对话轮次或扰动历史输入。
    原始媒体回调负责解码、重采样与编码，返回 CPU 特征 Tensor，不在这里创建模型。
    """

    def __init__(
        self,
        source: str | Path | Sequence[Mapping[str, Any]] | Sequence[str | Path],
        tokenizer: Any,
        config: OmniPlusConfig,
        max_length: int = 1200,
        training: bool = False,
        scheduled_sampling: float = 0.0,
        seed: int = 0,
        audio_encoder: Callable[[Any], Tensor] | None = None,
        vision_encoder: Callable[[Any], Tensor] | None = None,
    ) -> None:
        if max_length < 2 or not 0 <= scheduled_sampling <= 1:
            raise ValueError("max_length 必须 >= 2，scheduled_sampling 必须在 [0, 1] 内。")

        # 数据与配置；默认评估模式，不随机裁轮或扰动输入
        self.records = _load_records(source)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.training = training
        self.scheduled_sampling = scheduled_sampling if training else 0.0
        self.seed, self.epoch = seed, 0
        self.audio_encoder, self.vision_encoder = audio_encoder, vision_encoder
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = config.pad_token_id if config.pad_token_id is not None else 0

        # 占位符必须是对应的单个特殊 Token，才能与模型注入位置一致
        for token, ids in (
            (config.audio_special_token, config.audio_ids),
            (config.image_special_token, config.image_ids),
        ):
            if tokenizer.encode(token, add_special_tokens=False) != ids[:1]:
                raise ValueError(f"特殊 Token {token!r} 与 config 中的 id 不一致。")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        # 每个 epoch 可变换训练样本，同时不依赖 worker 的调用顺序
        self.epoch = epoch

    def _turn_features(
        self, record: dict, kind: str, users: int, total_users: int
    ) -> list[Tensor | None]:
        features = record.get(f"{kind}_features")
        if features is not None:
            # 单轮可直接传二维特征；多轮按 user 顺序传入，每轮可展平多张图
            if isinstance(features, (Tensor, np.ndarray)):
                features = [features] if features.ndim == 2 else list(features)
            elif (
                features
                and isinstance(features[0], list)
                and features[0]
                and isinstance(features[0][0], (int, float))
            ):
                features = [features]
            result = []
            for value in list(features)[:users]:
                if value is None:
                    result.append(None)
                else:
                    value = torch.as_tensor(value, dtype=torch.float32)
                    result.append(value.reshape(-1, value.shape[-1]))
            return result + [None] * (users - len(result))

        # bytes/path 的解码、重采样和编码由外部回调负责，不在这里安装媒体框架
        key, encoder = (
            ("question_audios", self.audio_encoder)
            if kind == "audio"
            else ("image_bytes", self.vision_encoder)
        )
        payloads = record.get(key)
        if payloads is None:
            return [None] * users
        if isinstance(payloads, (bytes, str, Path)):
            payloads = [payloads]
        elif kind == "image" and total_users == 1:
            if not payloads or not isinstance(payloads[0], (list, tuple)):
                payloads = [payloads]

        result = []
        for payload in list(payloads)[:users]:
            items = payload if kind == "image" and isinstance(payload, (list, tuple)) else [payload]
            items = [item for item in items if item is not None and bool(item)]
            if not items:
                result.append(None)
                continue
            if encoder is None:
                raise ValueError(f"{key} 是原始媒体，需要传入对应 encoder 回调。")

            encoded = [torch.as_tensor(encoder(item)).float() for item in items]
            result.append(torch.cat(encoded) if kind == "image" else encoded[0])
        return result + [None] * (users - len(result))

    def _expand_marker(self, content: str, feature: Tensor | None, kind: str) -> str:
        marker = f"<{kind}>"
        special = getattr(self.config, f"{kind}_special_token")
        if feature is None:
            if marker in content or special in content:
                raise ValueError(f"{kind} 占位符缺少对应的编码特征。")
            return content

        # 明确长度对应，保留多图位置，禁止无声截断特征或改变消息结构
        length, count = feature.shape[0], content.count(marker)
        if count:
            if length % count:
                raise ValueError(f"{kind} 特征数不能均分给各占位符。")
            return content.replace(marker, special * (length // count))
        count = content.count(special)
        if count == length:
            return content
        if count == 1:
            return content.replace(special, special * length)
        if count:
            raise ValueError(f"{kind} 特征数与显式占位符数不一致。")
        return special * length + "\n" + content

    def _audio_codes(self, value: Any) -> Tensor | None:
        if value is None or len(value) == 0:
            return None

        # Parquet 的扁平码本以 frame-major 排列；二维输入明确为 [codebooks, frames]
        codes = torch.as_tensor(value, dtype=torch.long)
        count = self.config.num_audio_codebooks
        if codes.ndim == 1:
            if codes.numel() % count:
                raise ValueError("扁平音频 codes 的长度必须能被码本数整除。")
            codes = codes.reshape(-1, count).T
        if codes.ndim != 2 or codes.shape[0] != count:
            raise ValueError("音频 codes 必须是 [codebooks, frames]。")
        if ((codes < 0) | (codes >= self.config.audio_codebook_size)).any():
            raise ValueError("原始音频 codes 只能包含声学词表 id，stop 由 dataset 添加。")
        return codes

    def _corrupt_inputs(
        self,
        tokens: Tensor,
        audio: Tensor,
        labels: Tensor,
        audio_labels: Tensor,
        rng: random.Random,
    ) -> None:
        if not self.scheduled_sampling:
            return

        # 只扰动监督区的历史输入，不改标签、媒体位置与 speaker/ref 前缀
        forbidden = set(self.config.audio_ids + self.config.image_ids + self.config.think_end_ids)
        forbidden.update(getattr(self.tokenizer, "all_special_ids", []))
        forbidden.update([self.config.bos_token_id, self.pad_id])
        eos = self.config.eos_token_id
        forbidden.update(eos if isinstance(eos, list) else [eos])
        text_choices = [token for token in range(len(self.tokenizer)) if token not in forbidden]
        for pos in range(tokens.shape[0]):
            if (
                text_choices
                and pos > 0
                and labels[pos - 1] != -100
                and tokens[pos].item() not in forbidden
                and rng.random() < self.scheduled_sampling
            ):
                tokens[pos] = rng.choice(text_choices)
            for codebook in range(audio.shape[0]):
                if (
                    pos > 0
                    and audio_labels[codebook, pos - 1] != -100
                    and audio[codebook, pos] < self.config.audio_codebook_size
                    and rng.random() < self.scheduled_sampling
                ):
                    audio[codebook, pos] = rng.randrange(self.config.audio_codebook_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回文本 [T]、音频码本 [C,T] 和各模态特征 [N,D]；标签已 shift。"""
        record = self.records[index]
        messages = record["conversations"]
        messages = json.loads(messages) if isinstance(messages, str) else messages
        total_users = sum(message["role"] == "user" for message in messages)
        assistants = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if not assistants:
            raise ValueError("对话至少需要一个 assistant 回合。")

        # 训练可抽取完整轮次前缀；评估始终使用最后一轮，原始记录保持不变
        rng = random.Random(self.seed + index + self.epoch * 104729)
        target = rng.randrange(len(assistants)) if self.training else len(assistants) - 1
        messages = [dict(message) for message in messages[: assistants[target] + 1]]
        users = sum(message["role"] == "user" for message in messages)
        audio_features = self._turn_features(record, "audio", users, total_users)
        image_features = self._turn_features(record, "image", users, total_users)
        user = 0
        for message in messages:
            if message["role"] == "user":
                content = self._expand_marker(message["content"], audio_features[user], "audio")
                message["content"] = self._expand_marker(content, image_features[user], "image")
                user += 1

        # 用同一个 chat template 确定最后 assistant 正文起点，不硬编码角色分隔符
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prefix = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        if not prefix or ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("chat template 的 generation prefix 与完整对话不一致。")
        assistant_start, text_end = len(prefix_ids), len(ids)
        audio_start = assistant_start
        think_end = self.config.think_end_ids
        if think_end:
            for pos in range(assistant_start, text_end - len(think_end) + 1):
                if ids[pos : pos + len(think_end)] == think_end:
                    audio_start = pos + len(think_end)
                    break

        # 最后 assistant 的声学码本与 stop；短文本也要为音频尾部留足时间轴
        answers = record.get("answer_audios")
        answer = None
        if answers is not None and len(answers):
            if isinstance(answers, (Tensor, np.ndarray)) or isinstance(answers[0], (int, float)):
                answers = [answers]
            if target < len(answers):
                answer = self._audio_codes(answers[target])
        count = self.config.num_audio_codebooks
        audio_end = audio_start + count + answer.shape[-1] + 1 if answer is not None else 0
        if audio_end > self.max_length:
            raise ValueError("max_length 会截断音频 codes 或 stop，请增大长度。")
        length = min(max(text_end, audio_end), self.max_length)
        if assistant_start >= length:
            raise ValueError("max_length 未保留任何 assistant 监督 token，请增大长度。")

        ids = ids[:length] + [self.pad_id] * max(0, length - text_end)
        tokens = torch.tensor(ids, dtype=torch.long)
        labels = torch.full((length,), -100, dtype=torch.long)
        labels[assistant_start : min(text_end, length)] = tokens[
            assistant_start : min(text_end, length)
        ]
        audio = torch.full((count, length), self.config.audio_pad_token, dtype=torch.long)
        audio_labels = torch.full_like(audio, -100)

        # 参考音频和 speaker 条件放在正文前，只作为输入，不计入监督
        speaker = record.get("spk_emb")
        speaker = (
            None
            if speaker is None or len(speaker) == 0
            else torch.as_tensor(speaker, dtype=torch.float32)
        )
        if speaker is not None and speaker.shape != (self.config.spk_emb_size,):
            raise ValueError("spk_emb 维度与 config.spk_emb_size 不一致。")
        reference = self._audio_codes(record.get("ref_audios"))
        if audio_start >= length and (reference is not None or speaker is not None):
            raise ValueError("max_length 截断了参考音频或 speaker 条件位置，请增大长度。")

        ref_len = (
            0
            if reference is None
            else min(reference.shape[-1], max(0, audio_start - int(speaker is not None)))
        )
        ref_start = audio_start - ref_len
        if ref_len:
            audio[:, ref_start:audio_start] = reference[:, -ref_len:]
        if speaker is not None and ref_start > 0 and ref_start <= length:
            audio[:, ref_start - 1] = self.config.audio_spk_token
        if answer is not None:
            for codebook in range(count):
                start = audio_start + codebook + 1
                codes = torch.cat(
                    (answer[codebook], answer.new_tensor([self.config.audio_stop_token]))
                )
                audio[codebook, start : start + len(codes)] = codes
                audio_labels[codebook, start : start + len(codes)] = codes

        # 数据侧只做一次 next-token shift，mask 根据有效时间轴而非 pad id 推断
        tokens, audio = tokens[:-1].clone(), audio[:, :-1].clone()
        labels, audio_labels = labels[1:], audio_labels[:, 1:]
        features = {}
        for kind, turns in (("audio", audio_features), ("image", image_features)):
            valid = [feature for feature in turns if feature is not None]
            feature = torch.cat(valid) if valid else None
            expected = 0 if feature is None else feature.shape[0]
            if tokens.eq(getattr(self.config, f"{kind}_ids")[0]).sum().item() != expected:
                raise ValueError("max_length 截断了模态占位符，或占位符与特征长度不一致。")
            features[f"{kind}_features"] = feature
        self._corrupt_inputs(tokens, audio, labels, audio_labels, rng)
        return dict(
            input_ids=tokens,
            audio_ids=audio,
            labels=labels,
            audio_labels=audio_labels,
            attention_mask=torch.ones_like(tokens),
            spk_emb=speaker,
            **features,
        )


class OmniCollator:
    def __init__(self, config: OmniPlusConfig) -> None:
        self.config = config

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """文本整理为 [B,T]、码本为 [B,C,T]，模态特征保留逐样本列表。"""
        # 只对 token 时间轴右 padding；特征保留 list 和 None，避免打乱混合 batch
        config = self.config
        batch = {}
        pads = {
            "input_ids": config.pad_token_id if config.pad_token_id is not None else 0,
            "labels": -100,
            "attention_mask": 0,
            "audio_ids": config.audio_pad_token,
            "audio_labels": -100,
        }
        for key, padding in pads.items():
            values = [
                sample[key].T if key.startswith("audio_") else sample[key] for sample in samples
            ]
            batch[key] = pad_sequence(values, batch_first=True, padding_value=padding)
            if key.startswith("audio_"):
                batch[key] = batch[key].transpose(1, 2)
        for kind in ("audio", "image"):
            batch[f"{kind}_features"] = [sample[f"{kind}_features"] for sample in samples]
        batch["spk_emb"] = torch.stack(
            [
                sample["spk_emb"]
                if sample["spk_emb"] is not None
                else torch.zeros(config.spk_emb_size)
                for sample in samples
            ]
        )
        return batch
