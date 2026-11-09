"""可组合的离线奖励。诊断奖励只验证更新，外部指标通过显式函数注入。"""

import math
from collections.abc import Callable, Mapping

import torch

from omni.evaluation.sft import transcription_errors
from omni.training.grpo_rollout import Rollout, decode_codes

BUILTIN_REWARDS = {"token_value", "audio_value", "text_match", "completeness", "asr_match"}


def validate_rewards(settings: dict, evaluators: Mapping | None = None) -> dict[str, float]:
    """未知指标或无效权重立即报错，不用缺省零分冒充指标已执行。"""
    if not isinstance(settings, dict) or set(settings) != {"weights"}:
        raise ValueError("Reward 配置需要 weights 映射。")
    weights = settings["weights"]
    known = BUILTIN_REWARDS | set(evaluators or {})
    if (
        not isinstance(weights, dict)
        or not weights
        or weights.keys() - known
        or any(not isinstance(w, (float, int)) or not math.isfinite(w) for w in weights.values())
        or not any(weights.values())
    ):
        raise ValueError("Reward 需要已实现的指标、有限权重及至少一个非零权重。")
    return weights


def reward_resources(settings: dict, root) -> dict:
    """仅显式选择 ASR 奖励时加载已有本地资源；普通 Tiny 不加载编码器。"""
    if not settings["grpo"]["reward"]["weights"].get("asr_match"):
        return {}
    from transformers import AutoTokenizer, MimiModel

    from omni.evaluation.sft import load_local_asr
    from omni.inference.codec import MimiCodec

    media = root / settings["assets"]["reference_root"] / "model"
    return dict(
        tokenizer=AutoTokenizer.from_pretrained(
            root / settings["assets"]["qwen_dir"], local_files_only=True
        ),
        codec=MimiCodec(
            MimiModel.from_pretrained(media / "mimi", local_files_only=True), num_codebooks=8
        ),
        asr=load_local_asr(media / "SenseVoiceSmall"),
    )


@torch.no_grad()
def reward_score(
    trajectory: Rollout,
    config,
    settings: dict,
    context: dict | None = None,
    *,
    tokenizer=None,
    codec=None,
    asr: Callable | None = None,
    evaluators: Mapping[str, Callable[[dict], float]] | None = None,
) -> tuple[float, dict[str, float]]:
    """返回 scalar reward 和每项原始分数；任何无效项拒绝整个 group。

    自定义函数接收 text_ids、完整 audio_codes [8,F]、可选 waveform [samples]、
    sampling_rate、解码 text、context 和停止状态，可接语义/音质等已审计指标。
    """
    weights = validate_rewards(settings, evaluators)
    context = {} if context is None else context
    text_ids = trajectory.text_ids[trajectory.text_mask].detach().cpu()
    codes = decode_codes(trajectory, config).detach().cpu()
    payload = dict(
        text_ids=text_ids,
        audio_codes=codes,
        waveform=None,
        sampling_rate=None,
        text=None
        if tokenizer is None
        else tokenizer.decode(text_ids.tolist(), skip_special_tokens=True),
        context=context,
        text_done=trajectory.text_done,
        audio_done=trajectory.audio_done,
        truncated=trajectory.truncated,
    )
    if codec is not None and codes.shape[-1]:
        waveform = codec.decode(codes[None]).detach().cpu().flatten()
        if not waveform.numel() or not waveform.isfinite().all():
            raise ValueError("奖励音频解码为空或非有限。")
        payload.update(waveform=waveform, sampling_rate=codec.sampling_rate)

    scores = {}
    for name, weight in weights.items():
        if weight == 0:
            continue
        if name == "token_value":
            # Token ID 仅提供稳定的 Tiny 数学信号，不能解释为文本质量。
            value = float(text_ids.float().mean()) / max(1, config.vocab_size - 1)
        elif name == "audio_value":
            active = trajectory.audio_mask & trajectory.audio_ids.lt(config.audio_codebook_size)
            raw = trajectory.audio_ids[active]
            value = (
                float(raw.float().mean()) / max(1, config.audio_codebook_size - 1)
                if raw.numel()
                else 0.0
            )
        elif name == "completeness":
            value = float(trajectory.text_done)
            if trajectory.mode != "text":
                value = (value + float(trajectory.audio_done and codes.shape[-1] > 0)) / 2
        elif name == "text_match":
            target = context.get("target_ids")
            if target is None or not len(target):
                raise ValueError("text_match 需要可信的目标 Token 序列。")
            value = float(torch.equal(text_ids, torch.as_tensor(target).long().cpu()))
        elif name == "asr_match":
            target = context.get("target_ids")
            if asr is None or codec is None or tokenizer is None or target is None:
                raise ValueError("asr_match 需要本地 ASR、Mimi、Tokenizer 与可信目标文本。")
            reference = tokenizer.decode(torch.as_tensor(target).tolist(), skip_special_tokens=True)
            if not reference.strip():
                raise ValueError("ASR Reward 的目标文本不能为空。")
            if payload["waveform"] is None:
                value = 0.0
            else:
                hypothesis = asr(payload["waveform"], payload["sampling_rate"])
                value = max(0.0, 1.0 - transcription_errors(reference, hypothesis)["cer"])
        else:
            value = float(evaluators[name](payload))
        if not math.isfinite(value):
            raise ValueError(f"Reward {name} 返回非有限值。")
        scores[name] = value

    total = sum(weights[name] * value for name, value in scores.items())
    if not math.isfinite(total):
        raise ValueError("Reward 加权结果非有限。")
    return total, scores
