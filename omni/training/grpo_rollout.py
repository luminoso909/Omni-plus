"""GRPO 的可重放音文动作轨迹；沿用 Omni forward 与八码本延迟协议。"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from omni.data.audio_alignment import restore_codes


@dataclass
class Rollout:
    """单条条件轨迹。T 为生成物理步数，C=8；forced PAD 不属于策略动作。"""

    prompt: dict
    text_ids: Tensor  # Long [T]，包含 EOS 和结束后的 forced PAD。
    audio_ids: Tensor  # Long [C,T]，包含独立 STOP 及延迟/结束 PAD。
    text_mask: Tensor  # Bool [T]，首次 EOS 自身为有效动作。
    audio_mask: Tensor  # Bool [C,T]，每路首次 STOP 自身为有效动作。
    old_logprob: Tensor  # Float [T]，行为策略有效动作的联合 logprob。
    policy_version: int
    mode: str
    temperature: float
    audio_temperature: float
    truncated: bool
    text_done: bool
    audio_done: bool

    @property
    def valid_mask(self) -> Tensor:
        if self.mode == "text":
            return self.text_mask
        if self.mode == "audio":
            return self.audio_mask.any(0)
        return self.text_mask | self.audio_mask.any(0)

    def to_record(self) -> dict:
        """仅保存可审查的动作、mask 与行为概率，不复制模型或媒体特征缓存。"""
        return dict(
            prompt_ids=self.prompt["input_ids"].detach().cpu().tolist(),
            text_ids=self.text_ids.detach().cpu().tolist(),
            audio_ids=self.audio_ids.detach().cpu().tolist(),
            text_mask=self.text_mask.detach().cpu().tolist(),
            audio_mask=self.audio_mask.detach().cpu().tolist(),
            old_logprob=self.old_logprob.detach().cpu().tolist(),
            policy_version=self.policy_version,
            mode=self.mode,
            temperature=self.temperature,
            audio_temperature=self.audio_temperature,
            truncated=self.truncated,
            text_done=self.text_done,
            audio_done=self.audio_done,
        )


@dataclass
class PolicyScores:
    logprob: Tensor  # Float [T]，按模式求和，未按码本数取平均。
    text_logprob: Tensor  # Float [T]，forced PAD 为零。
    audio_logprob: Tensor  # Float [C,T]，延迟/结束 PAD 为零。
    text_log_probs: Tensor  # Float [T,V_valid]，压缩合法 support 的完整 log 分布。
    audio_log_probs: Tensor  # Float [C,T,K+1]，真实声码 0..K-1 加 STOP。


def _support(config, device: torch.device) -> tuple[Tensor, Tensor]:
    forbidden = {config.pad_token_id, *config.audio_ids, *config.image_ids}
    text = torch.tensor(
        [token for token in range(config.vocab_size) if token not in forbidden], device=device
    )
    audio = torch.tensor(
        [*range(config.audio_codebook_size), config.audio_stop_token], device=device
    )
    if (
        not text.numel()
        or config.num_audio_codebooks != 8
        or config.audio_codebook_size < 1
        or not config.audio_codebook_size <= config.audio_pad_token < config.audio_vocab_size
        or config.audio_stop_token < config.audio_codebook_size
        or config.audio_stop_token == config.audio_pad_token
        or config.audio_stop_token >= config.audio_vocab_size
    ):
        raise ValueError("GRPO 需要非空文本 support、8 路声码及独立有效的 STOP/PAD。")
    return text.long(), audio.long()


def _log_probs(scores: Tensor, support: Tensor, temperature: float) -> Tensor:
    values = scores.index_select(-1, support)
    values = values if values.dtype == torch.float64 else values.float()
    if not values.isfinite().all():
        raise ValueError("策略 logits 必须有限。")
    result = (values / temperature).log_softmax(-1)
    if not result.isfinite().all():
        raise ValueError("温度缩放后的策略 logprob 必须有限。")
    return result


def _prompt(prompt: dict, config) -> dict:
    ids = prompt["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype != torch.long:
        raise ValueError("GRPO prompt 需要单条非空 Long [1,P]。")
    mask = prompt.get("attention_mask", torch.ones_like(ids)).bool()
    if mask.shape != ids.shape or not mask[0, -1] or (mask[:, :-1] & ~mask[:, 1:]).any():
        raise ValueError("GRPO prompt 只支持左 padding。")
    codes = prompt.get("audio_ids")
    if codes is None:
        codes = ids.new_full((1, 8, ids.shape[1]), config.audio_pad_token)
    if codes.shape != (1, 8, ids.shape[1]) or codes.dtype != torch.long:
        raise ValueError("prompt 声码需要 Long [1,8,P]。")

    # 轨迹条件固定，调用者之后修改原 batch 不会改变行为策略所见输入。
    copied = {}
    for name, value in prompt.items():
        if isinstance(value, Tensor):
            value = value.detach().clone()
        elif isinstance(value, list):
            value = [item.detach().clone() if isinstance(item, Tensor) else item for item in value]
        copied[name] = value
    return copied | dict(
        input_ids=ids.detach().clone(),
        attention_mask=mask.clone(),
        audio_ids=codes.detach().clone(),
    )


def _eos(config) -> set[int]:
    value = config.eos_token_id
    return set(value if isinstance(value, list) else [] if value is None else [value])


@torch.no_grad()
def rollout(
    model: nn.Module,
    prompt: dict,
    *,
    mode: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    audio_temperature: float = 1.0,
    policy_version: int = 0,
    use_cache: bool = True,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Rollout:
    """同一 eval 策略采样，行为分布为合法 support 上的温度 Softmax。"""
    if (
        mode not in {"text", "audio", "joint"}
        or type(max_new_tokens) is not int
        or max_new_tokens < 1
    ):
        raise ValueError("需要 text/audio/joint 模式及正整数生成预算。")
    if any(not math.isfinite(t) or t <= 0 for t in (temperature, audio_temperature)):
        raise ValueError("GRPO 采样温度必须为有限正数。")
    if top_k != 0 or top_p != 1:
        raise ValueError("GRPO 只接受 top_k=0、top_p=1 的完整 support 采样。")

    model.eval()  # Current 重算也保持 eval，关闭 dropout 而不关闭其 autograd。
    config = model.config
    prompt = _prompt(prompt, config)
    tokens, audio, mask = (prompt[key] for key in ("input_ids", "audio_ids", "attention_mask"))
    if tokens.shape[1] + max_new_tokens - 1 > config.max_position_embeddings:
        raise ValueError("GRPO 生成预算超出物理上下文。")
    text_support, audio_support = _support(config, tokens.device)
    features = {
        key: value
        for key, value in prompt.items()
        if key not in {"input_ids", "audio_ids", "attention_mask"}
    }
    text_done, stopped, cache = False, [False] * 8, None
    text_ids, audio_ids, text_masks, audio_masks, old = [], [], [], [], []

    for step in range(max_new_tokens):
        output = model(
            tokens[:, -1:] if cache is not None else tokens,
            audio_ids=audio[..., -1:] if cache is not None else audio,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=use_cache,
            logits_to_keep=1,
            **features,
        )
        cache = output.past_key_values
        active_text = not text_done
        token = tokens.new_tensor(config.pad_token_id)
        text_lp = output.logits.new_zeros((), dtype=torch.float32)
        if active_text:
            distribution = _log_probs(output.logits[0, -1], text_support, temperature)
            index = torch.multinomial(distribution.exp(), 1).squeeze(0)
            token, text_lp = text_support[index], distribution[index]
            text_done = int(token) in _eos(config)

        # 延迟/结束 PAD 是确定性控制动作，不进入概率；每路 STOP 仍是随机动作。
        next_audio = tokens.new_full((8,), config.audio_pad_token)
        active_audio = torch.zeros(8, dtype=torch.bool, device=tokens.device)
        audio_lp = text_lp.new_zeros(8)
        if mode != "text":
            for channel in range(8):
                if step < 1 + channel or stopped[channel]:
                    continue
                distribution = _log_probs(
                    output.audio_logits[0, channel, -1], audio_support, audio_temperature
                )
                index = torch.multinomial(distribution.exp(), 1).squeeze(0)
                next_audio[channel], audio_lp[channel] = audio_support[index], distribution[index]
                active_audio[channel] = True
                stopped[channel] = int(next_audio[channel]) == config.audio_stop_token
        joint = (text_lp if mode != "audio" else text_lp.new_zeros(())) + audio_lp.sum()
        text_ids.append(token)
        audio_ids.append(next_audio)
        text_masks.append(active_text)
        audio_masks.append(active_audio)
        old.append(joint)
        if text_done and (mode == "text" or all(stopped)):
            break

        tokens = torch.cat((tokens, token.reshape(1, 1)), -1)
        audio = torch.cat((audio, next_audio.reshape(1, 8, 1)), -1)
        mask = torch.cat((mask, mask.new_ones((1, 1))), -1)

    audio_done = mode == "text" or all(stopped)
    return Rollout(
        prompt=prompt,
        text_ids=torch.stack(text_ids),
        audio_ids=torch.stack(audio_ids, -1),
        text_mask=torch.tensor(text_masks, dtype=torch.bool, device=tokens.device),
        audio_mask=torch.stack(audio_masks, -1),
        old_logprob=torch.stack(old).detach(),
        policy_version=policy_version,
        mode=mode,
        temperature=temperature,
        audio_temperature=audio_temperature,
        truncated=not (text_done and audio_done),
        text_done=text_done,
        audio_done=audio_done,
    )


def rollout_group(model: nn.Module, prompt: dict, *, group_size: int, **sampling) -> list[Rollout]:
    if type(group_size) is not int or group_size < 2:
        raise ValueError("GRPO group_size 至少为 2。")
    return [rollout(model, prompt, **sampling) for _ in range(group_size)]


def _validate_trajectory(trajectory: Rollout, config) -> None:
    text, audio = trajectory.text_ids, trajectory.audio_ids
    if (
        text.ndim != 1
        or not text.numel()
        or audio.shape != (8, text.numel())
        or text.dtype != torch.long
        or audio.dtype != torch.long
        or trajectory.text_mask.shape != text.shape
        or trajectory.audio_mask.shape != audio.shape
        or trajectory.text_mask.dtype != torch.bool
        or trajectory.audio_mask.dtype != torch.bool
        or trajectory.mode not in {"text", "audio", "joint"}
        or any(
            not math.isfinite(t) or t <= 0
            for t in (trajectory.temperature, trajectory.audio_temperature)
        )
    ):
        raise ValueError("GRPO 动作、mask、模式或温度无效。")
    text_support, _ = _support(config, text.device)
    text_done, stopped = False, [False] * 8
    for step in range(text.numel()):
        if bool(trajectory.text_mask[step]) != (not text_done):
            raise ValueError("文本 action mask 与 EOS 不一致。")
        if not text_done:
            if not torch.isin(text[step], text_support):
                raise ValueError("文本动作超出合法 support。")
            text_done = int(text[step]) in _eos(config)
        elif text[step] != config.pad_token_id:
            raise ValueError("文本 EOS 后必须使用 forced PAD。")
        for channel in range(8):
            active = trajectory.mode != "text" and step >= 1 + channel and not stopped[channel]
            if bool(trajectory.audio_mask[channel, step]) != active:
                raise ValueError("声码 action mask 与 delay/STOP 不一致。")
            code = int(audio[channel, step])
            if active:
                if not 0 <= code < config.audio_codebook_size and code != config.audio_stop_token:
                    raise ValueError("声码动作超出合法 support。")
                stopped[channel] = code == config.audio_stop_token
            elif code != config.audio_pad_token:
                raise ValueError("延迟或 STOP 后必须使用 forced PAD。")
        finished = text_done and (trajectory.mode == "text" or all(stopped))
        if finished and step != text.numel() - 1:
            raise ValueError("已结束的轨迹不得追加伪动作。")
    if (
        trajectory.text_done != text_done
        or trajectory.audio_done != (trajectory.mode == "text" or all(stopped))
        or trajectory.truncated == finished
    ):
        raise ValueError("轨迹停止或截断标记与动作不一致。")


def score_rollout(model: nn.Module, trajectory: Rollout) -> PolicyScores:
    """一次 Teacher Forcing 重算 [T]；动作 t 只使用 prompt 与动作 <t。"""
    _validate_trajectory(trajectory, model.config)
    model.eval()
    prompt = _prompt(trajectory.prompt, model.config)
    text, audio = trajectory.text_ids, trajectory.audio_ids
    steps = text.numel()
    prompt["input_ids"] = torch.cat((prompt["input_ids"], text[None, :-1]), -1)
    prompt["audio_ids"] = torch.cat((prompt["audio_ids"], audio[None, :, :-1]), -1)
    prompt["attention_mask"] = torch.cat(
        (prompt["attention_mask"], prompt["attention_mask"].new_ones((1, steps - 1))), -1
    )
    output = model(**prompt, use_cache=False, logits_to_keep=steps)
    text_support, audio_support = _support(model.config, text.device)
    text_dist = _log_probs(output.logits[0], text_support, trajectory.temperature)
    audio_dist = _log_probs(output.audio_logits[0], audio_support, trajectory.audio_temperature)

    # forced PAD 使用任意合法索引 gather，再精确清零；不会对其计算伪 logprob。
    text_index = torch.searchsorted(
        text_support, text.masked_fill(~trajectory.text_mask, int(text_support[0]))
    )
    audio_index = audio.masked_fill(~trajectory.audio_mask, 0)
    audio_index = audio_index.masked_fill(
        audio_index == model.config.audio_stop_token, model.config.audio_codebook_size
    )
    text_lp = (
        text_dist.gather(-1, text_index[:, None]).squeeze(-1).masked_fill(~trajectory.text_mask, 0)
    )
    audio_lp = (
        audio_dist.gather(-1, audio_index[..., None])
        .squeeze(-1)
        .masked_fill(~trajectory.audio_mask, 0)
    )
    joint = audio_lp.sum(0) + (text_lp if trajectory.mode != "audio" else torch.zeros_like(text_lp))
    return PolicyScores(joint, text_lp, audio_lp, text_dist, audio_dist)


def decode_codes(trajectory: Rollout, config) -> Tensor:
    """去延迟得到完整连续前缀 [8,F]；不跳过残缺帧，不影响策略动作统计。"""
    _validate_trajectory(trajectory, config)
    count = 8
    frames = max(0, trajectory.audio_ids.shape[-1] - count)
    for frame in range(frames):
        diagonal = torch.stack(
            [trajectory.audio_ids[channel, 1 + channel + frame] for channel in range(count)]
        )
        if not ((diagonal >= 0) & (diagonal < config.audio_codebook_size)).all():
            frames = frame
            break
    if not frames:
        return trajectory.audio_ids.new_empty((count, 0))
    return restore_codes(
        trajectory.audio_ids, frames, start=1, codebook_size=config.audio_codebook_size
    )
