"""GRPO 数学核心。物理步联合概率与 Loss 的归一化分别处理。"""

import math

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Float [prompts,group] → detached 同形状优势，使用总体标准差。"""
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError("Reward 需要非空 [prompts,group>=2]。")
    if not math.isfinite(epsilon) or epsilon <= 0 or not rewards.isfinite().all():
        raise ValueError("Reward 必须有限，epsilon 必须为有限正数。")

    values = rewards.detach()
    values = values if values.dtype == torch.float64 else values.float()
    centered = values - values.mean(-1, keepdim=True)
    scale = centered.square().mean(-1, keepdim=True).sqrt()
    if not centered.isfinite().all() or not scale.isfinite().all():
        raise ValueError("Reward 标准化溢出。")
    advantages = centered / (scale + epsilon)
    if not advantages.isfinite().all():
        raise ValueError("Reward 标准化溢出。")
    return advantages


def exact_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Float [...,support] → [...]，精确 KL(current || fixed reference)。"""
    if (
        log_probs.ndim < 1
        or log_probs.shape != reference_log_probs.shape
        or log_probs.shape[-1] < 1
    ):
        raise ValueError("KL 需要相同的非空支持集。")
    current = log_probs if log_probs.dtype == torch.float64 else log_probs.float()
    reference = reference_log_probs.detach().to(current)
    for values in (current, reference):
        if not values.isfinite().all() or not torch.allclose(
            values.logsumexp(-1), values.new_zeros(values.shape[:-1]), atol=1e-5, rtol=0
        ):
            raise ValueError("KL 输入需要有限且已归一化的 log 分布。")

    return (current.exp() * (current - reference)).sum(-1)


def grpo_loss(
    current: Tensor,
    old: Tensor,
    kl: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip: float = 0.2,
    beta: float = 0.04,
    normalization: str = "sequence_mean",
) -> dict[str, Tensor]:
    """Float current/old/kl [N,T]、advantages [N]、Bool mask [N,T]。

    每个 logprob 已是该物理步的有效动作之和。old 与 advantage 仅作为常量。
    kl 必须由当前策略与冻结参考策略计算，保留 current 分支的梯度。
    """
    if (
        current.ndim != 2
        or current.shape != old.shape
        or current.shape != kl.shape
        or current.shape != mask.shape
        or advantages.shape != current.shape[:1]
        or mask.dtype != torch.bool
        or current.shape[0] == 0
        or not mask.any(-1).all()
    ):
        raise ValueError("GRPO 需要一致的 [N,T] 与 [N]，每条轨迹至少一个有效动作。")
    if (
        not math.isfinite(clip)
        or not 0 < clip < 1
        or not math.isfinite(beta)
        or beta < 0
        or normalization not in {"sequence_mean", "token_mean"}
    ):
        raise ValueError("clip、KL 系数或归一化设置无效。")
    if not advantages.isfinite().all() or any(
        not value[mask].isfinite().all() for value in (current, old, kl)
    ):
        raise ValueError("有效 GRPO 概率、KL 和优势必须有限。")

    # masked padding 在 exp 前归零，避免无效位置的 NaN/Inf 进入计算图。
    current = current.masked_fill(~mask, 0)
    old = old.detach().masked_fill(~mask, 0).to(current)
    advantage = advantages.detach().to(current)[:, None]
    ratio = (current - old).exp()
    if not ratio[mask].isfinite().all():
        raise ValueError("Importance ratio 溢出，拒绝截断数值冒充 PPO clipping。")
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)

    def reduce(values: Tensor) -> Tensor:
        values = values.masked_fill(~mask, 0)
        if normalization == "sequence_mean":
            return (values.sum(-1) / mask.sum(-1)).mean()
        return values.sum() / mask.sum()

    policy = -reduce(surrogate)
    regularizer = reduce(kl)
    return dict(
        total=policy + beta * regularizer,
        policy=policy,
        kl=regularizer,
        clip_fraction=reduce(((ratio - 1).abs() > clip).to(current)).detach(),
        ratio_mean=reduce(ratio).detach(),
    )
