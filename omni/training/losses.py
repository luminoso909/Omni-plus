"""Omni SFT 损失；Dataset 已 shift，文本和音频标签直接对齐 logits。"""

from torch import Tensor
from torch.nn import functional as F

from omni.models.model_omni import OmniOutput


def loss_counts(batches: list[dict]) -> tuple[int, Tensor]:
    """同一有效 batch 的文本 Token 数与各码本 Token 数 [C]。"""
    text = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
    audio = sum(batch["audio_labels"].ne(-100).sum(dim=(0, 2)).cpu() for batch in batches)
    return text, audio


def sft_loss(
    output: OmniOutput,
    labels: Tensor,
    audio_labels: Tensor,
    stop_token: int = 2050,
    stop_weight: float = 10.0,
    counts: tuple[int, Tensor] | None = None,
    aux_scale: float = 1.0,
) -> dict[str, Tensor]:
    """logits [B,T,V]/[B,C,T,A]，labels [B,T]/[B,C,T]；返回四个标量。

    stop 加权后仍按有效 Token 数归一化，再平均全部码本，与参考 SFT 定义一致。
    counts 可覆盖整个梯度累积组，使变长 micro-batch 与完整 batch 的 CE 相同。
    """
    if stop_weight < 0:
        raise ValueError("stop_weight 不能为负。")

    # 忽略区域的 CE 和梯度均为零；MoE 辅助项单独保留
    text_logits, audio_logits = output.logits, output.audio_logits
    text_raw = F.cross_entropy(
        text_logits.float().reshape(-1, text_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    audio_raw = F.cross_entropy(
        audio_logits.float().reshape(-1, audio_logits.shape[-1]),
        audio_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(audio_labels)

    # 每路码本独立归一化，避免长音频或 stop 个数隐式改变定义
    text_count, audio_count = counts or (
        int(labels.ne(-100).sum()),
        audio_labels.ne(-100).sum(dim=(0, 2)),
    )
    weights = 1 + audio_labels.eq(stop_token).to(audio_raw.dtype) * (stop_weight - 1)
    text_loss = text_raw.sum() / max(text_count, 1)
    audio_loss = (
        (audio_raw * weights).sum(dim=(0, 2)) / audio_count.to(audio_raw).clamp_min(1)
    ).mean()
    aux_loss = output.aux_loss * aux_scale if output.aux_loss is not None else text_loss * 0
    return dict(
        total=text_loss + audio_loss + aux_loss,
        text=text_loss,
        audio=audio_loss,
        aux=aux_loss,
    )
