"""单设备 SFT 步骤与完整状态恢复；调用方显式启动训练。"""

import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .losses import loss_counts, sft_loss


def configure_trainable(model: nn.Module, mode: str = "all") -> int:
    """all/audio/last_layer/audio_projector/vision_projector，返回可训练参数数。"""
    if mode not in {"all", "audio", "last_layer", "audio_projector", "vision_projector"}:
        raise ValueError(f"未知冻结策略：{mode}")

    # 只通过 requires_grad 设置冻结，不改 state_dict 键或权重共享
    last = len(model.model.layers) - 1
    for name, param in model.named_parameters():
        trainable = mode == "all"
        if mode in {"audio", "last_layer"}:
            trainable = name.startswith(("talker.", "audio_proj.", "vision_proj."))
        if mode == "last_layer":
            trainable |= name.startswith((f"model.layers.{last}.", "model.norm."))
        if mode.endswith("_projector"):
            prefix = "audio_proj." if mode == "audio_projector" else "vision_proj."
            trainable = name.startswith(prefix)
        param.requires_grad_(trainable)
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def move_batch(batch: dict, device: str | torch.device) -> dict:
    """Tensor 移到训练设备；模态列表保留 None 和原 batch 索引。"""
    result = {}
    for name, value in batch.items():
        if isinstance(value, Tensor):
            value = value.to(device)
        elif isinstance(value, list):
            value = [item.to(device) if isinstance(item, Tensor) else item for item in value]
        result[name] = value
    return result


def train_step(
    model: nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str = "cpu",
    amp_dtype: torch.dtype | None = None,
    stop_weight: float = 10.0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """一个 optimizer step；batches 为梯度累积组，末组允许不足预设长度。"""
    if not batches:
        raise ValueError("一次更新至少需要一个 batch。")

    # 先统计整组分母，再逐 micro-batch 反传，不保留多份计算图
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts = loss_counts(batches)
    metrics = dict.fromkeys(("total", "text", "audio", "aux"), 0.0)
    for batch in batches:
        batch = move_batch(batch, device)
        with torch.autocast(
            torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            output = model(
                **{k: v for k, v in batch.items() if k not in {"labels", "audio_labels"}}
            )
            losses = sft_loss(
                output,
                batch["labels"],
                batch["audio_labels"],
                model.config.audio_stop_token,
                stop_weight,
                counts,
                aux_scale=1 / len(batches),
            )
        scaler.scale(losses["total"]).backward()
        for name in metrics:
            metrics[name] += float(losses[name].detach())

    # 只在累积组结束时裁剪与更新；scaler 状态随 checkpoint 保存
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.requires_grad],
        grad_clip,
        error_if_nonfinite=not scaler.is_enabled(),
    )
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    metrics["grad_norm"] = float(norm)
    metrics["optimizer_updated"] = float(scaler.get_scale() >= previous_scale)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module, batches: list[dict], device: str = "cpu", stop_weight: float = 10
) -> dict:
    """固定评估，无增强；用全评估集有效 Token 数归一化。"""
    if not batches:
        raise ValueError("评估集不能为空。")

    model.eval()
    counts = loss_counts(batches)
    metrics = dict.fromkeys(("total", "text", "audio", "aux"), 0.0)
    for batch in batches:
        batch = move_batch(batch, device)
        output = model(**{k: v for k, v in batch.items() if k not in {"labels", "audio_labels"}})
        losses = sft_loss(
            output,
            batch["labels"],
            batch["audio_labels"],
            model.config.audio_stop_token,
            stop_weight,
            counts,
            aux_scale=1 / len(batches),
        )
        for name in metrics:
            metrics[name] += float(losses[name])
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    progress: dict,
) -> None:
    """仅在 optimizer step 边界保存：model/optimizer/scaler/RNG 与下个 batch 位置。"""
    # 将 NumPy RNG 转为基本类型，恢复时无需反序列化自定义 Python 对象
    numpy_state = np.random.get_state()
    device = next(model.parameters()).device.type
    rng = dict(
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch=torch.get_rng_state(),
        mps=torch.mps.get_rng_state() if device == "mps" else None,
        cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            config=model.config.to_dict(),
            trainable=[name for name, param in model.named_parameters() if param.requires_grad],
            progress=progress,
            rng=rng,
        ),
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict:
    """严格恢复参数及训练状态；返回调用方保存的进度元数据。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if checkpoint["config"] != model.config.to_dict() or checkpoint["trainable"] != trainable:
        raise ValueError("恢复时模型配置或冻结策略变化，不能沿用旧优化器状态。")

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    # 权重与优化器恢复后再恢复随机数，避免构造模型消耗 RNG 改变后续结果
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(rng["torch"])
    if rng["mps"] is not None:
        torch.mps.set_rng_state(rng["mps"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint["progress"]
