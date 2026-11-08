"""Qwen 线性层 LoRA 与独立多模态 Adapter；完整训练恢复仍由原 Checkpoint 负责。"""

import math
from pathlib import Path

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        self.scale = alpha / rank

        # 第一轮与原权重严格等价，后续梯度先进入 B，再进入 A。
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        """浮点输入 [...,in] → [...,out]，沿用原 Linear 的 dtype/device。"""
        return self.base(hidden) + self.lora_b(self.lora_a(hidden)) * self.scale


def inject_lora(model: nn.Module, settings: dict) -> list[str]:
    """仅替换显式列出的 Qwen 原生投影；返回完整模块路径，禁用时不修改模型。"""
    if not settings.get("enabled", False):
        return []
    if set(settings) - {"enabled", "rank", "alpha", "targets", "dropout"}:
        raise ValueError("未知 LoRA 配置键。")
    rank, alpha = settings["rank"], settings["alpha"]
    targets = settings["targets"]
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (
        type(rank) is not int
        or rank < 1
        or not math.isfinite(alpha)
        or alpha <= 0
        or not targets
        or len(set(targets)) != len(targets)
        or set(targets) - allowed
        or settings.get("dropout", 0) != 0
    ):
        raise ValueError("LoRA 需要正 rank/alpha、唯一合法 targets，当前 dropout 必须为 0。")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("LoRA 注入仅支持已核验的原生 Qwen3。")

    # 先核对全部目标，防止错名字或重复注入留下半修改模型。
    modules = []
    for index in range(model.config.num_hidden_layers):
        for target in targets:
            owner = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            name = f"model.layers.{index}.{owner}.{target}"
            base = model.get_submodule(name)
            if not isinstance(base, nn.Linear) or rank > min(base.in_features, base.out_features):
                raise ValueError(f"LoRA 目标不是未注入的 Linear，或 rank 太大：{name}")
            modules.append((name, base))
    model.requires_grad_(False)
    for name, base in modules:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(base, rank, alpha))
    model.omni_lora_config = dict(settings)
    return [name for name, _ in modules]


def _adapter_state(model: nn.Module) -> dict[str, Tensor]:
    # 同时保存冻结的新模块；独立 Adapter 不能依赖另一次随机初始化。
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("thinker.") or ".lora_a." in name or ".lora_b." in name
    }


def _metadata(model: nn.Module, identity: dict) -> dict:
    return dict(
        schema_version=1,
        identity=identity,
        config=model.config.to_dict(),
        lora=getattr(model.thinker.model, "omni_lora_config", {"enabled": False}),
    )


def export_adapter(path: Path, model: nn.Module, identity: dict) -> None:
    """保存 LoRA + 全部多模态模块；不含 Qwen 主权重或 optimizer。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 Adapter：{path}")
    if not identity:
        raise ValueError("Adapter 需要非空的基础模型、资产与 tokenizer 身份。")

    path.parent.mkdir(parents=True, exist_ok=True)
    values = {name: value.detach().cpu().clone() for name, value in _adapter_state(model).items()}
    with path.open("xb") as stream:
        torch.save(dict(metadata=_metadata(model, identity), state=values), stream)


def load_adapter(path: Path, model: nn.Module, identity: dict) -> None:
    """验证身份及完整键/shape/dtype 后再写入；不能把推理 Adapter 当作训练 Resume。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != _metadata(model, identity):
        raise ValueError("Adapter 基础模型、配置、tokenizer 或 LoRA 身份不一致。")
    expected, values = _adapter_state(model), payload["state"]
    if expected.keys() != values.keys():
        raise ValueError("Adapter 参数键缺失或多余。")
    for name, current in expected.items():
        saved = values[name]
        if saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"Adapter 参数形状或类型不匹配：{name}")
        if saved.is_floating_point() and not saved.isfinite().all():
            raise ValueError(f"Adapter 参数包含非有限值：{name}")

    with torch.no_grad():
        for name, current in expected.items():
            current.copy_(values[name].to(current.device))
