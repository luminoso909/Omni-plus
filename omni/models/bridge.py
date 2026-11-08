"""把选定的 Qwen block 输出融合成 Talker 条件。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import OmniPlusConfig
from .thinker.model_thinker import RMSNorm


def bridge_spec(config: OmniPlusConfig) -> dict:
    """解析零起算 block 索引，固定、末层各取一层，融合最多取四层。"""
    spec = getattr(config, "bridge", {})
    if not isinstance(spec, dict) or set(spec) - {"kind", "layers"}:
        raise ValueError("Bridge 配置仅接受 kind 和 layers。")
    kind = spec.get("kind", "fixed")
    if kind not in ("fixed", "last", "weighted", "gated"):
        raise ValueError(f"未知 Bridge 类型：{kind!r}。")

    last = config.num_hidden_layers - 1
    default = {"fixed": [config.bridge_layer], "last": [last]}.get(kind)
    layers = spec.get("layers", default)
    if (
        not isinstance(layers, (list, tuple))
        or any(type(i) is not int or not 0 <= i <= last for i in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("Bridge layers 必须为不同的零起算 Qwen Block 索引。")
    sizes = {"fixed": (1,), "last": (1,), "weighted": (2, 3, 4), "gated": (2,)}
    if len(layers) not in sizes[kind] or (kind == "last" and layers[0] != last):
        raise ValueError(f"Bridge {kind} 的层数或末层索引不正确。")
    return {"kind": kind, "layers": list(layers)}


class FixedBridge(nn.Module):
    def __init__(self, input_size: int, output_size: int, eps: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)
        self.norm = RMSNorm(output_size, eps)

    def forward(self, hidden: Tensor, valid: Tensor) -> Tensor:
        """Block 输出 [B,T,D] → 条件 [B,T,H]，valid [B,T] 的补齐位置清零。"""
        condition = self.norm(self.proj(hidden))
        return condition.masked_fill(~valid.bool().unsqueeze(-1), 0)


class WeightedBridge(FixedBridge):
    def __init__(self, input_size: int, output_size: int, eps: float, layers: int) -> None:
        super().__init__(input_size, output_size, eps)
        self.layer_logits = nn.Parameter(torch.zeros(layers))

    def forward(self, hidden: tuple[Tensor, ...], valid: Tensor) -> Tensor:
        """融合 K 个同形状 [B,T,D]，每层一个全局权重。"""
        logits = self.layer_logits
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        weights = logits.softmax(dim=0).to(hidden[0])
        fused = sum(weight * state for weight, state in zip(weights, hidden, strict=True))
        return super().forward(fused, valid)


class GatedBridge(FixedBridge):
    def __init__(self, input_size: int, output_size: int, eps: float) -> None:
        super().__init__(input_size, output_size, eps)
        self.gate_weight = nn.Parameter(torch.zeros(1, 2 * input_size))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: tuple[Tensor, Tensor], valid: Tensor) -> Tensor:
        """两个 [B,T,D] 按 token 做标量门控，初始各占一半。"""
        first, second = hidden
        score = F.linear(torch.cat((first, second), dim=-1), self.gate_weight, self.gate_bias)
        if score.dtype in (torch.float16, torch.bfloat16):
            score = score.float()
        gate = score.sigmoid().to(first)
        return super().forward(gate * first + (1 - gate) * second, valid)


def build_bridge(config: OmniPlusConfig, spec: dict) -> FixedBridge:
    sizes = (config.hidden_size, config.talker_hidden_size, config.rms_norm_eps)
    if spec["kind"] == "weighted":
        return WeightedBridge(*sizes, len(spec["layers"]))
    if spec["kind"] == "gated":
        return GatedBridge(*sizes)
    return FixedBridge(*sizes)
