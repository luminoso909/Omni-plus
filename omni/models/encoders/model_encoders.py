"""模态编码器后的特征投影；原始媒体与预训练编码器在外部处理。"""

from jaxtyping import Float
from torch import Tensor, nn


class Projector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        # 将编码器特征投影到 Thinker 宽度，不改变 token 数
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: Float[Tensor, "... features"]) -> Float[Tensor, "... hidden"]:
        return self.mlp(x)
