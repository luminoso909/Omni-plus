"""v2 Thinker 的最小接口。"""

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor
from transformers.cache_utils import Cache


@dataclass
class BackboneOutput:
    logits: Tensor  # 浮点 [batch, kept_seq, vocab]
    selected_hidden_state: Tensor | None  # 浮点 [batch, seq, hidden]
    past_key_values: Cache | None


class ThinkerBackend(Protocol):
    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """将 long [batch, seq] 映射为浮点 [batch, seq, hidden]。"""
        ...

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        hidden_layer: int | None = None,
        logits_to_keep: int = 0,
    ) -> BackboneOutput:
        """计算文本 logits、可选的 block 输出和原生 Cache。"""
        ...
