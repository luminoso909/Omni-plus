"""由调用方注入已加载的官方 Qwen3，复用原生前向、词表投影和 Cache。"""

from contextlib import ExitStack
from threading import get_ident

from torch import Tensor, nn
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import Cache

from .backend import BackboneOutput


class QwenBackend(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.model = model

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """输入 long [batch, seq]，输出模型精度的 [batch, seq, hidden]。"""
        return self.model.get_input_embeddings()(input_ids)

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
        hidden_layers: tuple[int, ...] | None = None,
        logits_to_keep: int = 0,
    ) -> BackboneOutput:
        """输入 token 或浮点 embedding，沿用模型精度与设备。

        attention_mask 为覆盖历史和当前步的 bool 或 0/1 long [batch, total_seq]。
        position_ids 为 long [batch 或 1, seq]，cache_position 为物理 long [seq]。
        Cache 可原地增长。新话轮传入 None，关闭 Cache 时从空状态计算。
        hidden_layer 与 hidden_layers 二选一，按零起算 block 编号取层。
        多层结果沿请求顺序，每项均为当前 [B,T,D]，末层也在最终 norm 前。
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True.")
        if hidden_layer is not None and not 0 <= hidden_layer < self.model.config.num_hidden_layers:
            raise ValueError("hidden_layer must select a zero-based decoder block.")
        if hidden_layers is not None and (
            hidden_layer is not None
            or not hidden_layers
            or any(
                type(i) is not int or not 0 <= i < self.model.config.num_hidden_layers
                for i in hidden_layers
            )
            or len(set(hidden_layers)) != len(hidden_layers)
        ):
            raise ValueError("hidden_layers needs distinct zero-based blocks and no hidden_layer.")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be nonnegative.")

        # 有 mask 时按 HF generate 的有效 token 编号计算 RoPE 逻辑位置。
        if attention_mask is not None:
            batch, seq = (input_ids if input_ids is not None else inputs_embeds).shape[:2]
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            if attention_mask.shape != (batch, past + seq):
                raise ValueError(
                    "attention_mask must cover the full batch and cached/current tokens."
                )
            if position_ids is None:
                positions = attention_mask.long().cumsum(-1) - 1
                position_ids = positions.masked_fill(attention_mask == 0, 1)[:, -seq:]

        # 只捕获本次调用的指定 block，退出时释放 hook，保留原始梯度。
        layers = (hidden_layer,) if hidden_layer is not None else hidden_layers or ()
        hidden: dict[int, Tensor] = {}
        with ExitStack() as scope:
            if layers:
                thread_id = get_ident()

                def capture(index: int):
                    def hook(_module: nn.Module, _args: tuple, output: Tensor) -> None:
                        if get_ident() == thread_id:
                            hidden[index] = output

                    return hook

                for index in layers:
                    layer = self.model.model.layers[index]
                    scope.enter_context(layer.register_forward_hook(capture(index)))

            # 物理 Cache 位置交给 HF 推断，显式传入时原样使用。
            out = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False,
                output_attentions=False,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )
        return BackboneOutput(
            out.logits,
            hidden[hidden_layer] if hidden_layer is not None else None,
            out.past_key_values,
            tuple(hidden[i] for i in layers) if hidden_layers is not None else (),
        )
