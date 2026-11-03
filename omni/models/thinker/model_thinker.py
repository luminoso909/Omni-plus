"""自研 Thinker 核心 + 薄 HF CausalLM 外壳。

形状约定：batch、seq、hidden；Cache 为 [batch, kv_heads, total_seq, head_dim]。
MiniMind 权重使用 split-half RoPE、无 bias 投影及相同的 state_dict 键名。
generate 显式实现自回归循环，支持采样、KV Cache 和 Streamer。
"""

import math

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from ..config import OmniPlusConfig

type KeyValue = tuple[
    Float[Tensor, "batch kv_heads total_seq head_dim"],
    Float[Tensor, "batch kv_heads total_seq head_dim"],
]
type PastKeyValues = tuple[KeyValue, ...]
type AttentionMask = Bool[Tensor, "batch total_seq"] | Int[Tensor, "batch total_seq"]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float = 1e-6) -> None:
        super().__init__()

        # 归一化精度与可学习缩放
        self.eps = rms_norm_eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
        # FP32 计算 RMS，再恢复输入精度
        norm: Float[Tensor, "... 1"] = torch.rsqrt(x.float().square().mean(-1, True) + self.eps)
        return (x.float() * norm * self.weight).to(x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        # RoPE 基础参数与可选 YaRN 配置
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_scaling = config.rope_scaling

    def forward(
        self,
        x: Float[Tensor, "batch heads seq head_dim"],
        position_ids: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch heads seq head_dim"]:
        # 按需以 FP32 计算，避免 from_pretrained 的 meta 初始化或半精度转换破坏频率。
        with torch.autocast(device_type=x.device.type, enabled=False):
            dims = torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
            freqs = self.rope_theta ** (-dims / self.head_dim)  # [head_dim // 2]

            # 按频率区间应用 YaRN 缩放
            scale = 1.0
            if self.rope_scaling is not None:
                scaling = self.rope_scaling
                scale = scaling["attention_factor"]
                original = scaling["original_max_position_embeddings"]
                if self.max_position_embeddings > original:
                    # YaRN 高频外推、低频插值；边界与 MiniMind 权重约定一致。
                    low = self.head_dim * math.log(original / (2 * math.pi * scaling["beta_fast"]))
                    high = self.head_dim * math.log(original / (2 * math.pi * scaling["beta_slow"]))
                    low = max(0, math.floor(low / (2 * math.log(self.rope_theta))))
                    high = min(
                        self.head_dim // 2 - 1, math.ceil(high / (2 * math.log(self.rope_theta)))
                    )
                    ramp = ((dims / 2 - low) / max(high - low, 0.001)).clamp(0, 1)
                    freqs = freqs * (1 - ramp + ramp / scaling["factor"])

            # 将位置与频率组合为旋转角度
            angles: Float[Tensor, "batch seq half_head"] = position_ids.float()[..., None] * freqs
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            cos = (angles.cos() * scale).to(x.dtype)
            sin = (angles.sin() * scale).to(x.dtype)

        # 前后半维配对旋转，保持 MiniMind 权重约定
        left, right = x.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        self.config = config

        # Q/K/V 投影与多头输出投影
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )

        # QK Norm、位置编码与输出 Dropout
        self.q_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        )
        self.rope = RotaryPositionalEmbedding(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Float[Tensor, "batch seq hidden"],
        position_ids: Int[Tensor, "batch seq"],
        attention_mask: AttentionMask | None = None,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Float[Tensor, "batch seq hidden"], KeyValue | None]:
        config = self.config

        # 投影并拆分 Query 头与 KV 头
        q: Float[Tensor, "batch heads seq head_dim"] = rearrange(
            self.q_proj(x), "b t (h d) -> b h t d", h=config.num_attention_heads
        )
        k: Float[Tensor, "batch kv_heads seq head_dim"] = rearrange(
            self.k_proj(x), "b t (h d) -> b h t d", h=config.num_key_value_heads
        )
        v: Float[Tensor, "batch kv_heads seq head_dim"] = rearrange(
            self.v_proj(x), "b t (h d) -> b h t d", h=config.num_key_value_heads
        )

        # 先归一化 Q/K，再应用位置旋转
        q = self.rope(self.q_norm(q), position_ids)
        k = self.rope(self.k_norm(k), position_ids)

        # 拼接历史 K/V，缓存保留未扩展的 KV 头
        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[-2]
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v) if use_cache else None

        # 扩展 KV 头以匹配 Query 头
        groups = config.num_attention_heads // config.num_key_value_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # 因果掩码叠加 padding 掩码
        query_pos = torch.arange(x.shape[1], device=x.device) + past_len
        key_pos = torch.arange(k.shape[-2], device=x.device)
        mask = (key_pos[None, :] <= query_pos[:, None])[None, None, :, :]
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        # 计算注意力：SDPA 或显式 softmax
        dropout = config.dropout if self.training else 0.0
        if config.flash_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        else:
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(config.head_dim)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = scores.softmax(-1).masked_fill(~mask, 0).to(v.dtype)
            out = F.dropout(probs, p=dropout, training=self.training) @ v

        # 合并多头并投影回 hidden_size
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.resid_dropout(self.o_proj(out)), present


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: OmniPlusConfig, intermediate_size: int | None = None) -> None:
        super().__init__()

        # 门控分支、值分支与降维投影
        intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
        # 激活后的门控与值分支逐元素相乘
        gate: Float[Tensor, "... intermediate"] = self.act_fn(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        self.config = config

        # Router 与独立专家 FFN
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUFeedForward(config, config.moe_intermediate_size)
            for _ in range(config.num_experts)
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq hidden"],
        attention_mask: Bool[Tensor, "batch seq"] | Int[Tensor, "batch seq"] | None = None,
    ) -> tuple[Float[Tensor, "batch seq hidden"], Float[Tensor, ""]]:
        # 展平 token，计算路由概率并选择 Top-K 专家
        flat: Float[Tensor, "tokens hidden"] = x.reshape(-1, self.config.hidden_size)
        scores: Float[Tensor, "tokens experts"] = F.linear(
            flat.float(), self.gate.weight.float()
        ).softmax(-1)
        weights, indices = scores.topk(self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 归一化专家权重，Top-1 保留路由梯度
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                weights = weights / weights.sum(-1, keepdim=True)
            else:
                detached_scores = F.linear(flat.detach().float(), self.gate.weight.float()).softmax(
                    -1
                )
                selected = detached_scores.gather(-1, indices)
                weights = selected + (1 - selected).detach()

        # 分发 token 到专家，再加权汇总
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel():
                expert_out = expert(flat[token_ids]) * weights[token_ids, slots, None]
                out = out.index_add(0, token_ids, expert_out.to(out.dtype))
            elif self.training:
                # 未命中专家仍连入计算图，DDP 不会误判为 unused parameter。
                out = out + sum(param.sum() * 0 for param in expert.parameters())

        # 仅训练时计算有效 token 的路由均衡损失
        aux_loss = scores.new_zeros(())
        if self.training and self.config.router_aux_loss_coef > 0:
            valid = torch.ones(flat.shape[0], device=x.device, dtype=torch.bool)
            if attention_mask is not None:
                valid = attention_mask.reshape(-1).bool()
            if valid.any():
                # 沿 Top-K 槽求和，保持 MiniMind 的辅助损失尺度。
                load = F.one_hot(indices[valid], self.config.num_experts).float().mean(0).sum(0)
                aux_loss = (load * scores[valid].mean(0)).sum()
                aux_loss = aux_loss * self.config.num_experts * self.config.router_aux_loss_coef
        return out.view_as(x), aux_loss


class OmniPlusTransformerBlock(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        # 两条 Pre-Norm 残差分支
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MOEFeedForward(config) if config.use_moe else SwiGLUFeedForward(config)

    def forward(
        self,
        x: Float[Tensor, "batch seq hidden"],
        position_ids: Int[Tensor, "batch seq"],
        attention_mask: AttentionMask | None = None,
        past_key_value: KeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Float[Tensor, "batch seq hidden"], KeyValue | None, Float[Tensor, ""]]:
        # 注意力分支与残差连接
        attn_out, present = self.self_attn(
            self.input_layernorm(x), position_ids, attention_mask, past_key_value, use_cache
        )
        x = x + attn_out

        # FFN / MoE 分支与路由辅助损失
        normed = self.post_attention_layernorm(x)
        aux_loss = x.new_zeros(())
        if isinstance(self.mlp, MOEFeedForward):
            token_mask = None if attention_mask is None else attention_mask[:, -x.shape[1] :]
            ffn_out, aux_loss = self.mlp(normed, token_mask)
        else:
            ffn_out = self.mlp(normed)
        return x + ffn_out, present, aux_loss


class OmniPlusModel(nn.Module):
    """返回 (hidden, cache, aux_loss, history)，不含词表投影。"""

    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        self.config = config

        # Token 嵌入、Transformer 堆叠与最终归一化
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            OmniPlusTransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"] | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: Int[Tensor, "batch seq"] | None = None,
        past_key_values: PastKeyValues | None = None,
        inputs_embeds: Float[Tensor, "batch seq hidden"] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[Float[Tensor, "batch seq hidden"], PastKeyValues | None, Tensor, tuple | None]:
        # 输入与 Cache 的必要边界检查
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 和 inputs_embeds 必须且只能提供一种。")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("KV Cache 层数不匹配。")

        batch, seq = (input_ids if inputs_embeds is None else inputs_embeds).shape[:2]
        past_len = past_key_values[0][0].shape[-2] if past_key_values else 0

        if past_len + seq > self.config.max_position_embeddings:
            raise ValueError("序列超出 max_position_embeddings。")
        if attention_mask is not None and attention_mask.shape != (batch, past_len + seq):
            raise ValueError("attention_mask 必须覆盖 [batch, past_len + seq]。")

        # 获取输入嵌入并确定当前 token 位置
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_mask is not None:
            attention_mask = attention_mask.to(x.device)
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(past_len, past_len + seq, device=x.device)[None, :]
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -seq:]
        position_ids = position_ids.to(x.device)

        # 逐层更新 hidden、Cache 与路由损失
        x = self.dropout(x)
        history = () if output_hidden_states else None
        presents = []
        aux_loss = x.new_zeros(())
        for layer_id, layer in enumerate(self.layers):
            if output_hidden_states:
                history += (x,)
            past = None if past_key_values is None else past_key_values[layer_id]
            x, present, layer_aux = layer(x, position_ids, attention_mask, past, use_cache)
            aux_loss = aux_loss + layer_aux
            if use_cache:
                presents.append(present)

        # 最终归一化，并按需保留逐层 hidden states
        x = self.norm(x)
        if output_hidden_states:
            history += (x,)
        return x, tuple(presents) if use_cache else None, aux_loss, history


class OmniPlusForCausalLM(PreTrainedModel, GenerationMixin):
    """Backbone + 词表投影；权重直接使用 load_state_dict/from_pretrained 加载。"""

    config_class = OmniPlusConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["OmniPlusTransformerBlock"]
    accepts_loss_kwargs = False

    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__(config)

        # Backbone 与词表输出头；初始化时按配置绑定权重
        self.model = OmniPlusModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"] | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: Int[Tensor, "batch seq"] | None = None,
        past_key_values: PastKeyValues | None = None,
        inputs_embeds: Float[Tensor, "batch seq hidden"] | None = None,
        labels: Int[Tensor, "batch seq"] | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
    ) -> MoeCausalLMOutputWithPast:
        """返回 logits [batch, kept_seq, vocab]；loss = next-token CE + aux_loss。"""
        # 训练必须保留完整 logits，避免标签错位
        if logits_to_keep < 0 or (labels is not None and logits_to_keep != 0):
            raise ValueError("logits_to_keep 必须 >= 0，且训练时必须为 0。")

        # Backbone 前向与词表投影
        hidden, cache, aux_loss, history = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_hidden_states,
        )
        logits: Float[Tensor, "batch kept_seq vocab"] = self.lm_head(hidden[:, -logits_to_keep:])

        # 标签右移，计算 next-token CE 与路由辅助损失
        loss = None
        if labels is not None:
            targets = labels[:, 1:].to(logits.device).reshape(-1)
            shifted = logits[:, :-1].float().reshape(-1, self.config.vocab_size)
            # sum / 有效 token 数：全 ignore 或单 token 输入时也不会出现 NaN。
            loss = F.cross_entropy(shifted, targets, ignore_index=-100, reduction="sum")
            loss = loss / targets.ne(-100).sum().clamp_min(1) + aux_loss

        # 返回 HF 标准输出对象
        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=cache,
            hidden_states=history,
        )

    def _filter_top_p(
        self, scores: Float[Tensor, "batch vocab"], top_p: float
    ) -> Float[Tensor, "batch vocab"]:
        """保留累计概率达到 top_p 的最小候选集，返回过滤后的 logits。"""
        if top_p == 1.0:
            return scores

        # 按概率降序确定候选集，保留首次跨过阈值的 token
        sorted_scores, indices = scores.sort(descending=True, dim=-1)
        remove = sorted_scores.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        # 将掩码还原到原词表顺序，屏蔽候选集以外的 logits
        return scores.masked_fill(remove.scatter(1, indices, remove), -torch.inf)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Int[Tensor, "batch seq"],
        attention_mask: AttentionMask | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | list[int] | None = None,
        use_cache: bool = True,
        streamer: BaseStreamer | None = None,
    ) -> Int[Tensor, "batch total_seq"]:
        """model.eval() 后调用；返回包含 prompt 的完整 token 序列。"""
        # 生成参数的必要检查
        if (
            max_new_tokens < 0
            or not math.isfinite(temperature)
            or temperature < 0
            or top_k < 0
            or not 0 < top_p <= 1
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0
        ):
            raise ValueError("检查生成长度、temperature、top_k、top_p 和 repetition_penalty。")

        # 结束标记、padding 与批量生成状态
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = torch.as_tensor(
            [] if eos_token_id is None else eos_token_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        ).reshape(-1)
        pad_id = self.config.pad_token_id
        pad_id = pad_id if pad_id is not None else (int(eos_ids[0]) if eos_ids.numel() else 0)
        mask = (
            torch.ones_like(input_ids)
            if attention_mask is None
            else attention_mask.to(input_ids.device)
        )
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        cache = None
        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 首轮处理 prompt，随后使用 Cache 逐 token 解码
        for _ in range(max_new_tokens):
            tokens = input_ids[:, -1:] if cache is not None else input_ids
            outputs = self(
                tokens,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=use_cache,
                logits_to_keep=1,
            )
            scores = outputs.logits[:, -1].float()

            # 对已出现的 token 应用重复惩罚
            if repetition_penalty != 1.0:
                seen = scores.gather(1, input_ids)
                seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
                scores.scatter_(1, input_ids, seen)

            # 温度缩放、Top-K / Top-P 过滤与采样
            if do_sample and temperature > 0:
                scores = scores / temperature
                if top_k > 0:
                    cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, -torch.inf)
                scores = self._filter_top_p(scores, top_p)

                next_token = torch.multinomial(scores.softmax(-1), num_samples=1)
            else:
                next_token = scores.argmax(-1, keepdim=True)

            # 追加 token、更新 Cache，并追踪各样本的结束状态
            next_token = next_token.masked_fill(finished[:, None], pad_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            mask = torch.cat((mask, (~finished[:, None]).to(mask.dtype)), dim=1)
            cache = outputs.past_key_values
            if streamer is not None:
                streamer.put(next_token.squeeze(-1).cpu())
            finished |= torch.isin(next_token.squeeze(-1), eos_ids)
            if finished.all():
                break

        # 结束流式输出，返回包含 prompt 的完整序列
        if streamer is not None:
            streamer.end()
        return input_ids


TransformerLM = OmniPlusForCausalLM
