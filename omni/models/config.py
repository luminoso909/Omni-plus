"""Thinker、Talker 与多模态接口配置；不构造编码器或下载资源。"""

import math
from typing import Any

from transformers import PretrainedConfig
from transformers.activations import ACT2FN


class OmniPlusConfig(PretrainedConfig):
    model_type = "omni-plus"

    def __init__(
        self,
        # Thinker / LLM：主干结构
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        vocab_size: int = 6400,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = 4,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "silu",
        # Thinker：正则化与初始化
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        # Thinker：位置编码与长上下文
        max_position_embeddings: int = 32768,
        rope_theta: float = 1e6,
        inference_rope_scaling: bool = False,
        rope_scaling: dict[str, Any] | None = None,
        # Thinker：注意力、输出头与推理开关
        qk_norm: bool = True,
        flash_attn: bool = True,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        # Thinker：文本 Token 与思考结束标记
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        pad_token_id: int | None = None,
        think_end_ids: list[int] | None = None,
        # Thinker：MoE 路由与专家（use_moe=True 时使用）
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        # Talker：独立主干维度与音频码本
        num_talker_hidden_layers: int = 4,
        talker_hidden_size: int = 768,
        talker_num_attention_heads: int = 8,
        talker_num_key_value_heads: int | None = 4,
        talker_intermediate_size: int | None = None,
        num_audio_codebooks: int = 8,
        talker_adapter_rank: int = 256,
        audio_codebook_size: int = 2048,
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        spk_emb_size: int = 192,
        # 多模态输入：音频占位符与编码器输出维度
        audio_ids: list[int] | None = None,
        audio_special_token: str = "<|audio_pad|>",
        audio_hidden_size: int = 512,
        # 多模态输入：图像占位符、编码器维度与 Token 数
        image_ids: list[int] | None = None,
        image_special_token: str = "<|image_pad|>",
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        # Bridge：Thinker 条件特征的取层位置
        bridge_layer: int | None = None,
        # Hugging Face 通用配置字段
        **kwargs: Any,
    ) -> None:
        # HF 通用元数据：文本特殊 Token 与权重绑定
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Thinker / LLM：主干结构
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads 必须为正数。")

        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.intermediate_size = (
            math.ceil(hidden_size * math.pi / 64) * 64
            if intermediate_size is None
            else intermediate_size
        )
        self.hidden_act = hidden_act

        # Thinker：正则化与初始化
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        # Thinker：位置编码与长上下文
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = None if rope_scaling is None else dict(rope_scaling)
        if self.rope_scaling is None and inference_rope_scaling:
            self.rope_scaling = dict(
                rope_type="yarn",
                factor=16.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_max_position_embeddings=2048,
                attention_factor=1.0,
            )

        # Thinker：注意力、输出头与推理开关
        self.qk_norm = qk_norm
        self.flash_attn = flash_attn
        self.use_cache = use_cache

        # Thinker：文本 Token 与思考结束标记
        self.think_end_ids = [26, 234, 234] if think_end_ids is None else list(think_end_ids)

        # Thinker：MoE 路由与专家（use_moe=True 时使用）
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            self.intermediate_size if moe_intermediate_size is None else moe_intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        # Talker：独立主干维度与音频码本
        self.num_talker_hidden_layers = num_talker_hidden_layers
        self.talker_hidden_size = talker_hidden_size
        self.talker_num_attention_heads = talker_num_attention_heads
        self.talker_num_key_value_heads = talker_num_key_value_heads
        self.talker_intermediate_size = talker_intermediate_size
        self.num_audio_codebooks = num_audio_codebooks
        self.talker_adapter_rank = talker_adapter_rank
        self.audio_codebook_size = audio_codebook_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.spk_emb_size = spk_emb_size

        # 多模态输入：音频占位符与编码器输出维度
        self.audio_ids = [16] if audio_ids is None else list(audio_ids)
        self.audio_special_token = audio_special_token
        self.audio_hidden_size = audio_hidden_size

        # 多模态输入：图像占位符、编码器维度与 Token 数
        self.image_ids = [12] if image_ids is None else list(image_ids)
        self.image_special_token = image_special_token
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len

        # Bridge：Thinker 条件特征的取层位置
        self.bridge_layer = (
            max(0, num_hidden_layers // 2 - 1) if bridge_layer is None else bridge_layer
        )

        # 统一校验配置，不在模型计算中重复检查
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        # 主干维度与 GQA 头数约束
        dims = (
            self.hidden_size,
            self.vocab_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.intermediate_size,
            self.max_position_embeddings,
            self.num_experts,
            self.moe_intermediate_size,
        )
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in dims):
            raise ValueError("模型维度、层数、头数和专家数必须为正整数。")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "head_dim 必须为偶数；num_attention_heads 必须能被 num_key_value_heads 整除。"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除。")

        # 路由、激活函数与数值范围
        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok 必须在 [1, num_experts] 内。")
        if self.hidden_act not in ACT2FN:
            raise ValueError(f"不支持 hidden_act={self.hidden_act!r}。")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内。")
        for name in ("rms_norm_eps", "rope_theta", "initializer_range"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数。")
        if not math.isfinite(self.router_aux_loss_coef) or self.router_aux_loss_coef < 0:
            raise ValueError("router_aux_loss_coef 必须为有限非负数。")

        # YaRN 缩放参数
        if self.rope_scaling is not None:
            scaling = self.rope_scaling
            if scaling.get("rope_type", scaling.get("type", "yarn")) != "yarn":
                raise ValueError("当前只支持 YaRN rope_scaling。")
            defaults = dict(
                factor=16.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_max_position_embeddings=2048,
                attention_factor=1.0,
            )
            for key, default in defaults.items():
                scaling.setdefault(key, default)
                if not math.isfinite(scaling[key]) or scaling[key] <= 0:
                    raise ValueError(f"rope_scaling.{key} 必须为有限正数。")
            if scaling["factor"] < 1 or scaling["beta_fast"] < scaling["beta_slow"]:
                raise ValueError("YaRN 要求 factor >= 1 且 beta_fast >= beta_slow。")
            if self.rope_theta <= 1:
                raise ValueError("YaRN 要求 rope_theta > 1。")
