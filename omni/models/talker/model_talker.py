"""Talker 的多码本 Embedding、输出头与独立 Transformer。"""

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from ..config import OmniPlusConfig
from ..thinker.model_lm import AttentionMask, OmniPlusTransformerBlock, PastKeyValues, RMSNorm

type AudioCodes = Int[Tensor, "batch codebooks frames"]


class TalkerHead(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        # 所有码本共享基础输出头，每个码本增加一个低秩残差
        hidden, rank = config.talker_hidden_size, config.talker_adapter_rank
        self.base = nn.Linear(hidden, config.audio_vocab_size, bias=False)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, config.audio_vocab_size, bias=False),
            )
            for _ in range(config.num_audio_codebooks)
        )

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch codebooks seq audio_vocab"]:
        base = self.base(x)
        return torch.stack([base + adapter(x) for adapter in self.adapters], dim=1)


class TalkerEmbedding(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        # 共享码本 Embedding，并用独立低秩分支区分各码本
        hidden, rank = config.talker_hidden_size, config.talker_adapter_rank
        self.base = nn.Embedding(config.audio_vocab_size, hidden)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Embedding(config.audio_vocab_size, rank),
                nn.GELU(),
                nn.Linear(rank, hidden, bias=False),
            )
            for _ in range(config.num_audio_codebooks)
        )

    def forward(self, audio_ids: AudioCodes) -> Float[Tensor, "batch seq hidden"]:
        base = self.base(audio_ids)
        codes = [base[:, i] + adapter(audio_ids[:, i]) for i, adapter in enumerate(self.adapters)]
        return torch.stack(codes, dim=1).mean(dim=1)


class TalkerModule(nn.Module):
    def __init__(self, config: OmniPlusConfig) -> None:
        super().__init__()

        # Talker 的头维度与 FFN 宽度独立计算，不沿用 Thinker 的 head_dim
        self.talker_config = OmniPlusConfig(
            hidden_size=config.talker_hidden_size,
            num_hidden_layers=config.num_talker_hidden_layers,
            num_attention_heads=config.talker_num_attention_heads,
            num_key_value_heads=config.talker_num_key_value_heads,
            intermediate_size=config.talker_intermediate_size,
            moe_intermediate_size=config.talker_intermediate_size,
            vocab_size=config.audio_vocab_size,
            hidden_act=config.hidden_act,
            dropout=config.dropout,
            rms_norm_eps=config.rms_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            qk_norm=config.qk_norm,
            flash_attn=config.flash_attn,
            use_moe=config.use_moe,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            router_aux_loss_coef=config.router_aux_loss_coef,
        )
        hidden = config.talker_hidden_size
        self.audio_spk_token = config.audio_spk_token

        # 历史音频编码与 Thinker 条件各自投影，再通过可学习比例融合
        self.embed_tokens = TalkerEmbedding(config)
        self.codec_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            RMSNorm(hidden, config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, hidden),
            RMSNorm(hidden, config.rms_norm_eps),
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(config.spk_emb_size, hidden, bias=False)

        # 复用自研 Transformer Block，输出各音频码本的 logits
        self.layers = nn.ModuleList(
            OmniPlusTransformerBlock(self.talker_config)
            for _ in range(config.num_talker_hidden_layers)
        )
        self.norm = RMSNorm(hidden, config.rms_norm_eps)
        self.lm_head = TalkerHead(config)

    def forward(
        self,
        bridge: Float[Tensor, "batch seq thinker_hidden"],
        audio_ids: AudioCodes,
        position_ids: Int[Tensor, "batch seq"],
        attention_mask: AttentionMask | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        spk_emb: Float[Tensor, "batch speaker_hidden"] | None = None,
    ) -> tuple[Float[Tensor, "batch codebooks seq audio_vocab"], PastKeyValues | None, Tensor]:
        # 在 speaker 标记处注入说话人条件
        audio = self.embed_tokens(audio_ids)
        if spk_emb is not None:
            speaker = self.spk_proj(spk_emb.to(audio))[:, None, :]
            audio = torch.where(
                (audio_ids[:, 0] == self.audio_spk_token)[..., None], speaker, audio
            )

        # 融合 Bridge 特征与历史音频码本
        hidden = (
            self.embed_proj(bridge) * self.text_scale + self.codec_proj(audio) * self.audio_scale
        )
        presents = []
        aux_loss = hidden.new_zeros(())

        # 逐层推进 Talker，Cache 与 Thinker 使用同一序列位置
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            hidden, present, layer_aux = layer(
                hidden, position_ids, attention_mask, past, use_cache
            )
            aux_loss = aux_loss + layer_aux
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.norm(hidden))
        return logits, tuple(presents) if use_cache else None, aux_loss
