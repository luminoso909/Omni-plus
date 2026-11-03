"""小尺寸随机权重测试：不下载模型、词表或数据，不启动训练任务。"""

import json

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from omni.models.thinker import OmniPlusConfig, OmniPlusForCausalLM, OmniPlusModel
from omni.models.thinker.model_thinker import MOEFeedForward, RMSNorm, RotaryPositionalEmbedding


@pytest.fixture
def config() -> OmniPlusConfig:
    return OmniPlusConfig(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=None,
    )


@pytest.fixture
def tokens() -> torch.Tensor:
    return torch.tensor([[1, 7, 9, 3], [1, 5, 6, 8]])


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_attention_heads": 0},
        {"hidden_size": 31},
        {"head_dim": 3},
        {"num_key_value_heads": 3},
        {"num_experts_per_tok": 5},
        {"dropout": 1.0},
        {"rms_norm_eps": 0},
        {"rope_scaling": {"factor": 0}},
    ],
)
def test_invalid_config(overrides: dict) -> None:
    with pytest.raises(ValueError):
        OmniPlusConfig(**overrides)


def test_config_roundtrip(config: OmniPlusConfig, tmp_path) -> None:
    config.rope_scaling = dict(
        type="yarn",
        factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=8,
        attention_factor=1.0,
    )
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, OmniPlusConfig)
    assert loaded.rope_scaling == config.rope_scaling
    assert loaded.audio_ids == config.audio_ids
    assert loaded.image_ids == config.image_ids
    assert loaded.tie_word_embeddings is True
    assert loaded.head_dim == 8


@pytest.mark.parametrize("use_moe", [False, True])
def test_backbone_loss_and_backward(
    config: OmniPlusConfig, tokens: torch.Tensor, use_moe: bool
) -> None:
    config.use_moe = use_moe
    model = OmniPlusForCausalLM(config).train()
    result = model(tokens, labels=tokens, output_hidden_states=True, use_cache=False)
    expected = F.cross_entropy(
        result.logits[:, :-1].reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(result.loss, expected + result.aux_loss)
    assert result.logits.shape == (2, 4, 64)
    assert len(result.hidden_states) == config.num_hidden_layers + 1
    assert result.hidden_states[-1].shape == (2, 4, 32)
    result.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    if use_moe:
        assert model.model.layers[0].mlp.gate.weight.grad.abs().sum() > 0
    assert not hasattr(model.model, "lm_head")
    assert isinstance(model.model, OmniPlusModel)


@pytest.mark.parametrize("flash_attn", [False, True])
@pytest.mark.parametrize("use_moe", [False, True])
def test_cache_and_padding(
    config: OmniPlusConfig, tokens: torch.Tensor, flash_attn: bool, use_moe: bool
) -> None:
    config.flash_attn = flash_attn
    config.use_moe = use_moe
    model = OmniPlusForCausalLM(config).eval()
    # 首行左 padding；缓存多 token Decode 应与全量因果前向一致。
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    with torch.no_grad():
        full = model(tokens, attention_mask=mask).logits
        prefix = model(tokens[:, :2], attention_mask=mask[:, :2], use_cache=True)
        suffix = model(
            tokens[:, 2:],
            attention_mask=mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(full[:, 2:], suffix.logits, rtol=2e-5, atol=2e-6)
        assert suffix.past_key_values[0][0].shape == (2, 2, 4, 8)
        changed = tokens.clone()
        changed[:, -1] = 20
        torch.testing.assert_close(model(tokens).logits[:, :-1], model(changed).logits[:, :-1])
        # padding token 的值不应影响其它位置。
        changed = tokens.clone()
        changed[0, 0] = 30
        torch.testing.assert_close(full[0, 1:], model(changed, attention_mask=mask).logits[0, 1:])
        embedded = model.model.embed_tokens(tokens)
        torch.testing.assert_close(model(inputs_embeds=embedded).logits, model(tokens).logits)


def test_manual_attention_matches_sdpa(config: OmniPlusConfig, tokens: torch.Tensor) -> None:
    model = OmniPlusForCausalLM(config).eval()
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    expected = model(tokens, attention_mask=mask).logits
    config.flash_attn = False
    torch.testing.assert_close(
        expected, model(tokens, attention_mask=mask).logits, rtol=2e-5, atol=2e-6
    )


def test_moe_routing_and_auxiliary_loss(config: OmniPlusConfig) -> None:
    config.num_experts = 3
    config.num_experts_per_tok = 2
    module = MOEFeedForward(config).train()
    x = torch.randn(2, 3, 32, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    actual, aux = module(x, mask)
    flat = x.reshape(-1, 32)
    probs = F.linear(flat, module.gate.weight).softmax(-1)
    weights, indices = probs.topk(2, dim=-1, sorted=False)
    weights = weights / weights.sum(-1, keepdim=True)
    # 独立逐 token 参考计算，验证专家选择与 dispatch/combine。
    expected = torch.stack(
        [
            sum(module.experts[int(indices[t, k])](flat[t]) * weights[t, k] for k in range(2))
            for t in range(6)
        ]
    ).view_as(x)
    torch.testing.assert_close(actual, expected)
    valid = mask.flatten().bool()
    load = F.one_hot(indices[valid], 3).float().mean(0).sum(0)
    expected_aux = (load * probs[valid].mean(0)).sum() * 3 * config.router_aux_loss_coef
    torch.testing.assert_close(aux, expected_aux)
    (actual.square().mean() + aux).backward()
    assert module.gate.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0
    module.eval()
    assert module(x)[1].item() == 0


def test_qk_norm_switch_and_rope(config: OmniPlusConfig) -> None:
    model = OmniPlusForCausalLM(config)
    assert isinstance(model.model.layers[0].self_attn.q_norm, RMSNorm)
    config.qk_norm = False
    assert isinstance(
        OmniPlusForCausalLM(config).model.layers[0].self_attn.q_norm, torch.nn.Identity
    )
    rope = RotaryPositionalEmbedding(config)
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
    positions = torch.tensor([[3]])
    freqs = config.rope_theta ** (-torch.arange(0, 8, 2).float() / 8)
    angles = torch.cat((freqs * 3, freqs * 3))
    rotated = torch.cat((-x[..., 4:], x[..., :4]), dim=-1)
    torch.testing.assert_close(rope(x, positions), x * angles.cos() + rotated * angles.sin())


def test_yarn_changes_low_frequency(config: OmniPlusConfig) -> None:
    config.head_dim = 64
    config.max_position_embeddings = 128
    rope = RotaryPositionalEmbedding(config)
    config.rope_scaling = dict(
        factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=16,
        attention_factor=1.0,
    )
    yarn = RotaryPositionalEmbedding(config)
    x = torch.ones(1, 1, 2, 64)
    positions = torch.tensor([[0, 100]])
    original, scaled = rope(x, positions), yarn(x, positions)
    torch.testing.assert_close(original[:, :, 0], scaled[:, :, 0])
    assert not torch.allclose(original[:, :, 1], scaled[:, :, 1])
    assert torch.isfinite(scaled).all()


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("use_moe", [False, True])
def test_hf_and_raw_weight_roundtrip(
    config: OmniPlusConfig, tokens: torch.Tensor, tied: bool, use_moe: bool, tmp_path
) -> None:
    config.tie_word_embeddings = tied
    config.use_moe = use_moe
    model = OmniPlusForCausalLM(config).eval()
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied
    model.save_pretrained(tmp_path / "hf", max_shard_size="10KB")
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "hf", local_files_only=True).eval()
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tied
    torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits)
    # 原始 .pth 路径与 HF safetensors 目录走不同的显式入口。
    torch.save(model.state_dict(), tmp_path / "raw.pth")
    raw = OmniPlusForCausalLM(config).eval()
    info = raw.load_state_dict(torch.load(tmp_path / "raw.pth", weights_only=True), strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    torch.testing.assert_close(raw(tokens).logits, model(tokens).logits)
    broken = dict(model.state_dict())
    del broken["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        raw.load_state_dict(broken, strict=True)


class RecordingStreamer:
    def __init__(self) -> None:
        self.chunks = []
        self.ended = False

    def put(self, value: torch.Tensor) -> None:
        self.chunks.append(value.clone())

    def end(self) -> None:
        self.ended = True


@pytest.mark.parametrize("use_cache", [False, True])
def test_generation_sampling_and_streamer(
    config: OmniPlusConfig, tokens: torch.Tensor, use_cache: bool
) -> None:
    model = OmniPlusForCausalLM(config).eval()
    streamer = RecordingStreamer()
    mask = torch.ones_like(tokens)
    greedy = model.generate(
        tokens, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=use_cache
    )
    sampled = model.generate(
        tokens,
        attention_mask=mask,
        max_new_tokens=3,
        do_sample=True,
        top_k=1,
        top_p=1.0,
        use_cache=use_cache,
        streamer=streamer,
    )
    assert torch.equal(greedy, sampled)
    assert streamer.ended and len(streamer.chunks) == 4
    assert torch.equal(streamer.chunks[0], tokens)
    assert torch.equal(torch.cat([c[:, None] for c in streamer.chunks[1:]], dim=1), sampled[:, 4:])
    cached = model.generate(
        tokens, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=not use_cache
    )
    assert torch.equal(greedy, cached)


def test_sampling_scores(config: OmniPlusConfig, tokens: torch.Tensor, monkeypatch) -> None:
    model = OmniPlusForCausalLM(config).eval()
    sampled_probs = []
    multinomial = torch.multinomial

    def record_probs(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        sampled_probs.append(probs.detach().clone())
        return multinomial(probs, num_samples)

    monkeypatch.setattr(torch, "multinomial", record_probs)
    model.generate(
        tokens,
        attention_mask=torch.ones_like(tokens),
        max_new_tokens=1,
        do_sample=True,
        temperature=0.7,
        top_k=5,
        top_p=0.8,
        repetition_penalty=1.5,
    )
    expected = model(tokens).logits[:, -1].float().clone()
    for row, ids in enumerate(tokens):
        seen = ids.unique()
        values = expected[row, seen]
        expected[row, seen] = torch.where(values < 0, values * 1.5, values / 1.5)
    expected /= 0.7
    expected[expected < expected.topk(5).values[:, -1:]] = -torch.inf
    sorted_scores, sorted_ids = expected.sort(descending=True, dim=-1)
    drop = sorted_scores.softmax(-1).cumsum(-1) > 0.8
    drop[:, 1:] = drop[:, :-1].clone()
    drop[:, 0] = False
    expected.masked_fill_(torch.zeros_like(drop).scatter(1, sorted_ids, drop), -torch.inf)
    torch.testing.assert_close(sampled_probs[0], expected.softmax(-1))


def test_logits_selection_and_ignored_loss(config: OmniPlusConfig, tokens: torch.Tensor) -> None:
    model = OmniPlusForCausalLM(config).eval()
    full = model(tokens).logits
    torch.testing.assert_close(model(tokens, logits_to_keep=2).logits, full[:, -2:])
    assert model(tokens, labels=torch.full_like(tokens, -100)).loss.item() == 0
    with pytest.raises(ValueError, match="logits_to_keep"):
        model(tokens, labels=tokens, logits_to_keep=1)
    with pytest.raises(ValueError, match="只能提供一种"):
        model(tokens, inputs_embeds=model.model.embed_tokens(tokens))
    bad_cache = model(tokens[:, :1], use_cache=True).past_key_values[:-1]
    with pytest.raises(ValueError, match="层数"):
        model(tokens[:, :1], past_key_values=bad_cache)
    config.max_position_embeddings = 1
    with pytest.raises(ValueError, match="超出"):
        model(tokens)


def test_tiny_config_file() -> None:
    with open("configs/model/thinker_tiny.json") as file:
        config = OmniPlusConfig(**json.load(file))
    model = OmniPlusForCausalLM(config)
    assert model(torch.tensor([[1, 2]])).logits.shape == (1, 2, 64)


def test_generation_batch_eos(config: OmniPlusConfig, tokens: torch.Tensor, monkeypatch) -> None:
    model = OmniPlusForCausalLM(config).eval()
    calls = []

    def scripted_forward(input_ids: torch.Tensor, **kwargs) -> MoeCausalLMOutputWithPast:
        calls.append(input_ids.shape[1])
        next_ids = [2, 5] if len(calls) == 1 else [7, 6]
        logits = torch.full((2, 1, config.vocab_size), -100.0)
        logits[torch.arange(2), 0, next_ids] = 100.0
        return MoeCausalLMOutputWithPast(logits=logits)

    monkeypatch.setattr(model, "forward", scripted_forward)
    result = model.generate(tokens, max_new_tokens=5, eos_token_id=[2, 6], use_cache=False)
    assert result[:, 4:].tolist() == [[2, 0], [5, 6]]
    assert calls == [4, 5]


def test_generation_zero_length_and_temperature(
    config: OmniPlusConfig, tokens: torch.Tensor
) -> None:
    model = OmniPlusForCausalLM(config).eval()
    assert torch.equal(model.generate(tokens, max_new_tokens=0), tokens)
    greedy = model.generate(tokens, max_new_tokens=2, do_sample=False)
    zero_temp = model.generate(tokens, max_new_tokens=2, do_sample=True, temperature=0)
    assert torch.equal(greedy, zero_temp)
