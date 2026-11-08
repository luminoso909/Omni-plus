"""Bridge 公式、选层、冻结梯度与原生 Qwen/Talker 的离线对照。"""

import math

import pytest
import torch
from torch import Tensor, nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from omni.models.bridge import GatedBridge, WeightedBridge, bridge_spec, build_bridge
from omni.models.config import OmniPlusConfig
from omni.models.model_omni_v2 import FixedBridge, OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.losses import sft_loss


def tiny_model(bridge: dict | None = None) -> OmniPlusOmni:
    native = Qwen3Config(
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
    )
    config = OmniPlusConfig(
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=3,
        num_key_value_heads=1,
        vocab_size=64,
        image_ids=[12],
        image_hidden_size=3,
        audio_ids=[13],
        audio_hidden_size=4,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=32,
        talker_adapter_rank=4,
        audio_codebook_size=8,
        audio_vocab_size=16,
        audio_pad_token=9,
        audio_stop_token=10,
        audio_spk_token=11,
        pad_token_id=0,
        eos_token_id=2,
        bridge=bridge,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return OmniPlusOmni(QwenBackend(Qwen3ForCausalLM(native)), config)


SPECS = [
    dict(kind="fixed", layers=[1]),
    dict(kind="last"),
    dict(kind="weighted", layers=[3, 1, 0]),
    dict(kind="gated", layers=[1, 3]),
]


def project_reference(bridge: FixedBridge, fused: Tensor, valid: Tensor) -> Tensor:
    value = (fused @ bridge.proj.weight.T).float()
    rms = (value.square().mean(dim=-1, keepdim=True) + bridge.norm.eps).sqrt()
    condition = (value / rms * bridge.norm.weight).to(fused.dtype)
    return torch.where(valid.bool().unsqueeze(-1), condition, 0.0)


def test_fixed_preserves_legacy_keys_config_and_initialization() -> None:
    legacy = tiny_model()
    assert "bridge" not in legacy.config.to_dict()
    assert legacy.bridge_spec == {"kind": "fixed", "layers": [1]}
    assert set(legacy.bridge.state_dict()) == {"proj.weight", "norm.weight"}

    # 每个候选使用同样的随机公共参数，且仅 fusion 参数增量不同。
    original = legacy.state_dict()
    for spec in SPECS:
        candidate = tiny_model(spec)
        assert OmniPlusConfig.from_dict(candidate.config.to_dict()).bridge == spec
        state = candidate.state_dict()
        for name, value in original.items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    explicit = tiny_model(SPECS[0])
    explicit.load_state_dict(original, strict=True)
    tokens = torch.tensor([[0, 1, 3, 4]])
    with torch.inference_mode():
        a = legacy.eval()(tokens, attention_mask=tokens.ne(0))
        b = explicit.eval()(tokens, attention_mask=tokens.ne(0))
    torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
    torch.testing.assert_close(a.audio_logits, b.audio_logits, rtol=0, atol=0)


@pytest.mark.parametrize("spec", SPECS, ids=[spec["kind"] for spec in SPECS])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bridge_shape_precision_mask_and_parameter_count(spec: dict, dtype: torch.dtype) -> None:
    config = OmniPlusConfig(hidden_size=16, num_hidden_layers=4, talker_hidden_size=8, bridge=spec)
    resolved = bridge_spec(config)
    bridge = build_bridge(config, resolved).to(dtype=dtype)
    hidden = tuple(torch.randn(3, 4, 16, dtype=dtype) for _ in resolved["layers"])
    valid = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]])
    output = bridge(hidden[0] if len(hidden) == 1 else hidden, valid)
    extras = {"weighted": 3, "gated": 33}.get(spec["kind"], 0)
    assert sum(p.numel() for p in bridge.parameters()) == 16 * 8 + 8 + extras
    assert output.shape == (3, 4, 8) and output.dtype == dtype
    assert output.device == hidden[0].device and output.isfinite().all()
    assert not output[~valid.bool()].any()
    assert output[valid.bool()].abs().sum() > 0


def test_weighted_softmax_reference_and_gradient() -> None:
    bridge = WeightedBridge(2, 3, 1e-6, 3).double()
    hidden = (torch.tensor([[[1.0, 2.0], [4.0, 3.0]]], dtype=torch.double),) + (
        torch.tensor([[[-2.0, 0.0], [2.0, 1.0]]], dtype=torch.double),
        torch.tensor([[[1.0, -2.0], [0.0, -3.0]]], dtype=torch.double),
    )
    valid = torch.tensor([[True, False]])
    probe = torch.tensor([[[2.0, -1.0], [0.0, 0.0]]], dtype=torch.double)
    torch.testing.assert_close(
        bridge(hidden, valid), project_reference(bridge, sum(hidden) / 3, valid)
    )
    with torch.no_grad():
        bridge.layer_logits.copy_(torch.tensor([0, math.log(2), math.log(3)], dtype=torch.double))
    weights = torch.tensor([1 / 6, 1 / 3, 1 / 2], dtype=torch.double)
    expected = sum(weight * state for weight, state in zip(weights, hidden, strict=True))
    projected = []
    with bridge.proj.register_forward_pre_hook(lambda _, args: projected.append(args[0])):
        output = bridge(hidden, valid)
    assert len(projected) == 1
    torch.testing.assert_close(projected[0], expected)
    torch.testing.assert_close(output, project_reference(bridge, expected, valid))

    actual_grad = torch.autograd.grad((projected[0] * probe).sum(), bridge.layer_logits)[0]
    expected_grad = weights * torch.stack([((state - expected) * probe).sum() for state in hidden])
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_grad.sum(), torch.zeros((), dtype=torch.double))


def test_gated_tokenwise_reference_and_gradient() -> None:
    bridge = GatedBridge(2, 3, 1e-6).double()
    first = torch.tensor([[[1.0, -1.0], [-2.0, 1.0]]], dtype=torch.double)
    second = torch.tensor([[[2.0, 3.0], [1.0, -4.0]]], dtype=torch.double)
    valid = torch.ones((1, 2), dtype=torch.bool)
    torch.testing.assert_close(
        bridge((first, second), valid), project_reference(bridge, (first + second) / 2, valid)
    )
    with torch.no_grad():
        bridge.gate_weight.copy_(torch.tensor([[0.2, -0.3, 0.4, 0.1]], dtype=torch.double))
        bridge.gate_bias.fill_(0.15)

    # 按明确的列系数独立计算 logit 和导数，不复用 F.linear/sigmoid。
    score = 0.2 * first[..., 0] - 0.3 * first[..., 1] + 0.4 * second[..., 0]
    score += 0.1 * second[..., 1] + 0.15
    gate = (1 / (1 + (-score).exp())).unsqueeze(-1)
    expected = second + gate * (first - second)
    projected = []
    with bridge.proj.register_forward_pre_hook(lambda _, args: projected.append(args[0])):
        output = bridge((first, second), valid)
    assert len(projected) == 1
    assert gate[0, 0, 0] != gate[0, 1, 0]
    torch.testing.assert_close(projected[0], expected)
    torch.testing.assert_close(output, project_reference(bridge, expected, valid))

    probe = torch.tensor([[[1.0, -2.0], [-0.5, 0.2]]], dtype=torch.double)
    grad_w, grad_b = torch.autograd.grad(
        (projected[0] * probe).sum(), (bridge.gate_weight, bridge.gate_bias)
    )
    d_score = gate * (1 - gate) * ((first - second) * probe).sum(dim=-1, keepdim=True)
    expected_w = (torch.cat((first, second), dim=-1) * d_score).sum((0, 1))[None]
    torch.testing.assert_close(grad_w, expected_w, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(grad_b, d_score.sum().view(1), atol=1e-12, rtol=1e-12)


def test_selected_hf_blocks_have_correct_index_order_and_norm() -> None:
    model = tiny_model(SPECS[2]).eval()
    backend = model.thinker
    tokens = torch.tensor([[0, 0, 1, 3], [1, 4, 6, 7]])
    mask = tokens.ne(0)
    positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 1)
    layers = (3, 1, 0)
    with torch.inference_mode():
        reference = backend.model(
            tokens,
            attention_mask=mask,
            position_ids=positions,
            output_hidden_states=True,
            use_cache=False,
        )
        selected = backend(input_ids=tokens, attention_mask=mask, hidden_layers=layers)
    assert selected.selected_hidden_state is None
    assert len(selected.selected_hidden_states) == 3
    torch.testing.assert_close(selected.logits, reference.logits, rtol=0, atol=0)
    for index, raw in zip(layers, selected.selected_hidden_states, strict=True):
        assert raw.shape == (2, 4, 24)
        assert raw.dtype == reference.logits.dtype and raw.device == tokens.device
        if index == 3:
            assert not torch.allclose(raw, reference.hidden_states[-1])
            raw = backend.model.model.norm(raw)
        torch.testing.assert_close(raw, reference.hidden_states[index + 1], rtol=0, atol=0)
    assert all(not layer._forward_hooks for layer in backend.model.model.layers)


@pytest.mark.parametrize(
    "spec",
    [
        dict(kind="typo"),
        dict(kind="last", layers=[1]),
        dict(kind="fixed", layers=[False]),
        dict(kind="weighted", layers=[0, 0]),
        dict(kind="weighted", layers=[0]),
        dict(kind="weighted", layers=[0, 1, 2, 3, 4]),
        dict(kind="gated", layers=[0, 4]),
        dict(kind="gated", layers=[1, 2, 3]),
        dict(kind="gated", layer=[0, 1]),
    ],
)
def test_invalid_bridge_specs_fail_early(spec: dict) -> None:
    with pytest.raises(ValueError, match="Bridge"):
        bridge_spec(OmniPlusConfig(num_hidden_layers=4, bridge=spec))


def test_multi_layer_hooks_clean_up_after_forward_failure() -> None:
    backend = tiny_model().thinker
    tokens = torch.tensor([[1, 3]])
    with pytest.raises(ValueError, match="hidden_layers"):
        backend(input_ids=tokens, hidden_layer=0, hidden_layers=(1, 3))
    with pytest.raises(ValueError, match="hidden_layers"):
        backend(input_ids=tokens, hidden_layers=(1, 1))
    with pytest.raises((IndexError, RuntimeError)):
        backend(input_ids=torch.tensor([[64]]), hidden_layers=(1, 3))
    assert all(not layer._forward_hooks for layer in backend.model.model.layers)


@pytest.mark.parametrize("spec", SPECS, ids=[spec["kind"] for spec in SPECS])
def test_mixed_supervision_updates_bridge_through_frozen_qwen(spec: dict) -> None:
    model = tiny_model(spec).train()
    assert isinstance(model.talker.embed_proj, nn.Identity)
    assert not model.thinker.training
    tokens = torch.tensor([[0, 1, 3, 4], [1, 12, 3, 4], [1, 13, 4, 0], [0, 1, 3, 5]])
    mask = tokens.ne(0)
    images = [None, torch.tensor([[0.3, 0.2, -0.1]]), None, None]
    audios = [None, None, torch.tensor([[0.1, -0.2, 0.3, 0.4]]), None]
    output = model(tokens, attention_mask=mask, image_features=images, audio_features=audios)
    assert output.logits.shape == (4, 4, 64) and output.audio_logits.shape == (4, 8, 4, 16)
    labels = torch.tensor(
        [[-100, -100, 4, 2], [-100, 3, 4, 2], [-100, 4, 2, -100], [-100, -100, 5, 2]]
    )
    audio_labels = torch.full((4, 8, 4), -100, dtype=torch.long)
    audio_labels[0, :, 2] = 4
    audio_labels[0, :, 3] = model.config.audio_stop_token
    loss = sft_loss(output, labels, audio_labels, stop_token=model.config.audio_stop_token)["total"]
    assert loss.isfinite()
    loss.backward()
    for module in (model.bridge, model.vision_proj, model.audio_proj):
        grads = [p.grad for p in module.parameters()]
        assert all(
            grad is not None and grad.isfinite().all() and grad.abs().max() > 0 for grad in grads
        )
    assert all(not p.requires_grad and p.grad is None for p in model.thinker.parameters())
    before = [p.detach().clone() for p in model.bridge.parameters()]
    torch.optim.AdamW(model.bridge.parameters(), lr=1e-3).step()
    assert all(
        not torch.equal(a, b) for a, b in zip(before, model.bridge.parameters(), strict=True)
    )


@pytest.mark.parametrize("spec", SPECS[1:], ids=[spec["kind"] for spec in SPECS[1:]])
@torch.inference_mode()
def test_candidate_cache_and_streaming_parity(spec: dict) -> None:
    model = tiny_model(spec).eval()
    tokens = torch.tensor([[0, 0, 1, 12, 3], [1, 13, 13, 5, 6]])
    mask = tokens.ne(0)
    image = torch.tensor([[0.2, 0.3, 0.1]])
    audio = torch.arange(8, dtype=torch.float32).view(2, 4)
    features = dict(image_features=[image, None], audio_features=[None, audio])
    prefill = model(tokens, attention_mask=mask, use_cache=True, **features)
    next_ids = torch.tensor([[4], [7]])
    mask = torch.cat((mask, torch.ones_like(next_ids).bool()), dim=-1)
    cached = model(
        next_ids,
        attention_mask=mask,
        past_key_values=prefill.past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    replay = model(
        torch.cat((tokens, next_ids), dim=-1), attention_mask=mask, logits_to_keep=1, **features
    )
    torch.testing.assert_close(cached.logits, replay.logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached.audio_logits, replay.audio_logits, rtol=1e-5, atol=1e-5)
    assert cached.past_key_values.thinker.get_seq_length() == 6
    assert cached.past_key_values.talker[0][0].shape[-2] == 6

    prompt = tokens[:1]
    kwargs = dict(attention_mask=prompt.ne(0), image_features=[image], max_new_tokens=12)

    # Tiny 随机头可能在声码延迟期间提前停止，强制产生完整帧来对照流式路径。
    def suppress_stop(_module: nn.Module, _args: tuple, logits: Tensor) -> None:
        logits[..., model.config.audio_stop_token] = -torch.inf

    with model.talker.lm_head.register_forward_hook(suppress_stop):
        with_cache = list(model.stream_generate(prompt, use_cache=True, **kwargs))
        without_cache = list(model.stream_generate(prompt, use_cache=False, **kwargs))
    assert len(with_cache) == len(without_cache) == 12
    frames = [frame for _, frame in with_cache if frame is not None]
    assert len(frames) == 4 and all(frame.shape == (1, 8, 1) for frame in frames)
    for cached_event, replayed_event in zip(with_cache, without_cache, strict=True):
        for actual, expected in zip(cached_event, replayed_event, strict=True):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
