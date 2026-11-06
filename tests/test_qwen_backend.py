"""用本地极小 Qwen3 对照 Backend 的张量、位置和缓存契约。"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache

from omni.models.thinker.qwen_backend import QwenBackend

RTOL = 1e-5
ATOL = 1e-5


@pytest.fixture
def model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=None,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return Qwen3ForCausalLM(config).eval().requires_grad_(False)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]]), torch.tensor(
        [[0, 0, 1, 1], [1, 1, 1, 1]]
    )


@pytest.mark.parametrize("hidden_layer", [0, 1, 2])
@torch.inference_mode()
def test_native_logits_and_selected_block(
    model: Qwen3ForCausalLM, batch: tuple[torch.Tensor, torch.Tensor], hidden_layer: int
) -> None:
    tokens, mask = batch
    backend = QwenBackend(model)
    positions = torch.tensor([[1, 1, 2, 4], [0, 2, 4, 7]])
    native = model(
        input_ids=tokens,
        attention_mask=mask,
        position_ids=positions,
        cache_position=torch.arange(tokens.shape[1]),
        output_hidden_states=True,
        use_cache=False,
    )
    actual = backend(
        input_ids=tokens,
        attention_mask=mask,
        position_ids=positions,
        cache_position=torch.arange(tokens.shape[1]),
        hidden_layer=hidden_layer,
        logits_to_keep=2,
    )

    assert actual.logits.shape == (2, 2, model.config.vocab_size)
    assert actual.selected_hidden_state.shape == (2, 4, model.config.hidden_size)
    assert actual.past_key_values is None
    torch.testing.assert_close(actual.logits, native.logits[:, -2:], rtol=RTOL, atol=ATOL)

    # HF 记录的末层已经过最终 norm，Backend 始终返回原始 Block 输出。
    hidden = actual.selected_hidden_state
    reference = native.hidden_states[hidden_layer + 1]
    if hidden_layer == model.config.num_hidden_layers - 1:
        assert not torch.allclose(hidden, reference, rtol=RTOL, atol=ATOL)
        hidden = model.model.norm(hidden)
    torch.testing.assert_close(hidden, reference, rtol=RTOL, atol=ATOL)


def test_embedding_input_keeps_gradient_through_frozen_model(
    model: Qwen3ForCausalLM, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    tokens, mask = batch
    backend = QwenBackend(model)
    embeddings = backend.embed_tokens(tokens).detach().requires_grad_(True)
    torch.testing.assert_close(embeddings, model.get_input_embeddings()(tokens))

    by_ids = backend(input_ids=tokens, attention_mask=mask)
    by_embeddings = backend(inputs_embeds=embeddings, attention_mask=mask, hidden_layer=1)
    assert by_ids.selected_hidden_state is None
    torch.testing.assert_close(by_embeddings.logits, by_ids.logits, rtol=RTOL, atol=ATOL)
    (gradient,) = torch.autograd.grad(
        by_embeddings.selected_hidden_state.square().mean(), embeddings
    )
    assert gradient.shape == embeddings.shape
    assert torch.isfinite(gradient).all()
    assert gradient[mask.bool()].abs().max() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


@torch.inference_mode()
def test_left_padding_uses_native_generation_positions(
    model: Qwen3ForCausalLM, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    tokens, mask = batch
    backend = QwenBackend(model)
    prepared = model.prepare_inputs_for_generation(
        tokens, attention_mask=mask, cache_position=torch.arange(tokens.shape[1]), use_cache=False
    )
    assert prepared["position_ids"].tolist() == [[1, 1, 0, 1], [0, 1, 2, 3]]
    native = model(**prepared).logits
    actual = backend(input_ids=tokens, attention_mask=mask.bool()).logits

    torch.testing.assert_close(actual, native, rtol=RTOL, atol=ATOL)
    unpadded = model(input_ids=tokens[:1, 2:], use_cache=False).logits
    torch.testing.assert_close(actual[:1, 2:], unpadded, rtol=RTOL, atol=ATOL)


@torch.inference_mode()
def test_prefill_decode_matches_native_and_full_recompute(
    model: Qwen3ForCausalLM, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    tokens, mask = batch
    backend = QwenBackend(model)
    native_ids = model.generate(
        input_ids=tokens,
        attention_mask=mask,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        use_model_defaults=False,
    )
    native = model(
        **model.prepare_inputs_for_generation(
            tokens,
            attention_mask=mask,
            cache_position=torch.arange(tokens.shape[1]),
            use_cache=True,
        )
    )
    actual = backend(input_ids=tokens, attention_mask=mask, use_cache=True, hidden_layer=1)
    cache = actual.past_key_values
    assert isinstance(cache, Cache)
    assert cache is not native.past_key_values
    assert cache.get_seq_length() == tokens.shape[1]
    torch.testing.assert_close(actual.logits, native.logits, rtol=RTOL, atol=ATOL)

    for next_native in native_ids[:, -2:].split(1, dim=1):
        next_token = actual.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.testing.assert_close(next_token, next_native)
        cache_position = torch.tensor([tokens.shape[1]])
        tokens = torch.cat((tokens, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        position_ids = mask.sum(dim=-1, keepdim=True) - 1
        assert position_ids[0, 0] < cache_position[0]

        native = model(
            **model.prepare_inputs_for_generation(
                tokens,
                attention_mask=mask,
                cache_position=cache_position,
                past_key_values=native.past_key_values,
                use_cache=True,
            ),
            output_hidden_states=True,
        )
        actual = backend(
            input_ids=next_token,
            attention_mask=mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            hidden_layer=1,
        )
        recomputed = backend(
            input_ids=tokens, attention_mask=mask, hidden_layer=1, logits_to_keep=1
        )
        assert actual.past_key_values is cache
        assert cache.get_seq_length() == tokens.shape[1]
        assert recomputed.past_key_values is None
        torch.testing.assert_close(actual.logits, native.logits, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            actual.selected_hidden_state, native.hidden_states[2], rtol=RTOL, atol=ATOL
        )
        torch.testing.assert_close(actual.logits, recomputed.logits, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            actual.selected_hidden_state,
            recomputed.selected_hidden_state[:, -1:],
            rtol=RTOL,
            atol=ATOL,
        )

    fresh = backend(input_ids=tokens[:, -2:], use_cache=True)
    assert fresh.past_key_values is not cache
    assert fresh.past_key_values.get_seq_length() == 2
    assert cache.get_seq_length() == tokens.shape[1]
    native_fresh = model(input_ids=tokens[:, -2:], use_cache=False)
    torch.testing.assert_close(fresh.logits, native_fresh.logits, rtol=RTOL, atol=ATOL)


def test_invalid_inputs_and_cache_are_rejected(model: Qwen3ForCausalLM) -> None:
    backend = QwenBackend(model)
    tokens = torch.tensor([[11, 12]])
    with pytest.raises(ValueError):
        backend()
    with pytest.raises(ValueError):
        backend(input_ids=tokens, inputs_embeds=backend.embed_tokens(tokens))
    with pytest.raises(ValueError, match="hidden_layer"):
        backend(input_ids=tokens, hidden_layer=model.config.num_hidden_layers)
    with pytest.raises(ValueError, match="nonnegative"):
        backend(input_ids=tokens, logits_to_keep=-1)

    cache = backend(input_ids=tokens, use_cache=True).past_key_values
    with pytest.raises(ValueError):
        backend(input_ids=tokens, past_key_values=cache, use_cache=False)
    with pytest.raises(ValueError, match="attention_mask"):
        backend(
            input_ids=tokens[:, -1:],
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
    assert cache.get_seq_length() == tokens.shape[1]


@torch.inference_mode()
def test_selected_layer_capture_is_removed_after_failure(model: Qwen3ForCausalLM) -> None:
    backend = QwenBackend(model)
    layers = model.model.layers
    original_forwards = tuple(layer.forward for layer in layers)
    original_hooks = len(layers[1]._forward_hooks)

    with pytest.raises(IndexError):
        backend(input_ids=torch.tensor([[model.config.vocab_size]]), hidden_layer=1)

    assert tuple(layer.forward for layer in layers) == original_forwards
    assert len(layers[1]._forward_hooks) == original_hooks
    tokens = torch.tensor([[11, 12]])
    actual = backend(input_ids=tokens, hidden_layer=1)
    assert len(layers[1]._forward_hooks) == original_hooks
    native = model(input_ids=tokens, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(actual.selected_hidden_state, native.hidden_states[2])
    torch.testing.assert_close(actual.logits, native.logits)
