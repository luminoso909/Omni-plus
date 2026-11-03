"""Omni 小模型测试：使用随机特征，不下载编码器、权重或数据。"""

import pytest
import torch

from omni.models.model_omni import OmniPlusOmni
from omni.models.thinker import OmniPlusConfig, OmniPlusForCausalLM


@pytest.fixture
def config() -> OmniPlusConfig:
    return OmniPlusConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        vocab_size=64,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=None,
        num_talker_hidden_layers=1,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=24,
        num_audio_codebooks=2,
        talker_adapter_rank=4,
        audio_codebook_size=16,
        audio_vocab_size=20,
        audio_pad_token=17,
        audio_stop_token=18,
        audio_spk_token=19,
        spk_emb_size=6,
        audio_hidden_size=8,
        image_hidden_size=10,
        bridge_layer=0,
    )


def test_forward_with_different_hidden_sizes(config: OmniPlusConfig) -> None:
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 3, 5, 7], [1, 4, 6, 8]])
    output = model(tokens, use_cache=True)

    assert output.logits.shape == (2, 4, 64)
    assert output.audio_logits.shape == (2, 2, 4, 20)
    assert output.aux_loss.ndim == 0
    assert len(output.past_key_values) == 3
    assert output.past_key_values[0][0].shape == (2, 2, 4, 8)
    assert output.past_key_values[-1][0].shape == (2, 1, 4, 8)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.audio_logits).all()
    assert not any(key.startswith("thinker.") for key in model.state_dict())


def test_text_thinker_matches_language_model(config: OmniPlusConfig) -> None:
    model = OmniPlusOmni(config).eval()
    lm = OmniPlusForCausalLM(config).eval()
    state = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("model.") or key == "lm_head.weight"
    }
    lm.load_state_dict(state, strict=True)
    tokens = torch.tensor([[1, 3, 4, 5]])

    torch.testing.assert_close(model(tokens).logits, lm(tokens).logits)
    torch.testing.assert_close(model(tokens, logits_to_keep=1).logits, model(tokens).logits[:, -1:])


def test_mixed_batch_feature_injection(config: OmniPlusConfig) -> None:
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 16, 16, 3, 4, 5], [1, 12, 12, 12, 6, 7]])
    audio = [torch.zeros(2, 8), None]
    images = [None, torch.randn(3, 10)]

    # 保留原 batch 索引；全零音频特征仍应替换占位符。
    expected = model.model.embed_tokens(tokens).clone()
    expected[0, 1:3] = model.audio_proj(audio[0])
    expected[1, 1:4] = model.vision_proj(images[1])
    hidden, _, _, _ = model.model(inputs_embeds=expected)
    actual = model(tokens, audio_features=audio, image_features=images)

    torch.testing.assert_close(actual.logits, model.lm_head(hidden))


def test_single_sample_multiple_images(config: OmniPlusConfig) -> None:
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 12, 12, 3, 12, 12, 4]])
    # 两张图各占两个 Token，特征按图片顺序展平。
    images = torch.randn(4, 10)
    expected = model.model.embed_tokens(tokens).clone()
    expected[0, tokens[0] == 12] = model.vision_proj(images)
    hidden, _, _, _ = model.model(inputs_embeds=expected)

    actual = model(tokens, image_features=[images])
    torch.testing.assert_close(actual.logits, model.lm_head(hidden))


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_feature_count_must_match_placeholders(config: OmniPlusConfig, modality: str) -> None:
    model = OmniPlusOmni(config)
    marker, width = (16, 8) if modality == "audio" else (12, 10)
    tokens = torch.tensor([[1, marker, marker, 3]])
    kwargs = {f"{modality}_features": [torch.randn(1, width)]}

    with pytest.raises(ValueError):
        model(tokens, **kwargs)


@pytest.mark.parametrize("use_moe", [False, True])
def test_multimodal_cache_matches_full_forward(config: OmniPlusConfig, use_moe: bool) -> None:
    config.use_moe = use_moe
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 16, 12, 3, 4, 5], [1, 12, 16, 6, 7, 8]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    audio_ids = torch.randint(0, 16, (2, 2, 6))
    audio = [torch.randn(1, 8), torch.randn(1, 8)]
    images = [torch.randn(1, 10), torch.randn(1, 10)]

    with torch.no_grad():
        full = model(
            tokens, audio_ids, attention_mask=mask, audio_features=audio, image_features=images
        )
        prefix = model(
            tokens[:, :4],
            audio_ids[:, :, :4],
            attention_mask=mask[:, :4],
            audio_features=audio,
            image_features=images,
            use_cache=True,
        )
        # Decode 不应再次消费原始模态特征或替换占位符。
        suffix = model(
            tokens[:, 4:],
            audio_ids[:, :, 4:],
            attention_mask=mask,
            audio_features=audio,
            image_features=images,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )

    torch.testing.assert_close(suffix.logits, full.logits[:, 4:], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        suffix.audio_logits, full.audio_logits[:, :, 4:], rtol=3e-5, atol=3e-6
    )
    assert all(pair[0].shape[-2] == 6 for pair in suffix.past_key_values)


@pytest.mark.parametrize("use_moe", [False, True])
def test_multimodal_backward(config: OmniPlusConfig, use_moe: bool) -> None:
    config.use_moe = use_moe
    model = OmniPlusOmni(config).train()
    tokens = torch.tensor([[1, 16, 12, 3]])
    audio_ids = torch.randint(0, 16, (1, 2, 4))
    audio_ids[:, :, 1] = config.audio_spk_token
    audio = torch.randn(1, 8, requires_grad=True)
    image = torch.randn(1, 10, requires_grad=True)
    speaker = torch.randn(1, 6, requires_grad=True)

    result = model(
        tokens, audio_ids, audio_features=[audio], image_features=[image], spk_emb=speaker
    )
    loss = result.logits.square().mean() + result.audio_logits.square().mean() + result.aux_loss
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    for feature in (audio, image, speaker):
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0


@pytest.mark.parametrize("tied", [False, True])
def test_omni_hf_weight_roundtrip(config: OmniPlusConfig, tied: bool, tmp_path) -> None:
    config.tie_word_embeddings = tied
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 3, 5, 7]])
    expected = model(tokens)
    model.save_pretrained(tmp_path)
    loaded = OmniPlusOmni.from_pretrained(tmp_path, local_files_only=True).eval()
    actual = loaded(tokens)

    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tied
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.audio_logits, expected.audio_logits)
    assert set(loaded.state_dict()) == set(model.state_dict())


@pytest.mark.parametrize("bridge_layer", [0, 1])
def test_bridge_uses_selected_layer_before_norm(config: OmniPlusConfig, bridge_layer: int) -> None:
    config.bridge_layer = bridge_layer
    model = OmniPlusOmni(config).eval()
    tokens = torch.tensor([[1, 3, 5, 7]])
    captured = {}

    def capture_layer(module, args, output) -> None:
        captured["layer"] = output[0].detach().clone()

    def capture_bridge(module, args, kwargs) -> None:
        bridge = args[0] if args else kwargs["bridge"]
        captured["bridge"] = bridge.detach().clone()

    layer_hook = model.model.layers[bridge_layer].register_forward_hook(capture_layer)
    bridge_hook = model.talker.register_forward_pre_hook(capture_bridge, with_kwargs=True)
    model(tokens)
    layer_hook.remove()
    bridge_hook.remove()

    torch.testing.assert_close(captured["bridge"], captured["layer"])
