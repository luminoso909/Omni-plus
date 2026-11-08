"""离线验证 Qwen 与 Talker 的组合契约。"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from omni.models.config import OmniPlusConfig
from omni.models.model_omni import OmniOutput
from omni.models.model_omni_v2 import OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.losses import sft_loss


@pytest.fixture
def model() -> OmniPlusOmni:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        qwen = Qwen3ForCausalLM(
            Qwen3Config(
                hidden_size=24,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=3,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=64,
            )
        )
        config = OmniPlusConfig(
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=3,
            num_key_value_heads=1,
            vocab_size=64,
            image_ids=[12],
            image_hidden_size=3,
            audio_ids=[13],
            audio_hidden_size=4,
            num_talker_hidden_layers=2,
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
            eos_token_id=[2, 0],
        )
        return OmniPlusOmni(QwenBackend(qwen), config)


def test_modal_projection_and_frozen_thinker_backward(model: OmniPlusOmni) -> None:
    model.train()
    assert not model.thinker.training
    assert isinstance(model.talker.embed_proj, torch.nn.Identity)
    tokens = torch.tensor([[1, 12, 12, 3, 4], [1, 13, 3, 4, 0], [1, 3, 4, 0, 0]])
    mask = tokens.ne(0)
    images = [torch.arange(6, dtype=torch.float32).view(2, 3), None, None]
    audios = [None, torch.zeros(1, 4), None]
    output = model(
        tokens,
        attention_mask=mask,
        image_features=images,
        audio_features=audios,
        spk_emb=torch.zeros(3, model.config.spk_emb_size),
    )
    assert output.logits.shape == (3, 5, 64)
    assert output.audio_logits.shape == (3, 8, 5, 16)

    labels = torch.full_like(tokens, -100)
    labels[:, 2] = 4
    audio_labels = torch.full((3, 8, 5), -100, dtype=torch.long)
    ignored = sft_loss(output, torch.full_like(labels, -100), audio_labels)
    assert all(value.item() == 0 for value in ignored.values())
    audio_labels[:2, :, 2] = 2
    loss = sft_loss(output, labels, audio_labels, stop_token=model.config.audio_stop_token)
    assert all(value.isfinite() for value in loss.values())
    loss["total"].backward()

    for module in (model.vision_proj, model.audio_proj, model.bridge, model.talker.lm_head):
        grads = [param.grad for param in module.parameters()]
        assert all(grad is not None and grad.isfinite().all() for grad in grads)
        assert any(grad.abs().max() > 0 for grad in grads)
    assert all(
        not param.requires_grad and param.grad is None for param in model.thinker.parameters()
    )
    assert model.talker.spk_proj.weight.grad is None


@torch.inference_mode()
def test_cached_left_padding_and_modal_reorder(model: OmniPlusOmni) -> None:
    model.eval()
    tokens = torch.tensor([[0, 0, 1, 12, 12, 3], [1, 13, 13, 13, 3, 4]])
    mask = tokens.ne(0)
    image = torch.arange(6, dtype=torch.float32).view(2, 3)
    audio = torch.arange(12, dtype=torch.float32).view(3, 4)
    features = dict(image_features=[image, None], audio_features=[None, audio])
    start = model(tokens, attention_mask=mask, use_cache=True, **features)
    original = start.past_key_values
    assert original.thinker.get_seq_length() == 6
    assert original.talker[0][0].shape == (2, 1, 6, 8)

    # 先核对去掉左 padding 和重排样本，再单独对照 Cache 解码。
    single = model(tokens[:1, 2:], image_features=[image])
    permuted = model(
        tokens.flip(0),
        attention_mask=mask.flip(0),
        image_features=[None, image],
        audio_features=[audio, None],
    )
    for key in ("logits", "audio_logits"):
        current = getattr(start, key)
        torch.testing.assert_close(current, getattr(permuted, key).flip(0), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            current[:1, ..., 2:, :], getattr(single, key), rtol=1e-5, atol=1e-5
        )

    next_ids = torch.tensor([[5], [6]])
    full_mask = torch.cat((mask, torch.ones_like(next_ids).bool()), dim=-1)
    step = model(next_ids, attention_mask=full_mask, past_key_values=original, use_cache=True)
    replay = model(
        torch.cat((tokens, next_ids), dim=-1),
        attention_mask=full_mask,
        logits_to_keep=1,
        **features,
    )
    torch.testing.assert_close(step.logits, replay.logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(step.audio_logits, replay.audio_logits, rtol=1e-5, atol=1e-5)
    assert step.past_key_values.thinker is original.thinker
    assert step.past_key_values.thinker.get_seq_length() == 7
    assert step.past_key_values.talker[0][0].shape[-2] == 7
    fresh = model(next_ids, use_cache=True).past_key_values
    assert fresh.thinker is not original.thinker and fresh.thinker.get_seq_length() == 1
    assert fresh.talker[0][0].shape[-2] == 1


@torch.inference_mode()
def test_future_codes_do_not_change_current_logits(model: OmniPlusOmni) -> None:
    model.eval()
    tokens = torch.tensor([[1, 3, 4, 5, 6]])
    audio = torch.full((1, 8, 5), model.config.audio_pad_token)
    changed = audio.clone()
    changed[..., 3:] = 2
    expected = model(tokens, audio_ids=audio).audio_logits
    actual = model(tokens, audio_ids=changed).audio_logits
    torch.testing.assert_close(expected[..., :3, :], actual[..., :3, :], rtol=1e-5, atol=1e-5)
    assert not torch.equal(expected[..., 3:, :], actual[..., 3:, :])


def test_missing_media_features_are_rejected(model: OmniPlusOmni) -> None:
    with pytest.raises(ValueError, match="占位缺少"):
        model(torch.tensor([[1, 12, 3]]))
    with pytest.raises(ValueError, match="长度"):
        model(torch.tensor([[1, 13, 3]]), audio_features=[torch.zeros(2, 4)])


def test_stream_eos_audio_tail_budget_and_cancel(model: OmniPlusOmni, monkeypatch) -> None:
    calls = []
    config = model.config

    def scripted(input_ids, **kwargs):
        step = len(calls)
        calls.append((input_ids.clone(), kwargs["audio_ids"].clone()))
        text = torch.zeros(1, 1, config.vocab_size)
        text[..., config.eos_token_id[0]] = 1
        audio = torch.zeros(1, config.num_audio_codebooks, 1, config.audio_vocab_size)
        for codebook in range(config.num_audio_codebooks):
            index = codebook if step == codebook + 1 else config.audio_stop_token
            audio[:, codebook, :, index] = 1
        return OmniOutput(logits=text, audio_logits=audio)

    monkeypatch.setattr(model, "forward", scripted)
    prompt = torch.tensor([[0, 1, 3]])
    mask = prompt.ne(0)
    with pytest.raises(ValueError, match="上下文"):
        next(model.stream_generate(prompt, max_new_tokens=config.max_position_embeddings))
    assert not calls
    events = list(model.stream_generate(prompt, attention_mask=mask, max_new_tokens=20))
    assert [token.item() for token, _ in events if token is not None] == [2]
    assert len(events) == 10
    frames = [frame for _, frame in events if frame is not None]
    torch.testing.assert_close(torch.cat(frames, dim=-1), torch.arange(8).view(1, 8, 1))

    calls.clear()
    truncated = list(model.stream_generate(prompt, attention_mask=mask, max_new_tokens=5))
    assert len(truncated) == 5 and all(frame is None for _, frame in truncated)
    calls.clear()
    stream = model.stream_generate(prompt, attention_mask=mask)
    next(stream)
    stream.close()
    assert list(stream) == [] and len(calls) == 1
