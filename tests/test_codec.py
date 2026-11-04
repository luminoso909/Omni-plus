"""延迟码本、冻结 Codec 与波形 IO 边界；不加载真实权重或联网。"""

import io
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from omni.data.audio_alignment import code_mask, delay_codes, restore_codes
from omni.inference.codec import MimiCodec, load_audio, save_audio
from omni.models import OmniPlusConfig, OmniPlusOmni


@pytest.mark.parametrize("frames", [0, 1, 2, 8, 31])
@pytest.mark.parametrize("start", [0, 1, 3])
def test_eight_codebook_delay_roundtrip(frames: int, start: int) -> None:
    codes = torch.arange(2 * 8 * frames).reshape(2, 8, frames) % 2048
    delayed = delay_codes(codes, start=start)
    assert delayed.shape == (2, 8, frames + 8 + start)
    assert code_mask(delayed).sum().item() == codes.numel()
    for index in range(8):
        assert delayed[:, index, start + index + frames].eq(2050).all()
        assert delayed[:, index, : start + index].eq(2049).all()
    torch.testing.assert_close(restore_codes(delayed, frames, start), codes, rtol=0, atol=0)


def test_alignment_rejects_special_tokens_as_acoustic_frames() -> None:
    delayed = delay_codes(torch.ones(8, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="stop/pad"):
        restore_codes(delayed, frames=3)
    with pytest.raises(ValueError, match="时间轴"):
        restore_codes(delayed, frames=100)


def test_omni_generation_restores_all_eight_codebooks_in_codec_order() -> None:
    config = OmniPlusConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        vocab_size=32,
        num_talker_hidden_layers=1,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=32,
        talker_adapter_rank=4,
        audio_codebook_size=16,
        audio_vocab_size=20,
        audio_pad_token=17,
        audio_stop_token=18,
        audio_spk_token=19,
        pad_token_id=0,
        max_position_embeddings=32,
    )
    model = OmniPlusOmni(config).eval()
    expected = (torch.arange(8)[:, None] + torch.arange(3)[None, :])[None]

    def forward(self, input_ids, **kwargs):
        # 文本立即 EOS；各路仍须按独立延迟完成三帧并 stop。
        step = input_ids.shape[-1] - 1
        text = torch.full((1, 1, 32), -100.0)
        text[..., 2] = 100.0
        audio = torch.full((1, 8, 1, 20), -100.0)
        for index in range(8):
            frame = step - 1 - index
            token = int(expected[0, index, frame]) if 0 <= frame < 3 else 18
            audio[:, index, :, token] = 100.0
        return SimpleNamespace(logits=text, audio_logits=audio, past_key_values=None)

    model.forward = MethodType(forward, model)
    _, codes = model.generate(
        torch.tensor([[1]]),
        max_new_tokens=20,
        temperature=0,
        audio_temperature=0,
        use_cache=False,
    )
    torch.testing.assert_close(codes, expected, rtol=0, atol=0)


class TinyCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(sampling_rate=24000, codebook_size=2048)

    def encode(self, waveform, num_quantizers, use_streaming, return_dict):
        assert waveform.shape == (1, 1, 3840)
        assert not use_streaming and return_dict
        return SimpleNamespace(audio_codes=torch.zeros(1, num_quantizers, 2, dtype=torch.long))

    def decode(self, codes, return_dict):
        assert return_dict and codes.dtype == torch.long
        return SimpleNamespace(audio_values=torch.zeros(1, 1, 3840))


def test_codec_remains_frozen_and_excludes_special_tokens() -> None:
    codec = MimiCodec(TinyCodec())
    codec.train()
    assert not codec.training and not codec.model.training
    assert not any(param.requires_grad for param in codec.parameters())
    codes = codec.encode(torch.zeros(3840))
    assert codes.shape == (1, 8, 2)
    assert codec.decode(codes, length=3000).shape == (3000,)
    with pytest.raises(ValueError, match="非空声学"):
        codec.decode(torch.full((1, 8, 1), 2050))
    with pytest.raises(ValueError, match="非空声学"):
        codec.decode(torch.empty(1, 8, 0, dtype=torch.long))


def test_waveform_io_stereo_bytes_and_resampling(tmp_path) -> None:
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")

    values = np.stack((np.ones(1600) * 0.2, np.ones(1600) * 0.4), axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, values, 16000, format="WAV", subtype="FLOAT")
    audio = load_audio(buffer.getvalue(), target_rate=24000)
    assert audio.shape == (2400,) and audio.dtype == torch.float32
    torch.testing.assert_close(audio[100:-100], torch.full((2200,), 0.3), atol=3e-4, rtol=1e-4)
    output = tmp_path / "audio.wav"
    save_audio(output, audio)
    restored = load_audio(output)
    torch.testing.assert_close(restored, audio, atol=1 / 32768, rtol=0)


def test_tensor_waveform_requires_rate_and_has_known_channel_axis() -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        load_audio(torch.zeros(100))
    audio = load_audio(torch.stack((torch.ones(100), torch.zeros(100))), sampling_rate=24000)
    torch.testing.assert_close(audio, torch.full((100,), 0.5))
    with pytest.raises(ValueError, match="非空"):
        load_audio(torch.empty(0), sampling_rate=24000)
