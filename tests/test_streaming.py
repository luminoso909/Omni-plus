"""Mimi 流式块、尾块和取消边界，使用随机初始化的极小 HF Codec。"""

import pytest
import torch
from transformers import MimiConfig, MimiModel

from omni.inference import MimiCodec, stream_waveform


@pytest.fixture(scope="module")
def codec() -> MimiCodec:
    config = MimiConfig(
        sampling_rate=800,
        frame_rate=50,
        hidden_size=16,
        num_filters=4,
        upsampling_ratios=[4, 2],
        codebook_size=16,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        num_quantizers=2,
        upsample_groups=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = MimiModel(config)
        for name, buffer in model.named_buffers():
            if name.endswith("embed_sum"):
                buffer.normal_()
    return MimiCodec(model, num_codebooks=2)


def test_stream_preserves_pcm_and_flushes_partial_chunk(codec: MimiCodec) -> None:
    codes = torch.arange(38).reshape(1, 2, 19) % 16
    events = ((torch.tensor([[i]]), frame) for i, frame in enumerate(codes.split(1, dim=-1)))
    output = list(stream_waveform(events, codec, chunk_frames=4))

    assert [int(event.token.item()) for event in output if event.token is not None] == list(
        range(19)
    )
    assert [event.frames for event in output if event.waveform is not None] == [4, 4, 4, 4, 3]
    pcm = torch.cat([event.waveform for event in output if event.waveform is not None])
    assert pcm.shape == (19 * 16,)
    torch.testing.assert_close(pcm, codec.decode(codes), rtol=1e-4, atol=1e-4)


def test_close_releases_upstream_and_next_request_starts_fresh(codec: MimiCodec) -> None:
    closed = []
    frame = torch.tensor([[[1], [2]]])

    def source():
        try:
            yield None, frame
            yield None, frame
        finally:
            closed.append(True)

    stream = stream_waveform(source(), codec, chunk_frames=1)
    first = next(stream).waveform
    stream.close()
    assert closed == [True]

    fresh = list(stream_waveform([(None, frame)], codec, chunk_frames=1))
    torch.testing.assert_close(first, fresh[0].waveform, rtol=0, atol=0)
    text_only = list(stream_waveform([(torch.tensor([[2]]), None)], codec))
    assert len(text_only) == 1 and text_only[0].waveform is None
