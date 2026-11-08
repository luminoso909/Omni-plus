"""将 Omni 逐步输出的八码本帧解码成连续波形块。"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache

from .codec import MimiCodec


@dataclass(frozen=True)
class WaveformEvent:
    """一次文本更新、CPU FP32 单声道波形 [samples]，或两者同时到达。"""

    token: Tensor | None = None
    waveform: Tensor | None = None
    frames: int = 0


class MimiStreamingDecoder:
    """单话轮的 Mimi 解码状态，复用 HF Attention KV 与有界因果卷积历史。"""

    def __init__(self, codec: MimiCodec) -> None:
        model = codec.model
        config = model.config
        if (
            not config.use_causal_conv
            or config.trim_right_ratio != 1.0
            or config.pad_mode != "constant"
            or config.audio_channels != 1
            or model.upsample is None
        ):
            raise ValueError("流式 Mimi 需要单声道、常量左填充、因果卷积及右裁剪配置。")

        self.codec = codec
        self._up_stride = model.upsample.conv.stride[0]
        up_kernel = model.upsample.conv.kernel_size[0]
        self._quant_context = math.ceil((up_kernel - self._up_stride) / self._up_stride)
        self._pcm_stride = math.prod(config.upsampling_ratios)

        # 将每级卷积的最大历史依赖换算成输出采样点，再向上取整为 latent 帧。
        residual_lag = (config.residual_kernel_size - 1) * sum(
            config.dilation_growth_rate**i for i in range(config.num_residual_layers)
        )
        lag = config.kernel_size - 1
        for stride in config.upsampling_ratios:
            lag = lag * stride + (2 * stride - 1) + residual_lag
        lag += config.last_kernel_size - 1
        self._latent_context = math.ceil(lag / self._pcm_stride)

        # 无滑窗的 KV 保持与本地 HF 整段 decode 的完整注意力口径一致。
        self._cache = DynamicCache()
        self._quant_tail: Tensor | None = None
        self._latent_tail: Tensor | None = None

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Tensor:
        """接收非空 Long [1,codebooks,frames]，只返回新增的单声道采样点。"""
        codec = self.codec
        if codes.ndim != 3 or codes.shape[:2] != (1, codec.num_codebooks):
            raise ValueError("codes 必须是 [1,num_codebooks,frames]。")
        if (
            codes.dtype != torch.long
            or not codes.shape[-1]
            or ((codes < 0) | (codes >= codec.codebook_size)).any()
        ):
            raise ValueError("流式解码只接受非空的 Long 声学码本。")

        # 上采样卷积只需要上一块末尾少量量化帧；丢弃重算的前缀输出。
        model = codec.model
        device = next(model.parameters()).device
        quantized = model.quantizer.decode(codes.to(device))
        prev_frames = 0
        if self._quant_tail is not None:
            prev_frames = self._quant_tail.shape[-1]
            quantized = torch.cat((self._quant_tail, quantized), dim=-1)
        upsampled = model.upsample(quantized)[..., prev_frames * self._up_stride :]
        self._quant_tail = quantized[
            ..., max(0, quantized.shape[-1] - self._quant_context) :
        ].clone()

        # Transformer 只计算新增帧，历史上下文由 HF KV Cache 提供。
        output = model.decoder_transformer(
            upsampled.transpose(1, 2), past_key_values=self._cache, use_cache=True, return_dict=True
        )
        self._cache = output.past_key_values
        latents = output.last_hidden_state.transpose(1, 2)

        # 最后一级波形网络同样保留有界历史；前缀不进入输出音频块。
        prev_latents = 0
        if self._latent_tail is not None:
            prev_latents = self._latent_tail.shape[-1]
            latents = torch.cat((self._latent_tail, latents), dim=-1)
        waveform = model.decoder(latents)[0, 0, prev_latents * self._pcm_stride :]
        self._latent_tail = latents[..., max(0, latents.shape[-1] - self._latent_context) :].clone()
        expected = codes.shape[-1] * self._up_stride * self._pcm_stride
        if waveform.numel() != expected:
            raise RuntimeError("Mimi 音频块长度与输入帧数不匹配。")
        return waveform.float().cpu().clone()


def stream_waveform(
    events: Iterable[tuple[Tensor | None, Tensor | None]],
    codec: MimiCodec,
    chunk_frames: int = 4,
) -> Iterator[WaveformEvent]:
    """接收 stream_generate 的 token/单帧事件，结束时解码余下的不足一块音频。"""
    if not isinstance(chunk_frames, int) or chunk_frames < 1:
        raise ValueError("chunk_frames 必须是正整数。")

    decoder = MimiStreamingDecoder(codec)
    pending = []
    source = iter(events)
    try:
        for token, frame in source:
            if frame is not None:
                if frame.shape != (1, codec.num_codebooks, 1):
                    raise ValueError("stream_generate 的音频事件必须是单个完整码本帧。")
                pending.append(frame)
            if len(pending) == chunk_frames:
                waveform = decoder.decode(torch.cat(pending, dim=-1))
                pending.clear()
                yield WaveformEvent(token, waveform, chunk_frames)
            elif token is not None:
                yield WaveformEvent(token)

        if pending:
            waveform = decoder.decode(torch.cat(pending, dim=-1))
            yield WaveformEvent(waveform=waveform, frames=len(pending))
    finally:
        # 消费方关闭流时同步释放上游生成器持有的 KV 与历史序列。
        close = getattr(source, "close", None)
        if close is not None:
            close()


def check_chunk_parity(codec: MimiCodec, codes: Tensor, chunk_frames: int = 4) -> dict:
    """把同一组声学码本流式解码与整段解码作逐采样点比较。"""
    reference = codec.decode(codes)
    events = ((None, frame) for frame in codes.split(1, dim=-1))
    chunks = [event.waveform for event in stream_waveform(events, codec, chunk_frames)]
    actual = torch.cat(chunks)
    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    diff = (actual - reference).abs()
    return dict(
        status="passed",
        frames=codes.shape[-1],
        samples=actual.numel(),
        chunk_frames=chunk_frames,
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        rtol=rtol,
        atol=atol,
    )
