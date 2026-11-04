"""本地 Mimi 与波形 IO；编码器不在 Omni 构造函数中隐式加载。"""

import io
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


def load_audio(
    source: str | Path | bytes | Tensor,
    sampling_rate: int | None = None,
    target_rate: int = 24000,
) -> Tensor:
    """返回 CPU float32 单声道 [samples]；Tensor 输入为 [samples] 或 [channels,samples]。"""
    # 文件自带采样率；内存 Tensor 必须明确原始采样率
    if isinstance(source, Tensor):
        if sampling_rate is None:
            raise ValueError("Tensor 波形需要 sampling_rate，避免误判时间轴。")
        audio = source.detach().float().cpu()
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
    else:
        import soundfile as sf

        stream = io.BytesIO(source) if isinstance(source, bytes) else source
        values, sampling_rate = sf.read(stream, dtype="float32", always_2d=True)
        audio = torch.from_numpy(values.mean(axis=1))
    if audio.ndim != 1 or not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError("波形必须是非空且有限的单声道序列。")
    if sampling_rate <= 0 or target_rate <= 0:
        raise ValueError("采样率必须为正数。")

    # 有理数重采样，避免用插值改变音频频谱
    if sampling_rate != target_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(sampling_rate, target_rate)
        values = resample_poly(audio.numpy(), target_rate // divisor, sampling_rate // divisor)
        audio = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return audio.contiguous()


def save_audio(path: str | Path, waveform: Tensor, sampling_rate: int = 24000) -> None:
    """将单声道 [samples] 写为可播放的 PCM16 WAV；不改变内存中的波形。"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().float().cpu().numpy(), sampling_rate, subtype="PCM_16")


class MimiCodec(nn.Module):
    def __init__(self, model: nn.Module, num_codebooks: int = 8) -> None:
        super().__init__()

        # 模型由调用方用标准 HF 接口离线加载，保留可测的依赖注入
        self.model = model.requires_grad_(False).eval()
        self.num_codebooks = num_codebooks
        self.sampling_rate = model.config.sampling_rate
        self.codebook_size = model.config.codebook_size
        self.eval()

    def train(self, mode: bool = True) -> "MimiCodec":
        # Codec 始终冻结，外部训练模式不能开启其 dropout
        super().train(False)
        return self

    @torch.inference_mode()
    def encode(self, waveform: Tensor) -> Tensor:
        """24kHz 单声道 [samples] → Long [1,codebooks,frames]，不保留流式 Cache。"""
        if waveform.ndim != 1 or not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("encode 需要非空、有限的单声道 [samples] 波形。")

        param = next(self.model.parameters())
        waveform = waveform.to(device=param.device, dtype=param.dtype).reshape(1, 1, -1)
        return self.model.encode(
            waveform, num_quantizers=self.num_codebooks, use_streaming=False, return_dict=True
        ).audio_codes

    @torch.inference_mode()
    def decode(self, codes: Tensor, length: int | None = None) -> Tensor:
        """Long [1,codebooks,frames] → 单声道 [samples]；length 可去掉末尾 codec padding。"""
        if (
            codes.ndim != 3
            or codes.shape[:2] != (1, self.num_codebooks)
            or codes.dtype != torch.long
        ):
            raise ValueError("Mimi 输入必须是 [1,num_codebooks,frames]。")
        if codes.shape[-1] == 0 or ((codes < 0) | (codes >= self.codebook_size)).any():
            raise ValueError("Mimi 只接受非空声学 codes；先移除 delay、stop 与 pad。")

        device = next(self.model.parameters()).device
        audio = self.model.decode(codes.to(device=device, dtype=torch.long), return_dict=True)
        return audio.audio_values[0, 0, :length].float().cpu()
