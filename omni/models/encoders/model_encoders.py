"""冻结的本地媒体编码器与特征投影；构造 Omni 时不隐式加载外部模型。"""

import io
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Projector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        # 将编码器特征投影到 Thinker 宽度，不改变 token 数
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: Float[Tensor, "... features"]) -> Float[Tensor, "... hidden"]:
        return self.mlp(x)


class SenseVoiceEncoder(nn.Module):
    """本地 SenseVoice 的声学特征；单样本返回 CPU FP32 [frames, 512]。"""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        from funasr import AutoModel

        # 只接受已有本地目录，关闭版本检查和远程代码
        path = Path(path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("SenseVoice path 必须为本地模型目录。")

        model = AutoModel(
            model=str(path),
            trust_remote_code=False,
            disable_update=True,
            disable_pbar=True,
            ignore_init_mismatch=False,
            device="cpu",
        )
        self.encoder = model.model.encoder
        self.frontend = model.kwargs["frontend"]

        # FunASR 负责构造；单独严格核对 encoder，避免上游宽松加载保留随机参数
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        self.encoder.load_state_dict(
            {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            },
            strict=True,
        )

        # 关闭 fbank 随机抖动，冻结两个模块且始终使用 eval
        self.frontend.dither = 0.0
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "SenseVoiceEncoder":
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self, source: str | Path | bytes | Tensor, sampling_rate: int | None = None
    ) -> Float[Tensor, "frames features"]:
        from omni.inference.codec import load_audio

        # fbank 在 CPU FP32 计算；波形统一为 16 kHz 单声道
        wav = load_audio(source, sampling_rate=sampling_rate, target_rate=16000)
        features, lengths = self.frontend(wav[None], torch.tensor([wav.numel()]))

        # Encoder 跟随 .to(device, dtype)，仅返回有效长度，不保留 batch padding
        param = next(self.encoder.parameters())
        features, lengths = self.encoder(features.to(param), lengths.to(device=param.device))
        return features[0, : int(lengths[0])].detach().float().cpu()


class SiglipEncoder(nn.Module):
    """本地 SigLIP2 patch 特征；单张图片返回 CPU FP32 [patches, 768]。"""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        from transformers import SiglipImageProcessor, SiglipVisionModel

        # 模型与预处理配置来自同一目录，禁止隐式下载
        path = Path(path).resolve(strict=True)
        self.encoder = SiglipVisionModel.from_pretrained(path, local_files_only=True)
        self.processor = SiglipImageProcessor.from_pretrained(path, local_files_only=True)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "SiglipEncoder":
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, source: str | Path | bytes) -> Float[Tensor, "patches features"]:
        from PIL import Image

        # 明确转换 RGB，bytes 与路径共用同一预处理
        content = io.BytesIO(source) if isinstance(source, bytes) else source
        with Image.open(content) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")

        # 仅编码单张图片，Dataset 负责保持各样本/多图的原始顺序
        param = next(self.encoder.parameters())
        pixels = inputs["pixel_values"].to(param)
        features = self.encoder(pixel_values=pixels).last_hidden_state
        return features[0].detach().float().cpu()
