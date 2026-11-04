"""冻结编码器的本地路径、生命周期和有效长度；不下载真实权重。"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder


class AudioFrontend(nn.Module):
    def forward(self, wav, lengths):
        assert wav.device.type == "cpu" and wav.dtype == torch.float32
        return torch.ones(1, 4, 3), torch.tensor([4])


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 5)

    def forward(self, features, lengths):
        assert features.dtype == self.proj.weight.dtype
        return self.proj(features), lengths - 2


def test_sensevoice_freezes_modules_and_trims_encoder_length(tmp_path, monkeypatch):
    options = {}

    def load_model(**kwargs):
        options.update(kwargs)
        return SimpleNamespace(
            model=SimpleNamespace(encoder=AudioEncoder()), kwargs={"frontend": AudioFrontend()}
        )

    def load_audio(source, sampling_rate, target_rate):
        assert source == b"audio" and sampling_rate is None and target_rate == 16000
        return torch.zeros(1600)

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=load_model))
    monkeypatch.setitem(sys.modules, "omni.inference.codec", SimpleNamespace(load_audio=load_audio))
    torch.save(
        {f"encoder.{key}": value for key, value in AudioEncoder().state_dict().items()},
        tmp_path / "model.pt",
    )
    model = SenseVoiceEncoder(tmp_path).to(dtype=torch.float64).train()

    features = model(b"audio")
    assert features.shape == (2, 5)
    assert features.dtype == torch.float32 and features.device.type == "cpu"
    assert not features.requires_grad
    assert not model.training and not model.encoder.training and not model.frontend.training
    assert not any(param.requires_grad for param in model.parameters())
    assert model.frontend.dither == 0
    assert options["disable_update"] and not options["trust_remote_code"]
    assert "encoder.proj.weight" in model.state_dict()


def test_missing_sensevoice_path_does_not_download(tmp_path, monkeypatch):
    def forbid_load(**kwargs):
        raise AssertionError("缺失本地目录时不能调用模型加载或下载")

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=forbid_load))
    with pytest.raises(FileNotFoundError):
        SenseVoiceEncoder(tmp_path / "missing")


def test_siglip_is_registered_frozen_and_preserves_patch_order(tmp_path, monkeypatch):
    options = []

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 5)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            options.append(kwargs)
            return cls()

        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=self.proj(pixel_values))

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            options.append(kwargs)
            return cls()

        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.arange(12).reshape(1, 4, 3).float()}

    class Image:
        @staticmethod
        def open(source):
            return Image()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return self

    # Transformers 的 LazyModule 可能被后续导入替换；隔离构造时实际使用的模块入口。
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(SiglipVisionModel=Vision, SiglipImageProcessor=Processor),
    )
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=Image))
    model = SiglipEncoder(tmp_path).to(dtype=torch.float64).train()

    features = model(b"image")
    expected = model.encoder.proj(torch.arange(12).reshape(4, 3).double()).float()
    torch.testing.assert_close(features, expected)
    assert features.shape == (4, 5) and not features.requires_grad
    assert not model.training and not model.encoder.training
    assert not any(param.requires_grad for param in model.parameters())
    assert options == [{"local_files_only": True}] * 2
    assert "encoder.proj.weight" in model.state_dict()
