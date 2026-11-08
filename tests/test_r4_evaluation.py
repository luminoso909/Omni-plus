"""评测数学、真实生成路径隔离和本地 ASR 接口；音频为机制 fixture。"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_r3_training import batch, tiny_model
from torch import nn

from omni.evaluation.sft import (
    evaluate_sft,
    generate_audio,
    load_local_asr,
    modality_swap,
    transcription_errors,
)


class ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_audio_codebooks=2, audio_stop_token=3)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        batch, length = input_ids.shape
        return SimpleNamespace(
            logits=torch.zeros(batch, length, 4),
            audio_logits=torch.zeros(batch, 2, length, 4),
            aux_loss=None,
        )


def test_weighted_per_codebook_ce_and_stop_aggregate() -> None:
    batches = [
        dict(
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[1, -100]]),
            audio_labels=torch.tensor([[[1, 3], [3, -100]]]),
        ),
        dict(
            input_ids=torch.tensor([[1]]),
            labels=torch.tensor([[2]]),
            audio_labels=torch.tensor([[[1], [2]]]),
        ),
    ]
    result = evaluate_sft(ConstantModel(), batches, stop_weight=2)
    expected = [math.log(4) * 4 / 3, math.log(4) * 3 / 2]
    assert result["codebook_loss"] == pytest.approx(expected)
    assert result["audio"] == pytest.approx(sum(expected) / 2)
    assert result["text"] == pytest.approx(math.log(4))
    assert result["audio_tokens"] == [3, 2]
    assert result["stop_targets"] == 2
    assert result["stop_hits"] == result["stop_predicted"] == 0
    assert result["stop_recall"] == 0
    assert result["stop_precision"] is None


def test_all_ignored_evaluation_has_no_fake_stop_score() -> None:
    result = evaluate_sft(
        ConstantModel(),
        [
            dict(
                input_ids=torch.tensor([[1]]),
                labels=torch.full((1, 1), -100),
                audio_labels=torch.full((1, 2, 1), -100),
            )
        ],
    )
    assert result["total"] == 0
    assert result["codebook_loss"] == [0, 0]
    assert result["stop_recall"] is result["stop_precision"] is None


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_modality_swap_is_measured_only_with_matching_pair(kind: str) -> None:
    model = tiny_model().eval()
    left, right = batch(0, modality=kind), batch(1, modality=kind)
    key = f"{kind}_features"
    assert modality_swap(model, left, kind)["status"] == "not_measured"
    merged = {name: torch.cat([left[name], right[name]]) for name in left if name != key}
    features = [torch.randn_like(left[key][0]), torch.randn_like(right[key][0])]
    merged[key] = features
    result = modality_swap(model, merged, kind)
    assert result["status"] == "measured" and result["text_max_abs_diff"] > 0
    assert merged[key] is features


def test_transcription_edit_distance_and_missing_reference() -> None:
    result = transcription_errors("a b", "a c d")
    assert result["character_edits"] == result["word_edits"] == 2
    assert result["cer"] == result["wer"] == 1
    assert transcription_errors("你好世界", "你好")["cer"] == 0.5
    assert transcription_errors("", "unexpected")["cer"] is None
    assert transcription_errors("", "unexpected")["wer"] is None


class GenerationFixture(nn.Module):
    def __init__(self, frames: bool = True) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(audio_pad_token=9)
        self.frames = frames
        self.seen = None

    def stream_generate(self, input_ids: torch.Tensor, **kwargs):
        self.seen = input_ids.clone()
        yield torch.tensor([[4]]), None
        if self.frames:
            yield None, torch.zeros(1, 8, 1, dtype=torch.long)


class CodecFixture:
    sampling_rate = 24000

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.shape == (1, 8, 1)
        return torch.linspace(-0.1, 0.1, 240)


class TokenizerFixture:
    def decode(self, tokens: list[int], **kwargs) -> str:
        return "a b" if tokens == [4, 2] else "a"


def generation_sample() -> dict:
    return dict(
        input_ids=torch.tensor([1, 3, 4, 2]),
        attention_mask=torch.ones(4, dtype=torch.long),
        labels=torch.tensor([-100, 4, 2, -100]),
        audio_ids=torch.full((8, 4), 9, dtype=torch.long),
        audio_features=None,
        image_features=None,
    )


@pytest.mark.parametrize("asr", [False, True])
def test_generation_uses_prompt_only_and_exports_prediction(tmp_path: Path, asr: bool) -> None:
    model = GenerationFixture()
    result = generate_audio(
        model,
        TokenizerFixture(),
        generation_sample(),
        CodecFixture(),
        tmp_path / "example",
        asr=(lambda waveform, sampling_rate: "a c d") if asr else None,
    )
    assert model.seen.tolist() == [[1, 3]]
    assert result["reference"] == "a b" and result["text"] == "a"
    assert result["status"] == "generated" and Path(result["wav"]).is_file()
    assert result["samples"] == 240
    assert result["asr_status"] == ("measured" if asr else "not_measured")
    assert result["cer"] == (1 if asr else None)


def test_no_generated_frames_is_reported_as_failure(tmp_path: Path) -> None:
    result = generate_audio(
        GenerationFixture(False),
        TokenizerFixture(),
        generation_sample(),
        CodecFixture(),
        tmp_path / "example",
    )
    assert result["status"] == "failed" and result["failure"] == "no_audio_frames"
    assert result["cer"] is None and "wav" not in result


def test_local_asr_resamples_without_reference_text(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    class LocalModel:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def generate(self, **kwargs) -> list[dict]:
            calls["generate"] = kwargs
            return [{"text": "<|zh|><|NEUTRAL|>你好<|Speech|>"}]

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=LocalModel))
    asr = load_local_asr(tmp_path)
    assert asr(torch.zeros(24000), 24000) == "你好"
    assert calls["init"]["model"] == str(tmp_path)
    assert calls["init"]["disable_update"] and not calls["init"]["trust_remote_code"]
    assert len(calls["generate"]["input"]) == 16000
    assert set(calls["generate"]) == {"input", "language", "use_itn"}
