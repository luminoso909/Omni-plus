"""审计源以临时合成 WAV 验证契约；不冒充真实模型或音质验收。"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from test_qwen_omni_data import formatter as formatter

from omni.data.r3_prepare import file_hash
from omni.data.sft_sources import encode_sample, read_sources


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.wav"
    sf.write(path, np.linspace(-0.2, 0.2, 800, dtype=np.float32), 8000)
    return path


def record(path: Path, **values) -> dict:
    return (
        dict(
            sample_id="LJSpeech:fixture",
            source="LJSpeech",
            revision="fixture-v1",
            license="fixture-only",
            task="t2a",
            split="train",
            prompt="Read aloud: fixture",
            answer="fixture",
            target_transcript="fixture",
            target_provenance="original_recording",
            target_audio=dict(path=path.name, sha256=file_hash(path), media_id="fixture/audio"),
        )
        | values
    )


def manifest(tmp_path: Path, rows: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "audited.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path, file_hash(path)


def test_audited_sources_preserve_paths_and_reject_unknown_target(audio_file, tmp_path) -> None:
    for name in ("LJSpeech", "AISHELL-3", "MiniMind-O"):
        path, digest = manifest(tmp_path, [record(audio_file, source=name)])
        rows = read_sources(path, digest)
        assert rows[0]["target_audio"]["path"] == str(audio_file)
        assert rows[0]["source"] == name
    with pytest.raises(ValueError, match="SHA-256"):
        read_sources(path, "0" * 64)
    for change in ({"target_provenance": "decoded_unknown_codes"}, {"target_transcript": "wrong"}):
        path, digest = manifest(tmp_path, [record(audio_file, **change)])
        with pytest.raises(ValueError, match="目标语音"):
            read_sources(path, digest)


def test_qa_requires_distinct_trusted_raw_target(audio_file, tmp_path) -> None:
    example = record(audio_file, task="a2a", a2a_kind="qa")
    example["input_audio"] = example["target_audio"].copy()
    path, digest = manifest(tmp_path, [example])
    with pytest.raises(ValueError, match="同一输入"):
        read_sources(path, digest)
    example["a2a_kind"] = "echo"
    path, digest = manifest(tmp_path, [example])
    assert read_sources(path, digest)[0]["a2a_kind"] == "echo"
    del example["target_audio"]
    path, digest = manifest(tmp_path, [example])
    with pytest.raises(ValueError, match="不完整"):
        read_sources(path, digest)

    target = tmp_path / "answer.wav"
    sf.write(target, np.zeros(400, dtype=np.float32), 8000)
    example["a2a_kind"] = "qa"
    example["target_audio"] = dict(
        path=target.name, sha256=file_hash(target), media_id="fixture/answer"
    )
    path, digest = manifest(tmp_path, [example])
    assert read_sources(path, digest)[0]["a2a_kind"] == "qa"


def test_cross_role_media_split_leakage_is_rejected(audio_file, tmp_path) -> None:
    first = record(audio_file)
    second = record(audio_file, sample_id="asr", task="a2t", split="test")
    second["input_audio"] = second.pop("target_audio")
    path, digest = manifest(tmp_path, [first, second])
    with pytest.raises(ValueError, match="不同历史 split"):
        read_sources(path, digest)
    first["target_audio"]["sha256"] = "f" * 64
    path, digest = manifest(tmp_path, [first])
    with pytest.raises(ValueError, match="媒体校验"):
        read_sources(path, digest)


def test_waveform_branches_and_raw_target_reencoding(audio_file, tmp_path) -> None:
    from types import SimpleNamespace

    calls = []

    class Formatter:
        config = SimpleNamespace(num_audio_codebooks=8, audio_codebook_size=2048)

        def sample(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return {"input_ids": torch.ones(8, dtype=torch.long)}

    class Codec:
        def encode(self, waveform):
            assert waveform.shape == (2400,)
            return torch.arange(24).reshape(1, 8, 3)

    def speech(waveform, sampling_rate):
        assert sampling_rate == 16000 and waveform.shape == (1600,)
        return torch.zeros(5, 512)

    example = record(audio_file, task="a2a", a2a_kind="echo")
    example["input_audio"] = example["target_audio"].copy()
    path, digest = manifest(tmp_path, [example])
    row = read_sources(path, digest)[0]
    sample = encode_sample(row, Formatter(), speech, None, Codec())
    assert sample["task"] == "a2a"
    assert calls[0][1]["audio_features"].shape == (5, 512)
    assert calls[0][1]["answer_codes"].shape == (8, 3)


def test_text_has_no_media_and_duplicate_identity_rejected(tmp_path) -> None:
    row = dict(
        sample_id="text:1",
        source="fixture",
        revision="1",
        license="fixture-only",
        task="text",
        split="train",
        prompt="q",
        answer="a",
    )
    path, digest = manifest(tmp_path, [row])
    assert read_sources(path, digest)[0]["media_ids"]
    path, digest = manifest(tmp_path, [row, row])
    with pytest.raises(ValueError, match="重复"):
        read_sources(path, digest)


def test_image_and_text_use_native_formatter_masks(formatter) -> None:
    image = dict(task="i2t", prompt="q", answer="a", image={"path": "fixture-image"})
    sample = encode_sample(image, formatter, None, lambda path: torch.zeros(64, 4), None)
    assert sample["input_ids"].eq(formatter.config.image_ids[0]).sum() == 64
    assert sample["audio_labels"].eq(-100).all()
    assert sample["labels"].ne(-100).sum() == 2  # 仅答案和 EOS。
    with pytest.raises(ValueError, match="64"):
        encode_sample(image, formatter, None, lambda path: torch.zeros(63, 4), None)

    text = encode_sample(dict(task="text", prompt="q", answer="a"), formatter, None, None, None)
    assert text["audio_features"] is text["image_features"] is None
    assert text["labels"].ne(-100).sum() == 2
