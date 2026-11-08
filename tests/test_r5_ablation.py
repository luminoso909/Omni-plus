"""留出评估的实际生成路径，以及矩阵证据和公平性校验。"""

import csv
import json
from pathlib import Path

import pytest
import yaml
from test_r4_evaluation import CodecFixture, GenerationFixture, TokenizerFixture, generation_sample

from omni.evaluation import bridge_benchmark
from scripts import bridge_ablation


def test_benchmark_saves_ordered_samples_and_failures(tmp_path: Path, monkeypatch) -> None:
    sample = generation_sample() | {"task": "t2a"}
    samples = [sample | {"task": "i2t"}, sample, sample]
    records = [
        {"task": row["task"], "record_id": name}
        for row, name in zip(samples, ("image", "left", "right"), strict=True)
    ]
    original = bridge_benchmark.generate_audio

    def generate(*args, **kwargs):
        if args[4].name == "sample-0001":
            raise ValueError("fixture decoding error")
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge_benchmark, "generate_audio", generate)
    output = tmp_path / "new"
    result = bridge_benchmark.benchmark_audio(
        GenerationFixture(),
        TokenizerFixture(),
        samples,
        records,
        CodecFixture(),
        output,
        count=2,
        max_new_tokens=4,
        asr=lambda wave, rate: "a c d",
    )
    assert result["sample_ids"] == ["left", "right"]
    assert result["failed_samples"] == 1 and result["timed_samples"] == 1
    assert result["asr_measured_samples"] == 1 and result["wer"] == 1
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["median_stream_seconds"] >= 0
    assert (output / "sample-0000/prediction.wav").is_file()
    with (output / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["record_id"] == "right" and rows[1]["error"] == "fixture decoding error"
    with pytest.raises(FileExistsError):
        bridge_benchmark.benchmark_audio(
            GenerationFixture(),
            TokenizerFixture(),
            samples,
            records,
            CodecFixture(),
            output,
            count=2,
            max_new_tokens=4,
        )


def test_benchmark_rejects_training_split_and_insufficient_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation"):
        bridge_benchmark.benchmark_options(dict(benchmark=dict(split="train", t2a_samples=1)))
    sample = generation_sample() | {"task": "t2a"}
    with pytest.raises(ValueError, match="独立"):
        bridge_benchmark.benchmark_audio(
            GenerationFixture(),
            TokenizerFixture(),
            [sample],
            [dict(task="t2a", record_id="one")],
            CodecFixture(),
            tmp_path / "too-few",
            count=2,
            max_new_tokens=4,
        )


def matrix_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    root = tmp_path
    base = dict(
        schema_version=2,
        model=dict(kind="tiny"),
        training=dict(steps=2, seed=42),
        data=dict(cache_dir="cache"),
        evaluation=dict(max_new_tokens=8, benchmark=dict(split="test", t2a_samples=1)),
    )
    identities = dict(
        initialization_sha256={},
        model_config_sha256="config hash",
        source_sha256={"bridge.py": "source hash"},
        asset_identity={"qwen_revision": "fixture"},
    )
    contract = {"kind": "fixture"}
    cache = root / "cache"
    cache.mkdir()
    (cache / "test.pt").write_bytes(b"fixture split")
    bridge_ablation.write_json(
        cache / "manifest.json",
        dict(
            contract=contract,
            files={"test.pt": bridge_ablation.digest(cache / "test.pt")},
            selections={"test": [dict(task="t2a", record_id="heldout-1")]},
        ),
    )
    identities["data_manifest_sha256"] = bridge_ablation.digest(cache / "manifest.json")
    monkeypatch.setattr(bridge_ablation, "read_settings", lambda *_: base)
    monkeypatch.setattr(bridge_ablation, "build_model", lambda *_, **kwargs: None)
    monkeypatch.setattr(bridge_ablation, "data_contract", lambda *_: contract)
    monkeypatch.setattr(bridge_ablation, "run_identity", lambda *_: identities)
    monkeypatch.setattr(
        bridge_ablation, "adapter_identity", lambda *_: identities["asset_identity"]
    )
    candidate_file = root / "candidates.yaml"
    candidate_file.write_text(
        "fixed: {kind: fixed, layers: [0]}\nweighted: {kind: weighted, layers: [0, 1]}\n"
    )
    output = root / "matrix"
    matrix = bridge_ablation.plan(root / "base.yaml", candidate_file, output, root)
    for run_id, spec in matrix["candidates"].items():
        assert yaml.safe_load(Path(spec["config_path"]).read_text()) == spec["config"]
        assert spec["config"]["training"] == base["training"]
        assert spec["config"]["model"]["bridge"]["kind"] == run_id
    with pytest.raises(FileExistsError):
        bridge_ablation.plan(root / "base.yaml", candidate_file, output, root)
    return output / "matrix.json", identities


def write_completed(matrix: dict, identities: dict, run_id: str, mismatch: str | None) -> None:
    entry = matrix["candidates"][run_id]
    train_dir, benchmark_dir = Path(entry["train"]), Path(entry["benchmark"])
    train_dir.mkdir(parents=True)
    benchmark_dir.mkdir(parents=True)
    common = identities | dict(
        config=entry["config"], checkpoint_sha256=f"weights-{run_id}", hardware={"device": "cpu"}
    )
    train = common | dict(
        status="passed",
        step=2,
        parameter_groups={"bridge": {"total": 5, "trainable": 5}},
        validation={"total": 1.0},
        gradient_groups={"bridge": 0.5},
        evaluations=[],
    )
    metrics = dict(
        protocol={"max_new_tokens": 8},
        sample_ids=["heldout-1"],
        samples=[{"record_id": "heldout-1", "status": "generated"}],
        failed_samples=0,
        truncated_samples=1,
        median_stream_seconds=0.1,
        cuda_peak_allocated_bytes=None,
        asr_measured_samples=0,
        cer=None,
        wer=None,
    )
    evaluation = common | dict(
        checkpoint_step=2, asr_assets=None, split="test", metrics={"total": 1.0}, benchmark=metrics
    )
    if mismatch == "checkpoint":
        evaluation["checkpoint_sha256"] = "another checkpoint"
    elif mismatch == "hardware":
        train["hardware"] = evaluation["hardware"] = {"device": "different cpu"}
    elif mismatch == "sample":
        metrics["samples"][0]["record_id"] = "another sample"
    elif mismatch == "generated":
        metrics["samples"][0].update(status="failed", failure="no_audio_frames")
        metrics["failed_samples"] = 1
    bridge_ablation.write_json(train_dir / "manifest.json", train)
    bridge_ablation.write_json(benchmark_dir / "evaluation.json", evaluation)


@pytest.mark.parametrize(
    "mismatch,status",
    [
        (None, "passed"),
        ("checkpoint", "invalid"),
        ("hardware", "passed"),
        ("sample", "invalid"),
        ("generated", "failed"),
        ("missing", "missing"),
    ],
)
def test_collect_checks_fairness_and_preserves_results(
    tmp_path: Path, monkeypatch, mismatch: str | None, status: str
) -> None:
    path, identities = matrix_fixture(tmp_path, monkeypatch)
    matrix = json.loads(path.read_text())
    write_completed(matrix, identities, "fixed", None)
    if mismatch != "missing":
        write_completed(matrix, identities, "weighted", mismatch)
    output = tmp_path / "summary"
    result = bridge_ablation.collect(path, output)
    assert result["comparable"] is (mismatch is None)
    assert result["runs"][1]["status"] == status
    assert (output / "summary.csv").is_file()
    assert json.loads((output / "summary.json").read_text()) == result
    with pytest.raises(FileExistsError):
        bridge_ablation.collect(path, output)
