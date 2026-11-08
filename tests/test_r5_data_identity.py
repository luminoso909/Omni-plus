"""真实 JSONL 选样与谱系检查，媒体和模型均为离线合成 fixture。"""

import copy
import json
import shutil
import wave
from pathlib import Path

import pytest
import torch
import yaml
from test_r3_training import assert_nested_equal
from test_r5_training import fixture_settings as fixture_settings

from omni.data.bridge_splits import check_bridge_splits
from omni.data.r3_prepare import file_hash
from omni.data.sft_sources import inspect_sources
from omni.training import r3_local
from omni.training.sft import save_checkpoint
from scripts import bridge_ablation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "audited-assets"
    shutil.copytree(ROOT / "omni", root / "omni", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "configs").mkdir()
    qwen = root / "qwen"
    qwen.mkdir()
    (qwen / "tokenizer.json").write_text("fixture tokenizer")
    (root / "configs/assets_v2.yaml").write_text(
        yaml.safe_dump(
            dict(
                assets=dict(
                    qwen3_1_7b=dict(
                        revision="fixture",
                        files={"tokenizer.json": {"sha256": file_hash(qwen / "tokenizer.json")}},
                    )
                )
            )
        )
    )
    reference = {}
    for name, folder, weight in (
        ("sensevoice", "SenseVoiceSmall", "model.pt"),
        ("siglip", "siglip2-base-p32-256-ve", "model.safetensors"),
        ("mimi", "mimi", "model.safetensors"),
    ):
        path = root / "reference/model" / folder / weight
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        reference[name] = {"sha256": file_hash(path)}
    manifest = root / "experiments/r2/20261108-cpu-fp32/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(reference_assets=reference)))
    (root / "model.yaml").write_text("model: {}\n")
    return root


def audio_row(root: Path, name: str, split: str, file: str, media_id: str, rate: int) -> dict:
    path = root / file
    if not path.exists():
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(bytes(rate // 2))
    return dict(
        sample_id=name,
        source="synthetic-regression",
        revision="1",
        license="fixture-only",
        task="t2a",
        split=split,
        prompt="Read fixture",
        answer=name,
        target_transcript=name,
        target_provenance="original_recording",
        target_audio=dict(path=file, sha256=file_hash(path), media_id=media_id),
    )


def contract(root: Path, name: str, rows: list[dict]) -> dict:
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    settings = dict(
        data=dict(
            tasks={"t2a": 3},
            seed=0,
            max_length=128,
            audited_manifest=path.name,
            audited_manifest_sha256=file_hash(path),
        ),
        assets=dict(qwen_dir="qwen", reference_root="reference"),
        model=dict(config="model.yaml"),
    )
    return inspect_sources(settings, root)


def source_and_target(
    root: Path, *, reverse: bool = False, independent: bool = False, cross_task: bool = False
) -> tuple[dict, dict]:
    split = "test" if reverse else "train"
    a = audio_row(root, "a-source", split, "a.wav", "source-id", 8000)
    b = audio_row(root, "b-alias", split, "b.wav", "source-id", 16000)
    alias_sha = b["target_audio"]["sha256"]
    if cross_task:
        # 未纳入 t2a 预算的 a2t 行仍是已审计的媒体别名。
        b["task"] = "a2t"
        b["input_audio"] = b.pop("target_audio")
        del b["target_provenance"], b["target_transcript"]
    grouped = contract(
        root,
        "grouped",
        [
            a,
            b,
            audio_row(root, "c-val", "validation", "val.wav", "val-id", 24000),
            audio_row(root, "d-other", "train" if reverse else "test", "other.wav", "other", 32000),
        ],
    )
    renamed = audio_row(
        root,
        "g-renamed",
        "train" if reverse else "test",
        "independent.wav" if independent else "b.wav",
        "new-id",
        22050,
    )
    single = contract(
        root,
        "single",
        [
            audio_row(root, "e-other", split, "new-other.wav", "new-other", 44100),
            audio_row(root, "f-val", "validation", "new-val.wav", "new-val", 48000),
            renamed,
        ],
    )
    selected = next(row for row in grouped["records"] if row["record_id"] == "a-source")
    assert all(row["record_id"] != "b-alias" for row in grouped["records"])
    assert "sha256:" + alias_sha in grouped["identity_groups"][selected["group_id"]]
    selected_alias = next(row for row in single["records"] if row["record_id"] == "g-renamed")
    assert not set(selected["media_ids"]) & set(selected_alias["media_ids"])
    assert selected["group_id"] != selected_alias["group_id"]
    return (single, grouped) if reverse else (grouped, single)


def lineage(tmp_path: Path, source: dict) -> tuple[dict, Path]:
    cache = tmp_path / "source-cache.json"
    cache.write_text(json.dumps(dict(contract=source)))
    identity = dict(settings=dict(model={}), data_manifest_sha256=file_hash(cache))
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps({"b1": dict(cache_manifest=cache.name)}))
    return identity, path


@pytest.mark.parametrize(("reverse", "cross_task"), [(False, False), (True, False), (False, True)])
def test_unselected_aliases_cannot_cross_stage_splits(audit_root, tmp_path, reverse, cross_task):
    source, target = source_and_target(audit_root, reverse=reverse, cross_task=cross_task)
    assert source["schema_version"] == target["schema_version"] == 3
    identity, path = lineage(tmp_path, source)
    with pytest.raises(ValueError, match="留出集冲突"):
        check_bridge_splits(target, identity, "b1", path, formal=True)


@pytest.mark.parametrize("case", ["alias", "same_split", "independent"])
def test_formal_plan_and_warm_start_share_alias_gate(
    audit_root, fixture_settings, tmp_path, monkeypatch, case
):
    source, target = source_and_target(audit_root, independent=case == "independent")
    if case == "same_split":
        target = copy.deepcopy(source)
    config = copy.deepcopy(fixture_settings)
    model = r3_local.build_model(config, ROOT)
    optimizer, _, scaler, _ = r3_local.optimizer_state(model, config)
    # 小型真实 Qwen/Talker 权重 fixture；formal 控制数据门槛，不加载 1.7B。
    config["model"]["kind"] = "qwen3"
    source_cache = tmp_path / "b1-cache.json"
    source_cache.write_text(json.dumps(dict(contract=source)))
    identity = dict(settings=copy.deepcopy(config), data_manifest_sha256=file_hash(source_cache))
    checkpoint = tmp_path / "b1.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, {"identity": identity, "step": 0})
    source_sha = file_hash(checkpoint)
    run = tmp_path / "b1-run.json"
    run.write_text(
        json.dumps(
            dict(
                config=identity["settings"],
                asset_identity=r3_local.adapter_identity(config, source),
                checkpoint_sha256=source_sha,
                data_manifest_sha256=identity["data_manifest_sha256"],
                data_contract=source,
            )
        )
    )
    provenance = tmp_path / "b1-lineage.json"
    provenance.write_text(json.dumps({source_sha: dict(cache_manifest=source_cache.name)}))
    config["model"]["initialization"] = dict(
        bridge_base_checkpoint=str(checkpoint),
        bridge_base_manifest=str(run),
        bridge_base_lineage=str(provenance),
    )
    config["evaluation"]["benchmark"] = {"split": "test", "t2a_samples": 1}
    split = tmp_path / "test.pt"
    split.write_bytes(b"encoded sample fixture")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                contract=target,
                files={split.name: file_hash(split)},
                selections={"test": [row for row in target["records"] if row["split"] == "test"]},
            )
        )
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("fixed: {kind: fixed, layers: [0]}\nlast: {kind: last, layers: [1]}\n")
    output = tmp_path / "matrix"
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(bridge_ablation, "read_settings", lambda *_: config)
    monkeypatch.setattr(bridge_ablation, "build_model", lambda *_, **__: None)
    monkeypatch.setattr(bridge_ablation, "data_contract", lambda *_: target)
    if case == "alias":
        with pytest.raises(ValueError, match="留出集冲突"):
            r3_local.initialize_bridge_base(
                model, config, ROOT, target, {"bridge_base_checkpoint": source_sha}
            )
        assert_nested_equal(before, model.state_dict())
        with pytest.raises(ValueError, match="留出集冲突"):
            bridge_ablation.plan(tmp_path / "base.yaml", candidates, output, ROOT)
        assert not output.exists()
    else:
        with torch.no_grad():
            model.bridge.proj.weight.zero_()
        r3_local.initialize_bridge_base(
            model, config, ROOT, target, {"bridge_base_checkpoint": source_sha}
        )
        assert_nested_equal(before, model.state_dict())
        matrix = bridge_ablation.plan(tmp_path / "base.yaml", candidates, output, ROOT)
        assert matrix["scope"] == "planned_ablation"
        assert (output / "matrix.json").is_file()
