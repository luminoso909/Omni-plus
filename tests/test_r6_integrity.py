"""R6 入口继续拒绝 R5 已修复的媒体别名和跨 Checkout 身份漏洞。"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_r5_data_identity import audit_root as audit_root
from test_r5_data_identity import source_and_target

from omni.data.r3_prepare import file_hash
from omni.training import r3_local, r6_local

ROOT = Path(__file__).resolve().parents[1]


def sft_source(tmp_path: Path, source: dict) -> dict:
    """只构造来源审计所需状态；身份检查应先于大模型构造。"""
    settings = dict(model=dict(kind="qwen3", initialization={}))
    cache = tmp_path / "b1-cache.json"
    cache.write_text(json.dumps(dict(contract=source)))
    identity = dict(settings=settings, data_manifest_sha256=file_hash(cache))
    checkpoint = tmp_path / "b1.pt"
    torch.save(
        dict(progress=dict(identity=identity), model={"sentinel": torch.tensor(7)}), checkpoint
    )
    sha = file_hash(checkpoint)
    manifest = tmp_path / "b1-run.json"
    manifest.write_text(
        json.dumps(
            dict(
                config=settings,
                checkpoint_sha256=sha,
                data_manifest_sha256=identity["data_manifest_sha256"],
                data_contract=source,
                asset_identity=r3_local.adapter_identity(settings, source),
            )
        )
    )
    lineage = tmp_path / "b1-lineage.json"
    lineage.write_text(json.dumps({sha: dict(cache_manifest=cache.name)}))
    return settings | {
        "grpo": {
            "policy": dict(
                source="sft",
                checkpoint=str(checkpoint),
                manifest=str(manifest),
                lineage=str(lineage),
            )
        }
    }


@pytest.mark.parametrize("cross_task", [False, True])
def test_grpo_rejects_dropped_alias_and_changed_sample_id(
    audit_root, tmp_path, monkeypatch, cross_task
):
    source, target = source_and_target(audit_root, cross_task=cross_task)
    settings = sft_source(tmp_path, source)

    def no_model(*args, **kwargs):
        pytest.fail("媒体身份冲突应在模型构造前拒绝。")

    monkeypatch.setattr(r3_local, "build_model", no_model)
    with pytest.raises(ValueError, match="留出集冲突"):
        r6_local.setup_model(settings, ROOT, target)


@pytest.mark.parametrize("missing", ["source", "target"])
def test_grpo_rejects_incomplete_media_identity(audit_root, tmp_path, missing):
    source, target = source_and_target(audit_root, independent=True)
    (source if missing == "source" else target).pop("identity_groups")
    settings = sft_source(tmp_path, source)
    with pytest.raises(ValueError, match="完整媒体组身份"):
        r6_local.policy_source(settings, ROOT, target)


@pytest.mark.parametrize("same_split", [False, True])
def test_grpo_accepts_legal_sft_source(audit_root, tmp_path, same_split):
    source, target = source_and_target(audit_root, independent=True)
    if same_split:
        target = copy.deepcopy(source)
    settings = sft_source(tmp_path, source)
    payload = r6_local.policy_source(settings, ROOT, target)
    assert payload["model"]["sentinel"].item() == 7
    assert payload["progress"]["identity"]["settings"] == {"model": settings["model"]}


def test_grpo_cli_rejects_code_imported_from_other_checkout(audit_root, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_experiment.py"),
            "grpo-smoke",
            "--config",
            str(tmp_path / "not-read.yaml"),
            "--dry-run",
        ],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(audit_root), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "执行源码与记录根目录不一致" in result.stderr
    assert str(audit_root / "omni/__init__.py") in result.stderr
    assert "FileNotFoundError" not in result.stderr
