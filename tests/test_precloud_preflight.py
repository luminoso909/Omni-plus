"""SFT 云前预检使用实际 split、Checkpoint 与初始化门槛，且不运行模型。"""

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_r4_training import samples
from test_r5_training import CONTRACT, ROOT, fixed_source
from test_r5_training import fixture_settings as fixture_settings

from omni.data.prepare import content_hash
from omni.data.r3_prepare import file_hash
from omni.training import r3_local
from omni.training.sft import save_checkpoint
from scripts import run_experiment


@pytest.fixture
def prepared(fixture_settings, tmp_path):
    config = fixture_settings
    config["evaluation"]["benchmark"] = {"split": "test", "t2a_samples": 1}
    rows = samples()[:4]
    files = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.pt"
        torch.save({"samples": rows, "contract_id": content_hash(CONTRACT)}, path)
        files[path.name] = file_hash(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"contract": CONTRACT, "files": files}))
    model = r3_local.build_model(config, ROOT)
    optimizer, scheduler, scaler, _ = r3_local.optimizer_state(model, config)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scaler,
        {
            "identity": r3_local.run_identity(config, ROOT),
            "step": 1,
            "scheduler": scheduler.state_dict(),
        },
    )
    return config, checkpoint


def inspect(config: dict, checkpoint: Path | None, command: str) -> dict:
    model = r3_local.build_model(config, ROOT, meta=True)

    def no_forward(*args, **kwargs):
        pytest.fail("Dry run 不得执行 forward。")

    model.forward = no_forward
    result = r3_local.inspect_sft_run(
        model, config, ROOT, CONTRACT, command=command, resume=checkpoint, target=4
    )
    assert all(parameter.is_meta for parameter in model.parameters())
    return result


@pytest.mark.parametrize("command", ["smoke", "resume", "evaluate", "demo"])
def test_preflight_uses_actual_split_without_model_execution(prepared, command):
    config, checkpoint = prepared
    result = inspect(config, None if command == "smoke" else checkpoint, command)
    assert result == {
        "split": "test" if command in {"evaluate", "demo"} else "train",
        "samples": 4,
        "checkpoint_step": None if command == "smoke" else 1,
    }


@pytest.mark.parametrize("command", ["evaluate", "demo"])
def test_missing_actual_test_split_is_rejected(prepared, command):
    config, checkpoint = prepared
    (Path(config["data"]["cache_dir"]) / "test.pt").unlink()
    with pytest.raises(FileNotFoundError, match="test.pt"):
        inspect(config, checkpoint, command)


def test_evaluation_does_not_require_unselected_splits(prepared):
    config, checkpoint = prepared
    for split in ("train", "validation"):
        (Path(config["data"]["cache_dir"]) / f"{split}.pt").unlink()
    assert inspect(config, checkpoint, "evaluate")["split"] == "test"


@pytest.mark.parametrize(
    "command,split", [("smoke", "validation"), ("evaluate", "test"), ("demo", "test")]
)
def test_preflight_rejects_missing_required_split_task(prepared, command, split):
    config, checkpoint = prepared
    directory = Path(config["data"]["cache_dir"])
    path = directory / f"{split}.pt"
    payload = torch.load(path, weights_only=True)
    payload["samples"] = [row for row in payload["samples"] if row["task"] != "i2t"]
    torch.save(payload, path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"][path.name] = file_hash(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"{split} 缓存缺少"):
        inspect(config, None if command == "smoke" else checkpoint, command)


@pytest.mark.parametrize(
    "defect, message",
    [
        ("identity", "配置、代码或数据"),
        ("config", "模型配置"),
        ("state", "参数键"),
        ("dtype", "dtype"),
        ("nan", "数值"),
        ("step", "恢复目标"),
        ("freeze", "冻结策略"),
        ("scheduler", "Scheduler"),
        ("optimizer", "Optimizer"),
    ],
)
def test_resume_preflight_rejects_invalid_state(prepared, defect, message):
    config, checkpoint = prepared
    payload = torch.load(checkpoint, weights_only=True)
    if defect == "identity":
        payload["progress"]["identity"]["source_schema_version"] = 0
    elif defect == "config":
        payload["config"]["hidden_size"] += 1
    elif defect == "state":
        payload["model"].pop(next(iter(payload["model"])))
    elif defect in {"dtype", "nan"}:
        name = next(iter(payload["model"]))
        payload["model"][name] = (
            payload["model"][name].double()
            if defect == "dtype"
            else torch.full_like(payload["model"][name], float("nan"))
        )
    elif defect == "step":
        payload["progress"]["step"] = 4
    elif defect == "freeze":
        payload["trainable"] = []
    elif defect == "scheduler":
        payload["progress"].pop("scheduler")
    else:
        payload.pop("optimizer")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        inspect(config, checkpoint, "resume")


@pytest.mark.parametrize(
    "defect,message",
    [
        ("empty_optimizer", "Optimizer"),
        ("state", "Optimizer"),
        ("groups", "Optimizer"),
        ("parameters", "Optimizer"),
        ("scheduler", "Scheduler"),
        ("scheduler_groups", "Scheduler"),
        ("rng", "RNG"),
        ("torch_rng", "RNG"),
    ],
)
def test_preflight_rejects_malformed_optimizer_scheduler_rng(prepared, defect, message):
    config, checkpoint = prepared
    payload = torch.load(checkpoint, weights_only=True)
    if defect == "empty_optimizer":
        payload["optimizer"] = {}
    elif defect == "state":
        payload["optimizer"]["state"] = []
    elif defect == "groups":
        payload["optimizer"]["param_groups"].pop()
    elif defect == "parameters":
        payload["optimizer"]["param_groups"][0]["params"].pop()
    elif defect == "scheduler":
        payload["progress"]["scheduler"] = {}
    elif defect == "scheduler_groups":
        payload["progress"]["scheduler"]["base_lrs"] = []
    elif defect == "rng":
        payload["rng"] = {}
    else:
        payload["rng"]["torch"] = torch.tensor([1], dtype=torch.uint8)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        inspect(config, checkpoint, "resume")


def test_resume_state_inspection_preserves_global_rng(prepared):
    config, checkpoint = prepared
    payload = torch.load(checkpoint, weights_only=True)
    model = r3_local.build_model(config, ROOT, meta=True)
    python, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state()
    r3_local.check_resume_state(payload, model, config)
    assert random.getstate() == python
    current = np.random.get_state()
    assert numpy[0] == current[0] and numpy[2:] == current[2:]
    np.testing.assert_array_equal(numpy[1], current[1])
    assert torch.equal(torch.get_rng_state(), tensor)


@pytest.mark.parametrize("command", ["smoke", "resume"])
def test_small_legacy_torch_save_remains_compatible(prepared, tmp_path, command):
    config, checkpoint = prepared
    payload = torch.load(checkpoint, weights_only=True)
    legacy = tmp_path / "legacy.pt"
    if command == "smoke":
        torch.save({"model": payload["model"]}, legacy, _use_new_zipfile_serialization=False)
        config["model"]["initialization"]["vision_checkpoint"] = str(legacy)
    else:
        torch.save(payload, legacy, _use_new_zipfile_serialization=False)
    assert inspect(config, legacy if command == "resume" else None, command)["split"] == "train"


def test_large_legacy_checkpoint_fails_before_loading(tmp_path, monkeypatch):
    legacy = tmp_path / "large-legacy.pt"
    with legacy.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    def no_load(*args, **kwargs):
        pytest.fail("过大的旧格式预检不能整体读入内存。")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match="64 MiB"):
        r3_local.inspection_checkpoint(legacy)


@pytest.mark.parametrize("invalid", [False, True])
def test_initialization_checks_shared_contract_without_copy(prepared, tmp_path, invalid):
    config, checkpoint = prepared
    state = torch.load(checkpoint, weights_only=True)["model"]
    if invalid:
        name = next(name for name in state if name.startswith("vision_proj."))
        state[name] = torch.zeros(1)
    initial = tmp_path / "vision.pt"
    torch.save({"model": state}, initial)
    config["model"]["initialization"]["vision_checkpoint"] = str(initial)
    if invalid:
        with pytest.raises(ValueError, match="初始化 Checkpoint"):
            inspect(config, None, "smoke")
    else:
        assert inspect(config, None, "smoke")["checkpoint_step"] is None


@pytest.mark.parametrize("invalid", [False, True])
def test_bridge_initialization_reuses_lineage_and_weight_checks(prepared, tmp_path, invalid):
    config, _ = prepared
    checkpoint, manifest, _ = fixed_source(config, tmp_path)
    config["model"].update(
        bridge={"kind": "weighted", "layers": [0, 1]},
        initialization={
            "bridge_base_checkpoint": str(checkpoint),
            "bridge_base_manifest": str(manifest),
        },
    )
    if invalid:
        metadata = json.loads(manifest.read_text())
        metadata["checkpoint_sha256"] = "not-the-source"
        manifest.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="SHA-256"):
            inspect(config, None, "smoke")
    else:
        assert inspect(config, None, "smoke")["checkpoint_step"] is None


@pytest.mark.parametrize("command", ["smoke", "resume", "evaluate", "demo"])
def test_cli_preflight_accepts_matching_checkpoint(
    prepared, tmp_path, monkeypatch, capsys, command
):
    config, checkpoint = prepared
    output = tmp_path / "never-created"
    monkeypatch.setattr(run_experiment, "read_settings", lambda *args: config)
    monkeypatch.setattr(run_experiment, "data_contract", lambda *args: CONTRACT)
    args = [
        "run_experiment.py",
        command,
        "--config",
        "fixture.yaml",
        "--dry-run",
        "--run-dir",
        str(output),
    ]
    if command != "smoke":
        args.extend(["--resume", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", args)
    run_experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result["execution"] == "preflight_no_forward"
    assert result["preflight"]["split"] == ("test" if command in {"evaluate", "demo"} else "train")
    assert not output.exists()


def test_cli_dry_run_rejects_checkpoint_before_creating_output(prepared, tmp_path, monkeypatch):
    config, checkpoint = prepared
    bad = tmp_path / "invalid.pt"
    torch.save({"unrelated": True}, bad)
    output = tmp_path / "never-created"
    monkeypatch.setattr(run_experiment, "read_settings", lambda *args: config)
    monkeypatch.setattr(run_experiment, "data_contract", lambda *args: CONTRACT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "resume",
            "--config",
            "fixture.yaml",
            "--dry-run",
            "--resume",
            str(bad),
            "--run-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="完整 Checkpoint"):
        run_experiment.main()
    assert not output.exists()


@pytest.mark.parametrize("command", ["inspect", "prepare"])
def test_metadata_commands_do_not_require_prepared_cache(
    prepared, tmp_path, monkeypatch, capsys, command
):
    config, _ = prepared
    config = copy.deepcopy(config)
    config["data"]["cache_dir"] = str(tmp_path / "not-prepared")
    monkeypatch.setattr(run_experiment, "read_settings", lambda *args: config)
    monkeypatch.setattr(run_experiment, "data_contract", lambda *args: CONTRACT)
    monkeypatch.setattr(
        sys, "argv", ["run_experiment.py", command, "--config", "fixture.yaml", "--dry-run"]
    )
    run_experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result["preflight"] is None
    assert result["execution"] == "metadata_only_no_forward"
    assert not Path(config["data"]["cache_dir"]).exists()
