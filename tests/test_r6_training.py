"""真实 Tiny Qwen/Talker 的 GRPO 参数更新、严格恢复及 CLI 路由。"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from test_r3_training import assert_nested_equal
from test_r4_training import samples
from test_r5_training import fixture_settings as fixture_settings

from omni.data.prepare import content_hash
from omni.data.r3_prepare import file_hash
from omni.training import r3_local, r6_local
from omni.training.adapters import load_adapter
from omni.training.sft import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {"qwen_revision": "fixture", "assets": {}}


def prepare_fixture(config: dict, cache: Path) -> dict:
    rows = samples()
    files, selections = {}, {}
    for split in ("train", "validation", "test"):
        selected = [
            dict(record_id=f"{split}-{i}", task=row["task"], a2a_kind="echo")
            for i, row in enumerate(rows)
        ]
        path = cache / f"{split}.pt"
        torch.save(dict(samples=rows, contract_id=content_hash(CONTRACT)), path)
        files[path.name] = file_hash(path)
        selections[split] = selected
    (cache / "manifest.json").write_text(
        json.dumps(dict(contract=CONTRACT, files=files, selections=selections))
    )
    values = r3_local.merge_settings(ROOT / "configs/v2/r6_joint_local.yaml")
    config["grpo"] = values["grpo"]
    config["training"].update(
        steps=2,
        batch_size=1,
        accumulation_steps=1,
        threads=1,
        warmup_steps=0,
        learning_rate=0.003,
        group_lrs={},
        tasks=["t2a"],
        task_weights={},
        sample_weights={},
        allow_diagnostic_a2a=False,
        max_seconds=0.0,
    )
    model_file = Path(config["model"]["config"])
    model = yaml.safe_load(model_file.read_text())
    model["model"]["num_talker_hidden_layers"] = 4
    model["model"]["eos_token_id"] = 2
    model_file.write_text(yaml.safe_dump(model))
    config["grpo"].update(max_new_tokens=11, group_size=3)
    return config


@pytest.fixture
def config(fixture_settings, tmp_path, monkeypatch):
    settings = prepare_fixture(fixture_settings, tmp_path)
    monkeypatch.setattr(r3_local, "data_contract", lambda *args: CONTRACT)
    return settings


@pytest.mark.parametrize("mode", ["text", "audio", "joint"])
def test_real_policy_update_freeze_and_evaluate(config, tmp_path, mode):
    config["grpo"]["mode"] = mode
    config["grpo"]["trainable"] = (
        ["lora"] if mode == "text" else ["bridge", "talker", "audio_embedding", "audio_heads"]
    )
    if mode == "joint":
        config["grpo"]["trainable"].append("lora")
    report = r6_local.run_grpo(config, ROOT, tmp_path / "run")
    assert report["status"] == "passed" and report["step"] == 2
    assert report["frozen_before"] == report["frozen_after"]
    assert report["trainable_before"] != report["trainable_after"]
    assert all(report["gradient_groups"][name] > 0 for name in config["grpo"]["trainable"])
    assert report["metrics"][0]["ratio_mean"] == pytest.approx(1, abs=2e-5)
    assert report["metrics"][0]["kl"] == pytest.approx(0, abs=2e-5)
    trajectory_rows = [
        json.loads(line) for line in (tmp_path / "run/rollouts.jsonl").read_text().splitlines()
    ]
    assert {row["policy_version"] for row in trajectory_rows} == {0, 1}
    assert all(row["record_id"].startswith("train-") for row in trajectory_rows)
    assert all(row["prompt_ids"] and "reward_components" in row for row in trajectory_rows)
    evaluation = r6_local.run_grpo(
        config, ROOT, tmp_path / "eval", resume=Path(report["checkpoint"]), command="grpo-evaluate"
    )
    assert evaluation["status"] == "evaluated" and evaluation["split"] == "validation"
    assert evaluation["frozen_before"] == evaluation["frozen_after"]
    assert evaluation["trainable_before"] == evaluation["trainable_after"]
    assert not (tmp_path / "eval/step-000002.pt").exists()
    baseline = r6_local.run_grpo(
        config, ROOT, tmp_path / "baseline", command="grpo-evaluate", baseline=True
    )
    assert baseline["status"] == "evaluated" and baseline["step"] == 0
    assert baseline["checkpoint"] is None
    assert baseline["evaluation_seed"] == evaluation["evaluation_seed"]
    assert baseline["evaluation_record_ids"] == evaluation["evaluation_record_ids"]
    assert baseline["trainable_before"] == baseline["trainable_after"]
    assert not list((tmp_path / "baseline").glob("*.pt"))


def test_reference_restores_after_exception_and_shared_base(config):
    model, _ = r6_local.setup_model(config, ROOT, CONTRACT)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    reference = {
        name: value.detach().clone().zero_()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    with pytest.raises(RuntimeError, match="reference error"):
        with r6_local.reference_policy(model, reference):
            assert all(
                torch.count_nonzero(dict(model.named_parameters())[name]) == 0 for name in reference
            )
            raise RuntimeError("reference error")
    assert_nested_equal(original, dict(model.named_parameters()))


def test_multiple_epochs_keep_behavior_version(config, tmp_path):
    config["grpo"]["update_epochs"] = 2
    report = r6_local.run_grpo(config, ROOT, tmp_path / "epochs")
    assert [row["behavior_version"] for row in report["metrics"]] == [0, 0]
    assert [row["rollout_epoch"] for row in report["metrics"]] == [0, 1]
    assert not (tmp_path / "epochs/step-000001.pt").exists()
    assert (tmp_path / "epochs/step-000002.pt").exists()
    with pytest.raises(ValueError, match="完整 rollout"):
        r6_local.run_grpo(config, ROOT, tmp_path / "partial", steps=1)
    assert not (tmp_path / "partial").exists()


@pytest.mark.parametrize("epochs,expected", [(1, [3, 4]), (2, [4])])
def test_save_interval_and_final_rollout_boundary(config, tmp_path, epochs, expected):
    config["training"].update(steps=4, save_interval=3)
    config["grpo"]["update_epochs"] = epochs
    report = r6_local.run_grpo(config, ROOT, tmp_path / "interval")
    saved = sorted(
        int(path.stem.removeprefix("step-"))
        for path in (tmp_path / "interval").glob("*.pt")
        if path.stem.startswith("step-")
    )
    assert report["step"] == 4 and saved == expected
    for step in saved:
        payload = torch.load(tmp_path / "interval" / f"step-{step:06d}.pt", weights_only=True)
        assert payload["progress"]["rollout_complete"]
        assert step % epochs == 0


def test_dry_run_selects_evaluation_split_and_budget(config, tmp_path):
    report = r6_local.inspect_run(config, ROOT, command="grpo-evaluate", baseline=True)
    assert report["split"] == "validation" and report["baseline"] and report["dry_run"]
    assert report["target_steps"] == 2
    for command, steps in (("grpo-smoke", 11), ("grpo-profile", 101)):
        invalid = copy.deepcopy(config)
        invalid["training"]["steps"] = steps
        with pytest.raises(ValueError, match="smoke 最多 10"):
            r6_local.inspect_run(invalid, ROOT, command=command)
    (tmp_path / "validation.pt").unlink()
    with pytest.raises(FileNotFoundError):
        r6_local.inspect_run(config, ROOT, command="grpo-evaluate", baseline=True)
    assert r6_local.inspect_run(config, ROOT)["split"] == "train"


def test_gradient_checkpointing_rejected(config):
    config["training"]["gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        r6_local.validate_grpo(config, ROOT)


def test_zero_variance_no_weight_decay_update(config, tmp_path, monkeypatch):
    import omni.training.grpo_rewards as rewards

    monkeypatch.setattr(rewards, "reward_score", lambda *args: (1.0, {"constant": 1.0}))
    config["grpo"]["kl_coef"] = 0
    report = r6_local.run_grpo(config, ROOT, tmp_path / "zero")
    assert report["status"] == "no_update" and report["step"] == 0
    assert report["trainable_before"] == report["trainable_after"]
    assert report["metrics"][0]["grad_norm"] == 0
    assert report["metrics"][0]["zero_variance_groups"] == 1


def test_invalid_reward_preserves_error_evidence(config, tmp_path, monkeypatch):
    import omni.training.grpo_rewards as rewards

    def invalid(*args):
        raise ValueError("nonfinite reward")

    monkeypatch.setattr(rewards, "reward_score", invalid)
    with pytest.raises(ValueError, match="nonfinite reward"):
        r6_local.run_grpo(config, ROOT, tmp_path / "invalid")
    error = json.loads((tmp_path / "invalid/error.json").read_text())
    assert error["step"] == 0 and error["record_ids"]
    assert not list((tmp_path / "invalid").glob("step-*.pt"))


@pytest.mark.parametrize(
    "field,value", [("top_p", 0.9), ("top_k", 5), ("group_size", 1), ("temperature", 0)]
)
def test_invalid_sampling_rejected(config, field, value):
    config["grpo"][field] = value
    with pytest.raises(ValueError):
        r6_local.validate_grpo(config, ROOT)


def test_audio_mode_cannot_update_text_policy(config):
    config["grpo"]["mode"] = "audio"
    with pytest.raises(ValueError, match="冻结文本策略"):
        r6_local.validate_grpo(config, ROOT)


def test_sft_alpha_and_lineage_are_checked(config, tmp_path):
    source_settings = copy.deepcopy(config)
    source_settings.pop("grpo")
    model = r3_local.build_model(source_settings, ROOT)
    optimizer, _, scaler, _ = r3_local.optimizer_state(model, source_settings)
    identity = r3_local.run_identity(source_settings, ROOT)
    checkpoint, manifest = tmp_path / "sft.pt", tmp_path / "sft.json"
    save_checkpoint(checkpoint, model, optimizer, scaler, dict(identity=identity, step=0))
    manifest.write_text(
        json.dumps(
            dict(
                config=source_settings,
                checkpoint_sha256=file_hash(checkpoint),
                data_manifest_sha256=identity["data_manifest_sha256"],
                data_contract=CONTRACT,
                asset_identity=r3_local.adapter_identity(source_settings, CONTRACT),
            )
        )
    )
    config["grpo"]["policy"] = dict(
        source="sft", checkpoint=str(checkpoint), manifest=str(manifest), lineage=None
    )
    loaded, _ = r6_local.setup_model(config, ROOT, CONTRACT)
    assert_nested_equal(model.state_dict(), loaded.state_dict())
    original = torch.load(checkpoint, weights_only=True)
    original_manifest = json.loads(manifest.read_text())
    for marker in ("algorithm", "settings"):
        payload, metadata = copy.deepcopy(original), copy.deepcopy(original_manifest)
        if marker == "algorithm":
            payload["progress"]["algorithm"] = "grpo-v1"
        else:
            payload["progress"]["identity"]["settings"]["grpo"] = config["grpo"]
            metadata["config"] = payload["progress"]["identity"]["settings"]
        torch.save(payload, checkpoint)
        metadata["checkpoint_sha256"] = file_hash(checkpoint)
        manifest.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="初始策略必须来自 SFT"):
            r6_local.setup_model(config, ROOT, CONTRACT)
    torch.save(original, checkpoint)
    original_manifest["checkpoint_sha256"] = file_hash(checkpoint)
    manifest.write_text(json.dumps(original_manifest))
    config["lora"]["alpha"] += 1
    with pytest.raises(ValueError, match="LoRA 配置"):
        r6_local.setup_model(config, ROOT, CONTRACT)
    config["model"]["kind"] = "qwen3"
    config["grpo"]["reward"] = {"weights": {"text_match": 1.0}}
    with pytest.raises(ValueError, match="完整 B1"):
        r6_local.validate_grpo(config, ROOT)


def worker(path: Path, run: Path, steps: int, resume: Path | None = None):
    config = yaml.safe_load(path.read_text())
    r3_local.data_contract = lambda *args: CONTRACT
    r6_local.run_grpo(config, ROOT, run, steps=steps, resume=resume)


def test_new_process_resume_matches_continuous_and_adapter(config, tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config))
    for name, steps, resume in (("full", 2, None), ("part", 1, None), ("continued", 2, "part")):
        command = [sys.executable, __file__, str(path), str(tmp_path / name), str(steps)]
        if resume:
            command.append(str(tmp_path / resume / "step-000001.pt"))
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=os.environ
            | {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
    full = torch.load(tmp_path / "full/step-000002.pt", weights_only=True)
    resumed = torch.load(tmp_path / "continued/step-000002.pt", weights_only=True)
    assert_nested_equal(full, resumed)
    model, _ = r6_local.setup_model(config, ROOT, CONTRACT)
    load_adapter(tmp_path / "full/adapter.pt", model, r3_local.adapter_identity(config, CONTRACT))
    for name, value in model.state_dict().items():
        if not name.startswith("thinker.") or ".lora_" in name:
            torch.testing.assert_close(value, full["model"][name], rtol=0, atol=0)

    for field, value in (("rollout_complete", False), ("policy_version", 99)):
        broken = copy.deepcopy(full)
        broken["progress"][field] = value
        torch.save(broken, tmp_path / "invalid.pt")
        for dry_run in (False, True):
            with pytest.raises(ValueError, match="完整 rollout"):
                if dry_run:
                    r6_local.inspect_run(
                        config, ROOT, resume=tmp_path / "invalid.pt", command="grpo-evaluate"
                    )
                else:
                    r6_local.run_grpo(
                        config,
                        ROOT,
                        tmp_path / "rejected",
                        resume=tmp_path / "invalid.pt",
                        command="grpo-evaluate",
                    )
        assert not (tmp_path / "rejected").exists()

    broken = copy.deepcopy(full)
    broken["progress"]["identity"]["grpo_schema_version"] = 99
    torch.save(broken, tmp_path / "wrong-identity.pt")
    with pytest.raises(ValueError, match="身份不一致"):
        r6_local.inspect_run(
            config, ROOT, resume=tmp_path / "wrong-identity.pt", command="grpo-evaluate"
        )

    # Reference 与 current 共享的冻结主权重不能随 Resume 漂移。
    broken = copy.deepcopy(full)
    name = next(
        key for key in broken["model"] if key.startswith("thinker.") and ".lora_" not in key
    )
    broken["model"][name].add_(0.25)
    torch.save(broken, tmp_path / "changed-base.pt")
    with pytest.raises(ValueError, match="冻结主权重"):
        r6_local.run_grpo(
            config,
            ROOT,
            tmp_path / "changed-base",
            resume=tmp_path / "changed-base.pt",
            command="grpo-evaluate",
        )
    assert not (tmp_path / "changed-base").exists()


def test_cli_and_reward_config_replacement():
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for name in ("grpo-smoke", "grpo-train", "grpo-resume", "grpo-evaluate", "grpo-profile"):
        assert name in result.stdout
    audio = r3_local.merge_settings(ROOT / "configs/v2/r6_audio_h100.yaml")
    assert audio["grpo"]["reward"]["weights"] == {"asr_match": 1.0, "completeness": 0.1}


if __name__ == "__main__":
    worker(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        int(sys.argv[3]),
        Path(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
