"""R4 共用训练器的任务加权、冻结边界、配置与完整进程恢复。"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_r3_training import assert_nested_equal, batch, tiny_model

from omni.data.omni_dataset import OmniCollator
from omni.training import r3_local
from omni.training.adapters import inject_lora, load_adapter
from omni.training.losses import loss_counts, sft_loss
from omni.training.sft import train_step

ROOT = Path(__file__).resolve().parents[1]


def settings(cache: Path, recipe: str = "joint") -> dict:
    values = r3_local.merge_settings(ROOT / "configs/v2/r4d_local.yaml")
    values["data"]["cache_dir"] = str(cache)
    values["training"].update(threads=1, recipe=recipe)
    return values


def samples() -> list[dict]:
    rows = []
    for index in range(3):
        for task in ("t2a", "i2t", "a2t", "a2a"):
            kind = "image" if task == "i2t" else "audio" if task in {"a2t", "a2a"} else None
            source = batch(index, modality=kind)
            row = {
                name: value[0] if isinstance(value, (torch.Tensor, list)) else value
                for name, value in source.items()
            }
            row.setdefault("image_features", None)
            row.setdefault("audio_features", None)
            row["spk_emb"] = None
            if task in {"i2t", "a2t"}:
                row["audio_labels"].fill_(-100)
            row["task"] = task
            rows.append(row)
    return rows


def fixture_model(config: dict, root: Path) -> torch.nn.Module:
    model = tiny_model()
    inject_lora(model.thinker.model, config["lora"])
    r3_local.configure_parameters(model, config["training"]["recipe"])
    return model


def test_task_weighting_matches_explicit_objective() -> None:
    torch.manual_seed(42)
    actual = tiny_model()
    # 关闭随机层，直接比较独立手算 Loss 和每个参数的梯度。
    actual.config.dropout = 0
    for layer in actual.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
        if isinstance(getattr(layer, "dropout", None), float):
            layer.dropout = 0.0
        if hasattr(getattr(layer, "config", None), "dropout"):
            layer.config.dropout = 0.0
    expected = copy.deepcopy(actual)
    for model in (actual, expected):
        r3_local.configure_parameters(model, "r4a")
    batches = [batch(0), batch(1)]
    optimizer = torch.optim.SGD([p for p in actual.parameters() if p.requires_grad], lr=0.0)
    metrics = train_step(
        actual,
        batches,
        optimizer,
        torch.amp.GradScaler("cpu", enabled=False),
        grad_clip=1e9,
        batch_tasks=["t2a", "a2a"],
        task_weights={"t2a": 3.0, "a2a": 1.0},
    )
    reference = 0
    for current, weight in zip(batches, (0.75, 0.25), strict=True):
        output = expected(
            **{k: v for k, v in current.items() if k not in {"labels", "audio_labels"}}
        )
        reference = (
            reference
            + sft_loss(
                output,
                current["labels"],
                current["audio_labels"],
                10,
                10.0,
                loss_counts([current]),
                aux_scale=1,
            )["total"]
            * weight
        )
    reference.backward()
    assert metrics["total"] == pytest.approx(float(reference.detach()), rel=1e-6)
    for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)


def test_single_task_window_keeps_configured_weight() -> None:
    base = tiny_model()
    values = []
    for weight in (1.0, 3.0):
        model = copy.deepcopy(base)
        r3_local.configure_parameters(model, "r4a")
        torch.manual_seed(17)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0)
        result = train_step(
            model,
            [batch(0)],
            optimizer,
            torch.amp.GradScaler("cpu", enabled=False),
            grad_clip=1e9,
            batch_tasks=["t2a"],
            task_weights={"t2a": weight, "i2t": 1.0},
        )
        values.append(result)
    # 固定配置分母：lambda=1 时 1/2，lambda=3 时 3/4；不因缺 I2T 消掉权重。
    assert values[1]["total"] == pytest.approx(values[0]["total"] * 1.5, rel=1e-6)
    assert values[1]["grad_norm"] == pytest.approx(values[0]["grad_norm"] * 1.5, rel=1e-6)


def test_weighted_task_sampler_and_cursor() -> None:
    config = settings(Path("unused"))["training"]
    config.update(accumulation_steps=4, sample_weights={"t2a": 3, "i2t": 1, "a2t": 2, "a2a": 2})
    model = tiny_model()
    progress, tasks = {}, []
    for _ in range(2):
        _, chosen = r3_local.next_batches(samples(), progress, config, OmniCollator(model.config))
        tasks.extend(chosen)
    assert {task: tasks.count(task) for task in set(tasks)} == {
        "t2a": 3,
        "i2t": 1,
        "a2t": 2,
        "a2a": 2,
    }
    resumed = copy.deepcopy(progress)
    left, task_left = r3_local.next_batches(samples(), progress, config, OmniCollator(model.config))
    right, task_right = r3_local.next_batches(
        samples(), resumed, config, OmniCollator(model.config)
    )
    assert task_left == task_right
    assert_nested_equal(left, right)
    assert progress == resumed


@pytest.mark.parametrize(
    "recipe,active",
    [
        ("vision", {"vision_proj"}),
        ("audio", {"audio_proj"}),
        (
            "joint",
            {
                "bridge",
                "talker",
                "audio_embedding",
                "audio_heads",
                "vision_proj",
                "audio_proj",
                "lora",
            },
        ),
    ],
)
def test_independent_recipes_and_group_learning_rates(recipe: str, active: set[str]) -> None:
    config = settings(Path("unused"), recipe)
    config["training"]["group_lrs"] = {"vision_proj": 0.002, "audio_proj": 0.004, "lora": 0.0001}
    model = fixture_model(config, ROOT)
    groups = r3_local.configure_parameters(model, recipe)
    assert {name for name, counts in groups.items() if counts["trainable"]} == active
    optimizer, _, _, _ = r3_local.optimizer_state(model, config)
    assert len({id(param) for group in optimizer.param_groups for param in group["params"]}) == sum(
        len(group["params"]) for group in optimizer.param_groups
    )
    for group in optimizer.param_groups:
        assert group["initial_lr"] == config["training"]["group_lrs"].get(
            group["name"], config["training"]["learning_rate"]
        )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.thinker.named_parameters()
        if ".lora_" not in name
    )


def test_explicit_initialization_combines_independent_stages(tmp_path: Path) -> None:
    original = tiny_model()
    paths = {}
    for index, name in enumerate(("components", "talker", "vision", "audio"), start=1):
        state = {key: torch.full_like(value, index) for key, value in original.state_dict().items()}
        path = tmp_path / f"{name}.pt"
        torch.save({"model": state}, path)
        paths[name + "_checkpoint"] = str(path)
    frozen = {name: value.clone() for name, value in original.thinker.state_dict().items()}
    config = {"model": {"initialization": paths}}
    r3_local.initialize_components(original, config, ROOT)
    for prefix, number in (("bridge", 1), ("talker", 2), ("vision_proj", 3), ("audio_proj", 4)):
        assert all(
            torch.equal(value, torch.full_like(value, number))
            for value in original.get_submodule(prefix).state_dict().values()
        )
    assert_nested_equal(frozen, original.thinker.state_dict())
    torch.save({"bad": torch.zeros(1)}, tmp_path / "invalid.pt")
    config["model"]["initialization"] = {"talker_checkpoint": str(tmp_path / "invalid.pt")}
    with pytest.raises(ValueError, match="不兼容"):
        r3_local.initialize_components(original, config, ROOT)


def test_inheritance_replaces_task_budget_and_weight_tables(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("data: {tasks: {t2a: 8, i2t: 8}}\ntraining: {task_weights: {t2a: 1, i2t: 2}}\n")
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\ndata: {tasks: {t2a: 4}}\ntraining: {task_weights: {t2a: 3}}\n"
    )
    value = r3_local.merge_settings(child)
    assert value["data"]["tasks"] == {"t2a": 4}
    assert value["training"]["task_weights"] == {"t2a": 3}


def test_joint_without_lora_and_invalid_config(tmp_path: Path) -> None:
    config = settings(tmp_path)
    config["lora"]["enabled"] = False
    r3_local.validate_r4(config, ROOT)
    config["training"]["gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="checkpointing"):
        r3_local.validate_r4(config, ROOT)
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="循环"):
        r3_local.merge_settings(path)


def test_qa_guard_rejects_echo_and_missing_qa(tmp_path: Path) -> None:
    config = settings(tmp_path)
    config["training"]["allow_diagnostic_a2a"] = False
    with pytest.raises(ValueError, match="echo"):
        r3_local.check_task_semantics(config, {})
    config["data"]["audited_manifest"] = "qa.jsonl"
    manifest = {"selections": {"train": [{"task": "a2a", "a2a_kind": "echo"}]}}
    with pytest.raises(ValueError, match="QA"):
        r3_local.check_task_semantics(config, manifest)
    manifest["selections"]["train"][0]["a2a_kind"] = "qa"
    r3_local.check_task_semantics(config, manifest)


def worker(
    output: Path, cache: Path, steps: int, resume: Path | None = None, early: bool = False
) -> None:
    import omni.data.r3_prepare

    config = settings(cache)
    if early:
        config["training"].update(min_delta=1e6, patience=1, eval_interval=1)
    r3_local.build_model = fixture_model
    r3_local.data_contract = lambda settings, root: {"qwen_revision": "fixture", "assets": {}}
    omni.data.r3_prepare.load_prepared = lambda cache, split, contract: samples()
    r3_local.smoke(config, ROOT, output, steps=steps, resume=resume)


def test_cross_process_joint_resume_and_adapter(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    for target, steps, source in (("full", 4, None), ("part", 2, None), ("resumed", 4, "part")):
        args = [sys.executable, __file__, str(tmp_path / target), str(cache), str(steps)]
        if source:
            args.append(str(tmp_path / source / "step-000002.pt"))
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        )
        assert result.returncode == 0, result.stderr
    full = torch.load(tmp_path / "full/step-000004.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/step-000004.pt", weights_only=True)
    assert_nested_equal(full, resumed)
    report = json.loads((tmp_path / "full/manifest.json").read_text())
    assert report["frozen_before"] == report["frozen_after"]
    assert set(report["validation"]["tasks"]) == {"t2a", "i2t", "a2t", "a2a"}
    assert len(report["validation"]["tasks"]["t2a"]["codebook_loss"]) == 8
    assert report["gradient_groups"]["lora"] > 0
    assert (tmp_path / "full/loss.svg").is_file()
    assert list((tmp_path / "full").glob("best-*.pt"))
    config = settings(cache)
    model = fixture_model(config, ROOT)
    load_adapter(
        tmp_path / "full/adapter.pt",
        model,
        r3_local.adapter_identity(config, {"qwen_revision": "fixture", "assets": {}}),
    )
    for name, value in model.state_dict().items():
        if not name.startswith("thinker.") or ".lora_" in name:
            torch.testing.assert_close(value, full["model"][name], rtol=0, atol=0)


def test_early_stopping_retains_checkpoint(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "run"), str(cache), "4", "early"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "run/manifest.json").read_text())
    assert report["stop_reason"] == "early_stopping" and report["step"] == 2
    assert report["best_step"] == 1
    assert Path(report["checkpoint"]).is_file()


def test_cli_help_exposes_implemented_commands() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for command in ("train", "resume", "evaluate", "demo", "profile"):
        assert command in result.stdout


if __name__ == "__main__":
    extra = sys.argv[4] if len(sys.argv) > 4 else None
    worker(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        int(sys.argv[3]),
        Path(extra) if extra and extra != "early" else None,
        extra == "early",
    )
