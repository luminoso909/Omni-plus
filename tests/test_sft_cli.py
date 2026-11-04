"""极小缓存的 CLI 集成测试；不下载媒体或模型，不冒充真实样本验收。"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml
from test_sft import samples, small_config

SPEC = importlib.util.spec_from_file_location(
    "train_sft_cli", Path(__file__).parents[1] / "scripts/train_sft.py"
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def settings_file(tmp_path: Path) -> Path:
    config = small_config(dropout=0.1).to_dict()
    for split, ids in (("train", ["t1", "t2", "t3"]), ("validation", ["v1"])):
        items = [copy.deepcopy(samples()[index % 2]) for index in range(len(ids))]
        selections = [
            dict(record_id=identity, task=task, source_row=index)
            for index, (identity, task) in enumerate(zip(ids, ("t2a", "a2a", "i2t"), strict=False))
        ]
        torch.save(
            dict(samples=items, selections=selections, config=config), tmp_path / f"{split}.pt"
        )
    settings = dict(
        data={split: str(tmp_path / f"{split}.pt") for split in ("train", "validation")},
        model={},
        training=dict(
            device="cpu",
            dtype="float32",
            seed=42,
            freeze="all",
            batch_size=1,
            accumulation_steps=2,
            steps=4,
            learning_rate=0.003,
            weight_decay=0,
            grad_clip=1,
            stop_weight=10,
            eval_every=2,
        ),
    )
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_cli_resume_matches_uninterrupted_training(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    config = settings_file(tmp_path)
    first = json.loads((CLI.run(config, steps=2) / "report.json").read_text())
    resumed = json.loads(
        (CLI.run(config, resume=Path(first["checkpoint"]), steps=4) / "report.json").read_text()
    )
    full = json.loads((CLI.run(config, steps=4) / "report.json").read_text())
    resumed_state = torch.load(resumed["checkpoint"], weights_only=True)
    full_state = torch.load(full["checkpoint"], weights_only=True)
    assert resumed["final_train"] == full["final_train"]
    assert resumed["optimizer_steps"] == full["optimizer_steps"] == 4
    assert resumed_state["progress"] == full_state["progress"]
    for name, value in resumed_state["model"].items():
        assert torch.equal(value, full_state["model"][name]), name


def test_cached_label_contract_cannot_be_overridden(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    path = settings_file(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["model"]["audio_stop_token"] = 11
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="audio_stop_token"):
        CLI.run(path)
