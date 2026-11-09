"""同轮补充复核：多任务 Projector 梯度与 Adaptive Bridge 的真实 RL 更新。"""

import json

import pytest
import torch
from test_r6_training import CONTRACT, ROOT
from test_r6_training import config as config
from test_r6_training import fixture_settings as fixture_settings

from omni.training import r6_local


def test_joint_multitask_updates_projectors_through_frozen_qwen(config, tmp_path):
    config["training"].update(tasks=["t2a", "i2t", "a2t"], accumulation_steps=3)
    config["grpo"]["trainable"] = [
        "vision_proj",
        "audio_proj",
        "bridge",
        "talker",
        "audio_embedding",
        "audio_heads",
    ]
    torch.manual_seed(config["training"]["seed"])
    initial, _ = r6_local.setup_model(config, ROOT, CONTRACT)
    before = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    assert all(not param.requires_grad for param in initial.thinker.parameters())

    report = r6_local.run_grpo(config, ROOT, tmp_path / "multitask")
    saved = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)["model"]
    assert report["status"] == "passed" and report["step"] == 2
    assert report["frozen_before"] == report["frozen_after"]
    for prefix in ("vision_proj.", "audio_proj."):
        assert report["gradient_groups"][prefix.removesuffix(".")] > 0
        assert any(
            not torch.equal(value, saved[name])
            for name, value in before.items()
            if name.startswith(prefix)
        )
    for name, value in before.items():
        if name.startswith("thinker."):
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)

    # 三种 prompt 都实际进入同一 joint 更新器，不能只构建而未被采样。
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    tasks = {row["record_id"]: row["task"] for row in manifest["selections"]["train"]}
    records = [
        json.loads(line)
        for line in (tmp_path / "multitask/rollouts.jsonl").read_text().splitlines()
    ]
    assert {tasks[row["record_id"]] for row in records} == {"t2a", "i2t", "a2t"}
    assert all(row["mode"] == "joint" for row in records)


@pytest.mark.parametrize(
    "kind,fusion_names",
    [
        ("weighted", ["bridge.layer_logits"]),
        ("gated", ["bridge.gate_weight", "bridge.gate_bias"]),
    ],
)
def test_adaptive_bridge_fusion_parameters_receive_grpo_updates(
    config, tmp_path, kind, fusion_names
):
    config["model"]["bridge"] = {"kind": kind, "layers": [0, 1]}
    config["grpo"]["trainable"] = ["bridge"]
    torch.manual_seed(config["training"]["seed"])
    initial, _ = r6_local.setup_model(config, ROOT, CONTRACT)
    before = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    assert all(torch.count_nonzero(before[name]) == 0 for name in fusion_names)

    report = r6_local.run_grpo(config, ROOT, tmp_path / kind)
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    assert report["status"] == "passed" and report["gradient_groups"]["bridge"] > 0
    assert report["frozen_before"] == report["frozen_after"]
    assert checkpoint["config"]["bridge"] == {"kind": kind, "layers": [0, 1]}
    for name in fusion_names:
        # 初值为零，AdamW weight decay 不能独自产生这些非零变化。
        assert checkpoint["model"][name].isfinite().all()
        assert torch.count_nonzero(checkpoint["model"][name]) > 0
    for name, value in before.items():
        if not name.startswith("bridge."):
            torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
