"""R3-L 参数边界、有限配置和跨新进程的训练恢复。"""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from omni.models.config import OmniPlusConfig
from omni.models.model_omni_v2 import OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.r3_local import (
    configure_parameters,
    optimizer_state,
    parameter_digest,
    read_settings,
)
from omni.training.sft import load_checkpoint, save_checkpoint, train_step


def tiny_model() -> OmniPlusOmni:
    config = OmniPlusConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        vocab_size=32,
        image_ids=[12],
        audio_ids=[13],
        image_hidden_size=3,
        audio_hidden_size=4,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=32,
        talker_adapter_rank=4,
        num_talker_hidden_layers=2,
        audio_codebook_size=8,
        audio_vocab_size=16,
        audio_pad_token=9,
        audio_stop_token=10,
        audio_spk_token=11,
        pad_token_id=0,
        dropout=0.2,
    )
    native = Qwen3Config(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        vocab_size=32,
        head_dim=8,
    )
    return OmniPlusOmni(QwenBackend(Qwen3ForCausalLM(native)), config)


def settings() -> dict:
    return dict(
        training=dict(
            device="cpu",
            dtype="float32",
            learning_rate=0.001,
            weight_decay=0.01,
            warmup_steps=1,
            steps=4,
        )
    )


def batch(index: int, *, modality: str | None = None) -> dict:
    tokens = torch.tensor([[1, 3, 4 + index, 5]])
    features = {}
    if modality:
        tokens[0, 1] = 12 if modality == "image" else 13
        features[f"{modality}_features"] = [torch.ones(1, 3 if modality == "image" else 4)]
    return dict(
        input_ids=tokens,
        audio_ids=torch.full((1, 8, 4), 9, dtype=torch.long),
        attention_mask=torch.ones_like(tokens),
        labels=torch.tensor([[-100, 4 + index, 5, 2]]),
        audio_labels=torch.full((1, 8, 4), index + 1, dtype=torch.long),
        **features,
    )


@pytest.mark.parametrize("recipe", ["r4a", "projectors"])
def test_parameter_groups_and_frozen_updates(recipe: str) -> None:
    torch.manual_seed(42)
    model = tiny_model()
    groups = configure_parameters(model, recipe)
    assert sum(group["total"] for group in groups.values()) == sum(
        p.numel() for p in model.parameters()
    )
    assert groups["thinker"]["trainable"] == groups["speaker"]["trainable"] == 0
    if recipe == "r4a":
        assert groups["vision_proj"]["trainable"] == groups["audio_proj"]["trainable"] == 0
        assert groups["audio_embedding"]["trainable"] > 0
        assert groups["audio_heads"]["trainable"] > 0
    before = parameter_digest(model, frozen=True)
    optimizer, _, scaler, dtype = optimizer_state(model, settings())
    losses = train_step(
        model,
        [batch(0, modality="image"), batch(1, modality="audio")],
        optimizer,
        scaler,
        "cpu",
        dtype,
    )
    assert losses["grad_norm"] > 0 and np.isfinite(losses["total"])
    assert parameter_digest(model, frozen=True) == before
    assert all(param.grad is None for param in model.thinker.parameters())
    if recipe == "projectors":
        for projector in (model.vision_proj, model.audio_proj):
            assert any(
                p.grad is not None and p.grad.abs().max() > 0 for p in projector.parameters()
            )


def worker(output: Path, steps: int, resume: Path | None = None) -> None:
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = tiny_model()
    configure_parameters(model, "r4a")
    optimizer, scheduler, scaler, dtype = optimizer_state(model, settings())
    progress = dict(step=0, epoch=0, cursor=0)
    if resume:
        progress = load_checkpoint(resume, model, optimizer, scaler)
        scheduler.load_state_dict(progress["scheduler"])
    while progress["step"] < steps:
        batches = []
        for _ in range(2):
            if progress["cursor"] == 3:
                progress["epoch"] += 1
                progress["cursor"] = 0
            order = torch.randperm(
                3, generator=torch.Generator().manual_seed(42 + progress["epoch"])
            )
            batches.append(batch(int(order[progress["cursor"]])))
            progress["cursor"] += 1
        train_step(model, batches, optimizer, scaler, "cpu", dtype)
        scheduler.step()
        progress["step"] += 1
        progress["scheduler"] = scheduler.state_dict()
    save_checkpoint(output, model, optimizer, scaler, progress)


def assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_fresh_process_resume_matches_uninterrupted(tmp_path: Path) -> None:
    for target, steps, source in (("full", 4, None), ("part", 2, None), ("resumed", 4, "part")):
        args = [sys.executable, __file__, str(tmp_path / f"{target}.pt"), str(steps)]
        if source:
            args.append(str(tmp_path / f"{source}.pt"))
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=os.environ
            | {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, result.stderr
    full = torch.load(tmp_path / "full.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed.pt", weights_only=True)
    assert full["progress"]["epoch"] == 2
    assert_nested_equal(full, resumed)


def test_reject_changed_freeze_policy(tmp_path: Path) -> None:
    worker(tmp_path / "state.pt", 1)
    model = tiny_model()
    configure_parameters(model, "projectors")
    optimizer, _, scaler, _ = optimizer_state(model, settings())
    with pytest.raises(ValueError, match="冻结策略"):
        load_checkpoint(tmp_path / "state.pt", model, optimizer, scaler)


@pytest.mark.parametrize("error", ["typo", "lora", "budget", "recipe", "missing_weights"])
def test_config_rejects_unsupported_or_missing_inputs(tmp_path: Path, error: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/v2/r3_local.yaml").read_text())
    if error == "typo":
        config["training"]["learinng_rate"] = 1
    elif error == "lora":
        config["lora"]["enabled"] = True
    elif error == "budget":
        config["training"]["steps"] = 11
    elif error == "recipe":
        config["training"]["recipe"] = "projectors"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, FileNotFoundError)):
        read_settings(path, tmp_path)


if __name__ == "__main__":
    worker(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
