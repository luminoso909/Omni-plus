"""R5 复用原训练/恢复入口，并验证从同一 Fixed 状态开始的权重迁移。"""

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml
from test_r3_training import assert_nested_equal, tiny_model
from test_r4_training import samples, settings

from omni.data import r3_prepare
from omni.training import r3_local
from omni.training.adapters import load_adapter
from omni.training.sft import save_checkpoint
from scripts import bridge_ablation

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {
    "qwen_revision": "fixture",
    "assets": {
        "qwen/tokenizer.json": "fixture-tokenizer",
        "mimi/model.safetensors": "fixture-mimi",
    },
}


@pytest.fixture
def fixture_settings(tmp_path: Path) -> dict:
    config = settings(tmp_path)
    original = tiny_model()
    path = tmp_path / "omni.yaml"
    path.write_text(yaml.safe_dump({"model": original.config.to_dict()}))
    config["model"]["config"] = str(path)
    config["model"]["tiny"] = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        talker_hidden_size=16,
    )
    config["assets"]["qwen_dir"] = str(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    native = original.thinker.model.config.to_dict()
    native["hidden_size"] = 2048  # read_settings 只检查官方资产元数据，此 fixture 始终 Tiny。
    (tmp_path / "config.json").write_text(json.dumps(native))
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).touch()
    return config


def fixed_source(
    config: dict, tmp_path: Path, contract: dict = CONTRACT
) -> tuple[Path, Path, dict]:
    torch.manual_seed(config["training"]["seed"])
    model = r3_local.build_model(config, ROOT)
    optimizer, _, scaler, _ = r3_local.optimizer_state(model, config)
    with torch.no_grad():
        # 模拟已训练的非零权重，确认迁移包含 LoRA 与所有共享组。
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.fill_(0.25 if ".lora_b." in name else 0.5)
    checkpoint = tmp_path / "fixed.pt"
    identity = r3_local.run_identity(config, ROOT)
    save_checkpoint(checkpoint, model, optimizer, scaler, {"identity": identity, "step": 1})
    manifest = tmp_path / "fixed.json"
    manifest.write_text(
        json.dumps(
            dict(
                config=config,
                asset_identity=r3_local.adapter_identity(config, contract),
                checkpoint_sha256=r3_local.digest(checkpoint),
                data_manifest_sha256=identity["data_manifest_sha256"],
                data_contract=contract,
            )
        )
    )
    return checkpoint, manifest, model.state_dict()


@pytest.mark.parametrize("kind", ["fixed", "last", "weighted", "gated"])
def test_warm_start_copies_shared_weights_and_keeps_fusion_init(
    fixture_settings: dict, tmp_path: Path, kind: str
) -> None:
    checkpoint, manifest, shared = fixed_source(fixture_settings, tmp_path)
    spec = {"kind": kind}
    if kind in {"weighted", "gated"}:
        spec["layers"] = [0, 1]
    fixture_settings["model"].update(
        bridge=spec,
        initialization={
            "bridge_base_checkpoint": str(checkpoint),
            "bridge_base_manifest": str(manifest),
        },
    )
    torch.manual_seed(999)  # 任意初始化值不能漏进共享组。
    model = r3_local.build_model(fixture_settings, ROOT)
    identity = r3_local.run_identity(fixture_settings, ROOT)
    assert identity["initialization_sha256"]["bridge_base_manifest"] == r3_local.digest(manifest)
    r3_local.initialize_bridge_base(
        model, fixture_settings, ROOT, CONTRACT, identity["initialization_sha256"]
    )
    state = model.state_dict()
    for name in shared:
        torch.testing.assert_close(state[name], shared[name], rtol=0, atol=0)
    for name in state.keys() - shared.keys():
        assert name.startswith("bridge.") and torch.count_nonzero(state[name]) == 0
    optimizer, _, _, _ = r3_local.optimizer_state(model, fixture_settings)
    assert not optimizer.state
    assert any(group["name"] == "bridge" for group in optimizer.param_groups)


@pytest.mark.parametrize("invalid", ["digest", "assets", "lora", "fixed_layer"])
def test_warm_start_rejects_incompatible_source(
    fixture_settings: dict, tmp_path: Path, invalid: str
) -> None:
    checkpoint, manifest, _ = fixed_source(fixture_settings, tmp_path)
    fixture_settings["model"].update(
        bridge={"kind": "weighted", "layers": [0, 1]},
        initialization={
            "bridge_base_checkpoint": str(checkpoint),
            "bridge_base_manifest": str(manifest),
        },
    )
    contract = copy.deepcopy(CONTRACT)
    data = json.loads(manifest.read_text())
    if invalid == "digest":
        data["checkpoint_sha256"] = "wrong"
        manifest.write_text(json.dumps(data))
    elif invalid == "assets":
        contract["assets"]["mimi/model.safetensors"] = "changed"
    elif invalid == "lora":
        fixture_settings["lora"]["alpha"] += 1
    else:
        fixture_settings["model"]["bridge"] = {"kind": "fixed", "layers": [1]}
    model = r3_local.build_model(fixture_settings, ROOT)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    identity = r3_local.run_identity(fixture_settings, ROOT)
    message = {
        "digest": "SHA-256",
        "assets": "资产身份",
        "lora": "LoRA 配置",
        "fixed_layer": "Fixed 对照组",
    }[invalid]
    with pytest.raises(ValueError, match=message):
        r3_local.initialize_bridge_base(
            model, fixture_settings, ROOT, contract, identity["initialization_sha256"]
        )
    assert_nested_equal(before, model.state_dict())


def test_leaked_media_rejected_by_plan_and_warm_start(
    fixture_settings: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = dict(
        task="t2a", record_id="original", group_id="original-group", media_ids=["sha256:voice"]
    )
    source = CONTRACT | {"records": [row | {"split": "train"}]}
    checkpoint, manifest, _ = fixed_source(fixture_settings, tmp_path, source)
    fixture_settings["model"]["initialization"] = dict(
        bridge_base_checkpoint=str(checkpoint), bridge_base_manifest=str(manifest)
    )
    heldout = row | {"split": "test", "record_id": "alias", "group_id": "different-group"}
    contract = CONTRACT | {"records": [heldout]}
    model = r3_local.build_model(fixture_settings, ROOT)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    identity = r3_local.run_identity(fixture_settings, ROOT)
    with pytest.raises(ValueError, match="留出集冲突"):
        r3_local.initialize_bridge_base(
            model, fixture_settings, ROOT, contract, identity["initialization_sha256"]
        )
    assert_nested_equal(before, model.state_dict())

    split_file = tmp_path / "test.pt"
    split_file.write_bytes(b"fixture split")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                contract=contract,
                files={split_file.name: r3_local.digest(split_file)},
                selections={"test": [heldout]},
            )
        )
    )
    fixture_settings["evaluation"]["benchmark"] = {"split": "test", "t2a_samples": 1}
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(fixture_settings))
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("fixed: {kind: fixed, layers: [0]}\nlast: {kind: last, layers: [1]}\n")
    output = tmp_path / "matrix"
    monkeypatch.setattr(bridge_ablation, "data_contract", lambda *_: contract)
    with pytest.raises(ValueError, match="留出集冲突"):
        bridge_ablation.plan(base, candidates, output, ROOT)
    assert not output.exists()


def test_components_rejects_last_with_fixed_keys(fixture_settings: dict, tmp_path: Path) -> None:
    checkpoint, _, _ = fixed_source(fixture_settings, tmp_path)
    fixture_settings["model"].update(
        bridge={"kind": "last"}, initialization={"components_checkpoint": str(checkpoint)}
    )
    with pytest.raises(ValueError, match="Bridge 类型或选层"):
        r3_local.build_model(fixture_settings, ROOT)


def test_yaml_bridge_resolution_and_invalid_layers(fixture_settings: dict, tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    fixture_settings["model"]["bridge"] = {"kind": "weighted", "layers": [0, 1]}
    path.write_text(yaml.safe_dump(fixture_settings))
    values = r3_local.read_settings(path, ROOT)
    model = r3_local.build_model(values, ROOT, meta=True)
    assert model.config.to_dict()["bridge"] == {"kind": "weighted", "layers": [0, 1]}
    fixture_settings["model"]["bridge"]["layers"] = [0, 2]
    path.write_text(yaml.safe_dump(fixture_settings))
    with pytest.raises(ValueError, match="Bridge"):
        r3_local.read_settings(path, ROOT)


def test_weighted_t2a_exact_resume_and_strict_adapter(
    fixture_settings: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r3_local, "data_contract", lambda settings, root: CONTRACT)
    monkeypatch.setattr(r3_prepare, "load_prepared", lambda path, split, contract: samples())
    fixture_settings["model"]["bridge"] = {"kind": "weighted", "layers": [0, 1]}
    fixture_settings["lora"]["enabled"] = False
    training = fixture_settings["training"]
    training.update(
        recipe="r4a", tasks=["t2a"], steps=2, accumulation_steps=1, eval_interval=1, save_interval=1
    )
    full = r3_local.smoke(fixture_settings, ROOT, tmp_path / "full")
    part = r3_local.smoke(fixture_settings, ROOT, tmp_path / "part", steps=1)
    resumed = r3_local.smoke(
        fixture_settings, ROOT, tmp_path / "resumed", resume=Path(part["checkpoint"])
    )
    full_checkpoint = torch.load(full["checkpoint"], weights_only=True)
    assert_nested_equal(full_checkpoint, torch.load(resumed["checkpoint"], weights_only=True))
    assert full["frozen_before"] == full["frozen_after"]
    assert full["bridge_gradient_max_abs"]["bridge.layer_logits"] > 0
    assert torch.count_nonzero(full_checkpoint["model"]["bridge.layer_logits"]) == 2

    different_order = copy.deepcopy(fixture_settings)
    different_order["model"]["bridge"]["layers"].reverse()
    with pytest.raises(ValueError, match="Resume 配置"):
        r3_local.smoke(different_order, ROOT, tmp_path / "bad", resume=Path(part["checkpoint"]))
    model = r3_local.build_model(different_order, ROOT)
    with pytest.raises(ValueError, match="配置"):
        load_adapter(
            Path(full["adapter"]), model, r3_local.adapter_identity(different_order, CONTRACT)
        )
