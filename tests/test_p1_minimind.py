"""P1 检查脚本的边界测试；不读取真实大权重，不下载资源。"""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "check_minimind", Path(__file__).resolve().parents[1] / "scripts/check_minimind.py"
)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_asset_collection_checks_and_repairs_copies(tmp_path, monkeypatch) -> None:
    reference, project = tmp_path / "reference", tmp_path / "project"
    for folder in ("out", "model"):
        (reference / folder).mkdir(parents=True)
    for name in ("llm_768.pth", "llm_768_moe.pth", "README.md"):
        (reference / "out" / name).write_text(name)
    (reference / "LICENSE").write_text("Apache License 2.0")
    (reference / "out/README.md").write_text("license: apache-2.0")
    for name in ("tokenizer.json", "tokenizer_config.json", "model_minimind.py"):
        (reference / "model" / name).write_text(name)
    monkeypatch.setattr(checks.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")

    manifest = checks.collect_assets(reference, project)
    assert manifest["reference_commit"] == "abc123"
    assert len(manifest["files"]) == 6
    for asset in manifest["files"]:
        assert asset["source"]["sha256"] == asset["local"]["sha256"]
        assert (
            Path(asset["source"]["path"]).read_bytes() == Path(asset["local"]["path"]).read_bytes()
        )

    damaged = project / "weights/minimind/llm_768.pth"
    damaged.write_text("damaged")
    repaired = checks.collect_assets(reference, project)
    assert repaired["files"][0]["local"]["sha256"] == manifest["files"][0]["local"]["sha256"]
    assert (reference / "out/llm_768.pth").read_text() == "llm_768.pth"


def test_numerical_comparison_does_not_relax_tolerance() -> None:
    expected = torch.tensor([1.0, -1.0])
    assert checks.compare(expected + 1e-5, expected) < 2e-5
    with pytest.raises(AssertionError):
        checks.compare(expected + 1e-2, expected)


def test_reference_import_keeps_source_directory_read_only(tmp_path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    source = folder / "model_minimind.py"
    source.write_text("VALUE = 42\n")

    reference = checks.load_reference(tmp_path)
    assert reference.VALUE == 42
    assert source.read_text() == "VALUE = 42\n"
    assert list(folder.iterdir()) == [source]


def test_inconsistent_source_tied_weights_are_rejected(tmp_path, monkeypatch) -> None:
    # 在大模型构造前拒绝冲突，避免后加载的 head 悄悄覆盖 embedding。
    state = {"model.embed_tokens.weight": torch.zeros(2, 2), "lm_head.weight": torch.ones(2, 2)}
    monkeypatch.setattr(checks.torch, "load", lambda *args, **kwargs: state)
    with pytest.raises(ValueError, match="embedding/head"):
        checks.check_variant("dense", tmp_path, tmp_path)


def test_layer_capture_snapshots_before_inplace_changes() -> None:
    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1, None

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

    model = Model()
    outputs, handles = checks.capture_layers(model)
    result = model.model.layers[0](torch.zeros(1))[0]
    result.add_(9)
    assert outputs[0].item() == 1
    for handle in handles:
        handle.remove()
    model.model.layers[0](torch.zeros(1))
    assert len(outputs) == 1
