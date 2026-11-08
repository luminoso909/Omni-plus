"""真实子进程导入来源与签名 checkout 一致，拒绝混用包或子模块。"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omni.training.r3_local import run_identity
from omni.training.run_source import CLI_FILES, check_source_root

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    target = tmp_path / "checkout"
    shutil.copytree(ROOT / "omni", target / "omni", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "scripts").mkdir()
    for name in CLI_FILES:
        shutil.copyfile(ROOT / name, target / name)
    return target


@pytest.mark.parametrize("script", CLI_FILES)
def test_cli_requires_actual_checkout_before_reading_assets(checkout: Path, script: str) -> None:
    # 模拟可编辑安装仍指向另一 checkout；调用真实 CLI，不替换任何函数。
    environment = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(checkout / script)]
    output = checkout / "must-not-exist"
    arguments = (
        ["plan", "--base-config", "absent.yaml", "--candidates", "absent.yaml"]
        if "bridge_ablation" in script
        else ["evaluate", "--config", "absent.yaml", "--resume", "absent.pt"]
    )
    arguments += ["--output-dir" if "bridge_ablation" in script else "--run-dir", str(output)]
    rejected = subprocess.run(
        command + arguments, cwd=checkout.parent, env=environment, capture_output=True, text=True
    )
    assert rejected.returncode != 0
    assert "执行源码与记录根目录不一致" in rejected.stderr
    assert "FileNotFoundError" not in rejected.stderr
    assert not output.exists()

    # 同根导入无需当前工作目录或 Git 元数据，也无需模型文件。
    accepted = subprocess.run(
        command + ["--help"],
        cwd=checkout.parent,
        env=environment | {"PYTHONPATH": str(checkout)},
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "usage:" in accepted.stdout


def test_run_identity_rejects_other_checkout_before_manifest_read(checkout: Path) -> None:
    check_source_root(ROOT)
    with pytest.raises(ValueError, match="执行源码与记录根目录不一致"):
        run_identity({}, checkout)


def test_loaded_submodule_and_package_search_paths_are_checked(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omni

    # 实际导入另一 checkout 的模块，主 omni 包仍来自正确目录。
    name = "omni.models.config"
    spec = importlib.util.spec_from_file_location(name, checkout / "omni/models/config.py")
    foreign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foreign)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, name, foreign)
        with pytest.raises(ValueError, match=name):
            check_source_root(ROOT)
    with monkeypatch.context() as patch:
        patch.setattr(omni, "__path__", [*omni.__path__, str(checkout / "omni")])
        with pytest.raises(ValueError, match="包搜索路径"):
            check_source_root(ROOT)
    check_source_root(ROOT)


@pytest.mark.parametrize("field", ("__file__", "__spec__"))
def test_missing_module_origin_is_not_accepted(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import omni

    monkeypatch.setattr(omni, field, None)
    with pytest.raises(ValueError, match="执行源码与记录根目录不一致"):
        check_source_root(ROOT)


def test_old_source_schema_cannot_certify_formal_matrix(tmp_path: Path) -> None:
    from scripts.bridge_ablation import collect

    matrix = {
        "schema_version": 2,
        "source_schema_version": 2,
        "source_sha256": {"omni/models/model_omni.py": "old hash"},
        "runtime": {"torch": "old", "transformers": "old"},
        "initialization_sha256": {"bridge_base_lineage": "old lineage"},
        "candidates": {
            name: {"config": {"model": {"kind": "qwen3"}}} for name in ("fixed", "weighted")
        },
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match="矩阵缺少完整源码"):
        collect(path, output)
    assert not output.exists()
