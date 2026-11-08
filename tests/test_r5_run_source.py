"""R5 源码身份覆盖实际执行路径，恢复和评测拒绝变化。"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_r4_training import settings

from omni.training.run_source import CLI_FILES, SOURCE_SCHEMA_VERSION, source_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    shutil.copytree(ROOT / "omni", root / "omni", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    for name in CLI_FILES:
        shutil.copyfile(ROOT / name, root / name)
    return root


def test_source_identity_tracks_code_and_is_independent_of_checkout_path(archive: Path) -> None:
    original = source_identity(archive)
    assert original["source_schema_version"] == SOURCE_SCHEMA_VERSION
    assert set(CLI_FILES) <= original["source_sha256"].keys()
    moved = archive.rename(archive.with_name("moved"))
    assert source_identity(moved) == original

    extra = moved / "omni" / "new_module.py"
    extra.write_text("enabled = True\n")
    assert "omni/new_module.py" in source_identity(moved)["source_sha256"]
    extra.unlink()
    assert source_identity(moved) == original

    script = moved / "scripts/run_experiment.py"
    script.write_text(script.read_text() + "\nprobe = True\n")
    assert source_identity(moved) != original
    script.unlink()
    with pytest.raises(FileNotFoundError, match="run_experiment.py"):
        source_identity(moved)


@pytest.mark.parametrize(
    "filename",
    ("omni/models/model_omni.py", "omni/data/omni_dataset.py", "omni/models/config.py"),
)
def test_source_change_rejects_resume_and_evaluate_before_model_load(
    archive: Path, tmp_path: Path, filename: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    config = settings(cache)
    config["model"]["config"] = str(ROOT / config["model"]["config"])
    config["model"]["bridge"] = {"kind": "fixed", "layers": [0]}
    config["training"].update(tasks=["t2a"], steps=2)
    config["evaluation"]["benchmark"] = {"split": "validation", "t2a_samples": 1}
    # 新进程从归档实际导入源码，静态指纹与运行来源同时受检。
    program = textwrap.dedent(
        r"""
        import json, sys
        from pathlib import Path
        import pytest
        import torch
        from omni.data import r3_prepare
        from omni.training import r3_local

        root, config, filename = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
        assert Path(r3_local.__file__).resolve().is_relative_to(root)
        original = r3_local.run_identity(config, root)
        checkpoint = root / "step-000001.pt"
        torch.save({"progress": {"identity": original, "step": 1}}, checkpoint)
        path = root / filename
        path.write_text(path.read_text() + "\nprobe = True\n")
        changed = r3_local.run_identity(config, root)
        assert changed["source_sha256"][filename] != original["source_sha256"][filename]

        r3_local.data_contract = lambda *args: {}
        r3_prepare.load_prepared = lambda *args: []
        r3_local.build_model = lambda *args: pytest.fail("加载了模型")
        resume, evaluation = root / "resume", root / "evaluation"
        with pytest.raises(ValueError, match="Resume 配置"):
            r3_local.smoke(config, root, resume, resume=checkpoint)
        with pytest.raises(ValueError, match="评估配置"):
            r3_local.evaluate_run(config, root, evaluation, checkpoint)
        assert not resume.exists() and not evaluation.exists()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(archive), json.dumps(config), filename],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(archive), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
