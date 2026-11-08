"""绑定训练、恢复及评测实际使用的项目源码。"""

import hashlib
from pathlib import Path

SOURCE_SCHEMA_VERSION = 2
CLI_FILES = ("scripts/run_experiment.py", "scripts/bridge_ablation.py")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _raise_scan_error(error: OSError) -> None:
    raise error


def source_identity(root: Path) -> dict:
    """按相对路径和实际文件字节固定执行源码，支持无 Git 元数据的归档。"""
    root = root.resolve()
    package = root / "omni"
    if package.is_symlink():
        raise ValueError(f"源码目录不能是符号链接：{package}")
    if not package.is_dir():
        raise FileNotFoundError(package)

    paths = {root / name for name in CLI_FILES}
    for directory, _, files in package.walk(on_error=_raise_scan_error):
        for name in files:
            path = directory / name
            if path.is_symlink() and path.is_dir():
                raise ValueError(f"源码目录不能是符号链接：{path}")
            if path.suffix == ".py":
                paths.add(path)

    hashes = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"源码路径超出仓库：{path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[path.relative_to(root).as_posix()] = digest(path)
    return dict(source_schema_version=SOURCE_SCHEMA_VERSION, source_sha256=hashes)
