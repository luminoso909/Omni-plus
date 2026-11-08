"""绑定训练、恢复及评测实际使用的项目源码。"""

import hashlib
import sys
from pathlib import Path

SOURCE_SCHEMA_VERSION = 3
CLI_FILES = ("scripts/run_experiment.py", "scripts/bridge_ablation.py")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _raise_scan_error(error: OSError) -> None:
    raise error


def check_source_root(root: Path) -> None:
    """拒绝从其他 checkout 导入源码，包括包搜索路径及已加载子模块。"""
    root = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "omni" and not name.startswith("omni."):
            continue
        location = root.joinpath(*name.split("."))
        package_paths = getattr(module, "__path__", None)
        expected = (
            location / "__init__.py" if package_paths is not None else location.with_suffix(".py")
        )
        origins = (
            getattr(module, "__file__", None),
            getattr(getattr(module, "__spec__", None), "origin", None),
        )
        if any(not isinstance(path, str) or Path(path).resolve() != expected for path in origins):
            raise ValueError(f"执行源码与记录根目录不一致：{name}，需要 {expected}，实际 {origins}")
        if package_paths is not None and [Path(path).resolve() for path in package_paths] != [
            location
        ]:
            raise ValueError(f"执行源码包搜索路径不一致：{name}，需要 {location}")


def source_identity(root: Path) -> dict:
    """静态摘要可用于归档比较；执行入口先用 check_source_root 核对实际导入。"""
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
