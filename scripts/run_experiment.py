"""OmniPlus R3-L 薄入口：检查、有限数据准备、有限训练步骤验收。"""

import argparse
import json
from pathlib import Path

from omni.data.r3_prepare import inspect_data, load_prepared, prepare_data
from omni.training.r3_local import build_model, configure_parameters, read_settings, smoke

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="train/resume/evaluate/profile 待后续阶段实现；smoke --resume 可验收完整状态恢复。",
    )
    parser.add_argument("command", choices=["inspect", "prepare", "smoke"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="只验证配置、资产与数据预算，不准备或训练"
    )
    parser.add_argument("--run-dir", type=Path, help="smoke 的新结果目录，禁止覆盖")
    parser.add_argument("--resume", type=Path, help="仅用于 smoke，读取旧完整 checkpoint")
    parser.add_argument("--steps", type=int, help="smoke 目标总步数，含恢复步数且不超过配置预算")
    args = parser.parse_args()
    if args.command != "smoke" and any(
        value is not None for value in (args.resume, args.steps, args.run_dir)
    ):
        parser.error("--resume / --steps / --run-dir 仅用于 smoke")
    if args.command == "smoke" and not args.dry_run and args.run_dir is None:
        parser.error("smoke 需要显式 --run-dir，且目录尚不存在")

    settings = read_settings(ROOT / args.config, ROOT)
    contract = inspect_data(settings, ROOT)
    if args.steps is not None and not 0 < args.steps <= settings["training"]["steps"]:
        parser.error("--steps 超出配置中的有限训练预算")
    if args.resume is not None and not (ROOT / args.resume).is_file():
        raise FileNotFoundError(ROOT / args.resume)
    if args.run_dir is not None and (ROOT / args.run_dir).exists():
        raise FileExistsError(ROOT / args.run_dir)
    if args.command == "smoke" and args.dry_run:
        load_prepared(ROOT / settings["data"]["cache_dir"], "train", contract)

    if args.dry_run or args.command == "inspect":
        model = build_model(settings, ROOT, meta=True)
        result = dict(
            status="validated",
            dry_run=args.dry_run,
            execution="metadata_only_no_forward",
            model_kind=settings["model"]["kind"],
            parameter_groups=configure_parameters(model, settings["training"]["recipe"]),
            model_config=model.config.to_dict(),
            assets=contract,
            training=settings["training"],
            future_commands="train/resume/evaluate/profile 尚未实现",
        )
    elif args.command == "prepare":
        result = prepare_data(settings, ROOT)
    else:
        result = smoke(
            settings,
            ROOT,
            ROOT / args.run_dir,
            resume=ROOT / args.resume if args.resume else None,
            steps=args.steps,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
