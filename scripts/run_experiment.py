"""OmniPlus 统一薄入口：有限数据准备、SFT、恢复、验证与生成。"""

import argparse
import json
from pathlib import Path

from omni.training.r3_local import (
    build_model,
    configure_parameters,
    data_contract,
    evaluate_run,
    inspect_sft_run,
    prepare,
    read_settings,
    smoke,
)
from omni.training.run_source import check_source_root

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    check_source_root(ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "prepare",
            "smoke",
            "train",
            "resume",
            "evaluate",
            "demo",
            "profile",
            "grpo-smoke",
            "grpo-train",
            "grpo-resume",
            "grpo-evaluate",
            "grpo-profile",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="只检查配置、资产、缓存及预算，不执行 forward"
    )
    parser.add_argument("--run-dir", type=Path, help="全新结果目录；历史运行目录不能覆盖")
    parser.add_argument("--resume", type=Path, help="恢复或评估使用的完整 Checkpoint")
    parser.add_argument("--baseline", action="store_true", help="GRPO 评测配置指定的初始策略")
    parser.add_argument("--steps", type=int, help="目标总 optimizer steps，含已恢复步数")
    args = parser.parse_args()
    if args.command.startswith("grpo-"):
        from omni.training.r6_local import inspect_run, read_grpo_settings, run_grpo

        settings = read_grpo_settings(ROOT / args.config, ROOT)
        if not args.dry_run and args.run_dir is None:
            parser.error("GRPO 执行需要新的 --run-dir")
        options = dict(
            resume=ROOT / args.resume if args.resume else None,
            steps=args.steps,
            command=args.command,
            baseline=args.baseline,
        )
        result = (
            inspect_run(
                settings, ROOT, run_dir=ROOT / args.run_dir if args.run_dir else None, **options
            )
            if args.dry_run
            else run_grpo(settings, ROOT, ROOT / args.run_dir, **options)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.baseline:
        parser.error("--baseline 仅用于 grpo-evaluate")

    training_commands = {"smoke", "train", "resume", "profile"}
    if args.command in {"inspect", "prepare"} and any(
        value is not None for value in (args.resume, args.steps, args.run_dir)
    ):
        parser.error("inspect / prepare 不使用 --resume / --steps / --run-dir")
    if args.command not in training_commands and args.steps is not None:
        parser.error("--steps 仅用于训练或资源 Profile")
    if args.command in {"resume", "evaluate", "demo"} and args.resume is None:
        parser.error("resume / evaluate / demo 需要 --resume CHECKPOINT")
    if args.command not in {"inspect", "prepare"} and not args.dry_run and args.run_dir is None:
        parser.error("执行需要显式 --run-dir，且目录尚不存在")

    settings = read_settings(ROOT / args.config, ROOT)
    contract = data_contract(settings, ROOT)
    target = args.steps if args.steps is not None else settings["training"]["steps"]
    if not 0 < target <= settings["training"]["steps"]:
        parser.error("--steps 超出配置中的预算")
    if (args.command == "smoke" and target > 10) or (args.command == "profile" and target > 100):
        parser.error("smoke 最多 10 步，profile 最多 100 步")
    if args.resume is not None and not (ROOT / args.resume).is_file():
        raise FileNotFoundError(ROOT / args.resume)
    if args.run_dir is not None and (ROOT / args.run_dir).exists():
        raise FileExistsError(ROOT / args.run_dir)
    if args.dry_run or args.command == "inspect":
        model = build_model(settings, ROOT, meta=True)
        preflight = None
        if args.command not in {"inspect", "prepare"}:
            preflight = inspect_sft_run(
                model,
                settings,
                ROOT,
                contract,
                command=args.command,
                resume=ROOT / args.resume if args.resume else None,
                target=target,
            )
        result = dict(
            status="validated",
            dry_run=args.dry_run,
            execution="preflight_no_forward"
            if preflight is not None
            else "metadata_only_no_forward",
            model_kind=settings["model"]["kind"],
            parameter_groups=configure_parameters(model, settings["training"]["recipe"]),
            model_config=model.config.to_dict(),
            assets=contract,
            training=settings["training"],
            preflight=preflight,
            commands=[
                "inspect",
                "prepare",
                "smoke",
                "train",
                "resume",
                "evaluate",
                "demo",
                "profile",
            ],
        )
    elif args.command == "prepare":
        result = prepare(settings, ROOT)
    elif args.command in {"evaluate", "demo"}:
        result = evaluate_run(
            settings, ROOT, ROOT / args.run_dir, ROOT / args.resume, demo=args.command == "demo"
        )
    else:
        result = smoke(
            settings,
            ROOT,
            ROOT / args.run_dir,
            resume=ROOT / args.resume if args.resume else None,
            steps=args.steps,
            command=args.command,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
