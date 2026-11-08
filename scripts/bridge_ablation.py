"""生成受控 Bridge 配置和汇总实测记录。训练由 run_experiment.py 显式执行。"""

import argparse
import json
import re
from pathlib import Path

import yaml

from omni.evaluation.bridge_benchmark import asr_identity, benchmark_options, write_csv
from omni.training.r3_local import (
    adapter_identity,
    build_model,
    data_contract,
    digest,
    read_bridge_base,
    read_settings,
    run_identity,
)
from omni.training.run_source import SOURCE_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIELDS = ("source_schema_version", "source_sha256", "runtime")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def plan(base_path: Path, candidates_path: Path, output: Path, root: Path = ROOT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    base = read_settings(base_path, root)
    options = benchmark_options(base.get("evaluation", {}))
    if base["schema_version"] != 2 or options is None:
        raise ValueError("矩阵需要 schema_version 2 及显式 evaluation.benchmark。")
    if base["model"]["kind"] == "qwen3":
        initial = base["model"].get("initialization", {})
        if not initial.get("bridge_base_checkpoint") or not initial.get("bridge_base_manifest"):
            raise ValueError("原生 Qwen 矩阵需要同一 B1 Checkpoint 与训练 Manifest。")
    specs = yaml.safe_load(candidates_path.read_text())
    if not isinstance(specs, dict) or len(specs) < 2:
        raise ValueError("候选清单至少需要两个不同的 run ID。")

    candidates = {}
    for run_id, spec in specs.items():
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", run_id):
            raise ValueError(f"无效 run ID：{run_id}")
        if not isinstance(spec, dict) or spec.keys() != {"kind", "layers"}:
            raise ValueError(f"候选需要显式 kind 与 layers：{run_id}")
        settings = base | {"model": base["model"] | {"bridge": spec}}
        # 校验真实尺寸与层号；meta 构造不载入模型权重。
        build_model(settings, root, meta=True)
        run = output / "runs" / run_id
        candidates[run_id] = dict(
            config=settings,
            config_path=str(output / "configs" / f"{run_id}.yaml"),
            train=str(run / "train"),
            benchmark=str(run / "benchmark"),
        )
    if not any(spec["kind"] == "fixed" for spec in specs.values()):
        raise ValueError("矩阵需要 Fixed 控制组。")

    settings = next(iter(candidates.values()))["config"]
    contract = data_contract(settings, root)
    cache = root / settings["data"]["cache_dir"]
    manifest = json.loads((cache / "manifest.json").read_text())
    split = options["split"]
    split_file = cache / f"{split}.pt"
    if manifest["contract"] != contract or manifest["files"][split_file.name] != digest(split_file):
        raise ValueError("准备缓存或留出 split 的校验失败。")
    ids = [row["record_id"] for row in manifest["selections"][split] if row["task"] == "t2a"][
        : options["t2a_samples"]
    ]
    if len(set(ids)) != options["t2a_samples"]:
        raise ValueError("准备缓存缺少足够独立留出 T2A 样本。")

    identity = run_identity(settings, root)
    read_bridge_base(settings, root, contract, identity["initialization_sha256"])
    asr_dir = settings["evaluation"].get("asr_dir")
    matrix = dict(
        schema_version=2,
        scope="tiny_diagnostic" if base["model"]["kind"] == "tiny" else "planned_ablation",
        base_config=str(base_path),
        candidates_source=str(candidates_path),
        candidates=candidates,
        data_manifest_sha256=identity["data_manifest_sha256"],
        initialization_sha256=identity["initialization_sha256"],
        model_config_sha256=identity["model_config_sha256"],
        **{key: identity[key] for key in SOURCE_FIELDS},
        asset_identity=adapter_identity(settings, contract),
        asr_assets=asr_identity(root / asr_dir if asr_dir else None),
        split=split,
        sample_ids=ids,
    )
    (output / "configs").mkdir(parents=True, exist_ok=False)
    for candidate in candidates.values():
        with Path(candidate["config_path"]).open("x", encoding="utf-8") as stream:
            yaml.safe_dump(candidate["config"], stream, allow_unicode=True, sort_keys=False)
    write_json(output / "matrix.json", matrix)
    return matrix


def completed_row(matrix: dict, entry: dict) -> dict:
    train = json.loads((Path(entry["train"]) / "manifest.json").read_text())
    evaluation = json.loads((Path(entry["benchmark"]) / "evaluation.json").read_text())
    benchmark = evaluation["benchmark"]
    budget = entry["config"]["training"]["steps"]
    source_fields = SOURCE_FIELDS if matrix["schema_version"] == 2 else ("source_sha256",)
    for document in (train, evaluation):
        if document["config"] != entry["config"]:
            raise ValueError("实测配置与矩阵不一致。")
        for field in (
            "asset_identity",
            "data_manifest_sha256",
            "initialization_sha256",
            "model_config_sha256",
            *source_fields,
        ):
            if document[field] != matrix[field]:
                raise ValueError(f"实测版本与矩阵不一致：{field}")
    if train["status"] != "passed" or train["step"] != budget:
        raise ValueError("训练未达到矩阵的共同更新步数。")
    if (
        evaluation["checkpoint_sha256"] != train["checkpoint_sha256"]
        or evaluation["checkpoint_step"] != budget
    ):
        raise ValueError("Benchmark 必须使用对应训练的终点 Checkpoint。")
    if evaluation["split"] != matrix["split"] or benchmark["sample_ids"] != matrix["sample_ids"]:
        raise ValueError("Benchmark split 或样本身份与矩阵不一致。")
    if [sample["record_id"] for sample in benchmark["samples"]] != matrix["sample_ids"]:
        raise ValueError("逐样本记录不完整或顺序有误。")
    failures = [
        {key: sample.get(key) for key in ("record_id", "failure", "error")}
        for sample in benchmark["samples"]
        if sample["status"] != "generated"
    ]
    if len(failures) != benchmark["failed_samples"]:
        raise ValueError("Benchmark 失败计数与逐样本记录不一致。")
    if evaluation["hardware"] != train["hardware"]:
        raise ValueError("训练与 Benchmark 的硬件或运行环境不一致。")
    if evaluation["asr_assets"] != matrix["asr_assets"]:
        raise ValueError("Benchmark ASR 资产与矩阵不一致。")
    if benchmark["protocol"]["max_new_tokens"] != entry["config"]["evaluation"].get(
        "max_new_tokens", 128
    ):
        raise ValueError("Benchmark 生成预算与矩阵不一致。")

    start_step = train["start_step"]
    if type(start_step) is not int or not 0 <= start_step < budget:
        raise ValueError("训练记录的本段更新区间无效。")
    cuda = train.get("cuda", {})
    counts = train["parameter_groups"]
    return dict(
        status="failed" if failures else "passed",
        reason="有生成失败样本" if failures else None,
        failures=failures,
        steps=train["step"],
        training_start_step=start_step,
        training_end_step=train["step"],
        training_update_count=train["step"] - start_step,
        training_loop_seconds=train["loop_seconds"],
        training_seconds_per_step=train["seconds_per_step"],
        training_timing_scope=train["timing_scope"],
        training_peak_allocated_bytes=cuda.get("peak_allocated_bytes"),
        training_peak_reserved_bytes=cuda.get("peak_reserved_bytes"),
        inference_median_stream_seconds=benchmark["median_stream_seconds"],
        inference_timing_scope=benchmark["protocol"]["timing"],
        inference_peak_allocated_bytes=benchmark["cuda_peak_allocated_bytes"],
        trainable_parameters=sum(group["trainable"] for group in counts.values()),
        bridge_parameters=counts["bridge"]["total"],
        validation_loss=train["validation"]["total"],
        heldout_loss=evaluation["metrics"]["total"],
        gradient_groups=train["gradient_groups"],
        evaluations=train["evaluations"],
        checkpoint_sha256=train["checkpoint_sha256"],
        hardware=evaluation["hardware"],
        asr_assets=evaluation["asr_assets"],
        protocol=benchmark["protocol"],
        **{
            key: benchmark[key]
            for key in (
                "failed_samples",
                "truncated_samples",
                "asr_measured_samples",
                "cer",
                "wer",
            )
        },
    )


def collect(matrix_path: Path, output: Path, overrides: dict | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    matrix = json.loads(matrix_path.read_text())
    candidates = matrix["candidates"]
    version = matrix["schema_version"]
    if version not in (1, 2) or len(candidates) < 2:
        raise ValueError("不支持的矩阵版本或候选数量。")
    native = any(entry["config"]["model"]["kind"] == "qwen3" for entry in candidates.values())
    if version == 1 and native:
        raise ValueError("历史矩阵缺少完整源码及 B1 谱系身份，原生 Qwen 需要重新规划。")
    if version == 2 and (
        matrix.get("source_schema_version") != SOURCE_SCHEMA_VERSION
        or not matrix.get("source_sha256")
        or not matrix.get("runtime")
        or (native and not matrix.get("initialization_sha256", {}).get("bridge_base_lineage"))
    ):
        raise ValueError("矩阵缺少完整源码、运行时或 B1 谱系身份。")
    shared = []
    for entry in candidates.values():
        config = entry["config"]
        shared.append(
            config | {"model": {k: v for k, v in config["model"].items() if k != "bridge"}}
        )
    if any(config != shared[0] for config in shared[1:]):
        raise ValueError("矩阵候选之间只能更改 Bridge。")
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or overrides.keys() - candidates.keys():
        raise ValueError("只可为矩阵内已有 run ID 指定结果路径。")

    rows, used_paths = [], set()
    for run_id, planned in candidates.items():
        paths = overrides.get(run_id, {})
        if not isinstance(paths, dict) or paths.keys() - {"train", "benchmark", "failure_log"}:
            raise ValueError("结果覆盖只接受 train、benchmark 和 failure_log。")
        entry = planned | paths
        for key in ("train", "benchmark"):
            path = Path(entry[key]).resolve()
            if path in used_paths:
                raise ValueError("多个实验不能共用同一结果目录。")
            used_paths.add(path)
        try:
            if entry.get("failure_log"):
                log = Path(entry["failure_log"])
                row = dict(status="failed", reason="执行失败日志", failure_log_sha256=digest(log))
            else:
                row = completed_row(matrix, entry)
        except FileNotFoundError as error:
            row = dict(status="missing", reason=str(error))
        except (ValueError, KeyError, TypeError) as error:
            row = dict(status="invalid", reason=str(error))
        rows.append(
            dict(
                run_id=run_id,
                bridge=entry["config"]["model"]["bridge"],
                train_dir=entry["train"],
                benchmark_dir=entry["benchmark"],
                failure_log=entry.get("failure_log"),
                **row,
            )
        )

    common = all(row["status"] == "passed" for row in rows)
    comparable = (
        version == 2
        and common
        and all(
            row[key] == rows[0][key]
            for row in rows
            for key in ("hardware", "asr_assets", "protocol")
        )
    )
    result = dict(
        scope=matrix["scope"],
        matrix=str(matrix_path),
        matrix_sha256=digest(matrix_path),
        comparable=comparable,
        fairness_failure=(
            None
            if comparable
            else "历史 Tiny 矩阵仅供诊断，缺少完整源码身份"
            if version == 1
            else "硬件、ASR 或协议不一致"
            if common
            else "有失败、无效或缺失的实验"
        ),
        runs=rows,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    fields = (
        "run_id",
        "bridge",
        "status",
        "reason",
        "steps",
        "trainable_parameters",
        "bridge_parameters",
        "validation_loss",
        "heldout_loss",
        "training_start_step",
        "training_end_step",
        "training_update_count",
        "training_loop_seconds",
        "training_seconds_per_step",
        "training_timing_scope",
        "training_peak_allocated_bytes",
        "training_peak_reserved_bytes",
        "inference_median_stream_seconds",
        "inference_timing_scope",
        "inference_peak_allocated_bytes",
        "failed_samples",
        "truncated_samples",
        "asr_measured_samples",
        "cer",
        "wer",
        "train_dir",
        "benchmark_dir",
        "failure_log",
    )
    write_csv(output / "summary.csv", rows, fields)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan", help="锁定共同配置并生成独立候选 YAML")
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    summary = commands.add_parser("collect", help="校验并汇总已存在的实际运行记录")
    summary.add_argument("--matrix", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--artifacts", type=Path, help="可选的 resume 路径及失败日志映射 JSON")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.base_config, args.candidates, args.output_dir.resolve())
    else:
        overrides = json.loads(args.artifacts.read_text()) if args.artifacts else None
        result = collect(args.matrix, args.output_dir, overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
