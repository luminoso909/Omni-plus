"""显式启动本地小样本 SFT；读取冻结特征缓存，不下载媒体或模型。"""

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from omni.data import OmniCollator
from omni.models import OmniPlusConfig, OmniPlusOmni
from omni.training.sft import (
    configure_trainable,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_step,
)

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(config_path: Path, resume: Path | None = None, steps: int | None = None) -> Path:
    started = time.perf_counter()
    settings = yaml.safe_load(config_path.read_text())
    training = settings["training"]
    device = training["device"]
    if any(training[name] <= 0 for name in ("batch_size", "accumulation_steps", "eval_every")):
        raise ValueError("batch_size、accumulation_steps、eval_every 必须为正数。")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用，请显式将 training.device 改为 cpu。")

    # 缓存缺失就停止，不以随机数据替代真实验收
    paths = {key: ROOT / value for key, value in settings["data"].items()}
    data = {
        key: torch.load(path, map_location="cpu", weights_only=True) for key, path in paths.items()
    }
    fingerprints = {key: file_hash(path) for key, path in paths.items()}
    train, validation = data["train"]["samples"], data["validation"]["samples"]
    train_ids = {row["record_id"] for row in data["train"]["selections"]}
    validation_ids = {row["record_id"] for row in data["validation"]["selections"]}
    if (
        not train
        or not validation
        or train_ids & validation_ids
        or len(train_ids) != len(train)
        or len(validation_ids) != len(validation)
        or len(data["train"]["selections"]) != len(train)
        or len(data["validation"]["selections"]) != len(validation)
    ):
        raise ValueError("需要非空且互不重叠的训练与验证缓存。")

    # 模型配置可在缓存的媒体接口上覆盖主干尺寸；tiny 过拟合与全尺寸 SFT 分开记录
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    config = OmniPlusConfig(**(data["train"]["config"] | settings["model"]))
    interface = (
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "think_end_ids",
        "audio_ids",
        "image_ids",
        "audio_special_token",
        "image_special_token",
        "num_audio_codebooks",
        "audio_codebook_size",
        "audio_vocab_size",
        "audio_pad_token",
        "audio_stop_token",
        "audio_spk_token",
        "spk_emb_size",
        "audio_hidden_size",
        "image_hidden_size",
        "image_token_len",
    )
    for name in interface:
        values = [part["config"][name] for part in data.values()]
        if any(value != getattr(config, name) for value in values):
            raise ValueError(f"训练/验证缓存与模型的媒体或标签接口不一致：{name}")

    model = OmniPlusOmni(config).to(device)
    if not resume and settings.get("init_weights"):
        state = torch.load(ROOT / settings["init_weights"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    trainable = configure_trainable(model, training["freeze"])
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    dtype = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        training["dtype"]
    ]
    scaler = torch.amp.GradScaler(torch.device(device).type, enabled=dtype == torch.float16)
    collate = OmniCollator(config)
    batch_size = training["batch_size"]
    train_eval = [collate(train[i : i + batch_size]) for i in range(0, len(train), batch_size)]
    val_eval = [
        collate(validation[i : i + batch_size]) for i in range(0, len(validation), batch_size)
    ]

    # 每次新建 run；恢复只读取旧 checkpoint，日志和新权重写到新的 run
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "outputs/p4" / run_id
    report_dir = ROOT / "experiments/p4" / run_id
    output.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    progress = dict(
        step=0, updates=0, epoch=0, cursor=0, data_hashes=fingerprints, training=training
    )
    if resume:
        progress = load_checkpoint(resume, model, optimizer, scaler)
        if progress["data_hashes"] != fingerprints or progress["training"] != training:
            raise ValueError("恢复时缓存或训练配置变化，不能沿用旧采样位置和优化器。")

    stop_weight = training["stop_weight"]
    initial = evaluate(model, train_eval, device, stop_weight)
    progress.setdefault("initial_train", initial)
    history = [dict(step=progress["step"], split="train", **initial)]
    max_steps = steps if steps is not None else training["steps"]
    if max_steps <= progress["step"]:
        raise ValueError("目标步数必须大于已恢复步数。")
    while progress["step"] < max_steps:
        # 独立 shuffle Generator，由 epoch 决定，不消耗 dropout 的全局 RNG
        batches = []
        for _ in range(training["accumulation_steps"]):
            if progress["cursor"] >= len(train):
                progress["epoch"] += 1
                progress["cursor"] = 0
            generator = torch.Generator().manual_seed(seed + progress["epoch"])
            order = torch.randperm(len(train), generator=generator).tolist()
            start = progress["cursor"]
            selected = order[start : start + batch_size]
            batches.append(collate([train[index] for index in selected]))
            progress["cursor"] += len(selected)
        losses = train_step(
            model,
            batches,
            optimizer,
            scaler,
            device,
            dtype,
            stop_weight,
            grad_clip=training["grad_clip"],
        )
        progress["step"] += 1
        progress["updates"] += int(losses["optimizer_updated"])
        step = progress["step"]
        history.append(dict(step=step, split="update", **losses))
        if step % training["eval_every"] == 0 or step == max_steps:
            metrics = evaluate(model, train_eval, device, stop_weight)
            val_metrics = evaluate(model, val_eval, device, stop_weight)
            history.extend(
                [
                    dict(step=step, split="train", **metrics),
                    dict(step=step, split="validation", **val_metrics),
                ]
            )
            save_checkpoint(output / f"step-{step}.pt", model, optimizer, scaler, progress.copy())
            print(
                f"step={step} train={metrics['total']:.4f} validation={val_metrics['total']:.4f}",
                flush=True,
            )

    # 只从实际日志绘制曲线；无正式大规模训练或模型质量结论
    final = evaluate(model, train_eval, device, stop_weight)
    fields = ("step", "split", "total", "text", "audio", "aux", "grad_norm", "optimizer_updated")
    with (report_dir / "loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    for split in ("train", "validation"):
        rows = [row for row in history if row["split"] == split]
        axis.plot([row["step"] for row in rows], [row["total"] for row in rows], label=split)
    axis.set(xlabel="Update attempt", ylabel="Masked SFT loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "loss.png", dpi=160)
    plt.close(figure)
    report = dict(
        run_id=run_id,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[1])): file_hash(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "omni/training/losses.py",
                Path(__file__).resolve().parents[1] / "omni/training/sft.py",
            )
        },
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        timestamp_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        config_path=str(config_path),
        settings=settings,
        model_config=config.to_dict(),
        train_samples=len(train),
        validation_samples=len(validation),
        data_hashes=fingerprints,
        train_tasks={
            task: sum(row["task"] == task for row in data["train"]["selections"])
            for task in ("t2a", "a2a", "i2t")
        },
        trainable_parameters=trainable,
        device=device,
        dtype=training["dtype"],
        seed=seed,
        initial_train=progress["initial_train"],
        segment_initial_train=initial,
        final_train=final,
        final_validation=val_metrics,
        optimizer_steps=progress["updates"],
        resume_from=str(resume) if resume else None,
        progress=progress,
        loss_ratio=final["total"] / max(progress["initial_train"]["total"], 1e-12),
        overfit_loss_halved=final["total"] < progress["initial_train"]["total"] * 0.5,
        checkpoint=str(output / f"step-{progress['step']}.pt"),
        checkpoint_sha256=file_hash(output / f"step-{progress['step']}.pt"),
        scope="本地小样本工程过拟合；不等于全尺寸正式SFT或模型质量评测",
    )
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"P4 运行记录：{report_dir}")
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/p4_overfit.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, help="目标总更新尝试数（含已有步数；FP16溢出跳步另计）"
    )
    args = parser.parse_args()
    run(ROOT / args.config, ROOT / args.resume if args.resume else None, args.steps)


if __name__ == "__main__":
    main()
