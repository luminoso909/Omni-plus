"""显式验收真实缓存的三任务反传，以及已训练 checkpoint 的恢复一致性。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from train_sft import run

from omni.data import OmniCollator
from omni.models import OmniPlusConfig, OmniPlusOmni
from omni.training.sft import train_step

ROOT = Path(__file__).resolve().parents[1]


def check_smoke(config_path: Path, device: str) -> dict:
    settings = yaml.safe_load(config_path.read_text())
    cache = torch.load(ROOT / settings["data"]["train"], weights_only=True)
    config = OmniPlusConfig(**(cache["config"] | settings["model"]))
    collate = OmniCollator(config)
    results = []
    for task in ("t2a", "a2a", "i2t"):
        # 各任务独立初始化，真实特征与监督来自同一官方记录
        index = next(i for i, row in enumerate(cache["selections"]) if row["task"] == task)
        sample = cache["samples"][index]
        batch = collate([sample])
        torch.manual_seed(42)
        model = OmniPlusOmni(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        losses = train_step(
            model, [batch], optimizer, torch.amp.GradScaler(device, enabled=False), device
        )
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert losses["optimizer_updated"] == 1
        results.append(
            dict(
                task=task,
                record_id=cache["selections"][index]["record_id"],
                input_shape=list(batch["input_ids"].shape),
                audio_shape=list(batch["audio_ids"].shape),
                has_audio_features=sample["audio_features"] is not None,
                has_image_features=sample["image_features"] is not None,
                text_supervised=int(sample["labels"].ne(-100).sum()),
                audio_supervised=int(sample["audio_labels"].ne(-100).sum()),
                losses=losses,
            )
        )
    return dict(
        status="passed",
        device=device,
        dtype="float32",
        tasks=results,
        scope="真实官方样本+冻结特征；小主干各任务一步更新，不是质量评测",
    )


def check_resume(original_report: Path) -> dict:
    original = json.loads(original_report.read_text())
    interval = original["settings"]["training"]["eval_every"]
    source_dir = Path(original["checkpoint"]).parent
    resumed_dir = run(
        Path(original["config_path"]), source_dir / f"step-{interval}.pt", steps=interval * 2
    )
    resumed_report = json.loads((resumed_dir / "report.json").read_text())
    expected = torch.load(
        source_dir / f"step-{interval * 2}.pt", map_location="cpu", weights_only=True
    )
    actual = torch.load(resumed_report["checkpoint"], map_location="cpu", weights_only=True)

    # MPS 比较预先固定 1e-4 容差，同时记录是否逐元素完全相等；不因失败放宽阈值
    differences = []
    exact = True
    for name, value in expected["model"].items():
        other = actual["model"][name]
        torch.testing.assert_close(other, value, rtol=1e-4, atol=1e-4)
        differences.append(float((other.float() - value.float()).abs().max()))
        exact &= torch.equal(other, value)
    assert expected["progress"] == actual["progress"]
    assert expected["scaler"] == actual["scaler"]
    return dict(
        status="passed",
        device=original["device"],
        from_step=interval,
        to_step=interval * 2,
        model_max_abs_diff=max(differences),
        model_bitwise_equal=exact,
        progress_equal=True,
        scaler_equal=True,
        rtol=1e-4,
        atol=1e-4,
        original_report=str(original_report),
        resumed_report=str(resumed_dir / "report.json"),
        scope="真实64条缓存的恢复分支与连续训练同一步权重对照；CPU完整状态精确对照另见单测",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/p4_overfit.yaml")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--resume-check", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = (
        check_resume(ROOT / args.resume_check)
        if args.resume_check
        else check_smoke(ROOT / args.config, args.device)
    )
    report["checked_at_utc"] = datetime.now(UTC).isoformat()
    name = "resume" if args.resume_check else "smoke"
    path = ROOT / "experiments/p4" / f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"P4 {name}: passed; {path}")


if __name__ == "__main__":
    main()
