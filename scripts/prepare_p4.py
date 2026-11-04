"""显式准备本地 P4 小样本；源数据缺失时停止，不下载或复用评测媒体凑数。"""

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from omni.data.prepare import cache_split, prepare_splits
from omni.models import OmniPlusConfig

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/p4_data.yaml"))
    parser.add_argument("--cache-features", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load((ROOT / args.config).read_text())
    output_dir = ROOT / settings["output_dir"]
    report_path = ROOT / settings["report"]
    if report_path.exists() and not args.cache_features:
        parser.error("报告已存在，请在配置中使用新的输出路径保留旧记录。")

    # 数据先划分，模型只在显式要求缓存时加载
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = prepare_splits(
            {task: ROOT / path for task, path in settings["sources"].items()},
            output_dir,
            settings["train_size"],
            settings["validation_size"],
            settings["seed"],
            settings["pool_per_task"],
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.cache_features:
        from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder

        report_path = ROOT / settings["feature_report"]
        if report_path.exists():
            parser.error("特征报告已存在；请使用新路径保留旧记录。")

        reference = ROOT / settings["reference_root"]
        audio_encoder = SenseVoiceEncoder(reference / "model/SenseVoiceSmall")
        vision_encoder = SiglipEncoder(reference / "model/siglip2-base-p32-256-ve")
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / settings["tokenizer"], local_files_only=True
        )
        config = OmniPlusConfig(pad_token_id=tokenizer.pad_token_id)
        report["cache"] = {
            split: cache_split(
                Path(path),
                ROOT / settings["feature_dir"] / f"{split}_features.pt",
                tokenizer,
                config,
                audio_encoder,
                vision_encoder,
                settings["max_length"],
            )
            for split, path in report["paths"].items()
        }
        report["feature_paths"] = {
            split: str(ROOT / settings["feature_dir"] / f"{split}_features.pt")
            for split in report["paths"]
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"P4 本地数据准备完成：{report_path}")


if __name__ == "__main__":
    main()
