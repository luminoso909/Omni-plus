"""本地小样本划分与冻结特征缓存；不下载、不扩增样本数。"""

import hashlib
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch import Tensor

from omni.models.config import OmniPlusConfig

from .omni_dataset import OmniDataset

TASKS = ("t2a", "a2a", "i2t")


def content_hash(value: Any) -> str:
    # 媒体只保留内容摘要参与记录哈希，不把大段 bytes 转成 JSON 文本
    def normalize(item: Any) -> Any:
        if isinstance(item, bytes):
            return {"bytes_sha256": hashlib.sha256(item).hexdigest()}
        if isinstance(item, Mapping):
            return {key: normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item

    text = json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def media_hashes(record: dict) -> set[str]:
    # 同一张图或相同输入/输出音频不能分别落入训练集与验证集；参考音色不作分组键
    groups = set()
    for key in ("image_bytes", "question_audios", "answer_audios"):
        values = record.get(key)
        if not values:
            continue
        values = [values] if isinstance(values, (bytes, str)) else values
        if key == "image_bytes":
            values = [
                item for turn in values for item in (turn if isinstance(turn, list) else [turn])
            ]
        elif key == "answer_audios" and isinstance(values[0], (int, float)):
            values = [values]
        for value in values:
            if value:
                groups.add(content_hash({"kind": key, "value": value}))
    return groups


def prepare_splits(
    sources: Mapping[str, str | Path],
    output_dir: Path,
    train_size: int = 64,
    validation_size: int = 8,
    seed: int = 42,
    pool_per_task: int = 128,
) -> dict:
    """扫描各任务开头的固定候选池，均衡划分真实独立记录并保留原始行号。"""
    if (
        set(sources) != set(TASKS)
        or min(train_size, validation_size) < len(TASKS)
        or pool_per_task < 1
    ):
        raise ValueError("需要 t2a/a2a/i2t 三种本地源，且每个 split 至少覆盖三种任务。")

    paths = {task: Path(sources[task]).resolve() for task in TASKS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少本地真实数据，不自动下载：" + ", ".join(missing))
    if any((output_dir / f"{split}.pt").exists() for split in ("train", "validation")):
        raise FileExistsError("目标 split 已存在；使用新目录保留旧实验。")

    # 固定候选池与种子，避免读取全部大型 Parquet；不足时明确失败，不重复凑数
    rng = random.Random(seed)
    records = {split: [] for split in ("train", "validation")}
    selected = {split: [] for split in records}
    seen_records, seen_media = set(), set()
    source_info = []
    for task_index, (task, path) in enumerate(paths.items()):
        parquet = pq.ParquetFile(path)
        pool = []
        for batch in parquet.iter_batches(batch_size=min(pool_per_task, 32)):
            pool.extend(batch.to_pylist()[: pool_per_task - len(pool)])
            if len(pool) >= pool_per_task:
                break
        candidates = list(enumerate(pool))
        rng.shuffle(candidates)
        quotas = {
            split: size // 3 + int(task_index < size % 3)
            for split, size in (("train", train_size), ("validation", validation_size))
        }
        for split, quota in quotas.items():
            count = 0
            while candidates and count < quota:
                row_index, record = candidates.pop()
                required = (
                    ("image_bytes",)
                    if task == "i2t"
                    else ("answer_audios", "question_audios")
                    if task == "a2a"
                    else ("answer_audios",)
                )
                if not record.get("conversations") or any(not record.get(key) for key in required):
                    raise ValueError(
                        f"{task} 第 {row_index} 行缺少对话或必需模态，不能作为该任务样本。"
                    )

                identity = content_hash(
                    {key: value for key, value in record.items() if not key.startswith("_source_")}
                )
                media = media_hashes(record)
                if identity in seen_records or media & seen_media:
                    continue
                seen_records.add(identity)
                seen_media.update(media)
                records[split].append(record)
                selected[split].append(
                    dict(
                        task=task,
                        source_row=record.get("_source_row", row_index),
                        source_file=record.get("_source_file", path.name),
                        subset_row=row_index,
                        record_id=identity,
                    )
                )
                count += 1
            if count != quota:
                raise ValueError(
                    f"{task} 候选池中独立样本不足：{split} 需要 {quota} 条，得到 {count} 条。"
                )

        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source_info.append(
            dict(task=task, path=str(path), sha256=digest, rows=parquet.metadata.num_rows)
        )

    # 原始记录仅存入被忽略的本地缓存，公开 manifest 不包含文本、媒体或完整波形
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in records.items():
        torch.save(dict(records=values, selections=selected[split]), output_dir / f"{split}.pt")
    return dict(
        seed=seed,
        pool_per_task=pool_per_task,
        sources=source_info,
        splits=selected,
        paths={split: str(output_dir / f"{split}.pt") for split in records},
        overlap=0,
        augmentation=False,
        license_note="仅本地诊断使用；各数据集及其上游媒体许可需分别核实，不沿用代码许可证。",
    )


def cache_split(
    split_path: Path,
    output_path: Path,
    tokenizer: Any,
    config: OmniPlusConfig,
    audio_encoder: Callable[[Any], Tensor] | None = None,
    vision_encoder: Callable[[Any], Tensor] | None = None,
    max_length: int = 1200,
) -> dict:
    """冻结媒体特征与最后一轮监督；缓存样本可直接交给 OmniCollator。"""
    if output_path.exists():
        raise FileExistsError("目标特征缓存已存在；使用新文件保留旧实验。")

    source = torch.load(split_path, map_location="cpu", weights_only=True)
    dataset = OmniDataset(
        source["records"],
        tokenizer,
        config,
        max_length=max_length,
        training=False,
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    statistics = dict(
        samples=len(samples),
        text_tokens=sum(int(sample["labels"].ne(-100).sum()) for sample in samples),
        audio_tokens=sum(int(sample["audio_labels"].ne(-100).sum()) for sample in samples),
        audio_samples=sum(sample["audio_features"] is not None for sample in samples),
        image_samples=sum(sample["image_features"] is not None for sample in samples),
        max_tokens=max(sample["input_ids"].numel() for sample in samples),
        augmentation=False,
        labels_shifted_once=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            samples=samples,
            selections=source["selections"],
            config=config.to_dict(),
            max_length=max_length,
            statistics=statistics,
        ),
        output_path,
    )
    return statistics
