"""R3 媒体组、可重复划分及不可混用的缓存契约。"""

import json

import pytest
import torch

from omni.data.prepare import content_hash
from omni.data.r3_prepare import (
    file_hash,
    group_records,
    inspect_data,
    load_prepared,
    select_records,
)


def test_connected_media_groups_preserve_historical_split() -> None:
    records = [
        dict(record_id="a", media_ids=["audio", "image"]),
        dict(record_id="b", media_ids=["image", "other"]),
        dict(record_id="c", media_ids=["other"]),
        dict(record_id="d", media_ids=["unused"]),
    ]
    rows = group_records(records, {"a": "train"})
    assert len({row["group_id"] for row in rows[:3]}) == 1
    assert [row["split"] for row in rows] == ["train", "train", "train", "test"]
    with pytest.raises(ValueError, match="不同历史 split"):
        group_records(records, {"a": "train", "c": "validation"})


def test_selection_is_seeded_disjoint_and_does_not_repeat_media() -> None:
    rows = [
        dict(record_id=str(i), group_id=str(i), split=split)
        for i, split in enumerate(["train"] * 12 + ["validation"] * 3 + ["test"] * 3)
    ]
    selected = select_records(rows, count=8, seed=42)
    assert selected == select_records(rows[::-1], count=8, seed=42)
    assert selected != select_records(rows, count=8, seed=43)
    assert [row["split"] for row in selected] == ["train"] * 6 + ["validation", "test"]
    assert len({row["group_id"] for row in selected}) == 8
    with pytest.raises(ValueError, match="独立媒体不足"):
        select_records([dict(row, group_id="same") for row in rows], count=8, seed=42)


def test_cache_rejects_version_and_tampering(tmp_path) -> None:
    contract = {"processor": "version1", "tokenizer": "qwen3", "code_policy": "raw_reencode"}
    path = tmp_path / "train.pt"
    torch.save(
        dict(
            samples=[{"input_ids": torch.tensor([151644, 42])}], contract_id=content_hash(contract)
        ),
        path,
    )
    manifest = dict(contract=contract, files={path.name: file_hash(path)})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_prepared(tmp_path, "train", contract)
    assert loaded[0]["input_ids"].tolist() == [151644, 42]
    with pytest.raises(ValueError, match="版本不匹配"):
        load_prepared(tmp_path, "train", dict(contract, processor="version2"))
    with pytest.raises(ValueError, match="版本不匹配"):
        load_prepared(tmp_path, "train", dict(contract, tokenizer="minimind"))
    torch.save(dict(samples=[], contract_id="wrong"), path)
    with pytest.raises(ValueError, match="缓存校验失败"):
        load_prepared(tmp_path, "train", contract)
    manifest["files"][path.name] = file_hash(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="摘要不一致"):
        load_prepared(tmp_path, "train", contract)


def test_dry_run_rejects_unbounded_tasks_before_any_asset_loading(tmp_path) -> None:
    settings = dict(
        data=dict(tasks=dict(t2a=1000, a2t=8, a2a=8, i2t=8), seed=42, max_length=1200), assets={}
    )
    with pytest.raises(ValueError, match="本地预算"):
        inspect_data(settings, tmp_path)
    settings["data"]["tasks"]["t2a"] = 8
    settings["assets"] = dict(qwen_dir="missing", reference_root="missing")
    with pytest.raises(FileNotFoundError):
        inspect_data(settings, tmp_path)


def test_derived_views_keep_original_media_group() -> None:
    records = [
        dict(record_id=task, media_ids=["same_original_audio"]) for task in ("t2a", "a2t", "a2a")
    ]
    grouped = group_records(records, {"a2a": "validation"})
    assert {row["split"] for row in grouped} == {"validation"}
    assert len({row["group_id"] for row in grouped}) == 1
