"""P4 数据划分与特征缓存边界；全部使用临时记录，不读取真实大资产。"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_dataset import TinyTokenizer, conversation

from omni.data import OmniCollator
from omni.data.prepare import cache_split, content_hash, media_hashes, prepare_splits
from omni.models import OmniPlusConfig


def make_sources(tmp_path, count: int = 16) -> dict:
    sources = {}
    for task in ("t2a", "a2a", "i2t"):
        records = []
        for index in range(count):
            record = dict(conversations=json.dumps(conversation(f"answer-{task}-{index}")))
            if task == "i2t":
                record["image_bytes"] = [f"image-{index}".encode()]
            else:
                record["answer_audios"] = [[index, index + 1, index + 2, index + 3]]
                if task == "a2a":
                    record["question_audios"] = [f"audio-{index}".encode()]
            records.append(record)
        sources[task] = tmp_path / f"{task}.parquet"
        pq.write_table(pa.Table.from_pylist(records), sources[task])
    return sources


def test_split_is_reproducible_balanced_and_media_disjoint(tmp_path) -> None:
    sources = make_sources(tmp_path)
    first = prepare_splits(sources, tmp_path / "first", 12, 6)
    second = prepare_splits(sources, tmp_path / "second", 12, 6)
    assert first["splits"] == second["splits"]
    assert all(source["rows"] == 16 for source in first["sources"])

    groups, identities = {}, {}
    for split, size in (("train", 12), ("validation", 6)):
        payload = torch.load(first["paths"][split], weights_only=True)
        assert len(payload["records"]) == size
        assert {row["task"] for row in payload["selections"]} == {"t2a", "a2a", "i2t"}
        identities[split] = {content_hash(record) for record in payload["records"]}
        groups[split] = set().union(*(media_hashes(record) for record in payload["records"]))
    assert identities["train"].isdisjoint(identities["validation"])
    assert groups["train"].isdisjoint(groups["validation"])
    with pytest.raises(FileExistsError, match="旧实验"):
        prepare_splits(sources, tmp_path / "first", 12, 6)


def test_missing_source_and_insufficient_unique_rows_do_not_create_cache(tmp_path) -> None:
    sources = {task: tmp_path / f"{task}.parquet" for task in ("t2a", "a2a", "i2t")}
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError, match="不自动下载"):
        prepare_splits(sources, output)
    assert not output.exists()

    sources = make_sources(tmp_path, count=1)
    with pytest.raises(ValueError, match="独立样本不足"):
        prepare_splits(sources, output, 6, 3)
    assert not output.exists()


def test_media_grouping_detects_partial_multiimage_overlap() -> None:
    first = media_hashes({"image_bytes": [[b"shared", b"first"]]})
    second = media_hashes({"image_bytes": [b"other", b"shared"]})
    assert len(first & second) == 1
    assert media_hashes({"answer_audios": [1, 2, 3]}) == media_hashes(
        {"answer_audios": [[1, 2, 3]]}
    )


def test_text_only_rows_cannot_be_mislabeled_as_t2a(tmp_path) -> None:
    sources = make_sources(tmp_path)
    pq.write_table(
        pa.Table.from_pylist([dict(conversations=json.dumps(conversation()))] * 16), sources["t2a"]
    )
    with pytest.raises(ValueError, match="必需模态"):
        prepare_splits(sources, tmp_path / "prepared", 6, 3)
    assert not (tmp_path / "prepared").exists()


def test_feature_cache_preserves_mixed_indices_and_single_shift(tmp_path) -> None:
    config = OmniPlusConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        pad_token_id=0,
        num_audio_codebooks=2,
        audio_codebook_size=16,
        audio_vocab_size=20,
        audio_pad_token=17,
        audio_stop_token=18,
        audio_spk_token=19,
    )
    records = [
        dict(conversations=conversation("ab")),
        dict(conversations=conversation("cd"), question_audios=[b"question"]),
        dict(
            conversations=conversation("ef", "<image>"),
            image_bytes=[b"image"],
            answer_audios=[[3, 4, 5, 6]],
        ),
    ]
    selections = [dict(task=task) for task in ("t2a", "a2a", "i2t")]
    split_path, output_path = tmp_path / "split.pt", tmp_path / "cached.pt"
    torch.save(dict(records=records, selections=selections), split_path)
    tokenizer = TinyTokenizer()
    audio_feature, image_feature = torch.ones(3, 8), torch.full((2, 10), 2.0)
    stats = cache_split(
        split_path,
        output_path,
        tokenizer,
        config,
        lambda value: audio_feature,
        lambda value: image_feature,
    )
    cached = torch.load(output_path, weights_only=True)
    assert cached["selections"] == selections
    assert stats["samples"] == 3 and stats["labels_shifted_once"]
    assert stats["audio_samples"] == stats["image_samples"] == 1
    assert stats["augmentation"] is False
    samples = cached["samples"]
    assert samples[0]["audio_features"] is samples[0]["image_features"] is None
    assert torch.equal(samples[1]["audio_features"], audio_feature)
    assert torch.equal(samples[2]["image_features"], image_feature)
    labels = samples[0]["labels"][samples[0]["labels"] != -100]
    assert labels.tolist() == tokenizer.encode("ab</s>\n")
    batch = OmniCollator(config)(samples)
    assert batch["audio_features"][0] is None and batch["audio_features"][2] is None
    assert batch["image_features"][0] is None and batch["image_features"][1] is None
    assert batch["audio_labels"][2].eq(config.audio_stop_token).sum() == 2
    with pytest.raises(FileExistsError, match="旧实验"):
        cache_split(split_path, output_path, tokenizer, config)
