import json
from pathlib import Path

import pytest
import torch

from omni.data.bridge_splits import check_bridge_splits
from omni.data.r3_prepare import file_hash


def record(name: str, split: str, *identities: str, task: str = "t2a") -> dict:
    return dict(record_id=name, group_id=name, split=split, task=task, media_ids=list(identities))


def audited(*rows: dict) -> dict:
    return dict(kind="audited_raw_media", records=list(rows))


@pytest.mark.parametrize(
    ("source_split", "target_split", "identity"),
    [
        ("train", "validation", "id:shared-media"),
        ("train", "test", "sha256:shared-bytes"),
        ("validation", "test", "text:shared-prompt-answer"),
        ("test", "train", "sha256:sealed-test"),
        ("test", "validation", "sha256:sealed-test"),
    ],
)
def test_rejects_cross_stage_roles(source_split, target_split, identity):
    task = "text" if identity.startswith("text:") else "a2t"
    source = audited(record("old-id", source_split, identity, task=task))
    target = audited(record("new-id", target_split, identity, task=task))
    if task != "text":
        target["records"][0]["task"] = "t2a"
    with pytest.raises(ValueError, match="留出集冲突"):
        check_bridge_splits(target, {}, "source", None, formal=False, source_contract=source)


def test_unchanged_splits_and_disjoint_additions_are_allowed():
    source = audited(
        record("old-train", "train", "id:train"),
        record("old-val", "validation", "id:val"),
        record("old-test", "test", "id:test"),
    )
    target = audited(*source["records"], record("independent", "test", "id:new"))
    check_bridge_splits(target, {}, "source", None, formal=False, source_contract=source)


def test_source_aliases_cannot_join_distinct_target_splits():
    source = audited(record("old-val", "validation", "id:alias-A", "sha256:alias-B"))
    target = audited(
        record("new-train", "train", "id:alias-A"),
        record("new-val", "validation", "sha256:alias-B"),
    )
    with pytest.raises(ValueError, match="同一数据组横跨目标 train, validation"):
        check_bridge_splits(target, {}, "source", None, formal=False, source_contract=source)


def test_transitive_aliases_preserve_sealed_test():
    source = audited(
        record("old-test", "test", "id:old-A"),
        record("linked-test", "test", "id:old-A", "sha256:alias-B"),
    )
    target = audited(record("new-train", "train", "sha256:alias-B"))
    with pytest.raises(ValueError, match="来源 test 与目标 train"):
        check_bridge_splits(target, {}, "source", None, formal=False, source_contract=source)


def write_cache(path: Path, data: dict, **initialization: str) -> dict:
    path.write_text(json.dumps(dict(contract=data)))
    return dict(
        settings=dict(model=dict(initialization={key: key + ".pt" for key in initialization})),
        data_manifest_sha256=file_hash(path),
        initialization_sha256=initialization,
    )


def test_verifies_ancestor_bytes_and_transitive_media_aliases(tmp_path):
    ancestor_cache = tmp_path / "ancestor-cache.json"
    ancestor = tmp_path / "ancestor.pt"
    ancestor_identity = write_cache(ancestor_cache, audited(record("A", "train", "id:alias-A")))
    torch.save(dict(progress=dict(identity=ancestor_identity)), ancestor)
    ancestor_sha = file_hash(ancestor)
    # The intermediate B1 row links the ancestor's original media ID to a renamed target.
    source = audited(record("B", "validation", "id:alias-A", "sha256:bytes-B"))
    source_cache = tmp_path / "b1-cache.json"
    source_identity = write_cache(source_cache, source, components_checkpoint=ancestor_sha)
    lineage = tmp_path / "lineage.json"
    entries = {
        "b1": dict(cache_manifest=source_cache.name),
        ancestor_sha: dict(checkpoint=ancestor.name, cache_manifest=ancestor_cache.name),
    }
    lineage.write_text(json.dumps(entries))

    target = audited(record("C", "validation", "sha256:bytes-B"))
    with pytest.raises(ValueError, match="来源 train 与目标 validation"):
        check_bridge_splits(target, source_identity, "b1", lineage, formal=True)

    independent = audited(record("D", "test", "id:independent"))
    check_bridge_splits(independent, source_identity, "b1", lineage, formal=True)

    lineage.write_text(json.dumps({"b1": entries["b1"]}))
    with pytest.raises(ValueError, match="缺少继承组件"):
        check_bridge_splits(independent, source_identity, "b1", lineage, formal=True)
    lineage.write_text(json.dumps(entries))

    ancestor_cache.write_text("{}")
    with pytest.raises(ValueError, match="缓存清单的 SHA-256"):
        check_bridge_splits(independent, source_identity, "b1", lineage, formal=True)
    ancestor.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="继承组件 Checkpoint SHA-256"):
        check_bridge_splits(independent, source_identity, "b1", lineage, formal=True)


def test_formal_requires_complete_audited_provenance(tmp_path):
    target = audited(record("test", "test", "id:independent"))
    with pytest.raises(ValueError, match="需要 bridge_base_lineage"):
        check_bridge_splits(target, {}, "b1", None, formal=True)

    cache = tmp_path / "cache.json"
    identity = write_cache(cache, {})
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"b1": dict(cache_manifest=cache.name)}))
    with pytest.raises(ValueError, match="需要包含 split 身份"):
        check_bridge_splits(target, identity, "b1", lineage, formal=True)
    with pytest.raises(ValueError, match="运行清单.*不一致"):
        check_bridge_splits(target, identity, "b1", lineage, formal=True, source_contract=target)

    # Existing local Tiny diagnostics may still use their historical cache contract.
    check_bridge_splits({}, {}, "b1", None, formal=False)
