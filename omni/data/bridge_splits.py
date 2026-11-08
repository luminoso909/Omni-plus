"""用 Checkpoint 绑定的数据清单保护 G5 留出集。"""

import hashlib
import json
from pathlib import Path

import torch

from omni.data.prepare import content_hash
from omni.data.r3_prepare import SPLITS, file_hash, group_records


def lineage_contracts(
    path: Path, source_sha: str, source_identity: dict, source_contract: dict | None
) -> list[dict]:
    """读取 B1 及其组件来源的原始缓存清单，按 Checkpoint 摘要去重。"""
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict):
        raise ValueError("B1 数据来源清单需要以 Checkpoint SHA-256 为键。")
    pending = [(source_sha, source_identity)]
    seen, contracts = set(), []
    while pending:
        sha, identity = pending.pop()
        if sha in seen:
            continue
        seen.add(sha)
        entry = entries.get(sha)
        if not isinstance(entry, dict) or not isinstance(entry.get("cache_manifest"), str):
            raise ValueError(f"B1 数据来源缺少 Checkpoint {sha} 的缓存清单。")
        cache = (path.parent / entry["cache_manifest"]).read_bytes()
        if hashlib.sha256(cache).hexdigest() != identity.get("data_manifest_sha256"):
            raise ValueError(f"B1 来源缓存清单的 SHA-256 与 Checkpoint 不一致：{sha}")
        contract = json.loads(cache).get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"B1 来源缓存清单缺少数据契约：{sha}")
        if sha == source_sha and source_contract is not None and contract != source_contract:
            raise ValueError("B1 运行清单与 Checkpoint 绑定的缓存数据契约不一致。")
        contracts.append(contract)

        initial = identity.get("settings", {}).get("model", {}).get("initialization", {})
        for name, value in initial.items():
            if not name.endswith("_checkpoint") or not value:
                continue
            parent_sha = identity.get("initialization_sha256", {}).get(name)
            parent = entries.get(parent_sha)
            if not isinstance(parent, dict) or not isinstance(parent.get("checkpoint"), str):
                raise ValueError(f"B1 数据来源缺少继承组件的 Checkpoint：{name}")
            if parent_sha in seen or any(parent_sha == queued for queued, _ in pending):
                continue
            checkpoint = path.parent / parent["checkpoint"]
            if file_hash(checkpoint) != parent_sha:
                raise ValueError(f"B1 继承组件 Checkpoint SHA-256 不一致：{name}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            parent_identity = payload.get("progress", {}).get("identity")
            if not isinstance(parent_identity, dict) or not parent_identity.get("settings"):
                raise ValueError(f"B1 继承组件缺少可核验的训练身份：{name}")
            pending.append((parent_sha, parent_identity))
            del payload
    return contracts


def check_bridge_splits(
    target_contract: dict,
    source_identity: dict,
    source_sha: str,
    lineage_path: Path | None,
    *,
    formal: bool,
    source_contract: dict | None = None,
) -> None:
    """合并两阶段已记录的底层媒体/文本身份，检查相连记录的 split。"""
    if formal and lineage_path is None:
        raise ValueError("正式 G5 需要 bridge_base_lineage 核验 B1 及继承组件的数据来源。")
    if lineage_path is not None:
        sources = lineage_contracts(lineage_path, source_sha, source_identity, source_contract)
    else:
        sources = [source_contract or {}]

    records, roles = [], []
    for is_target, contract in [(False, source) for source in sources] + [(True, target_contract)]:
        rows = contract.get("records")
        if formal and (contract.get("kind") != "audited_raw_media" or not rows):
            raise ValueError("正式 G5 的 B1 来源与目标需要包含 split 身份的审计数据契约。")
        if rows is None:
            continue  # 历史 Tiny 诊断只检查已有身份。
        identities = contract.get("identity_groups")
        if (formal or identities is not None) and not isinstance(identities, dict):
            raise ValueError("正式 G5 缺少完整媒体组身份，请从原始审计清单另建缓存。")
        for row in rows:
            split, group, media = row.get("split"), row.get("group_id"), row.get("media_ids")
            if (
                split not in SPLITS
                or not isinstance(group, str)
                or not group
                or not isinstance(media, list)
                or not media
                or any(not isinstance(item, str) or not item for item in media)
            ):
                raise ValueError("G5 数据身份缺少有效的 split、group_id 或 media_ids。")

            # group_id 绑定完整集合，不能只比较选中行或不可逆的组摘要。
            if identities is not None:
                members = identities.get(group)
                if (
                    not isinstance(members, list)
                    or not members
                    or any(not isinstance(item, str) or not item for item in members)
                    or members != sorted(set(members))
                    or content_hash(members) != group
                    or not set(media) <= set(members)
                ):
                    raise ValueError("G5 完整媒体组身份缺失或与 group_id、media_ids 不一致。")
                media = members

            records.append(dict(record_id=len(roles), media_ids=[*media, "group:" + group]))
            roles.append((is_target, split, row.get("record_id")))

    groups: dict[str, dict[bool, dict[str, str]]] = {}
    for row in group_records(records, {}):
        is_target, split, record = roles[row["record_id"]]
        groups.setdefault(row["group_id"], {}).setdefault(is_target, {})[split] = record
    allowed_sources = {
        "train": {"train", "validation"},
        "validation": {"validation"},
        "test": {"test"},
    }
    for group in groups.values():
        source = group.get(False, {})
        target = group.get(True, {})
        if len(target) > 1:
            raise ValueError(f"B1→G5 留出集冲突：同一数据组横跨目标 {', '.join(sorted(target))}。")
        for target_split, record in target.items():
            conflict = source.keys() - allowed_sources[target_split]
            if conflict:
                raise ValueError(
                    f"B1→G5 留出集冲突：来源 {', '.join(sorted(conflict))} "
                    f"与目标 {target_split} 样本 {record!r} 的数据身份重合。"
                )
