"""R3 本地数据契约：真实媒体重编码、分组划分和不可混用的冻结特征缓存。"""

import hashlib
import importlib.metadata
import json
import random
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from omni.data.audio_alignment import delay_codes, restore_codes
from omni.data.prepare import content_hash
from omni.data.qwen_omni import QwenOmniFormatter, audio_codes
from omni.models.config import OmniPlusConfig

TASKS = ("t2a", "i2t", "a2t", "a2a")
SPLITS = ("train", "validation", "test")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_data(settings: dict, root: Path) -> dict:
    """只读核对预算、已有资产与处理器版本；不初始化模型或写缓存。"""
    data, assets = settings["data"], settings["assets"]
    tasks = data["tasks"]
    if (
        not tasks
        or not set(tasks) <= set(TASKS)
        or any(
            type(count) is not int or not 8 <= count <= (32 if task in {"t2a", "i2t"} else 16)
            for task, count in tasks.items()
        )
    ):
        raise ValueError("本地预算需要 T2A/I2T 各 8–32 条，A2T/A2A 各 8–16 条。")
    if type(data["seed"]) is not int or not 32 <= data["max_length"] <= 2048:
        raise ValueError("需要整数 seed 和 32–2048 的 max_length。")

    qwen = root / assets["qwen_dir"]
    reference = root / assets["reference_root"]
    r2 = json.loads((root / "experiments/r2/20261108-cpu-fp32/manifest.json").read_text())
    lock = yaml.safe_load((root / "configs/assets_v2.yaml").read_text())["assets"]["qwen3_1_7b"]
    files = {}
    for name, expected in lock["files"].items():
        path = qwen / name
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Qwen 资产版本不匹配：{name}")
        files[f"qwen/{name}"] = expected["sha256"]
    for name, directory, weight in (
        ("sensevoice", "SenseVoiceSmall", "model.pt"),
        ("siglip", "siglip2-base-p32-256-ve", "model.safetensors"),
        ("mimi", "mimi", "model.safetensors"),
    ):
        path = reference / "model" / directory
        if file_hash(path / weight) != r2["reference_assets"][name]["sha256"]:
            raise ValueError(f"{name} 权重与 R2 资产版本不匹配。")
        # 配置、归一化与前端词表也绑定；不只比较大权重。
        for item in sorted(path.iterdir()):
            if item.is_file() and (
                item.name == weight or item.suffix in {".json", ".yaml", ".txt"}
            ):
                files[f"{name}/{item.name}"] = file_hash(item)
    source_manifest = root / data["sources_manifest"]
    history = json.loads(source_manifest.read_text())
    sources = {}
    for source in history["sources"]:
        name = Path(source["path"]).name
        actual = file_hash(root / data["source_dir"] / name)
        if actual != source["sha256"]:
            raise ValueError(f"已有数据源校验值不匹配：{name}")
        sources[source["task"]] = dict(file=name, sha256=actual)
    model = yaml.safe_load((root / settings["model"]["config"]).read_text())["model"]
    source_lock = json.loads((root / "experiments/p4/subset-assets-v2.json").read_text())
    return dict(
        schema_version=1,
        tasks=tasks,
        seed=data["seed"],
        max_length=data["max_length"],
        model=model,
        source_revision=source_lock["revision"],
        source_license=source_lock["declared_licenses"],
        license_note=source_lock["license_note"],
        sources=sources,
        history_sha256=file_hash(source_manifest),
        qwen_revision=lock["revision"],
        encoder_revisions={
            name: "local-sha256:" + r2["reference_assets"][name]["sha256"]
            for name in ("sensevoice", "siglip", "mimi")
        },
        assets=files,
        processors={
            name: importlib.metadata.version(name)
            for name in ("transformers", "funasr", "torch", "scipy", "soundfile", "pillow")
        },
        implementation={
            name: file_hash(root / name)
            for name in (
                "omni/data/r3_prepare.py",
                "omni/data/qwen_omni.py",
                "omni/data/audio_alignment.py",
                "omni/models/encoders/model_encoders.py",
                "omni/inference/codec.py",
            )
        },
        sampling_rates=dict(sensevoice=16000, mimi=24000),
        code_policy="reencode_raw_question_audio; quarantine_all_imported_answer_codes",
    )


def group_records(records: list[dict], history: dict[str, str]) -> list[dict]:
    """同媒体跨字段/来源去重；连接多个媒体的记录属于同一个传递闭包。"""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, item in enumerate(records):
        for identity in item["media_ids"]:
            if identity in owners:
                parents[find(index)] = find(owners[identity])
            owners[identity] = index
    groups = {}
    for index, item in enumerate(records):
        groups.setdefault(find(index), []).append(item)
    result = []
    for items in groups.values():
        assignments = {history[item["record_id"]] for item in items if item["record_id"] in history}
        if len(assignments) > 1:
            raise ValueError("同一媒体组已出现在不同历史 split，拒绝重新分配。")
        split = next(iter(assignments), "test")
        group_id = content_hash(
            sorted({identity for item in items for identity in item["media_ids"]})
        )
        for item in items:
            result.append(dict(item, group_id=group_id, split=split))
    return result


def select_records(records: list[dict], count: int, seed: int) -> list[dict]:
    """保留历史训练/验证归属，未使用媒体才进入本地工程 Test。"""
    holdout = max(1, count // 10)
    quotas = dict(train=count - 2 * holdout, validation=holdout, test=holdout)
    chosen, seen = [], set()
    rng = random.Random(seed)
    for split, quota in quotas.items():
        pool = sorted(
            (row for row in records if row["split"] == split), key=lambda row: row["record_id"]
        )
        rng.shuffle(pool)
        accepted = []
        for row in pool:
            if row["group_id"] not in seen:
                seen.add(row["group_id"])
                accepted.append(row)
            if len(accepted) == quota:
                break
        if len(accepted) != quota:
            raise ValueError(f"{split} 独立媒体不足：需要 {quota}，得到 {len(accepted)}。")
        chosen.extend(accepted)
    return chosen


def load_prepared(cache_dir: Path, split: str, expected_contract: dict) -> list[dict]:
    """校验文件内容和完整处理契约，再读 CPU Tensor；Test 仅供显式数据审计。"""
    if split not in SPLITS:
        raise ValueError("split 必须是 train/validation/test。")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["contract"] != expected_contract:
        raise ValueError("缓存版本不匹配，必须使用新目录重新准备。")
    path = cache_dir / f"{split}.pt"
    if file_hash(path) != manifest["files"][path.name]:
        raise ValueError(f"缓存校验失败：{path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["contract_id"] != content_hash(manifest["contract"]):
        raise ValueError("缓存契约摘要不一致。")
    return payload["samples"]


def prepare_data(settings: dict, root: Path) -> dict:
    """有限真实媒体 → Qwen 样本和冻结特征；相同契约可复用，变更必须另建目录。"""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, MimiModel

    from omni.inference.codec import MimiCodec, load_audio
    from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder

    contract = inspect_data(settings, root)
    data, assets = settings["data"], settings["assets"]
    output = root / data["cache_dir"]
    if output.exists():
        for split in SPLITS:
            load_prepared(output, split, contract)
        return dict(json.loads((output / "manifest.json").read_text()), cache_hit=True)

    history = json.loads((root / data["sources_manifest"]).read_text())
    assignments = {
        row["record_id"]: split for split, rows in history["splits"].items() for row in rows
    }
    records = []
    for task in ("a2a", "i2t"):
        path = root / data["source_dir"] / contract["sources"][task]["file"]
        for raw in pq.read_table(path).to_pylist():
            if raw["_source_revision"] != contract["source_revision"]:
                raise ValueError("数据 revision 不匹配。")
            media = [raw["image_bytes"]] if task == "i2t" else raw["question_audios"]
            if not media or any(not isinstance(item, bytes) or not item for item in media):
                raise ValueError("需要真实非空原始媒体 bytes。")
            records.append(
                dict(
                    task=task,
                    raw=raw,
                    record_id=content_hash(
                        {key: val for key, val in raw.items() if not key.startswith("_source_")}
                    ),
                    media_ids=[content_hash(item) for item in media],
                )
            )
    grouped = group_records(records, assignments)
    task_rows = {
        task: select_records(
            [row for row in grouped if row["task"] == ("i2t" if task == "i2t" else "a2a")],
            count,
            data["seed"],
        )
        for task, count in data["tasks"].items()
    }
    selected_ids = {task: {row["record_id"] for row in rows} for task, rows in task_rows.items()}
    chosen = list({row["record_id"]: row for rows in task_rows.values() for row in rows}.values())
    reference = root / assets["reference_root"]
    tokenizer = AutoTokenizer.from_pretrained(root / assets["qwen_dir"], local_files_only=True)
    config = OmniPlusConfig(**contract["model"])
    formatter = QwenOmniFormatter(tokenizer, config, max_length=data["max_length"])
    speech = SenseVoiceEncoder(reference / "model/SenseVoiceSmall")
    vision = SiglipEncoder(reference / "model/siglip2-base-p32-256-ve")
    codec = MimiCodec(
        MimiModel.from_pretrained(reference / "model/mimi", local_files_only=True), num_codebooks=8
    )
    if (codec.sampling_rate, codec.codebook_size, codec.num_codebooks) != (24000, 2048, 8):
        raise ValueError("Mimi 配置必须为 24kHz、8 路、2048 个声学 ID。")

    samples = {split: [] for split in SPLITS}
    selections = {split: [] for split in SPLITS}
    for row in chosen:
        raw, split = row["raw"], row["split"]
        if row["task"] == "i2t":
            views = {"i2t": formatter.sample(raw, image_features=vision(raw["image_bytes"]))}
            rates = None
        else:
            messages = json.loads(raw["conversations"])
            if len(messages) != 2 or len(raw["question_audios"]) != 1:
                raise ValueError("当前真实语音诊断仅支持已核对的单轮记录。")
            transcript = messages[0]["content"].replace("<audio>", "").strip()
            if not transcript:
                raise ValueError("缺少原音频对应的转写，不可构造监督。")
            wav16 = load_audio(raw["question_audios"][0], target_rate=16000)
            wav24 = load_audio(raw["question_audios"][0], target_rate=24000)
            feature = speech(wav16, sampling_rate=16000)
            codes = audio_codes(codec.encode(wav24)[0], config)
            torch.testing.assert_close(
                restore_codes(delay_codes(codes[None]), codes.shape[-1])[0], codes, rtol=0, atol=0
            )
            views = {}
            for task, prompt in (
                ("t2a", "Read this text aloud: " + transcript),
                ("a2t", "<audio> Transcribe the speech."),
                ("a2a", "<audio> Repeat the speech aloud and transcribe it."),
            ):
                record = dict(
                    conversations=[
                        dict(role="user", content=prompt),
                        dict(role="assistant", content=transcript),
                    ]
                )
                views[task] = formatter.sample(
                    record,
                    audio_features=feature if task != "t2a" else None,
                    answer_codes=codes if task != "a2t" else None,
                )
            rates = dict(
                sensevoice_samples=wav16.numel(), mimi_samples=wav24.numel(), frames=codes.shape[-1]
            )
        for task, sample in views.items():
            if row["record_id"] not in selected_ids.get(task, set()):
                continue
            sample["task"] = task
            samples[split].append(sample)
            selections[split].append(
                dict(
                    task=task,
                    record_id=row["record_id"],
                    group_id=row["group_id"],
                    media_ids=row["media_ids"],
                    source_row=raw["_source_row"],
                    source_file=raw["_source_file"],
                    derivation="original_i2t" if task == "i2t" else "user_audio_transcript_" + task,
                    sampling=rates,
                    tokens=sample["input_ids"].numel(),
                )
            )

    # 原子发布新目录。任何处理中断都不生成可被误用的完整缓存。
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            torch.save(
                dict(
                    samples=samples[split],
                    selections=selections[split],
                    contract_id=content_hash(contract),
                ),
                temporary / f"{split}.pt",
            )
        manifest = dict(
            status="prepared",
            contract=contract,
            selections=selections,
            counts={
                split: {
                    task: sum(row["task"] == task for row in selections[split])
                    for task in data["tasks"]
                }
                for split in SPLITS
            },
            files={f"{split}.pt": file_hash(temporary / f"{split}.pt") for split in SPLITS},
            unique_media=len(chosen),
            cache_hit=False,
            limitations=(
                "工程诊断；语音三任务为同一原音频的不同视图，A2A 是复述任务。"
                "Test 来自 P4 未选择的媒体组，不是正式 Benchmark。"
                "源数据所有预编码回答因缺少精确编码器版本而隔离，未用于训练。"
            ),
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        if output.exists():
            raise FileExistsError("目标缓存已被创建，拒绝覆盖。")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
