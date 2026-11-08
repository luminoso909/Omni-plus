"""审计后的本地 JSONL 数据源：统一重编码和版本契约，不推断声码来源。"""

import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from omni.data.audio_alignment import delay_codes, restore_codes
from omni.data.prepare import content_hash
from omni.data.qwen_omni import QwenOmniFormatter, audio_codes
from omni.data.r3_prepare import SPLITS, file_hash, group_records, load_prepared, select_records
from omni.inference.codec import MimiCodec, load_audio
from omni.models.config import OmniPlusConfig

TASKS = {"t2a", "i2t", "a2t", "a2a", "text"}
MEDIA = ("image", "input_audio", "target_audio")


def read_sources(path: Path, expected_sha256: str) -> list[dict]:
    """读取逐行审计记录；相对媒体路径从 JSONL 所在目录解析。"""
    if file_hash(path) != expected_sha256:
        raise ValueError("数据 Manifest SHA-256 不匹配。")

    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("sample_id", "source", "revision", "license", "prompt", "answer")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError("每条样本需要非空身份、来源、版本、许可、prompt 和 answer。")
        if (
            row["sample_id"] in seen
            or row.get("task") not in TASKS
            or row.get("split") not in SPLITS
        ):
            raise ValueError("样本 ID 重复、任务未知或 split 无效。")
        seen.add(row["sample_id"])
        task = row["task"]
        expected_media = {
            "t2a": {"target_audio"},
            "i2t": {"image"},
            "a2t": {"input_audio"},
            "a2a": {"input_audio", "target_audio"},
            "text": set(),
        }[task]
        if {key for key in MEDIA if row.get(key) is not None} != expected_media:
            raise ValueError("任务的输入/目标媒体不完整或包含不允许的媒体。")

        # 同一个媒体 ID 或内容摘要跨输入/目标角色仍属于同一组。
        identities = []
        for kind in sorted(expected_media):
            media = row[kind]
            if set(media) != {"path", "sha256", "media_id"} or not all(
                isinstance(value, str) and value for value in media.values()
            ):
                raise ValueError("媒体需要 path、sha256 和全局唯一的原始 media_id。")
            source = (path.parent / media["path"]).resolve(strict=True)
            if not source.is_file() or file_hash(source) != media["sha256"]:
                raise ValueError(f"媒体校验失败：{row['sample_id']} / {kind}")
            identities.extend(("id:" + media["media_id"], "sha256:" + media["sha256"]))
            row[kind] = dict(media, path=str(source))
        if "target_audio" in expected_media:
            if (
                row.get("target_provenance") != "original_recording"
                or row.get("target_transcript") != row["answer"]
            ):
                raise ValueError("目标语音需要原始录音来源声明及与 answer 一致的审计转写。")
        if task == "a2a":
            if row.get("a2a_kind") not in {"qa", "echo"}:
                raise ValueError("A2A 必须明确标注 qa 或 echo；禁止自动改写任务语义。")
            if row["a2a_kind"] == "qa" and any(
                row["input_audio"][key] == row["target_audio"][key]
                for key in ("sha256", "media_id")
            ):
                raise ValueError("问答式 A2A 的目标不能是同一输入语音。")
        if not identities:
            identities = ["text:" + content_hash((row["prompt"], row["answer"]))]
        row.update(record_id=row["sample_id"], media_ids=identities)
        rows.append(row)
    if not rows:
        raise ValueError("审计数据源不能为空。")

    grouped = group_records(rows, {row["record_id"]: row["split"] for row in rows})
    return grouped


def inspect_sources(settings: dict, root: Path) -> dict:
    """检查显式数据预算与固定资产；不读取旧 P4 数据，不创建编码器。"""
    data, assets = settings["data"], settings["assets"]
    tasks = data["tasks"]
    if (
        not tasks
        or not set(tasks) <= TASKS
        or any(type(n) is not int or n < 3 for n in tasks.values())
    ):
        raise ValueError("每个审计任务预算至少 3 条，分别保留 train/validation/test。")
    if type(data["seed"]) is not int or not 32 <= data["max_length"] <= 32768:
        raise ValueError("需要整数 seed 以及 32–32768 的 max_length。")
    manifest = root / data["audited_manifest"]
    rows = read_sources(manifest, data["audited_manifest_sha256"])
    chosen = [
        row
        for task, count in tasks.items()
        for row in select_records([row for row in rows if row["task"] == task], count, data["seed"])
    ]

    # 复用 R1/R2 固定的模型资产身份，不接受任意同名目录。
    r2 = json.loads((root / "experiments/r2/20261108-cpu-fp32/manifest.json").read_text())
    lock = yaml.safe_load((root / "configs/assets_v2.yaml").read_text())["assets"]["qwen3_1_7b"]
    files = {}
    for name, expected in lock["files"].items():
        digest = file_hash(root / assets["qwen_dir"] / name)
        if digest != expected["sha256"]:
            raise ValueError(f"Qwen 资产版本不匹配：{name}")
        files[f"qwen/{name}"] = digest
    reference = root / assets["reference_root"] / "model"
    for name, folder, weight in (
        ("sensevoice", "SenseVoiceSmall", "model.pt"),
        ("siglip", "siglip2-base-p32-256-ve", "model.safetensors"),
        ("mimi", "mimi", "model.safetensors"),
    ):
        directory = reference / folder
        if file_hash(directory / weight) != r2["reference_assets"][name]["sha256"]:
            raise ValueError(f"{name} 权重与 R2 不一致。")
        for path in sorted(directory.iterdir()):
            if path.is_file() and (
                path.name == weight or path.suffix in {".json", ".yaml", ".txt"}
            ):
                files[f"{name}/{path.name}"] = file_hash(path)
    return dict(
        schema_version=2,
        kind="audited_raw_media",
        tasks=tasks,
        seed=data["seed"],
        max_length=data["max_length"],
        model=yaml.safe_load((root / settings["model"]["config"]).read_text())["model"],
        qwen_revision=lock["revision"],
        assets=files,
        manifest_sha256=data["audited_manifest_sha256"],
        records=chosen,
        processors={
            name: importlib.metadata.version(name)
            for name in ("transformers", "funasr", "torch", "scipy", "soundfile", "pillow")
        },
        implementation={
            name: file_hash(root / name)
            for name in (
                "omni/data/sft_sources.py",
                "omni/data/r3_prepare.py",
                "omni/data/qwen_omni.py",
                "omni/data/audio_alignment.py",
                "omni/models/encoders/model_encoders.py",
                "omni/inference/codec.py",
            )
        },
        sampling_rates=dict(sensevoice=16000, mimi=24000),
        code_policy="raw_target_audio_reencoded_with_locked_mimi; no_imported_codes",
    )


def encode_sample(row: dict, formatter: QwenOmniFormatter, speech, vision, codec) -> dict:
    """真实媒体 → 文本 [T]/声码 [8,T]；依赖注入用于离线接口单测。"""
    audio = image = codes = None
    if row.get("input_audio"):
        waveform = load_audio(row["input_audio"]["path"], target_rate=16000)
        audio = speech(waveform, sampling_rate=16000)
    if row.get("image"):
        image = vision(row["image"]["path"])
        if image.shape != (64, formatter.config.image_hidden_size):
            raise ValueError("I2T 需要 64 个有效视觉 Tokens。")
    if row.get("target_audio"):
        waveform = load_audio(row["target_audio"]["path"], target_rate=24000)
        codes = audio_codes(codec.encode(waveform)[0], formatter.config)
        delayed = delay_codes(codes)
        torch.testing.assert_close(restore_codes(delayed, codes.shape[-1]), codes, rtol=0, atol=0)
    sample = formatter.sample(
        dict(
            conversations=[
                dict(role="user", content=row["prompt"]),
                dict(role="assistant", content=row["answer"]),
            ]
        ),
        audio_features=audio,
        image_features=image,
        answer_codes=codes,
    )
    sample["task"] = row["task"]
    return sample


def prepare_sources(settings: dict, root: Path) -> dict:
    """LJSpeech/AISHELL-3/MiniMind-O 等已审计记录共享同一个薄准备流程。"""
    from transformers import AutoTokenizer, MimiModel

    from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder

    contract = inspect_sources(settings, root)
    output = root / settings["data"]["cache_dir"]
    if output.exists():
        for split in SPLITS:
            load_prepared(output, split, contract)
        return dict(json.loads((output / "manifest.json").read_text()), cache_hit=True)

    config = OmniPlusConfig(**contract["model"])
    formatter = QwenOmniFormatter(
        AutoTokenizer.from_pretrained(root / settings["assets"]["qwen_dir"], local_files_only=True),
        config,
        max_length=contract["max_length"],
    )
    reference = root / settings["assets"]["reference_root"] / "model"
    speech = (
        SenseVoiceEncoder(reference / "SenseVoiceSmall")
        if set(contract["tasks"]) & {"a2t", "a2a"}
        else None
    )
    vision = (
        SiglipEncoder(reference / "siglip2-base-p32-256-ve") if "i2t" in contract["tasks"] else None
    )
    codec = (
        MimiCodec(
            MimiModel.from_pretrained(reference / "mimi", local_files_only=True), num_codebooks=8
        )
        if set(contract["tasks"]) & {"t2a", "a2a"}
        else None
    )
    if codec is not None and (codec.sampling_rate, codec.codebook_size) != (24000, 2048):
        raise ValueError("目标编码器需要同版本的 24 kHz、8×2048 Mimi。")

    samples, selections = {split: [] for split in SPLITS}, {split: [] for split in SPLITS}
    for row in contract["records"]:
        sample = encode_sample(row, formatter, speech, vision, codec)
        samples[row["split"]].append(sample)
        selections[row["split"]].append(
            dict(row, tokens=len(sample["input_ids"]), derivation="audited_" + row["task"])
        )

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
        result = dict(
            status="prepared",
            contract=contract,
            selections=selections,
            counts={
                split: {
                    task: sum(row["task"] == task for row in selections[split])
                    for task in contract["tasks"]
                }
                for split in SPLITS
            },
            files={f"{split}.pt": file_hash(temporary / f"{split}.pt") for split in SPLITS},
            cache_hit=False,
            limitations="来源与转写依赖输入审计清单；字节核验不替代人工语义/许可核查。",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        if output.exists():
            raise FileExistsError("目标缓存已存在，拒绝覆盖。")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result
