"""R3-L 配置、参数分组和有限步数验收；训练及恢复复用 P4。"""

import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from omni.data.omni_dataset import OmniCollator
from omni.models.config import OmniPlusConfig
from omni.models.model_omni_v2 import OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.sft import load_checkpoint, save_checkpoint, train_step


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_settings(path: Path, root: Path) -> dict:
    settings = yaml.safe_load(path.read_text())
    allowed = {
        "root": {"schema_version", "assets", "data", "model", "training", "lora"},
        "assets": {"qwen_dir", "reference_root"},
        "data": {"sources_manifest", "source_dir", "cache_dir", "seed", "tasks", "max_length"},
        "model": {"kind", "config", "tiny"},
        "training": {
            "recipe",
            "tasks",
            "device",
            "dtype",
            "seed",
            "threads",
            "steps",
            "batch_size",
            "accumulation_steps",
            "learning_rate",
            "weight_decay",
            "scheduler",
            "warmup_steps",
            "stop_weight",
            "grad_clip",
        },
        "lora": {"enabled"},
    }
    for name, keys in allowed.items():
        section = settings if name == "root" else settings[name]
        if not isinstance(section, dict) or set(section) - keys:
            raise ValueError(f"未知配置键或错误配置类型：{name}")
    tiny_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "talker_hidden_size",
    }
    if set(settings["model"].get("tiny", {})) - tiny_keys:
        raise ValueError("未知 tiny 模型配置键。")
    training = settings["training"]
    if settings.get("schema_version") != 1:
        raise ValueError("需要 schema_version: 1。")
    if settings["model"]["kind"] not in {"tiny", "qwen3"}:
        raise ValueError("model.kind 仅支持 tiny 或 qwen3。")
    if settings.get("lora", {}).get("enabled", False):
        raise ValueError("LoRA 仅预留配置；完整实现和验收在 R4-L。")
    if training["recipe"] not in {"r4a", "projectors"}:
        raise ValueError("recipe 仅支持 r4a 或 projectors。")
    if training["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("device 需要 cpu、mps 或 cuda。")
    if training["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("未知 dtype。")
    for key in ("steps", "batch_size", "accumulation_steps", "threads"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} 必须为正整数。")
    if training["steps"] > 10:
        raise ValueError("R3-L smoke 最多 10 步；100-step Profile 属于后续授权。")
    for key in ("learning_rate", "grad_clip"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} 必须为有限正数。")
    for key in ("weight_decay", "stop_weight"):
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} 必须为有限非负数。")
    if training["scheduler"] != "linear" or not 0 <= training["warmup_steps"] < training["steps"]:
        raise ValueError("需要 linear scheduler，warmup_steps 应小于 steps。")
    if not training["tasks"] or set(training["tasks"]) - set(settings["data"]["tasks"]):
        raise ValueError("训练任务必须属于有限数据任务清单。")
    if training["recipe"] == "r4a" and set(training["tasks"]) != {"t2a"}:
        raise ValueError("R4-A 初始配方只使用 T2A；多模态投影验证使用 projectors。")
    if training["recipe"] == "projectors" and "t2a" in training["tasks"]:
        raise ValueError("Projector 验证需要 I2T、A2T 或 A2A 输入模态。")

    # 模型路径始终指向现成资产，tiny 也不能悄悄换用 MiniMind tokenizer。
    model_dir = root / settings["assets"]["qwen_dir"]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    native = json.loads((model_dir / "config.json").read_text())
    if native.get("model_type") != "qwen3" or native.get("hidden_size") != 2048:
        raise ValueError("需要已验收的 Qwen3-1.7B 资产。")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        files = set(json.loads(index.read_text())["weight_map"].values())
    else:
        files = {"model.safetensors"}
    for name in files:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    return settings


def model_config(settings: dict, root: Path) -> tuple[OmniPlusConfig, Qwen3Config]:
    values = yaml.safe_load((root / settings["model"]["config"]).read_text())["model"]
    native = Qwen3Config.from_pretrained(
        root / settings["assets"]["qwen_dir"], local_files_only=True
    )
    if settings["model"]["kind"] == "tiny":
        tiny = settings["model"]["tiny"]
        # 只缩小计算宽度；词表、媒体维度和八码本接口保持真实数据语义。
        native = Qwen3Config(
            hidden_size=tiny["hidden_size"],
            intermediate_size=tiny["intermediate_size"],
            num_hidden_layers=tiny["num_hidden_layers"],
            num_attention_heads=tiny["num_attention_heads"],
            num_key_value_heads=tiny["num_key_value_heads"],
            head_dim=tiny["hidden_size"] // tiny["num_attention_heads"],
            vocab_size=native.vocab_size,
            tie_word_embeddings=True,
            attention_dropout=0.1,
        )
        values |= {
            key: getattr(native, key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            )
        }
        values |= dict(
            bridge_layer=0,
            talker_hidden_size=tiny["talker_hidden_size"],
            talker_num_attention_heads=4,
            talker_num_key_value_heads=2,
            talker_intermediate_size=tiny["talker_hidden_size"] * 2,
            talker_adapter_rank=8,
            dropout=0.1,
        )
    return OmniPlusConfig(**values), native


def build_model(settings: dict, root: Path, *, meta: bool = False) -> OmniPlusOmni:
    config, native = model_config(settings, root)
    if meta:
        with torch.device("meta"):
            thinker = Qwen3ForCausalLM(native)
            model = OmniPlusOmni(QwenBackend(thinker), config)
    else:
        if settings["model"]["kind"] == "tiny":
            thinker = Qwen3ForCausalLM(native)
        else:
            thinker, loading = Qwen3ForCausalLM.from_pretrained(
                root / settings["assets"]["qwen_dir"],
                local_files_only=True,
                dtype=torch.float32,
                use_safetensors=True,
                output_loading_info=True,
            )
            if any(loading.values()):
                raise ValueError(f"Qwen 权重未严格加载：{loading}")
        model = OmniPlusOmni(QwenBackend(thinker), config)
    configure_parameters(model, settings["training"]["recipe"])
    return model


def parameter_group(name: str) -> str:
    if name.startswith("talker.embed_tokens."):
        return "audio_embedding"
    if name.startswith("talker.lm_head."):
        return "audio_heads"
    if name.startswith("talker.spk_proj."):
        return "speaker"
    return name.split(".", 1)[0]


def configure_parameters(model: nn.Module, recipe: str) -> dict:
    active = {"bridge", "talker", "audio_embedding", "audio_heads"}
    if recipe == "projectors":
        active = {"vision_proj", "audio_proj"}
    elif recipe != "r4a":
        raise ValueError("未知冻结配方。")
    groups = {}
    for name, param in model.named_parameters():
        group = parameter_group(name)
        param.requires_grad_(group in active)
        counts = groups.setdefault(group, dict(total=0, trainable=0))
        counts["total"] += param.numel()
        counts["trainable"] += param.numel() if param.requires_grad else 0
    model.thinker.eval()
    return groups


def parameter_digest(model: nn.Module, *, frozen: bool) -> str:
    result = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad != frozen:
            result.update(name.encode())
            result.update(
                param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return result.hexdigest()


def optimizer_state(model: nn.Module, settings: dict) -> tuple:
    training = settings["training"]
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    warmup, total = training["warmup_steps"], training["steps"]

    def scale(step: int) -> float:
        return min((step + 1) / max(1, warmup), max(0.0, (total - step) / (total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        training["dtype"]
    ]
    scaler = torch.amp.GradScaler(training["device"], enabled=dtype == torch.float16)
    return optimizer, scheduler, scaler, dtype


def smoke(
    settings: dict,
    root: Path,
    run_dir: Path,
    *,
    resume: Path | None = None,
    steps: int | None = None,
) -> dict:
    from omni.data.r3_prepare import inspect_data, load_prepared

    training = settings["training"]
    device = training["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 尚未在本机验收；需要实际可用的 GPU。")
    if device != "cuda" and training["dtype"] != "float32":
        raise ValueError("本地验收使用 FP32；混合精度留给 CUDA Profile。")
    target = training["steps"] if steps is None else steps
    if not 0 < target <= training["steps"]:
        raise ValueError("--steps 必须在配置步数预算内。")
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖运行目录：{run_dir}")

    # 在加载模型前验证数据及完整的运行身份，错误恢复不能消耗训练步骤。
    contract = inspect_data(settings, root)
    samples = load_prepared(root / settings["data"]["cache_dir"], "train", contract)
    manifest_path = root / settings["data"]["cache_dir"] / "manifest.json"
    identity = dict(
        settings=settings,
        data_manifest_sha256=digest(manifest_path),
        runtime=dict(torch=str(torch.__version__), transformers=transformers.__version__),
        source_sha256={
            name: digest(root / name)
            for name in (
                "omni/training/r3_local.py",
                "omni/training/sft.py",
                "omni/training/losses.py",
                "omni/models/model_omni_v2.py",
                "omni/models/thinker/qwen_backend.py",
                "omni/models/talker/model_talker.py",
                "omni/models/thinker/model_thinker.py",
                "omni/data/r3_prepare.py",
                "omni/data/qwen_omni.py",
            )
        },
    )
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["progress"]["identity"] != identity:
            raise ValueError("Resume 配置或数据版本不一致。")
        if checkpoint["progress"]["step"] >= target:
            raise ValueError("目标 step 必须大于恢复 step。")

    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(training["threads"])
    started = time.perf_counter()
    model = build_model(settings, root).to(device)
    groups = configure_parameters(model, training["recipe"])
    optimizer, scheduler, scaler, dtype = optimizer_state(model, settings)
    progress = dict(step=0, epoch=0, cursor=0, identity=identity)
    if resume:
        del checkpoint
        progress = load_checkpoint(resume, model, optimizer, scaler)
        scheduler.load_state_dict(progress["scheduler"])

    # 每个缓存样本携带任务 metadata；只把 Collator 所需张量交给旧训练步骤。
    chosen = [row for row in samples if row["task"] in training["tasks"]]
    if not chosen:
        raise ValueError("缓存中没有配置要求的训练任务。")
    collate = OmniCollator(model.config)
    frozen_before = parameter_digest(model, frozen=True)
    trainable_before = parameter_digest(model, frozen=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    start_step = progress["step"]
    while progress["step"] < target:
        batches = []
        for _ in range(training["accumulation_steps"]):
            if progress["cursor"] >= len(chosen):
                progress["epoch"], progress["cursor"] = progress["epoch"] + 1, 0
            order = torch.randperm(
                len(chosen), generator=torch.Generator().manual_seed(seed + progress["epoch"])
            ).tolist()
            start = progress["cursor"]
            indices = order[start : start + training["batch_size"]]
            batches.append(collate([chosen[index] for index in indices]))
            progress["cursor"] += len(indices)
        metrics = train_step(
            model,
            batches,
            optimizer,
            scaler,
            device,
            dtype,
            training["stop_weight"],
            training["grad_clip"],
        )
        if not all(math.isfinite(value) for value in metrics.values()) or metrics["grad_norm"] <= 0:
            raise RuntimeError("Smoke 需要有限损失和非零梯度。")
        if metrics["optimizer_updated"] != 1:
            raise RuntimeError("Smoke 优化器没有更新。")
        scheduler.step()
        progress["step"] += 1
        progress["scheduler"] = scheduler.state_dict()
        row = dict(step=progress["step"], learning_rate=scheduler.get_last_lr()[0], **metrics)
        rows.append(row)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    frozen_after = parameter_digest(model, frozen=True)
    trainable_after = parameter_digest(model, frozen=False)
    if frozen_before != frozen_after or trainable_before == trainable_after:
        raise RuntimeError("冻结参数变化或目标参数未更新。")
    save_checkpoint(run_dir / f"step-{progress['step']:06d}.pt", model, optimizer, scaler, progress)
    grad_groups = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not param.requires_grad or not param.grad.isfinite().all():
                raise RuntimeError("冻结参数收到梯度或梯度非有限。")
            group = parameter_group(name)
            grad_groups[group] = max(grad_groups.get(group, 0), float(param.grad.abs().max()))
    result = dict(
        status="passed",
        scope="R3-L 有限步数工程验收，非正式 SFT 或质量评测",
        model_kind=settings["model"]["kind"],
        device=device,
        dtype=training["dtype"],
        python=platform.python_version(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        config=settings,
        parameter_groups=groups,
        train_samples=len(chosen),
        tasks=training["tasks"],
        start_step=start_step,
        step=progress["step"],
        resume_from=str(resume) if resume else None,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        trainable_before=trainable_before,
        trainable_after=trainable_after,
        gradient_groups=grad_groups,
        metrics=rows,
        data_manifest_sha256=digest(manifest_path),
        checkpoint=str(run_dir / f"step-{progress['step']:06d}.pt"),
        elapsed_seconds=time.perf_counter() - started,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_sha256={
            name: digest(root / name)
            for name in (
                "omni/training/r3_local.py",
                "omni/training/sft.py",
                "omni/training/losses.py",
                "omni/models/model_omni_v2.py",
                "omni/models/thinker/qwen_backend.py",
                "omni/models/talker/model_talker.py",
                "omni/models/thinker/model_thinker.py",
                "omni/data/r3_prepare.py",
                "omni/data/qwen_omni.py",
            )
        },
    )
    result["checkpoint_sha256"] = digest(Path(result["checkpoint"]))
    (run_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
