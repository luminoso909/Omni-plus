"""R3–R5 共用配置、数据调度和 SFT 入口；单步训练及完整恢复沿用 P4。"""

import hashlib
import json
import math
import platform
import random
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from omni.data.omni_dataset import OmniCollator
from omni.models.bridge import bridge_spec
from omni.models.config import OmniPlusConfig
from omni.models.model_omni_v2 import OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.run_source import check_source_root, digest, source_identity
from omni.training.sft import load_checkpoint, save_checkpoint, train_step


def merge_settings(path: Path, parents: tuple[Path, ...] = ()) -> dict:
    """显式相对路径继承；配置区合并，任务预算/权重/LR表与列表完整替换。"""
    path = path.resolve()
    if path in parents:
        raise ValueError("配置 extends 形成循环。")

    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("配置必须是 YAML 字典。")
    parent = values.pop("extends", None)
    base = merge_settings(path.parent / parent, (*parents, path)) if parent else {}

    def merge(left: dict, right: dict) -> dict:
        for key, value in right.items():
            left[key] = (
                merge(dict(left[key]), value)
                if isinstance(left.get(key), dict)
                and isinstance(value, dict)
                and key
                not in {
                    "tasks",
                    "sample_weights",
                    "task_weights",
                    "task_batch_sizes",
                    "group_lrs",
                    "bridge",
                    "weights",
                }
                else value
            )
        return left

    return merge(base, values)


def read_settings(path: Path, root: Path, *, allow_grpo: bool = False) -> dict:
    settings = merge_settings(path)
    version = settings.get("schema_version")
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
    if allow_grpo:
        allowed["root"].add("grpo")
    if version == 2:
        allowed["root"] |= {"evaluation"}
        allowed["data"] |= {"audited_manifest", "audited_manifest_sha256"}
        allowed["model"] |= {"initialization", "bridge"}
        allowed["training"] |= {
            "eval_interval",
            "save_interval",
            "patience",
            "min_delta",
            "group_lrs",
            "sample_weights",
            "task_weights",
            "task_batch_sizes",
            "length_bucket_size",
            "gradient_checkpointing",
            "allow_diagnostic_a2a",
            "max_seconds",
        }
        allowed["lora"] |= {"rank", "alpha", "targets", "dropout"}
        allowed["evaluation"] = {"max_new_tokens", "asr_dir", "benchmark"}
        settings.setdefault("evaluation", {})
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
    if version not in {1, 2}:
        raise ValueError("需要 schema_version: 1 或 2。")
    if settings["model"]["kind"] not in {"tiny", "qwen3"}:
        raise ValueError("model.kind 仅支持 tiny 或 qwen3。")
    if version == 1 and settings.get("lora", {}).get("enabled", False):
        raise ValueError("LoRA 仅预留配置；完整实现和验收在 R4-L。")
    recipes = (
        {"r4a", "projectors"}
        if version == 1
        else {"r4a", "r4b", "projectors", "vision", "audio", "joint"}
    )
    if training["recipe"] not in recipes:
        raise ValueError("未知训练 recipe。")
    if training["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("device 需要 cpu、mps 或 cuda。")
    if training["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("未知 dtype。")
    for key in ("steps", "batch_size", "accumulation_steps", "threads"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} 必须为正整数。")
    if version == 1 and training["steps"] > 10:
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
    if training["recipe"] in {"r4a", "r4b"} and set(training["tasks"]) != {"t2a"}:
        raise ValueError("R4-A 初始配方只使用 T2A；多模态投影验证使用 projectors。")
    if training["recipe"] == "projectors" and "t2a" in training["tasks"]:
        raise ValueError("Projector 验证需要 I2T、A2T 或 A2A 输入模态。")

    if version == 2:
        validate_r4(settings, root)

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
    if "bridge" in settings["model"]:
        model_config(settings, root)
    return settings


def validate_r4(settings: dict, root: Path) -> None:
    training = settings["training"]
    for key in ("eval_interval", "save_interval", "length_bucket_size"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} 必须为正整数。")
    if type(training["patience"]) is not int or training["patience"] < 0:
        raise ValueError("patience 必须为非负整数；0 表示禁用早停。")
    for key in ("min_delta", "max_seconds"):
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} 必须为有限非负数。")
    if training["gradient_checkpointing"]:
        raise ValueError("冻结 Qwen 使用 eval；未验收的 gradient checkpointing 不能启用。")
    tasks = set(training["tasks"])
    for key in ("sample_weights", "task_weights", "task_batch_sizes"):
        values = training[key]
        if set(values) - tasks or any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError(f"{key} 只能配置选定任务的有限正数。")
    if any(type(n) is not int for n in training["task_batch_sizes"].values()):
        raise ValueError("task_batch_sizes 必须为整数。")
    groups = {
        "bridge",
        "talker",
        "audio_embedding",
        "audio_heads",
        "vision_proj",
        "audio_proj",
        "lora",
    }
    if set(training["group_lrs"]) - groups or any(
        not math.isfinite(v) or v <= 0 for v in training["group_lrs"].values()
    ):
        raise ValueError("group_lrs 需要合法参数组和有限正学习率。")
    if training["recipe"] == "vision" and tasks != {"i2t"}:
        raise ValueError("vision 独立对齐只使用 I2T。")
    if training["recipe"] == "audio" and tasks != {"a2t"}:
        raise ValueError("audio 独立对齐只使用 A2T。")
    lora = settings["lora"]
    if lora.get("enabled") and training["recipe"] != "joint":
        raise ValueError("LoRA 仅在 joint 配方开启。")
    if lora.get("dropout", 0) != 0:
        raise ValueError("已验收 LoRA 的 dropout 必须为 0。")
    initial = settings["model"].get("initialization", {})
    checkpoints = {
        "talker_checkpoint",
        "components_checkpoint",
        "vision_checkpoint",
        "audio_checkpoint",
    }
    bridge_sources = {"bridge_base_checkpoint", "bridge_base_manifest", "bridge_base_lineage"}
    if set(initial) - {
        "talker",
        *checkpoints,
        *bridge_sources,
    }:
        raise ValueError("未知 initialization 配置键。")
    if initial.get("talker", "random") not in {"random", "pretrained"}:
        raise ValueError("Talker 初始化需要 random 或 pretrained。")
    if (initial.get("talker", "random") == "pretrained") != bool(initial.get("talker_checkpoint")):
        raise ValueError("pretrained Talker 需要显式兼容权重；random 不能指定它。")
    has_base = bool(initial.get("bridge_base_checkpoint"))
    if has_base != bool(initial.get("bridge_base_manifest")):
        raise ValueError("Bridge 热启动需要同时指定 Fixed Checkpoint 和对应 run Manifest。")
    has_lineage = bool(initial.get("bridge_base_lineage"))
    if (has_lineage and not has_base) or (
        has_base and settings["model"]["kind"] == "qwen3" and not has_lineage
    ):
        raise ValueError("原生 Bridge 热启动需要 B1 数据谱系 bridge_base_lineage。")
    if has_base and any(initial.get(key) for key in checkpoints):
        raise ValueError("Bridge 热启动不能与独立组件初始化混用。")
    for key in sorted(checkpoints | bridge_sources):
        if initial.get(key) and not (root / initial[key]).is_file():
            raise FileNotFoundError(root / initial[key])
    if "audited_manifest" in settings["data"]:
        if not settings["data"].get("audited_manifest_sha256"):
            raise ValueError("审计源需要 audited_manifest_sha256。")
    if not 1 <= settings["evaluation"].get("max_new_tokens", 128) <= 4096:
        raise ValueError("生成预算必须在 1–4096。")
    if "benchmark" in settings["evaluation"]:
        from omni.evaluation.bridge_benchmark import benchmark_options

        benchmark_options(settings["evaluation"])


def data_contract(settings: dict, root: Path) -> dict:
    if settings["data"].get("audited_manifest"):
        from omni.data.sft_sources import inspect_sources

        return inspect_sources(settings, root)
    from omni.data.r3_prepare import inspect_data

    return inspect_data(settings, root)


def prepare(settings: dict, root: Path) -> dict:
    if settings["data"].get("audited_manifest"):
        from omni.data.sft_sources import prepare_sources

        return prepare_sources(settings, root)
    from omni.data.r3_prepare import prepare_data

    return prepare_data(settings, root)


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
    if "bridge" in settings["model"]:
        values["bridge"] = settings["model"]["bridge"]
    config = OmniPlusConfig(**values)
    bridge_spec(config)
    return config, native


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
    from omni.training.adapters import inject_lora

    inject_lora(model.thinker.model, settings["lora"])
    if not meta:
        initialize_components(model, settings, root)
    configure_parameters(model, settings["training"]["recipe"])
    return model


def inspection_checkpoint(path: Path) -> dict:
    """预检优先 mmap；旧格式仅允许有限 CPU 回退，不改写来源文件。"""
    mapped = zipfile.is_zipfile(path)
    if not mapped and path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("旧格式 Checkpoint 超过 64 MiB；请另存为新版 torch.save 格式后预检。")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mapped)


def initialize_components(
    model: nn.Module, settings: dict, root: Path, *, validate_only: bool = False
) -> None:
    """先导入完整多模态组件，再按显式文件覆盖 Talker/两个独立 Projector。"""
    initial = settings["model"].get("initialization", {})
    for key, prefix in (
        ("components_checkpoint", ""),
        ("talker_checkpoint", "talker"),
        ("vision_checkpoint", "vision_proj"),
        ("audio_checkpoint", "audio_proj"),
    ):
        if not initial.get(key):
            continue
        payload = (
            inspection_checkpoint(root / initial[key])
            if validate_only
            else torch.load(root / initial[key], map_location="cpu", weights_only=True)
        )
        state = payload.get("model", payload)
        if prefix:
            module = model.get_submodule(prefix)
            values = {
                name.removeprefix(prefix + "."): value
                for name, value in state.items()
                if name.startswith(prefix + ".")
            }
            if not values:
                values = state
            expected = module.state_dict()
        else:
            config = payload.get("config") or payload.get("metadata", {}).get("config")
            if config is not None:
                if bridge_spec(OmniPlusConfig(**config)) != bridge_spec(model.config):
                    raise ValueError("组件 Checkpoint 的 Bridge 类型或选层不匹配。")
            elif hasattr(model.config, "bridge"):
                raise ValueError("显式 Bridge 的组件初始化需要可核验的模型配置。")
            expected = {
                name: value
                for name, value in model.state_dict().items()
                if not name.startswith("thinker.")
            }
            values = {
                name: value for name, value in state.items() if not name.startswith("thinker.")
            }
        if expected.keys() != values.keys() or any(
            expected[name].shape != values[name].shape
            or expected[name].dtype != values[name].dtype
            or (values[name].is_floating_point() and not values[name].isfinite().all())
            for name in expected
        ):
            raise ValueError(f"初始化 Checkpoint 的键、shape、dtype 或数值不兼容：{key}")

        if validate_only:
            continue
        with torch.no_grad():
            for name, value in expected.items():
                value.copy_(values[name])


def read_bridge_base(settings: dict, root: Path, contract: dict, source_sha: dict) -> dict | None:
    """在矩阵规划和权重导入前核验同一份来源和数据谱系。"""
    from omni.data.bridge_splits import check_bridge_splits

    initial = settings["model"].get("initialization", {})
    if not initial.get("bridge_base_checkpoint"):
        return None
    manifest = json.loads((root / initial["bridge_base_manifest"]).read_text())
    if manifest.get("checkpoint_sha256") != source_sha["bridge_base_checkpoint"]:
        raise ValueError("Bridge 热启动 Checkpoint 与来源 Manifest 的 SHA-256 不一致。")

    # macOS 复制到 Linux 时，AppleDouble 侧车文件可能不存在。
    def asset_hashes(assets: dict) -> dict:
        return {name: sha for name, sha in assets.items() if not Path(name).name.startswith("._")}

    if manifest.get("asset_identity") != adapter_identity(settings, contract) or asset_hashes(
        manifest.get("data_contract", {}).get("assets", {})
    ) != asset_hashes(contract["assets"]):
        raise ValueError("Bridge 热启动的 Qwen、Tokenizer 或媒体资产身份不一致。")

    payload = torch.load(
        root / initial["bridge_base_checkpoint"], map_location="cpu", weights_only=True, mmap=True
    )
    identity = payload.get("progress", {}).get("identity", {})
    source_settings = identity.get("settings")
    if (
        not source_settings
        or source_settings != manifest.get("config")
        or not identity.get("data_manifest_sha256")
        or identity["data_manifest_sha256"] != manifest.get("data_manifest_sha256")
    ):
        raise ValueError("Bridge 热启动需要与 Manifest 匹配的完整训练 Checkpoint。")

    lineage = initial.get("bridge_base_lineage")
    check_bridge_splits(
        contract,
        identity,
        source_sha["bridge_base_checkpoint"],
        root / lineage if lineage else None,
        formal=settings["model"]["kind"] == "qwen3",
        source_contract=manifest["data_contract"],
    )
    return payload


def initialize_bridge_base(
    model: OmniPlusOmni,
    settings: dict,
    root: Path,
    contract: dict,
    source_sha: dict,
    *,
    validate_only: bool = False,
) -> None:
    """从同一个 Fixed 训练结果导入共享权重，融合参数保留本候选初始化。"""
    payload = read_bridge_base(settings, root, contract, source_sha)
    if payload is None:
        return
    source_settings = payload["progress"]["identity"]["settings"]
    source = dict(payload["config"])
    target = model.config.to_dict()
    source_bridge = bridge_spec(OmniPlusConfig(**source))
    if source_bridge["kind"] != "fixed":
        raise ValueError("Bridge 热启动只接受 Fixed 来源。")
    target_bridge = bridge_spec(model.config)
    if target_bridge["kind"] == "fixed" and target_bridge != source_bridge:
        raise ValueError("Fixed 对照组必须沿用来源 Checkpoint 的选层。")
    for config in (source, target):
        config.pop("bridge", None)
        config.pop("bridge_layer", None)
    if source != target:
        raise ValueError("Bridge 热启动的非 Bridge 模型配置不一致。")
    source_lora, target_lora = source_settings["lora"], settings["lora"]
    if source_lora.get("enabled", False) != target_lora.get("enabled", False) or (
        source_lora.get("enabled", False) and source_lora != target_lora
    ):
        raise ValueError("Bridge 热启动的 LoRA 配置不一致。")

    expected, values = model.state_dict(), payload["model"]
    shared = {
        name
        for name in expected
        if not name.startswith("bridge.") or name.startswith(("bridge.proj.", "bridge.norm."))
    }
    if shared != values.keys() or any(
        expected[name].shape != value.shape
        or expected[name].dtype != value.dtype
        or (value.is_floating_point() and not value.isfinite().all())
        for name, value in values.items()
    ):
        raise ValueError("Bridge 热启动的共享权重键、shape、dtype 或数值不兼容。")

    if not validate_only:
        model.load_state_dict(expected | values, strict=True)


def parameter_group(name: str) -> str:
    if ".lora_a." in name or ".lora_b." in name:
        return "lora"
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
    elif recipe in {"vision", "audio"}:
        active = {recipe + "_proj"}
    elif recipe == "joint":
        active |= {"vision_proj", "audio_proj", "lora"}
    elif recipe not in {"r4a", "r4b"}:
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
    params = [param for param in model.parameters() if param.requires_grad]
    if settings.get("schema_version") == 2:
        groups = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                groups.setdefault(parameter_group(name), []).append(param)
        params = [
            dict(
                params=values,
                name=name,
                lr=training["group_lrs"].get(name, training["learning_rate"]),
            )
            for name, values in groups.items()
        ]
    optimizer = torch.optim.AdamW(
        params,
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


def run_identity(settings: dict, root: Path) -> dict:
    check_source_root(root)

    bridge_identity = {}
    if "bridge" in settings["model"] or settings["model"].get("initialization", {}).get(
        "bridge_base_checkpoint"
    ):
        bridge_identity["model_config_sha256"] = digest(root / settings["model"]["config"])
    return dict(
        settings=settings,
        data_manifest_sha256=digest(root / settings["data"]["cache_dir"] / "manifest.json"),
        initialization_sha256={
            name: digest(root / value)
            for name, value in settings["model"].get("initialization", {}).items()
            if (
                name.endswith("_checkpoint")
                or name in {"bridge_base_manifest", "bridge_base_lineage"}
            )
            and value
        },
        runtime=dict(torch=str(torch.__version__), transformers=transformers.__version__),
        **source_identity(root),
        **bridge_identity,
    )


def adapter_identity(settings: dict, contract: dict) -> dict:
    """独立 Adapter 绑定基座与 tokenizer；tiny 还绑定随机基座的构造 seed。"""
    return dict(
        qwen_revision=contract["qwen_revision"],
        qwen_assets={
            name: value for name, value in contract["assets"].items() if name.startswith("qwen/")
        },
        model_kind=settings["model"]["kind"],
        tiny=settings["model"].get("tiny") if settings["model"]["kind"] == "tiny" else None,
        seed=settings["training"]["seed"] if settings["model"]["kind"] == "tiny" else None,
    )


def next_batches(samples: list[dict], progress: dict, training: dict, collate) -> tuple[list, list]:
    """平滑加权任务轮转 + 任务内长度分桶；所有游标都放进 Checkpoint。"""
    tasks = training["tasks"]
    state = progress.setdefault(
        "sampler", {task: dict(score=0.0, epoch=0, cursor=0) for task in tasks}
    )
    weights = {task: training.get("sample_weights", {}).get(task, 1.0) for task in tasks}
    batches, selected = [], []
    for _ in range(training["accumulation_steps"]):
        for task in tasks:
            state[task]["score"] += weights[task]
        task = max(tasks, key=lambda name: state[name]["score"])
        state[task]["score"] -= sum(weights.values())
        pool = [row for row in samples if row["task"] == task]
        if not pool:
            raise ValueError(f"训练缓存缺少任务：{task}")
        cursor = state[task]
        if cursor["cursor"] >= len(pool):
            cursor["epoch"], cursor["cursor"] = cursor["epoch"] + 1, 0
        seed = training["seed"] + tasks.index(task) * 100003 + cursor["epoch"]
        order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)).tolist()
        size = training.get("task_batch_sizes", {}).get(task, training["batch_size"])
        bucket = size * training.get("length_bucket_size", 1)
        order = [
            index
            for start in range(0, len(order), bucket)
            for index in sorted(
                order[start : start + bucket], key=lambda i: len(pool[i]["input_ids"])
            )
        ]
        indices = order[cursor["cursor"] : cursor["cursor"] + size]
        cursor["cursor"] += len(indices)
        batches.append(collate([pool[index] for index in indices]))
        selected.append(task)
    return batches, selected


def task_evaluation(model: nn.Module, samples: list[dict], training: dict) -> dict:
    from omni.evaluation.sft import evaluate_sft

    collate = OmniCollator(model.config)
    result = {}
    for task in training["tasks"]:
        rows = [row for row in samples if row["task"] == task]
        if not rows:
            raise ValueError(f"验证缓存缺少任务：{task}")
        size = training.get("task_batch_sizes", {}).get(task, training["batch_size"])
        batches = [collate(rows[start : start + size]) for start in range(0, len(rows), size)]
        result[task] = evaluate_sft(model, batches, training["device"], training["stop_weight"])
    weights = training.get("task_weights", {})
    total = sum(weights.get(task, 1.0) for task in result)
    return dict(
        tasks=result,
        total=sum(weights.get(task, 1.0) * value["total"] for task, value in result.items())
        / total,
    )


def check_device(training: dict) -> None:
    device = training["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 尚未在本机验收；需要实际可用的 GPU。")
    if device != "cuda" and training["dtype"] != "float32":
        raise ValueError("本地验收使用 FP32；混合精度留给 CUDA Profile。")


def check_task_semantics(settings: dict, manifest: dict) -> None:
    if "a2a" not in settings["training"]["tasks"] or settings["schema_version"] == 1:
        return
    if settings["training"]["allow_diagnostic_a2a"]:
        return
    if not settings["data"].get("audited_manifest"):
        raise ValueError("旧缓存 A2A 仅为 echo 诊断；正式联合训练需要可信 QA 目标语音。")
    rows = [row for split in manifest.get("selections", {}).values() for row in split]
    qa = [row for row in rows if row["task"] == "a2a"]
    if not qa or any(row.get("a2a_kind") != "qa" for row in qa):
        raise ValueError("问答式 A2A 必须提供逐条已审计 QA 目标，不能复用 echo。")


def inspect_sft_run(
    model: nn.Module,
    settings: dict,
    root: Path,
    contract: dict,
    *,
    command: str,
    resume: Path | None,
    target: int,
) -> dict:
    """以 meta 模型核对实际数据/初始化/恢复门槛，不前向、不恢复 RNG 或写产物。"""
    from omni.data.r3_prepare import load_prepared
    from omni.evaluation.bridge_benchmark import benchmark_options

    evaluation = command in {"evaluate", "demo"}
    options = benchmark_options(settings.get("evaluation", {}))
    split = (options["split"] if options else "validation") if evaluation else "train"
    directory = root / settings["data"]["cache_dir"]
    samples = load_prepared(directory, split, contract)
    required = [split] + (
        ["validation"] if not evaluation and settings["schema_version"] == 2 else []
    )
    for name in required:
        rows = samples if name == split else load_prepared(directory, name, contract)
        if set(settings["training"]["tasks"]) - {row["task"] for row in rows}:
            raise ValueError(f"{name} 缓存缺少配置要求的任务。")
    check_task_semantics(settings, json.loads((directory / "manifest.json").read_text()))
    if command == "demo":
        count = options["t2a_samples"] if options else 1
        if sum(row["task"] == "t2a" for row in samples) < count:
            raise ValueError("音频 demo 的实际评测 split 缺少足够 T2A 样本。")

    # 复用热启动的资产、谱系、键、形状及数值检查；meta 参数不写入权重。
    identity = run_identity(settings, root)
    initialize_components(model, settings, root, validate_only=True)
    if resume is None:
        initialize_bridge_base(
            model, settings, root, contract, identity["initialization_sha256"], validate_only=True
        )
        return dict(split=split, samples=len(samples), checkpoint_step=None)

    payload = inspection_checkpoint(resume)
    if not isinstance(payload, dict) or not {"model", "config", "progress"} <= payload.keys():
        raise ValueError("需要包含 model、config 和 progress 的完整 Checkpoint。")
    progress = payload["progress"]
    if not isinstance(progress, dict) or progress.get("identity") != identity:
        raise ValueError("预检配置、代码或数据与 Checkpoint 不一致。")
    if payload["config"] != model.config.to_dict():
        raise ValueError("预检模型配置与 Checkpoint 不一致。")
    state, expected = payload["model"], model.state_dict()
    if (
        not isinstance(state, dict)
        or state.keys() != expected.keys()
        or any(
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or (value.is_floating_point() and not value.isfinite().all())
            for name, value in state.items()
        )
    ):
        raise ValueError("Checkpoint 参数键、shape、dtype 或数值不兼容。")
    step = progress.get("step")
    if type(step) is not int or step < 0 or (not evaluation and step >= target):
        raise ValueError("Checkpoint step 无效，恢复目标必须大于已保存 step。")
    if not evaluation:
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        if not {"optimizer", "scaler", "rng"} <= payload.keys() or "scheduler" not in progress:
            raise ValueError("完整恢复需要 Optimizer、Scaler、RNG 和 Scheduler 状态。")
        if payload.get("trainable") != trainable:
            raise ValueError("恢复时冻结策略变化，不能沿用旧优化器状态。")
        check_resume_state(payload, model, settings)
    return dict(split=split, samples=len(samples), checkpoint_step=step)


def check_resume_state(payload: dict, model: nn.Module, settings: dict) -> None:
    """检查恢复状态结构；只使用 meta 优化器及私有 CPU RNG，不改变全局随机状态。"""
    optimizer, scheduler, _, _ = optimizer_state(model, settings)
    saved = payload["optimizer"]
    groups = optimizer.state_dict()["param_groups"]
    if (
        not isinstance(saved, dict)
        or not isinstance(saved.get("state"), dict)
        or not isinstance(saved.get("param_groups"), list)
        or len(saved["param_groups"]) != len(groups)
        or any(
            not isinstance(old, dict)
            or not old.keys() >= new.keys()
            or not isinstance(old["params"], list)
            or len(old["params"]) != len(new["params"])
            for old, new in zip(saved["param_groups"], groups, strict=True)
        )
    ):
        raise ValueError("Optimizer state/param_groups 结构或参数分组不兼容。")

    state = payload["progress"]["scheduler"]
    if (
        not isinstance(state, dict)
        or not state.keys() >= scheduler.state_dict().keys()
        or type(state["last_epoch"]) is not int
        or any(
            not isinstance(state[key], list) or len(state[key]) != len(groups)
            for key in ("base_lrs", "_last_lr", "lr_lambdas")
        )
        or not isinstance(payload["scaler"], dict)
    ):
        raise ValueError("Scheduler/Scaler 状态结构不兼容。")

    # CPU 随机状态由私有实例验证；设备 RNG 只核对基本张量结构，设备恢复留给实测。
    rng = payload["rng"]
    try:
        random.Random().setstate(rng["python"])
        numpy = rng["numpy"]
        np.random.RandomState().set_state(
            (numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:])
        )
        torch.Generator(device="cpu").set_state(rng["torch"])
        device_states = ([rng["mps"]] if rng["mps"] is not None else []) + (
            rng["cuda"] if rng["cuda"] is not None else []
        )
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or not value.numel()
            for value in device_states
        ):
            raise ValueError("设备 RNG 必须为非空 ByteTensor。")
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("RNG 状态结构不兼容。") from error


def loss_curve(path: Path, rows: list[dict], evaluations: list[dict]) -> None:
    """不增加绘图库依赖；输出可直接浏览的真实 Loss 曲线。"""
    series = [
        ("train", "#2166ac", [(row["step"], row["total"]) for row in rows]),
        ("validation", "#b2182b", [(row["step"], row["total"]) for row in evaluations]),
    ]
    points = [point for _, _, values in series for point in values]
    if not points:
        return
    xmax = max(x for x, _ in points) or 1
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    span = ymax - ymin or 1.0
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">',
        '<rect width="640" height="320" fill="white"/>',
        '<path d="M55 20V275H620" fill="none" stroke="black"/>',
        '<text x="280" y="310">optimizer step</text>',
    ]
    for index, (label, color, values) in enumerate(series):
        coords = " ".join(
            f"{55 + x / xmax * 555:.1f},{260 - (y - ymin) / span * 220:.1f}" for x, y in values
        )
        svg.append(f'<polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"/>')
        svg.append(f'<text x="{350 + index * 120}" y="18" fill="{color}">{label}</text>')
    svg.extend(
        [
            f'<text x="2" y="45">{ymax:.3f}</text>',
            f'<text x="2" y="265">{ymin:.3f}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n")


def smoke(
    settings: dict,
    root: Path,
    run_dir: Path,
    *,
    resume: Path | None = None,
    steps: int | None = None,
    command: str = "smoke",
) -> dict:
    from omni.data.r3_prepare import load_prepared
    from omni.training.adapters import export_adapter

    training = settings["training"]
    modern = settings["schema_version"] == 2
    device = training["device"]
    check_device(training)
    target = training["steps"] if steps is None else steps
    if not 0 < target <= training["steps"]:
        raise ValueError("--steps 必须在配置步数预算内。")
    if (command == "smoke" and target > 10) or (command == "profile" and target > 100):
        raise ValueError("smoke 最多 10 步；profile 最多 100 步。")
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖运行目录：{run_dir}")

    # 先验证缓存及恢复身份，再创建模型。历史测试集不用于训练或选择模型。
    contract = data_contract(settings, root)
    cache_dir = root / settings["data"]["cache_dir"]
    samples = load_prepared(cache_dir, "train", contract)
    validation = load_prepared(cache_dir, "validation", contract) if modern else []
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    check_task_semantics(settings, manifest)
    identity = run_identity(settings, root)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["progress"]["identity"] != identity:
            raise ValueError("Resume 配置、代码或数据版本不一致。")
        if checkpoint["progress"]["step"] >= target:
            raise ValueError("目标 step 必须大于恢复 step。")
        del checkpoint

    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(training["threads"])
    started = time.perf_counter()
    model = build_model(settings, root)
    if resume is None:
        initialize_bridge_base(model, settings, root, contract, identity["initialization_sha256"])
    model.to(device)
    groups = configure_parameters(model, training["recipe"])
    optimizer, scheduler, scaler, dtype = optimizer_state(model, settings)
    progress = dict(step=0, epoch=0, cursor=0, identity=identity)
    if modern:
        progress.update(best_loss=float("inf"), bad_evaluations=0, best_step=None, evaluations=[])
    if resume:
        progress = load_checkpoint(resume, model, optimizer, scaler)
        scheduler.load_state_dict(progress["scheduler"])
    chosen = [row for row in samples if row["task"] in training["tasks"]]
    if set(row["task"] for row in chosen) != set(training["tasks"]):
        raise ValueError("缓存缺少配置要求的训练任务。")
    collate = OmniCollator(model.config)
    frozen_before = parameter_digest(model, frozen=True)
    trainable_before = parameter_digest(model, frozen=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False)
    )
    rows, gradient_groups, bridge_gradients = [], {}, {}
    start_step = progress["step"]
    stop_reason = "step_budget"
    # 固定训练探针用于观察过拟合；不把不同随机 batch 的 Loss 当作下降证据。
    probe = [next(row for row in chosen if row["task"] == task) for task in training["tasks"]]
    initial_probe = task_evaluation(model, probe, training) if modern else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    loop_started = time.perf_counter()
    while progress["step"] < target:
        if modern:
            batches, tasks = next_batches(chosen, progress, training, collate)
        else:
            batches, tasks = [], None
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
            tasks,
            {task: training.get("task_weights", {}).get(task, 1.0) for task in training["tasks"]},
        )
        if not all(math.isfinite(value) for value in metrics.values()) or metrics["grad_norm"] <= 0:
            raise RuntimeError("训练步骤需要有限损失和非零梯度。")
        if metrics["optimizer_updated"] != 1:
            raise RuntimeError("优化器没有更新；保存上一有效 Checkpoint 后检查 AMP。")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not param.requires_grad or not param.grad.isfinite().all():
                    raise RuntimeError("冻结参数收到梯度或梯度非有限。")
                group = parameter_group(name)
                grad = float(param.grad.abs().max())
                gradient_groups[group] = max(gradient_groups.get(group, 0), grad)
                if group == "bridge":
                    bridge_gradients[name] = max(bridge_gradients.get(name, 0), grad)
        scheduler.step()
        progress["step"] += 1
        progress["scheduler"] = scheduler.state_dict()
        row = dict(
            step=progress["step"], learning_rate=scheduler.get_last_lr()[0], tasks=tasks, **metrics
        )
        rows.append(row)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        # 验证与最佳状态都在 optimizer 边界；恢复沿用 patience 和调度器进度。
        if modern and progress["step"] % training["eval_interval"] == 0:
            scores = task_evaluation(model, validation, training)
            if not math.isfinite(scores["total"]):
                raise RuntimeError("验证 Loss 非有限。")
            progress["evaluations"].append(dict(step=progress["step"], **scores))
            if scores["total"] < progress["best_loss"] - training["min_delta"]:
                progress.update(
                    best_loss=scores["total"], bad_evaluations=0, best_step=progress["step"]
                )
                save_checkpoint(
                    run_dir / f"best-{progress['step']:06d}.pt", model, optimizer, scaler, progress
                )
            else:
                progress["bad_evaluations"] += 1
        if modern and progress["step"] % training["save_interval"] == 0:
            save_checkpoint(
                run_dir / f"step-{progress['step']:06d}.pt", model, optimizer, scaler, progress
            )
        if modern and training["patience"] and progress["bad_evaluations"] >= training["patience"]:
            stop_reason = "early_stopping"
            break
        if (
            modern
            and training["max_seconds"]
            and time.perf_counter() - loop_started >= training["max_seconds"]
        ):
            stop_reason = "time_budget"
            break
    if device == "cuda":
        torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - loop_started

    frozen_after = parameter_digest(model, frozen=True)
    trainable_after = parameter_digest(model, frozen=False)
    if frozen_before != frozen_after or trainable_before == trainable_after:
        raise RuntimeError("冻结参数变化或目标参数未更新。")
    checkpoint_path = run_dir / f"step-{progress['step']:06d}.pt"
    if not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, model, optimizer, scaler, progress)
    result = dict(
        status="passed",
        scope="本地/显式运行的 SFT 工程验证；不代表正式 G4 或音质通过",
        command=command,
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
        stop_reason=stop_reason,
        resume_from=str(resume) if resume else None,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        trainable_before=trainable_before,
        trainable_after=trainable_after,
        gradient_groups=gradient_groups,
        metrics=rows,
        data_manifest_sha256=identity["data_manifest_sha256"],
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest(checkpoint_path),
        elapsed_seconds=time.perf_counter() - started,
        loop_seconds=loop_seconds,
        seconds_per_step=loop_seconds / len(rows),
        timing_scope="optimizer loop including validation and checkpoint IO",
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_schema_version=identity["source_schema_version"],
        source_sha256=identity["source_sha256"],
        runtime=identity["runtime"],
        asset_identity=adapter_identity(settings, contract),
        data_contract=contract,
    )
    if "model_config_sha256" in identity:
        from omni.evaluation.bridge_benchmark import hardware_info

        result.update(
            scope="Bridge 训练运行记录，阶段验收与质量结论按独立评测确认",
            bridge=bridge_spec(model.config),
            bridge_gradient_max_abs=bridge_gradients,
            initialization_sha256=identity["initialization_sha256"],
            model_config_sha256=identity["model_config_sha256"],
            hardware=hardware_info(device),
        )
    if modern:
        result.update(
            initial_train_probe=initial_probe,
            final_train_probe=task_evaluation(model, probe, training),
            validation=task_evaluation(model, validation, training),
            evaluations=progress["evaluations"],
            best_step=progress["best_step"],
            best_validation_loss=progress["best_loss"] if progress["best_step"] else None,
        )
        export_adapter(run_dir / "adapter.pt", model, adapter_identity(settings, contract))
        result["adapter"] = str(run_dir / "adapter.pt")
        loss_curve(run_dir / "loss.svg", rows, progress["evaluations"])
    if device == "cuda":
        result["cuda"] = dict(
            device=torch.cuda.get_device_name(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    (run_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def evaluate_run(
    settings: dict, root: Path, run_dir: Path, checkpoint: Path, *, demo: bool = False
) -> dict:
    from omni.data.r3_prepare import load_prepared
    from omni.evaluation.bridge_benchmark import (
        asr_identity,
        benchmark_audio,
        benchmark_options,
        hardware_info,
    )
    from omni.evaluation.sft import generate_audio, load_local_asr, modality_swap
    from omni.training.sft import move_batch

    if run_dir.exists():
        raise FileExistsError(run_dir)
    check_device(settings["training"])
    contract = data_contract(settings, root)
    cache_dir = root / settings["data"]["cache_dir"]
    evaluation = settings.get("evaluation", {})
    benchmark = benchmark_options(evaluation)
    split = benchmark["split"] if benchmark else "validation"
    samples = load_prepared(cache_dir, split, contract)
    identity = run_identity(settings, root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["progress"]["identity"] != identity:
        raise ValueError("评估配置、代码或数据与 Checkpoint 不一致。")
    if benchmark:
        torch.set_num_threads(settings["training"]["threads"])
    torch.manual_seed(settings["training"]["seed"])
    model = build_model(settings, root).to(settings["training"]["device"])
    if payload["config"] != model.config.to_dict():
        raise ValueError("评估模型配置与 Checkpoint 不一致。")
    model.load_state_dict(payload["model"], strict=True)
    payload_step = payload["progress"]["step"]
    del payload
    run_dir.mkdir(parents=True, exist_ok=False)
    result = dict(
        checkpoint=str(checkpoint),
        split=split,
        metrics=task_evaluation(model, samples, settings["training"]),
        modality_swap={},
    )
    if benchmark:
        result.update(
            config=settings,
            checkpoint_sha256=digest(checkpoint),
            checkpoint_step=payload_step,
            data_manifest_sha256=identity["data_manifest_sha256"],
            initialization_sha256=identity["initialization_sha256"],
            model_config_sha256=identity.get("model_config_sha256"),
            source_schema_version=identity["source_schema_version"],
            source_sha256=identity["source_sha256"],
            runtime=identity["runtime"],
            hardware=hardware_info(settings["training"]["device"]),
            asset_identity=adapter_identity(settings, contract),
        )
    collate = OmniCollator(model.config)
    for task, kind in (("i2t", "image"), ("a2t", "audio")):
        rows = [row for row in samples if row["task"] == task]
        if task in settings["training"]["tasks"] and rows:
            result["modality_swap"][kind] = modality_swap(
                model, move_batch(collate(rows[:2]), settings["training"]["device"]), kind
            )
    if demo:
        from transformers import AutoTokenizer, MimiModel

        from omni.inference.codec import MimiCodec

        tokenizer = AutoTokenizer.from_pretrained(
            root / settings["assets"]["qwen_dir"], local_files_only=True
        )
        codec = MimiCodec(
            MimiModel.from_pretrained(
                root / settings["assets"]["reference_root"] / "model/mimi", local_files_only=True
            )
        )
        asr_path = root / evaluation["asr_dir"] if evaluation.get("asr_dir") else None
        max_new_tokens = evaluation.get("max_new_tokens", 128)
        if benchmark:
            result["asr_assets"] = asr_identity(asr_path)
        asr = load_local_asr(asr_path) if asr_path else None
        if benchmark:
            selections = json.loads((cache_dir / "manifest.json").read_text())["selections"][split]
            result["benchmark"] = benchmark_audio(
                model,
                tokenizer,
                samples,
                selections,
                codec,
                run_dir / "audio",
                benchmark["t2a_samples"],
                max_new_tokens,
                asr,
            )
        else:
            sample = next((row for row in samples if row["task"] == "t2a"), None)
            if sample is None:
                raise ValueError("音频 demo 需要验证集 T2A 样本。")
            result["generation"] = generate_audio(
                model,
                tokenizer,
                sample,
                codec,
                run_dir / "audio",
                max_new_tokens,
                asr,
            )
    (run_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result
