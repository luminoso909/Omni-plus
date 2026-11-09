"""R6 单设备 GRPO；沿用 SFT 的模型、采样游标、Optimizer 和完整 Checkpoint。"""

import contextlib
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from omni.data.r3_prepare import load_prepared
from omni.training import r3_local
from omni.training.adapters import export_adapter
from omni.training.run_source import check_source_root, digest
from omni.training.sft import load_checkpoint, save_checkpoint

GROUPS = {"lora", "bridge", "talker", "audio_embedding", "audio_heads", "vision_proj", "audio_proj"}


def read_grpo_settings(path: Path, root: Path) -> dict:
    settings = r3_local.read_settings(path, root, allow_grpo=True)
    validate_grpo(settings, root)
    return settings


def validate_grpo(settings: dict, root: Path) -> None:
    from omni.training.grpo_rewards import validate_rewards

    options, training = settings.get("grpo", {}), settings["training"]
    keys = {
        "schema_version",
        "mode",
        "group_size",
        "max_new_tokens",
        "temperature",
        "audio_temperature",
        "top_k",
        "top_p",
        "use_cache",
        "clip",
        "kl_coef",
        "advantage_epsilon",
        "normalization",
        "update_epochs",
        "trainable",
        "policy",
        "reference",
        "reward",
        "evaluation_split",
        "evaluation_prompts",
    }
    if set(options) != keys or options["schema_version"] != 1:
        raise ValueError("GRPO 需要完整且无未知键的 schema_version: 1 配置。")
    mode = options["mode"]
    if mode not in {"text", "audio", "joint"}:
        raise ValueError("GRPO mode 需要 text、audio 或 joint。")
    for key in ("group_size", "max_new_tokens", "update_epochs", "evaluation_prompts"):
        if type(options[key]) is not int or options[key] < (2 if key == "group_size" else 1):
            raise ValueError(f"grpo.{key} 需要有效正整数。")
    if mode != "text" and options["max_new_tokens"] < 2:
        raise ValueError("音频 GRPO 至少需要两个物理步骤以启动首路声码。")
    for key in ("temperature", "audio_temperature", "advantage_epsilon"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"grpo.{key} 必须为有限正数。")
    if options["top_k"] != 0 or options["top_p"] != 1:
        raise ValueError("GRPO 使用完整合法动作分布，需要 top_k=0、top_p=1。")
    if type(options["use_cache"]) is not bool:
        raise ValueError("use_cache 必须是布尔值。")
    if (
        not 0 < options["clip"] < 1
        or not math.isfinite(options["kl_coef"])
        or options["kl_coef"] < 0
    ):
        raise ValueError("GRPO clip 必须在 (0,1)，kl_coef 必须有限非负。")
    if options["normalization"] not in {"sequence_mean", "token_mean"}:
        raise ValueError("未知 GRPO Loss 归一化。")
    if training["steps"] % options["update_epochs"]:
        raise ValueError("steps 必须整除 update_epochs，在完整 rollout 更新边界保存。")
    if training.get("gradient_checkpointing"):
        raise ValueError("GRPO 使用 eval 固定策略，尚不支持 gradient_checkpointing。")
    active = options["trainable"]
    if (
        not isinstance(active, list)
        or not active
        or len(set(active)) != len(active)
        or set(active) - GROUPS
    ):
        raise ValueError("GRPO trainable 必须为不重复的合法参数组。")
    if "lora" in active and not settings["lora"].get("enabled"):
        raise ValueError("更新 lora 需要启用 Qwen LoRA。")
    if mode == "text" and set(active) - {"lora", "vision_proj", "audio_proj"}:
        raise ValueError("Text GRPO 仅更新文本策略 LoRA 或输入 Projector。")
    if mode == "audio" and set(active) - {"bridge", "talker", "audio_embedding", "audio_heads"}:
        raise ValueError("Audio GRPO 必须冻结文本策略及输入 Projector，保证文本概率抵消。")
    if options["reference"] != {"source": "policy_initial"}:
        raise ValueError("当前 Reference 固定为同一初始 SFT 策略，不接受其它来源。")
    validate_rewards(options["reward"])
    if settings["model"]["kind"] == "qwen3" and set(options["reward"].get("weights", {})) & {
        "token_value",
        "audio_value",
    }:
        raise ValueError("诊断 Token/Code 奖励仅限 Tiny，正式 G6 需要经审定的 Reward。")
    if options["evaluation_split"] not in {"validation", "test"}:
        raise ValueError("GRPO 评估只能显式使用 validation 或 test。")
    policy = options["policy"]
    if set(policy) != {"source", "checkpoint", "manifest", "lineage"}:
        raise ValueError("policy 需要 source、checkpoint、manifest、lineage。")
    initial = settings["model"].get("initialization", {})
    if (
        any(value for name, value in initial.items() if name != "talker")
        or initial.get("talker", "random") != "random"
    ):
        raise ValueError("GRPO 初始化只使用显式 policy 来源，不能混入组件热启动。")
    if policy["source"] == "tiny_random":
        if settings["model"]["kind"] != "tiny" or any(
            policy[k] for k in ("checkpoint", "manifest", "lineage")
        ):
            raise ValueError("tiny_random 仅用于 Tiny，且不应配置 SFT 来源文件。")
    elif policy["source"] == "sft":
        for key in ("checkpoint", "manifest", "lineage"):
            if not policy[key] and (key != "lineage" or settings["model"]["kind"] == "qwen3"):
                raise ValueError("正式 GRPO 需要完整 B1 Checkpoint、Manifest 和数据谱系。")
            if policy[key] and not (root / policy[key]).is_file():
                raise FileNotFoundError(root / policy[key])
    else:
        raise ValueError("policy.source 需要 tiny_random 或 sft。")


def configure_grpo(model: nn.Module, active: list[str]) -> dict:
    groups = {}
    for name, param in model.named_parameters():
        group = r3_local.parameter_group(name)
        param.requires_grad_(group in active)
        counts = groups.setdefault(group, dict(total=0, trainable=0))
        counts["total"] += param.numel()
        counts["trainable"] += param.numel() if param.requires_grad else 0
    if any(not groups.get(name, {}).get("trainable") for name in active):
        raise ValueError("GRPO 指定的参数组没有可训练参数。")
    # 关闭 dropout 保证行为策略与重算策略一致；eval 不关闭 autograd。
    model.eval()
    return groups


def prompt_from_sample(sample: dict, device: str) -> dict:
    valid = sample["labels"].ne(-100).nonzero().flatten()
    if not len(valid):
        raise ValueError("RL 样本缺少可确定 generation prefix 的文本监督。")
    length = int(valid[0]) + 1
    prompt = {
        "input_ids": sample["input_ids"][:length].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"][:length].unsqueeze(0).to(device),
        "audio_ids": sample["audio_ids"][:, :length].unsqueeze(0).to(device),
    }
    for name in ("image_features", "audio_features"):
        value = sample.get(name)
        prompt[name] = [value.to(device) if value is not None else None]
    speaker = sample.get("spk_emb")
    prompt["spk_emb"] = speaker.unsqueeze(0).to(device) if speaker is not None else None
    return prompt


@contextlib.contextmanager
def reference_policy(model: nn.Module, reference: dict):
    """只暂换会更新的参数；须在当前策略计算图创建前完成参考前向。"""
    params = dict(model.named_parameters())
    expected = {name for name, value in params.items() if value.requires_grad}
    if reference.keys() != expected:
        raise ValueError("Reference 快照参数组与当前可训练范围不匹配。")
    current = {name: params[name].detach().cpu().clone() for name in reference}
    try:
        with torch.no_grad():
            for name, value in reference.items():
                if (
                    value.shape != params[name].shape
                    or value.dtype != params[name].dtype
                    or not value.isfinite().all()
                ):
                    raise ValueError("Reference 参数形状、dtype 或数值不兼容。")
                params[name].copy_(value.to(params[name].device))
        yield
    finally:
        with torch.no_grad():
            for name, value in current.items():
                params[name].copy_(value.to(params[name].device))


def policy_source(settings: dict, root: Path, contract: dict) -> dict | None:
    """复用 R5 完整来源与媒体隔离检查；GRPO 不更换 Bridge 架构。"""
    policy = settings["grpo"]["policy"]
    if policy["source"] == "tiny_random":
        return None
    values = copy.deepcopy(settings)
    values["model"]["initialization"] = {
        "bridge_base_checkpoint": policy["checkpoint"],
        "bridge_base_manifest": policy["manifest"],
        "bridge_base_lineage": policy["lineage"],
    }
    payload = r3_local.read_bridge_base(
        values, root, contract, {"bridge_base_checkpoint": digest(root / policy["checkpoint"])}
    )
    progress = payload["progress"]
    if progress.get("algorithm") == "grpo-v1" or "grpo" in progress["identity"]["settings"]:
        raise ValueError("GRPO 初始策略必须来自 SFT；RL Checkpoint 请使用 resume。")
    return payload


def run_identity(settings: dict, root: Path) -> dict:
    identity = r3_local.run_identity(settings, root)
    identity["grpo_schema_version"] = 1
    identity["policy_source_sha256"] = {
        key: digest(root / value)
        for key, value in settings["grpo"]["policy"].items()
        if key != "source" and value
    }
    return identity


def prepared_samples(settings: dict, root: Path, split: str) -> tuple[dict, list[dict]]:
    contract = r3_local.data_contract(settings, root)
    directory = root / settings["data"]["cache_dir"]
    samples = load_prepared(directory, split, contract)
    manifest = json.loads((directory / "manifest.json").read_text())
    r3_local.check_task_semantics(settings, manifest)
    records = manifest["selections"][split]
    if len(samples) != len(records):
        raise ValueError("缓存样本与来源记录数量不一致。")
    chosen = [
        row | {"record_id": record["record_id"], "provenance": record}
        for row, record in zip(samples, records, strict=True)
        if row["task"] in settings["training"]["tasks"]
    ]
    if set(row["task"] for row in chosen) != set(settings["training"]["tasks"]):
        raise ValueError("RL 缓存缺少配置要求的任务。")
    if settings["model"]["kind"] == "qwen3" and "a2a" in settings["training"]["tasks"]:
        if any(row["provenance"].get("a2a_kind") != "qa" for row in chosen if row["task"] == "a2a"):
            raise ValueError("正式 A2A Reward 需要经审计的 QA 目标，echo 仅供 Tiny 诊断。")
    return contract, chosen


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def setup_model(settings: dict, root: Path, contract: dict, *, meta: bool = False):
    payload = policy_source(settings, root, contract)
    model = r3_local.build_model(settings, root, meta=meta)
    if payload is not None:
        if payload["config"] != model.config.to_dict():
            raise ValueError("GRPO SFT 来源的模型或 Bridge 配置不一致。")
        if payload["progress"]["identity"]["settings"].get("lora") != settings["lora"]:
            raise ValueError("GRPO SFT 来源的 LoRA 配置不一致。")
        expected, saved = model.state_dict(), payload["model"]
        if expected.keys() != saved.keys() or any(
            expected[name].shape != saved[name].shape
            or expected[name].dtype != saved[name].dtype
            or not saved[name].isfinite().all()
            for name in expected
        ):
            raise ValueError("GRPO SFT 参数键、shape、dtype 或数值不兼容。")
        if not meta:
            model.load_state_dict(payload["model"], strict=True)
    groups = configure_grpo(model, settings["grpo"]["trainable"])
    return model, groups


def prepare_run(
    settings: dict,
    root: Path,
    *,
    command: str,
    resume: Path | None,
    steps: int | None,
    baseline: bool,
    run_dir: Path | None,
) -> tuple[bool, int, dict, list[dict], dict]:
    """Dry run 与执行共用预算、数据 Split 和恢复身份检查。"""
    check_source_root(root)
    validate_grpo(settings, root)
    training, options = settings["training"], settings["grpo"]
    if command not in {"grpo-smoke", "grpo-train", "grpo-resume", "grpo-evaluate", "grpo-profile"}:
        raise ValueError("未知 GRPO 命令。")
    evaluate = command == "grpo-evaluate"
    target = training["steps"] if steps is None else steps
    epochs = options["update_epochs"]
    if not 0 < target <= training["steps"] or target % epochs:
        raise ValueError("目标 steps 必须处于预算内且位于完整 rollout 更新边界。")
    if (command == "grpo-smoke" and target > 10) or (command == "grpo-profile" and target > 100):
        raise ValueError("本地 smoke 最多 10 步，显式 profile 最多 100 步。")
    if evaluate and steps is not None:
        raise ValueError("GRPO evaluate 不接受 steps。")
    if run_dir is not None and run_dir.exists():
        raise FileExistsError(run_dir)
    if baseline and (not evaluate or resume is not None):
        raise ValueError("baseline 仅用于初始策略评测，不能同时指定 resume。")
    if command in {"grpo-resume", "grpo-evaluate"} and resume is None and not baseline:
        raise ValueError("GRPO resume/evaluate 需要完整 Checkpoint。")
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)

    split = options["evaluation_split"] if evaluate else "train"
    contract, samples = prepared_samples(settings, root, split)
    identity = run_identity(settings, root)
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
        saved = payload["progress"]
        if saved.get("identity") != identity or saved.get("algorithm") != "grpo-v1":
            raise ValueError("GRPO Resume 配置、源码、数据或初始策略身份不一致。")
        if (
            not saved.get("rollout_complete")
            or saved["step"] % epochs
            or saved.get("policy_version") != saved["step"]
        ):
            raise ValueError("只能恢复完整 rollout 更新边界，禁止沿用过期 Old Logprob。")
        if not evaluate and saved["step"] >= target:
            raise ValueError("恢复目标必须大于已保存 step。")
        if reference_digest(saved["reference"]) != saved["reference_sha256"]:
            raise ValueError("Reference 固定快照校验失败。")
    return evaluate, target, contract, samples, identity


def inspect_run(
    settings: dict,
    root: Path,
    *,
    resume: Path | None = None,
    steps: int | None = None,
    command: str = "grpo-smoke",
    baseline: bool = False,
    run_dir: Path | None = None,
) -> dict:
    evaluate, target, contract, samples, identity = prepare_run(
        settings,
        root,
        command=command,
        resume=resume,
        steps=steps,
        baseline=baseline,
        run_dir=run_dir,
    )
    model, groups = setup_model(settings, root, contract, meta=True)
    length = settings["grpo"]["max_new_tokens"]
    for row in samples:
        prompt = prompt_from_sample(row, "cpu")
        if prompt["input_ids"].shape[-1] + length - 1 > model.config.max_position_embeddings:
            raise ValueError("GRPO prompt + rollout 超出上下文预算。")
    return dict(
        status="validated",
        dry_run=True,
        command=command,
        baseline=baseline,
        target_steps=target,
        split=settings["grpo"]["evaluation_split"] if evaluate else "train",
        config=settings,
        parameter_groups=groups,
        identity=identity,
        samples=len(samples),
        data_contract=contract,
    )


def reference_digest(reference: dict) -> str:
    import hashlib

    result = hashlib.sha256()
    for name, value in sorted(reference.items()):
        result.update(name.encode())
        result.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def rollout_options(options: dict) -> dict:
    return {
        key: options[key]
        for key in (
            "group_size",
            "mode",
            "max_new_tokens",
            "temperature",
            "audio_temperature",
            "use_cache",
            "top_k",
            "top_p",
        )
    }


def collect_groups(
    model: nn.Module,
    samples: list[dict],
    options: dict,
    version: int,
    device: str,
    resources: dict | None = None,
):
    from omni.training.grpo import group_advantages
    from omni.training.grpo_rewards import reward_score
    from omni.training.grpo_rollout import rollout_group

    trajectories, rewards, records = [], [], []
    for sample in samples:
        group = rollout_group(
            model,
            prompt_from_sample(sample, device),
            policy_version=version,
            **rollout_options(options),
        )
        values = []
        for trajectory in group:
            if trajectory.policy_version != version or not trajectory.valid_mask.any():
                raise ValueError("Rollout 行为版本错误或没有有效动作。")
            context = dict(sample.get("provenance", {}))
            context["target_ids"] = sample["labels"][sample["labels"].ne(-100)].tolist()
            value, components = reward_score(
                trajectory, model.config, options["reward"], context, **(resources or {})
            )
            values.append(value)
            records.append(
                dict(
                    record_id=sample.get("record_id"),
                    reward=value,
                    reward_components=components,
                    **trajectory.to_record(),
                )
            )
        rewards.append(values)
        trajectories.extend(group)
    reward = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = group_advantages(reward, epsilon=options["advantage_epsilon"]).flatten()
    return trajectories, advantages, reward, records


def update_policy(
    model, trajectories, advantages, reference, optimizer, scaler, dtype, options, grad_clip
):
    from omni.training.grpo import exact_kl, grpo_loss
    from omni.training.grpo_rollout import score_rollout

    optimizer.zero_grad(set_to_none=True)
    count = len(trajectories)
    total_tokens = sum(int(row.valid_mask.sum()) for row in trajectories)
    if not count or not total_tokens:
        raise ValueError("GRPO 更新需要有效轨迹及动作。")
    metrics = dict.fromkeys(("total", "policy", "kl", "clip_fraction", "ratio_mean"), 0.0)
    device = next(model.parameters()).device.type
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
            # 参考前向完成并恢复原参数后才构建 current 图，避免原地换权重破坏梯度。
            with reference_policy(model, reference), torch.no_grad():
                ref = score_rollout(model, trajectory)
            current = score_rollout(model, trajectory)
            kl = torch.zeros_like(current.logprob)
            if trajectory.mode != "audio":
                kl = (
                    kl + exact_kl(current.text_log_probs, ref.text_log_probs) * trajectory.text_mask
                )
            if trajectory.mode != "text":
                kl = kl + (
                    exact_kl(current.audio_log_probs, ref.audio_log_probs) * trajectory.audio_mask
                ).sum(0)
            losses = grpo_loss(
                current.logprob.unsqueeze(0),
                trajectory.old_logprob.unsqueeze(0),
                kl.unsqueeze(0),
                advantage.reshape(1),
                trajectory.valid_mask.unsqueeze(0),
                clip=options["clip"],
                beta=options["kl_coef"],
                normalization=options["normalization"],
            )
            scale = (
                1 / count
                if options["normalization"] == "sequence_mean"
                else int(trajectory.valid_mask.sum()) / total_tokens
            )
            loss = losses["total"] * scale
        scaler.scale(loss).backward()
        for key in metrics:
            metrics[key] += float(losses[key].detach()) * scale
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], grad_clip, error_if_nonfinite=True
    )
    metrics["grad_norm"] = float(norm)
    if not math.isfinite(float(norm)):
        raise RuntimeError("GRPO 梯度非有限。")
    if float(norm) == 0:
        metrics["optimizer_updated"] = False
        return metrics
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    metrics["optimizer_updated"] = scaler.get_scale() >= previous
    if not metrics["optimizer_updated"]:
        raise RuntimeError("GRPO AMP 跳过更新，拒绝推进策略版本。")
    return metrics


def run_grpo(
    settings: dict,
    root: Path,
    run_dir: Path,
    *,
    resume: Path | None = None,
    steps: int | None = None,
    command: str = "grpo-smoke",
    baseline: bool = False,
) -> dict:
    from omni.evaluation.bridge_benchmark import hardware_info
    from omni.training.grpo_rewards import reward_resources
    from omni.training.grpo_rollout import score_rollout

    training, options = settings["training"], settings["grpo"]
    r3_local.check_device(training)
    evaluate, target, contract, samples, identity = prepare_run(
        settings,
        root,
        command=command,
        resume=resume,
        steps=steps,
        baseline=baseline,
        run_dir=run_dir,
    )
    epochs = options["update_epochs"]
    split = options["evaluation_split"] if evaluate else "train"

    seed, device = training["seed"], training["device"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(training["threads"])
    started = time.perf_counter()
    model, groups = setup_model(settings, root, contract)
    model.to(device)
    optimizer, scheduler, scaler, dtype = r3_local.optimizer_state(model, settings)
    resources = reward_resources(settings, root)
    initial_frozen = r3_local.parameter_digest(model, frozen=True)
    reference = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    progress = dict(
        algorithm="grpo-v1",
        step=0,
        policy_version=0,
        identity=identity,
        reference=reference,
        reference_sha256=reference_digest(reference),
        frozen_initial_sha256=initial_frozen,
        rollout_complete=True,
        scheduler=scheduler.state_dict(),
    )
    if resume:
        initial_reference = progress["reference_sha256"]
        progress = load_checkpoint(resume, model, optimizer, scaler)
        if (
            progress["reference_sha256"] != initial_reference
            or progress.get("frozen_initial_sha256") != initial_frozen
            or r3_local.parameter_digest(model, frozen=True) != initial_frozen
        ):
            raise ValueError("GRPO Reference 或冻结主权重与初始 SFT 策略不一致。")
        scheduler.load_state_dict(progress["scheduler"])
        reference = progress["reference"]
    model.eval()
    evaluation_seed = seed + 1_000_003 if evaluate else None
    if evaluate:
        random.seed(evaluation_seed)
        np.random.seed(evaluation_seed)
        torch.manual_seed(evaluation_seed)
    start_step = progress["step"]
    frozen_before = r3_local.parameter_digest(model, frozen=True)
    trainable_before = r3_local.parameter_digest(model, frozen=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False)
    )
    metrics, gradients, stop_reason = [], {}, "step_budget"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    loop_started = time.perf_counter()
    selected = []
    try:
        while evaluate or progress["step"] < target:
            if evaluate:
                selected = samples[: options["evaluation_prompts"]]
            else:
                batches, _ = r3_local.next_batches(samples, progress, training, lambda rows: rows)
                selected = [row for batch in batches for row in batch]
            behavior_version = progress["policy_version"]
            with torch.autocast(torch.device(device).type, dtype=dtype, enabled=dtype is not None):
                trajectories, advantages, reward, records = collect_groups(
                    model, selected, options, behavior_version, device, resources
                )
                # 独立 teacher-forcing 重算先验证行为概率，更新前误差不得静默放宽。
                with torch.no_grad():
                    for row in trajectories:
                        current = score_rollout(model, row).logprob
                        torch.testing.assert_close(current, row.old_logprob, rtol=1e-4, atol=1e-4)
            with (run_dir / "rollouts.jsonl").open("a") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if evaluate:
                metrics.append(
                    dict(
                        step=progress["step"],
                        reward_mean=float(reward.mean()),
                        reward_std=float(reward.std(unbiased=False)),
                        trajectories=len(records),
                    )
                )
                stop_reason = "evaluation"
                break
            progress["rollout_complete"] = False
            for epoch in range(epochs):
                result = update_policy(
                    model,
                    trajectories,
                    advantages,
                    reference,
                    optimizer,
                    scaler,
                    dtype,
                    options,
                    training["grad_clip"],
                )
                row = dict(
                    step=progress["step"] + int(result["optimizer_updated"]),
                    behavior_version=behavior_version,
                    rollout_epoch=epoch,
                    reward_mean=float(reward.mean()),
                    reward_std=float(reward.std(unbiased=False)),
                    zero_variance_groups=int((reward.var(dim=-1, unbiased=False) == 0).sum()),
                    learning_rate=scheduler.get_last_lr()[0],
                    **result,
                )
                metrics.append(row)
                with (run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if not result["optimizer_updated"]:
                    stop_reason = "no_gradient"
                    break
                for name, value in model.named_parameters():
                    if value.grad is not None:
                        if not value.requires_grad or not value.grad.isfinite().all():
                            raise RuntimeError("冻结参数收到梯度或梯度非有限。")
                        group = r3_local.parameter_group(name)
                        gradients[group] = max(
                            gradients.get(group, 0), float(value.grad.abs().max())
                        )
                scheduler.step()
                progress["step"] += 1
                progress["policy_version"] = progress["step"]
            progress["scheduler"] = scheduler.state_dict()
            progress["rollout_complete"] = (
                stop_reason != "no_gradient" and progress["step"] % epochs == 0
            )
            if progress["rollout_complete"] and progress["step"] % training["save_interval"] == 0:
                save_checkpoint(
                    run_dir / f"step-{progress['step']:06d}.pt", model, optimizer, scaler, progress
                )
            if stop_reason == "no_gradient":
                break
            if (
                training["max_seconds"]
                and time.perf_counter() - loop_started >= training["max_seconds"]
            ):
                stop_reason = "time_budget"
                break
    except Exception as error:
        write_json(
            run_dir / "error.json",
            dict(
                status="failed",
                step=progress["step"],
                policy_version=progress["policy_version"],
                error_type=type(error).__name__,
                error=str(error),
                record_ids=[row.get("record_id") for row in selected],
            ),
        )
        raise
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - loop_started
    frozen_after = r3_local.parameter_digest(model, frozen=True)
    trainable_after = r3_local.parameter_digest(model, frozen=False)
    if frozen_before != frozen_after:
        raise RuntimeError("GRPO 冻结参数发生变化。")
    updated = progress["step"] > start_step
    if updated and trainable_before == trainable_after:
        raise RuntimeError("GRPO 记录了更新但可训练参数未变化。")
    checkpoint = run_dir / f"step-{progress['step']:06d}.pt"
    if not evaluate and not checkpoint.exists():
        save_checkpoint(checkpoint, model, optimizer, scaler, progress)
    if not evaluate:
        export_adapter(run_dir / "adapter.pt", model, r3_local.adapter_identity(settings, contract))
    result = dict(
        status="evaluated"
        if evaluate
        else "no_update"
        if stop_reason == "no_gradient"
        else "partial"
        if progress["step"] < target
        else "passed",
        scope="local GRPO engineering; formal G6 quality and CUDA unvalidated",
        command=command,
        baseline=baseline,
        evaluation_seed=evaluation_seed,
        evaluation_record_ids=[row["record_id"] for row in selected] if evaluate else None,
        config=settings,
        identity=identity,
        parameter_groups=groups,
        gradient_groups=gradients,
        data_contract=contract,
        split=split,
        samples=len(samples),
        start_step=start_step,
        step=progress["step"],
        policy_version=progress["policy_version"],
        stop_reason=stop_reason,
        reference_sha256=progress["reference_sha256"],
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        trainable_before=trainable_before,
        trainable_after=trainable_after,
        metrics=metrics,
        loop_seconds=seconds,
        seconds_per_update=seconds / (progress["step"] - start_step) if updated else None,
        hardware=hardware_info(device),
        elapsed_seconds=time.perf_counter() - started,
        checkpoint=str(resume if evaluate else checkpoint) if not baseline else None,
        checkpoint_sha256=digest(resume if evaluate else checkpoint) if not baseline else None,
    )
    if device == "cuda":
        result["cuda"] = dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    write_json(run_dir / ("evaluation.json" if evaluate else "manifest.json"), result)
    return result
