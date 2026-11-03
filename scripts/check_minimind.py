"""MiniMind 本地资产与真实权重验收；不下载、不训练、不运行优化器。"""

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer

from omni.models import OmniPlusConfig, OmniPlusForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def collect_assets(reference_root: Path, asset_root: Path = ROOT) -> dict:
    """只复制指定资产，验证前后哈希；许可证信息来自本地权重卡。"""
    pairs = [
        (reference_root / "out" / name, asset_root / "weights/minimind" / name)
        for name in ("llm_768.pth", "llm_768_moe.pth")
    ] + [
        (reference_root / "model" / name, asset_root / "tokenizers/minimind" / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
    ]
    pairs += [
        (reference_root / "LICENSE", asset_root / "weights/minimind/LICENSE"),
        (reference_root / "out/README.md", asset_root / "weights/minimind/README.md"),
    ]
    if "license: apache-2.0" not in (reference_root / "out/README.md").read_text().lower():
        raise ValueError("权重卡中的许可证需要重新核实，不能沿用历史假设。")
    files = []
    for source, destination in pairs:
        before = file_record(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or file_record(destination)["sha256"] != before["sha256"]:
            shutil.copy2(source, destination)
        copied = file_record(destination)
        if before["sha256"] != copied["sha256"] or file_record(source) != before:
            raise RuntimeError(f"资产复制校验失败：{source.name}")
        files.append({"source": before, "local": copied})

    code = file_record(reference_root / "model/model_minimind.py")
    return {
        "reference_root": str(reference_root),
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "reference_code": code,
        "license": "Apache-2.0",
        "license_source": file_record(reference_root / "out/README.md"),
        "files": files,
    }


def load_reference(reference_root: Path):
    # 仅导入文本核心，不加载旧仓库中的编码器、训练入口或全局配置
    path = reference_root / "model/model_minimind.py"
    spec = importlib.util.spec_from_file_location("minimind_p1_reference", path)
    reference = importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(), str(path), "exec"), reference.__dict__)
    return reference


def compare(actual: Tensor, expected: Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return float((actual.detach().float() - expected.detach().float()).abs().max())


def capture_layers(model) -> tuple[list, list]:
    outputs, hooks = [], []

    def capture(module, args, output) -> None:
        outputs.append(output[0].detach().clone())

    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(capture))
    return outputs, hooks


def check_models(model, reference, tokens: Tensor) -> dict:
    """同权重 FP32 对照：层输出、KV、单步/多步 Decode、padding 与因果性。"""
    model.eval()
    reference.eval()
    actual_layers, actual_hooks = capture_layers(model)
    expected_layers, expected_hooks = capture_layers(reference)
    with torch.no_grad():
        actual = model(tokens, use_cache=True)
        expected = reference(tokens, use_cache=True)
    for hook in actual_hooks + expected_hooks:
        hook.remove()

    result = {
        "logits_max_abs_diff": compare(actual.logits, expected.logits),
        "layer_max_abs_diff": [
            compare(a, b) for a, b in zip(actual_layers, expected_layers, strict=True)
        ],
        "cache_max_abs_diff": max(
            compare(value, other.transpose(1, 2))
            for pair, ref_pair in zip(actual.past_key_values, expected.past_key_values, strict=True)
            for value, other in zip(pair, ref_pair, strict=True)
        ),
    }
    with torch.no_grad():
        for chunk in (1, 2):
            prefix = model(tokens[:, :-chunk], use_cache=True)
            suffix = model(
                tokens[:, -chunk:], past_key_values=prefix.past_key_values, use_cache=True
            )
            result[f"decode_{chunk}_token_max_abs_diff"] = compare(
                suffix.logits, actual.logits[:, -chunk:]
            )
        padded = F.pad(tokens, (2, 0), value=model.config.pad_token_id)
        mask = F.pad(torch.ones_like(tokens), (2, 0), value=0)
        result["padding_max_abs_diff"] = compare(
            model(padded, attention_mask=mask).logits[:, 2:], actual.logits
        )
        changed = tokens.clone()
        changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
        result["causal_prefix_max_abs_diff"] = compare(
            model(changed).logits[:, :-1], actual.logits[:, :-1]
        )
    return result


def check_gradients(model, reference, tokens: Tensor) -> dict:
    # 只启用指定参数的梯度，避免双模型同时分配全量梯度；没有 optimizer/update
    names = ["model.layers.0.self_attn.q_proj.weight"]
    if model.config.use_moe:
        names.append("model.layers.0.mlp.gate.weight")
    for current in (model, reference):
        current.requires_grad_(False)
        for name, param in current.named_parameters():
            param.requires_grad_(name in names)
        current.train()

    actual = model(tokens, labels=tokens)
    expected = reference(tokens, labels=tokens)
    actual.loss.backward()
    (expected.loss + expected.aux_loss).backward()
    result = {
        "checked_parameters": names,
        "parameter_max_abs_diff": {
            name: compare(
                dict(model.named_parameters())[name].grad,
                dict(reference.named_parameters())[name].grad,
            )
            for name in names
        },
        "aux_loss": float(actual.aux_loss.detach()),
        "aux_loss_max_abs_diff": compare(actual.aux_loss, expected.aux_loss),
        "optimizer_steps": 0,
    }
    for current in (model, reference):
        current.zero_grad(set_to_none=True)
        current.requires_grad_(False)
        current.eval()
    return result


def check_routers(model, reference, tokens: Tensor) -> dict:
    if not model.config.use_moe:
        return {"enabled": False}

    captured = [[], []]
    handles = []
    for bucket, current in zip(captured, (model, reference), strict=True):

        def capture(module, args, bucket=bucket) -> None:
            bucket.append(args[0].detach().clone())

        for layer in current.model.layers:
            handles.append(layer.mlp.register_forward_pre_hook(capture))
    with torch.no_grad():
        model(tokens)
        reference(tokens)
    for handle in handles:
        handle.remove()

    routes = []
    for layer_id, (hidden, ref_hidden) in enumerate(zip(*captured, strict=True)):
        mlp, ref_mlp = model.model.layers[layer_id].mlp, reference.model.layers[layer_id].mlp
        scores = mlp.gate(hidden.reshape(-1, model.config.hidden_size)).softmax(-1)
        ref_scores = ref_mlp.gate(ref_hidden.reshape(-1, model.config.hidden_size)).softmax(-1)
        indices = scores.topk(model.config.num_experts_per_tok, sorted=False).indices
        ref_indices = ref_scores.topk(model.config.num_experts_per_tok, sorted=False).indices
        if not torch.equal(indices, ref_indices):
            raise AssertionError(f"第 {layer_id} 层专家选择不一致。")
        routes.append(
            {
                "layer": layer_id,
                "scores_max_abs_diff": compare(scores, ref_scores),
                "expert_counts": torch.bincount(
                    indices.flatten(), minlength=model.config.num_experts
                ).tolist(),
            }
        )
    shared = sum(p.numel() for name, p in model.named_parameters() if ".experts." not in name)
    active_experts = sum(
        model.config.num_experts_per_tok * sum(p.numel() for p in layer.mlp.experts[0].parameters())
        for layer in model.model.layers
    )
    return {
        "enabled": True,
        "top_k": model.config.num_experts_per_tok,
        "routes": routes,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "shared_parameters": shared,
        "expert_parameters_per_token": active_experts,
        "parameter_budget_per_token": shared + active_experts,
        "counting_note": "共享参数按完整张量计数；不是 FLOPs、显存占用或速度测量。",
    }


def check_variant(variant: str, reference_root: Path, asset_root: Path = ROOT) -> dict:
    torch.manual_seed(42)
    config = OmniPlusConfig(use_moe=variant == "moe", pad_token_id=0, dropout=0.0)
    weight = asset_root / "weights/minimind" / f"llm_768{'_moe' if config.use_moe else ''}.pth"
    state = torch.load(weight, map_location="cpu", weights_only=True, mmap=True)
    if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
        raise ValueError("源 embedding/head 不相等，不能按绑定权重配置加载。")

    model = OmniPlusForCausalLM(config).float().eval()
    info = model.load_state_dict(state, strict=True)
    source = load_reference(reference_root)
    reference = (
        source.MiniMindForCausalLM(source.MiniMindConfig(use_moe=config.use_moe)).float().eval()
    )
    reference.load_state_dict(state, strict=True)
    del state

    # tokenizer 与权重独立存放，但必须配对核验特殊 Token 和可用 id 范围
    tokenizer = AutoTokenizer.from_pretrained(
        asset_root / "tokenizers/minimind", local_files_only=True, trust_remote_code=False
    )
    token_ids = list(tokenizer.get_vocab().values())
    if min(token_ids) < 0 or max(token_ids) >= config.vocab_size:
        raise AssertionError("tokenizer Token ID 超出 embedding 范围。")
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (
        config.bos_token_id,
        config.eos_token_id,
        config.pad_token_id,
    ):
        raise AssertionError("tokenizer 特殊 Token 与模型 config 不匹配。")
    tokens = torch.tensor(
        [tokenizer.encode("你好，人工智能。", add_special_tokens=False)], dtype=torch.long
    )
    result = {
        "variant": variant,
        "status": "passed",
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "rtol": 1e-4,
        "atol": 1e-4,
        "config": config.to_dict(),
        "weight": file_record(weight),
        "state_dict_keys": len(model.state_dict()),
        "source_tied_weights_equal": True,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "missing_keys": info.missing_keys,
        "unexpected_keys": info.unexpected_keys,
        "tokenizer": {
            "length": len(tokenizer),
            "max_token_id": max(token_ids),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "forward": check_models(model, reference, tokens),
        "router": check_routers(model, reference, tokens),
        "gradients": check_gradients(model, reference, tokens),
    }
    del reference
    gc.collect()

    # HF 回环只写临时目录，不再永久保存一份重复的大权重
    with tempfile.TemporaryDirectory(prefix="omni-minimind-hf-") as directory:
        model.save_pretrained(directory)
        restored = (
            OmniPlusForCausalLM.from_pretrained(directory, local_files_only=True).float().eval()
        )
        with torch.no_grad():
            result["hf_roundtrip_max_abs_diff"] = compare(
                restored(tokens).logits, model(tokens).logits
            )
        if restored.lm_head.weight is not restored.model.embed_tokens.weight:
            raise AssertionError("HF 权重回环后 embedding/head 绑定丢失。")
        del restored

    # 对同一提示词分别打开/关闭 Cache；固定贪心生成避免抽样差异
    prompt = "你好，请简单介绍一下人工智能。"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
    generated = model.generate(ids, max_new_tokens=24, do_sample=False, use_cache=True)
    uncached = model.generate(ids, max_new_tokens=24, do_sample=False, use_cache=False)
    if not torch.equal(generated, uncached):
        raise AssertionError("Cache 开关下的贪心生成结果不同。")
    result["generation"] = {
        "prompt": prompt,
        "rendered_prompt": text,
        "max_new_tokens": 24,
        "cache_tokens_identical": True,
        "generated_ids": generated[:, ids.shape[1] :].tolist(),
        "text": tokenizer.decode(generated[0, ids.shape[1] :].tolist(), skip_special_tokens=True),
        "quality_note": "LLM 预训练权重的运行烟测，不等同于聊天能力或模型质量评测。",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dense", "moe", "all"], default="all")
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/p1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.reference_root = (ROOT / args.reference_root).resolve()
    args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = collect_assets(args.reference_root.resolve())
    (args.output_dir / "minimind-assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for variant in ["dense", "moe"] if args.variant == "all" else [args.variant]:
        result = check_variant(variant, args.reference_root.resolve())
        if (
            file_record(args.reference_root / "model/model_minimind.py")
            != manifest["reference_code"]
        ):
            raise AssertionError("参考源码在验收期间发生变化。")
        (args.output_dir / f"minimind-{variant}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"MiniMind {variant}: passed, "
            f"logits diff={result['forward']['logits_max_abs_diff']:.3g}",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
