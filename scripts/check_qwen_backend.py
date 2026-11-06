"""核对固定版本的本地 Qwen3 资产与 Thinker Backend 数值，输出独立验收记录。"""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
import yaml
from torch import Tensor
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase, Qwen3ForCausalLM

from omni.models.thinker.qwen_backend import QwenBackend

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "法国首都在哪里？只回答城市名。",
    "What is the capital of France? Answer with the city name only.",
)


def check_files(model_dir: Path, expected: dict) -> dict:
    # 多出的单文件权重或模板可能优先于固定清单中的文件被 HF 加载。
    entries = {path.name for path in model_dir.iterdir()} - {".cache", ".DS_Store"}
    if entries != expected.keys():
        missing = sorted(expected.keys() - entries)
        extra = sorted(entries - expected.keys())
        raise ValueError(f"本地目录与固定资产清单不一致，缺少 {missing}，多出 {extra}。")

    files = {}
    for name, entry in expected.items():
        path = model_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        actual = {"bytes": path.stat().st_size, "sha256": digest}
        if any(actual[key] != entry[key] for key in actual):
            raise ValueError(f"本地资产与固定 revision 的记录不一致：{path}")
        files[name] = actual
    return files


def check_tokenizer(tokenizer: PreTrainedTokenizerBase, model: Qwen3ForCausalLM) -> dict:
    vocab = tokenizer.get_vocab()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    min_id, max_id = min(vocab.values()), max(vocab.values())
    tied = input_weight is output_weight
    if min_id < 0 or max_id >= input_weight.shape[0] or input_weight.shape != output_weight.shape:
        raise ValueError("Tokenizer ID 或词表形状与 embedding／输出头不匹配。")
    if tied != model.config.tie_word_embeddings:
        raise ValueError("输入 embedding 与输出头的绑定状态与配置不一致。")

    markers = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>")
    marker_ids = {marker: vocab[marker] for marker in markers}
    for marker, token_id in marker_ids.items():
        if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
            raise ValueError(f"模板标记没有独立编码为单个 Token：{marker}")
    generation = model.generation_config
    stop_ids = generation.eos_token_id
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
    if (
        tokenizer.pad_token_id != marker_ids["<|endoftext|>"]
        or tokenizer.eos_token_id != marker_ids["<|im_end|>"]
        or generation.pad_token_id != tokenizer.pad_token_id
        or not stop_ids
        or tokenizer.eos_token_id not in stop_ids
        or not set(stop_ids).issubset(vocab.values())
    ):
        raise ValueError("Tokenizer 与原生生成配置的 PAD/EOS 定义不一致。")
    sample = "你好，世界。Hello, world."
    if tokenizer.decode(tokenizer.encode(sample, add_special_tokens=False)) != sample:
        raise ValueError("Tokenizer 中英文编码往返不一致。")
    return dict(
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=len(tokenizer),
        id_range=[min_id, max_id],
        model_vocab_size=input_weight.shape[0],
        padded_vocab_rows=input_weight.shape[0] - len(tokenizer),
        bos_id=tokenizer.bos_token_id,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
        generation_eos_ids=stop_ids,
        markers=marker_ids,
        all_special_ids=tokenizer.all_special_ids,
        weights_tied=tied,
        padding_side=tokenizer.padding_side,
        roundtrip=True,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    )


def render_chat(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], *, thinking: bool
) -> str:
    kwargs = dict(add_generation_prompt=True, enable_thinking=thinking)
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    direct = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    suffix = "<|im_start|>assistant\n"
    if not thinking:
        suffix += "<think>\n\n</think>\n\n"
    if not text.endswith(suffix) or direct != tokenizer.encode(text, add_special_tokens=False):
        raise ValueError("官方模板前缀与 Token 编码不一致。")
    return text


def check_parity(
    backend: QwenBackend, ids: Tensor, mask: Tensor, turn_ids: Tensor, *, rtol: float, atol: float
) -> dict:
    """比较 [B,T] 左 padding 输入、[B,1] decode 和完整重放。"""
    model = backend.model
    layer = model.config.num_hidden_layers // 2 - 1
    metrics = {}

    def compare(name: str, actual: Tensor, expected: Tensor) -> None:
        if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise ValueError(f"出现非有限值：{name}")
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, msg=lambda detail: f"{name}\n{detail}"
        )
        metrics[name] = dict(
            shape=list(actual.shape), max_abs_diff=(actual - expected).abs().max().item()
        )

    # 对照 HF generate 的逻辑 RoPE 位置与连续的物理 Cache 位置
    width = ids.shape[1]
    slots = torch.arange(width, device=ids.device)
    logical = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    native_inputs = model.prepare_inputs_for_generation(
        ids, attention_mask=mask, cache_position=slots
    )
    torch.testing.assert_close(native_inputs["position_ids"], logical)
    native = model(**native_inputs, use_cache=True, output_hidden_states=True)
    prefill = backend(ids, attention_mask=mask, use_cache=True, hidden_layer=layer)
    compare("prefill_logits", prefill.logits, native.logits)
    compare("prefill_hidden", prefill.selected_hidden_state, native.hidden_states[layer + 1])
    cache = prefill.past_key_values
    cache_shape = list(cache.layers[0].keys.shape)
    if cache.get_seq_length() != width or cache_shape != [
        ids.shape[0],
        model.config.num_key_value_heads,
        width,
        model.config.head_dim,
    ]:
        raise ValueError(f"Prefill KV 形状异常：{cache_shape}")

    # 裸 HF forward 默认使用物理位置。Embedding 路径用显式逻辑位置对照 generate
    raw_native = model(ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    raw = backend(ids, attention_mask=mask, position_ids=slots[None], logits_to_keep=1)
    compare("native_forward_physical_positions", raw.logits, raw_native.logits)
    embeds = backend.embed_tokens(ids)
    compare("token_embeddings", embeds, model.get_input_embeddings()(ids))
    embedded = backend(
        inputs_embeds=embeds,
        attention_mask=mask.bool(),
        position_ids=logical,
        cache_position=slots,
        hidden_layer=layer,
        logits_to_keep=1,
    )
    compare("embeds_logits_tail", embedded.logits, prefill.logits[:, -1:])
    compare("embeds_hidden_full", embedded.selected_hidden_state, prefill.selected_hidden_state)
    final = backend(
        ids, attention_mask=mask, hidden_layer=model.config.num_hidden_layers - 1, logits_to_keep=1
    )
    compare(
        "final_block_after_norm",
        model.model.norm(final.selected_hidden_state),
        native.hidden_states[-1],
    )

    # 去掉左侧 padding 后，有效位置应与逐样本运行一致
    for row, valid in enumerate(mask.bool()):
        single = backend(ids[row, valid][None], hidden_layer=layer)
        compare(f"unpadded_logits_{row}", prefill.logits[row, valid][None], single.logits)
        compare(
            f"unpadded_hidden_{row}",
            prefill.selected_hidden_state[row, valid][None],
            single.selected_hidden_state,
        )

    # 用同一 Token 继续两条独立 Cache，再与关闭 Cache 的完整重放比较
    next_ids = native.logits[:, -1].argmax(dim=-1, keepdim=True)
    full_ids = torch.cat((ids, next_ids), dim=1)
    full_mask = torch.cat((mask, torch.ones_like(next_ids)), dim=1)
    step_slot = slots[-1:] + 1
    native_inputs = model.prepare_inputs_for_generation(
        full_ids,
        past_key_values=native.past_key_values,
        attention_mask=full_mask,
        cache_position=step_slot,
    )
    torch.testing.assert_close(native_inputs["position_ids"], mask.sum(-1, keepdim=True))
    native_step = model(**native_inputs, use_cache=True, output_hidden_states=True)
    step = backend(
        next_ids,
        attention_mask=full_mask,
        cache_position=step_slot,
        past_key_values=cache,
        use_cache=True,
        hidden_layer=layer,
    )
    replay = backend(full_ids, attention_mask=full_mask, hidden_layer=layer, logits_to_keep=1)
    if step.past_key_values is not cache or cache.get_seq_length() != width + 1:
        raise ValueError("Decode 未在原生 Cache 继续写入一个位置。")
    if replay.past_key_values is not None:
        raise ValueError("关闭 Cache 后仍返回状态。")
    compare("decode_logits", step.logits, native_step.logits)
    compare("decode_hidden", step.selected_hidden_state, native_step.hidden_states[layer + 1])
    compare("cache_vs_replay_logits", step.logits, replay.logits)
    compare(
        "cache_vs_replay_hidden", step.selected_hidden_state, replay.selected_hidden_state[:, -1:]
    )

    # 开新话轮时不传历史 Cache，复用同一 Backend 并验证旧 Cache 长度保持
    reset = backend(turn_ids, use_cache=True, logits_to_keep=1)
    fresh = model(turn_ids, use_cache=False, logits_to_keep=1)
    if (
        reset.past_key_values is cache
        or reset.past_key_values.get_seq_length() != turn_ids.shape[1]
        or cache.get_seq_length() != width + 1
    ):
        raise ValueError("新话轮复用了旧 Cache，或改变了旧 Cache 长度。")
    compare("multiturn_reset_logits", reset.logits, fresh.logits)
    return dict(
        hidden_layer=layer,
        layer_convention="zero_based_post_block_before_final_norm",
        cache_type=type(cache).__name__,
        prefill_cache_shape=cache_shape,
        decode_cache_length=cache.get_seq_length(),
        reset_cache_length=reset.past_key_values.get_seq_length(),
        prefill_cache_positions=slots.tolist(),
        prefill_position_ids=logical.tolist(),
        decode_cache_position=step_slot.tolist(),
        decode_position_ids=native_inputs["position_ids"].tolist(),
        metrics=metrics,
    )


def greedy_tokens(
    backend: QwenBackend,
    ids: Tensor,
    mask: Tensor,
    generation: GenerationConfig,
    *,
    use_cache: bool,
) -> Tensor:
    """验收用贪心解码。缓存模式首步 prefill，后续每次只传最后一个 Token。"""
    width = ids.shape[1]
    stop_ids = torch.tensor(generation.eos_token_id, device=ids.device)
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    cache = None
    for _ in range(generation.max_new_tokens):
        current = ids if cache is None else ids[:, -1:]
        out = backend(
            current,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=use_cache,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        chosen = out.logits[:, -1].argmax(dim=-1)
        chosen = chosen.masked_fill(finished, generation.pad_token_id)
        ids = torch.cat((ids, chosen[:, None]), dim=1)
        mask = torch.cat((mask, torch.ones_like(chosen[:, None])), dim=1)
        finished |= (chosen[:, None] == stop_ids).any(dim=-1)
        if bool(finished.all()):
            break
    return ids[:, width:]


@torch.inference_mode()
def check_model(model_dir: Path, validation: dict, device: str) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(validation["seed"])
    dtype = getattr(torch, validation["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    started = time.perf_counter()
    model, loading = Qwen3ForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=dtype,
        output_loading_info=True,
    )
    model = model.eval().to(device)
    load_seconds = time.perf_counter() - started
    if any(loading.values()):
        raise ValueError(f"存在未解释的权重加载差异：{loading}")
    backend = QwenBackend(model).eval()
    token_info = check_tokenizer(tokenizer, model)

    # 核对原生模板生成前缀。固定关闭 thinking 进行本轮短生成
    thinking = validation["enable_thinking"]
    rendered = [
        render_chat(tokenizer, [{"role": "user", "content": prompt}], thinking=thinking)
        for prompt in PROMPTS
    ]
    render_chat(tokenizer, [{"role": "user", "content": PROMPTS[0]}], thinking=not thinking)
    turn = render_chat(
        tokenizer,
        [
            {"role": "user", "content": PROMPTS[0]},
            {"role": "assistant", "content": "巴黎。"},
            {"role": "user", "content": "那座城市在哪个国家？"},
        ],
        thinking=thinking,
    )
    turn_ids = tokenizer(turn, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    batch = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt").to(
        device
    )
    ids, mask = batch.input_ids, batch.attention_mask
    if bool(mask[:, 0].all()) or not bool(mask[:, -1].all()):
        raise ValueError("验收提示词未形成有效的左 padding 测试。")
    parity = check_parity(
        backend, ids, mask, turn_ids, rtol=validation["rtol"], atol=validation["atol"]
    )
    token_info["thinking_modes_checked"] = [False, True]

    # 同一官方模型的 generate 与 Backend 贪心路径比较完整生成 ID
    generation = GenerationConfig(
        do_sample=False,
        use_cache=True,
        max_new_tokens=validation["max_new_tokens"],
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=token_info["generation_eos_ids"],
        pad_token_id=token_info["pad_id"],
    )
    started = time.perf_counter()
    native = model.generate(
        input_ids=ids, attention_mask=mask, generation_config=generation, use_model_defaults=False
    )[:, ids.shape[1] :]
    cached = greedy_tokens(backend, ids, mask, generation, use_cache=True)
    uncached = greedy_tokens(backend, ids, mask, generation, use_cache=False)
    torch.testing.assert_close(cached, native, rtol=0, atol=0)
    torch.testing.assert_close(uncached, native, rtol=0, atol=0)
    if not set(native.flatten().tolist()).issubset(tokenizer.get_vocab().values()):
        raise ValueError("生成 ID 超出真实 tokenizer 词表。")
    decoded = tokenizer.batch_decode(native, skip_special_tokens=True)
    if not all(text.strip() for text in decoded):
        raise ValueError("中英文短生成应产生非空文本。")

    return dict(
        status="passed",
        runtime=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            platform=platform.platform(),
            device=device,
            dtype=validation["dtype"],
            threads=torch.get_num_threads(),
            attention=model.config._attn_implementation,
            seed=validation["seed"],
            local_files_only=True,
        ),
        model=dict(
            model_type=model.config.model_type,
            loading_info=loading,
            load_seconds=load_seconds,
            parameters=sum(param.numel() for param in model.parameters()),
            **{
                key: getattr(model.config, key)
                for key in (
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "vocab_size",
                    "max_position_embeddings",
                )
            },
        ),
        tokenizer=token_info,
        parity=dict(rtol=validation["rtol"], atol=validation["atol"], **parity),
        generation=dict(
            enable_thinking=thinking,
            do_sample=False,
            max_new_tokens=generation.max_new_tokens,
            original=model.generation_config.to_diff_dict(),
            native_matches_cached_and_uncached=True,
            elapsed_seconds=time.perf_counter() - started,
            samples=[
                dict(
                    prompt=prompt,
                    rendered=template,
                    input_ids=seq[valid].tolist(),
                    generated_ids=new.tolist(),
                    text=text,
                )
                for prompt, template, seq, valid, new, text in zip(
                    PROMPTS, rendered, ids, mask.bool(), native, decoded, strict=True
                )
            ],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="本地目录，文件需匹配资产清单。")
    parser.add_argument("--output-dir", type=Path, required=True, help="本次验收独立的新目录。")
    parser.add_argument("--device", choices=("cpu", "mps"), help="默认使用模型配置的验收设备。")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/model/qwen3_1_7b.yaml").read_text())
    assets = yaml.safe_load((ROOT / "configs/assets_v2.yaml").read_text())
    asset = assets["assets"][config["asset"]]
    model_dir = ROOT / (args.model_dir or asset["local_path"])
    output_dir = ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"请使用新的验收目录：{output_dir}")
    device = args.device or config["validation"]["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前主机 MPS 不可用。")

    files = check_files(model_dir, asset["files"])
    report = check_model(model_dir, config["validation"], device)
    report["asset"] = dict(
        repo_id=asset["repo_id"],
        revision=asset["revision"],
        license=asset["license"],
        source=f"https://huggingface.co/{asset['repo_id']}/tree/{asset['revision']}",
        model_dir=str(args.model_dir or asset["local_path"]),
        files=files,
    )
    report["checked_at_utc"] = datetime.now(UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Qwen Backend 离线验收通过。报告位置 {manifest}")


if __name__ == "__main__":
    main()
