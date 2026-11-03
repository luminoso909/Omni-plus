"""收集官方 Qwen3-0.6B，并验证独立的离线文本基线；不训练、不接入 Omni。"""

import argparse
import gc
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
PROMPTS = ("请用一句话解释什么是语言模型。", "Explain what a language model is in one sentence.")


def collect_assets(model_dir: Path) -> dict:
    """固定官方 revision，只收集一套未量化 Safetensors 与配套 tokenizer。"""
    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True, timeout=20)
    snapshot_download(
        REPO_ID,
        revision=REVISION,
        local_dir=model_dir,
        allow_patterns=list(FILES),
        max_workers=3,
        etag_timeout=20,
    )

    # 本地逐文件校验，LFS 权重另与官方 SHA256 对照
    files = []
    metadata = {file.rfilename: file for file in info.siblings}
    for name in FILES:
        path = model_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = metadata[name]
        lfs_hash = expected.lfs.sha256 if expected.lfs else None
        if path.stat().st_size != expected.size or (lfs_hash and digest != lfs_hash):
            raise ValueError(f"官方资产校验失败：{name}")

        files.append(
            dict(path=name, bytes=path.stat().st_size, sha256=digest, official_sha256=lfs_hash)
        )
    return dict(
        repo_id=REPO_ID,
        revision=info.sha,
        source=f"https://huggingface.co/{REPO_ID}/tree/{REVISION}",
        model_dir=str(model_dir),
        license="Apache-2.0",
        files=files,
    )


@torch.inference_mode()
def check_model(model_dir: Path, device: str = "mps", max_new_tokens: int = 32) -> dict:
    """只读取本地 HF 目录；先做 CPU FP32 前向，再做指定设备短文本生成。"""
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用；请在真实主机运行，或显式指定 --device cpu。")

    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    load_start = time.perf_counter()
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    if any(loading.values()):
        raise ValueError(f"存在未解释的权重加载差异：{loading}")

    # Tokenizer 允许小于模型 padding 后的词表，但不能产生越界 Token ID
    ids = list(tokenizer.get_vocab().values())
    vocab_size = model.get_input_embeddings().num_embeddings
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if model.config.tie_word_embeddings and not tied:
        raise ValueError("输入 Embedding 与输出头未按配置绑定权重。")
    if min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError("Tokenizer Token ID 超出模型 Embedding 范围。")
    sample_text = "你好，世界。Hello, world."
    roundtrip = tokenizer.decode(tokenizer.encode(sample_text, add_special_tokens=False))
    if roundtrip != sample_text:
        raise ValueError("Tokenizer 文本编码/解码未保持输入文本。")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        rendered, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False
    )
    started = time.perf_counter()
    logits = model(**inputs, logits_to_keep=1).logits
    finite = bool(torch.isfinite(logits).all())
    if not finite:
        raise ValueError("CPU FP32 前向包含非有限 logits。")
    report = dict(
        model_type=model.config.model_type,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_length=len(tokenizer),
        max_token_id=max(ids),
        embedding_vocab_size=vocab_size,
        special_token_ids={
            name: getattr(tokenizer, name)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        all_special_ids=tokenizer.all_special_ids,
        tokenizer_roundtrip=True,
        weights_tied=tied,
        loading_info=loading,
        cpu=dict(
            status="passed",
            device="cpu",
            dtype="float32",
            load_seconds=load_seconds,
            forward_seconds=time.perf_counter() - started,
            logits_shape=list(logits.shape),
            logits_finite=finite,
        ),
    )

    # 释放 CPU 大模型后再加载生成模型，避免同时持有两份权重
    del model, logits, inputs
    gc.collect()
    dtype = torch.float16 if device == "mps" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=dtype)
        .eval()
        .to(device)
    )
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    samples = []
    for prompt in PROMPTS:
        template = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            template, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False
        ).to(device)
        torch.manual_seed(42)
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        output = model.generate(**inputs, generation_config=generation, use_model_defaults=False)
        if device == "mps":
            torch.mps.synchronize()
        new_ids = output[0, inputs.input_ids.shape[1] :].cpu().tolist()
        samples.append(
            dict(
                prompt=prompt,
                rendered_template=template,
                input_ids=inputs.input_ids[0].cpu().tolist(),
                generated_ids=new_ids,
                text=tokenizer.decode(new_ids, skip_special_tokens=True),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    report["generation"] = dict(
        status="passed",
        device=device,
        dtype=str(dtype),
        enable_thinking=False,
        seed=42,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        samples=samples,
    )
    report["checked_at_utc"] = datetime.now(UTC).isoformat()
    report["offline"] = dict(local_files_only=True, trained=False, integrated_with_omni=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("weights/qwen3-0.6b"))
    parser.add_argument("--output", type=Path, default=Path("experiments/p1/qwen3.json"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--download", action="store_true", help="显式下载固定 revision 的官方资产")
    args = parser.parse_args()
    args.model_dir = ROOT / args.model_dir
    args.output = ROOT / args.output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.download:
        assets = collect_assets(args.model_dir)
        args.output.with_name("qwen3-assets.json").write_text(
            json.dumps(assets, ensure_ascii=False, indent=2) + "\n"
        )
    report = check_model(args.model_dir, args.device, args.max_new_tokens)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Qwen3 离线验收通过：CPU FP32 前向；{args.device} 中英文生成。报告：{args.output}")


if __name__ == "__main__":
    main()
