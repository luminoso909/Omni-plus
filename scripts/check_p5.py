"""用本地 SFT 和 Mimi 实测确定性流式推理。"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import Tensor
from torch.nn.functional import pad
from transformers import AutoTokenizer, MimiModel, PreTrainedTokenizerBase

from omni.inference.codec import MimiCodec
from omni.inference.streaming import check_chunk_parity, stream_waveform
from omni.models import OmniPlusConfig, OmniPlusOmni

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "short": "只回答：你好。",
    "long": "你好，请用一句话介绍春天。",
}
MAX_STEPS = 256
SEED = 42


@dataclass
class Trial:
    tokens: list[int]
    codes: Tensor  # Long [1,codebooks,frames]，已去除 delay、stop 和 pad
    steps: int
    metrics: dict


def file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def peak_rss_mib() -> float:
    # macOS 的 ru_maxrss 为 bytes，Linux 为 KiB。
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if platform.system() == "Darwin" else 1024)


def run_trial(
    model: OmniPlusOmni,
    tokenizer: PreTrainedTokenizerBase,
    codec: MimiCodec,
    prompt: str,
    chunk_frames: int,
    use_cache: bool,
    max_steps: int = MAX_STEPS,
    left_pad: int = 0,
) -> Trial:
    device = model.device

    def now() -> float:
        if device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    # 热启动请求从原始文本开始，包含模板、分词及模型到 PCM 的完整链路。
    started = now()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(rendered, add_special_tokens=False)], device=device)
    mask = pad(torch.ones_like(ids), (left_pad, 0))
    ids = pad(ids, (left_pad, 0), value=model.config.pad_token_id)
    input_seconds = now() - started
    events = model.stream_generate(
        ids,
        max_new_tokens=max_steps,
        temperature=0,
        audio_temperature=0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        attention_mask=mask,
        open_thinking=False,
        use_cache=use_cache,
    )

    # 只累计推进模型 iterator 的时间。波形解码在外层进行。
    frames, steps, model_seconds = [], 0, 0.0

    def measure_model() -> Iterator[tuple[Tensor | None, Tensor | None]]:
        nonlocal steps, model_seconds
        tick = now()
        for token, frame in events:
            model_seconds += now() - tick
            steps += 1
            if frame is not None:
                frames.append(frame)
            yield token, frame
            tick = now()
        model_seconds += now() - tick

    # 对外事件的到达时刻用于首字、首段音频及 chunk 间隔。
    tokens, chunks, chunk_sizes, arrivals = [], [], [], []
    ttft = None
    specials = set(tokenizer.all_special_ids)
    for event in stream_waveform(measure_model(), codec, chunk_frames=chunk_frames):
        elapsed = now() - started
        if event.token is not None:
            token = int(event.token.item())
            tokens.append(token)
            if ttft is None and token not in specials:
                ttft = elapsed
        if event.waveform is not None:
            chunks.append(event.waveform)
            chunk_sizes.append(event.frames)
            arrivals.append(elapsed)
    end_to_end = now() - started

    if not frames or not chunks:
        raise AssertionError("真实案例未产生完整声学帧及 PCM。")
    codes = torch.cat(frames, dim=-1).cpu()
    waveform = torch.cat(chunks)
    audio_seconds = waveform.numel() / codec.sampling_rate
    hop = round(codec.sampling_rate / codec.model.config.frame_rate)
    if (
        ttft is None
        or not torch.isfinite(waveform).all()
        or not bool(waveform.abs().max() > 0)
        or [chunk.numel() for chunk in chunks] != [size * hop for size in chunk_sizes]
        or sum(chunk_sizes) != codes.shape[-1]
        or chunk_sizes[:-1] != [chunk_frames] * (len(chunks) - 1)
        or not 0 < chunk_sizes[-1] <= chunk_frames
        or ((codes < 0) | (codes >= codec.codebook_size)).any()
    ):
        raise AssertionError("文本、声学码本或 PCM chunk 不符合流式输出约定。")

    gaps = [second - first for first, second in zip(arrivals, arrivals[1:], strict=False)]
    metrics = dict(
        ttft_seconds=ttft,
        ttfa_seconds=arrivals[0],
        model_seconds=model_seconds,
        end_to_end_seconds=end_to_end,
        end_to_end_minus_model_seconds=end_to_end - model_seconds,
        input_preparation_seconds=input_seconds,
        audio_seconds=audio_seconds,
        audio_samples=waveform.numel(),
        model_rtf=model_seconds / audio_seconds,
        end_to_end_rtf=end_to_end / audio_seconds,
        p95_chunk_gap_seconds=float(np.percentile(gaps, 95)) if gaps else None,
        chunk_gap_seconds=gaps,
        chunk_arrival_seconds=arrivals,
        chunk_frames=chunk_sizes,
    )
    return Trial(tokens, codes, steps, metrics)


def compare_trials(actual: Trial, expected: Trial) -> None:
    if (
        actual.tokens != expected.tokens
        or not torch.equal(actual.codes, expected.codes)
        or actual.steps != expected.steps
    ):
        raise AssertionError("相同确定性请求在 Cache、padding 或重复运行时产生不同输出。")


def benchmark_prompt(
    model: OmniPlusOmni,
    tokenizer: PreTrainedTokenizerBase,
    codec: MimiCodec,
    prompt: str,
    chunk_frames: int,
    repeats: int,
) -> dict:
    # 两种 Cache 模式先热身，再交替测量，减少固定运行顺序的影响。
    options = (True, False)
    for cache in options:
        run_trial(model, tokenizer, codec, prompt, chunk_frames, cache, max(16, chunk_frames + 8))

    trials = {True: [], False: []}
    for index in range(repeats):
        order = options if index % 2 == 0 else options[::-1]
        for cache in order:
            trials[cache].append(run_trial(model, tokenizer, codec, prompt, chunk_frames, cache))

    baseline = trials[True][0]
    for runs in trials.values():
        for trial in runs:
            compare_trials(trial, baseline)

    # EOS 后文本不再出现，达到预算前结束意味着全部码本已停止。
    eos = tokenizer.eos_token_id
    if (
        baseline.steps >= MAX_STEPS
        or baseline.tokens[-1] != eos
        or eos in baseline.tokens[:-1]
        or tokenizer.pad_token_id in baseline.tokens
        or baseline.steps <= len(baseline.tokens)
    ):
        raise AssertionError("真实案例未通过文本 EOS、音频尾部及自然停止检查。")

    left_pad = 3
    padded = run_trial(model, tokenizer, codec, prompt, chunk_frames, True, left_pad=left_pad)
    compare_trials(padded, baseline)
    summary = dict(
        prompt=prompt,
        text=tokenizer.decode(baseline.tokens, skip_special_tokens=True),
        generated_ids=baseline.tokens,
        codes_shape=list(baseline.codes.shape),
        codes_sha256=hashlib.sha256(baseline.codes.numpy().tobytes()).hexdigest(),
        forward_steps=baseline.steps,
        text_eos_step=len(baseline.tokens),
        natural_stop=True,
        cache_tokens_and_codes_equal=True,
        repeat_tokens_and_codes_equal=True,
        left_padding=dict(count=left_pad, mask_value=0, tokens_and_codes_equal=True),
        mimi_pcm_parity=check_chunk_parity(codec, baseline.codes, chunk_frames=chunk_frames),
    )
    for cache, runs in trials.items():
        samples = [run.metrics for run in runs]
        scalars = {
            key: statistics.median(sample[key] for sample in samples)
            for key, value in samples[0].items()
            if isinstance(value, (int, float))
        }
        summary["cache_on" if cache else "cache_off"] = dict(samples=samples, median=scalars)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.threads, args.chunk_frames, args.repeats) < 1:
        parser.error("线程数、chunk 帧数及重复次数必须为正数。")
    mps_available = torch.backends.mps.is_available()
    if args.device == "mps" and not mps_available:
        parser.error("当前进程的 MPS 不可用。")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = ROOT / (args.report or Path(f"experiments/p5/{run_id}.json"))
    if report_path.exists():
        raise FileExistsError(f"请指定新的报告路径，已有结果：{report_path}")

    reference = (ROOT / args.reference_root).resolve()
    weights = reference / "out/sft_omni_768.pth"
    mimi = reference / "model/mimi"
    assets = [weights, mimi / "model.safetensors", mimi / "config.json"]
    assets += [reference / "model" / name for name in ("tokenizer.json", "tokenizer_config.json")]
    sources = [
        ROOT / "omni/models" / name
        for name in (
            "config.py",
            "model_omni.py",
            "thinker/model_thinker.py",
            "talker/model_talker.py",
        )
    ]
    sources += [ROOT / "omni/inference" / name for name in ("codec.py", "streaming.py")]
    sources.append(Path(__file__).resolve())
    source_records = [file_record(path) for path in sources]

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    before_load = peak_rss_mib()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(reference / "model", local_files_only=True)
    config = OmniPlusConfig(pad_token_id=tokenizer.pad_token_id)
    state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
        raise AssertionError("源 SFT embedding 与绑定的输出头不相等。")
    model = OmniPlusOmni(config).float().eval()
    model.load_state_dict(state, strict=True)
    del state
    model.to(args.device)
    mimi_model, info = MimiModel.from_pretrained(
        mimi, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if any(info.values()):
        raise AssertionError(f"Mimi 权重加载差异：{info}")
    codec = MimiCodec(mimi_model.to(args.device), num_codebooks=config.num_audio_codebooks)
    if args.device == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    after_load = peak_rss_mib()

    cases = {}
    for name, prompt in PROMPTS.items():
        cases[name] = benchmark_prompt(
            model, tokenizer, codec, prompt, args.chunk_frames, args.repeats
        )
    if cases["short"]["codes_shape"][-1] >= cases["long"]["codes_shape"][-1]:
        raise AssertionError("长案例未生成比短案例更长的语音。")

    # 限制在第 13 步，检验未自然结束时仍能冲刷所有已对齐的尾帧。
    cap = 13
    limited = run_trial(model, tokenizer, codec, PROMPTS["long"], args.chunk_frames, True, cap)
    if limited.steps != cap or limited.codes.shape[-1] != cap - config.num_audio_codebooks:
        raise AssertionError("生成步数预算或预算截断后的完整帧数不符。")
    if [file_record(path) for path in sources] != source_records:
        raise AssertionError("计时期间源码发生变化，请在稳定快照上重跑。")

    report = dict(
        status="passed",
        run_id=run_id,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        reference_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=reference, text=True
        ).strip(),
        source_files=source_records,
        assets=[file_record(path) for path in assets],
        environment=dict(
            platform=platform.platform(),
            machine=platform.machine(),
            logical_cpus=os.cpu_count(),
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=args.device,
            dtype="float32",
            mps_available=mps_available,
            cpu_threads=torch.get_num_threads(),
            local_files_only=True,
            hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"),
        ),
        parameters=dict(
            seed=SEED,
            temperature=0,
            audio_temperature=0,
            top_k=0,
            top_p=1,
            repetition_penalty=1,
            open_thinking=False,
            max_new_tokens=MAX_STEPS,
            warmups_per_mode=1,
            warmup_max_new_tokens=max(16, args.chunk_frames + 8),
            repeats_per_mode=args.repeats,
            chunk_frames=args.chunk_frames,
            sampling_rate=codec.sampling_rate,
            model_config=config.to_dict(),
        ),
        methodology=dict(
            end_to_end="单请求热启动，从模板、分词到最终 PCM 事件流耗尽，不含加载、写盘及播放。",
            model="逐次推进模型 iterator 的活跃墙钟时间，包含前向、采样和码本对齐。",
            end_to_end_minus_model="包含分词、Mimi 解码、调度和计时开销，不能视为纯 Codec 耗时。",
            ttft="从原始文本请求开始到首个非特殊文本 Token 事件。",
            ttfa="从原始文本请求开始到首段实际解码的非空 PCM。",
            rtf="耗时除以生成 PCM 的时长。",
            chunk_gap="连续 PCM 事件到达差，含最后一个部分 chunk。每次试验取线性 P95。",
            memory="进程 ru_maxrss 累计高水位，覆盖两份模型及全部运行，不归因于单个 Cache 模式。",
            synchronization="MPS 模式计时边界显式同步，CPU 模式直接使用单调墙钟。",
        ),
        cold_load_seconds=load_seconds,
        process_peak_rss_mib=dict(
            before_load=before_load, after_load=after_load, final=peak_rss_mib()
        ),
        cases=cases,
        budget_limit=dict(
            max_new_tokens=cap,
            forward_steps=limited.steps,
            codes_shape=list(limited.codes.shape),
            chunk_frames=limited.metrics["chunk_frames"],
            audio_samples=limited.metrics["audio_samples"],
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"P5 {args.device} 真实流式基准已保存到 {report_path}")


if __name__ == "__main__":
    main()
