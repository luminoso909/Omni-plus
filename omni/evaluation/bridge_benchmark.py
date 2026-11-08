"""固定留出集的 Bridge 诊断，保存逐样本的生成、时延、质量与失败。"""

import csv
import hashlib
import platform
from collections.abc import Callable
from pathlib import Path
from statistics import median

import torch
from torch import Tensor, nn

from omni.evaluation.sft import generate_audio
from omni.inference.codec import MimiCodec


def benchmark_options(evaluation: dict) -> dict | None:
    if "benchmark" not in evaluation:
        return None
    options = evaluation["benchmark"]
    if not isinstance(options, dict) or options.keys() != {"split", "t2a_samples"}:
        raise ValueError("evaluation.benchmark 需要 split 与 t2a_samples。")
    if options["split"] not in ("validation", "test"):
        raise ValueError("Benchmark split 必须是 validation 或 test。")
    count = options["t2a_samples"]
    if type(count) is not int or count < 1:
        raise ValueError("Benchmark t2a_samples 必须为正整数。")
    return options


def hardware_info(device: str) -> dict:
    info = dict(
        device=device,
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_threads=torch.get_num_threads(),
    )
    if torch.device(device).type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        info.update(gpu=gpu.name, gpu_bytes=gpu.total_memory, cuda=torch.version.cuda)
    return info


def asr_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    # 固定整个本地识别资产，避免同一路径下静默切换模型或 tokenizer。
    result = {}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            with file.open("rb") as stream:
                result[file.relative_to(path).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    if not result:
        raise FileNotFoundError(path)
    return result


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_audio(
    model: nn.Module,
    tokenizer,
    samples: list[dict],
    selections: list[dict],
    codec: MimiCodec,
    output_dir: Path,
    count: int,
    max_new_tokens: int,
    asr: Callable[[Tensor, int], str] | None = None,
) -> dict:
    """选择缓存顺序前 N 条 T2A，保留全部失败与质量覆盖。"""
    if len(samples) != len(selections) or any(
        sample["task"] != selected["task"]
        for sample, selected in zip(samples, selections, strict=True)
    ):
        raise ValueError("Benchmark 样本顺序与缓存清单不一致。")
    selected = [
        (sample, item["record_id"])
        for sample, item in zip(samples, selections, strict=True)
        if sample["task"] == "t2a"
    ][:count]
    if len(selected) != count or len({record_id for _, record_id in selected}) != count:
        raise ValueError("留出集内没有足够独立 T2A 样本。")

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (sample, record_id) in enumerate(selected):
        sample_dir = output_dir / f"sample-{index:04d}"
        try:
            row = generate_audio(
                model,
                tokenizer,
                sample,
                codec,
                sample_dir,
                max_new_tokens,
                asr,
                measure_stream=True,
            )
        except (ValueError, RuntimeError, OSError) as error:
            row = dict(status="failed", failure=type(error).__name__, error=str(error))
            if (sample_dir / "prediction.wav").is_file():
                row["wav"] = str(sample_dir / "prediction.wav")
        rows.append(dict(record_id=record_id, **row))

    fields = (
        "record_id",
        "status",
        "failure",
        "error",
        "stream_seconds",
        "cuda_peak_allocated_bytes",
        "frames",
        "budget_boundary",
        "text_eos_seen",
        "asr_status",
        "cer",
        "wer",
        "wav",
    )
    write_csv(output_dir / "samples.csv", rows, fields)
    timings = [row["stream_seconds"] for row in rows if "stream_seconds" in row]
    peaks = [
        row["cuda_peak_allocated_bytes"]
        for row in rows
        if row.get("cuda_peak_allocated_bytes") is not None
    ]
    quality = [row for row in rows if row.get("asr_status") == "measured"]
    chars = sum(row["reference_characters"] for row in quality)
    words = sum(row["reference_words"] for row in quality)
    return dict(
        protocol=dict(
            order="prepared_t2a_first_n",
            generation="greedy_cached_prompt_only",
            max_new_tokens=max_new_tokens,
            warmup=0,
            model_dtype=str(next(model.parameters()).dtype),
            timing="full model stream with device synchronization; excludes Mimi, ASR and file IO",
            cuda_memory="peak allocated bytes during model stream, including resident model",
        ),
        sample_ids=[record_id for _, record_id in selected],
        samples=rows,
        failed_samples=sum(row["status"] != "generated" for row in rows),
        truncated_samples=sum(row.get("budget_boundary", False) for row in rows),
        timed_samples=len(timings),
        median_stream_seconds=median(timings) if timings else None,
        cuda_peak_allocated_bytes=max(peaks, default=None),
        asr_measured_samples=len(quality),
        cer=sum(row["character_edits"] for row in quality) / chars if chars else None,
        wer=sum(row["word_edits"] for row in quality) / words if words else None,
    )
