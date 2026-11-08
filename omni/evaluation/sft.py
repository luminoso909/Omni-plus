"""SFT 验证、模态对照及有界音频评估；缺少 ASR 时不填写质量分数。"""

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from omni.inference.codec import MimiCodec, load_audio, save_audio
from omni.training.losses import loss_counts, sft_loss
from omni.training.sft import move_batch


def _inputs(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key not in {"labels", "audio_labels"}}


@torch.inference_mode()
def evaluate_sft(
    model: nn.Module, batches: list[dict], device: str = "cpu", stop_weight: float = 10
) -> dict:
    """聚合变长 batch 的有效监督，返回原损失、每路码本 CE 和 STOP 混淆计数。"""
    if not batches:
        raise ValueError("验证集不能为空。")

    model.eval()
    counts = loss_counts(batches)
    metrics = dict.fromkeys(("total", "text", "audio", "aux"), 0.0)
    books = model.config.num_audio_codebooks
    book_sums = torch.zeros(books, dtype=torch.float64)
    hits, targets, predicted = 0, 0, 0
    for batch in batches:
        batch = move_batch(batch, device)
        output = model(**_inputs(batch))
        losses = sft_loss(
            output,
            batch["labels"],
            batch["audio_labels"],
            model.config.audio_stop_token,
            stop_weight,
            counts,
            aux_scale=1 / len(batches),
        )
        for name in metrics:
            metrics[name] += float(losses[name])

        labels, logits = batch["audio_labels"], output.audio_logits.float()
        raw = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        stop = labels.eq(model.config.audio_stop_token)
        weights = 1 + stop.to(raw) * (stop_weight - 1)
        book_sums += (raw * weights).sum((0, 2)).double().cpu()
        chosen = logits.argmax(-1).eq(model.config.audio_stop_token) & labels.ne(-100)
        hits += int((chosen & stop).sum())
        targets += int(stop.sum())
        predicted += int(chosen.sum())

    metrics.update(
        codebook_loss=(book_sums / counts[1].clamp_min(1)).tolist(),
        text_tokens=counts[0],
        audio_tokens=counts[1].tolist(),
        stop_hits=hits,
        stop_targets=targets,
        stop_predicted=predicted,
        stop_recall=hits / targets if targets else None,
        stop_precision=hits / predicted if predicted else None,
    )
    return metrics


@torch.inference_mode()
def modality_swap(model: nn.Module, batch: dict, kind: str) -> dict:
    """交换两条同形状的真实特征，其他输入固定；不将变长媒体裁切冒充替换。"""
    if kind not in {"audio", "image"}:
        raise ValueError("kind 必须为 audio 或 image。")

    key = f"{kind}_features"
    features = batch[key]
    pair = next(
        (
            (i, j)
            for i, value in enumerate(features)
            if value is not None
            for j in range(i + 1, len(features))
            if features[j] is not None
            and value.shape == features[j].shape
            and not torch.equal(value, features[j])
        ),
        None,
    )
    if pair is None:
        return dict(status="not_measured", reason="需要两条同形状且内容不同的模态特征。")

    model.eval()
    altered = dict(batch)
    altered[key] = list(features)
    i, j = pair
    altered[key][i], altered[key][j] = features[j], features[i]
    base, changed = model(**_inputs(batch)), model(**_inputs(altered))
    # 只比较目标文本，避免把输入占位符本身变化误判为回答依赖模态。
    mask = batch["attention_mask"].bool() & batch["labels"].ne(-100)
    if not mask.any():
        return dict(status="not_measured", reason="缺少有效文本目标。")
    delta = (base.logits - changed.logits).abs()[mask].max()
    return dict(
        status="measured",
        kind=kind,
        swapped_rows=[i, j],
        answer_tokens=int(mask.sum()),
        text_max_abs_diff=float(delta),
    )


def _edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual))
            )
        previous = current
    return previous[-1]


def transcription_errors(reference: str, hypothesis: str) -> dict:
    """只压缩空白，不隐式改变大小写/标点；CER 去空白，WER 用空白分词。"""
    ref_chars, hyp_chars = "".join(reference.split()), "".join(hypothesis.split())
    ref_words, hyp_words = reference.split(), hypothesis.split()
    char_edits = _edit_distance(ref_chars, hyp_chars)
    word_edits = _edit_distance(ref_words, hyp_words)
    return dict(
        cer=char_edits / len(ref_chars) if ref_chars else None,
        wer=word_edits / len(ref_words) if ref_words else None,
        character_edits=char_edits,
        reference_characters=len(ref_chars),
        word_edits=word_edits,
        reference_words=len(ref_words),
        normalization="whitespace_only; case_and_punctuation_preserved",
    )


def load_local_asr(path: Path) -> Callable[[Tensor, int], str]:
    """加载本地 SenseVoice；返回仅接收波形的转写函数，不传入参考答案。"""
    from funasr import AutoModel

    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    model = AutoModel(model=str(path), device="cpu", disable_update=True, trust_remote_code=False)

    def transcribe(waveform: Tensor, sampling_rate: int) -> str:
        audio = load_audio(waveform, sampling_rate=sampling_rate, target_rate=16000)
        result = model.generate(input=audio.numpy(), language="auto", use_itn=False)
        if not result or "text" not in result[0]:
            raise RuntimeError("ASR 未返回转写文本。")
        return re.sub(r"<\|.*?\|>", "", result[0]["text"]).strip()

    return transcribe


@torch.inference_mode()
def generate_audio(
    model: nn.Module,
    tokenizer,
    sample: dict,
    codec: MimiCodec,
    output_dir: Path,
    max_new_tokens: int = 128,
    asr: Callable[[Tensor, int], str] | None = None,
    *,
    measure_stream: bool = False,
) -> dict:
    """从样本的 prompt 自回归生成；导出真实预测 PCM，绝不解码答案冒充预测。"""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖评估目录：{output_dir}")
    if max_new_tokens < 1:
        raise ValueError("生成预算必须为正整数。")
    positions = sample["labels"].ne(-100).nonzero().flatten()
    if not len(positions):
        raise ValueError("生成评估需要非空文本目标以确定 prompt 边界。")
    start = int(positions[0]) + 1
    if sample["audio_ids"][:, :start].ne(model.config.audio_pad_token).any():
        raise ValueError("此评估入口不隐式使用参考音频或 speaker 条件。")

    device = next(model.parameters()).device
    features = {
        key: [None if sample[key] is None else sample[key].to(device)]
        for key in ("audio_features", "image_features")
    }
    model.eval()
    ids = sample["input_ids"][None, :start].to(device)
    mask = sample["attention_mask"][None, :start].to(device)
    if measure_stream:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        start_time = time.perf_counter()

    events = list(
        model.stream_generate(
            ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **features,
        )
    )
    timing = {}
    if measure_stream:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        timing = dict(
            stream_seconds=time.perf_counter() - start_time,
            cuda_peak_allocated_bytes=(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        )
    text_ids = [int(token.item()) for token, _ in events if token is not None]
    frames = [codes for _, codes in events if codes is not None]
    eos = getattr(model.config, "eos_token_id", [])
    eos = [eos] if isinstance(eos, int) else eos
    reference = tokenizer.decode(sample["labels"][positions].tolist(), skip_special_tokens=True)
    result = dict(
        status="generated" if frames else "failed",
        failure=None if frames else "no_audio_frames",
        text=tokenizer.decode(text_ids, skip_special_tokens=True),
        reference=reference,
        generation_steps=len(events),
        frames=len(frames),
        budget_boundary=len(events) >= max_new_tokens,
        text_eos_seen=any(token in eos for token in text_ids),
        stream_ended_before_budget=len(events) < max_new_tokens,
        audio_stop_detail="not_exposed_by_stream_api",
        asr_status="not_measured",
        cer=None,
        wer=None,
        **timing,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    if frames:
        waveform = codec.decode(torch.cat(frames, dim=-1))
        if not waveform.numel() or not waveform.isfinite().all():
            result.update(status="failed", failure="invalid_decoded_waveform")
            return result
        save_audio(output_dir / "prediction.wav", waveform, codec.sampling_rate)
        result.update(
            wav=str(output_dir / "prediction.wav"),
            samples=waveform.numel(),
            sampling_rate=codec.sampling_rate,
        )
        if asr is not None:
            hypothesis = asr(waveform, codec.sampling_rate)
            result.update(
                asr_status="measured",
                transcription=hypothesis,
                **transcription_errors(reference, hypothesis),
            )
    return result
