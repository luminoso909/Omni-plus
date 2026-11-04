"""P3 显式真实资产验收：Mimi codec-only、旧版 logits 对照、文本到可播放 WAV。"""

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoTokenizer, MimiModel

from omni.data.audio_alignment import code_mask, delay_codes, restore_codes
from omni.inference import MimiCodec, load_audio, save_audio
from omni.models import OmniPlusConfig, OmniPlusOmni

ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def load_reference(reference_root: Path):
    # 原样执行旧文本核心与 Omni 类定义；跳过未使用的媒体包 import，不写旧目录缓存
    namespace = {"__name__": "omni_p3_reference", "os": os}
    path = reference_root / "model/model_minimind.py"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    import warnings

    namespace["warnings"] = warnings
    path = reference_root / "model/model_omni.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {
        "OmniConfig",
        "MMAudioProjector",
        "MMVisionProjector",
        "TalkerHead",
        "TalkerEmbedding",
        "TalkerModule",
        "MiniMindOmni",
    }
    tree.body = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
    ]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def compare(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return float((actual - expected).abs().max())


def check_codec(reference_root: Path, audio_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    model, info = MimiModel.from_pretrained(
        reference_root / "model/mimi",
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(info.values()):
        raise AssertionError(f"Mimi 权重加载差异：{info}")

    # codec-only 先验收真实波形，截取至 2 秒，避免混入 Talker 生成质量
    codec = MimiCodec(model)
    waveform = load_audio(audio_path, target_rate=codec.sampling_rate)[: 2 * codec.sampling_rate]
    codes = codec.encode(waveform)
    delayed = delay_codes(codes)
    restored = restore_codes(delayed, codes.shape[-1])
    torch.testing.assert_close(restored, codes, rtol=0, atol=0)
    reconstruction = codec.decode(restored, length=waveform.numel())
    assert reconstruction.shape == waveform.shape and torch.isfinite(reconstruction).all()
    assert bool(reconstruction.abs().max() > 0)
    source_path = output_dir / "codec-source.wav"
    reconstruction_path = output_dir / "codec-reconstruction.wav"
    save_audio(source_path, waveform)
    save_audio(reconstruction_path, reconstruction)
    mse = float((waveform - reconstruction).square().mean())
    signal = float(waveform.square().mean())
    return dict(
        status="passed",
        source=file_record(audio_path),
        waveform_samples=waveform.numel(),
        sampling_rate=codec.sampling_rate,
        codes_shape=list(codes.shape),
        code_id_range=[int(codes.min()), int(codes.max())],
        delayed_shape=list(delayed.shape),
        mask_acoustic_tokens=int(code_mask(delayed).sum()),
        alignment_exact=True,
        reconstruction_samples=reconstruction.numel(),
        reconstruction_finite=True,
        source_rms=math.sqrt(signal),
        reconstruction_rms=float(reconstruction.square().mean().sqrt()),
        waveform_mse=mse,
        waveform_snr_db=10 * math.log10(signal / max(mse, 1e-12)),
        metric_note="逐样点 SNR 未做时延校正，不等于感知音质或 ASR 准确率。",
        source_wav=str(source_path),
        reconstruction_wav=str(reconstruction_path),
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def check_omni(reference_root: Path) -> tuple[dict, torch.Tensor]:
    path = reference_root / "out/sft_omni_768.pth"
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
        raise AssertionError("源 embedding/head 不相等，不能以绑定方式加载。")
    config = OmniPlusConfig(pad_token_id=0)
    model = OmniPlusOmni(config).float().eval()
    info = model.load_state_dict(state, strict=True)
    source = load_reference(reference_root)
    reference = (
        source["MiniMindOmni"](
            source["OmniConfig"](), audio_encoder_path="/__p3_no_encoder__", vision_model_path=None
        )
        .float()
        .eval()
    )
    ref_info = reference.load_state_dict(state, strict=True)
    del state

    # 同权重/同输入：包含 pad 历史和 speaker 条件，比较全部八码本 logits
    tokens = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2]])
    audio_ids = torch.randint(0, config.audio_codebook_size, (1, 8, tokens.shape[-1]))
    audio_ids[:, :, :3] = config.audio_pad_token
    audio_ids[:, :, 1] = config.audio_spk_token
    speaker = torch.randn(1, config.spk_emb_size)
    current = model(tokens, audio_ids, spk_emb=speaker, use_cache=True)
    previous = reference(
        torch.cat((audio_ids, tokens[:, None]), dim=1), spk_emb=speaker, use_cache=True
    )
    old_audio = torch.stack(previous.audio_logits, dim=1)
    parity = dict(
        status="passed",
        weights=file_record(path),
        keys=len(model.state_dict()),
        missing_keys=info.missing_keys,
        unexpected_keys=info.unexpected_keys,
        reference_missing_keys=ref_info.missing_keys,
        reference_unexpected_keys=ref_info.unexpected_keys,
        source_tied_weights_equal=True,
        text_logits_max_abs_diff=compare(current.logits, previous.logits),
        audio_logits_max_abs_diff=compare(current.audio_logits, old_audio),
        audio_logits_shape=list(current.audio_logits.shape),
        cache_max_abs_diff=max(
            compare(value, other.transpose(1, 2))
            for pair, ref_pair in zip(
                current.past_key_values, previous.past_key_values, strict=True
            )
            for value, other in zip(pair, ref_pair, strict=True)
        ),
        rtol=1e-4,
        atol=1e-4,
    )
    del reference, current, previous, old_audio
    gc.collect()

    # 明确使用已有 SFT 权重；固定贪心生成，不训练、不修补输出 codes
    tokenizer = AutoTokenizer.from_pretrained(
        reference_root / "model", local_files_only=True, trust_remote_code=False
    )
    prompt = "你好，请用一句话介绍春天。"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(rendered, add_special_tokens=False)])
    started = time.perf_counter()
    steps = []
    handle = model.register_forward_hook(lambda *args: steps.append(None))
    text, codes = model.generate(
        ids, max_new_tokens=256, temperature=0, audio_temperature=0, top_k=0, top_p=1.0
    )
    handle.remove()
    if codes.shape[-1] == 0:
        raise AssertionError("真实 Omni 未生成完整声学帧，不能以伪造 codes 替代 TTS 验收。")
    if len(steps) == 256:
        raise AssertionError("音频尾部尚未全部 stop，不能将达到步数上限的样本标为完整生成。")
    assert ((codes >= 0) & (codes < config.audio_codebook_size)).all()
    result = dict(
        parity=parity,
        generation=dict(
            status="passed",
            prompt=prompt,
            rendered_prompt=rendered,
            text=tokenizer.decode(text[0, ids.shape[-1] :], skip_special_tokens=True),
            generated_ids=text[:, ids.shape[-1] :].tolist(),
            codes_shape=list(codes.shape),
            code_id_range=[int(codes.min()), int(codes.max())],
            max_new_tokens=256,
            forward_steps=len(steps),
            stopped_before_budget=True,
            temperature=0,
            audio_temperature=0,
            top_k=0,
            top_p=1.0,
            use_cache=True,
            speaker_condition=False,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    return result, codes.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--audio", type=Path, default=Path("output_benchmark/cer_00.wav"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p3"))
    parser.add_argument("--report", type=Path, default=Path("experiments/p3/results.json"))
    args = parser.parse_args()
    reference_root = (ROOT / args.reference_root).resolve()
    audio_path = reference_root / args.audio
    output_dir = ROOT / args.output_dir
    report_path = ROOT / args.report
    if report_path.exists():
        raise FileExistsError("报告已存在；请使用 --report 指定新 run，保留旧实验结果。")
    if any(
        (output_dir / name).exists()
        for name in ("codec-source.wav", "codec-reconstruction.wav", "omni-generated.wav")
    ):
        raise FileExistsError("波形输出已存在；请使用 --output-dir 指定新 run。")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    report = dict(
        run_id=report_path.stem,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        reference_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=reference_root, text=True
        ).strip(),
        checked_at_utc=datetime.now(UTC).isoformat(),
        device="cpu",
        dtype="float32",
        seed=42,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        train_steps=0,
        codec_weights=file_record(reference_root / "model/mimi/model.safetensors"),
        codec_model_card=file_record(reference_root / "model/mimi/README.md"),
        codec_license="CC-BY-4.0",
        codec_source_note="本地 Kyutai Mimi FP16 转换资产，验收时以 FP32 加载。",
        reference_code=[
            file_record(reference_root / f"model/{name}.py")
            for name in ("model_minimind", "model_omni")
        ],
    )
    report["codec_only"] = check_codec(reference_root, audio_path, output_dir)
    gc.collect()
    omni, codes = check_omni(reference_root)
    report.update(omni)
    gc.collect()

    # 已释放 Omni 后再加载 Mimi，生成 WAV 回读确认采样率和时长
    codec = MimiCodec(
        MimiModel.from_pretrained(
            reference_root / "model/mimi", local_files_only=True, dtype=torch.float32
        )
    )
    waveform = codec.decode(codes)
    assert torch.isfinite(waveform).all() and bool(waveform.abs().max() > 0)
    path = output_dir / "omni-generated.wav"
    save_audio(path, waveform, codec.sampling_rate)
    readback = load_audio(path, target_rate=codec.sampling_rate)
    assert readback.numel() == waveform.numel()
    report["generation"].update(
        waveform_path=str(path),
        waveform_samples=waveform.numel(),
        sampling_rate=codec.sampling_rate,
        waveform_rms=float(waveform.square().mean().sqrt()),
        playable_wav_readback=True,
        quality_note="只验证有效 PCM 波形与声学 token 顺序，未做听评、ASR 或说话人相似度评测。",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"P3 passed: codec-only、音频 logits 对照及 TTS WAV。报告：{report_path}")


if __name__ == "__main__":
    main()
