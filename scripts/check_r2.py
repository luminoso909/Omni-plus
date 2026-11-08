"""离线验收 1.7B Omni 的真实特征、监督、梯度与 PCM，独立保存每次结果。"""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
import yaml
from check_p2 import file_record
from check_qwen_backend import check_files, check_tokenizer
from torch import Tensor, nn
from torch.nn.functional import pad
from transformers import AutoTokenizer, MimiModel, Qwen3ForCausalLM

from omni.data.audio_alignment import delay_codes, restore_codes
from omni.data.omni_dataset import OmniCollator
from omni.data.prepare import content_hash
from omni.data.qwen_omni import QwenOmniFormatter, audio_codes
from omni.inference.codec import MimiCodec, load_audio
from omni.inference.streaming import check_chunk_parity, stream_waveform
from omni.models.config import OmniPlusConfig
from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder
from omni.models.model_omni_v2 import OmniPlusOmni
from omni.models.thinker.qwen_backend import QwenBackend
from omni.training.losses import sft_loss

ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = (
    "input_ids",
    "audio_ids",
    "attention_mask",
    "audio_features",
    "image_features",
    "spk_emb",
)
RTOL, ATOL = 1e-4, 1e-4


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def reference_assets(reference: Path) -> dict:
    """与历史 P2/P3 资产指纹核对，允许旧仓库搬迁。"""
    p2 = read_json("experiments/p2/20261104T065000931843Z/report.json")
    p3 = read_json("experiments/p3/run-02.json")
    records = {}
    for key, expected, relative in (
        ("sensevoice", p2["assets"][0], "model/SenseVoiceSmall/model.pt"),
        ("siglip", p2["assets"][1], "model/siglip2-base-p32-256-ve/model.safetensors"),
        ("mimi", p3["codec_weights"], "model/mimi/model.safetensors"),
    ):
        actual = file_record(reference / relative)
        if any(actual[field] != expected[field] for field in ("bytes", "sha256")):
            raise ValueError(f"{key} 的本地权重与历史验收 SHA 不符。")
        records[key] = actual
    return records


def pick_records(path: Path, revision: str) -> dict:
    """只读历史 P4 小样本，按来源行号与内容哈希选固定记录。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    indexed = {
        (meta["task"], meta["source_row"]): (record, meta)
        for record, meta in zip(payload["records"], payload["selections"], strict=True)
    }
    historical = {
        (meta["task"], meta["source_row"]): meta
        for meta in read_json("experiments/p4/data.json")["splits"]["train"]
    }
    picks = {
        "short": ("t2a", 7),
        "second": ("t2a", 14),
        "long": ("t2a", 0),
        "a2a": ("a2a", 31),
        "i2t": ("i2t", 91),
    }
    selected = {}
    for name, key in picks.items():
        record, meta = indexed[key]
        actual = content_hash({k: v for k, v in record.items() if not k.startswith("_source_")})
        if meta != historical[key] or actual != meta["record_id"]:
            raise ValueError(f"P4 样本 {key} 的内容与 manifest 不符。")
        if record["_source_revision"] != revision:
            raise ValueError(f"P4 样本 {key} 不属于固定的官方 revision。")
        selected[name] = (record, meta)
    return selected


def max_diff(left: Tensor, right: Tensor) -> float:
    torch.testing.assert_close(left, right, rtol=RTOL, atol=ATOL)
    return float((left - right).abs().max())


def check_alignment(row: dict, codes: Tensor, config: OmniPlusConfig, eos: int) -> dict:
    """t 位置的监督只在 t+1 写回模型历史，STOP 为每路最后一个目标。"""
    labels, history = row["audio_labels"], row["audio_ids"]
    text_positions = row["labels"].ne(-100).nonzero().flatten()
    start = int(text_positions[0]) + 1
    text_stop = int(text_positions[-1])
    length, frames = labels.shape[-1], codes.shape[-1]
    delayed = delay_codes(codes, config.audio_pad_token, config.audio_stop_token)
    expected = torch.full_like(labels, -100)
    shifted = delayed[:, 1:]
    expected[:, start : start + shifted.shape[-1]] = shifted.masked_fill(
        shifted.eq(config.audio_pad_token), -100
    )
    torch.testing.assert_close(labels, expected, rtol=0, atol=0)
    shifted_history = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), config.audio_pad_token)
    expected_history = pad(shifted_history, (1, 0), value=config.audio_pad_token)
    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
    aligned = labels[:, start:].masked_fill(labels[:, start:].eq(-100), config.audio_pad_token)
    torch.testing.assert_close(restore_codes(aligned, frames, start=0), codes, rtol=0, atol=0)
    assert row["labels"][text_stop] == eos
    assert row["attention_mask"].all()
    assert row["input_ids"][text_stop + 2 :].eq(config.pad_token_id).all()
    return dict(
        prompt_tokens=start,
        text_target_tokens=text_positions.numel(),
        text_first_target_at=int(text_positions[0]),
        text_eos_target_at=text_stop,
        audio_frames=frames,
        audio_label_shape=list(labels.shape),
        delay_shape=list(delayed.shape),
        valid_positions=length,
        lanes=[
            dict(
                first_target=start + i,
                last_frame_target=start + i + frames - 1,
                first_history=start + i + 1,
                stop_target=start + i + frames,
            )
            for i in range(config.num_audio_codebooks)
        ],
        audio_tail_after_text_eos=max(0, length - text_stop - 1),
        no_answer_codes_in_prompt=True,
        exact_restore=True,
    )


def inputs(batch: dict) -> dict:
    return {key: batch[key] for key in INPUT_KEYS}


def check_backward(model: OmniPlusOmni, batch: dict, frozen: dict[str, nn.Module]) -> dict:
    """复用 SFT CE；反传到投影和 Talker，并验证冻结 Qwen 的输入梯度路径。"""
    model.train()
    for module in frozen.values():
        if module is not model.thinker:
            module.train()
        assert not any(child.training for child in module.modules())
        assert not any(param.requires_grad for param in module.parameters())
    output = model(**inputs(batch))
    losses = sft_loss(output, batch["labels"], batch["audio_labels"], model.config.audio_stop_token)
    assert torch.isfinite(losses["total"])

    # 只用声学损失抽查 Audio Projector，排除纯文本 CE 提供梯度的可能。
    probe = torch.autograd.grad(losses["audio"], model.audio_proj.mlp[1].weight, retain_graph=True)[
        0
    ]
    assert probe.isfinite().all() and probe.abs().max() > 0
    losses["total"].backward()
    trainable = {
        "audio_projector": model.audio_proj,
        "vision_projector": model.vision_proj,
        "bridge": model.bridge,
        "talker": model.talker,
    }
    stats = {}
    for name, module in trainable.items():
        params = dict(module.named_parameters())
        missing = {key for key, param in params.items() if param.grad is None}
        expected = {"spk_proj.weight"} if name == "talker" else set()
        assert missing == expected, (name, sorted(missing))
        grads = [param.grad for param in params.values() if param.grad is not None]
        assert grads and all(grad.isfinite().all() for grad in grads)
        nonzero = sum(int(grad.abs().max() > 0) for grad in grads)
        assert nonzero == len(grads), (name, nonzero, len(grads))
        stats[name] = dict(
            parameters=sum(p.numel() for p in params.values()),
            nonzero_gradient_tensors=nonzero,
            disconnected_tensors=sorted(missing),
        )
    for name, module in frozen.items():
        assert all(param.grad is None for param in module.parameters())
        stats[name] = dict(parameters=sum(p.numel() for p in module.parameters()), frozen=True)
    result = dict(
        text_logits=list(output.logits.shape),
        audio_logits=list(output.audio_logits.shape),
        loss={key: float(value.detach()) for key, value in losses.items()},
        modules=stats,
        audio_loss_to_audio_projector_max_gradient=float(probe.abs().max()),
        optimizer_steps=0,
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


@torch.inference_mode()
def check_mixed(model: OmniPlusOmni, collate: OmniCollator, rows: list[dict]) -> dict:
    batch = collate(rows)
    padding = ~batch["attention_mask"].bool()
    assert padding.any() and batch["labels"][padding].eq(-100).all()
    assert batch["audio_labels"].masked_select(padding[:, None]).eq(-100).all()
    assert batch["audio_labels"][1:].eq(-100).all()
    output = model(**inputs(batch))
    assert output.logits.isfinite().all() and output.audio_logits.isfinite().all()
    order = [2, 0, 1]
    reversed_output = model(**inputs(collate([rows[i] for i in order])))
    return dict(
        routes=["A2A", "I2T", "A2T_diagnostic"],
        order=order,
        text_logits=list(output.logits.shape),
        audio_logits=list(output.audio_logits.shape),
        padding_loss_ignored=True,
        text_permutation_max_abs_diff=max_diff(reversed_output.logits, output.logits[order]),
        audio_permutation_max_abs_diff=max_diff(
            reversed_output.audio_logits, output.audio_logits[order]
        ),
    )


@torch.inference_mode()
def check_causal_cache(model: OmniPlusOmni, row: dict, timing: dict) -> dict:
    config = model.config
    ids, audio = row["input_ids"][None], row["audio_ids"][None]
    start = timing["prompt_tokens"]
    mask = torch.ones_like(ids[:, : start + 1])
    prefix = model(ids[:, :start], audio_ids=audio[:, :, :start], use_cache=True, logits_to_keep=1)
    cache = prefix.past_key_values
    assert cache.thinker.get_seq_length() == start
    assert len(cache.talker) == config.num_talker_hidden_layers
    assert all(key.shape[-2] == start for key, _ in cache.talker)
    cached = model(
        ids[:, start : start + 1],
        audio_ids=audio[:, :, start : start + 1],
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    replay = model(ids[:, : start + 1], audio_ids=audio[:, :, : start + 1], logits_to_keep=1)
    padded = model(
        pad(ids[:, : start + 1], (3, 0), value=config.pad_token_id),
        audio_ids=pad(audio[:, :, : start + 1], (3, 0), value=config.audio_pad_token),
        attention_mask=pad(mask, (3, 0)),
        use_cache=True,
        logits_to_keep=1,
    )
    updated = cached.past_key_values
    assert updated.thinker.get_seq_length() == start + 1
    assert all(key.shape[-2] == start + 1 for key, _ in updated.talker)
    assert padded.past_key_values.thinker is not updated.thinker
    assert padded.past_key_values.thinker.get_seq_length() == start + 4

    # 改动更晚的声学 history，未来 token 不得改变当前位置及以前的预测。
    limit = start + 5
    altered = audio.clone()
    valid = altered[:, :, limit:].lt(config.audio_codebook_size)
    assert valid.any()
    altered[:, :, limit:] = torch.where(
        valid, (altered[:, :, limit:] + 1) % config.audio_codebook_size, altered[:, :, limit:]
    )
    base = model(ids, audio_ids=audio)
    future = model(ids, audio_ids=altered)
    return dict(
        prefill_length=start,
        cache_length=start + 1,
        independent_padded_cache_length=start + 4,
        thinker_cache=type(updated.thinker).__name__,
        talker_cache_layers=len(updated.talker),
        text_cache_max_abs_diff=max_diff(cached.logits, replay.logits),
        audio_cache_max_abs_diff=max_diff(cached.audio_logits, replay.audio_logits),
        text_left_padding_max_abs_diff=max_diff(padded.logits, replay.logits),
        audio_left_padding_max_abs_diff=max_diff(padded.audio_logits, replay.audio_logits),
        future_change_from=limit,
        causal_audio_max_abs_diff=max_diff(
            base.audio_logits[:, :, :limit], future.audio_logits[:, :, :limit]
        ),
    )


def check_stream(
    model: OmniPlusOmni, codec: MimiCodec, prompt: Tensor, known: Tensor, eos: int
) -> dict:
    """实际运行模型和缓存，固定 logits 选择，用真实码本验收 STOP/PCM 边界。"""
    config = model.config
    frames, books = known.shape[-1], known.shape[0]

    def run(budget: int, *, cancel: bool = False) -> tuple[list, int]:
        step = 0

        def force(_module, _args, output) -> None:
            nonlocal step
            output.logits.fill_(-torch.inf)
            output.logits[..., eos] = 0
            output.audio_logits.fill_(-torch.inf)
            for i in range(books):
                frame = step - 1 - i
                target = int(known[i, frame]) if 0 <= frame < frames else config.audio_stop_token
                output.audio_logits[:, i, :, target] = 0
            step += 1

        with model.register_forward_hook(force):
            source = model.stream_generate(prompt, max_new_tokens=budget, use_cache=True)
            waveform = stream_waveform(source, codec, chunk_frames=4)
            if cancel:
                next(waveform)
                waveform.close()
                assert source.gi_frame is None
                return [], step
            return list(waveform), step

    natural, steps = run(frames + books + 4)
    chunks = [event for event in natural if event.waveform is not None]
    assert [int(event.token.item()) for event in natural if event.token is not None] == [eos]
    assert [event.frames for event in chunks] == [4, 1] and steps == frames + books + 1
    pcm = torch.cat([event.waveform for event in chunks])
    pcm_error = max_diff(pcm, codec.decode(known[None]))
    capped, capped_steps = run(books + 3)
    assert sum(event.frames for event in capped) == 3 and capped_steps == books + 3
    _, cancelled_steps = run(frames + books + 4, cancel=True)
    assert cancelled_steps == 1
    return dict(
        controlled_logits=True,
        natural_steps=steps,
        natural_chunk_frames=[4, 1],
        pcm_samples=pcm.numel(),
        pcm_max_abs_diff=pcm_error,
        capped_steps=capped_steps,
        capped_complete_frames=3,
        cancellation_closed_upstream=True,
    )


def check_r2(reference: Path, model_dir: Path, split: Path, threads: int) -> dict:
    started = time.perf_counter()
    config_file = ROOT / "configs/model/qwen_omni_r2.yaml"
    settings = yaml.safe_load(config_file.read_text())
    lock = yaml.safe_load((ROOT / "configs/assets_v2.yaml").read_text())["assets"]["qwen3_1_7b"]
    config = OmniPlusConfig(**settings["model"])
    torch.set_num_threads(threads)
    torch.manual_seed(settings["seed"])
    locked_files = check_files(model_dir, lock["files"])
    assets = reference_assets(reference)
    data_lock = read_json("experiments/p4/subset-assets-v2.json")
    picked = pick_records(split, data_lock["revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    formatter = QwenOmniFormatter(tokenizer, config)
    collate = OmniCollator(config)
    speech = SenseVoiceEncoder(reference / "model/SenseVoiceSmall")
    vision = SiglipEncoder(reference / "model/siglip2-base-p32-256-ve")
    a2a = picked["a2a"][0]
    question = a2a["question_audios"][0]
    audio_features = speech(question)
    image_features = vision(picked["i2t"][0]["image_bytes"])
    assert audio_features.ndim == 2 and audio_features.shape[1] == 512
    assert image_features.shape == (64, 768)
    assert not audio_features.requires_grad and not image_features.requires_grad
    rows = dict(
        a2a=formatter.sample(a2a, audio_features=audio_features),
        i2t=formatter.sample(picked["i2t"][0], image_features=image_features),
    )
    transcript = json.loads(a2a["conversations"])[0]["content"].replace("<audio>", "").strip()
    assert transcript
    rows["a2t"] = formatter.sample(
        dict(
            conversations=[
                dict(role="user", content="<audio> Transcribe the speech."),
                dict(role="assistant", content=transcript),
            ]
        ),
        audio_features=audio_features,
    )

    timing, codes = {}, {}
    for name in ("short", "second", "long", "a2a"):
        record, _ = picked[name]
        codes[name] = audio_codes(record["answer_audios"][0], config)
        if name != "a2a":
            rows[name] = formatter.sample(record)
        timing[name] = check_alignment(rows[name], codes[name], config, formatter.eos)
    assert timing["long"]["audio_frames"] == 300 and timing["long"]["audio_tail_after_text_eos"] > 0
    assert codes["short"].shape[-1] != codes["second"].shape[-1]

    mimi, mimi_info = MimiModel.from_pretrained(
        reference / "model/mimi",
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    assert not any(mimi_info.values())
    codec = MimiCodec(mimi, num_codebooks=config.num_audio_codebooks)
    waveform = load_audio(question, target_rate=codec.sampling_rate)
    reencoded = codec.encode(waveform)
    torch.testing.assert_close(
        restore_codes(delay_codes(reencoded), reencoded.shape[-1]), reencoded, rtol=0, atol=0
    )
    assert codec.decode(reencoded, length=waveform.numel()).shape == waveform.shape
    pcm = {
        name: check_chunk_parity(codec, codes[name][None], chunk_frames=4)
        for name in ("short", "second")
    }
    pcm["encode_waveform"] = dict(
        samples=waveform.numel(),
        codes_shape=list(reencoded.shape),
        exact_restore=True,
        sampling_rate=codec.sampling_rate,
    )

    thinker, thinker_info = Qwen3ForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    assert not any(thinker_info.values())
    token_info = check_tokenizer(tokenizer, thinker)
    model = OmniPlusOmni(QwenBackend(thinker), config).eval()
    assert len(model.talker.layers) == 4 and model.config.talker_hidden_size == 768
    assert codec.codebook_size == config.audio_codebook_size
    frozen = dict(thinker=model.thinker, sensevoice=speech, siglip=vision, mimi=codec)
    gradients = check_backward(model, collate([rows["a2a"], rows["i2t"]]), frozen)
    mixed = check_mixed(model, collate, [rows["a2a"], rows["i2t"], rows["a2t"]])
    causal = check_causal_cache(model, rows["short"], timing["short"])
    second_causal = check_causal_cache(model, rows["second"], timing["second"])
    with torch.inference_mode():
        for name in ("short", "second", "long"):
            output = model(**inputs(collate([rows[name]])), logits_to_keep=1)
            assert output.logits.isfinite().all() and output.audio_logits.isfinite().all()
            timing[name]["forward_text_logits"] = list(output.logits.shape)
            timing[name]["forward_audio_logits"] = list(output.audio_logits.shape)
    stream = check_stream(
        model,
        codec,
        rows["short"]["input_ids"][None, : timing["short"]["prompt_tokens"]],
        codes["short"][:, :5],
        formatter.eos,
    )
    return dict(
        status="passed",
        environment=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            device="cpu",
            dtype="float32",
            threads=threads,
            seed=settings["seed"],
            local_files_only=True,
        ),
        qwen=dict(
            repo=lock["repo_id"],
            revision=lock["revision"],
            license=lock["license"],
            assets=locked_files,
            loading=thinker_info,
            tokenizer=token_info,
        ),
        reference_assets=assets,
        config=file_record(config_file),
        mimi_loading=mimi_info,
        data=dict(
            split=file_record(split),
            source_repo=data_lock["repo_id"],
            source_revision=data_lock["revision"],
            selected={name: meta for name, (_, meta) in picked.items()},
            a2t="diagnostic from the A2A question audio and source transcript",
            reference_policy="upstream reference audio and speaker columns intentionally ignored",
            features=dict(audio=list(audio_features.shape), image=list(image_features.shape)),
        ),
        timing=timing,
        gradients=gradients,
        mixed=mixed,
        causal_cache=causal,
        second_causal_cache=second_causal,
        codec=pcm,
        streaming=stream,
        tolerances=dict(rtol=RTOL, atol=ATOL),
        elapsed_seconds=time.perf_counter() - started,
        checked_at_utc=datetime.now(UTC).isoformat(),
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/qwen3-1.7b")
    parser.add_argument("--split", type=Path, default=ROOT / ".cache/p4/prepared/train.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"验收目录已存在，请用新路径：{output}")
    report = check_r2(
        ROOT / args.reference_root, ROOT / args.model_dir, ROOT / args.split, args.threads
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"R2 真实资产验收通过：{output / 'manifest.json'}")


if __name__ == "__main__":
    main()
