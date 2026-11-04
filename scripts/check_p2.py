"""本地 SenseVoice/SigLIP2 与真实 Omni 权重的混合模态前向验收。"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

from omni.data import OmniCollator, OmniDataset
from omni.models import OmniPlusConfig, OmniPlusOmni
from omni.models.encoders import SenseVoiceEncoder, SiglipEncoder

ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


@torch.no_grad()
def check_p2(reference: Path, device: str = "cpu") -> dict:
    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = OmniPlusConfig(pad_token_id=0)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizers/minimind", local_files_only=True)

    # 只读本地外部编码器；前端保持 CPU FP32，编码器跟随指定设备
    audio_encoder = SenseVoiceEncoder(reference / "model/SenseVoiceSmall").to(device)
    vision_encoder = SiglipEncoder(reference / "model/siglip2-base-p32-256-ve").to(device)
    audio = reference / "dataset/eval_omni/audio-zh-01_你平时喜欢吃什么.mp3"
    image = reference / "dataset/eval_omni/image-01-orange-cat-moon-desk.jpg"
    audio_features = audio_encoder(audio)
    image_features = vision_encoder(image)
    torch.testing.assert_close(audio_encoder(audio), audio_features, rtol=0, atol=0)
    torch.testing.assert_close(vision_encoder(image.read_bytes()), image_features, rtol=0, atol=0)
    for encoder in (audio_encoder, vision_encoder):
        encoder.train()
        assert not any(module.training for module in encoder.modules())
        assert not any(param.requires_grad for param in encoder.parameters())
    for features, width in ((audio_features, 512), (image_features, 768)):
        assert features.ndim == 2 and features.shape[0] > 0 and features.shape[1] == width
        assert features.device.type == "cpu" and features.dtype == torch.float32
        assert torch.isfinite(features).all() and not features.requires_grad

    # 四种组合保留原始 batch 索引；使用真实特征，避免重复编码同一媒体
    records = []
    for has_audio, has_image in ((False, False), (True, False), (False, True), (True, True)):
        question = "请简要描述输入。"
        question = ("<audio>" if has_audio else "") + ("<image>" if has_image else "") + question
        records.append(
            dict(
                conversations=[
                    dict(role="user", content=question),
                    dict(role="assistant", content="这是一次前向验收。"),
                ],
                audio_features=audio_features if has_audio else None,
                image_features=image_features if has_image else None,
            )
        )
    dataset = OmniDataset(records, tokenizer, config, max_length=640, training=False)
    samples = [dataset[index] for index in range(len(dataset))]
    collate = OmniCollator(config)

    # 完整 195 键严格加载，既不重命名也不忽略缺失项
    torch.manual_seed(42)
    weights = reference / "out/sft_omni_768.pth"
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    assert torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"])
    model = OmniPlusOmni(config).eval()
    loaded = model.load_state_dict(state, strict=True)
    model.to(device)

    def forward(batch: dict):
        keys = (
            "input_ids",
            "audio_ids",
            "attention_mask",
            "audio_features",
            "image_features",
            "spk_emb",
        )
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
            if key in keys
        }
        return model(**inputs)

    batch = collate(samples)
    output = forward(batch)
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.audio_logits).all()
    rows = []
    for index, sample in enumerate(samples):
        length = len(sample["input_ids"])
        assert batch["attention_mask"][index].sum() == length
        assert batch["input_ids"][index].eq(config.audio_ids[0]).sum() == (
            0 if sample["audio_features"] is None else len(sample["audio_features"])
        )
        assert batch["input_ids"][index].eq(config.image_ids[0]).sum() == (
            0 if sample["image_features"] is None else len(sample["image_features"])
        )
        single = forward(collate([sample]))
        torch.testing.assert_close(
            output.logits[index, :length], single.logits[0], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            output.audio_logits[index, :, :length], single.audio_logits[0], rtol=1e-4, atol=1e-4
        )
        rows.append(
            dict(
                row=index,
                has_audio=sample["audio_features"] is not None,
                has_image=sample["image_features"] is not None,
                valid_tokens=length,
                text_max_abs_diff=float(
                    (output.logits[index, :length] - single.logits[0]).abs().max()
                ),
                audio_max_abs_diff=float(
                    (output.audio_logits[index, :, :length] - single.audio_logits[0]).abs().max()
                ),
            )
        )

    # 交换样本顺序仍与原 batch 对应，padding 的文本/音频标签均被忽略
    order = [3, 0, 2, 1]
    reordered = forward(collate([samples[index] for index in order]))
    torch.testing.assert_close(reordered.logits, output.logits[order], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        reordered.audio_logits, output.audio_logits[order], rtol=1e-4, atol=1e-4
    )
    padding = batch["attention_mask"] == 0
    assert batch["labels"][padding].eq(-100).all()
    assert (
        batch["audio_labels"]
        .masked_select(padding[:, None].expand_as(batch["audio_labels"]))
        .eq(-100)
        .all()
    )
    return dict(
        status="passed",
        device=device,
        dtype="float32",
        seed=42,
        threads=4,
        rtol=1e-4,
        atol=1e-4,
        weights=file_record(weights),
        loading=dict(
            keys=len(state), missing=loaded.missing_keys, unexpected=loaded.unexpected_keys
        ),
        assets=[
            file_record(reference / "model/SenseVoiceSmall/model.pt"),
            file_record(reference / "model/siglip2-base-p32-256-ve/model.safetensors"),
            file_record(audio),
            file_record(image),
        ],
        features=dict(
            audio_shape=list(audio_features.shape), image_shape=list(image_features.shape)
        ),
        batch=dict(
            text_logits=list(output.logits.shape), audio_logits=list(output.audio_logits.shape)
        ),
        frozen_eval=True,
        deterministic_features=True,
        permutation_invariant=True,
        padding_labels_ignored=True,
        rows=rows,
        quality_note="真实预训练编码器和 SFT 权重的前向、对齐验收；未评估识别/理解正确率。",
        checked_at_utc=datetime.now(UTC).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_p2((ROOT / args.reference_root).resolve(), args.device)
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / args.output if args.output else ROOT / "experiments/p2" / run / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"P2 passed: {path}")


if __name__ == "__main__":
    main()
