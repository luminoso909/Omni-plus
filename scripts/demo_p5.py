"""本地文本到语音 Demo，逐块把 Mimi 的 PCM 写入 WAV。"""

import argparse
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, MimiModel

from omni.inference import MimiCodec, stream_waveform
from omni.models import OmniPlusConfig, OmniPlusOmni

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="要交给 MiniMind 的文本提示")
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "minimind-o")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if not args.prompt.strip() or args.chunk_frames < 1 or args.max_new_tokens < 1:
        parser.error("需要非空提示、正的 chunk 帧数与生成步数。")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("当前进程的 MPS 不可用。")

    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / (args.output or Path(f"outputs/p5/{run}.wav"))
    if output.exists():
        raise FileExistsError(f"输出已存在，请指定新路径：{output}")

    # 复用本地 SFT、tokenizer 与 Mimi。
    torch.set_num_threads(4)
    reference = (ROOT / args.reference_root).resolve()
    tokenizer = AutoTokenizer.from_pretrained(reference / "model", local_files_only=True)
    config = OmniPlusConfig(pad_token_id=tokenizer.pad_token_id)
    model = OmniPlusOmni(config).eval()
    state = torch.load(
        reference / "out/sft_omni_768.pth", map_location="cpu", weights_only=True, mmap=True
    )
    if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
        raise ValueError("源 SFT embedding 与绑定的输出头不相等。")
    model.load_state_dict(state, strict=True)
    del state
    model.to(args.device)
    mimi = MimiModel.from_pretrained(
        reference / "model/mimi", local_files_only=True, dtype=torch.float32
    )
    codec = MimiCodec(mimi.to(args.device), num_codebooks=config.num_audio_codebooks)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(rendered, add_special_tokens=False)], device=args.device)
    events = model.stream_generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0,
        audio_temperature=0,
        use_cache=not args.no_cache,
    )

    # 首段 PCM 到达时创建文件，后续按块写入。关闭文件时写回 WAV 头。
    tokens, chunks = [], 0
    with ExitStack() as stack:
        stream = stack.enter_context(closing(stream_waveform(events, codec, args.chunk_frames)))
        wave = None
        for event in stream:
            if event.token is not None:
                tokens.append(int(event.token.item()))
            if event.waveform is not None:
                if wave is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    file = stack.enter_context(output.open("xb"))
                    wave = stack.enter_context(
                        sf.SoundFile(
                            file,
                            mode="w",
                            samplerate=codec.sampling_rate,
                            channels=1,
                            format="WAV",
                            subtype="PCM_16",
                        )
                    )
                wave.write(event.waveform.numpy())
                chunks += 1

    print(tokenizer.decode(tokens, skip_special_tokens=True))
    print(f"WAV {chunks} 块，{output}" if chunks else "本轮生成 0 个音频块。")


if __name__ == "__main__":
    main()
