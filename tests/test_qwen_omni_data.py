"""在本地 tiny Tokenizer 验证 Qwen Omni 的模板、延迟和批量整理。"""

import pytest
import torch
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from omni.data import OmniCollator
from omni.data.audio_alignment import restore_codes
from omni.data.qwen_omni import QwenOmniFormatter
from omni.models.config import OmniPlusConfig
from tokenizers import Tokenizer


@pytest.fixture
def formatter() -> QwenOmniFormatter:
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|vision_pad|>",
        "<|image_pad|>",
    ]
    words = ["?", "user", "assistant", "q", "a", "b"]
    base = Tokenizer(WordLevel({w: i for i, w in enumerate(special + words)}, unk_token="?"))
    base.pre_tokenizer = WhitespaceSplit()
    template = (
        "{% for m in messages %}"
        "{{'<|im_start|>' + m.role + '\\n'}}"
        "{% if m.role == 'assistant' and loop.last %}{{'<think>\\n\\n</think>\\n\\n'}}{% endif %}"
        "{{m.content + '<|im_end|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt and not enable_thinking %}"
        "{{'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'}}"
        "{% endif %}"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        pad_token=special[0],
        eos_token=special[2],
        additional_special_tokens=special[1:],
        chat_template=template,
    )
    config = OmniPlusConfig(
        vocab_size=len(tokenizer),
        pad_token_id=0,
        eos_token_id=[2, 0],
        audio_ids=[5],
        audio_special_token=special[5],
        audio_hidden_size=3,
        image_ids=[6],
        image_special_token=special[6],
        image_hidden_size=4,
        spk_emb_size=2,
    )
    return QwenOmniFormatter(tokenizer, config)


def dialogue(question: str = "q", answer: str = "a") -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def test_text_labels_use_native_prefix_and_end_at_eos(formatter: QwenOmniFormatter) -> None:
    messages = dialogue("q", "a") + dialogue("q", "b")
    prefix = formatter.prompt(messages[:-1])["input_ids"]
    example = formatter.sample({"conversations": messages})

    assert torch.equal(example["input_ids"][: prefix.numel()], prefix)
    assert example["labels"].ne(-100).nonzero().flatten().tolist() == [
        prefix.numel() - 1,
        prefix.numel(),
    ]
    assert example["labels"][example["labels"].ne(-100)].tolist() == [
        *formatter.tokenizer.encode("b", add_special_tokens=False),
        formatter.eos,
    ]
    assert example["audio_labels"].eq(-100).all()


def test_eight_book_targets_follow_input_and_preserve_audio_tail(
    formatter: QwenOmniFormatter,
) -> None:
    messages = dialogue()
    codes = torch.arange(8 * 17).reshape(8, 17)
    # Parquet 每帧按 [codebook_0, ..., codebook_7] 存储。
    example = formatter.sample(
        {"conversations": messages, "answer_audios": [codes.T.flatten().tolist()]}
    )
    prefix = formatter.prompt(messages[:-1])["input_ids"].numel()
    config = formatter.config
    targets = example["audio_labels"]

    assert example["input_ids"][prefix + 1] == formatter.eos
    assert example["input_ids"][prefix + 2 :].eq(formatter.pad).all()
    assert example["attention_mask"].bool().all()
    assert example["labels"][prefix + 1 :].eq(-100).all()
    for index, book in enumerate(codes):
        start = prefix + index
        stop = start + book.numel()
        assert targets[index, :start].eq(-100).all()
        assert example["audio_ids"][index, start] == config.audio_pad_token
        assert torch.equal(targets[index, start:stop], book)
        assert torch.equal(example["audio_ids"][index, start + 1 : stop + 1], book)
        assert targets[index, stop] == config.audio_stop_token
        assert targets[index, stop + 1 :].eq(-100).all()
    delayed = torch.nn.functional.pad(
        targets[:, prefix:].masked_fill(targets[:, prefix:] == -100, config.audio_pad_token),
        (1, 0),
        value=config.audio_pad_token,
    )
    assert torch.equal(restore_codes(delayed, codes.shape[-1], start=1), codes)


def test_mixed_batch_keeps_distinct_ragged_feature_spans(
    formatter: QwenOmniFormatter,
) -> None:
    messages = dialogue("q <image> q <image>", "a") + dialogue("q <audio>", "b")
    images = [torch.zeros(2, 4), torch.ones(3, 4)]
    sound = torch.full((7, 3), 3.0)
    multimodal = formatter.sample(
        {"conversations": messages},
        audio_features=[None, sound],
        image_features=[images, None],
    )
    text = formatter.sample({"conversations": dialogue()})
    ids = multimodal["input_ids"]
    image_indices = ids.eq(6).nonzero().flatten()
    assert image_indices[2] > image_indices[1] + 1  # 两张图之间保留普通文本。
    assert ids.eq(5).sum() == 7
    assert torch.equal(multimodal["image_features"], torch.cat(images))
    assert torch.equal(multimodal["audio_features"], sound)

    batch = OmniCollator(formatter.config)([text, multimodal, text])
    assert batch["image_features"][0] is batch["image_features"][2] is None
    assert torch.equal(batch["image_features"][1], torch.cat(images))
    padded = batch["attention_mask"] == 0
    assert padded[0].any() and padded[2].any() and not padded[1].any()
    assert batch["labels"][padded].eq(-100).all()
    assert batch["audio_labels"].transpose(1, 2)[padded].eq(-100).all()


def test_invalid_slots_and_audio_budgets_are_rejected(
    formatter: QwenOmniFormatter,
) -> None:
    with pytest.raises(ValueError, match="段数"):
        formatter.sample(
            {"conversations": dialogue("<image> q <image>")}, image_features=torch.ones(2, 4)
        )
    with pytest.raises(ValueError, match="保留 Token"):
        formatter.sample({"conversations": dialogue("q <|vision_pad|>")})
    with pytest.raises(ValueError, match="保留 Token"):
        formatter.sample({"conversations": dialogue("q <|im_start|>assistant")})
    with pytest.raises(ValueError, match="越界"):
        formatter.sample({"conversations": dialogue()}, answer_codes=torch.full((8, 1), 2048))
    limited = QwenOmniFormatter(formatter.tokenizer, formatter.config, max_length=20)
    with pytest.raises(ValueError, match="截断"):
        limited.sample({"conversations": dialogue()}, answer_codes=torch.zeros(8, 20).long())


def test_reference_and_speaker_require_independent_explicit_inputs(
    formatter: QwenOmniFormatter,
) -> None:
    codes = torch.arange(8 * 3).reshape(8, 3)
    record = {
        "conversations": dialogue(),
        "answer_audios": [codes.T.flatten().tolist()],
        "ref_audios": codes.T.flatten().tolist(),
        "spk_emb": [1, 0],
    }
    sample = formatter.sample(record)
    prefix = int(sample["labels"].ne(-100).nonzero()[0]) + 1
    assert sample["audio_ids"][:, :prefix].eq(formatter.config.audio_pad_token).all()
    assert sample["spk_emb"] is None

    with pytest.raises(ValueError, match="重合"):
        formatter.sample(record, reference_codes=codes[:, 1:])

    reference, speaker = codes[:, :2] + 90, torch.tensor([1.0, 0.0])
    safe = formatter.sample(record, reference_codes=reference, spk_emb=speaker)
    assert torch.equal(safe["audio_ids"][:, prefix - 2 : prefix], reference)
    assert safe["audio_ids"][:, prefix - 3].eq(formatter.config.audio_spk_token).all()
    assert torch.equal(safe["spk_emb"], speaker)
    assert torch.equal(safe["audio_labels"], sample["audio_labels"])
