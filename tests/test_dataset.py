"""本地合成记录验证数据对齐；不下载 tokenizer、媒体或模型。"""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from omni.data import OmniCollator, OmniDataset
from omni.models import OmniPlusConfig, OmniPlusOmni


class TinyTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 16]

    def __init__(self) -> None:
        self.special = {
            "<s>": 1,
            "</s>": 2,
            "user\n": 4,
            "assistant\n": 5,
            "system\n": 6,
            "\n": 7,
            "<think>": 10,
            "</think>": 11,
            "<|image_pad|>": 12,
            "<|audio_pad|>": 16,
        }

    def __len__(self) -> int:
        return 64

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        while text:
            match = next((token for token in self.special if text.startswith(token)), None)
            if match is not None:
                ids.append(self.special[match])
                text = text[len(match) :]
            else:
                ids.append(20 + ord(text[0]) - ord("a") if "a" <= text[0] <= "z" else 49)
                text = text[1:]
        return ids

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(input_ids=self.encode(text))

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str | list[int]:
        text = "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in messages)
        if add_generation_prompt:
            text += "<s>assistant\n"
        return self.encode(text) if tokenize else text


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def config() -> OmniPlusConfig:
    return OmniPlusConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        vocab_size=64,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=2,
        num_talker_hidden_layers=1,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=24,
        num_audio_codebooks=2,
        talker_adapter_rank=4,
        audio_codebook_size=16,
        audio_vocab_size=20,
        audio_pad_token=17,
        audio_stop_token=18,
        audio_spk_token=19,
        spk_emb_size=6,
        audio_hidden_size=8,
        image_hidden_size=10,
        image_token_len=2,
        think_end_ids=[11, 7, 7],
    )


def conversation(answer: str = "abcdef", question: str = "q") -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def assistant_start(tokenizer: TinyTokenizer, messages: list[dict]) -> int:
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    return len(prefix)


def test_text_labels_are_shifted_once(tokenizer: TinyTokenizer, config: OmniPlusConfig) -> None:
    messages = conversation("ab")
    row = OmniDataset([{"conversations": messages}], tokenizer, config)[0]
    full = torch.tensor(tokenizer.apply_chat_template(messages, tokenize=True))
    start = assistant_start(tokenizer, messages)
    expected = torch.full_like(full, -100)
    expected[start:] = full[start:]

    assert torch.equal(row["input_ids"], full[:-1])
    assert torch.equal(row["labels"], expected[1:])
    assert row["attention_mask"].bool().all()
    assert row["audio_labels"].eq(-100).all()
    assert row["audio_ids"].eq(config.audio_pad_token).all()


def test_only_last_assistant_is_supervised(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation("ab") + conversation("cd", "r")
    row = OmniDataset([{"conversations": json.dumps(messages)}], tokenizer, config)[0]
    targets = row["labels"][row["labels"] != -100].tolist()
    assert targets == tokenizer.encode("cd</s>\n")


def test_audio_delay_and_stop_labels(tokenizer: TinyTokenizer, config: OmniPlusConfig) -> None:
    messages = conversation()
    row = OmniDataset(
        [{"conversations": messages, "answer_audios": [[3, 8, 4, 9]]}], tokenizer, config
    )[0]
    start = assistant_start(tokenizer, messages)
    for codebook, expected in enumerate(([3, 4, 18], [8, 9, 18])):
        first = start + codebook
        assert row["audio_labels"][codebook, first : first + 3].tolist() == expected
        assert row["audio_ids"][codebook, first + 1 : first + 4].tolist() == expected
        assert row["audio_labels"][codebook, :first].eq(-100).all()
        assert row["audio_labels"][codebook, first + 3 :].eq(-100).all()


def test_audio_starts_after_thinking(tokenizer: TinyTokenizer, config: OmniPlusConfig) -> None:
    messages = conversation("<think>ab</think>\n\ncdefgh")
    row = OmniDataset([{"conversations": messages, "answer_audios": [[3, 8]]}], tokenizer, config)[
        0
    ]
    before_audio = tokenizer.encode("<think>ab</think>\n\n")
    first = assistant_start(tokenizer, messages) + len(before_audio)
    assert row["audio_labels"][0, first].item() == 3
    assert row["audio_labels"][1, first + 1].item() == 8
    assert row["audio_labels"][:, :first].eq(-100).all()


def test_reference_and_speaker_prefix_are_not_targets(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation()
    row = OmniDataset(
        [
            {
                "conversations": messages,
                "answer_audios": [[3, 8]],
                "ref_audios": [5, 10, 6, 11],
                "spk_emb": [0.2] * 6,
            }
        ],
        tokenizer,
        config,
    )[0]
    start = assistant_start(tokenizer, messages)
    assert row["audio_ids"][:, start - 2 : start].tolist() == [[5, 6], [10, 11]]
    assert row["audio_ids"][:, start - 3].eq(config.audio_spk_token).all()
    assert row["audio_labels"][:, :start].eq(-100).all()
    torch.testing.assert_close(row["spk_emb"], torch.full((6,), 0.2))


def test_short_text_retains_long_audio_tail(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation("a")
    codes = [[i % 16 for i in range(10)], [(i + 3) % 16 for i in range(10)]]
    row = OmniDataset([{"conversations": messages, "answer_audios": [codes]}], tokenizer, config)[0]
    for codebook in range(2):
        valid = row["audio_labels"][codebook] != -100
        assert valid.sum().item() == 11
        assert row["audio_labels"][codebook, valid][-1].item() == config.audio_stop_token
    text_end = len(tokenizer.apply_chat_template(messages, tokenize=True)) - 1
    assert row["input_ids"].shape[0] > text_end
    assert row["attention_mask"][text_end:].bool().all()
    assert row["labels"][text_end:].eq(-100).all()


def test_too_small_length_does_not_silently_drop_audio(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {"conversations": conversation("a"), "answer_audios": [[3, 8] * 10]}
    with pytest.raises(ValueError):
        OmniDataset([record], tokenizer, config, max_length=15)[0]


def test_multi_turn_features_keep_user_order(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation("abc", "<audio><image>") + conversation("def", "<audio><image>")
    audios = [torch.ones(1, 8), torch.full((2, 8), 2.0)]
    images = [torch.zeros(2, 10), torch.full((3, 10), 3.0)]
    row = OmniDataset(
        [{"conversations": messages, "audio_features": audios, "image_features": images}],
        tokenizer,
        config,
    )[0]
    torch.testing.assert_close(row["audio_features"], torch.cat(audios))
    torch.testing.assert_close(row["image_features"], torch.cat(images))
    assert row["input_ids"].eq(16).sum().item() == 3
    assert row["input_ids"].eq(12).sum().item() == 5


def test_multiple_images_and_zero_features(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation(question="<image>q<image>"),
        "image_features": torch.zeros(4, 10),
    }
    row = OmniDataset([record], tokenizer, config)[0]
    assert row["input_ids"].eq(12).sum().item() == 4
    assert row["image_features"].shape == (4, 10)
    assert row["image_features"].eq(0).all()


def test_truncation_cannot_cut_modal_span(tokenizer: TinyTokenizer, config: OmniPlusConfig) -> None:
    record = {
        "conversations": conversation(question="<audio>"),
        "audio_features": torch.randn(5, 8),
    }
    with pytest.raises(ValueError):
        OmniDataset([record], tokenizer, config, max_length=7)[0]


def test_raw_media_callbacks_preserve_turn_order(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    seen = []

    def encode_audio(payload: bytes) -> torch.Tensor:
        seen.append(payload)
        return torch.full((1, 8), float(payload[0]))

    messages = conversation("abc", "<audio>") + conversation("def", "<audio>")
    row = OmniDataset(
        [{"conversations": messages, "question_audios": [b"a", b"b"]}],
        tokenizer,
        config,
        audio_encoder=encode_audio,
    )[0]
    assert seen == [b"a", b"b"]
    assert row["audio_features"][:, 0].tolist() == [97.0, 98.0]


def test_eval_is_deterministic_even_with_sampling_enabled(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation("abc") + conversation("def", "r"),
        "answer_audios": [[3, 8], [4, 9]],
    }
    dataset = OmniDataset([record], tokenizer, config, training=False, scheduled_sampling=1.0)
    first, second = dataset[0], dataset[0]
    for key in ("input_ids", "labels", "audio_ids", "audio_labels", "attention_mask"):
        assert torch.equal(first[key], second[key])
    assert first["labels"][first["labels"] != -100].tolist() == tokenizer.encode("def</s>\n")


def test_training_corruption_preserves_modal_tokens(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation(question="<audio><image>q"),
        "audio_features": torch.ones(2, 8),
        "image_features": torch.ones(2, 10),
        "answer_audios": [[3, 8, 4, 9]],
    }
    plain = OmniDataset([record], tokenizer, config)[0]
    sampled = OmniDataset([record], tokenizer, config, training=True, scheduled_sampling=1.0)[0]
    for marker in (12, 16):
        assert torch.equal(plain["input_ids"] == marker, sampled["input_ids"] == marker)
    assert torch.equal(plain["labels"], sampled["labels"])
    assert torch.equal(plain["audio_labels"], sampled["audio_labels"])


def test_local_parquet_and_multiple_sources(
    tokenizer: TinyTokenizer, config: OmniPlusConfig, tmp_path
) -> None:
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for path, answer in zip(paths, ("ab", "cd"), strict=True):
        table = pa.Table.from_pylist([{"conversations": json.dumps(conversation(answer))}])
        pq.write_table(table, path)
    dataset = OmniDataset(paths, tokenizer, config)
    comma_dataset = OmniDataset(",".join(map(str, paths)), tokenizer, config)
    assert len(dataset) == len(comma_dataset) == 2
    assert dataset[0]["labels"][dataset[0]["labels"] != -100].tolist() == tokenizer.encode(
        "ab</s>\n"
    )
    assert torch.equal(dataset[1]["input_ids"], comma_dataset[1]["input_ids"])


def test_collator_to_model_forward_and_backward(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    records = [
        {
            "conversations": conversation("abc", "<audio><image>q"),
            "audio_features": torch.randn(2, 8),
            "image_features": torch.randn(2, 10),
            "answer_audios": [[3, 8, 4, 9]],
            "spk_emb": [0.1] * 6,
        },
        {"conversations": conversation("a")},
    ]
    dataset = OmniDataset(records, tokenizer, config)
    items = [dataset[0], dataset[1]]
    batch = OmniCollator(config)(items)
    assert batch["audio_features"][1] is None and batch["image_features"][1] is None
    assert batch["audio_ids"].shape[:2] == (2, 2)
    length = items[1]["input_ids"].shape[0]
    assert batch["attention_mask"][1, length:].eq(0).all()
    assert batch["labels"][1, length:].eq(-100).all()
    assert batch["audio_labels"][1].eq(-100).all()

    model = OmniPlusOmni(config).train()
    model_args = {
        key: value for key, value in batch.items() if key not in ("labels", "audio_labels")
    }
    output = model(**model_args)
    text_loss = F.cross_entropy(output.logits.reshape(-1, 64), batch["labels"].reshape(-1))
    audio_loss = F.cross_entropy(
        output.audio_logits.reshape(-1, 20), batch["audio_labels"].reshape(-1)
    )
    loss = text_loss + audio_loss + output.aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert model.audio_proj.mlp[1].weight.grad is not None
    assert model.vision_proj.mlp[1].weight.grad is not None


def test_training_turn_selection_keeps_matching_media(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation("abc", "<audio>") + conversation("def", "<audio>"),
        "audio_features": [torch.ones(1, 8), torch.full((2, 8), 2.0)],
    }
    for seed in range(4):
        row = OmniDataset([record], tokenizer, config, training=True, seed=seed)[0]
        target = row["labels"][row["labels"] != -100].tolist()
        count = 1 if target == tokenizer.encode("abc</s>\n") else 3
        assert row["audio_features"].shape == (count, 8)
        assert row["input_ids"].eq(16).sum().item() == count
    assert record["conversations"][0]["content"] == "<audio>"


def test_empty_raw_media_does_not_require_encoders(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation("abc"),
        "question_audios": [None],
        "image_bytes": [None, b""],
    }
    row = OmniDataset([record], tokenizer, config)[0]
    assert row["audio_features"] is None and row["image_features"] is None


def test_single_user_nested_images_are_flattened_once(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    seen = []

    def encode_image(payload: bytes) -> torch.Tensor:
        seen.append(payload)
        return torch.full((1, 10), float(payload[0]))

    record = {"conversations": conversation(question="<image>"), "image_bytes": [[b"a", b"b"]]}
    row = OmniDataset([record], tokenizer, config, vision_encoder=encode_image)[0]
    assert seen == [b"a", b"b"]
    assert row["image_features"][:, 0].tolist() == [97.0, 98.0]


def test_training_media_turn_selection_does_not_merge_future_images(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    record = {
        "conversations": conversation("abc", "<image>") + conversation("def", "<image>"),
        "image_bytes": [[b"a"], [b"b", b"c"]],
    }

    def encode_image(payload: bytes) -> torch.Tensor:
        return torch.full((1, 10), float(payload[0]))

    for seed in range(4):
        row = OmniDataset(
            [record], tokenizer, config, training=True, seed=seed, vision_encoder=encode_image
        )[0]
        target = row["labels"][row["labels"] != -100].tolist()
        expected = [97.0] if target == tokenizer.encode("abc</s>\n") else [97.0, 98.0, 99.0]
        assert row["image_features"][:, 0].tolist() == expected


def test_scheduled_sampling_does_not_corrupt_assistant_header(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation("abcdef")
    plain = OmniDataset([{"conversations": messages}], tokenizer, config)[0]
    noisy = OmniDataset(
        [{"conversations": messages}], tokenizer, config, training=True, scheduled_sampling=1.0
    )[0]
    start = assistant_start(tokenizer, messages)
    assert torch.equal(noisy["input_ids"][:start], plain["input_ids"][:start])


@pytest.mark.parametrize("reference", [None, [3, 8]])
def test_truncation_must_retain_assistant_supervision(
    tokenizer: TinyTokenizer, config: OmniPlusConfig, reference
) -> None:
    record = {"conversations": conversation("abc"), "ref_audios": reference}
    with pytest.raises(ValueError):
        OmniDataset([record], tokenizer, config, max_length=3)[0]


def test_real_hf_tokenizer_chat_template(config: OmniPlusConfig) -> None:
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from tokenizers import Tokenizer

    # 本地构造真正的 HF tokenizer，避免只验证测试替身的模板行为。
    names = [f"token{i}" for i in range(64)]
    for index, name in {
        0: "<pad>",
        1: "<s>",
        2: "</s>",
        3: "<unk>",
        4: "user",
        5: "assistant",
        6: "q",
        7: "a",
        8: "b",
        12: "<|image_pad|>",
        16: "<|audio_pad|>",
    }.items():
        names[index] = name
    raw = Tokenizer(WordLevel({name: i for i, name in enumerate(names)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|image_pad|>", "<|audio_pad|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<s>' + message['role'] + '\n' + message['content'] + '</s>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant\n' }}{% endif %}"
    )
    row = OmniDataset([{"conversations": conversation("a b")}], tokenizer, config)[0]
    assert row["labels"][row["labels"] != -100].tolist() == [7, 8, 2]
    batch = OmniCollator(config)([row])
    assert batch["attention_mask"].bool().all()


def test_scheduled_sampling_preserves_and_never_injects_special_tokens(
    tokenizer: TinyTokenizer, config: OmniPlusConfig
) -> None:
    messages = conversation("<think>ab</think>\n\ncdefgh")
    record = {"conversations": messages, "answer_audios": [[3, 8] * 12]}
    plain = OmniDataset([record], tokenizer, config)[0]
    noisy = OmniDataset([record], tokenizer, config, training=True, scheduled_sampling=1.0, seed=1)[
        0
    ]
    special = torch.tensor(tokenizer.all_special_ids)
    protected = torch.isin(plain["input_ids"], special)
    assert torch.equal(noisy["input_ids"][protected], plain["input_ids"][protected])
    assert not torch.isin(noisy["input_ids"][~protected], special).any()
