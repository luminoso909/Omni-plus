"""本地极小 Qwen3 fixture 验证离线检查脚本，不下载大模型、不训练。"""

import importlib.util
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

SPEC = importlib.util.spec_from_file_location(
    "check_qwen3", Path(__file__).parents[1] / "scripts/check_qwen3.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def make_local_qwen(path: Path, extra_vocab: int = 8) -> int:
    # ByteLevel 保留中文、标点和空格，覆盖真实 tokenizer 的文本往返语义
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {
        token: index for index, token in enumerate(alphabet + ["<pad>", "<s>", "</s>", "<unk>"])
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(path)
    config = Qwen3Config(
        vocab_size=len(tokenizer) + extra_vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.6
    model.generation_config.top_p = 0.95
    model.generation_config.top_k = 20
    model.save_pretrained(path)
    return len(tokenizer)


def test_local_qwen_offline_and_padded_vocabulary(tmp_path, monkeypatch) -> None:
    size = make_local_qwen(tmp_path)

    def forbid_network(*args, **kwargs):
        raise AssertionError("离线验收不应访问网络")

    monkeypatch.setattr("socket.create_connection", forbid_network)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("确定性基线不能继承模型文件中的随机采样设置")

    monkeypatch.setattr("torch.multinomial", forbid_sampling)
    report = CHECK.check_model(tmp_path, device="cpu", max_new_tokens=2)
    assert report["tokenizer_length"] == size
    assert report["embedding_vocab_size"] == size + 8
    assert report["tokenizer_roundtrip"]
    assert report["cpu"]["logits_shape"] == [1, 1, size + 8]
    assert report["cpu"]["logits_finite"]
    assert not report["offline"]["trained"] and report["offline"]["local_files_only"]
    assert report["generation"]["enable_thinking"] is False
    for sample in report["generation"]["samples"]:
        assert sample["prompt"] in sample["rendered_template"]
        assert len(sample["generated_ids"]) <= 2
        assert all(0 <= token < size + 8 for token in sample["generated_ids"])


def test_tokenizer_outside_embedding_is_rejected(tmp_path) -> None:
    make_local_qwen(tmp_path, extra_vocab=-1)
    with pytest.raises(ValueError, match="Token ID"):
        CHECK.check_model(tmp_path, device="cpu", max_new_tokens=1)
