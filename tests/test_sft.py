"""独立手算损失、变长梯度累积及带 RNG 的完整训练状态恢复。"""

import copy
import random

import numpy as np
import pytest
import torch

from omni.data import OmniCollator
from omni.models import OmniPlusConfig, OmniPlusOmni
from omni.models.model_omni import OmniOutput
from omni.training.losses import sft_loss
from omni.training.sft import (
    configure_trainable,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def small_config(dropout: float = 0) -> OmniPlusConfig:
    return OmniPlusConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        vocab_size=32,
        num_talker_hidden_layers=1,
        talker_hidden_size=16,
        talker_num_attention_heads=2,
        talker_num_key_value_heads=1,
        talker_intermediate_size=24,
        talker_adapter_rank=4,
        num_audio_codebooks=2,
        audio_codebook_size=8,
        audio_vocab_size=12,
        audio_pad_token=9,
        audio_stop_token=10,
        audio_spk_token=11,
        audio_hidden_size=4,
        image_hidden_size=6,
        max_position_embeddings=32,
        dropout=dropout,
        pad_token_id=0,
    )


def samples() -> list[dict]:
    result = []
    for length in (3, 5):
        result.append(
            dict(
                input_ids=torch.tensor([1, 3, 4, 5, 6][:length]),
                labels=torch.tensor([-100, 4, 5, 6, 2][:length]),
                audio_ids=torch.full((2, length), 9, dtype=torch.long),
                audio_labels=torch.tensor([[1, 2, 10, -100, -100], [3, 4, 5, 10, -100]])[
                    :, :length
                ],
                attention_mask=torch.ones(length, dtype=torch.long),
                audio_features=None,
                image_features=None,
                spk_emb=None,
            )
        )
    return result


def test_loss_matches_hand_calculation_and_masked_gradients() -> None:
    text = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 1.0, 2.0]]], requires_grad=True)
    audio = torch.tensor(
        [[[[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]], [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]]]],
        requires_grad=True,
    )
    labels = torch.tensor([[1, -100]])
    targets = torch.tensor([[[2, -100], [0, 2]]])
    losses = sft_loss(OmniOutput(logits=text, audio_logits=audio), labels, targets, 2, 3)

    # 直接从 log-softmax 取监督位置，独立写出各码本加权和及分母
    expected_text = -text.log_softmax(-1)[0, 0, 1]
    logp = audio.log_softmax(-1)
    expected_audio = (-3 * logp[0, 0, 0, 2] + (-logp[0, 1, 0, 0] - 3 * logp[0, 1, 1, 2]) / 2) / 2
    torch.testing.assert_close(losses["text"], expected_text)
    torch.testing.assert_close(losses["audio"], expected_audio)
    losses["total"].backward()
    assert torch.equal(text.grad[0, 1], torch.zeros(3))
    assert torch.equal(audio.grad[0, 0, 1], torch.zeros(3))


def test_all_ignored_labels_return_differentiable_zero() -> None:
    text = torch.randn(1, 3, 4, requires_grad=True)
    audio = torch.randn(1, 2, 3, 5, requires_grad=True)
    labels = torch.full((1, 3), -100)
    targets = torch.full((1, 2, 3), -100)
    losses = sft_loss(OmniOutput(logits=text, audio_logits=audio), labels, targets)
    assert losses["total"].item() == 0
    losses["total"].backward()
    assert text.grad.eq(0).all() and audio.grad.eq(0).all()


def test_accumulation_matches_complete_variable_length_batch() -> None:
    config = small_config()
    full = OmniPlusOmni(config)
    accumulated = copy.deepcopy(full)
    collate = OmniCollator(config)
    items = samples()
    first = torch.optim.SGD(full.parameters(), lr=0.01)
    second = torch.optim.SGD(accumulated.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    metrics = train_step(full, [collate(items)], first, scaler, grad_clip=100)
    micro = train_step(
        accumulated, [collate([item]) for item in items], second, scaler, grad_clip=100
    )
    assert metrics["total"] == pytest.approx(micro["total"], rel=1e-6)
    for actual, expected in zip(full.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_checkpoint_resume_restores_optimizer_and_all_cpu_rng(tmp_path) -> None:
    config = small_config(dropout=0.1)
    model = OmniPlusOmni(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    batches = [OmniCollator(config)(samples())]
    train_step(model, batches, optimizer, scaler)
    checkpoint = tmp_path / "step1.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, dict(step=1, next_batch=2, epoch=0))
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    expected_loss = train_step(model, batches, optimizer, scaler)

    # 新模型构造会消耗随机数，必须在构造后恢复 RNG 和 Adam 状态
    resumed = OmniPlusOmni(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.123)
    progress = load_checkpoint(checkpoint, resumed, resumed_optimizer, scaler)
    assert progress == dict(step=1, next_batch=2, epoch=0)
    assert (random.random(), np.random.random()) == expected_rng[:2]
    assert torch.equal(torch.rand(3), expected_rng[2])
    actual_loss = train_step(resumed, batches, resumed_optimizer, scaler)
    assert actual_loss == expected_loss
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_bfloat16_autocast_step() -> None:
    config = small_config()
    model = OmniPlusOmni(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    result = train_step(
        model, [OmniCollator(config)(samples())], optimizer, scaler, amp_dtype=torch.bfloat16
    )
    assert all(np.isfinite(value) for value in result.values())


def test_freeze_policy_preserves_weight_keys_and_updates_only_selected_module() -> None:
    config = small_config()
    model = OmniPlusOmni(config)
    keys = set(model.state_dict())
    count = configure_trainable(model, "vision_projector")
    assert count == sum(param.numel() for param in model.vision_proj.parameters())
    assert all(not param.requires_grad for param in model.model.parameters())
    assert all(not param.requires_grad for param in model.talker.parameters())
    assert set(model.state_dict()) == keys
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    batch = OmniCollator(config)(samples())
    batch["input_ids"][:, 1] = 12
    batch["image_features"] = [torch.randn(1, 6), torch.randn(1, 6)]
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad])
    train_step(model, [batch], optimizer, torch.amp.GradScaler("cpu", enabled=False))
    changed = [
        name for name, param in model.named_parameters() if not torch.equal(before[name], param)
    ]
    assert changed and all(name.startswith("vision_proj.") for name in changed)
    configure_trainable(model, "all")
    assert all(param.requires_grad for param in model.parameters())


def test_empty_evaluation_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        evaluate(OmniPlusOmni(small_config()), [])


def test_fp16_overflow_skips_update_and_reduces_scale() -> None:
    config = small_config()
    model = OmniPlusOmni(config)
    before = [param.detach().clone() for param in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cpu", init_scale=1e12, enabled=True)
    result = train_step(
        model, [OmniCollator(config)(samples())], optimizer, scaler, amp_dtype=torch.float16
    )
    assert result["optimizer_updated"] == 0 and scaler.get_scale() < 1e12
    assert all(
        torch.equal(first, second) for first, second in zip(before, model.parameters(), strict=True)
    )


def test_resume_rejects_changed_freeze_policy(tmp_path) -> None:
    model = OmniPlusOmni(small_config())
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scaler, dict(step=0))
    configure_trainable(model, "vision_projector")
    with pytest.raises(ValueError, match="冻结策略"):
        load_checkpoint(path, model, optimizer, scaler)
