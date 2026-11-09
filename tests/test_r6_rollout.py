"""真实 Tiny Qwen/Talker 概率重放及独立的 delay/STOP 边界期望。"""

import copy
from dataclasses import replace

import pytest
import torch
from test_r3_training import tiny_model

from omni.data.audio_alignment import delay_codes
from omni.models.model_omni import OmniOutput
from omni.training.adapters import inject_lora
from omni.training.grpo_rollout import (
    decode_codes,
    rollout,
    rollout_group,
    score_rollout,
)


@pytest.fixture
def model():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    return tiny_model().eval()


def prompt():
    return dict(input_ids=torch.tensor([[1, 3, 4]]))


@pytest.mark.parametrize("mode", ["text", "audio", "joint"])
def test_real_tiny_behavior_matches_teacher_forcing_and_cache(model, mode):
    values = dict(mode=mode, max_new_tokens=12, temperature=0.8, audio_temperature=1.2)
    torch.manual_seed(67)
    cached = rollout(model, prompt(), policy_version=4, **values)
    replay = score_rollout(model, cached)
    torch.testing.assert_close(cached.old_logprob, replay.logprob, rtol=2e-5, atol=2e-5)
    assert not cached.old_logprob.requires_grad
    assert cached.policy_version == 4
    assert replay.text_log_probs.shape[0] == cached.text_ids.numel()
    assert replay.audio_log_probs.shape[:2] == cached.audio_ids.shape
    assert replay.text_log_probs.isfinite().all() and replay.audio_log_probs.isfinite().all()
    torch.testing.assert_close(
        replay.text_log_probs.exp().sum(-1), torch.ones_like(cached.old_logprob)
    )
    torch.testing.assert_close(
        replay.audio_log_probs.exp().sum(-1), torch.ones_like(cached.audio_ids, dtype=torch.float32)
    )
    expected = replay.audio_logprob.sum(0)
    if mode != "audio":
        expected = expected + replay.text_logprob
    torch.testing.assert_close(replay.logprob, expected)

    torch.manual_seed(67)
    uncached = rollout(model, prompt(), use_cache=False, policy_version=4, **values)
    assert torch.equal(cached.text_ids, uncached.text_ids)
    assert torch.equal(cached.audio_ids, uncached.audio_ids)
    torch.testing.assert_close(cached.old_logprob, uncached.old_logprob, rtol=2e-5, atol=2e-5)
    assert cached.to_record()["policy_version"] == 4


def test_left_padding_and_prompt_audio_history_are_preserved(model):
    plain = prompt()
    plain["audio_ids"] = torch.full((1, 8, 3), 9, dtype=torch.long)
    plain["audio_ids"][..., -1] = 3
    padded = dict(
        input_ids=torch.tensor([[0, 0, 1, 3, 4]]),
        audio_ids=torch.cat((torch.full((1, 8, 2), 9, dtype=torch.long), plain["audio_ids"]), -1),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
    )
    torch.manual_seed(5)
    left = rollout(model, plain, mode="joint", max_new_tokens=10)
    torch.manual_seed(5)
    right = rollout(model, padded, mode="joint", max_new_tokens=10)
    torch.testing.assert_close(left.old_logprob, right.old_logprob, rtol=2e-5, atol=2e-5)
    assert torch.equal(left.audio_ids, right.audio_ids)
    assert torch.equal(left.prompt["audio_ids"], plain["audio_ids"])
    plain["audio_ids"].zero_()
    assert left.prompt["audio_ids"][..., -1].eq(3).all()
    baseline = score_rollout(model, left)
    changed = copy.deepcopy(left)
    changed.prompt["audio_ids"].fill_(9)
    assert not torch.allclose(
        baseline.audio_log_probs, score_rollout(model, changed).audio_log_probs
    )


def test_joint_logprobs_have_real_lora_bridge_and_talker_gradients(model):
    inject_lora(
        model.thinker.model,
        dict(enabled=True, rank=2, alpha=4, targets=["q_proj", "v_proj"], dropout=0),
    )
    torch.manual_seed(54)
    trajectory = rollout(model, prompt(), mode="joint", max_new_tokens=12)
    score = score_rollout(model, trajectory)
    (-score.logprob[trajectory.valid_mask].mean()).backward()
    for prefix in ("bridge.", "talker.embed_tokens.", "talker.lm_head.", "talker.layers."):
        assert any(
            p.grad is not None and p.grad.abs().max() > 0
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        )
    assert any(
        p.grad is not None and p.grad.abs().max() > 0
        for name, p in model.named_parameters()
        if ".lora_b." in name
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_multimodal_prompt_and_projector_gradients_survive_replay(model):
    source = dict(
        input_ids=torch.tensor([[1, 12, 12, 13, 3]]),
        image_features=[torch.arange(6, dtype=torch.float32).reshape(2, 3)],
        audio_features=[torch.arange(4, dtype=torch.float32).reshape(1, 4)],
    )
    trajectory = rollout(model, source, mode="joint", max_new_tokens=10)
    score = score_rollout(model, trajectory)
    torch.testing.assert_close(score.logprob, trajectory.old_logprob, rtol=2e-5, atol=2e-5)
    (-score.logprob.mean()).backward()
    for module in (model.vision_proj, model.audio_proj):
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.thinker.parameters())


def test_future_joint_action_does_not_change_earlier_conditionals(model):
    trajectory = rollout(
        ScheduledPolicy(model.config, frames=20),
        prompt(),
        mode="joint",
        max_new_tokens=12,
        use_cache=False,
    )
    changed = copy.deepcopy(trajectory)
    active = changed.audio_mask.nonzero()[-1]
    channel, step = map(int, active)
    if int(changed.audio_ids[channel, step]) == model.config.audio_stop_token:
        changed.audio_ids[channel, step] = 2
        changed.audio_done = False
        changed.truncated = True
    else:
        changed.audio_ids[channel, step] = (changed.audio_ids[channel, step] + 1) % 8
    before = score_rollout(model, trajectory)
    after = score_rollout(model, changed)
    torch.testing.assert_close(
        before.text_log_probs[: step + 1], after.text_log_probs[: step + 1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        before.audio_log_probs[:, : step + 1], after.audio_log_probs[:, : step + 1], rtol=0, atol=0
    )


class ScheduledPolicy(torch.nn.Module):
    """仅用于有独立期望的协议边界；概率/梯度验收使用上面的真实模型。"""

    def __init__(self, config, *, frames=1):
        super().__init__()
        self.config, self.frames = config, frames

    def forward(self, input_ids, **kwargs):
        step = input_ids.shape[-1] - 3
        text = torch.full((1, 1, self.config.vocab_size), -1000.0)
        text[..., self.config.eos_token_id] = 0
        audio = torch.full((1, 8, 1, self.config.audio_vocab_size), -1000.0)
        for channel in range(8):
            code = channel if step < 1 + channel + self.frames else self.config.audio_stop_token
            audio[:, channel, :, code] = 0
        return OmniOutput(logits=text, audio_logits=audio, aux_loss=torch.tensor(0.0))


def test_eos_then_audio_tail_matches_independent_delay_layout(model):
    trajectory = rollout(
        ScheduledPolicy(model.config), prompt(), mode="joint", max_new_tokens=15, use_cache=False
    )
    expected = delay_codes(torch.arange(8).reshape(8, 1), pad_token=9, stop_token=10)
    assert torch.equal(trajectory.audio_ids, expected)
    assert trajectory.text_ids.tolist() == [2] + [0] * 9
    assert trajectory.text_mask.tolist() == [True] + [False] * 9
    assert trajectory.audio_mask.sum(1).tolist() == [2] * 8  # 声码与首次 STOP。
    assert trajectory.text_done and trajectory.audio_done and not trajectory.truncated
    assert decode_codes(trajectory, model.config).tolist() == [[c] for c in range(8)]


def test_immediate_stops_and_incomplete_truncated_frames_remain_actions(model):
    stopped = rollout(
        ScheduledPolicy(model.config, frames=0),
        prompt(),
        mode="joint",
        max_new_tokens=15,
        use_cache=False,
    )
    assert stopped.audio_mask.sum().item() == 8
    assert decode_codes(stopped, model.config).shape == (8, 0)
    truncated = rollout(
        ScheduledPolicy(model.config), prompt(), mode="audio", max_new_tokens=7, use_cache=False
    )
    assert truncated.truncated and not truncated.audio_done
    assert truncated.audio_mask.sum() > 0
    assert decode_codes(truncated, model.config).shape == (8, 0)
    assert truncated.valid_mask.tolist() == [False] + [True] * 6


def test_uniform_eight_head_joint_probability_is_product_not_average(model):
    # 完整合法 support 均匀，任一有效声码/STOP 的概率都为 1/9。
    for param in model.parameters():
        param.data.zero_()
    trajectory = rollout(
        ScheduledPolicy(model.config, frames=2),
        prompt(),
        mode="joint",
        max_new_tokens=20,
        use_cache=False,
    )
    scored = score_rollout(model, trajectory)
    counts = trajectory.audio_mask.sum(0)
    expected = -counts * torch.log(torch.tensor(9.0)) - trajectory.text_mask * torch.log(
        torch.tensor(29.0)
    )
    torch.testing.assert_close(scored.logprob, expected)
    assert scored.logprob[8].exp().item() == pytest.approx(9.0**-3, rel=1e-6)


@pytest.mark.parametrize("damage", ["text_mask", "audio_mask", "invalid_code", "post_stop"])
def test_replay_rejects_forged_alignment(model, damage):
    trajectory = rollout(
        ScheduledPolicy(model.config), prompt(), mode="joint", max_new_tokens=15, use_cache=False
    )
    if damage == "text_mask":
        trajectory.text_mask[1] = True
    elif damage == "audio_mask":
        trajectory.audio_mask[0, 0] = True
    elif damage == "invalid_code":
        trajectory.audio_ids[0, 1] = 15
    else:
        trajectory.audio_ids[0, 4] = 1
    with pytest.raises(ValueError):
        score_rollout(model, trajectory)


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"top_k": 3}, {"top_p": 0.9}, {"max_new_tokens": 0}]
)
def test_sampling_rejects_ambiguous_or_empty_behavior(model, kwargs):
    values = dict(mode="joint", max_new_tokens=10) | kwargs
    with pytest.raises(ValueError):
        rollout(model, prompt(), **values)


def test_group_rollouts_are_distinct_detached_snapshots(model):
    group = rollout_group(
        model, prompt(), group_size=3, mode="text", max_new_tokens=6, policy_version=7
    )
    assert all(row.policy_version == 7 and not row.old_logprob.requires_grad for row in group)
    assert len({tuple(row.text_ids.tolist()) for row in group}) > 1
    with pytest.raises(ValueError, match="group_size"):
        rollout_group(model, prompt(), group_size=1, mode="text", max_new_tokens=6)
    bad = replace(group[0], text_done=not group[0].text_done)
    with pytest.raises(ValueError, match="停止或截断"):
        score_rollout(model, bad)
