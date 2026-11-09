"""奖励的独立数值与资源接口；不把诊断 token/code 均值当作质量指标。"""

from dataclasses import replace

import pytest
import torch
from test_r3_training import tiny_model

from omni.training.grpo import group_advantages
from omni.training.grpo_rewards import reward_resources, reward_score, validate_rewards
from omni.training.grpo_rollout import rollout


@pytest.fixture
def sample():
    torch.manual_seed(91)
    torch.set_num_threads(1)
    model = tiny_model()
    # 均匀声码分布配固定种子，得到完整帧供 decoder/ASR 接口验证。
    with torch.no_grad():
        model.talker.lm_head.base.weight.zero_()
        for head in model.talker.lm_head.adapters:
            head[-1].weight.zero_()
    torch.manual_seed(2)
    row = rollout(model, {"input_ids": torch.tensor([[1, 3]])}, mode="joint", max_new_tokens=14)
    return row, model.config


def test_deterministic_components_are_weighted_without_hidden_normalization(sample):
    row, config = sample
    weights = {"token_value": 2.0, "audio_value": -0.5}
    total, scores = reward_score(row, config, {"weights": weights})
    tokens = row.text_ids[row.text_mask].tolist()
    audio = [
        int(row.audio_ids[c, t])
        for c in range(8)
        for t in range(row.text_ids.numel())
        if row.audio_mask[c, t] and row.audio_ids[c, t] < config.audio_codebook_size
    ]
    text_mean = sum(tokens) / len(tokens) / (config.vocab_size - 1)
    audio_mean = sum(audio) / len(audio) / (config.audio_codebook_size - 1)
    assert scores == pytest.approx({"token_value": text_mean, "audio_value": audio_mean})
    assert total == pytest.approx(2 * text_mean - 0.5 * audio_mean)


def test_exact_text_reward_requires_audited_target(sample):
    row, config = sample
    settings = {"weights": {"text_match": 1.0}}
    target = row.text_ids[row.text_mask].tolist()
    assert reward_score(row, config, settings, {"target_ids": target})[0] == 1
    assert reward_score(row, config, settings, {"target_ids": target + [2]})[0] == 0
    with pytest.raises(ValueError, match="可信"):
        reward_score(row, config, settings)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_reward_cannot_be_silently_zeroed(sample, bad):
    row, config = sample
    with pytest.raises(ValueError, match="非有限"):
        reward_score(row, config, {"weights": {"custom": 1}}, evaluators={"custom": lambda _: bad})


def test_audio_decode_and_custom_quality_interface(sample):
    row, config = sample
    calls = []

    class Codec:
        sampling_rate = 24000

        def decode(self, codes):
            assert codes.shape[0:2] == (1, 8)
            assert ((codes >= 0) & (codes < config.audio_codebook_size)).all()
            calls.append(codes.clone())
            return torch.ones(codes.shape[-1] * 4)

    def quality(payload):
        assert payload["sampling_rate"] == 24000
        assert payload["waveform"].numel() == payload["audio_codes"].shape[-1] * 4
        assert payload["context"]["source"] == "fixture"
        return 0.75

    value, scores = reward_score(
        row,
        config,
        {"weights": {"quality": 2}},
        {"source": "fixture"},
        codec=Codec(),
        evaluators={"quality": quality},
    )
    assert calls and value == 1.5 and scores == {"quality": 0.75}


def test_missing_asr_is_explicit_and_unused_resources_are_not_loaded(sample, tmp_path):
    row, config = sample
    with pytest.raises(ValueError, match="ASR"):
        reward_score(row, config, {"weights": {"asr_match": 1}}, {"target_ids": [1, 2]})
    assert reward_resources({"grpo": {"reward": {"weights": {"token_value": 1}}}}, tmp_path) == {}


def test_asr_reward_uses_decoded_audio_and_trusted_target(sample):
    row, config = sample

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return "abc"

    class Codec:
        sampling_rate = 24000

        def decode(self, codes):
            assert codes.shape[0:2] == (1, 8) and codes.shape[-1] > 0
            return torch.ones(8)

    def asr(waveform, sampling_rate):
        assert waveform.shape == (8,) and sampling_rate == 24000
        return "adc"

    total, components = reward_score(
        row,
        config,
        {"weights": {"asr_match": 1}},
        {"target_ids": [1, 2, 3]},
        tokenizer=Tokenizer(),
        codec=Codec(),
        asr=asr,
    )
    # abc -> adc 是一次替换，独立期望 CER=1/3。
    assert total == pytest.approx(2 / 3)
    assert components == pytest.approx({"asr_match": 2 / 3})


def test_invalid_reward_config_and_extreme_normalization_fail():
    for settings in (
        {},
        {"weights": {}},
        {"weights": {"quality": 1}},
        {"weights": {"token_value": 0}},
    ):
        with pytest.raises(ValueError, match="Reward"):
            validate_rewards(settings)
    with pytest.raises(ValueError, match="溢出"):
        group_advantages(torch.tensor([[3e38, -3e38]]))


def test_empty_audio_is_not_completed(sample):
    row, config = sample
    text = replace(row, mode="text")
    # Text-only 要由真实 sampler 建轨迹，不能只换 mode 后沿用音频动作。
    with pytest.raises(ValueError, match="mask"):
        reward_score(text, config, {"weights": {"completeness": 1}})
