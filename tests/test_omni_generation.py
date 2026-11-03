"""音频码本生成：用确定性 logits 验证延迟、停止和流式边界，不下载模型。"""

import pytest
import torch

from omni.models.config import OmniPlusConfig
from omni.models.model_omni import OmniOutput, OmniPlusOmni


@pytest.fixture
def omni_config() -> OmniPlusConfig:
    return OmniPlusConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        vocab_size=64,
        max_position_embeddings=48,
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
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def script_model(
    config: OmniPlusConfig,
    monkeypatch,
    text_steps: list[int],
    audio_steps: list[tuple[int, int]],
) -> tuple[OmniPlusOmni, list[dict]]:
    model = OmniPlusOmni(config).eval()
    calls = []

    def forward(input_ids, audio_ids=None, attention_mask=None, **kwargs) -> OmniOutput:
        step = len(calls)
        calls.append(
            {
                "input_ids": input_ids.clone(),
                "audio_ids": audio_ids.clone(),
                "mask": attention_mask.clone(),
                "grad": torch.is_grad_enabled(),
                "inference": torch.is_inference_mode_enabled(),
            }
        )
        text = input_ids.new_full((1, 1, config.vocab_size), -100, dtype=torch.float32)
        text[0, 0, text_steps[min(step, len(text_steps) - 1)]] = 50
        audio = text.new_full((1, config.num_audio_codebooks, 1, config.audio_vocab_size), -100)
        # 无效 special token 分数更高，也不能进入可采样的声学候选集。
        audio[..., config.audio_codebook_size] = 200
        audio[..., config.audio_pad_token] = 200
        audio[..., config.audio_spk_token] = 200
        for codebook, code in enumerate(audio_steps[min(step, len(audio_steps) - 1)]):
            audio[0, codebook, 0, code] = 50
        return OmniOutput(logits=text, audio_logits=audio)

    monkeypatch.setattr(model, "forward", forward)
    return model, calls


def test_delayed_frames_and_stream_inference(omni_config: OmniPlusConfig, monkeypatch) -> None:
    # 码本 0 在 step=1 启动，码本 1 在 step=2 启动；第一帧为 [3, 7]。
    model, calls = script_model(
        omni_config,
        monkeypatch,
        text_steps=[4, 5, 2, 1, 1],
        audio_steps=[(15, 15), (3, 15), (4, 7), (18, 8), (15, 9), (15, 10), (15, 18)],
    )
    prompt = torch.tensor([[0, 1, 11, 12]])
    mask = torch.tensor([[0, 1, 1, 1]])
    chunks = list(
        model.stream_generate(
            prompt,
            max_new_tokens=8,
            temperature=0,
            audio_temperature=0,
            attention_mask=mask,
            use_cache=False,
        )
    )
    frame_chunks = [(step, frame) for step, (_, frame) in enumerate(chunks) if frame is not None]
    assert frame_chunks[0][0] == 2
    frames = torch.cat([frame for _, frame in frame_chunks], dim=-1)
    assert frames.shape == (1, 2, 2)
    assert frames.tolist() == [[[3, 4], [7, 8]]]
    text = torch.cat([token for token, _ in chunks if token is not None], dim=-1)
    assert text.tolist() == [[4, 5, 2]]
    assert all(not call["grad"] and call["inference"] for call in calls)
    for step, call in enumerate(calls):
        assert call["mask"].shape == (1, prompt.shape[1] + step)
        assert torch.equal(call["mask"][:, :4], mask)
    # 码本 0 停止后，下一轮音频历史只追加 pad，不能再次发出声学码。
    assert calls[5]["audio_ids"][0, 0, -1].item() == omni_config.audio_pad_token
    assert calls[6]["audio_ids"][0, 0, -1].item() == omni_config.audio_pad_token


def test_generate_keeps_text_when_only_audio_remains(
    omni_config: OmniPlusConfig, monkeypatch
) -> None:
    model, _ = script_model(
        omni_config,
        monkeypatch,
        text_steps=[4, 5, 2, 1, 1],
        audio_steps=[(15, 15), (3, 15), (4, 7), (18, 8), (15, 9), (15, 10), (15, 18)],
    )
    prompt = torch.tensor([[1, 11, 12]])
    text, codes = model.generate(
        prompt, max_new_tokens=8, temperature=0, audio_temperature=0, use_cache=False
    )
    assert text.tolist() == [[1, 11, 12, 4, 5, 2]]
    assert codes.tolist() == [[[3, 4], [7, 8]]]


def test_thinking_end_delays_audio_start(omni_config: OmniPlusConfig, monkeypatch) -> None:
    omni_config.think_end_ids = [9, 10]
    model, calls = script_model(
        omni_config,
        monkeypatch,
        text_steps=[9, 10, 4, 5, 2, 1, 1],
        audio_steps=[(15, 15), (15, 15), (15, 15), (6, 15), (7, 8), (18, 9), (15, 18)],
    )
    prompt = torch.tensor([[1, 11, 12]])
    chunks = list(
        model.stream_generate(
            prompt,
            max_new_tokens=9,
            temperature=0,
            audio_temperature=0,
            open_thinking=True,
            use_cache=False,
        )
    )
    frame_chunks = [(step, frame) for step, (_, frame) in enumerate(chunks) if frame is not None]
    assert frame_chunks[0][0] == 4
    frames = torch.cat([frame for _, frame in frame_chunks], dim=-1)
    assert frames.tolist() == [[[6, 7], [8, 9]]]
    # </think> 在 step=1 匹配；step=3 才采样第一路音频。
    assert calls[3]["audio_ids"][0, :, -1].tolist() == [17, 17]
    assert calls[4]["audio_ids"][0, :, -1].tolist() == [6, 17]


def test_reference_codes_and_speaker_prompt(omni_config: OmniPlusConfig, monkeypatch) -> None:
    model, calls = script_model(
        omni_config,
        monkeypatch,
        text_steps=[4],
        audio_steps=[(3, 7)],
    )
    prompt = torch.tensor([[1, 11, 12, 13]])
    ref_codes = torch.tensor([[[1, 2], [3, 4]]])
    list(
        model.stream_generate(
            prompt,
            max_new_tokens=1,
            temperature=0,
            audio_temperature=0,
            ref_codes=ref_codes,
            spk_emb=torch.zeros(1, 6),
            use_cache=False,
        )
    )
    assert calls[0]["audio_ids"].tolist() == [[[17, 19, 1, 2], [17, 19, 3, 4]]]


def test_generate_cache_equivalence(omni_config: OmniPlusConfig) -> None:
    torch.manual_seed(19)
    model = OmniPlusOmni(omni_config).eval()
    prompt = torch.tensor([[1, 11, 12]])
    args = dict(max_new_tokens=6, temperature=0, audio_temperature=0)
    cached_text, cached_codes = model.generate(prompt, use_cache=True, **args)
    full_text, full_codes = model.generate(prompt, use_cache=False, **args)
    assert torch.equal(cached_text, full_text)
    assert torch.equal(cached_codes, full_codes)


def test_generation_empty_budget_and_batch_limit(omni_config: OmniPlusConfig) -> None:
    model = OmniPlusOmni(omni_config).eval()
    prompt = torch.tensor([[1, 11, 12]])
    text, codes = model.generate(prompt, max_new_tokens=0)
    assert torch.equal(text, prompt)
    assert codes.shape == (1, omni_config.num_audio_codebooks, 0)
    with pytest.raises(ValueError, match="batch"):
        list(model.stream_generate(prompt.repeat(2, 1), max_new_tokens=1))
