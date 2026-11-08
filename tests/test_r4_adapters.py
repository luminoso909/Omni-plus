"""原生 Qwen LoRA 的零输出初始化、训练边界和严格独立恢复。"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_r3_training import batch, tiny_model

from omni.training.adapters import export_adapter, inject_lora, load_adapter
from omni.training.losses import sft_loss


def lora_settings() -> dict:
    return dict(enabled=True, rank=2, alpha=4, targets=["q_proj", "v_proj"], dropout=0)


def test_native_qwen_lora_init_and_gradients() -> None:
    torch.manual_seed(42)
    model = tiny_model().eval()
    inputs = batch(0)
    labels, audio_labels = inputs.pop("labels"), inputs.pop("audio_labels")
    expected = model(**inputs).logits.detach()
    base = {name: value.clone() for name, value in model.thinker.model.state_dict().items()}
    paths = inject_lora(model.thinker.model, lora_settings())
    assert paths == [
        f"model.layers.{index}.self_attn.{target}"
        for index in range(2)
        for target in ("q_proj", "v_proj")
    ]
    torch.testing.assert_close(model(**inputs).logits, expected, rtol=0, atol=0)

    # Omni.train() 保持 Qwen.eval()，dropout=0 的 Adapter 仍有可训练梯度。
    model.train()
    assert not model.thinker.training
    optimizer = torch.optim.SGD(
        [param for param in model.thinker.parameters() if param.requires_grad], lr=0.1
    )
    for _ in range(2):
        optimizer.zero_grad()
        output = model(**inputs)
        sft_loss(output, labels, audio_labels, stop_token=10)["total"].backward()
        optimizer.step()
    params = dict(model.thinker.model.named_parameters())
    for name, param in params.items():
        if ".lora_" in name:
            assert param.grad is not None and param.grad.abs().max() > 0
        else:
            assert param.grad is None and not param.requires_grad
            original = name.replace(".base.", ".")
            torch.testing.assert_close(param, base[original], rtol=0, atol=0)


@pytest.mark.parametrize(
    "change", [{"dropout": 0.1}, {"targets": ["wrong"]}, {"rank": 100}, {"rank": 0}]
)
def test_invalid_lora_leaves_native_model_unchanged(change: dict) -> None:
    model = tiny_model()
    native = model.thinker.model
    keys = tuple(native.state_dict())
    with pytest.raises(ValueError):
        inject_lora(native, lora_settings() | change)
    assert tuple(native.state_dict()) == keys


def test_adapter_restores_multimodal_and_lora_without_base_weights(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = tiny_model()
    restored = deepcopy(model)
    inject_lora(model.thinker.model, lora_settings())
    inject_lora(restored.thinker.model, lora_settings())
    for name, param in model.named_parameters():
        if ".lora_b." in name:
            with torch.no_grad():
                param.fill_(0.05)
    for name, param in restored.named_parameters():
        if not name.startswith("thinker.") or ".lora_" in name:
            with torch.no_grad():
                param.add_(1)
    identity = {"base_revision": "tiny-fixture", "tokenizer_sha256": "fixture-only"}
    path = tmp_path / "adapter.pt"
    export_adapter(path, model, identity)
    payload = torch.load(path, weights_only=True)
    assert all(not name.startswith("thinker.") or ".lora_" in name for name in payload["state"])
    load_adapter(path, restored, identity)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    inputs = {key: val for key, val in batch(0).items() if "labels" not in key}
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(**inputs).audio_logits, restored(**inputs).audio_logits)
    with pytest.raises(FileExistsError):
        export_adapter(path, model, identity)
    with pytest.raises(ValueError, match="身份"):
        load_adapter(path, restored, {"base_revision": "wrong"})


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nan"])
def test_adapter_validation_is_strict_before_writing(tmp_path: Path, damage: str) -> None:
    model = tiny_model()
    inject_lora(model.thinker.model, lora_settings())
    path = tmp_path / "adapter.pt"
    identity = {"base_revision": "tiny-fixture"}
    export_adapter(path, model, identity)
    payload = torch.load(path, weights_only=True)
    state = payload["state"]
    name = next(iter(state))
    if damage == "missing":
        del state[name]
    elif damage == "extra":
        state["unknown"] = torch.zeros(1)
    elif damage == "shape":
        state[name] = state[name].flatten()[:1]
    elif damage == "dtype":
        state[name] = state[name].double()
    else:
        state[name].fill_(float("nan"))
    torch.save(payload, path)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_adapter(path, model, identity)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
