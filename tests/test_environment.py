"""P0 environment checks only; no models or network access."""

import importlib

import pytest
import torch

import omni


def test_import_omni() -> None:
    assert omni.__name__ == "omni"


@pytest.mark.parametrize(
    "module_name", ["torch", "transformers", "datasets", "numpy", "einops", "safetensors", "yaml"]
)
def test_dependency_import(module_name: str) -> None:
    assert importlib.import_module(module_name).__name__ == module_name


def test_cpu_tensor_arithmetic() -> None:
    # x: [2, 2]; result: [2, 2]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
    torch.testing.assert_close(x @ x, expected)


def test_mps_availability_probe() -> None:
    # Availability is host-dependent; Ubuntu CPU CI must also pass.
    assert isinstance(torch.backends.mps.is_available(), bool)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable on this host")
def test_mps_tensor_arithmetic() -> None:
    # x: [2, 2]; result returns to CPU to synchronize the MPS operation.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
    torch.testing.assert_close((x @ x).cpu(), expected)
