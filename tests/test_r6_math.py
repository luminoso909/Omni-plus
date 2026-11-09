"""GRPO 期望值来自概率乘积、分段导数和独立标量公式。"""

import math

import pytest
import torch

from omni.training.grpo import exact_kl, group_advantages, grpo_loss


def test_group_advantage_matches_population_standard_deviation() -> None:
    rewards = torch.tensor([[0.0, 2.0, 4.0], [7.0, 7.0, 7.0]], dtype=torch.float64)
    rewards.requires_grad_()
    actual = group_advantages(rewards, epsilon=1e-6)
    scale = math.sqrt(8 / 3) + 1e-6
    expected = torch.tensor([[-2 / scale, 0, 2 / scale], [0, 0, 0]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not actual.requires_grad


def test_advantage_groups_do_not_mix_prompts() -> None:
    rewards = torch.tensor([[0.0, 2.0], [100.0, 104.0]], dtype=torch.float64)
    expected = torch.tensor(
        [[-1 / 1.000001, 1 / 1.000001], [-2 / 2.000001, 2 / 2.000001]], dtype=torch.float64
    )
    torch.testing.assert_close(group_advantages(rewards), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "rewards",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0]]),
        torch.empty(0, 2),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[float("inf"), 2.0]]),
    ],
)
def test_invalid_rewards_reject_whole_group(rewards: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        group_advantages(rewards)


def test_exact_kl_direction_and_reference_gradient() -> None:
    logits = torch.tensor([math.log(3.0), 0.0], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([math.log(0.5), math.log(0.5)], dtype=torch.float64)
    reference.requires_grad_()
    actual = exact_kl(logits.log_softmax(-1), reference)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    reverse = 0.5 * math.log(2 / 3) + 0.5 * math.log(2)
    assert abs(actual.item() - reverse) > 1e-3
    actual.backward()
    expected_gradient = torch.tensor(
        [0.75 * (math.log(1.5) - expected), 0.25 * (math.log(0.5) - expected)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    assert reference.grad is None


def test_exact_kl_has_zero_gradient_for_identical_distributions() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    log_probs = logits.log_softmax(-1)
    value = exact_kl(log_probs, log_probs.detach())
    torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=1e-14)
    value.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-14)


def test_exact_kl_gradcheck() -> None:
    logits = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64).log()
    assert torch.autograd.gradcheck(
        lambda value: exact_kl(value.log_softmax(-1), reference), logits
    )


def test_clipped_objective_piecewise_value_and_gradient() -> None:
    # 四种情况依次为正优势上裁剪、正优势未裁剪、负优势未裁剪、负优势下裁剪。
    ratios = torch.tensor([[1.5], [0.5], [1.5], [0.5]], dtype=torch.float64)
    old = torch.full_like(ratios, math.log(0.4), requires_grad=True)
    current = (old.detach() + ratios.log()).requires_grad_()
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64, requires_grad=True)
    result = grpo_loss(
        current,
        old,
        torch.zeros_like(current),
        advantage,
        torch.ones_like(current, dtype=torch.bool),
        clip=0.2,
        beta=0.0,
    )
    assert result["total"].item() == pytest.approx((-1.2 - 0.5 + 1.5 + 0.8) / 4, abs=1e-12)
    result["total"].backward()
    expected = torch.tensor([[0.0], [-0.125], [0.375], [0.0]], dtype=torch.float64)
    torch.testing.assert_close(current.grad, expected, rtol=1e-12, atol=1e-12)
    assert old.grad is None and advantage.grad is None


@pytest.mark.parametrize("normalization,expected", [("sequence_mean", 2.0), ("token_mean", 2.5)])
def test_loss_normalization_uses_valid_physical_steps(normalization: str, expected: float) -> None:
    current = torch.full((2, 3), math.log(0.5), dtype=torch.float64, requires_grad=True)
    old = current.detach().clone()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    # ratio=1、A=-1/-3，所以逐步 Policy Loss 分别为 [1] 与 [3,3,3]。
    advantage = torch.tensor([-1.0, -3.0], dtype=torch.float64)
    result = grpo_loss(
        current,
        old,
        torch.zeros_like(current),
        advantage,
        mask,
        beta=0.0,
        normalization=normalization,
    )
    assert result["total"].item() == pytest.approx(expected, abs=1e-12)
    result["total"].backward()
    assert torch.equal(current.grad[0, 1:], torch.zeros(2, dtype=torch.float64))


def test_padding_cannot_change_loss_or_kl_normalization() -> None:
    current = torch.tensor([[math.log(0.5), -10.0]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[math.log(0.5), -20.0]], dtype=torch.float64)
    kl = torch.tensor([[0.25, 99.0]], dtype=torch.float64, requires_grad=True)
    result = grpo_loss(
        current, old, kl, torch.tensor([1.0]), torch.tensor([[True, False]]), beta=0.4
    )
    assert result["total"].item() == pytest.approx(-0.9, abs=1e-12)
    assert result["kl"].detach().item() == pytest.approx(0.25, abs=1e-12)
    result["total"].backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-1.0, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(kl.grad, torch.tensor([[0.4, 0.0]], dtype=torch.float64))


def test_eight_codebook_joint_ratio_is_probability_product() -> None:
    # 每路 Old=1/4、Current=1/2，因此八路联合 Ratio=2**8，而非2。
    old = torch.tensor([[8 * math.log(0.25)]], dtype=torch.float64)
    current = torch.tensor([[8 * math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    result = grpo_loss(
        current,
        old,
        torch.zeros_like(current),
        torch.tensor([-1.0]),
        torch.ones_like(current, dtype=torch.bool),
        beta=0.0,
    )
    assert result["total"].item() == pytest.approx(256.0, rel=1e-12)
    result["total"].backward()
    assert current.grad.item() == pytest.approx(256.0, rel=1e-12)


def test_joint_surrogate_gradcheck_away_from_clip_boundaries() -> None:
    old = torch.tensor([[-1.1, -0.7], [-0.6, -1.2]], dtype=torch.float64)
    current = (old + torch.tensor([[0.03, -0.04], [-0.05, 0.02]])).requires_grad_()
    mask = torch.ones_like(current, dtype=torch.bool)
    advantage = torch.tensor([0.8, -0.6], dtype=torch.float64)

    def objective(value):
        return grpo_loss(value, old, torch.zeros_like(value), advantage, mask, beta=0.0)["total"]

    assert torch.autograd.gradcheck(objective, current)


def test_zero_advantage_retains_only_kl_gradient() -> None:
    current = torch.tensor([[-0.7]], dtype=torch.float64, requires_grad=True)
    kl = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    result = grpo_loss(
        current,
        current.detach(),
        kl,
        torch.zeros(1),
        torch.ones_like(current, dtype=torch.bool),
        beta=0.5,
    )
    assert result["total"].item() == pytest.approx(0.05, abs=1e-12)
    result["total"].backward()
    assert current.grad.item() == 0 and kl.grad.item() == 0.5


@pytest.mark.parametrize("defect", ["empty", "current_nan", "old_inf", "advantage_nan", "kl_inf"])
def test_invalid_objective_inputs_fail_clearly(defect: str) -> None:
    current = torch.tensor([[-1.0]], requires_grad=True)
    old, kl, advantage = current.detach().clone(), torch.zeros(1, 1), torch.ones(1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    if defect == "empty":
        mask.zero_()
    elif defect == "current_nan":
        current = torch.tensor([[float("nan")]], requires_grad=True)
    elif defect == "old_inf":
        old.fill_(float("inf"))
    elif defect == "advantage_nan":
        advantage.fill_(float("nan"))
    else:
        kl.fill_(float("inf"))
    with pytest.raises(ValueError):
        grpo_loss(current, old, kl, advantage, mask)
