import pytest
import torch
import torch.nn.functional as F

from optuna_saturate.vectorized.losses import per_member_cross_entropy


def test_returns_one_loss_value_per_member() -> None:
    logits = torch.randn(3, 5, 4)
    targets = torch.randint(0, 4, (5,))

    losses = per_member_cross_entropy(logits, targets)

    assert losses.shape == (3,)


def test_matches_torch_cross_entropy_computed_member_by_member() -> None:
    logits = torch.randn(4, 6, 3)
    targets = torch.randint(0, 3, (6,))

    losses = per_member_cross_entropy(logits, targets)

    for i in range(4):
        expected = F.cross_entropy(logits[i], targets)
        torch.testing.assert_close(losses[i], expected, atol=1e-6, rtol=1e-5)


def test_temperature_divides_the_logits_of_each_member() -> None:
    logits = torch.randn(2, 5, 3)
    targets = torch.randint(0, 3, (5,))
    temperature = torch.tensor([0.5, 2.0])

    losses = per_member_cross_entropy(logits, targets, temperature=temperature)

    for i, temp in enumerate([0.5, 2.0]):
        expected = F.cross_entropy(logits[i] / temp, targets)
        torch.testing.assert_close(losses[i], expected, atol=1e-6, rtol=1e-5)


def test_temperature_of_one_leaves_the_loss_unchanged() -> None:
    logits = torch.randn(3, 5, 4)
    targets = torch.randint(0, 4, (5,))

    without = per_member_cross_entropy(logits, targets)
    with_unit = per_member_cross_entropy(logits, targets, temperature=torch.ones(3))

    torch.testing.assert_close(without, with_unit)


def test_summing_the_losses_keeps_member_gradients_separate() -> None:
    logits = torch.randn(3, 5, 4, requires_grad=True)
    targets = torch.randint(0, 4, (5,))

    per_member_cross_entropy(logits, targets).sum().backward()

    grad = logits.grad
    assert grad is not None
    # Each member's gradient must equal the gradient of its own loss alone.
    for i in range(3):
        solo = logits.detach()[i].clone().requires_grad_(True)
        F.cross_entropy(solo, targets).backward()
        assert solo.grad is not None
        torch.testing.assert_close(grad[i], solo.grad, atol=1e-6, rtol=1e-5)


def test_targets_may_be_supplied_per_member() -> None:
    logits = torch.randn(2, 5, 3)
    targets = torch.randint(0, 3, (2, 5))

    losses = per_member_cross_entropy(logits, targets)

    for i in range(2):
        expected = F.cross_entropy(logits[i], targets[i])
        torch.testing.assert_close(losses[i], expected, atol=1e-6, rtol=1e-5)


def test_temperature_length_must_match_the_member_count() -> None:
    logits = torch.randn(3, 5, 4)
    targets = torch.randint(0, 4, (5,))

    with pytest.raises(ValueError, match="temperature"):
        per_member_cross_entropy(logits, targets, temperature=torch.ones(2))
