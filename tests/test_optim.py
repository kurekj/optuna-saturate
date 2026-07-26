import pytest
import torch

from optuna_saturate.vectorized.optim import VectorizedSGD


def _params_with_unit_gradient(k: int = 3) -> dict[str, torch.Tensor]:
    weight = torch.zeros(k, 2, 2, requires_grad=True)
    weight.grad = torch.ones(k, 2, 2)
    return {"weight": weight}


def test_member_with_zero_learning_rate_is_left_untouched() -> None:
    params = _params_with_unit_gradient(k=3)
    optimiser = VectorizedSGD(params, lr=torch.tensor([0.0, 0.1, 0.2]))

    optimiser.step()

    torch.testing.assert_close(params["weight"][0], torch.zeros(2, 2))


def test_step_size_is_proportional_to_the_member_learning_rate() -> None:
    params = _params_with_unit_gradient(k=2)
    optimiser = VectorizedSGD(params, lr=torch.tensor([0.1, 0.2]))

    optimiser.step()

    # Gradient is 1 everywhere, so each member moves by exactly -lr.
    torch.testing.assert_close(params["weight"][0], torch.full((2, 2), -0.1))
    torch.testing.assert_close(params["weight"][1], torch.full((2, 2), -0.2))


def test_learning_rate_broadcasts_across_parameters_of_different_rank() -> None:
    weight = torch.zeros(2, 3, 4, requires_grad=True)
    weight.grad = torch.ones(2, 3, 4)
    bias = torch.zeros(2, 4, requires_grad=True)
    bias.grad = torch.ones(2, 4)
    optimiser = VectorizedSGD({"weight": weight, "bias": bias}, lr=torch.tensor([0.5, 1.0]))

    optimiser.step()

    torch.testing.assert_close(weight[0], torch.full((3, 4), -0.5))
    torch.testing.assert_close(bias[1], torch.full((4,), -1.0))


def test_parameters_without_a_gradient_are_skipped() -> None:
    weight = torch.zeros(2, 2, requires_grad=True)  # grad stays None
    optimiser = VectorizedSGD({"weight": weight}, lr=torch.tensor([0.1, 0.1]))

    optimiser.step()  # must not raise

    torch.testing.assert_close(weight, torch.zeros(2, 2))


def test_zero_grad_clears_gradients() -> None:
    params = _params_with_unit_gradient(k=2)
    optimiser = VectorizedSGD(params, lr=torch.tensor([0.1, 0.1]))

    optimiser.zero_grad()

    assert params["weight"].grad is None


def test_learning_rate_length_must_match_the_member_count() -> None:
    params = _params_with_unit_gradient(k=3)
    with pytest.raises(ValueError, match="learning rate"):
        VectorizedSGD(params, lr=torch.tensor([0.1, 0.2]))


def test_step_does_not_record_itself_on_the_autograd_graph() -> None:
    params = _params_with_unit_gradient(k=2)
    optimiser = VectorizedSGD(params, lr=torch.tensor([0.1, 0.1]))

    optimiser.step()

    assert params["weight"].grad_fn is None
    assert params["weight"].is_leaf
