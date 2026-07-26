import copy

import torch

from optuna_saturate.vectorized.ensemble import StackedEnsemble
from tests.models import build_members


def test_forward_returns_one_output_row_per_member() -> None:
    members = build_members(k=3, in_features=8, hidden=16, out_features=4)
    ensemble = StackedEnsemble(members)
    x = torch.randn(5, 8)

    out = ensemble.forward(x)

    assert out.shape == (3, 5, 4)


def test_forward_matches_each_member_evaluated_independently() -> None:
    members = build_members(k=4, in_features=8, hidden=16, out_features=4)
    reference = [copy.deepcopy(m).eval() for m in members]
    ensemble = StackedEnsemble(members)
    x = torch.randn(6, 8)

    stacked = ensemble.forward(x)

    for i, model in enumerate(reference):
        with torch.no_grad():
            expected = model(x)
        torch.testing.assert_close(stacked[i], expected, atol=1e-5, rtol=1e-4)


def test_k_reports_the_number_of_members() -> None:
    ensemble = StackedEnsemble(build_members(k=7))
    assert ensemble.k == 7


def test_stacked_parameters_carry_a_leading_member_dimension() -> None:
    ensemble = StackedEnsemble(build_members(k=3, in_features=8, hidden=16, out_features=4))
    assert ensemble.params["fc1.weight"].shape == (3, 16, 8)
    assert ensemble.params["fc2.bias"].shape == (3, 4)


def test_stacked_parameters_require_gradients() -> None:
    ensemble = StackedEnsemble(build_members(k=2))
    for tensor in ensemble.params.values():
        assert tensor.requires_grad


def test_backward_populates_a_gradient_for_every_stacked_parameter() -> None:
    ensemble = StackedEnsemble(build_members(k=3, in_features=8, hidden=16, out_features=4))
    x = torch.randn(5, 8)

    ensemble.forward(x).sum().backward()

    for name, tensor in ensemble.params.items():
        assert tensor.grad is not None, f"no gradient for {name}"
        assert tensor.grad.shape == tensor.shape


def test_zero_grad_clears_previously_accumulated_gradients() -> None:
    ensemble = StackedEnsemble(build_members(k=2, in_features=8, hidden=16, out_features=4))
    x = torch.randn(4, 8)
    ensemble.forward(x).sum().backward()

    ensemble.zero_grad()

    for tensor in ensemble.params.values():
        assert tensor.grad is None or torch.count_nonzero(tensor.grad) == 0


def test_gradients_of_one_member_do_not_leak_into_another() -> None:
    """Only member 1 contributes to the loss, so only its slice may receive gradient."""
    ensemble = StackedEnsemble(build_members(k=3, in_features=8, hidden=16, out_features=4))
    x = torch.randn(4, 8)

    ensemble.forward(x)[1].sum().backward()

    grad = ensemble.params["fc1.weight"].grad
    assert grad is not None
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[2]) == 0
    assert torch.count_nonzero(grad[1]) > 0
