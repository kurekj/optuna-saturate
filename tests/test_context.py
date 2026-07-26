from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from optuna_saturate.core.context import VectorizedContext
from optuna_saturate.exceptions import ShapeChangingHyperparameterError


class Net(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _batch(n: int = 12) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(n, 8, generator=generator),
        torch.randint(0, 4, (n,), generator=generator),
    )


def test_stack_builds_an_ensemble_of_the_declared_size() -> None:
    ctx = VectorizedContext(k=3, declared=("lr",))
    ensemble = ctx.stack([Net() for _ in range(3)])
    assert ensemble.k == 3


def test_stack_rejects_a_member_count_other_than_k() -> None:
    ctx = VectorizedContext(k=3, declared=("lr",))
    with pytest.raises(ValueError, match="3"):
        ctx.stack([Net() for _ in range(2)])


def test_stack_names_the_hyperparameter_that_actually_changes_shapes() -> None:
    """The load-bearing check: a wrong `over` declaration must fail loudly.

    Declaring a width as vectorisable produces members whose weights differ in
    shape. Without this the stack would fail with an opaque tensor-shape error,
    or silently compute something wrong.
    """
    ctx = VectorizedContext(k=2, declared=("hidden",))

    with pytest.raises(ShapeChangingHyperparameterError) as excinfo:
        ctx.stack([Net(hidden=16), Net(hidden=32)])

    message = str(excinfo.value)
    assert "hidden" in message
    assert "fc1.weight" in message


def test_the_shape_error_reports_the_differing_shapes() -> None:
    ctx = VectorizedContext(k=2, declared=("hidden",))
    with pytest.raises(ShapeChangingHyperparameterError, match=r"16|32"):
        ctx.stack([Net(hidden=16), Net(hidden=32)])


def test_sgd_accepts_the_learning_rate_vector_from_the_trial_proxy() -> None:
    ctx = VectorizedContext(k=3, declared=("lr",))
    ensemble = ctx.stack([Net() for _ in range(3)])

    optimiser = ctx.sgd(ensemble, lr=torch.tensor([0.1, 0.2, 0.3]))

    assert optimiser.lr.shape == (3,)


def test_step_returns_one_loss_per_member() -> None:
    ctx = VectorizedContext(k=3, declared=("lr",))
    ensemble = ctx.stack([Net() for _ in range(3)])
    optimiser = ctx.sgd(ensemble, lr=torch.full((3,), 0.1))
    x, y = _batch()

    losses = ctx.step(ensemble, optimiser, x, y)

    assert losses.shape == (3,)


def test_step_actually_updates_the_parameters() -> None:
    ctx = VectorizedContext(k=2, declared=("lr",))
    ensemble = ctx.stack([Net() for _ in range(2)])
    optimiser = ctx.sgd(ensemble, lr=torch.full((2,), 0.5))
    before = ensemble.params["fc1.weight"].detach().clone()
    x, y = _batch()

    ctx.step(ensemble, optimiser, x, y)

    assert not torch.equal(ensemble.params["fc1.weight"], before)


def test_accuracy_returns_one_plain_float_per_member() -> None:
    ctx = VectorizedContext(k=3, declared=("lr",))
    ensemble = ctx.stack([Net() for _ in range(3)])

    scores = ctx.accuracy(ensemble, [_batch(), _batch()])

    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_accuracy_is_one_when_every_prediction_is_correct() -> None:
    """A member that always predicts class 0, scored against all-zero targets."""
    ctx = VectorizedContext(k=1, declared=())
    ensemble = ctx.stack([Net() for _ in range(1)])
    # Force class 0 to dominate for any input.
    with torch.no_grad():
        ensemble.params["fc2.bias"][0] = torch.tensor([50.0, 0.0, 0.0, 0.0])

    x = torch.randn(10, 8)
    y = torch.zeros(10, dtype=torch.long)

    assert ctx.accuracy(ensemble, [(x, y)]) == [1.0]
