"""The load-bearing test of this project.

If training K configurations as one batched model does not reproduce K
independent trainings, the vectorisation layer is not a speed-up but a different
algorithm, and no benchmark built on it would mean anything.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.losses import per_member_cross_entropy
from optuna_saturate.vectorized.optim import VectorizedSGD
from tests.models import TinyMLP, build_members

K = 4
STEPS = 20
LEARNING_RATES = [0.05, 0.1, 0.2, 0.4]
TEMPERATURES = [0.5, 1.0, 1.5, 2.0]


def _fixed_batches(
    n: int, batch: int = 12, in_features: int = 8, classes: int = 4
) -> list[tuple[Tensor, Tensor]]:
    """Deterministic batches, identical for both training paths."""
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(batch, in_features, generator=generator),
            torch.randint(0, classes, (batch,), generator=generator),
        )
        for _ in range(n)
    ]


def _train_independently(
    models: list[TinyMLP],
    batches: list[tuple[Tensor, Tensor]],
    learning_rates: list[float],
    temperatures: list[float],
) -> list[dict[str, Tensor]]:
    """Reference path: each configuration trained on its own, the ordinary way."""
    for model, lr, temp in zip(models, learning_rates, temperatures, strict=True):
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        for inputs, targets in batches:
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(inputs) / temp, targets)
            loss.backward()
            optimiser.step()
    return [dict(model.state_dict()) for model in models]


def test_vectorised_training_reproduces_independent_training() -> None:
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=16, out_features=4)
    reference_models = [copy.deepcopy(m) for m in members]
    batches = _fixed_batches(STEPS)

    expected = _train_independently(reference_models, batches, LEARNING_RATES, TEMPERATURES)

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.tensor(LEARNING_RATES))
    train_steps(
        ensemble,
        optimiser,
        batches,
        temperature=torch.tensor(TEMPERATURES),
    )

    for i in range(K):
        actual = ensemble.member_state_dict(i)
        for name, reference_tensor in expected[i].items():
            torch.testing.assert_close(
                actual[name],
                reference_tensor,
                atol=1e-5,
                rtol=1e-4,
                msg=lambda s, i=i, name=name: f"member {i}, parameter {name}: {s}",
            )


def test_losses_reported_by_the_loop_match_independent_computation() -> None:
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=16, out_features=4)
    reference_models = [copy.deepcopy(m) for m in members]
    batches = _fixed_batches(3)

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.tensor(LEARNING_RATES))
    history = train_steps(ensemble, optimiser, batches, temperature=torch.tensor(TEMPERATURES))

    # Only the first step is compared: afterwards the reference path would need
    # its own optimiser state, which the parity test above already covers.
    first_inputs, first_targets = batches[0]
    for i, (model, temp) in enumerate(zip(reference_models, TEMPERATURES, strict=True)):
        with torch.no_grad():
            expected = torch.nn.functional.cross_entropy(model(first_inputs) / temp, first_targets)
        torch.testing.assert_close(history[0][i], expected, atol=1e-5, rtol=1e-4)


def test_parity_holds_when_every_member_shares_one_learning_rate() -> None:
    """Degenerate case: with identical hyperparameters all members must converge alike."""
    torch.manual_seed(0)
    members = build_members(k=3, in_features=8, hidden=16, out_features=4)
    # Force identical initial weights across members.
    for model in members[1:]:
        model.load_state_dict(members[0].state_dict())
    batches = _fixed_batches(10)

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.full((3,), 0.1))
    train_steps(ensemble, optimiser, batches)

    first = ensemble.member_state_dict(0)
    for i in (1, 2):
        other = ensemble.member_state_dict(i)
        for name in first:
            torch.testing.assert_close(other[name], first[name], atol=1e-6, rtol=1e-5)


def test_a_member_with_zero_learning_rate_keeps_its_initial_weights() -> None:
    torch.manual_seed(0)
    members = build_members(k=3, in_features=8, hidden=16, out_features=4)
    initial = copy.deepcopy(members[0].state_dict())
    batches = _fixed_batches(10)

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.tensor([0.0, 0.1, 0.1]))
    train_steps(ensemble, optimiser, batches)

    frozen = ensemble.member_state_dict(0)
    for name, tensor in initial.items():
        torch.testing.assert_close(frozen[name], tensor, atol=1e-7, rtol=1e-6)


def test_loss_decreases_over_training_for_every_member() -> None:
    """Sanity check that the loop optimises rather than merely running."""
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=16, out_features=4)
    # A single repeated batch is easy to overfit, so the loss must drop.
    batch = _fixed_batches(1)[0]
    batches = [batch] * 60

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.full((K,), 0.5))
    history = train_steps(ensemble, optimiser, batches)

    for i in range(K):
        assert history[-1][i] < history[0][i], f"member {i} did not improve"


def test_per_member_cross_entropy_is_used_consistently_by_the_loop() -> None:
    """Guards against the loop silently switching to a batch-wide reduction."""
    torch.manual_seed(0)
    members = build_members(k=2, in_features=8, hidden=16, out_features=4)
    batches = _fixed_batches(1)
    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.zeros(2))

    history = train_steps(ensemble, optimiser, batches)

    inputs, targets = batches[0]
    expected = per_member_cross_entropy(ensemble.forward(inputs), targets)
    torch.testing.assert_close(history[0], expected.detach(), atol=1e-6, rtol=1e-5)
