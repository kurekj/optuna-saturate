"""Numerical parity for a convolutional network.

Plan 1 proved parity for linear layers only. The benchmark trains convolutions,
and the paper claims vectorisation in general, so the proof has to cover them
too. Small sizes and CPU: this runs in CI.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from benchmarks.models import SmallCNN, build_cnn_members, parameter_count
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedSGD

K = 3
STEPS = 8
WIDTH = 4
LEARNING_RATES = [0.02, 0.05, 0.1]


def _batches(n: int, size: int = 8) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(size, 1, 28, 28, generator=generator),
            torch.randint(0, 10, (size,), generator=generator),
        )
        for _ in range(n)
    ]


def test_the_model_carries_no_batch_normalisation() -> None:
    """Batch norm buffers do not survive the vectorised call; guard against a regression."""
    model = SmallCNN(width=WIDTH)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_the_model_has_no_buffers_to_lose() -> None:
    assert list(SmallCNN(width=WIDTH).buffers()) == []


def test_parameter_count_is_in_the_range_the_project_targets() -> None:
    """The spec calls 1e4-1e6 parameters 'small models'; the benchmark must sit inside it."""
    assert 10_000 < parameter_count(SmallCNN(width=16)) < 1_000_000


def test_stacked_forward_matches_each_convolutional_member_alone() -> None:
    members = build_cnn_members(k=K, width=WIDTH)
    reference = [copy.deepcopy(m).eval() for m in members]
    ensemble = StackedEnsemble(members)
    x = torch.randn(5, 1, 28, 28)

    stacked = ensemble.forward(x)

    for i, model in enumerate(reference):
        with torch.no_grad():
            expected = model(x)
        torch.testing.assert_close(stacked[i], expected, atol=1e-5, rtol=1e-4)


def test_vectorised_training_of_a_cnn_reproduces_independent_training() -> None:
    """The load-bearing claim, extended from linear layers to convolutions."""
    members = build_cnn_members(k=K, width=WIDTH)
    reference_models = [copy.deepcopy(m) for m in members]
    batches = _batches(STEPS)

    for model, lr in zip(reference_models, LEARNING_RATES, strict=True):
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        for inputs, targets in batches:
            optimiser.zero_grad()
            torch.nn.functional.cross_entropy(model(inputs), targets).backward()
            optimiser.step()

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(ensemble.params, lr=torch.tensor(LEARNING_RATES))
    train_steps(ensemble, optimiser, batches)

    for i, model in enumerate(reference_models):
        actual = ensemble.member_state_dict(i)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(
                actual[name],
                expected,
                atol=1e-5,
                rtol=1e-4,
                msg=lambda s, i=i, name=name: f"member {i}, parameter {name}: {s}",
            )
