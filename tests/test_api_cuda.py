"""Public API checks that need a physical device.

The CPU suite cannot see device-placement bugs: with one device everything is
trivially co-located. These tests exist to catch a tensor handed to the user, or
taken from them, that ends up on the wrong device.
"""

from __future__ import annotations

from typing import Any

import optuna
import pytest
import torch

import optuna_saturate as osat
from optuna_saturate.core.context import VectorizedContext
from tests.objectives import Net, fixed_batches, make_objective

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def _study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize")


def test_saturate_runs_end_to_end_on_the_device() -> None:
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4, device="cuda:0")

    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_the_learning_rate_vector_lands_on_the_model_device() -> None:
    """The user never writes .to(device) for something the library handed them."""
    seen: dict[str, torch.device] = {}
    study = _study()

    def objective(trial: Any, ctx: Any) -> list[float]:
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        seen["lr"] = lr.device
        ensemble = ctx.stack([Net() for _ in range(ctx.k)])
        seen["params"] = ensemble.params["fc1.weight"].device
        ctx.sgd(ensemble, lr=lr)  # would raise on a device mismatch
        return [0.0] * ctx.k

    osat.saturate(
        study,
        osat.vectorizable(over=["lr"])(objective),
        n_trials=4,
        group_size=4,
        device="cuda:0",
    )

    assert seen["lr"].type == "cuda"
    assert seen["params"].type == "cuda"


def test_member_hp_is_moved_to_the_device_by_step() -> None:
    """A per-member hyperparameter built on the CPU must still work."""
    ctx = VectorizedContext(k=2, declared=("dropout",), device="cuda:0")
    ensemble = ctx.stack([Net() for _ in range(2)])
    optimiser = ctx.sgd(ensemble, lr=torch.full((2,), 0.1))  # CPU tensor on purpose
    x, y = fixed_batches(n=1)[0]  # CPU tensors on purpose

    losses = ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": torch.tensor([0.0, 0.5])})

    assert losses.shape == (2,)
    assert losses.device.type == "cuda"


def test_accuracy_returns_plain_floats_not_device_tensors() -> None:
    ctx = VectorizedContext(k=3, declared=(), device="cuda:0")
    ensemble = ctx.stack([Net() for _ in range(3)])

    scores = ctx.accuracy(ensemble, fixed_batches())

    assert all(isinstance(s, float) for s in scores)
