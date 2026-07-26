"""Objective functions used to exercise the public API."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class Net(nn.Module):
    """Two-layer MLP whose width is a shape-changing hyperparameter."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def fixed_batches(n: int = 2, size: int = 12) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(size, 8, generator=generator),
            torch.randint(0, 4, (size,), generator=generator),
        )
        for _ in range(n)
    ]


def make_objective(calls: list[int], device: str | None = None) -> Any:
    """An objective that trains a group and records how often it was called."""

    def objective(trial: Any, ctx: Any) -> list[float]:
        hidden = trial.suggest_int("hidden", 8, 32)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)

        ensemble = ctx.stack([Net(hidden) for _ in range(ctx.k)])
        # Recorded after stack(), so the space probe -- which stops there -- is
        # not counted. Only real group runs land in `calls`.
        calls.append(ctx.k)
        optimiser = ctx.sgd(ensemble, lr=lr)
        batches = fixed_batches()
        for x, y in batches:
            ctx.step(ensemble, optimiser, x, y)
        return ctx.accuracy(ensemble, batches)

    return objective


def make_bad_over_objective() -> Any:
    """Declares the width as vectorisable, which it is not."""

    def objective(trial: Any, ctx: Any) -> list[float]:
        hidden = trial.suggest_int("hidden", 8, 32)
        # Each member gets its own width -- exactly what cannot be vectorised.
        ensemble = ctx.stack([Net(hidden + i) for i in range(ctx.k)])
        return [0.0] * ensemble.k

    return objective
