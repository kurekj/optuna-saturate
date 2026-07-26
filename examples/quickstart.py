"""Smallest complete example of a vectorised Optuna study.

Trains eight learning rates at once as a single batched model, on synthetic data
so the example needs no download. Run:

    python examples/quickstart.py
"""

from __future__ import annotations

from typing import Any

import optuna
import torch
from torch import Tensor, nn

import optuna_saturate as osat


class Net(nn.Module):
    """A two-layer classifier. `hidden` changes shapes, so it is not vectorised."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def batches(count: int, size: int = 64) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(size, 8, generator=generator),
            torch.randint(0, 4, (size,), generator=generator),
        )
        for _ in range(count)
    ]


TRAIN = batches(20)
VALID = batches(5)


# Every suggest_* call must come before ctx.stack(): that is how the library
# discovers the search space without paying for a training run.
@osat.vectorizable(over=["learning_rate"])
def objective(trial: Any, ctx: Any) -> list[float]:
    hidden = trial.suggest_int("hidden", 16, 64)  # one value for the group
    lr = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)  # one per member

    ensemble = ctx.stack([Net(hidden) for _ in range(ctx.k)])
    optimiser = ctx.sgd(ensemble, lr=lr)
    for _ in range(3):
        for x, y in TRAIN:
            ctx.step(ensemble, optimiser, x, y)
    return ctx.accuracy(ensemble, VALID)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else None
    study = optuna.create_study(direction="maximize")
    osat.saturate(study, objective, n_trials=32, group_size=8, device=device)

    print(f"trials run     : {len(study.trials)}")
    print(f"best accuracy  : {study.best_value:.4f}")
    print(f"best parameters: {study.best_params}")


if __name__ == "__main__":
    main()
