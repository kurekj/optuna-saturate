"""A vectorised study with AdamW, per-member weight decay and bfloat16 autocast.

Three shape-preserving hyperparameters vary within a group: the learning rate
and the weight decay go to the optimiser, the dropout probability goes to the
model through ``member_hp``. Synthetic data, so the example needs no download:

    python examples/adamw_mixed_precision.py
"""

from __future__ import annotations

from typing import Any

import optuna
import torch
from torch import Tensor, nn

import optuna_saturate as osat
from optuna_saturate.vectorized.dropout import member_dropout


class Net(nn.Module):
    """A two-layer classifier with dropout on the hidden layer.

    ``hidden`` changes shapes and is shared by the group; the dropout
    probability arrives per member as a zero-dimensional tensor in ``hp``.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        h = torch.relu(self.fc1(x))
        if hp is not None and "dropout" in hp:
            h = member_dropout(h, p=hp["dropout"], training=self.training)
        return self.fc2(h)


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


@osat.vectorizable(over=["learning_rate", "weight_decay", "dropout"])
def objective(trial: Any, ctx: Any) -> list[float]:
    hidden = trial.suggest_int("hidden", 16, 64)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    ensemble = ctx.stack([Net(hidden) for _ in range(ctx.k)])
    optimiser = ctx.adamw(ensemble, lr=lr, weight_decay=weight_decay)
    for _ in range(3):
        for x, y in TRAIN:
            ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": dropout})
    return ctx.accuracy(ensemble, VALID)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else None
    study = optuna.create_study(direction="maximize")
    osat.saturate(
        study,
        objective,
        n_trials=32,
        group_size=8,
        device=device,
        autocast_dtype=torch.bfloat16,
    )

    print(f"trials run     : {len(study.trials)}")
    print(f"best accuracy  : {study.best_value:.4f}")
    print(f"best parameters: {study.best_params}")


if __name__ == "__main__":
    main()
