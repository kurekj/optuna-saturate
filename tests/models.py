"""Minimal models used across the test suite."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from optuna_saturate.vectorized.dropout import member_dropout


class TinyMLP(nn.Module):
    """A two-layer MLP honouring the StackedEnsemble forward contract.

    The ``hp`` argument carries per-member hyperparameters as zero-dimensional
    tensors. This model ignores it; models exercising per-member dropout use it.
    """

    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def build_members(k: int, seed: int = 0, **kwargs: int) -> list[TinyMLP]:
    """Build ``k`` independently initialised models with reproducible weights."""
    members = []
    for i in range(k):
        torch.manual_seed(seed + i)
        members.append(TinyMLP(**kwargs))
    return members


class DropoutMLP(nn.Module):
    """MLP applying per-member dropout supplied through the ``hp`` mapping."""

    def __init__(self, in_features: int = 8, hidden: int = 512, out_features: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        h = torch.relu(self.fc1(x))
        if hp is not None and "dropout" in hp:
            h = member_dropout(h, p=hp["dropout"], training=self.training)
        return self.fc2(h)


def build_dropout_members(k: int, seed: int = 0, **kwargs: int) -> list[DropoutMLP]:
    """Build ``k`` dropout-enabled models with reproducible weights."""
    members = []
    for i in range(k):
        torch.manual_seed(seed + i)
        model = DropoutMLP(**kwargs)
        model.train()
        members.append(model)
    return members
