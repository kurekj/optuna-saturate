"""Parity with stochastic dropout, where exact equality is not the right test.

With dropout active, batched and independent training draw different random
masks, so no two runs agree bit for bit, not even two independent runs. The
test compares the mean training loss over 24 replicates of each path, for a
member trained inside a batched group and the same member trained alone with
``torch.nn.functional.dropout`` at the same rate. It checks mean agreement, not
distributional equivalence; a member trained at a deliberately wrong rate must
fail the same criterion. Both paths use inverted dropout, so the expected
activations agree and only the mask differs.

The test draws replicates of each path, each with its own seed, and compares
the mean training loss of each member over its last steps, a quantity the
dropout rate drives directly. The two means must agree to within three pooled
standard errors, and a companion test applies the same rule to a member trained
at a deliberately wrong rate and checks that it is rejected.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedSGD
from tests.models import DropoutMLP, build_dropout_members

K = 3
DROPOUT = [0.0, 0.3, 0.6]
LR = 0.1
STEPS = 30
REPLICATES = 24


class ReferenceDropoutMLP(nn.Module):
    """The same network as DropoutMLP, with PyTorch's own dropout for one rate."""

    def __init__(self, source: DropoutMLP, p: float) -> None:
        super().__init__()
        self.fc1 = copy.deepcopy(source.fc1)
        self.fc2 = copy.deepcopy(source.fc2)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.fc1(x))
        h = F.dropout(h, p=self.p, training=self.training)
        return self.fc2(h)


def _batches(n: int) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(2024)
    return [
        (torch.randn(32, 8, generator=generator), torch.randint(0, 4, (32,), generator=generator))
        for _ in range(n)
    ]


LAST_STEPS = 10


def _independent_final_losses(members: list[DropoutMLP], batches, seed: int) -> list[float]:
    """Mean training loss of each member over the last steps, trained alone."""
    torch.manual_seed(seed)
    losses = []
    for member, p in zip(members, DROPOUT, strict=True):
        model = ReferenceDropoutMLP(member, p)
        model.train()
        optimiser = torch.optim.SGD(model.parameters(), lr=LR)
        history = []
        for inputs, targets in batches:
            optimiser.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimiser.step()
            history.append(float(loss.detach()))
        losses.append(sum(history[-LAST_STEPS:]) / LAST_STEPS)
    return losses


def _batched_final_losses(
    members: list[DropoutMLP], batches, seed: int, rates: list[float] = DROPOUT
) -> list[float]:
    """Mean training loss of each member over the last steps, trained as a group."""
    torch.manual_seed(seed)
    ensemble = StackedEnsemble([copy.deepcopy(m) for m in members])
    optimiser = VectorizedSGD(ensemble.params, lr=LR)
    history = train_steps(ensemble, optimiser, batches, member_hp={"dropout": torch.tensor(rates)})
    tail = torch.stack(history[-LAST_STEPS:]).mean(dim=0)
    return [float(v) for v in tail]


def _mean_and_stderr(samples: list[float]) -> tuple[float, float]:
    n = len(samples)
    mean = sum(samples) / n
    variance = sum((s - mean) ** 2 for s in samples) / (n - 1)
    return mean, math.sqrt(variance / n)


def _agrees(a: list[float], b: list[float]) -> bool:
    """The acceptance rule: means within three pooled standard errors."""
    mean_a, se_a = _mean_and_stderr(a)
    mean_b, se_b = _mean_and_stderr(b)
    return abs(mean_a - mean_b) < 3 * math.sqrt(se_a**2 + se_b**2) + 1e-6


def test_dropout_outcomes_agree_in_the_mean_on_both_paths() -> None:
    members = build_dropout_members(k=K, in_features=8, hidden=64, out_features=4)
    batches = _batches(STEPS)

    # Different seed ranges on the two paths, so the replicates are independent draws.
    independent = [_independent_final_losses(members, batches, seed) for seed in range(REPLICATES)]
    batched = [
        _batched_final_losses(members, batches, seed) for seed in range(100, 100 + REPLICATES)
    ]

    for i, p in enumerate(DROPOUT):
        a = [run[i] for run in independent]
        b = [run[i] for run in batched]
        if p == 0.0:
            # No randomness at all: the two paths must agree exactly.
            assert abs(_mean_and_stderr(a)[0] - _mean_and_stderr(b)[0]) < 1e-5
        else:
            assert _agrees(a, b), (p, _mean_and_stderr(a), _mean_and_stderr(b))


def test_the_acceptance_rule_rejects_a_member_trained_at_the_wrong_rate() -> None:
    """Power check: the same rule, applied to a deliberately wrong rate, must fail."""
    members = build_dropout_members(k=K, in_features=8, hidden=64, out_features=4)
    batches = _batches(STEPS)
    wrong = [0.0, 0.6, 0.3]  # members 1 and 2 swapped

    independent = [_independent_final_losses(members, batches, seed) for seed in range(REPLICATES)]
    control = [
        _batched_final_losses(members, batches, seed, rates=wrong)
        for seed in range(100, 100 + REPLICATES)
    ]

    for i in (1, 2):
        assert not _agrees([run[i] for run in independent], [run[i] for run in control]), i
