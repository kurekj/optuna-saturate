"""Numerical parity for the Transformer used in the evaluation, trained with AdamW.

Attention, layer normalisation, residual connections and GELU are a different
set of operations from the linear and convolutional layers the other parity
tests cover, and the optimiser is AdamW with per-member weight decay. Small
sizes and CPU: this runs in CI.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from benchmarks.tasks import TinyViT
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedAdamW

K = 3
STEPS = 6
LR = [1e-3, 3e-3, 1e-2]
WEIGHT_DECAY = [0.0, 1e-2, 1e-1]


def _members() -> list[TinyViT]:
    members = []
    for i in range(K):
        torch.manual_seed(i)
        members.append(TinyViT(embed_dim=32, depth=2, heads=4))
    return members


def _batches(n: int, size: int = 6) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(11)
    return [
        (
            torch.randn(size, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (size,), generator=generator),
        )
        for _ in range(n)
    ]


def test_the_model_carries_no_buffers() -> None:
    """LayerNorm has parameters only; nothing is lost in the vectorised call."""
    assert list(TinyViT(embed_dim=32).buffers()) == []


def test_stacked_forward_matches_each_transformer_member_alone() -> None:
    members = _members()
    reference = [copy.deepcopy(m).eval() for m in members]
    ensemble = StackedEnsemble(members).eval()
    x = torch.randn(4, 3, 32, 32)

    stacked = ensemble.forward(x)

    for i, model in enumerate(reference):
        with torch.no_grad():
            expected = model(x)
        torch.testing.assert_close(stacked[i], expected, atol=1e-5, rtol=1e-4)


def test_vectorised_adamw_training_of_a_transformer_matches_independent_training() -> None:
    members = _members()
    reference_models = [copy.deepcopy(m) for m in members]
    batches = _batches(STEPS)

    for model, lr, wd in zip(reference_models, LR, WEIGHT_DECAY, strict=True):
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        for inputs, targets in batches:
            optimiser.zero_grad()
            torch.nn.functional.cross_entropy(model(inputs), targets).backward()
            optimiser.step()

    ensemble = StackedEnsemble(members)
    optimiser_v = VectorizedAdamW(
        ensemble.params, lr=torch.tensor(LR), weight_decay=torch.tensor(WEIGHT_DECAY)
    )
    train_steps(ensemble, optimiser_v, batches)

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
