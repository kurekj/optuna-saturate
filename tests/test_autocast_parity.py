"""Parity under mixed precision.

Under ``torch.autocast`` the forward pass rounds to bfloat16, and a batched
matrix product accumulates in a different order from a single one, so the two
paths cannot agree bit for bit the way they do in float32. The claim that can
be tested is a bound: for every member, the gap between batched and independent
training under bfloat16 must be no larger than the gap that bfloat16 itself
opens between the independent path and its float32 counterpart. Batching then
adds no error beyond what mixed precision already costs. For SGD the two gaps
come out equal to within rounding, so a margin of one half is allowed on the
comparison; for Adam, whose normalised update amplifies rounding noise the
most, the batching gap is several times the smaller. The bound is checked for
both.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedAdam, VectorizedSGD
from tests.models import build_members

K = 3
STEPS = 10
LR = [0.05, 0.1, 0.2]


def _batches(n: int, batch: int = 16) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(99)
    return [
        (
            torch.randn(batch, 8, generator=generator),
            torch.randint(0, 4, (batch,), generator=generator),
        )
        for _ in range(n)
    ]


def _train_independently(
    models: list[nn.Module],
    batches: list[tuple[Tensor, Tensor]],
    adam: bool,
    dtype: torch.dtype | None,
) -> list[dict[str, Tensor]]:
    for model, lr in zip(models, LR, strict=True):
        optimiser = (
            torch.optim.Adam(model.parameters(), lr=lr)
            if adam
            else torch.optim.SGD(model.parameters(), lr=lr)
        )
        for inputs, targets in batches:
            optimiser.zero_grad()
            with torch.autocast(device_type="cpu", dtype=dtype, enabled=dtype is not None):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            loss.backward()
            optimiser.step()
    return [dict(model.state_dict()) for model in models]


def _train_batched(
    models: list[nn.Module],
    batches: list[tuple[Tensor, Tensor]],
    adam: bool,
    dtype: torch.dtype | None,
) -> list[dict[str, Tensor]]:
    ensemble = StackedEnsemble(models)
    optimiser = (
        VectorizedAdam(ensemble.params, lr=torch.tensor(LR))
        if adam
        else VectorizedSGD(ensemble.params, lr=torch.tensor(LR))
    )
    train_steps(ensemble, optimiser, batches, autocast_dtype=dtype)
    return [ensemble.member_state_dict(i) for i in range(K)]


def _largest_gap(a: dict[str, Tensor], b: dict[str, Tensor]) -> float:
    return max(float((a[name] - b[name]).abs().max()) for name in a)


@pytest.mark.parametrize("adam", [False, True], ids=["sgd", "adam"])
def test_batching_adds_no_error_beyond_what_bfloat16_itself_costs(adam: bool) -> None:
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=32, out_features=4)
    batches = _batches(STEPS)
    clones = [[copy.deepcopy(m) for m in members] for _ in range(3)]

    independent_fp32 = _train_independently(clones[0], batches, adam, dtype=None)
    independent_bf16 = _train_independently(clones[1], batches, adam, dtype=torch.bfloat16)
    batched_bf16 = _train_batched(clones[2], batches, adam, dtype=torch.bfloat16)

    for i in range(K):
        batching_gap = _largest_gap(batched_bf16[i], independent_bf16[i])
        precision_gap = _largest_gap(independent_bf16[i], independent_fp32[i])
        assert batching_gap <= 1.5 * precision_gap, (i, batching_gap, precision_gap)


def test_sgd_under_bfloat16_matches_to_the_precision_of_bfloat16() -> None:
    """For SGD the bound is loose; the paths in fact agree to a few bf16 ulps."""
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=32, out_features=4)
    batches = _batches(STEPS)

    expected = _train_independently(
        [copy.deepcopy(m) for m in members], batches, False, torch.bfloat16
    )
    actual = _train_batched([copy.deepcopy(m) for m in members], batches, False, torch.bfloat16)

    for i in range(K):
        for name, tensor in expected[i].items():
            torch.testing.assert_close(actual[i][name], tensor, atol=5e-3, rtol=2e-2)


def test_parameters_stay_in_float32_under_autocast() -> None:
    """Autocast touches activations only; the stacked parameters keep full precision."""
    torch.manual_seed(0)
    ensemble = StackedEnsemble(build_members(k=2, in_features=8, hidden=16, out_features=4))
    optimiser = VectorizedSGD(ensemble.params, lr=0.1)
    train_steps(ensemble, optimiser, _batches(2), autocast_dtype=torch.bfloat16)
    for tensor in ensemble.params.values():
        assert tensor.dtype == torch.float32


def test_bfloat16_training_differs_from_float32_training() -> None:
    """Confirms the autocast flag reaches the forward pass rather than being ignored."""
    torch.manual_seed(0)
    members = build_members(k=K, in_features=8, hidden=32, out_features=4)
    batches = _batches(STEPS)
    fp32 = _train_batched([copy.deepcopy(m) for m in members], batches, False, None)
    bf16 = _train_batched([copy.deepcopy(m) for m in members], batches, False, torch.bfloat16)
    assert not torch.equal(fp32[0]["fc1.weight"], bf16[0]["fc1.weight"])
