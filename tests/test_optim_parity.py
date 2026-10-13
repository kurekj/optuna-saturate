"""Numerical parity for every optimiser the library ships, against torch.optim.

``test_parity.py`` covers SGD with a per-member learning rate only. Each
optimiser here is run on K members with K different settings of every
hyperparameter it accepts (momentum, weight decay, the Adam betas), and compared
member by member with the reference optimiser applied to the same member alone.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from benchmarks.models import SmallCNN, build_cnn_members
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedAdam, VectorizedAdamW, VectorizedSGD
from tests.models import TinyMLP, build_members

K = 4
STEPS = 25
LR = [0.05, 0.1, 0.02, 0.2]
MOMENTUM = [0.0, 0.5, 0.9, 0.99]
WEIGHT_DECAY = [0.0, 1e-3, 1e-2, 5e-2]
BETA1 = [0.9, 0.8, 0.95, 0.5]
BETA2 = [0.999, 0.99, 0.9, 0.9999]
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def _batches(n: int, batch: int = 12) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(batch, 8, generator=generator),
            torch.randint(0, 4, (batch,), generator=generator),
        )
        for _ in range(n)
    ]


def _train_reference(
    model: nn.Module, optimiser: torch.optim.Optimizer, batches: list[tuple[Tensor, Tensor]]
) -> dict[str, Tensor]:
    for inputs, targets in batches:
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), targets).backward()
        optimiser.step()
    return dict(model.state_dict())


def _assert_members_match(ensemble: StackedEnsemble, expected: list[dict[str, Tensor]]) -> None:
    for i, reference in enumerate(expected):
        actual = ensemble.member_state_dict(i)
        for name, tensor in reference.items():
            torch.testing.assert_close(
                actual[name],
                tensor,
                **TOLERANCE,
                msg=lambda s, i=i, name=name: f"member {i}, parameter {name}: {s}",
            )


def _members() -> list[TinyMLP]:
    torch.manual_seed(0)
    return build_members(k=K, in_features=8, hidden=16, out_features=4)


# --- SGD -------------------------------------------------------------------


@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_with_momentum_and_weight_decay_matches_torch(nesterov: bool) -> None:
    members = _members()
    batches = _batches(STEPS)
    momentum = [0.5, 0.9, 0.99, 0.8] if nesterov else MOMENTUM  # nesterov needs momentum > 0

    expected = []
    for i, model in enumerate(members):
        clone = copy.deepcopy(model)
        optimiser = torch.optim.SGD(
            clone.parameters(),
            lr=LR[i],
            momentum=momentum[i],
            weight_decay=WEIGHT_DECAY[i],
            nesterov=nesterov,
        )
        expected.append(_train_reference(clone, optimiser, batches))

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedSGD(
        ensemble.params,
        lr=torch.tensor(LR),
        momentum=torch.tensor(momentum),
        weight_decay=torch.tensor(WEIGHT_DECAY),
        nesterov=nesterov,
    )
    train_steps(ensemble, optimiser, batches)

    _assert_members_match(ensemble, expected)


# --- Adam and AdamW --------------------------------------------------------


@pytest.mark.parametrize(
    ("vectorised", "reference"),
    [(VectorizedAdam, torch.optim.Adam), (VectorizedAdamW, torch.optim.AdamW)],
    ids=["adam", "adamw"],
)
def test_adam_family_matches_torch_with_per_member_betas_and_weight_decay(
    vectorised: Callable[..., Any], reference: Callable[..., torch.optim.Optimizer]
) -> None:
    members = _members()
    batches = _batches(STEPS)

    expected = []
    for i, model in enumerate(members):
        clone = copy.deepcopy(model)
        optimiser = reference(
            clone.parameters(),
            lr=LR[i],
            betas=(BETA1[i], BETA2[i]),
            weight_decay=WEIGHT_DECAY[i],
        )
        expected.append(_train_reference(clone, optimiser, batches))

    ensemble = StackedEnsemble(members)
    optimiser_v = vectorised(
        ensemble.params,
        lr=torch.tensor(LR),
        betas=(torch.tensor(BETA1), torch.tensor(BETA2)),
        weight_decay=torch.tensor(WEIGHT_DECAY),
    )
    train_steps(ensemble, optimiser_v, batches)

    _assert_members_match(ensemble, expected)


def test_adam_and_adamw_differ_when_weight_decay_is_non_zero() -> None:
    """Guards against the decoupled flag being silently ignored."""
    members = _members()
    batches = _batches(5)
    results = []
    for cls in (VectorizedAdam, VectorizedAdamW):
        ensemble = StackedEnsemble([copy.deepcopy(m) for m in members])
        optimiser = cls(ensemble.params, lr=0.1, weight_decay=0.1)
        train_steps(ensemble, optimiser, batches)
        results.append(ensemble.member_state_dict(0)["fc1.weight"])
    assert not torch.allclose(results[0], results[1])


def test_shared_float_hyperparameters_match_torch() -> None:
    """A plain float, not a vector, must behave as the same value on every member."""
    members = _members()
    batches = _batches(STEPS)
    expected = []
    for model in members:
        clone = copy.deepcopy(model)
        optimiser = torch.optim.AdamW(clone.parameters(), lr=0.01, weight_decay=0.02)
        expected.append(_train_reference(clone, optimiser, batches))

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedAdamW(ensemble.params, lr=0.01, weight_decay=0.02)
    train_steps(ensemble, optimiser, batches)

    _assert_members_match(ensemble, expected)


# --- the convolutional benchmark model, Adam, to the same tolerance ---------


def test_cnn_trained_with_adam_matches_independent_training() -> None:
    members = build_cnn_members(k=3, width=4)
    reference_models = [copy.deepcopy(m) for m in members]
    generator = torch.Generator().manual_seed(7)
    batches = [
        (
            torch.randn(8, 1, 28, 28, generator=generator),
            torch.randint(0, 10, (8,), generator=generator),
        )
        for _ in range(8)
    ]
    lrs = [1e-3, 3e-3, 1e-2]
    expected = [
        _train_reference(model, torch.optim.Adam(model.parameters(), lr=lr), batches)
        for model, lr in zip(reference_models, lrs, strict=True)
    ]

    ensemble = StackedEnsemble(members)
    optimiser = VectorizedAdam(ensemble.params, lr=torch.tensor(lrs))
    train_steps(ensemble, optimiser, batches)

    _assert_members_match(ensemble, expected)


def test_benchmark_model_is_a_small_cnn() -> None:
    assert isinstance(SmallCNN(width=4), nn.Module)


# --- a parameter that receives its gradient late -----------------------------


class _GatedMLP(nn.Module):
    """An MLP whose second head is used only once a flag is set.

    Until then its parameters receive no gradient, so the optimiser must keep a
    separate step count for them, as torch.optim does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.head_a = nn.Linear(16, 4)
        self.head_b = nn.Linear(16, 4)
        self.use_b = False

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        h = torch.relu(self.fc1(x))
        return self.head_b(h) if self.use_b else self.head_a(h)


@pytest.mark.parametrize(
    "reference_cls",
    [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD],
    ids=["adam", "adamw", "sgd"],
)
def test_a_parameter_first_used_on_a_later_step_gets_its_own_step_count(
    reference_cls: Any,
) -> None:
    members = []
    for i in range(3):
        torch.manual_seed(i)
        members.append(_GatedMLP())
    batches = _batches(12)
    lrs = [0.01, 0.03, 0.1]
    extra = {"momentum": 0.9} if reference_cls is torch.optim.SGD else {}

    expected = []
    for model, lr in zip(members, lrs, strict=True):
        clone = copy.deepcopy(model)
        optimiser = reference_cls(clone.parameters(), lr=lr, **extra)
        for step, (inputs, targets) in enumerate(batches):
            clone.use_b = step >= 5  # head_b first receives a gradient on step 5
            optimiser.zero_grad()
            torch.nn.functional.cross_entropy(clone(inputs), targets).backward()
            optimiser.step()
        expected.append(dict(clone.state_dict()))

    ensemble = StackedEnsemble(members)
    vectorised_cls = {
        torch.optim.Adam: VectorizedAdam,
        torch.optim.AdamW: VectorizedAdamW,
        torch.optim.SGD: VectorizedSGD,
    }[reference_cls]
    optimiser_v = vectorised_cls(ensemble.params, lr=torch.tensor(lrs), **extra)
    for step, (inputs, targets) in enumerate(batches):
        ensemble._base.use_b = step >= 5
        train_steps(ensemble, optimiser_v, [(inputs, targets)])

    _assert_members_match(ensemble, expected)


def test_adamw_defaults_to_the_same_weight_decay_as_torch() -> None:
    members = _members()
    batches = _batches(STEPS)
    expected = []
    for model in members:
        clone = copy.deepcopy(model)
        optimiser = torch.optim.AdamW(clone.parameters(), lr=0.01)
        expected.append(_train_reference(clone, optimiser, batches))

    ensemble = StackedEnsemble(members)
    train_steps(ensemble, VectorizedAdamW(ensemble.params, lr=0.01), batches)

    _assert_members_match(ensemble, expected)
    assert VectorizedAdamW(ensemble.params, lr=0.01).weight_decay.exact.tolist() == [0.01] * K


def test_betas_close_to_one_keep_their_precision() -> None:
    """0.99999999 rounds to 1.0 in float32; the bias correction must not see that."""
    ensemble = StackedEnsemble(_members())
    optimiser = VectorizedAdam(ensemble.params, lr=0.01, betas=(0.9, 0.99999999))
    assert optimiser.beta2.exact[0].item() == 0.99999999
    train_steps(ensemble, optimiser, _batches(3))  # must not produce NaN or inf
    for tensor in ensemble.params.values():
        assert torch.isfinite(tensor).all()


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": -0.1}, "learning rate"),
        ({"lr": 0.1, "momentum": torch.tensor([0.9, -0.1, 0.5, 0.5])}, "momentum"),
        ({"lr": 0.1, "weight_decay": -1e-3}, "weight decay"),
        ({"lr": 0.1, "momentum": torch.tensor([0.9, 0.0, 0.5, 0.5]), "nesterov": True}, "nesterov"),
    ],
)
def test_sgd_rejects_values_torch_optim_rejects(kwargs: dict[str, Any], message: str) -> None:
    ensemble = StackedEnsemble(_members())
    with pytest.raises(ValueError, match=message):
        VectorizedSGD(ensemble.params, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": 0.1, "betas": (0.9, 1.0)}, "beta2"),
        ({"lr": 0.1, "betas": (torch.tensor([0.9, 0.9, 1.5, 0.9]), 0.999)}, "beta1"),
        ({"lr": 0.1, "eps": -1e-8}, "eps"),
    ],
)
def test_adam_rejects_values_torch_optim_rejects(kwargs: dict[str, Any], message: str) -> None:
    ensemble = StackedEnsemble(_members())
    with pytest.raises(ValueError, match=message):
        VectorizedAdam(ensemble.params, **kwargs)


def test_a_hyperparameter_vector_of_the_wrong_length_is_rejected_by_name() -> None:
    ensemble = StackedEnsemble(_members())
    with pytest.raises(ValueError, match="weight decay"):
        VectorizedAdam(ensemble.params, lr=0.1, weight_decay=torch.tensor([0.0, 0.1]))
