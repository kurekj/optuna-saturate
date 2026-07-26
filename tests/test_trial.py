from __future__ import annotations

import optuna
import pytest
import torch

from optuna_saturate.core.trial import ProbeComplete, ProbeTrial, VectorizedTrial
from optuna_saturate.exceptions import ShapeChangingHyperparameterError


def _study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize")


def _group(study: optuna.Study, k: int) -> list[optuna.Trial]:
    return [study.ask() for _ in range(k)]


def test_a_vectorised_name_yields_one_value_per_member() -> None:
    study = _study()
    proxy = VectorizedTrial(_group(study, 4), preserving=frozenset({"lr"}))

    lr = proxy.suggest_float("lr", 1e-4, 1e-1, log=True)

    assert isinstance(lr, torch.Tensor)
    assert lr.shape == (4,)


def test_a_vectorised_name_yields_distinct_values_across_members() -> None:
    study = _study()
    proxy = VectorizedTrial(_group(study, 8), preserving=frozenset({"lr"}))

    lr = proxy.suggest_float("lr", 1e-4, 1e-1, log=True)

    assert len(set(lr.tolist())) > 1


def test_a_non_vectorised_name_yields_a_plain_scalar() -> None:
    study = _study()
    proxy = VectorizedTrial(_group(study, 4), preserving=frozenset({"lr"}))

    hidden = proxy.suggest_int("hidden", 32, 256)

    assert isinstance(hidden, int)


def test_declaring_an_integer_as_vectorisable_is_rejected_at_the_declaration() -> None:
    """An int sizes tensors, so it cannot vary inside a group.

    Rejecting it here matters: handing back a tensor instead would break the
    obvious use, `nn.Linear(8, hidden)`, with a PyTorch error that says nothing
    about the declaration being wrong.
    """
    study = _study()
    proxy = VectorizedTrial(_group(study, 3), preserving=frozenset({"hidden"}))

    with pytest.raises(ShapeChangingHyperparameterError, match="hidden"):
        proxy.suggest_int("hidden", 32, 256)


def test_every_member_records_the_shared_parameter() -> None:
    """Optuna requires each trial to declare its own params, even shared ones."""
    study = _study()
    trials = _group(study, 3)
    proxy = VectorizedTrial(trials, preserving=frozenset())

    proxy.suggest_int("hidden", 32, 256)

    assert all("hidden" in t.params for t in trials)


def test_k_reports_the_group_size() -> None:
    study = _study()
    proxy = VectorizedTrial(_group(study, 5), preserving=frozenset())
    assert proxy.k == 5


def test_categorical_is_never_vectorised_even_if_declared() -> None:
    """Categorical values are not numbers, so they cannot become a tensor."""
    study = _study()
    proxy = VectorizedTrial(_group(study, 3), preserving=frozenset({"kind"}))

    kind = proxy.suggest_categorical("kind", ["a", "b"])

    assert kind in ("a", "b")


def test_the_tensor_lands_on_the_requested_device() -> None:
    study = _study()
    proxy = VectorizedTrial(_group(study, 3), preserving=frozenset({"lr"}), device="cpu")

    lr = proxy.suggest_float("lr", 0.1, 0.2)

    assert lr.device.type == "cpu"


def test_the_probe_records_every_declared_name() -> None:
    study = _study()
    probe = ProbeTrial(study.ask())

    probe.suggest_int("hidden", 32, 256)
    probe.suggest_float("lr", 1e-4, 1e-1)

    assert probe.seen == ["hidden", "lr"]


def test_the_probe_stops_without_running_the_training_body() -> None:
    """The probe must learn the search space without paying for a training run."""
    study = _study()
    probe = ProbeTrial(study.ask())
    probe.suggest_int("hidden", 32, 256)

    with pytest.raises(ProbeComplete):
        probe.stop_probe()
