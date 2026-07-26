from __future__ import annotations

from typing import Any

import optuna
import pytest

import optuna_saturate as osat
from optuna_saturate.exceptions import (
    ShapeChangingHyperparameterError,
    UnknownHyperparameterError,
)
from tests.objectives import make_bad_over_objective, make_objective


def _study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize")


def test_every_member_of_a_group_shares_the_shape_changing_value() -> None:
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    first_group = study.trials[:4]
    assert len({t.params["hidden"] for t in first_group}) == 1


def test_members_of_a_group_differ_in_the_vectorised_value() -> None:
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    first_group = study.trials[:4]
    assert len({t.params["lr"] for t in first_group}) > 1


def test_the_objective_runs_once_per_group_not_once_per_trial() -> None:
    """This is the whole point: one training pass covers the group."""
    calls: list[int] = []
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective(calls))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    assert calls == [4, 4]


def test_all_trials_land_in_the_study_as_complete() -> None:
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_each_trial_receives_its_own_score() -> None:
    study = _study()

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        return [float(i) for i in range(ctx.k)]

    osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=4, group_size=4)

    assert [t.value for t in study.trials] == [0.0, 1.0, 2.0, 3.0]
    assert study.best_value == 3.0


def test_a_final_partial_group_is_still_run() -> None:
    calls: list[int] = []
    study = _study()
    objective = osat.vectorizable(over=["lr"])(make_objective(calls))

    osat.saturate(study, objective, n_trials=10, group_size=4)

    assert calls == [4, 4, 2]
    assert len(study.trials) == 10


def test_a_typo_in_over_is_rejected_before_any_training_runs() -> None:
    calls: list[int] = []
    study = _study()
    objective = osat.vectorizable(over=["learnig_rate"])(make_objective(calls))

    with pytest.raises(UnknownHyperparameterError, match="learnig_rate"):
        osat.saturate(study, objective, n_trials=4, group_size=2)

    assert calls == []  # the probe stopped before any model was built


def test_declaring_a_shape_changing_hyperparameter_fails_loudly() -> None:
    """Critical test 3 of the project spec, end to end through the public API."""
    study = _study()
    objective = osat.vectorizable(over=["hidden"])(make_bad_over_objective())

    with pytest.raises(ShapeChangingHyperparameterError, match="hidden"):
        osat.saturate(study, objective, n_trials=4, group_size=2)


def test_an_undecorated_objective_is_rejected_with_a_useful_message() -> None:
    study = _study()

    with pytest.raises(ValueError, match="vectorizable"):
        osat.saturate(study, lambda trial, ctx: [0.0], n_trials=2, group_size=2)


def test_returning_the_wrong_number_of_scores_is_an_error() -> None:
    study = _study()

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        return [0.0, 1.0]  # only two, whatever the group size

    with pytest.raises(ValueError, match="3"):
        osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=3, group_size=3)


def test_a_failing_objective_marks_the_group_failed_and_leaves_the_study_usable() -> None:
    study = _study()

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        raise RuntimeError("training blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=2, group_size=2)

    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_the_public_surface_is_importable_from_the_package_root() -> None:
    """A reviewer installs the package and reads `dir(optuna_saturate)`."""
    for name in ("saturate", "vectorizable", "__version__"):
        assert hasattr(osat, name), name
