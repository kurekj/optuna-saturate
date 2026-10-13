"""What of the surrounding Optuna machinery keeps working through saturate().

Samplers, storages, callbacks, timeout, user attributes and the result
accessors are exercised on a study filled by saturate(). Pruning calls,
GridSampler and multi-objective studies raise the documented errors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import optuna
import pytest

import optuna_saturate as osat
from tests.objectives import Net, make_objective


def _quiet() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize(
    "sampler",
    [
        optuna.samplers.TPESampler(seed=0),
        optuna.samplers.RandomSampler(seed=0),
        optuna.samplers.TPESampler(seed=0, multivariate=True),
    ],
    ids=["tpe", "random", "tpe-multivariate"],
)
def test_any_ask_and_tell_sampler_drives_the_vectorised_values(sampler: Any) -> None:
    _quiet()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len({t.params["lr"] for t in study.trials[:4]}) == 4


def test_grid_sampler_is_rejected_with_an_explanation() -> None:
    """GridSampler refuses partially specified trials, which is how a group is grown."""
    _quiet()
    grid = {"hidden": [8, 16], "lr": [0.01, 0.1]}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(grid))
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    with pytest.raises(ValueError, match="GridSampler"):
        osat.saturate(study, objective, n_trials=4, group_size=2)
    assert study.trials == []


def test_the_gaussian_process_sampler_drives_the_vectorised_values() -> None:
    pytest.importorskip("scipy")  # GPSampler needs it
    _quiet()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.GPSampler(seed=0, n_startup_trials=2)
    )
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=6, group_size=2)

    assert len(study.trials) == 6
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_the_cma_es_sampler_drives_the_vectorised_values() -> None:
    pytest.importorskip("cmaes")  # CmaEsSampler needs it
    _quiet()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=0))
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=6, group_size=2)

    assert len(study.trials) == 6


def test_a_relational_storage_records_every_member(tmp_path: Path) -> None:
    _quiet()
    url = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    study = optuna.create_study(direction="maximize", storage=url, study_name="s")
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=6, group_size=3)

    reloaded = optuna.load_study(study_name="s", storage=url)
    assert len(reloaded.trials) == 6
    assert reloaded.best_trial.number == study.best_trial.number


def test_a_journal_storage_records_every_member(tmp_path: Path) -> None:
    _quiet()
    path = str(tmp_path / "journal.log")
    # The default lock is a symlink, which an unprivileged Windows account cannot create.
    lock = optuna.storages.journal.JournalFileOpenLock(path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(path, lock_obj=lock)
    )
    study = optuna.create_study(direction="maximize", storage=storage)
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=4, group_size=2)

    assert len(study.trials) == 4


def test_callbacks_fire_once_per_member_in_member_order() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    objective = osat.vectorizable(over=["lr"])(make_objective([]))
    seen: list[int] = []

    def record(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        seen.append(trial.number)

    osat.saturate(study, objective, n_trials=6, group_size=3, callbacks=[record])

    assert seen == [0, 1, 2, 3, 4, 5]


def test_timeout_stops_new_groups_from_starting() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=100, group_size=4, timeout=0.0)

    # The timeout is checked before each group, so a zero timeout runs nothing.
    assert len(study.trials) == 0


def test_user_attributes_can_be_set_for_the_group_or_per_member() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        trial.set_user_attr("splits", ["train", "validation"])  # a list, stored unchanged
        trial.set_member_attr("rank", list(range(ctx.k)))
        return [0.0] * ctx.k

    osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=3, group_size=3)

    assert all(t.user_attrs["splits"] == ["train", "validation"] for t in study.trials)
    assert [t.user_attrs["rank"] for t in study.trials] == [0, 1, 2]


def test_a_member_attribute_list_of_the_wrong_length_is_rejected() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        trial.set_member_attr("rank", [0, 1])
        return [0.0] * ctx.k

    with pytest.raises(ValueError, match="one per member"):
        osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=3, group_size=3)


def test_a_callback_can_stop_the_run_after_the_current_group() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    def stop_after_first_group(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number >= 1:
            raise osat.StopSaturate

    osat.saturate(study, objective, n_trials=12, group_size=4, callbacks=[stop_after_first_group])

    # The whole first group is recorded, and no second group was started.
    assert len(study.trials) == 4
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_a_failing_callback_leaves_every_member_of_the_group_recorded() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    def explode(_: optuna.Study, __: optuna.trial.FrozenTrial) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        osat.saturate(study, objective, n_trials=4, group_size=4, callbacks=[explode])

    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len(study.trials) == 4


def test_a_positive_timeout_stops_once_it_has_elapsed() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    seen: list[int] = []

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        ctx.stack([Net() for _ in range(ctx.k)])  # ends the probe, as a real objective does
        time.sleep(0.05)
        seen.append(ctx.k)
        return [0.0] * ctx.k

    osat.saturate(
        study, osat.vectorizable(over=["lr"])(objective), n_trials=1000, group_size=2, timeout=0.2
    )

    # A few groups fit in the budget; nowhere near the trial cap.
    assert 1 <= len(seen) < 50
    assert len(study.trials) == 2 * len(seen)


def test_trial_numbers_are_exposed_for_every_member() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize")
    seen: list[list[int]] = []

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        if hasattr(trial, "numbers"):
            seen.append(trial.numbers)
        return [0.0] * ctx.k

    osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=4, group_size=2)

    assert seen == [[0, 1], [2, 3]]


def test_pruning_calls_fail_with_an_explanation() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    def objective(trial: Any, ctx: Any) -> list[float]:
        trial.suggest_float("lr", 0.1, 0.2)
        trial.report(0.5, step=0)
        return [0.0] * ctx.k

    with pytest.raises(NotImplementedError, match="prune"):
        osat.saturate(study, osat.vectorizable(over=["lr"])(objective), n_trials=2, group_size=2)
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_multi_objective_studies_are_rejected_up_front() -> None:
    _quiet()
    study = optuna.create_study(directions=["maximize", "minimize"])
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    with pytest.raises(ValueError, match="single-objective"):
        osat.saturate(study, objective, n_trials=2, group_size=2)


def test_the_result_accessors_read_the_filled_study() -> None:
    _quiet()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    objective = osat.vectorizable(over=["lr"])(make_objective([]))

    osat.saturate(study, objective, n_trials=8, group_size=4)

    pytest.importorskip("pandas")  # trials_dataframe() needs it; CI installs the [dev] extra
    frame = study.trials_dataframe()
    assert len(frame) == 8
    assert {"params_hidden", "params_lr", "value"} <= set(frame.columns)
    assert study.best_params.keys() == {"hidden", "lr"}
