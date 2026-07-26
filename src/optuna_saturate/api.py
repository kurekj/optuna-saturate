"""The public entry points: :func:`saturate` and :func:`vectorizable`.

``saturate`` replaces ``study.optimize`` for studies whose search space contains
hyperparameters that do not change tensor shapes. Those configurations train as
one batched model, so a group of K costs roughly one training run rather than K.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import optuna

from optuna_saturate.core.context import VectorizedContext
from optuna_saturate.core.space import SpaceClassification, classify
from optuna_saturate.core.trial import ProbeComplete, ProbeTrial, VectorizedTrial

_OVER_ATTRIBUTE = "_osat_over"


def vectorizable(over: Sequence[str]) -> Callable[[Any], Any]:
    """Declare which hyperparameters of an objective preserve tensor shapes.

    The declaration cannot be inferred: whether a hyperparameter changes the
    architecture depends on how the objective uses it. It is checked at runtime,
    and a wrong declaration raises rather than quietly producing wrong numbers.

    Args:
        over: Names that leave every tensor shape untouched, such as a learning
            rate, a dropout probability or a softmax temperature.

    Returns:
        The objective, tagged for :func:`saturate`.
    """

    def decorate(objective: Any) -> Any:
        setattr(objective, _OVER_ATTRIBUTE, tuple(over))
        return objective

    return decorate


class _ProbeContext:
    """A context that ends the objective the moment it asks for a model."""

    k = 1

    def __init__(self, probe: ProbeTrial) -> None:
        self._probe = probe

    def stack(self, models: Any) -> Any:
        self._probe.stop_probe()

    def __getattr__(self, name: str) -> Any:
        # Reaching for anything else also means the declarations are done.
        def stop(*args: Any, **kwargs: Any) -> Any:
            self._probe.stop_probe()

        return stop


def _declare_only(
    objective: Any,
    trial: optuna.Trial,
    declared: frozenset[str],
    k: int,
    device: str | None,
) -> tuple[str, ...]:
    """Run the objective just far enough to populate ``trial.params``.

    There is a cycle to break. A freshly asked trial has empty ``params``: Optuna
    fills them as ``suggest_*`` is called. So the group's architecture is unknown
    until the objective has run, but the objective cannot run until the group
    exists. The probe cuts it by stopping at ``ctx.stack`` -- after every
    declaration, before any training.

    Re-suggesting a name on the same trial returns Optuna's cached value, so the
    probed trial then serves as an ordinary member of its own group and nothing
    is wasted.

    Returns:
        The hyperparameter names the objective declared, in order.
    """
    probe = ProbeTrial(trial, declared=declared, k=k, device=device)
    # contextlib.suppress rather than try/except/pass: ruff SIM105.
    with contextlib.suppress(ProbeComplete):
        objective(probe, _ProbeContext(probe))
    return tuple(probe.seen)


def saturate(
    study: optuna.Study,
    objective: Any,
    n_trials: int,
    group_size: int = 8,
    device: str | None = None,
) -> None:
    """Run ``n_trials`` of ``study``, training vectorisable groups as one model.

    Args:
        study: The study to fill. Trials are recorded exactly as
            ``study.optimize`` would record them.
        objective: A callable ``(trial, ctx)`` decorated with
            :func:`vectorizable`, returning one score per group member.
        n_trials: Total number of trials to run. The final group may be smaller.
        group_size: How many configurations to train together.
        device: Device for models and tensors, for example ``"cuda:0"``.

    Raises:
        ValueError: ``objective`` was not decorated with :func:`vectorizable`,
            or returned a number of scores other than the group size.
        UnknownHyperparameterError: ``over`` names something outside the space.
        ShapeChangingHyperparameterError: A declared name changes the architecture.
    """
    declared = getattr(objective, _OVER_ATTRIBUTE, None)
    if declared is None:
        raise ValueError(
            "saturate() needs an objective declared with @vectorizable(over=[...]). "
            "Without the declaration there is nothing to vectorise over."
        )

    classification: SpaceClassification | None = None
    remaining = n_trials

    while remaining > 0:
        k = min(group_size, remaining)
        leader = study.ask()
        try:
            discovered = _declare_only(objective, leader, frozenset(declared), k, device)
            if classification is None:
                # Raises on a typo, before anything has been trained.
                classification = classify(param_names=discovered, declared_preserving=declared)
        except BaseException:
            study.tell(leader, state=optuna.trial.TrialState.FAIL)
            raise

        _run_group(
            study,
            objective,
            leader,
            k,
            frozenset(classification.preserving),
            declared,
            device,
        )
        remaining -= k


def _fail_all(study: optuna.Study, trials: Sequence[optuna.Trial]) -> None:
    for trial in trials:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


def _run_group(
    study: optuna.Study,
    objective: Any,
    leader: optuna.Trial,
    k: int,
    preserving: frozenset[str],
    declared: tuple[str, ...],
    device: str | None,
) -> None:
    """Grow ``leader`` into a group of ``k`` trials sharing its architecture."""
    # The leader's shape-changing values fix the architecture; the sampler stays
    # free to pick the vectorised ones for every follower.
    shared = {name: value for name, value in leader.params.items() if name not in preserving}
    for _ in range(k - 1):
        study.enqueue_trial(shared, skip_if_exists=False)
    trials = [leader] + [study.ask() for _ in range(k - 1)]

    proxy = VectorizedTrial(trials, preserving=preserving, device=device)
    context = VectorizedContext(k=k, declared=declared, device=device)

    try:
        scores = objective(proxy, context)
    except BaseException:
        _fail_all(study, trials)
        raise

    if len(scores) != k:
        _fail_all(study, trials)
        raise ValueError(f"objective returned {len(scores)} scores but the group holds {k}")

    for trial, score in zip(trials, scores, strict=True):
        study.tell(trial, float(score))
