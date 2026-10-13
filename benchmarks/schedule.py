"""A fixed schedule of configurations, so that every arm evaluates the same ones.

An adaptive sampler proposes different configurations under different execution
strategies: workers running in parallel see a different history, and a group
shares its architecture. That is part of what a strategy is, and the main
comparison keeps it. For a comparison of execution alone, this module replaces
the sampler with a schedule drawn once per seed: ``trials`` configurations in
blocks of ``block`` that share the shape-changing values, which is the shape a
vectorised group needs, handed out by trial number so that every arm,
sequential or parallel, threaded or batched, trains exactly the same list.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution


def _draw(rng: np.random.RandomState, distribution: BaseDistribution) -> Any:
    if isinstance(distribution, FloatDistribution):
        if distribution.log:
            return float(np.exp(rng.uniform(np.log(distribution.low), np.log(distribution.high))))
        return float(rng.uniform(distribution.low, distribution.high))
    if isinstance(distribution, IntDistribution):
        if distribution.log:
            value = int(
                round(np.exp(rng.uniform(np.log(distribution.low), np.log(distribution.high))))
            )
            return min(max(value, distribution.low), distribution.high)
        choices = range(distribution.low, distribution.high + 1, distribution.step)
        return int(rng.choice(list(choices)))
    raise TypeError(f"no draw for {type(distribution).__name__}")


def make_schedule(
    distributions: dict[str, BaseDistribution],
    shared: tuple[str, ...],
    trials: int,
    block: int,
    seed: int,
) -> list[dict[str, Any]]:
    """``trials`` configurations in blocks of ``block`` sharing the ``shared`` names."""
    rng = np.random.RandomState(seed)
    schedule: list[dict[str, Any]] = []
    for _ in range(math.ceil(trials / block)):
        fixed = {name: _draw(rng, distributions[name]) for name in shared}
        for _ in range(block):
            if len(schedule) == trials:
                break
            entry = dict(fixed)
            for name, distribution in distributions.items():
                if name not in shared:
                    entry[name] = _draw(rng, distribution)
            schedule.append(entry)
    return schedule


class ScheduleSampler(optuna.samplers.BaseSampler):
    """An Optuna sampler that replays a schedule by trial number.

    Fixed values on an enqueued trial bypass the sampler, which is how a
    vectorised group's followers receive their shared values; the schedule is
    built in blocks so that those values agree with it.
    """

    def __init__(self, schedule: list[dict[str, Any]]) -> None:
        self._schedule = schedule

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if trial.number >= len(self._schedule):
            raise IndexError(
                f"trial {trial.number} is beyond the schedule of {len(self._schedule)}"
            )
        return self._schedule[trial.number][param_name]
