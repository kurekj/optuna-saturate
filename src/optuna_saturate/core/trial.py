"""Trial proxies used while a group of configurations runs as one.

A vectorised group is K real Optuna trials that share an architecture. The user
writes one objective, so ``suggest_*`` has to answer for all K at once: a tensor
of K values for a hyperparameter declared vectorisable, a plain scalar for one
that is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

import optuna
import torch
from torch import Tensor

from optuna_saturate.exceptions import ShapeChangingHyperparameterError


# N818 wants an "Error" suffix, but this signals success, not failure: the probe
# has everything it came for. Naming it an error would misdescribe it.
class ProbeComplete(Exception):  # noqa: N818
    """Raised to end the probe once the search space has been declared.

    Not an error: the probe deliberately abandons the objective after the
    ``suggest_*`` calls so that discovering the space costs no training.
    """


def _reject_vectorised_int(name: str) -> NoReturn:
    """Refuse to hand back an integer hyperparameter as a per-member vector.

    An integer in a search space almost always sizes a tensor -- a width, a
    depth, a number of heads -- so configurations differing in one cannot share a
    batched model. Returning a tensor instead would break the obvious use,
    ``nn.Linear(8, hidden)``, deep inside PyTorch with an error that says nothing
    about the real mistake.
    """
    raise ShapeChangingHyperparameterError(
        f"{name!r} was declared vectorisable, but it is an integer hyperparameter. "
        "Integers size tensors, and members of a group must share every tensor "
        "shape. Remove it from `over` so that it varies between groups instead of "
        "within one."
    )


def _reject_pruning() -> NoReturn:
    """Pruning is not available inside a vectorised group.

    A pruner stops one trial part-way through its training. Members of a group
    train as one batched model, so stopping one member early would mean either
    stopping the whole group or carrying a dead member to the end; neither is
    what a pruner promises. The methods exist so that an objective written for
    ``study.optimize`` fails with an explanation instead of an AttributeError.
    """
    raise NotImplementedError(
        "trial.report() and trial.should_prune() are not supported inside a "
        "vectorised group: members train together and cannot be stopped one at a "
        "time. Run the objective through study.optimize() if it needs a pruner."
    )


class ProbeTrial:
    """Runs an objective far enough to learn which hyperparameters exist.

    Answers ``suggest_*`` exactly as :class:`VectorizedTrial` will during the real
    run -- a tensor for a declared name, a scalar otherwise -- so that objective
    code between the declarations and ``ctx.stack`` behaves identically in both
    passes. Getting a float during the probe and a tensor afterwards would make
    any use of the value in between fail only on the first pass.

    Args:
        trial: The real trial whose parameters are being declared.
        declared: Names the user passed to ``over``. The classification is not
            known yet -- discovering it is what this probe is for -- so the raw
            declaration is used.
        k: Group size, so the probe's tensors have the same length as the real
            run's.
        device: Where the returned tensors are placed.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        declared: frozenset[str] = frozenset(),
        k: int = 1,
        device: str | None = None,
    ) -> None:
        self._trial = trial
        self._declared = declared
        self._k = k
        self._device = device
        self.seen: list[str] = []

    @property
    def number(self) -> int:
        return self._trial.number

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> Tensor | float:
        self.seen.append(name)
        value = self._trial.suggest_float(name, low, high, **kwargs)
        if name in self._declared:
            return torch.full((self._k,), value, dtype=torch.float32, device=self._device)
        return value

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        self.seen.append(name)
        if name in self._declared:
            _reject_vectorised_int(name)
        return self._trial.suggest_int(name, low, high, **kwargs)

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        self.seen.append(name)
        return self._trial.suggest_categorical(name, choices)

    def set_user_attr(self, key: str, value: Any) -> None:
        self._trial.set_user_attr(key, value)

    def set_member_attr(self, key: str, values: Sequence[Any]) -> None:
        # The probe stands in for the group's first member only.
        self._trial.set_user_attr(key, values[0])

    def report(self, value: float, step: int) -> NoReturn:
        _reject_pruning()

    def should_prune(self) -> NoReturn:
        _reject_pruning()

    def stop_probe(self) -> NoReturn:
        """Abandon the objective now that every hyperparameter has been seen."""
        raise ProbeComplete


class VectorizedTrial:
    """One objective call standing in for K real trials.

    Args:
        trials: The K trials of the group, all sharing one architecture.
        preserving: Names declared vectorisable. These return a tensor of K
            values; every other name returns the scalar the group shares.
        device: Where the returned tensors are placed. ``None`` leaves them on
            the default device.

    Attributes:
        k: Group size.
        trials: The underlying Optuna trials, in member order, for callers that
            need the per-trial handle (its number, its user attributes).
    """

    def __init__(
        self,
        trials: Sequence[optuna.Trial],
        preserving: frozenset[str],
        device: str | None = None,
    ) -> None:
        if len(trials) == 0:
            raise ValueError("a vectorised group needs at least one trial")
        self.trials = list(trials)
        self._preserving = preserving
        self._device = device
        self.k = len(self.trials)

    @property
    def number(self) -> int:
        """Number of the group's first trial; see ``trials`` for all of them."""
        return self.trials[0].number

    @property
    def numbers(self) -> list[int]:
        """Trial numbers of every member, in member order."""
        return [trial.number for trial in self.trials]

    def _as_tensor(self, values: Sequence[float]) -> Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> Tensor | float:
        # Every trial must record the parameter, otherwise Optuna stores an
        # incomplete trial and the sampler cannot learn from it.
        values = [t.suggest_float(name, low, high, **kwargs) for t in self.trials]
        if name in self._preserving:
            return self._as_tensor(values)
        return values[0]

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        """Integers are never vectorised. See :func:`_reject_vectorised_int`."""
        if name in self._preserving:
            _reject_vectorised_int(name)
        values = [t.suggest_int(name, low, high, **kwargs) for t in self.trials]
        return values[0]

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        # Categorical values need not be numeric, so they never become a tensor
        # even when declared vectorisable.
        values = [t.suggest_categorical(name, choices) for t in self.trials]
        return values[0]

    def set_user_attr(self, key: str, value: Any) -> None:
        """Record ``value`` on every member of the group, unchanged."""
        for trial in self.trials:
            trial.set_user_attr(key, value)

    def set_member_attr(self, key: str, values: Sequence[Any]) -> None:
        """Record one value per member: member ``i`` receives ``values[i]``."""
        if len(values) != self.k:
            raise ValueError(f"expected {self.k} values, one per member, got {len(values)}")
        for trial, value in zip(self.trials, values, strict=True):
            trial.set_user_attr(key, value)

    def report(self, value: float, step: int) -> NoReturn:
        _reject_pruning()

    def should_prune(self) -> NoReturn:
        _reject_pruning()
