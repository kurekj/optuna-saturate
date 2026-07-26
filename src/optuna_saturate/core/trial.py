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
    """

    def __init__(
        self,
        trials: Sequence[optuna.Trial],
        preserving: frozenset[str],
        device: str | None = None,
    ) -> None:
        if len(trials) == 0:
            raise ValueError("a vectorised group needs at least one trial")
        self._trials = list(trials)
        self._preserving = preserving
        self._device = device
        self.k = len(self._trials)

    def _as_tensor(self, values: Sequence[float]) -> Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> Tensor | float:
        # Every trial must record the parameter, otherwise Optuna stores an
        # incomplete trial and the sampler cannot learn from it.
        values = [t.suggest_float(name, low, high, **kwargs) for t in self._trials]
        if name in self._preserving:
            return self._as_tensor(values)
        return values[0]

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        """Integers are never vectorised. See :func:`_reject_vectorised_int`."""
        if name in self._preserving:
            _reject_vectorised_int(name)
        values = [t.suggest_int(name, low, high, **kwargs) for t in self._trials]
        return values[0]

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        # Categorical values need not be numeric, so they never become a tensor
        # even when declared vectorisable.
        values = [t.suggest_categorical(name, choices) for t in self._trials]
        return values[0]
