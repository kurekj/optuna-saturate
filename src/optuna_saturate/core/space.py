"""Classification of a hyperparameter search space into vectorisable and non-vectorisable parts.

A hyperparameter is *shape-preserving* when changing its value leaves every tensor
shape in the model untouched — learning rate, dropout probability, softmax
temperature, loss weights. Configurations that differ only in shape-preserving
hyperparameters can be trained as one batched model.

A hyperparameter is *shape-changing* when it alters the architecture — embedding
size, hidden width, layer count, batch size. Such configurations need separate
model instances and therefore separate trials.

The distinction cannot be inferred automatically, so the user declares the
shape-preserving names and this module validates the declaration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from optuna_saturate.exceptions import UnknownHyperparameterError


@dataclass(frozen=True)
class SpaceClassification:
    """Names of a search space split by whether they preserve tensor shapes.

    Both tuples are sorted alphabetically so that planning is deterministic.
    """

    preserving: tuple[str, ...]
    changing: tuple[str, ...]


def classify(
    param_names: Iterable[str],
    declared_preserving: Iterable[str],
) -> SpaceClassification:
    """Split ``param_names`` according to ``declared_preserving``.

    Args:
        param_names: Every hyperparameter name in the search space.
        declared_preserving: Names the user declared as shape-preserving.

    Returns:
        The classification, with both tuples sorted alphabetically.

    Raises:
        UnknownHyperparameterError: A declared name is absent from ``param_names``.
            Silently ignoring it would let a typo disable vectorisation without
            any visible symptom.
    """
    all_names = set(param_names)
    declared = set(declared_preserving)

    unknown = declared - all_names
    if unknown:
        known = ", ".join(sorted(all_names))
        raise UnknownHyperparameterError(
            f"declared as vectorisable but not present in the search space: "
            f"{', '.join(sorted(unknown))}. Known names: {known}"
        )

    return SpaceClassification(
        preserving=tuple(sorted(declared)),
        changing=tuple(sorted(all_names - declared)),
    )
