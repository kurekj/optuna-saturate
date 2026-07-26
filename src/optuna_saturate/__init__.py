"""Saturate a single GPU during Optuna hyperparameter optimisation."""

from optuna_saturate.api import saturate, vectorizable
from optuna_saturate.core.space import SpaceClassification, classify
from optuna_saturate.exceptions import (
    MissingDependencyError,
    OptunaSaturateError,
    ShapeChangingHyperparameterError,
    UnknownHyperparameterError,
)

__version__ = "0.1.0"

__all__ = [
    "MissingDependencyError",
    "OptunaSaturateError",
    "ShapeChangingHyperparameterError",
    "SpaceClassification",
    "UnknownHyperparameterError",
    "__version__",
    "classify",
    "saturate",
    "vectorizable",
]
