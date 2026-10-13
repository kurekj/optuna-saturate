"""Saturate a single GPU during Optuna hyperparameter optimisation."""

from optuna_saturate.api import saturate, vectorizable
from optuna_saturate.core.plan import MemoryModel, fit_memory_model, recommend_group_size
from optuna_saturate.core.space import SpaceClassification, classify
from optuna_saturate.exceptions import (
    MissingDependencyError,
    OptunaSaturateError,
    ShapeChangingHyperparameterError,
    StopSaturate,
    UnknownHyperparameterError,
)

__version__ = "0.2.0"

__all__ = [
    "MemoryModel",
    "MissingDependencyError",
    "OptunaSaturateError",
    "ShapeChangingHyperparameterError",
    "SpaceClassification",
    "StopSaturate",
    "UnknownHyperparameterError",
    "__version__",
    "classify",
    "fit_memory_model",
    "recommend_group_size",
    "saturate",
    "vectorizable",
]
