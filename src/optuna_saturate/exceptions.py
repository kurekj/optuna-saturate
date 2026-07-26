"""Exception types raised by optuna-saturate."""


class OptunaSaturateError(Exception):
    """Base class for every error raised by this library."""


class UnknownHyperparameterError(OptunaSaturateError, ValueError):
    """A hyperparameter was declared vectorisable but is absent from the search space."""


class MissingDependencyError(OptunaSaturateError, ImportError):
    """An optional dependency is required for this feature but is not installed."""


class ShapeChangingHyperparameterError(OptunaSaturateError, ValueError):
    """A hyperparameter declared vectorisable turned out to change tensor shapes."""
