"""Exception types raised by optuna-saturate."""


class OptunaSaturateError(Exception):
    """Base class for every error raised by this library."""


class UnknownHyperparameterError(OptunaSaturateError, ValueError):
    """A hyperparameter was declared vectorisable but is absent from the search space."""


class MissingDependencyError(OptunaSaturateError, ImportError):
    """An optional dependency is required for this feature but is not installed."""


class ShapeChangingHyperparameterError(OptunaSaturateError, ValueError):
    """A hyperparameter declared vectorisable turned out to change tensor shapes."""


class StopSaturate(OptunaSaturateError):  # noqa: N818
    """Raise from a callback to end ``saturate()`` after the current group.

    ``study.stop()`` works only inside ``study.optimize()``; this is its
    counterpart for ``saturate()``. Every member of the current group has
    already been recorded when the callbacks run, so nothing is lost.
    """
