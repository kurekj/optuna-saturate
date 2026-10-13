"""Choosing a group size from what the device can hold.

Holding K models at once costs memory that grows linearly in K: a fixed part
(the CUDA context, the data, the allocator's slack) and a per-member part (the
member's parameters, gradients, optimiser state and activations). A line fitted
to measured peaks gives how many members fit in a given budget. The speedup
flattens once the device stops idling between kernels, so the rule is to grow K
until either the memory budget or a speedup plateau is reached, whichever comes
first. The fit excludes the CUDA context and the allocator's reserved blocks, so
a margin is kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryModel:
    """Peak memory as a straight line in the group size.

    Attributes:
        fixed_bytes: Memory used whatever the group size.
        per_member_bytes: Additional memory per stacked member.
    """

    fixed_bytes: float
    per_member_bytes: float

    def peak_bytes(self, group_size: int) -> float:
        """Predicted peak memory for ``group_size`` members."""
        return self.fixed_bytes + self.per_member_bytes * group_size

    def largest_group_within(self, budget_bytes: float, cap: int = 256) -> int:
        """The largest group whose predicted peak stays inside ``budget_bytes``.

        Returns:
            At most ``cap``; zero when not even one member fits. A fitted slope
            of zero or below, which measurements this close to the allocator's
            granularity can produce, is treated as a constant cost: the whole
            cap fits if that constant does, and nothing fits otherwise.
        """
        if self.per_member_bytes <= 0:
            return cap if self.fixed_bytes <= budget_bytes else 0
        fits = int((budget_bytes - self.fixed_bytes) // self.per_member_bytes)
        return max(0, min(cap, fits))


def fit_memory_model(group_sizes: Sequence[int], peak_bytes: Sequence[float]) -> MemoryModel:
    """Fit the line through measured peaks by ordinary least squares.

    Args:
        group_sizes: The group sizes that were measured, at least two distinct.
        peak_bytes: The peak memory observed at each.

    Raises:
        ValueError: Fewer than two distinct group sizes, or mismatched lengths.
    """
    if len(group_sizes) != len(peak_bytes):
        raise ValueError("group_sizes and peak_bytes must have the same length")
    if len(set(group_sizes)) < 2:
        raise ValueError("fitting a line needs at least two distinct group sizes")

    n = len(group_sizes)
    mean_k = sum(group_sizes) / n
    mean_m = sum(peak_bytes) / n
    pairs = zip(group_sizes, peak_bytes, strict=True)
    covariance = sum((k - mean_k) * (m - mean_m) for k, m in pairs)
    variance = sum((k - mean_k) ** 2 for k in group_sizes)
    slope = covariance / variance
    intercept = mean_m - slope * mean_k
    return MemoryModel(fixed_bytes=intercept, per_member_bytes=slope)


def recommend_group_size(
    group_sizes: Sequence[int],
    peak_bytes: Sequence[float],
    device_bytes: float,
    headroom: float = 0.8,
    speedups: Sequence[float] | None = None,
    plateau: float = 0.05,
    cap: int = 256,
) -> int:
    """Pick a group size from measured peaks, and optionally measured speedups.

    Args:
        group_sizes: Group sizes that were measured.
        peak_bytes: Peak memory observed at each, as PyTorch's own allocation
            (``peak_vram_torch_bytes`` from the profiler) rather than NVML's
            figure, which includes the context and other processes.
        device_bytes: Total memory of the device.
        headroom: Fraction of ``device_bytes`` the group may use. The rest is
            left for the CUDA context, the allocator's fragmentation and other
            processes.
        speedups: Speedup over independent training at each group size, if
            measured. When given, the recommendation stops growing once the
            next measured step improves the speedup by less than ``plateau``
            of its value, since further members then cost memory for nothing.
            When no plateau is reached within the measured sizes, the
            recommendation does not extrapolate beyond the largest size
            measured: the speedup curve is known only that far.
        plateau: Relative improvement below which the speedup counts as flat.
        cap: Upper bound on the recommendation.

    Returns:
        The largest group size that fits the memory budget and, when speedups
        were given, lies before the plateau.

    Raises:
        ValueError: Not even one member fits the budget.
    """
    model = fit_memory_model(group_sizes, peak_bytes)
    choice = model.largest_group_within(device_bytes * headroom, cap=cap)
    if choice < 1:
        raise ValueError(
            f"no group fits: the fitted fixed cost of {model.fixed_bytes:.0f} bytes "
            f"already exceeds the budget of {device_bytes * headroom:.0f} bytes"
        )

    if speedups is not None:
        if len(speedups) != len(group_sizes):
            raise ValueError("speedups must have one entry per group size")
        ordered = sorted(zip(group_sizes, speedups, strict=True))
        flat_from = ordered[-1][0]
        for (k_small, s_small), (_, s_large) in zip(ordered, ordered[1:], strict=False):
            if s_small > 0 and (s_large - s_small) / s_small < plateau:
                flat_from = k_small
                break
        choice = min(choice, flat_from)

    return choice
