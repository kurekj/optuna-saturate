import pytest

from optuna_saturate.core.plan import MemoryModel, fit_memory_model, recommend_group_size

GIB = 1024**3


def test_fit_recovers_an_exact_line() -> None:
    model = fit_memory_model([2, 4, 8], [300 + 2 * 50, 300 + 4 * 50, 300 + 8 * 50])
    assert model.fixed_bytes == pytest.approx(300)
    assert model.per_member_bytes == pytest.approx(50)


def test_fit_needs_two_distinct_group_sizes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        fit_memory_model([4, 4], [100, 100])


def test_prediction_is_linear_in_the_group_size() -> None:
    model = MemoryModel(fixed_bytes=100, per_member_bytes=10)
    assert model.peak_bytes(16) == 260


def test_largest_group_stays_inside_the_budget() -> None:
    model = MemoryModel(fixed_bytes=100, per_member_bytes=10)
    assert model.largest_group_within(budget_bytes=305) == 20
    assert model.peak_bytes(20) <= 305 < model.peak_bytes(21)


def test_largest_group_is_zero_when_nothing_fits() -> None:
    model = MemoryModel(fixed_bytes=1000, per_member_bytes=10)
    assert model.largest_group_within(budget_bytes=5) == 0


def test_a_flat_fit_is_a_constant_cost_checked_against_the_budget() -> None:
    assert MemoryModel(fixed_bytes=1000, per_member_bytes=0).largest_group_within(80) == 0
    assert MemoryModel(fixed_bytes=50, per_member_bytes=0).largest_group_within(80, cap=16) == 16


def test_recommendation_raises_when_nothing_fits() -> None:
    with pytest.raises(ValueError, match="no group fits"):
        recommend_group_size([2, 4], [30 * GIB, 31 * GIB], device_bytes=24 * GIB)


def test_without_a_plateau_the_recommendation_stays_within_the_measured_sizes() -> None:
    sizes = [2, 4, 8]
    peaks = [1 * GIB + 0.01 * GIB * k for k in sizes]
    speedups = [0.9, 1.5, 2.5]  # still rising at the largest size measured
    k = recommend_group_size(sizes, peaks, device_bytes=80 * GIB, speedups=speedups)
    assert k == 8


def test_recommendation_is_bounded_by_memory_when_no_speedups_are_given() -> None:
    # 1 GiB fixed, 1 GiB per member, 24 GiB device at 80% headroom: 18 members fit.
    k = recommend_group_size([2, 4], [3 * GIB, 5 * GIB], device_bytes=24 * GIB)
    assert k == 18


def test_recommendation_stops_at_the_speedup_plateau() -> None:
    sizes = [2, 4, 8, 16, 32]
    peaks = [2 * GIB + 0.1 * GIB * k for k in sizes]
    speedups = [0.9, 1.5, 2.2, 3.0, 3.05]  # flat between 16 and 32
    k = recommend_group_size(sizes, peaks, device_bytes=80 * GIB, speedups=speedups)
    assert k == 16


def test_recommendation_takes_the_tighter_of_the_two_limits() -> None:
    sizes = [2, 4, 8, 16, 32]
    peaks = [2 * GIB + 1.0 * GIB * k for k in sizes]
    speedups = [0.9, 1.5, 2.2, 3.0, 3.05]
    # 12 GiB at 80% leaves 9.6 GiB: seven members fit, below the plateau at 16.
    k = recommend_group_size(sizes, peaks, device_bytes=12 * GIB, speedups=speedups)
    assert k == 7


def test_speedups_must_match_the_group_sizes() -> None:
    with pytest.raises(ValueError, match="speedups"):
        recommend_group_size([2, 4], [1.0, 2.0], device_bytes=10, speedups=[1.0])
