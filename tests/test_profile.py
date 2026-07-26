import dataclasses
import time

import pytest

from optuna_saturate.runtime.profile import GpuProfile, build_profile, profile_gpu
from tests.fakes import FakeNvmlBackend

MIB = 1024 * 1024


def test_busy_fraction_counts_samples_with_any_kernel_resident() -> None:
    profile = build_profile(
        utilizations=[0, 100, 100, 0],
        memories=[0, 0, 0, 0],
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=1.0,
    )
    assert profile.busy_fraction == 0.5


def test_busy_fraction_is_zero_when_no_sample_saw_a_kernel() -> None:
    profile = build_profile(
        utilizations=[0, 0, 0],
        memories=[0, 0, 0],
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=1.0,
    )
    assert profile.busy_fraction == 0.0


def test_peak_vram_is_the_maximum_not_the_final_reading() -> None:
    profile = build_profile(
        utilizations=[0, 0, 0],
        memories=[100 * MIB, 900 * MIB, 200 * MIB],
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=1.0,
    )
    assert profile.peak_vram_nvml_bytes == 900 * MIB


def test_energy_is_the_difference_between_counter_readings() -> None:
    """The NVML counter is cumulative since driver load, so only deltas mean anything."""
    profile = build_profile(
        utilizations=[0],
        memories=[0],
        energy_start_mj=5_000_000,
        energy_end_mj=5_002_500,
        wall_time_s=1.0,
    )
    assert profile.energy_j == pytest.approx(2.5)


def test_energy_is_absent_when_the_counter_is_unsupported() -> None:
    profile = build_profile(
        utilizations=[0],
        memories=[0],
        energy_start_mj=None,
        energy_end_mj=None,
        wall_time_s=1.0,
    )
    assert profile.energy_j is None


def test_effective_sample_rate_reports_what_was_achieved() -> None:
    """NVML queries cost tens of milliseconds, so the requested rate is not the real one."""
    profile = build_profile(
        utilizations=[0] * 48,
        memories=[0] * 48,
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=2.0,
    )
    assert profile.sample_count == 48
    assert profile.effective_sample_hz == pytest.approx(24.0)


def test_a_profile_with_no_samples_does_not_divide_by_zero() -> None:
    profile = build_profile(
        utilizations=[],
        memories=[],
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=0.5,
    )
    assert profile.busy_fraction == 0.0
    assert profile.effective_sample_hz == 0.0
    assert profile.peak_vram_nvml_bytes == 0


def test_the_profile_is_immutable() -> None:
    profile = build_profile(
        utilizations=[0],
        memories=[0],
        energy_start_mj=0,
        energy_end_mj=0,
        wall_time_s=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.busy_fraction = 1.0  # type: ignore[misc]


def test_the_context_manager_collects_samples_while_the_block_runs() -> None:
    backend = FakeNvmlBackend(utilizations=[100], memories=[42 * MIB], energies=[0, 1000])

    with profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=False) as prof:
        time.sleep(0.1)

    report = prof.report
    assert isinstance(report, GpuProfile)
    assert report.sample_count > 0
    assert report.busy_fraction == 1.0
    assert report.peak_vram_nvml_bytes == 42 * MIB
    assert report.wall_time_s >= 0.1


def test_an_exception_inside_the_block_still_yields_a_report() -> None:
    """A benchmark loop must be able to record what happened before a failure."""
    backend = FakeNvmlBackend(utilizations=[100], memories=[7 * MIB])

    with pytest.raises(RuntimeError, match="boom"), profile_gpu(
        backend=backend, interval_s=0.005, kernel_profiling=False
    ) as prof:
        time.sleep(0.05)
        raise RuntimeError("boom")

    # `prof` stays bound after the block, so the report survives the failure.
    assert prof.report.sample_count > 0
    assert prof.report.peak_vram_nvml_bytes == 7 * MIB


def test_the_sampler_thread_is_stopped_after_the_block() -> None:
    backend = FakeNvmlBackend()

    with profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=False) as prof:
        time.sleep(0.02)

    assert not prof.is_sampling()


def test_the_backend_is_closed_only_when_the_profiler_created_it() -> None:
    """An injected backend belongs to the caller and must outlive the block."""
    backend = FakeNvmlBackend()

    with profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=False):
        pass

    assert not backend.closed


def test_reading_the_report_before_the_block_finishes_is_an_error() -> None:
    backend = FakeNvmlBackend()

    with (
        profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=False) as prof,
        pytest.raises(RuntimeError, match="not finished"),
    ):
        _ = prof.report


def test_kernel_profiling_disabled_leaves_the_occupancy_axis_empty() -> None:
    backend = FakeNvmlBackend()

    with profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=False) as prof:
        time.sleep(0.02)

    assert prof.report.kernel_time_fraction is None
    assert prof.report.peak_vram_torch_bytes is None


def test_kernel_profiling_is_skipped_when_cuda_is_unavailable() -> None:
    """On a CPU-only machine the occupancy axis is unmeasurable, not zero."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("this test describes the CPU-only path")

    backend = FakeNvmlBackend()

    with profile_gpu(backend=backend, interval_s=0.005, kernel_profiling=True) as prof:
        time.sleep(0.02)

    assert prof.report.kernel_time_fraction is None
    assert prof.report.peak_vram_torch_bytes is None
