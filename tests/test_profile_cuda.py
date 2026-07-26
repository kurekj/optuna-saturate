"""Profiler checks that need a physical device.

Skipped without CUDA. Assertions are deliberately about relations and signs, not
absolute values: what the rest of the machine is doing moves every absolute
number, and a test that pins one down would fail for reasons unrelated to the
code.
"""

from __future__ import annotations

import importlib.util
import time

import pytest
import torch

from optuna_saturate.runtime.profile import profile_gpu

# A device is not enough: these read NVML, which arrives with the [gpu] extra.
# Without the second guard, a GPU machine that installed only [dev] would report
# six failures where it should report six skips.
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
    pytest.mark.skipif(
        importlib.util.find_spec("pynvml") is None,
        reason='requires NVML bindings: pip install "optuna-saturate[gpu]"',
    ),
]


def _busy_work(seconds: float = 0.6, size: int = 1024) -> None:
    """Keep the device busy for a wall-clock duration, not a fixed iteration count.

    NVML averages utilisation over a window on the order of a second, so a
    workload measured in tens of milliseconds can finish entirely inside one
    window and leave every sample reading zero. Driving the load by elapsed time
    keeps these tests independent of how fast the device happens to be.
    """
    a = torch.randn(size, size, device="cuda")
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(50):
            a = a @ a
            a = a / a.norm()
        torch.cuda.synchronize()


def test_a_real_workload_reports_a_positive_busy_fraction() -> None:
    with profile_gpu(interval_s=0.02, kernel_profiling=False) as prof:
        _busy_work()

    assert prof.report.busy_fraction > 0.0
    assert prof.report.sample_count > 0


def test_the_achieved_sample_rate_is_below_the_requested_one() -> None:
    """An NVML query costs tens of milliseconds; the report must not pretend otherwise."""
    with profile_gpu(interval_s=0.001, kernel_profiling=False) as prof:
        _busy_work()

    assert prof.report.effective_sample_hz < 1000.0


def test_energy_drawn_during_real_work_is_positive() -> None:
    with profile_gpu(interval_s=0.02, kernel_profiling=False) as prof:
        _busy_work()

    energy = prof.report.energy_j
    if energy is None:
        pytest.skip("this device exposes no energy counter")
    assert energy > 0.0


def test_the_occupancy_axis_is_populated_on_real_hardware() -> None:
    with profile_gpu(interval_s=0.02, kernel_profiling=True) as prof:
        _busy_work()

    fraction = prof.report.kernel_time_fraction
    assert fraction is not None
    assert 0.0 < fraction <= 1.5  # >1 is possible: concurrent kernels overlap in time


def test_peak_torch_memory_grows_with_the_allocation() -> None:
    with profile_gpu(interval_s=0.02, kernel_profiling=True) as small:
        torch.randn(256, 256, device="cuda")
        torch.cuda.synchronize()

    with profile_gpu(interval_s=0.02, kernel_profiling=True) as large:
        torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()

    small_peak = small.report.peak_vram_torch_bytes
    large_peak = large.report.peak_vram_torch_bytes
    assert small_peak is not None and large_peak is not None
    assert large_peak > small_peak


def test_nvml_memory_exceeds_torch_memory_because_it_includes_the_context() -> None:
    """The two memory fields answer different questions and must not be conflated."""
    with profile_gpu(interval_s=0.02, kernel_profiling=True) as prof:
        torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()

    torch_peak = prof.report.peak_vram_torch_bytes
    assert torch_peak is not None
    assert prof.report.peak_vram_nvml_bytes > torch_peak
