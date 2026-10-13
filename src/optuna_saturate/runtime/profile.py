"""Measuring how far a workload is from saturating the device.

Under-use has two independent axes and one number cannot express both.

*Temporal* under-use is time when no kernel is resident at all — gaps left by the
data loader and by kernel launch overhead. NVML measures this, and it is what
``busy_fraction`` reports.

*Occupancy* under-use is a kernel too small to fill the multiprocessors. NVML
cannot see it: its utilisation reading is the percentage of time during which at
least one kernel was executing, so a kernel occupying a single multiprocessor
still reads 100. ``kernel_time_fraction``, taken from the CUDA kernel trace,
covers this axis.

Neither field is called "SM utilisation", because neither one is that.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from optuna_saturate.runtime.nvml import NvmlBackend, RealNvmlBackend


@dataclass(frozen=True)
class GpuProfile:
    """What a measured block cost, on both axes of under-use.

    Attributes:
        wall_time_s: Duration of the measured block.
        busy_fraction: Share of NVML samples that saw at least one resident
            kernel. Temporal axis. NVML averages over a window on the order of a
            second, so this lags: a block shorter than that window can read zero
            while genuinely busy, and can read non-zero because of work that ran
            just before it. Trust it over seconds, not milliseconds.
        kernel_time_fraction: Summed CUDA kernel duration over wall time. The
            records of successive kernels overlap once kernels are dense, so the
            sum bounds the share of wall time with a kernel running from above;
            ``benchmarks/kernel_overlap.py`` measures the difference.
            Occupancy axis. ``None`` when kernel profiling was disabled.
        peak_vram_nvml_bytes: Highest device memory reading. Includes the CUDA
            context and any other process on the device.
        peak_vram_torch_bytes: Highest allocation PyTorch itself made. ``None``
            when CUDA is unavailable.
        energy_j: Energy drawn by the GPU board during the block, as the
            difference of NVML's cumulative energy counter, or ``None`` if the
            device has no such counter. This is the device's own draw as the
            driver reports it; host CPU, memory and power-supply losses are not
            included, so it is a GPU figure, not a system one.
        mean_power_w: ``energy_j / wall_time_s``, the board's average power over
            the block, or ``None`` when the energy is unavailable.
        mean_sm_clock_mhz: Mean of the sampled streaming-multiprocessor clock.
            A device idling between small kernels runs at a fraction of its
            peak clock, which lengthens the kernels it does run; comparisons
            across runs should check that this figure is comparable.
        max_temperature_c: Highest sampled device temperature.
        sample_count: Number of NVML samples taken.
        effective_sample_hz: Sampling rate actually achieved, which is well below
            the requested rate because an NVML query costs tens of milliseconds.
    """

    wall_time_s: float
    busy_fraction: float
    kernel_time_fraction: float | None
    peak_vram_nvml_bytes: int
    peak_vram_torch_bytes: int | None
    energy_j: float | None
    mean_power_w: float | None
    mean_sm_clock_mhz: float
    max_temperature_c: int
    sample_count: int
    effective_sample_hz: float


def build_profile(
    utilizations: Sequence[int],
    memories: Sequence[int],
    energy_start_mj: int | None,
    energy_end_mj: int | None,
    wall_time_s: float,
    kernel_time_fraction: float | None = None,
    peak_vram_torch_bytes: int | None = None,
    sm_clocks: Sequence[int] = (),
    temperatures: Sequence[int] = (),
) -> GpuProfile:
    """Turn raw samples into a profile.

    Kept free of threads and hardware so the arithmetic can be tested directly.

    Args:
        utilizations: Utilisation readings, one per sample.
        memories: Memory readings in bytes, one per sample.
        energy_start_mj: Energy counter before the block, or ``None`` if absent.
        energy_end_mj: Energy counter after the block, or ``None`` if absent.
        wall_time_s: Duration of the measured block.
        kernel_time_fraction: Occupancy axis, when available.
        peak_vram_torch_bytes: PyTorch's own peak allocation, when available.
        sm_clocks: Clock readings in MHz, one per sample.
        temperatures: Temperature readings in degrees Celsius, one per sample.

    Returns:
        The assembled profile. A run that collected no samples yields zeros
        rather than raising, so a block too short to sample still reports.
    """
    sample_count = len(utilizations)

    busy_fraction = (
        sum(1 for value in utilizations if value > 0) / sample_count if sample_count else 0.0
    )

    energy_j: float | None = None
    mean_power_w: float | None = None
    if energy_start_mj is not None and energy_end_mj is not None:
        energy_j = (energy_end_mj - energy_start_mj) / 1000.0
        if wall_time_s > 0:
            mean_power_w = energy_j / wall_time_s

    return GpuProfile(
        wall_time_s=wall_time_s,
        busy_fraction=busy_fraction,
        kernel_time_fraction=kernel_time_fraction,
        peak_vram_nvml_bytes=max(memories) if memories else 0,
        peak_vram_torch_bytes=peak_vram_torch_bytes,
        energy_j=energy_j,
        mean_power_w=mean_power_w,
        mean_sm_clock_mhz=sum(sm_clocks) / len(sm_clocks) if sm_clocks else 0.0,
        max_temperature_c=max(temperatures) if temperatures else 0,
        sample_count=sample_count,
        effective_sample_hz=sample_count / wall_time_s if wall_time_s > 0 else 0.0,
    )


def _kernel_time_seconds(profiler: Any) -> float:
    """Total time CUDA kernels spent executing, from a finished torch profile.

    ``key_averages`` aggregates by operator; ``self_device_time_total`` is that
    operator's own device time in microseconds, excluding children, so summing it
    counts each kernel once.
    """
    total_us: float = sum(
        entry.self_device_time_total
        for entry in profiler.key_averages()
        if entry.self_device_time_total > 0
    )
    return total_us / 1e6


class _Sampler(threading.Thread):
    """Polls the backend until asked to stop."""

    def __init__(self, backend: NvmlBackend, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._backend = backend
        self._interval_s = interval_s
        # Not `_stop`: Thread already has a private `_stop()` that join() calls,
        # and shadowing it with an Event makes join() raise TypeError.
        self._stop_event = threading.Event()
        self.utilizations: list[int] = []
        self.memories: list[int] = []
        self.sm_clocks: list[int] = []
        self.temperatures: list[int] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.utilizations.append(self._backend.utilization_percent())
            self.memories.append(self._backend.memory_used_bytes())
            self.sm_clocks.append(self._backend.sm_clock_mhz())
            self.temperatures.append(self._backend.temperature_c())
            # wait() rather than sleep(): a stop request ends the thread at once
            # instead of after one more full interval.
            self._stop_event.wait(self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()


class _Profiler:
    """Handle yielded by :func:`profile_gpu`."""

    def __init__(self, sampler: _Sampler) -> None:
        self._sampler = sampler
        self._report: GpuProfile | None = None

    @property
    def report(self) -> GpuProfile:
        """The finished profile.

        Raises:
            RuntimeError: The measured block has not finished yet.
        """
        if self._report is None:
            raise RuntimeError("the profiled block has not finished, so there is no report yet")
        return self._report

    def is_sampling(self) -> bool:
        """Whether the sampling thread is still alive."""
        return self._sampler.is_alive()

    def _finish(self, report: GpuProfile) -> None:
        self._report = report


@contextmanager
def profile_gpu(
    device: int = 0,
    interval_s: float = 0.02,
    kernel_profiling: bool = True,
    backend: NvmlBackend | None = None,
) -> Iterator[_Profiler]:
    """Measure the device cost of the enclosed block.

    Args:
        device: Device index, used for both the NVML handle and the PyTorch
            memory statistics. The two numbering schemes coincide on a
            single-device machine but can diverge under ``CUDA_VISIBLE_DEVICES``
            or a differing PCI enumeration order; selecting devices separately
            is deferred until multi-GPU support exists. Ignored for NVML when
            ``backend`` is supplied.
        interval_s: Requested gap between samples. Values below roughly 0.02 buy
            nothing: a single NVML query already costs tens of milliseconds, and
            a tighter loop only steals CPU from the workload being measured.
        kernel_profiling: Collect the CUDA kernel trace for the occupancy axis.
            Carries its own overhead, so it can be turned off for short blocks.
        backend: Reading source. Supplying one keeps the caller in charge of its
            lifetime; otherwise a real NVML backend is created and closed here.

    Yields:
        A handle whose ``report`` holds the profile once the block has finished.
        The report is produced even when the block raises, so a failing run still
        records what happened before the failure.
    """
    owns_backend = backend is None
    active_backend = backend if backend is not None else RealNvmlBackend(device_index=device)

    sampler = _Sampler(active_backend, interval_s)
    profiler = _Profiler(sampler)

    # The kernel trace needs both a CUDA build and a visible device; PyTorch's
    # memory statistics need only the device, and are cheap, so they are
    # collected whether or not the trace is.
    track_memory = torch.cuda.is_available()
    collect_kernels = kernel_profiling and track_memory

    torch_profiler: Any = None
    if track_memory:
        torch.cuda.init()  # type: ignore[no-untyped-call]
        torch.cuda.reset_peak_memory_stats(device)
    if collect_kernels:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torch_profile

        torch_profiler = torch_profile(activities=[ProfilerActivity.CUDA])
        torch_profiler.__enter__()

    energy_start = active_backend.total_energy_mj()
    started_at = time.perf_counter()
    sampler.start()
    try:
        yield profiler
    finally:
        wall_time_s = time.perf_counter() - started_at
        sampler.stop()
        sampler.join()
        energy_end = active_backend.total_energy_mj()

        kernel_time_fraction: float | None = None
        peak_vram_torch_bytes: int | None = None
        if track_memory:
            # Kernels are asynchronous; without this the trace is short by
            # whatever was still in flight when the block ended.
            torch.cuda.synchronize(device)
            peak_vram_torch_bytes = int(torch.cuda.max_memory_allocated(device))
        if collect_kernels:
            torch_profiler.__exit__(None, None, None)
            if wall_time_s > 0:
                kernel_time_fraction = _kernel_time_seconds(torch_profiler) / wall_time_s

        profiler._finish(
            build_profile(
                utilizations=sampler.utilizations,
                memories=sampler.memories,
                energy_start_mj=energy_start,
                energy_end_mj=energy_end,
                wall_time_s=wall_time_s,
                kernel_time_fraction=kernel_time_fraction,
                peak_vram_torch_bytes=peak_vram_torch_bytes,
                sm_clocks=sampler.sm_clocks,
                temperatures=sampler.temperatures,
            )
        )

        if owns_backend:
            active_backend.close()
