"""The only point of contact with NVML.

Everything the profiler needs from the driver goes through the ``NvmlBackend``
protocol, so tests can substitute a scripted double and run on machines with no
NVIDIA device at all.
"""

from __future__ import annotations

from typing import Any, Protocol

from optuna_saturate.exceptions import MissingDependencyError


class NvmlBackend(Protocol):
    """A source of per-device readings."""

    def utilization_percent(self) -> int:
        """Percent of the last sample period with at least one kernel resident.

        This is a *temporal* measure. It says nothing about how many
        multiprocessors were busy: a kernel occupying one of them reads 100.
        """

    def memory_used_bytes(self) -> int:
        """Device memory in use, including the CUDA context and other processes."""

    def total_energy_mj(self) -> int | None:
        """Cumulative energy draw in millijoules, or ``None`` if unsupported.

        The counter runs since the driver was last reloaded, so only differences
        between two readings are meaningful.
        """

    def close(self) -> None:
        """Release the NVML handle."""


class RealNvmlBackend:
    """Readings from a physical device via NVML.

    Args:
        device_index: Zero-based NVML device index.

    Raises:
        MissingDependencyError: ``nvidia-ml-py`` is not installed.
    """

    def __init__(self, device_index: int = 0) -> None:
        try:
            import pynvml
        except ImportError as exc:
            raise MissingDependencyError(
                "GPU profiling requires NVML bindings. Install them with: "
                'pip install "optuna-saturate[gpu]"  (provides nvidia-ml-py)'
            ) from exc

        self._nvml: Any = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)

    def utilization_percent(self) -> int:
        rates = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        return int(rates.gpu)

    def memory_used_bytes(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def total_energy_mj(self) -> int | None:
        try:
            return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            # Reported as unsupported on pre-Volta parts and inside some VMs.
            # A missing counter must degrade the report, not abort the run.
            return None

    def close(self) -> None:
        self._nvml.nvmlShutdown()
