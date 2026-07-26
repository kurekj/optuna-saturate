"""Test doubles used across the test suite."""

from __future__ import annotations

from collections.abc import Sequence


class FakeNvmlBackend:
    """An NvmlBackend whose readings are scripted by the test.

    Each call advances through the supplied sequences. Once a sequence is
    exhausted its last value repeats, so a test never has to predict exactly how
    many times the sampler thread will poll.
    """

    def __init__(
        self,
        utilizations: Sequence[int] = (0,),
        memories: Sequence[int] = (0,),
        energies: Sequence[int] | None = (0,),
    ) -> None:
        self._utilizations = list(utilizations)
        self._memories = list(memories)
        self._energies = None if energies is None else list(energies)
        self._utilization_calls = 0
        self._memory_calls = 0
        self._energy_calls = 0
        self.closed = False

    @staticmethod
    def _advance(values: list[int], index: int) -> int:
        return values[min(index, len(values) - 1)]

    def utilization_percent(self) -> int:
        value = self._advance(self._utilizations, self._utilization_calls)
        self._utilization_calls += 1
        return value

    def memory_used_bytes(self) -> int:
        value = self._advance(self._memories, self._memory_calls)
        self._memory_calls += 1
        return value

    def total_energy_mj(self) -> int | None:
        if self._energies is None:
            return None
        value = self._advance(self._energies, self._energy_calls)
        self._energy_calls += 1
        return value

    def close(self) -> None:
        self.closed = True
