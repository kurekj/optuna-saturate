import builtins

import pytest

from optuna_saturate.exceptions import MissingDependencyError
from optuna_saturate.runtime.nvml import NvmlBackend, RealNvmlBackend
from tests.fakes import FakeNvmlBackend


def test_the_fake_satisfies_the_backend_protocol() -> None:
    backend: NvmlBackend = FakeNvmlBackend()
    assert isinstance(backend.utilization_percent(), int)
    assert isinstance(backend.memory_used_bytes(), int)


def test_readings_advance_through_the_scripted_sequence() -> None:
    backend = FakeNvmlBackend(utilizations=[10, 20, 30])
    assert [backend.utilization_percent() for _ in range(3)] == [10, 20, 30]


def test_an_exhausted_sequence_repeats_its_last_value() -> None:
    backend = FakeNvmlBackend(utilizations=[7, 9])
    assert [backend.utilization_percent() for _ in range(4)] == [7, 9, 9, 9]


def test_energy_may_be_absent_to_model_unsupported_hardware() -> None:
    backend = FakeNvmlBackend(energies=None)
    assert backend.total_energy_mj() is None


def test_a_missing_pynvml_raises_a_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the [gpu] extra the failure must name the package to install."""
    real_import = builtins.__import__

    def refuse_pynvml(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynvml":
            raise ImportError("No module named 'pynvml'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse_pynvml)

    with pytest.raises(MissingDependencyError, match="nvidia-ml-py"):
        RealNvmlBackend(device_index=0)
