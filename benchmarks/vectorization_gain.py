"""Measure what vectorising a hyperparameter group buys, and what it costs.

Trains K configurations twice: once as K independent models, once as a single
stacked model, and reports both axes of device under-use for each. Writes one CSV
row per (K, hidden width) pair.

Run:
    python benchmarks/vectorization_gain.py --out results/vectorization_gain.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from optuna_saturate.runtime.profile import GpuProfile, profile_gpu
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_steps
from optuna_saturate.vectorized.optim import VectorizedSGD

IN_FEATURES = 8
CLASSES = 4
BATCH = 64
STEPS = 300
DEVICE = "cuda"


class Net(torch.nn.Module):
    """Two-layer MLP honouring the StackedEnsemble forward contract."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(IN_FEATURES, hidden)
        self.fc2 = torch.nn.Linear(hidden, CLASSES)

    def forward(self, x: torch.Tensor, hp: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def build_members(k: int, hidden: int) -> list[Net]:
    members = []
    for i in range(k):
        torch.manual_seed(i)
        members.append(Net(hidden).to(DEVICE))
    return members


def build_batches(n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(BATCH, IN_FEATURES, generator=generator).to(DEVICE),
            torch.randint(0, CLASSES, (BATCH,), generator=generator).to(DEVICE),
        )
        for _ in range(n)
    ]


def run_sequential(k: int, hidden: int) -> None:
    batches = build_batches(STEPS)
    for model in build_members(k, hidden):
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        for inputs, targets in batches:
            optimiser.zero_grad()
            torch.nn.functional.cross_entropy(model(inputs), targets).backward()
            optimiser.step()
    torch.cuda.synchronize()


def run_vectorised(k: int, hidden: int) -> None:
    batches = build_batches(STEPS)
    ensemble = StackedEnsemble(build_members(k, hidden))
    optimiser = VectorizedSGD(ensemble.params, lr=torch.full((k,), 0.1, device=DEVICE))
    train_steps(ensemble, optimiser, batches)
    torch.cuda.synchronize()


def measure(k: int, hidden: int, vectorised: bool) -> GpuProfile:
    run = run_vectorised if vectorised else run_sequential
    run(k, hidden)  # warm-up: CUDA context, autotuning, allocator growth
    with profile_gpu(interval_s=0.02, kernel_profiling=True) as prof:
        run(k, hidden)
    return prof.report


def format_row(k: int, hidden: int, label: str, profile: GpuProfile) -> dict[str, object]:
    return {
        "k": k,
        "hidden": hidden,
        "mode": label,
        "wall_time_s": round(profile.wall_time_s, 4),
        "busy_fraction": round(profile.busy_fraction, 4),
        "kernel_time_fraction": (
            None if profile.kernel_time_fraction is None else round(profile.kernel_time_fraction, 4)
        ),
        "peak_vram_nvml_mib": profile.peak_vram_nvml_bytes // 1024**2,
        "peak_vram_torch_mib": (
            None
            if profile.peak_vram_torch_bytes is None
            else profile.peak_vram_torch_bytes // 1024**2
        ),
        "energy_j": None if profile.energy_j is None else round(profile.energy_j, 2),
        "effective_sample_hz": round(profile.effective_sample_hz, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/vectorization_gain.csv"))
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 128, 1024])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "No CUDA device available. This benchmark measures GPU under-use and "
            "cannot run on CPU. Install a CUDA build of PyTorch.",
            file=sys.stderr,
        )
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    rows: list[dict[str, object]] = []

    for hidden in args.hidden:
        for k in args.k:
            sequential = measure(k, hidden, vectorised=False)
            vectorised = measure(k, hidden, vectorised=True)
            rows.append(format_row(k, hidden, "sequential", sequential))
            rows.append(format_row(k, hidden, "vectorised", vectorised))
            speedup = sequential.wall_time_s / vectorised.wall_time_s
            print(
                f"hidden={hidden:5d} k={k:3d}  "
                f"wall {sequential.wall_time_s:6.3f}s -> {vectorised.wall_time_s:6.3f}s "
                f"({speedup:4.2f}x)  "
                f"busy {sequential.busy_fraction:5.1%} -> {vectorised.busy_fraction:5.1%}  "
                f"kernel {sequential.kernel_time_fraction or 0:5.1%} -> "
                f"{vectorised.kernel_time_fraction or 0:5.1%}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
