"""Compare saturate() with plain Optuna on Fashion-MNIST.

Three arms get the same search space, the same trial count and the same seed:

    sequential  study.optimize(objective, n_trials=N)
    n_jobs      study.optimize(objective, n_trials=N, n_jobs=K)
    saturate    osat.saturate(study, vectorised_objective, group_size=K)

Run from the repository root, as a module so that ``benchmarks`` is importable:

    python -m benchmarks.fashion_mnist --trials 24 --group-size 8 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import optuna
import torch

import optuna_saturate as osat
from benchmarks.data import load_fashion_mnist
from benchmarks.models import SmallCNN, build_cnn_members
from optuna_saturate.runtime.profile import GpuProfile, profile_gpu

EPOCHS = 1
BATCH_SIZE = 128
DEVICE = "cuda:0"

# Search space, shared by every arm. Two vectorisable names and one that is not.
LR_RANGE = (1e-3, 3e-1)
DROPOUT_RANGE = (0.0, 0.6)
WIDTH_RANGE = (8, 48)


def _accuracy(model: SmallCNN, loader: Any) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += int((model(x).argmax(dim=-1) == y).sum())
            total += int(y.shape[0])
    model.train()
    return correct / total


def make_plain_objective(train_loader: Any, valid_loader: Any) -> Any:
    """One model per trial -- what Optuna does on its own."""

    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_int("width", *WIDTH_RANGE)
        lr = trial.suggest_float("learning_rate", *LR_RANGE, log=True)
        dropout = trial.suggest_float("dropout", *DROPOUT_RANGE)

        torch.manual_seed(0)
        model = SmallCNN(width=width, dropout_default=dropout).to(DEVICE)
        optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        for _ in range(EPOCHS):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimiser.zero_grad()
                torch.nn.functional.cross_entropy(model(x), y).backward()
                optimiser.step()
        return _accuracy(model, valid_loader)

    return objective


def make_vectorised_objective(train_loader: Any, valid_loader: Any) -> Any:
    """One batched model per group -- the same training, executed together."""

    @osat.vectorizable(over=["learning_rate", "dropout"])
    def objective(trial: Any, ctx: Any) -> list[float]:
        width = trial.suggest_int("width", *WIDTH_RANGE)
        lr = trial.suggest_float("learning_rate", *LR_RANGE, log=True)
        dropout = trial.suggest_float("dropout", *DROPOUT_RANGE)

        ensemble = ctx.stack(build_cnn_members(k=ctx.k, width=width, seed=0))
        optimiser = ctx.sgd(ensemble, lr=lr)
        for _ in range(EPOCHS):
            for x, y in train_loader:
                ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": dropout})
        return ctx.accuracy(ensemble, valid_loader)

    return objective


def _study(seed: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))


def run_arm(
    arm: str, seed: int, trials: int, group_size: int, train_loader: Any, valid_loader: Any
) -> tuple[GpuProfile, float, int]:
    """Run one arm end to end and return its profile, best accuracy and trial count."""
    torch.manual_seed(seed)
    study = _study(seed)

    with profile_gpu(interval_s=0.05, kernel_profiling=True) as prof:
        if arm == "sequential":
            study.optimize(make_plain_objective(train_loader, valid_loader), n_trials=trials)
        elif arm == "n_jobs":
            study.optimize(
                make_plain_objective(train_loader, valid_loader),
                n_trials=trials,
                n_jobs=group_size,
            )
        elif arm == "saturate":
            osat.saturate(
                study,
                make_vectorised_objective(train_loader, valid_loader),
                n_trials=trials,
                group_size=group_size,
                device=DEVICE,
            )
        else:
            raise ValueError(f"unknown arm {arm!r}")

    return prof.report, float(study.best_value), len(study.trials)


def format_row(
    arm: str, seed: int, profile: GpuProfile, best: float, trials: int
) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": seed,
        "trials_completed": trials,
        "wall_time_s": round(profile.wall_time_s, 3),
        "best_accuracy": round(best, 4),
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
        "energy_j": None if profile.energy_j is None else round(profile.energy_j, 1),
    }


def _summarise(rows: list[dict[str, object]]) -> None:
    """Mean over seeds per arm, with speedup against the sequential arm."""

    def mean(arm: str, field: str) -> float:
        values = [float(r[field]) for r in rows if r["arm"] == arm]  # type: ignore[arg-type]
        return sum(values) / len(values)

    base_time = mean("sequential", "wall_time_s")
    base_acc = mean("sequential", "best_accuracy")
    print(f"\n{'arm':<10} {'wall(s)':>9} {'speedup':>9} {'best acc':>9} {'delta acc':>10}")
    for arm in ("sequential", "n_jobs", "saturate"):
        wall = mean(arm, "wall_time_s")
        acc = mean(arm, "best_accuracy")
        print(f"{arm:<10} {wall:9.2f} {base_time / wall:8.2f}x {acc:9.4f} {acc - base_acc:+10.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-subset", type=int, default=8192)
    parser.add_argument("--valid-subset", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/fashion_mnist.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "No CUDA device available. This benchmark measures GPU under-use and "
            "cannot run on CPU.",
            file=sys.stderr,
        )
        return 1

    train_loader, valid_loader = load_fashion_mnist(
        batch_size=BATCH_SIZE,
        train_subset=args.train_subset,
        valid_subset=args.valid_subset,
    )
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"train batches: {len(train_loader)}  valid batches: {len(valid_loader)}")

    rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        for arm in ("sequential", "n_jobs", "saturate"):
            profile, best, trials = run_arm(
                arm, seed, args.trials, args.group_size, train_loader, valid_loader
            )
            rows.append(format_row(arm, seed, profile, best, trials))
            print(
                f"seed {seed}  {arm:<10} "
                f"wall {profile.wall_time_s:7.2f}s  "
                f"best {best:.4f}  "
                f"busy {profile.busy_fraction:5.1%}  "
                f"kernel {profile.kernel_time_fraction or 0:5.1%}  "
                f"energy {profile.energy_j or 0:7.1f}J"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _summarise(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
