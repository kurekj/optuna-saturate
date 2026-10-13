"""How far the summed kernel duration over-counts the device's busy time.

    python benchmarks/kernel_overlap.py --group-size 64 --trials 128
    python benchmarks/kernel_overlap.py --arm sequential --trials 24

``kernel_time_fraction`` sums the recorded duration of every CUDA kernel in a
block and divides by the wall time. The records of successive kernels overlap
once kernels are dense, so the sum can exceed the length of their union, and
even the wall time. This script runs one arm of the Fashion-MNIST comparison
with the trace kept, and prints the sum, the union and the breakdown of device
events, so that the kernel column of the results can be read as the upper bound
it is.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch.profiler as torch_profiler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import compare_arms  # noqa: E402
from benchmarks.tasks import TASKS  # noqa: E402

captured: dict[str, torch_profiler.profile] = {}


class _KeepingProfile(torch_profiler.profile):
    """The profiler ``profile_gpu`` creates, kept after the block for inspection."""

    def __exit__(self, *args: object) -> None:
        super().__exit__(*args)
        captured["profile"] = self


def union_seconds(intervals: list[tuple[float, float]]) -> tuple[float, int]:
    """Length of the union of intervals (in seconds) and the number of overlaps."""
    intervals = sorted(intervals)
    total = 0.0
    overlaps = 0
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s < end:
            overlaps += 1
            end = max(end, e)
        else:
            total += end - start
            start, end = s, e
    total += end - start
    return total / 1e6, overlaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="fashion_cnn", choices=sorted(TASKS))
    parser.add_argument("--arm", default="saturate", choices=["sequential", "saturate"])
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=8192)
    parser.add_argument("--valid-subset", type=int, default=2048)
    args = parser.parse_args(argv)

    torch_profiler.profile = _KeepingProfile  # type: ignore[misc]
    task = TASKS[args.task]
    train_loader, valid_loader = task.load(
        compare_arms.BATCH_SIZE, train_subset=args.train_subset, valid_subset=args.valid_subset
    )
    report, _, _ = compare_arms._run_study_arm(
        task,
        args.arm,
        args.seed,
        args.trials,
        args.group_size,
        None,
        train_loader,
        valid_loader,
        1,
        None,
        kernel_profiling=True,
    )
    assert report.kernel_time_fraction is not None
    print(
        f"wall {report.wall_time_s:.3f} s, kernel_time_fraction {report.kernel_time_fraction:.4f}"
    )

    events = [e for e in captured["profile"].events() if e.device_type.name == "CUDA"]
    by_kind: dict[str, float] = defaultdict(float)
    intervals = []
    for event in events:
        kind = (
            "memcpy" if "Memcpy" in event.name else "memset" if "Memset" in event.name else "kernel"
        )
        by_kind[kind] += (event.time_range.end - event.time_range.start) / 1e6
        intervals.append((event.time_range.start, event.time_range.end))
    total = sum(by_kind.values())
    union, overlaps = union_seconds(intervals)
    print(
        f"{len(events)} device events: sum {total:.3f} s, union {union:.3f} s, "
        f"{overlaps} overlapping"
    )
    for kind, seconds in sorted(by_kind.items()):
        print(f"  {kind}: {seconds:.3f} s")
    print(
        f"sum / wall {total / report.wall_time_s:.4f}, "
        f"union / wall {union / report.wall_time_s:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
