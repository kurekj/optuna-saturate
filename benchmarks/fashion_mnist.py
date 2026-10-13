"""Compare saturate() with plain Optuna on Fashion-MNIST.

Kept as the entry point the first release documented. It runs the
``fashion_cnn`` task of :mod:`benchmarks.compare_arms` with the three original
arms; every other option of that script is accepted and passed through:

    python -m benchmarks.fashion_mnist --trials 24 --group-sizes 8 --seeds 3
"""

from __future__ import annotations

import sys

from benchmarks.compare_arms import main


def run() -> int:
    argv = ["--task", "fashion_cnn"]
    if "--arms" not in sys.argv:
        argv += ["--arms", "sequential", "n_jobs", "saturate"]
    return main(argv + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(run())
