# Measurements

Raw per-seed measurements behind every table and figure of the article,
written by `benchmarks/compare_arms.py` and `benchmarks/vectorization_gain.py`
through `run_all_benchmarks.sh`. One row per (task, arm, requested group size,
seed). All runs: NVIDIA RTX PRO 5000 Blackwell Laptop GPU, driver 596.53,
PyTorch 2.14.1+cu130, Optuna 5.0.0, Ray 2.59.0, Python 3.10.9, Windows 11.

| File | Contents |
|---|---|
| `fashion_cnn.csv` | six strategies, K in {2, 4, 8, 16, 32}, 24 trials, 5 seeds |
| `fashion_cnn_replay.csv` | the same 24 configurations in every arm (fixed schedule), K = 8, 5 seeds |
| `kmnist_mlp.csv`, `cifar_vit.csv` | the other two tasks, K = 8, 24 trials, 5 seeds |
| `fashion_cnn_full_k16/32/64.csv` | full groups: 2K trials, sequential against saturate, 3 seeds |
| `fashion_cnn_long.csv` | full training set, 3 epochs per trial, 48 trials, 3 seeds, loader in the training process |
| `fashion_cnn_long_workers.csv` | as above with four persistent DataLoader workers per loader |
| `vectorization_gain.csv` | synthetic two-layer network, K up to 64, three hidden widths |

## Columns of the `compare_arms` files

| Column | Meaning |
|---|---|
| `task` | `fashion_cnn`, `kmnist_mlp` or `cifar_vit`, as defined in `benchmarks/tasks.py` |
| `seed` | the seed of the row: sampler, initial weights and batch order |
| `arm` | `sequential`, `n_jobs`, `streams`, `process`, `ray`, `saturate`, `saturate_equal_time` |
| `group_size` | requested K: the group size of `saturate`, the thread, process or trial concurrency of the others; empty for `sequential` |
| `workers` | processes or trials actually admitted (the process arm caps them by free memory) |
| `schedule` | `tpe` (the seeded sampler proposes) or `replay` (fixed schedule by trial number) |
| `trials_completed` | trials recorded; for `saturate_equal_time`, those recorded inside the budget |
| `distinct_architectures` | distinct shape-changing values among the recorded trials |
| `wall_time_s` | elapsed time of the measured block (data loaded and CUDA context created before it) |
| `budget_s` | the equal-time arm's budget: the sequential arm's wall time on the same seed |
| `best_accuracy` | best validation accuracy among the recorded trials; the search objective, not a test estimate |
| `busy_fraction` | share of NVML utilisation polls that returned a non-zero value (temporal) |
| `kernel_time_fraction` | summed CUDA kernel duration over wall time, an upper bound on the share of wall time with a kernel running (the records of successive kernels overlap once kernels are dense; `benchmarks/kernel_overlap.py` measures the difference); for multi-process arms the sum over processes; empty when the trace was off |
| `multi_process_bounds` | `True` when the kernel and memory figures are the multi-process bounds above |
| `peak_vram_nvml_mib` | highest NVML device memory reading (includes contexts and other processes) |
| `peak_vram_torch_mib` | PyTorch's own peak allocation; for multi-process arms the sum of the processes' peaks |
| `energy_j` | GPU-board energy from NVML's cumulative counter |
| `mean_power_w` | `energy_j / wall_time_s` |
| `mean_sm_clock_mhz` | mean sampled streaming-multiprocessor clock during the block |
| `max_temperature_c` | highest sampled device temperature |
| `epochs`, `train_subset`, `loader_workers` | the training protocol of the row (`train_subset` empty: full set) |
| `device`, `torch` | the GPU name and the PyTorch version string of the process that measured the row |

Rows for an arm were measured in a fresh process after a three-second warm-up;
the Ray arm's timing includes Ray's per-trial scheduling but not `ray.init`.

## Columns of `vectorization_gain.csv`

One row per (K, hidden width, mode) for a synthetic two-layer network trained
for 300 steps on random data. `k` is the number of configurations, `hidden`
the width of the hidden layer, `mode` is `sequential` (K independent models,
one after another) or `vectorised` (one stacked model). The profiler columns
have the meanings above; `effective_sample_hz` is the NVML polling rate the
profiler actually achieved during the block.

`fashion_mnist.csv` is the measurement behind the first version of the
article (v0.1.0 of the library, three seeds) and is kept for reference; the
tables of the revised article are built from the files listed above.
