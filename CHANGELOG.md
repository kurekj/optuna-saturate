# Changelog

## 0.2.0

Added
- `VectorizedSGD` accepts per-member momentum and weight decay, and Nesterov momentum.
- `VectorizedAdam` and `VectorizedAdamW`, with per-member learning rate, betas and weight decay;
  `ctx.adam()` and `ctx.adamw()` build them.
- `autocast_dtype` on `saturate()` and `VectorizedContext`: bfloat16 or float16 forward pass and loss.
- `timeout` and `callbacks` on `saturate()`, as on `study.optimize()`; `StopSaturate` to end a run from a callback.
- `trial.set_user_attr()`, `trial.set_member_attr()`, `trial.number` and `trial.numbers` on the group's trial proxy.
- `recommend_group_size()`, `fit_memory_model()` and `MemoryModel` for choosing K from measured peaks.
- `benchmarks/kernel_overlap.py`: how far the summed kernel duration over-counts the device's busy time.
- `StackedEnsemble.train()` / `.eval()`; `ctx.accuracy()` evaluates in evaluation mode.
- `GpuProfile.mean_power_w`, `mean_sm_clock_mhz` and `max_temperature_c`; the energy field is documented as
  the GPU board's draw from NVML; PyTorch's peak allocation is recorded whether or not the kernel trace is.
- Parity tests for momentum, weight decay, Adam, AdamW, a Transformer encoder, bfloat16 autocast and
  stochastic dropout; Optuna compatibility tests for samplers, storages, callbacks and user attributes.
- Benchmarks: `compare_arms` with process-based, CUDA-stream and Ray Tune arms, an equal-time arm, a
  group-size sweep and a replay schedule that hands every arm the same configurations; KMNIST MLP and
  CIFAR-10 Transformer tasks.

Changed
- Multi-objective studies and `GridSampler` are rejected with an explanation instead of failing later.
- `trial.report()` and `trial.should_prune()` raise `NotImplementedError` with an explanation.

## 0.1.0

First release.
