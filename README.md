# optuna-saturate

Saturate a single GPU during Optuna hyperparameter optimisation.

Hyperparameter search spaces split into two classes that need different
execution strategies:

- **Shape-preserving** hyperparameters (learning rate, dropout, temperature,
  weight decay) do not change tensor shapes. K such configurations can be
  trained **as a single batched model** via `torch.func.stack_module_state`
  and `vmap`.
- **Shape-changing** hyperparameters (embedding size, hidden width, number of
  layers) alter the architecture. They must run as **concurrent trials** with
  memory-aware admission control.

Existing tools make no such distinction. `optuna-saturate` classifies the
search space, plans the execution, and keeps the device busy.

## Status

Early development. The vectorisation layer is implemented and covered by a
numerical parity test: training K configurations as one batched model reproduces
K independent trainings to within `atol=1e-5`. A GPU profiler reports both axes
of device under-use — time with no kernel resident, and kernel time against wall
time — together with energy draw and peak memory. The concurrent-trial scheduler
and the native core are not implemented yet.

## Installation

```bash
pip install optuna-saturate          # library
pip install "optuna-saturate[gpu]"   # plus NVML bindings for profiling
```

PyTorch must be a CUDA build for anything to run on the device.

## Usage

Declare which hyperparameters leave tensor shapes untouched, and `saturate`
trains them as one batched model:

```python
import optuna
import optuna_saturate as osat

@osat.vectorizable(over=["learning_rate"])
def objective(trial, ctx):
    hidden = trial.suggest_int("hidden", 16, 64)                     # one per group
    lr = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)  # one per member

    ensemble = ctx.stack([Net(hidden) for _ in range(ctx.k)])
    optimiser = ctx.sgd(ensemble, lr=lr)
    for x, y in train_loader:
        ctx.step(ensemble, optimiser, x, y)
    return ctx.accuracy(ensemble, val_loader)

study = optuna.create_study(direction="maximize")
osat.saturate(study, objective, n_trials=200, group_size=8, device="cuda:0")
```

Trials are recorded exactly as `study.optimize` records them, so
`optuna-dashboard` and the rest of the Optuna ecosystem work unchanged. A
complete runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Results

Fashion-MNIST, a small convolutional network, 24 trials over a mixed search
space (`learning_rate` and `dropout` vectorised, `width` not), averaged over
3 seeds:

| Arm | Wall time | Speedup | Best accuracy | Energy |
|---|---|---|---|---|
| Optuna, sequential | 30.6 s | 1.00× | 0.8050 | 1041 J |
| Optuna, `n_jobs=8` | 71.6 s | 0.43× | 0.7996 | 2110 J |
| `saturate`, group 8 | **4.7 s** | **6.47×** | 0.8006 | **323 J** |

Two things are worth reading carefully. `n_jobs=8` is *slower* than running
trials one at a time: `study.optimize` parallelises with threads, which contend
for the GIL and for one device. And `saturate` matches the sequential arm's
accuracy to within 0.44 percentage points — less than the spread between seeds —
despite exploring fewer distinct architectures, because a group shares its
shape-changing hyperparameters.

The share of wall time spent executing CUDA kernels rises from 8.9% to 36.9%.
The speedup comes from filling that idle time, not from doing less work.

Measured on an NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB, driver 596.53)
with PyTorch 2.13.0+cu130. Three seeds are too few for confidence intervals.
Reproduce with:

```bash
pip install -e ".[dev,gpu,bench]"
python -m benchmarks.fashion_mnist --trials 24 --group-size 8 --seeds 3
```

### Rules and limits

- **Declare `over` honestly.** A hyperparameter that changes any tensor shape
  cannot be vectorised. Declaring one raises
  `ShapeChangingHyperparameterError` naming the offender, rather than computing
  something wrong. Integer hyperparameters are rejected outright: an integer in
  a search space almost always sizes a tensor.
- **Call every `suggest_*` before `ctx.stack(...)`.** The library discovers the
  search space by running the objective up to that point, so a hyperparameter
  declared after it is invisible.
- **Per-member values go through `member_hp`, not the model constructor.** The
  group shares one flattened set of parameters; a per-member dropout probability
  is applied inside the forward pass.
- **`BatchNorm` is not supported.** Running statistics are buffers, and buffer
  updates do not propagate back out of the vectorised call. Use a network
  without batch normalisation, or run those trials unvectorised.
- **`batch_size` cannot be vectorised.** The group shares one batch.
- Groups run one after another; concurrent trials for shape-changing
  hyperparameters are not implemented yet.

## Licence

MIT
