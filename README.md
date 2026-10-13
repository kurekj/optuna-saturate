# optuna-saturate

Train hyperparameter configurations as one batched model so that a single GPU
stops idling during an Optuna study.

Hyperparameter search spaces split into two classes that need different
execution strategies:

- **Shape-preserving** hyperparameters (learning rate, momentum, weight decay,
  dropout, temperature, loss weights) do not change tensor shapes. K such
  configurations can be trained **as a single batched model** via
  `torch.func.stack_module_state` and `vmap`.
- **Shape-changing** hyperparameters (embedding size, hidden width, number of
  layers) alter the architecture. They must run as separate trials.

Existing tools make no such distinction. `optuna-saturate` classifies the
search space, trains each shape-preserving group as one model, and reports
every member back to Optuna as an ordinary trial.

## Installation

```bash
pip install "optuna-saturate @ git+https://github.com/kurekj/optuna-saturate@v0.2.0"
pip install "optuna-saturate[gpu] @ git+https://github.com/kurekj/optuna-saturate@v0.2.0"   # plus NVML bindings for profiling
```

PyTorch must be a CUDA build for anything to run on the device.

## Usage

Declare which hyperparameters leave tensor shapes untouched, and `saturate`
trains them as one batched model:

```python
import optuna
import optuna_saturate as osat

@osat.vectorizable(over=["learning_rate", "weight_decay", "dropout"])
def objective(trial, ctx):
    hidden = trial.suggest_int("hidden", 16, 64)                       # one per group
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)    # one per member
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)     # one per member
    p = trial.suggest_float("dropout", 0.0, 0.5)                       # one per member

    ensemble = ctx.stack([Net(hidden) for _ in range(ctx.k)])
    optimiser = ctx.adamw(ensemble, lr=lr, weight_decay=wd)
    for x, y in train_loader:
        ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": p})
    return ctx.accuracy(ensemble, val_loader)

study = optuna.create_study(direction="maximize")
osat.saturate(study, objective, n_trials=200, group_size=8, device="cuda:0")
```

Trials are recorded exactly as `study.optimize` records them, so
`optuna-dashboard` and the rest of the Optuna ecosystem work unchanged. A
complete runnable version is in [`examples/quickstart.py`](examples/quickstart.py);
[`examples/adamw_mixed_precision.py`](examples/adamw_mixed_precision.py) adds
AdamW, per-member weight decay and bfloat16 autocast.

### Optimisers

`ctx.sgd`, `ctx.adam` and `ctx.adamw` mirror `torch.optim.SGD`, `Adam` and
`AdamW`. Every scalar hyperparameter of the reference optimiser (learning rate,
momentum, weight decay, the Adam betas) may be a float shared by the group or a
`[K]` tensor holding one value per member, which is what the trial returns for
a name listed in `over`. Nesterov momentum is supported. The update rules follow
`torch.optim` operation for operation; see below for how closely.

### Mixed precision

`saturate(..., autocast_dtype=torch.bfloat16)` runs the forward pass and the
loss under `torch.autocast`; parameters and optimiser state stay in float32, as
they do in an ordinary autocast training loop.

### Choosing the group size

Memory grows linearly with K while the speedup saturates once the device stops
idling. `osat.recommend_group_size()` fits the line through measured peaks and
returns the largest K that fits the memory budget and, when speedups are
supplied, lies before the plateau:

```python
k = osat.recommend_group_size(
    group_sizes=[2, 4, 8],
    peak_bytes=[peak2, peak4, peak8],      # profiler's peak_vram_torch_bytes
    device_bytes=torch.cuda.mem_get_info()[1],
    speedups=[0.9, 1.5, 2.2],
)
```

### What of Optuna keeps working

| Supported (covered by tests) | Not available inside a group |
|---|---|
| Samplers driven through ask-and-tell: TPE, random, Gaussian process, CMA-ES are tested | `GridSampler` (refuses the partially specified trials a group is grown from) |
| Storages: in-memory, RDB and journal are tested | Pruners: `trial.report` / `trial.should_prune` raise with an explanation |
| `callbacks` and `timeout`, as on `study.optimize`; callbacks run once the whole group is recorded; raise `StopSaturate` from a callback to stop | `study.stop()`: Optuna allows it only inside `study.optimize` |
| `trial.set_user_attr` for the group, `trial.set_member_attr` for one value per member | Multi-objective studies: rejected up front |
| `n_jobs`: the group is the unit of parallelism | |
| `study.best_trial`, `trials_dataframe`, `optuna-dashboard` | Per-member batch sizes: the group shares one batch; conditional search spaces: the probe follows one path |

## Numerical parity

The claim that a batched group reproduces K independent trainings is tested. On CPU, in continuous integration (`atol=1e-5`, `rtol=1e-4` is the
tolerance of `torch.testing.assert_close`, checked on every parameter after 6-25 steps):

| Setting | Agreement with independent training |
|---|---|
| SGD, per-member learning rate and temperature, MLP and CNN | `atol=1e-5`, `rtol=1e-4` |
| SGD with per-member momentum, weight decay, Nesterov | `atol=1e-5`, `rtol=1e-4` |
| Adam and AdamW with per-member betas and weight decay | `atol=1e-5`, `rtol=1e-4` |
| Transformer encoder (attention, LayerNorm, GELU) with AdamW | `atol=1e-5`, `rtol=1e-4` |
| A parameter that first receives a gradient on a later step (SGD, Adam, AdamW) | `atol=1e-5`, `rtol=1e-4` |
| bfloat16 autocast, SGD and Adam | batching gap within 1.5x the bf16-vs-fp32 gap of independent training (equal to rounding for SGD, several times smaller for Adam) |
| Stochastic dropout | mean training loss per member agrees within 3 pooled standard errors over 24 replicates per path; the same rule rejects a member trained at the wrong rate |

## Evaluation

The scripts in `benchmarks/` compare `saturate` with five other ways of running
a study on one device: sequential `study.optimize`, thread-based `n_jobs`,
threads on separate CUDA streams, process-based workers with memory admission
control, and Ray Tune. Three tasks cover three model families:

| Task | Dataset | Model | Vectorised | Shared |
|---|---|---|---|---|
| `fashion_cnn` | Fashion-MNIST | two-block CNN, SGD | learning rate, dropout | width |
| `kmnist_mlp` | KMNIST | three-layer MLP, SGD with momentum | learning rate, momentum, dropout, weight decay | hidden |
| `cifar_vit` | CIFAR-10 | small Transformer encoder, AdamW | learning rate, weight decay, dropout | embedding size |

Reproduce with:

```bash
pip install -e ".[dev,gpu,bench,raytune]"   # raytune only for the Ray Tune arm
python -m benchmarks.compare_arms --task fashion_cnn --trials 24 --group-sizes 2 4 8 16 32 --seeds 5
python -m benchmarks.compare_arms --task fashion_cnn --schedule replay --group-sizes 8 --seeds 5
python benchmarks/vectorization_gain.py --k 2 4 8 16 32 64
python benchmarks/kernel_overlap.py --group-size 64 --trials 128   # the kernel column as an upper bound
```

`--schedule replay` hands every arm the same fixed list of configurations, so
the comparison is of execution alone; the default lets the seeded TPE sampler
propose, as a study would in practice. For the multi-process arms the kernel
time sums every process's kernel seconds over the wall time and the memory
figure sums the processes' own peaks; both are upper bounds, flagged in the
CSV, rather than one measured trace.

`results/` holds the raw per-seed measurements behind every table and figure
of the accompanying article; `run_all_benchmarks.sh` is the exact sequence
that produced them. Energy figures are the GPU board's own draw from NVML's
cumulative energy counter, reported with the mean power over the run; host
energy is not included.

### Rules and limits

- **Declare `over` honestly.** A hyperparameter that changes any parameter shape
  cannot be vectorised. Declaring one raises
  `ShapeChangingHyperparameterError` naming the offender, rather than computing
  something wrong. The check sees parameter shapes only: a hyperparameter that
  changes the computation without changing a shape (a head count over a fixed
  embedding, say) is not caught and must not be declared. Integer hyperparameters
  are rejected outright: an integer in a search space almost always sizes a tensor.
- **Call every `suggest_*` before `ctx.stack(...)`.** The library discovers the
  search space by running the objective up to that point, so a hyperparameter
  declared after it is invisible.
- **Per-member values go through `member_hp` or the optimiser, not the model
  constructor.** The group shares one flattened set of parameters; a per-member
  dropout probability is applied inside the forward pass.
- **`BatchNorm` is not supported in this release.** The stacked buffers are handed
  to `functional_call`, but updated running statistics are not written back, so a
  batch-normalised member would evaluate with the statistics it started with.
  LayerNorm and GroupNorm carry parameters only and work.
- **`batch_size` cannot be vectorised.** The group shares one batch.
- Groups run one after another; concurrent execution of shape-changing trials
  is not implemented.

## Licence

MIT
