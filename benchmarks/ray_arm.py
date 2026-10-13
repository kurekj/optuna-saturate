"""Ray Tune as a baseline: a practical scheduler running K trials concurrently.

Each trial runs in a Ray actor holding its own CUDA context and granted ``1/K``
of the GPU, which is how Ray Tune is normally configured to pack small trials
onto one device. Actors are reused across trials and keep their data loaders,
so a trial pays for its own training and for Ray's scheduling, not for loading
the dataset again. The search algorithm is Ray's wrapper around an Optuna
sampler, seeded like the other arms; concurrent execution can still change the
sampler's history, which is what the replay schedule is for: under it the arm
uses the same schedule sampler as the others.

Ray is an optional dependency of the benchmarks: ``pip install "ray[tune]"``.
"""

from __future__ import annotations

from typing import Any

import optuna
import torch

from benchmarks.tasks import TASKS, Task
from optuna_saturate.runtime.profile import GpuProfile, profile_gpu

_LOADERS: dict[tuple[Any, ...], Any] = {}


def _trainable(config: dict[str, Any]) -> None:
    from ray import tune
    from torch.profiler import ProfilerActivity, profile

    from optuna_saturate.runtime.profile import _kernel_time_seconds

    task = TASKS[config["task"]]
    key = (task.name, config["batch_size"], config["train_subset"], config["valid_subset"])
    if key not in _LOADERS:  # once per actor, since actors are reused
        _LOADERS[key] = task.load(
            config["batch_size"],
            train_subset=config["train_subset"],
            valid_subset=config["valid_subset"],
        )
    train_loader, valid_loader = _LOADERS[key]
    objective = task.plain_objective(train_loader, valid_loader, config["epochs"])

    # Ray hands the sampled values back as a config; a fixed trial replays them
    # through the ordinary objective so that the model code is shared verbatim.
    params = {name: config[name] for name in task.distributions}
    trial = optuna.trial.FixedTrial(params)
    torch.cuda.init()  # the memory statistics need a live context
    torch.cuda.reset_peak_memory_stats(0)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        accuracy = objective(trial)
        torch.cuda.synchronize()
    tune.report(
        {
            "accuracy": accuracy,
            "kernel_s": _kernel_time_seconds(prof),
            "peak_torch": int(torch.cuda.max_memory_allocated(0)),
        }
    )


def run_ray_arm(
    task: Task,
    seed: int,
    trials: int,
    concurrency: int,
    max_concurrency: int,
    epochs: int,
    train_subset: int | None,
    valid_subset: int | None,
    batch_size: int,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> tuple[GpuProfile, float, int, int, float | None, int]:
    """Run ``trials`` trials through Ray Tune, ``concurrency`` at a time.

    Args:
        max_concurrency: The largest concurrency of the whole sweep, so that
            Ray is started once with enough CPU slots for every arm.
        sampler: The Optuna sampler to drive the search; the seeded TPE
            sampler by default, or a replay schedule.

    Returns:
        The device profile, best accuracy, completed trials, distinct
        architectures, the kernel seconds of all trials over the wall time, and
        the sum of the ``concurrency`` largest per-trial peak allocations, which
        bounds the simultaneous total from above.
    """
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    search = OptunaSearch(
        space=task.distributions,
        metric="accuracy",
        mode="max",
        sampler=sampler or optuna.samplers.TPESampler(seed=seed),
    )
    config = {
        "task": task.name,
        "batch_size": batch_size,
        "train_subset": train_subset,
        "valid_subset": valid_subset,
        "epochs": epochs,
    }
    if not ray.is_initialized():
        ray.init(
            num_cpus=max(2, max_concurrency),
            num_gpus=1,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
        )

    trainable = tune.with_resources(
        tune.with_parameters(_trainable),
        {"cpu": 1, "gpu": 1.0 / concurrency},
    )
    tuner = tune.Tuner(
        trainable,
        param_space=config,
        tune_config=tune.TuneConfig(
            search_alg=search,
            num_samples=trials,
            max_concurrent_trials=concurrency,
            reuse_actors=True,
            # Short directory names: the default ones exceed Windows' path limit.
            trial_dirname_creator=lambda trial: trial.trial_id,
        ),
        # Retry a trial whose actor dies, up to three times.
        run_config=tune.RunConfig(verbose=0, failure_config=tune.FailureConfig(max_failures=3)),
    )
    with profile_gpu(interval_s=0.05, kernel_profiling=False) as prof:
        results = tuner.fit()

    frame = results.get_dataframe()
    done = frame[frame["accuracy"].notna()]
    shared_columns = [f"config/{name}" for name in task.shared]
    distinct = len(done[shared_columns].drop_duplicates())
    wall = prof.report.wall_time_s
    kernel = float(done["kernel_s"].sum()) / wall if wall > 0 else None
    peak = int(done["peak_torch"].nlargest(concurrency).sum())
    return prof.report, float(done["accuracy"].max()), int(len(done)), distinct, kernel, peak


def shutdown() -> None:
    """Release Ray's workers so that later arms have the device to themselves."""
    import ray

    if ray.is_initialized():
        ray.shutdown()
