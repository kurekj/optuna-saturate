"""Compare saturate() with the other ways of running an Optuna study on one GPU.

Every arm receives the same task, the same number of trials and the same seed.

    sequential   study.optimize(objective, n_trials=N)
    n_jobs       study.optimize(objective, n_trials=N, n_jobs=K): threads
    streams      as n_jobs, each thread issuing its work on its own CUDA stream
                 from a pool of K streams
    process      K worker processes sharing one study through a journal file,
                 admitted up to what the device memory allows
    ray          Ray Tune with the Optuna search algorithm, K concurrent trials
                 each granted 1/K of the GPU
    saturate     osat.saturate(study, objective, group_size=K), for every K in
                 --group-sizes
    saturate_equal_time
                 saturate() given the wall-clock time the sequential arm took on
                 the same seed; only groups finished inside that budget count

Every row of the result (one arm at one K on one seed) runs in a fresh
process, so that nothing an arm leaves behind in the CUDA allocator, in Ray or
in a thread pool can affect the next. Before its clock starts, each row loads
its data, creates its CUDA context and runs a short warm-up so that the device
has left its idle power state; the clock and temperature the device actually
ran at are sampled during the row and recorded with it.

Two schedules are available. ``--schedule tpe`` (the default) lets the seeded
TPE sampler propose configurations, as a study would in practice; the arms
then differ in their sampling history as well as in execution. ``--schedule
replay`` draws one fixed list of configurations per seed and hands it to every
arm by trial number, so that all arms train exactly the same list and the
comparison is of execution alone.

Run from the repository root, as a module so that ``benchmarks`` is importable:

    python -m benchmarks.compare_arms --task fashion_cnn --trials 24 \
        --group-sizes 8 --seeds 3 --arms sequential n_jobs saturate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
import torch

import optuna_saturate as osat
from benchmarks.schedule import ScheduleSampler, make_schedule
from benchmarks.tasks import DEVICE, TASKS, Task
from optuna_saturate.runtime.profile import GpuProfile, profile_gpu

BATCH_SIZE = 128
ALL_ARMS = ("sequential", "n_jobs", "streams", "process", "ray", "saturate", "saturate_equal_time")
MIB = 1024**2
WARMUP_S = 3.0


@dataclass(frozen=True)
class RowSpec:
    """Everything one row needs, picklable so that a fresh process can run it."""

    task: str
    arm: str
    group_size: int | None
    seed: int
    trials: int
    schedule_kind: str
    replay_block: int
    epochs: int
    train_subset: int | None
    valid_subset: int | None
    context_mib: int
    max_group_size: int
    workdir: str
    sequential_wall: float | None
    sequential_peak: int | None
    kernel_profiling: bool = True
    loader_workers: int = 0


def _sampler(seed: int, schedule: list[dict[str, Any]] | None) -> optuna.samplers.BaseSampler:
    return ScheduleSampler(schedule) if schedule else optuna.samplers.TPESampler(seed=seed)


def _study(
    seed: int,
    schedule: list[dict[str, Any]] | None = None,
    storage: Any = None,
    name: str | None = None,
) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="maximize",
        sampler=_sampler(seed, schedule),
        storage=storage,
        study_name=name,
        load_if_exists=storage is not None,
    )


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def _distinct_architectures(trials: list[optuna.trial.FrozenTrial], shared: tuple[str, ...]) -> int:
    """How many different shape-changing configurations were evaluated."""
    return len({tuple(t.params.get(name) for name in shared) for t in trials})


def _best(trials: list[optuna.trial.FrozenTrial]) -> float:
    return max(float(t.value) for t in trials) if trials else math.nan


def _warm_up(seconds: float = WARMUP_S) -> None:
    """Keep the device busy briefly so that it leaves its idle power state.

    Every arm gets the same warm-up, outside its measured block.
    """
    x = torch.randn(2048, 2048, device=DEVICE)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        x = x @ x * 1e-3
    torch.cuda.synchronize()
    del x


# --- the thread-based arms ----------------------------------------------------


def _with_stream_pool(objective: Any, pool_size: int) -> Any:
    """Wrap a plain objective so that each thread issues its trials on its own stream.

    The streams are a fixed pool, one per thread, reused across trials. A fresh
    stream per trial would leak: blocks freed on a stream are reclaimed only
    when something next allocates on that stream, which never happens to a
    stream nobody uses again.
    """
    pool = [torch.cuda.Stream() for _ in range(pool_size)]
    counter = itertools.count()
    local = threading.local()

    def on_stream(trial: optuna.Trial) -> float:
        if not hasattr(local, "stream"):
            local.stream = pool[next(counter) % pool_size]
        with torch.cuda.stream(local.stream):
            value = objective(trial)
        local.stream.synchronize()
        return value

    return on_stream


# --- the process-based arm ----------------------------------------------------


def _journal_storage(path: Path) -> optuna.storages.JournalStorage:
    # The open-file lock works on every platform; the default symlink lock does not.
    lock = optuna.storages.journal.JournalFileOpenLock(str(path))
    backend = optuna.storages.journal.JournalFileBackend(str(path), lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


def _process_worker(
    task_name: str,
    seed: int,
    index: int,
    trials: int,
    journal: str,
    schedule: list[dict[str, Any]] | None,
    epochs: int,
    train_subset: int | None,
    valid_subset: int | None,
    kernel_profiling: bool,
    loader_workers: int,
    ready: Any,
    go: Any,
    queue: Any,
) -> None:
    """One worker: its own CUDA context, its own loaders, a share of the trials.

    Setup (imports, data, context) happens before the ``ready`` barrier; the
    parent then warms the device up and releases the ``go`` barrier as it
    starts its clock, so the measured time is the study's, as it is for the
    in-process arms whose loaders are built before timing.
    """
    import traceback

    from torch.profiler import ProfilerActivity, profile

    from optuna_saturate.runtime.profile import _kernel_time_seconds

    try:
        task = TASKS[task_name]
        train_loader, valid_loader = task.load(
            BATCH_SIZE,
            train_subset=train_subset,
            valid_subset=valid_subset,
            num_workers=loader_workers,
        )
        # Under TPE each worker needs its own seed, or they would all propose
        # the same start-up points; under a schedule the trial number decides.
        storage = _journal_storage(Path(journal))
        study = _study(seed * 1000 + index, schedule, storage=storage, name="shared")
        objective = task.plain_objective(train_loader, valid_loader, epochs)
        torch.cuda.init()
        torch.zeros(1, device=DEVICE)  # a live context before the clock starts
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        ready.wait()
        go.wait()
        if kernel_profiling:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                study.optimize(objective, n_trials=trials)
                torch.cuda.synchronize()
            kernel_seconds: float | None = _kernel_time_seconds(prof)
        else:
            # The kernel trace of a long study is too large to hold; the other
            # axes are still measured by the parent's NVML sampler.
            study.optimize(objective, n_trials=trials)
            torch.cuda.synchronize()
            kernel_seconds = None
        queue.put(("ok", kernel_seconds, int(torch.cuda.max_memory_allocated(0))))
    except BaseException:
        queue.put(("error", traceback.format_exc(), 0))
        raise


def _collect(queue: Any, processes: list[Any]) -> list[tuple[float | None, int]]:
    """Gather one report per worker, failing fast if a worker dies silently."""
    import queue as queue_module

    results: list[tuple[float | None, int]] = []
    while len(results) < len(processes):
        try:
            status, kernel_seconds, peak = queue.get(timeout=5.0)
        except queue_module.Empty:
            if not any(p.is_alive() for p in processes):
                raise RuntimeError("a worker process exited without reporting") from None
            continue
        if status != "ok":
            raise RuntimeError(f"worker process failed: {kernel_seconds}")
        results.append((None if kernel_seconds is None else float(kernel_seconds), int(peak)))
    return results


def _admitted_workers(requested: int, per_trial_bytes: int, context_bytes: int) -> int:
    """Admission control: as many workers as the free device memory allows.

    ``per_trial_bytes`` is the peak PyTorch allocation of the sequential arm on
    the same seed, which is the calibration a practitioner has to hand; it is a
    sampled maximum, not a guarantee, and the 20% headroom is for what it does
    not cover: allocator reservations and the largest architecture a parallel
    sampler may still propose.

    Raises:
        RuntimeError: Not even one worker fits.
    """
    free, _total = torch.cuda.mem_get_info(0)
    per_worker = per_trial_bytes + context_bytes
    fits = int(0.8 * free // per_worker)
    if fits < 1:
        raise RuntimeError(
            f"no worker fits: {per_worker / MIB:.0f} MiB per worker against "
            f"{free / MIB:.0f} MiB free"
        )
    return min(requested, fits)


def run_process_arm(
    task: Task,
    seed: int,
    trials: int,
    workers: int,
    schedule: list[dict[str, Any]] | None,
    epochs: int,
    train_subset: int | None,
    valid_subset: int | None,
    workdir: Path,
    kernel_profiling: bool = True,
    loader_workers: int = 0,
) -> tuple[GpuProfile, optuna.Study, float | None, int]:
    """Spawn ``workers`` processes on one shared study.

    Returns:
        The device profile, the shared study, the workers' kernel seconds
        summed over the wall time, and the sum of the workers' peak PyTorch
        allocations (an upper bound on the simultaneous total).
    """
    journal = workdir / f"process_{task.name}_{seed}.log"
    if journal.exists():
        journal.unlink()
    _study(seed, schedule, storage=_journal_storage(journal), name="shared")

    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    shares = [trials // workers + (1 if i < trials % workers else 0) for i in range(workers)]
    shares = [share for share in shares if share > 0]
    ready = context.Barrier(len(shares) + 1)
    go = context.Barrier(len(shares) + 1)
    processes = [
        context.Process(
            target=_process_worker,
            args=(
                task.name,
                seed,
                i,
                share,
                str(journal),
                schedule,
                epochs,
                train_subset,
                valid_subset,
                kernel_profiling,
                loader_workers,
                ready,
                go,
                queue,
            ),
        )
        for i, share in enumerate(shares)
    ]
    for process in processes:
        process.start()
    # A worker that dies during setup never reaches the barrier; the timeout
    # turns that into a failed row instead of a hang.
    ready.wait(timeout=900.0)  # every worker has its data and its CUDA context
    _warm_up()

    # The parent launches no kernels, so only the NVML axes are measured here;
    # the kernel trace comes back from each worker.
    with profile_gpu(interval_s=0.05, kernel_profiling=False) as prof:
        go.wait(timeout=60.0)
        results = _collect(queue, processes)
        for process in processes:
            process.join()

    peak_torch = sum(m for _, m in results)
    wall = prof.report.wall_time_s
    kernel: float | None = None
    if kernel_profiling and wall > 0:
        kernel = sum(k or 0.0 for k, _ in results) / wall
    study = optuna.load_study(study_name="shared", storage=_journal_storage(journal))
    return prof.report, study, kernel, peak_torch


# --- the in-process arms --------------------------------------------------------


def _run_study_arm(
    task: Task,
    arm: str,
    seed: int,
    trials: int,
    group_size: int,
    schedule: list[dict[str, Any]] | None,
    train_loader: Any,
    valid_loader: Any,
    epochs: int,
    timeout: float | None = None,
    kernel_profiling: bool = True,
) -> tuple[GpuProfile, optuna.Study, dict[int, float]]:
    """Run one in-process arm.

    Returns:
        The profile, the study, and the time at which each trial was recorded,
        in seconds from the start of the measured block.
    """
    torch.manual_seed(seed)
    study = _study(seed, schedule)
    plain = task.plain_objective(train_loader, valid_loader, epochs)
    recorded: dict[int, float] = {}
    _warm_up()
    started = time.perf_counter()

    def stamp(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        recorded[trial.number] = time.perf_counter() - started

    with profile_gpu(interval_s=0.05, kernel_profiling=kernel_profiling) as prof:
        if arm == "sequential":
            study.optimize(plain, n_trials=trials, callbacks=[stamp])
        elif arm == "n_jobs":
            study.optimize(plain, n_trials=trials, n_jobs=group_size, callbacks=[stamp])
        elif arm == "streams":
            study.optimize(
                _with_stream_pool(plain, group_size),
                n_trials=trials,
                n_jobs=group_size,
                callbacks=[stamp],
            )
        elif arm in ("saturate", "saturate_equal_time"):
            osat.saturate(
                study,
                task.vectorised_objective(train_loader, valid_loader, epochs),
                n_trials=trials,
                group_size=group_size,
                device=DEVICE,
                timeout=timeout,
                callbacks=[stamp],
            )
        else:
            raise ValueError(f"unknown arm {arm!r}")
        torch.cuda.synchronize()
    return prof.report, study, recorded


# --- one row of results --------------------------------------------------------


def format_row(
    task: Task,
    spec: RowSpec,
    workers: int | None,
    profile: GpuProfile,
    trials: list[optuna.trial.FrozenTrial],
    kernel_time_fraction: float | None = None,
    peak_torch_bytes: int | None = None,
    budget_s: float | None = None,
) -> dict[str, object]:
    multi_process = peak_torch_bytes is not None
    kernel = kernel_time_fraction if multi_process else profile.kernel_time_fraction
    peak = peak_torch_bytes if multi_process else profile.peak_vram_torch_bytes
    return {
        "task": task.name,
        "arm": spec.arm,
        "group_size": spec.group_size,
        "workers": workers,
        "seed": spec.seed,
        "schedule": spec.schedule_kind,
        "trials_completed": len(trials),
        "distinct_architectures": _distinct_architectures(trials, task.shared),
        "wall_time_s": round(profile.wall_time_s, 3),
        "budget_s": None if budget_s is None else round(budget_s, 3),
        "best_accuracy": round(_best(trials), 4),
        "busy_fraction": round(profile.busy_fraction, 4),
        "kernel_time_fraction": None if kernel is None else round(kernel, 4),
        # For the multi-process arms the kernel figure sums every process's
        # kernel seconds over the wall time, and the torch peak sums the
        # processes' own peaks: both are upper bounds, not one measured trace.
        "multi_process_bounds": multi_process,
        "peak_vram_nvml_mib": profile.peak_vram_nvml_bytes // MIB,
        "peak_vram_torch_mib": None if peak is None else peak // MIB,
        "energy_j": None if profile.energy_j is None else round(profile.energy_j, 1),
        "mean_power_w": None if profile.mean_power_w is None else round(profile.mean_power_w, 1),
        "mean_sm_clock_mhz": round(profile.mean_sm_clock_mhz),
        "max_temperature_c": profile.max_temperature_c,
        "epochs": spec.epochs,
        "train_subset": spec.train_subset,
        "loader_workers": spec.loader_workers,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def run_row(spec: RowSpec) -> dict[str, object]:
    """Run one row in the current process. ``_row_worker`` calls this in a fresh one."""
    task = TASKS[spec.task]
    schedule = None
    if spec.schedule_kind == "replay":
        schedule = make_schedule(
            task.distributions, task.shared, spec.trials, spec.replay_block, spec.seed
        )
    k = spec.group_size
    workdir = Path(spec.workdir)

    if spec.arm == "process":
        assert k is not None and spec.sequential_peak is not None
        workers = _admitted_workers(k, spec.sequential_peak, spec.context_mib * MIB)
        profile, study, kernel, peak = run_process_arm(
            task,
            spec.seed,
            spec.trials,
            workers,
            schedule,
            spec.epochs,
            spec.train_subset,
            spec.valid_subset,
            workdir,
            spec.kernel_profiling,
            spec.loader_workers,
        )
        return format_row(task, spec, workers, profile, _completed(study), kernel, peak)

    if spec.arm == "ray":
        from benchmarks.ray_arm import run_ray_arm, shutdown

        assert k is not None
        try:
            _warm_up()
            profile, best, completed, distinct, kernel, peak = run_ray_arm(
                task,
                spec.seed,
                spec.trials,
                k,
                spec.max_group_size,
                spec.epochs,
                spec.train_subset,
                spec.valid_subset,
                BATCH_SIZE,
                sampler=ScheduleSampler(schedule) if schedule else None,
            )
        finally:
            shutdown()
        row = format_row(task, spec, k, profile, [], kernel, peak)
        row.update(
            trials_completed=completed,
            distinct_architectures=distinct,
            best_accuracy=round(best, 4),
        )
        return row

    train_loader, valid_loader = task.load(
        BATCH_SIZE,
        train_subset=spec.train_subset,
        valid_subset=spec.valid_subset,
        num_workers=spec.loader_workers,
    )
    timeout = None
    trials = spec.trials
    budget = None
    if spec.arm == "saturate_equal_time":
        assert spec.sequential_wall is not None
        timeout, trials, budget = spec.sequential_wall, 10**6, spec.sequential_wall
    profile, study, recorded = _run_study_arm(
        task,
        spec.arm,
        spec.seed,
        trials,
        k or 1,
        schedule,
        train_loader,
        valid_loader,
        spec.epochs,
        timeout,
        kernel_profiling=spec.kernel_profiling,
    )
    completed = _completed(study)
    if budget is not None:
        # Only what was recorded inside the budget counts; a group that started
        # just before the deadline and finished after it is excluded, and the
        # overshoot is visible as wall_time_s minus budget_s.
        completed = [t for t in completed if recorded.get(t.number, math.inf) <= budget]
    return format_row(task, spec, k, profile, completed, budget_s=budget)


def _row_worker(spec_dict: dict[str, Any], queue: Any) -> None:
    import traceback

    try:
        queue.put(("ok", run_row(RowSpec(**spec_dict))))
    except BaseException:
        queue.put(("error", traceback.format_exc()))
        raise


def _run_once_isolated(spec: RowSpec, deadline_s: float) -> dict[str, object]:
    import queue as queue_module

    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_row_worker, args=(asdict(spec), queue))
    process.start()
    started = time.perf_counter()
    try:
        while True:
            try:
                status, payload = queue.get(timeout=5.0)
                break
            except queue_module.Empty:
                if not process.is_alive():
                    raise RuntimeError("the row's process exited without reporting") from None
                if time.perf_counter() - started > deadline_s:
                    raise RuntimeError(
                        f"the row did not finish within {deadline_s:.0f} s"
                    ) from None
    finally:
        if process.is_alive():
            process.kill()
        process.join()
    if status != "ok":
        raise RuntimeError(f"row failed:\n{payload}")
    assert isinstance(payload, dict)
    return payload


def run_row_isolated(spec: RowSpec, deadline_s: float, retries: int = 1) -> dict[str, object]:
    """Run one row in a fresh process and return its result.

    A row that fails or hangs (a scheduler that never starts its workers, say)
    is killed and retried once before the whole sweep gives up on it.
    """
    label = f"{spec.arm} K={spec.group_size} seed={spec.seed}"
    for attempt in range(retries + 1):
        try:
            return _run_once_isolated(spec, deadline_s)
        except RuntimeError as error:
            if attempt == retries:
                raise RuntimeError(f"row {label} failed after {retries + 1} attempts") from error
            print(f"row {label} failed ({str(error).splitlines()[0]}); retrying", flush=True)
            time.sleep(5.0)
    raise AssertionError("unreachable")


def _read_rows(path: Path) -> list[dict[str, object]]:
    """Rows of an earlier run, with the fields the loop needs typed again."""

    def typed(value: str) -> object:
        if value == "":
            return None
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return {"True": True, "False": False}.get(value, value)

    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: typed(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _summarise(rows: list[dict[str, object]]) -> None:
    """Mean over seeds per (arm, K), with speedup against the sequential arm."""
    keys = sorted({(str(r["arm"]), r["group_size"]) for r in rows}, key=str)

    def mean(arm: str, k: object, field: str) -> float:
        values = [
            float(r[field])  # type: ignore[arg-type]
            for r in rows
            if r["arm"] == arm and r["group_size"] == k and r[field] is not None
        ]
        return sum(values) / len(values) if values else math.nan

    base_time = mean("sequential", None, "wall_time_s")
    base_acc = mean("sequential", None, "best_accuracy")
    print(
        f"\n{'arm':<22}{'K':>4} {'wall(s)':>9} {'speedup':>9} {'best acc':>9} "
        f"{'delta':>8} {'kernel':>7} {'J':>7} {'W':>6} {'MHz':>5} {'archs':>6} {'trials':>7}"
    )
    for arm, k in keys:
        wall = mean(arm, k, "wall_time_s")
        acc = mean(arm, k, "best_accuracy")
        print(
            f"{arm:<22}{'' if k is None else k:>4} {wall:9.2f} {base_time / wall:8.2f}x "
            f"{acc:9.4f} {acc - base_acc:+8.4f} {mean(arm, k, 'kernel_time_fraction'):7.1%} "
            f"{mean(arm, k, 'energy_j'):7.0f} {mean(arm, k, 'mean_power_w'):6.1f} "
            f"{mean(arm, k, 'mean_sm_clock_mhz'):5.0f} "
            f"{mean(arm, k, 'distinct_architectures'):6.1f} {mean(arm, k, 'trials_completed'):7.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="fashion_cnn")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-subset", type=int, default=8192)
    parser.add_argument("--valid-subset", type=int, default=2048)
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    parser.add_argument(
        "--schedule",
        choices=("tpe", "replay"),
        default="tpe",
        help="tpe: the seeded sampler proposes; replay: one fixed list of configurations "
        "per seed, in blocks of --replay-block sharing the shape-changing values",
    )
    parser.add_argument("--replay-block", type=int, default=8)
    parser.add_argument(
        "--context-mib",
        type=int,
        default=600,
        help="memory a worker's CUDA context costs, for the process arm's admission control",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader worker processes per loader (persistent); 0 loads in the main thread",
    )
    parser.add_argument(
        "--no-kernel-profiling",
        action="store_true",
        help="skip the CUDA kernel trace; for long studies whose trace would not fit in memory",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="seconds to leave the device idle between rows",
    )
    parser.add_argument(
        "--row-deadline",
        type=float,
        default=1800.0,
        help="seconds after which a row is killed and retried",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the rows already in --out and run only the missing ones",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=Path("results/tmp"))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "No CUDA device available. This benchmark measures GPU under-use and "
            "cannot run on CPU.",
            file=sys.stderr,
        )
        return 1
    needs_sequential = {"process", "saturate_equal_time"} & set(args.arms)
    if needs_sequential and (
        "sequential" not in args.arms
        or any(args.arms.index(a) < args.arms.index("sequential") for a in needs_sequential)
    ):
        print(
            "the process and saturate_equal_time arms need the sequential arm first",
            file=sys.stderr,
        )
        return 1
    if args.schedule == "replay" and "saturate_equal_time" in args.arms:
        print("saturate_equal_time has no meaning under a replay schedule", file=sys.stderr)
        return 1

    task = TASKS[args.task]
    out = args.out or Path(f"results/{task.name}.csv")
    args.workdir.mkdir(parents=True, exist_ok=True)
    train_subset = args.train_subset if args.train_subset > 0 else None
    valid_subset = args.valid_subset if args.valid_subset > 0 else None
    task.load(BATCH_SIZE, train_subset=train_subset, valid_subset=valid_subset)  # download once
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(
        f"task: {task.name}  parameters {task.parameter_range[0]:,}-{task.parameter_range[1]:,}  "
        f"schedule: {args.schedule}"
    )

    rows: list[dict[str, object]] = []
    if args.resume and out.exists():
        rows = _read_rows(out)
        print(f"resuming: {len(rows)} rows already in {out}")
    done = {(str(r["arm"]), r["group_size"], int(r["seed"])) for r in rows}  # type: ignore[arg-type]

    for seed in range(args.first_seed, args.first_seed + args.seeds):
        sequential_wall: float | None = None
        sequential_peak: int | None = None
        for arm in args.arms:
            group_sizes = args.group_sizes if arm != "sequential" else [None]
            for k in group_sizes:
                if (arm, k, seed) in done:
                    if arm == "sequential":
                        previous = next(
                            r
                            for r in rows
                            if r["arm"] == "sequential" and int(r["seed"]) == seed  # type: ignore[arg-type]
                        )
                        sequential_wall = float(previous["wall_time_s"])  # type: ignore[arg-type]
                        sequential_peak = int(previous["peak_vram_torch_mib"]) * MIB  # type: ignore[arg-type]
                    continue
                started = time.perf_counter()
                spec = RowSpec(
                    task=task.name,
                    arm=arm,
                    group_size=k,
                    seed=seed,
                    trials=args.trials,
                    schedule_kind=args.schedule,
                    replay_block=args.replay_block,
                    epochs=args.epochs,
                    train_subset=train_subset,
                    valid_subset=valid_subset,
                    context_mib=args.context_mib,
                    max_group_size=max(args.group_sizes),
                    workdir=str(args.workdir),
                    sequential_wall=sequential_wall,
                    sequential_peak=sequential_peak,
                    kernel_profiling=not args.no_kernel_profiling,
                    loader_workers=args.loader_workers,
                )
                row = run_row_isolated(spec, args.row_deadline)
                if arm == "sequential":
                    sequential_wall = float(row["wall_time_s"])  # type: ignore[arg-type]
                    sequential_peak = int(row["peak_vram_torch_mib"]) * MIB  # type: ignore[arg-type]
                rows.append(row)
                print(
                    f"seed {seed}  {arm:<20} K={k!s:<4} wall {row['wall_time_s']:8.2f}s  "
                    f"best {row['best_accuracy']:.4f}  trials {row['trials_completed']:>4}  "
                    f"archs {row['distinct_architectures']:>3}  "
                    f"kernel {row['kernel_time_fraction'] or 0:5.1%}  "
                    f"mem {row['peak_vram_torch_mib']} MiB  energy {row['energy_j'] or 0:7.1f} J  "
                    f"{row['mean_sm_clock_mhz']} MHz {row['max_temperature_c']}C  "
                    f"({time.perf_counter() - started:.0f}s incl. setup)"
                )
                # Write after every row so that a long sweep survives an interruption.
                out.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = list(dict.fromkeys(key for row in rows for key in row))
                with out.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, restval=None)
                    writer.writeheader()
                    writer.writerows(rows)
                time.sleep(args.cooldown)

    _summarise(rows)
    print(f"\nwrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
