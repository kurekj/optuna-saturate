#!/usr/bin/env bash
# The full evaluation behind the article, in the order it was run. Every stage
# resumes from its CSV, so an interrupted run continues where it stopped.
# One GPU; nothing else should use it while this runs.
set -u
cd "$(dirname "$0")"
PY=./.venv-gpu/Scripts/python.exe   # a CUDA build of PyTorch with the [dev,gpu,bench,raytune] extras
LOG=results/logs
mkdir -p $LOG
run() {
    echo "=== $(date '+%F %T') ${*:3}" >> $LOG/progress.log
    "$@" 2>&1 | grep --line-buffered -v -E "USDT|trial.py:.*WARNING"
    echo "=== $(date '+%F %T') done rc=${PIPESTATUS[0]}" >> $LOG/progress.log
}

# 1. Six execution strategies on Fashion-MNIST, K from 2 to 32, five seeds.
run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials 24 --group-sizes 2 4 8 16 32 --seeds 5 \
    --arms sequential n_jobs streams process ray saturate saturate_equal_time \
    --resume --out results/fashion_cnn.csv > $LOG/fashion_cnn.log

# 2. The same configurations in every arm (fixed schedule in blocks of eight).
run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials 24 --group-sizes 8 --seeds 5 \
    --schedule replay --replay-block 8 --arms sequential n_jobs streams process ray saturate \
    --resume --out results/fashion_cnn_replay.csv > $LOG/fashion_cnn_replay.log

# 3. Two further model families.
run $PY -u -m benchmarks.compare_arms --task kmnist_mlp --trials 24 --group-sizes 8 --seeds 5 \
    --arms sequential n_jobs streams process ray saturate saturate_equal_time \
    --resume --out results/kmnist_mlp.csv > $LOG/kmnist_mlp.log
run $PY -u -m benchmarks.compare_arms --task cifar_vit --trials 24 --group-sizes 8 --seeds 5 \
    --arms sequential n_jobs streams process ray saturate saturate_equal_time \
    --resume --out results/cifar_vit.csv > $LOG/cifar_vit.log

# 4. Full groups at large K: 2K trials, so that the batched arm runs two complete groups.
for K in 16 32 64; do
    run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials $((2 * K)) --group-sizes $K --seeds 3 \
        --arms sequential saturate --resume --out results/fashion_cnn_full_k$K.csv > $LOG/fashion_cnn_full_k$K.log
done

# 5. The synthetic group-size sweep (data resident on the device).
run $PY -u benchmarks/vectorization_gain.py --k 2 4 8 16 32 64 --out results/vectorization_gain.csv \
    > $LOG/vectorization_gain.log

# 6. A longer workload: full training set, three epochs per trial, 48 trials. The CUDA
#    kernel trace is switched off (an hour-long trace does not fit in memory); once with
#    the data loaded in the training process, once with four persistent loader workers
#    (the process arm is left out of the loader-worker runs: its eight processes already
#    load data in parallel, and eight of them with four loader workers each did not
#    finish a row within the deadline).
run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials 48 --group-sizes 8 --seeds 3 \
    --epochs 3 --train-subset 0 --valid-subset 0 --no-kernel-profiling \
    --arms sequential process saturate saturate_equal_time \
    --resume --out results/fashion_cnn_long.csv > $LOG/fashion_cnn_long.log
run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials 48 --group-sizes 8 --seeds 3 \
    --epochs 3 --train-subset 0 --valid-subset 0 --no-kernel-profiling --loader-workers 4 \
    --arms sequential saturate saturate_equal_time \
    --resume --out results/fashion_cnn_long_workers.csv > $LOG/fashion_cnn_long_workers.log

# 7. The short Fashion-MNIST study with four loader workers, for the arms the loader affects.
run $PY -u -m benchmarks.compare_arms --task fashion_cnn --trials 24 --group-sizes 8 --seeds 5 \
    --loader-workers 4 --arms sequential saturate saturate_equal_time \
    --resume --out results/fashion_cnn_workers.csv > $LOG/fashion_cnn_workers.log

echo "=== $(date '+%F %T') ALL DONE" >> $LOG/progress.log
