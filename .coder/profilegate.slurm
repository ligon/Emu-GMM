#!/bin/bash
#SBATCH -A co_carleton
#SBATCH -p savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J emu-profile
#SBATCH -o /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM/.coder/profile_%j.log
# Duration-profiling + parallel-gate trial: FULL suite under pytest-xdist with
# one single-threaded worker per core (OMP/XLA caps prevent 16 workers x
# 16-thread JAX pools oversubscribing the cgroup). --durations=0 emits every
# test's setup/call time -- the ranking that decides what goes behind @slow.
set -o pipefail
cd /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM || exit 2
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS=--xla_force_host_platform_device_count=1
echo "=== branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) on $(hostname); workers=$SLURM_CPUS_ON_NODE ==="
.venv/bin/python -m pytest -q -p no:cacheprovider -n "$SLURM_CPUS_ON_NODE" \
    --durations=0 --durations-min=1 2>&1 | tail -400
echo "pytest_exit=${PIPESTATUS[0]}"
echo "=== DONE $(date) ==="
