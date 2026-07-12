#!/usr/bin/env bash
#SBATCH --job-name=kappa_sweep
#SBATCH --account=co_carleton
#SBATCH --partition=savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/kappa_sweep_%j.log
#
# Direct kappa_target x fixture sweep (exploratory; see kappa_sweep.py).
# 16 (kappa, fixture) pairs on 8 two-core affinity slots, two waves.
# Submit from the run-clone root: sbatch evolve/sbatch_kappa_sweep.sh
set -eu

REPO_ROOT=${EVOLVE_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit via sbatch from the repo root}}
PY=${EVOLVE_EMU_PY:-/global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM/.venv/bin/python}

KAPPAS="1e2 1e3 1e4 1e5 1e7 1e8 1e10 1e12"
FIXTURES="fb_few_eps_k31 fb_strathet_eps_k32"

# Build 2-core slot list from the job's affinity.
mapfile -t CORES < <($PY -c "import os; [print(c) for c in sorted(os.sched_getaffinity(0))]")
NSLOTS=$(( ${#CORES[@]} / 2 ))

i=0
pids=()
for k in $KAPPAS; do
  for f in $FIXTURES; do
    slot=$(( i % NSLOTS ))
    c1=${CORES[$((slot * 2))]}
    c2=${CORES[$((slot * 2 + 1))]}
    # Wait for a free wave: at most NSLOTS concurrent.
    while [ "$(jobs -rp | wc -l)" -ge "$NSLOTS" ]; do sleep 5; done
    taskset -c "$c1,$c2" "$PY" "$REPO_ROOT/evolve/kappa_sweep.py" \
        --kappa "$k" --fixture "$f" 2>/dev/null &
    pids+=($!)
    i=$((i + 1))
  done
done
wait
echo "kappa sweep complete: $i evaluations"
