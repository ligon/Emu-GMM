#!/usr/bin/env bash
# Submit only after reviewing the branch SHA and resource request:
# sbatch --export=ALL,EMU_GMM_145_SHA=<published-sha> \
#   scripts/validation/design_aware_fixed_strata_burst.sh
#SBATCH --job-name=emu145-mc
#SBATCH --account=co_carleton
#SBATCH --partition=savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/ligon/emu-gmm-145-%j.log

set -euo pipefail

: "${SLURM_JOB_ID:?submit with sbatch}"
: "${EMU_GMM_145_SHA:?pass the published branch SHA via --export}"

job_root="/local/job${SLURM_JOB_ID}"
repo_dir="${job_root}/mirrors/Emu-GMM"
mkdir -p "${job_root}/mirrors" "${job_root}/tmp" "${job_root}/poetry-cache"
export TMPDIR="${job_root}/tmp"
export POETRY_CACHE_DIR="${job_root}/poetry-cache"
export POETRY_VIRTUALENVS_IN_PROJECT=true
export XLA_FLAGS=--xla_force_host_platform_device_count=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

git clone --branch design/145-centering \
  /global/home/users/ligon/mirrors/Emu-GMM "$repo_dir"
git -C "$repo_dir" checkout --detach "$EMU_GMM_145_SHA"
actual_sha="$(git -C "$repo_dir" rev-parse HEAD)"
test "$actual_sha" = "$EMU_GMM_145_SHA"

cd "$repo_dir"
echo "job_id=$SLURM_JOB_ID sha=$actual_sha cpus=$SLURM_CPUS_PER_TASK"
echo "allowed_cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
echo "command=poetry run python scripts/validation/design_aware_fixed_strata_pilot.py --fit-reps 300 --paired-reps 4096 --seed 20260610"
nice -n 19 poetry install --no-interaction
nice -n 19 poetry run python \
  scripts/validation/design_aware_fixed_strata_pilot.py \
  --fit-reps 300 --paired-reps 4096 --seed 20260610
