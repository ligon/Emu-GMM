#!/bin/bash
#SBATCH -A co_carleton
#SBATCH -p savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J emu-reviewgate
#SBATCH -o /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM/.coder/reviewgate_%j.log
# Full gate (ruff + black + mypy + FULL pytest incl slow) for the
# fix/review-20260702 branch (2026-07-02 comprehensive-review fixes).
# co_carleton condo QoS on savio4_htc: free for fc_jevons, no preemption,
# per-core scheduling -- modest 8-core ask per scrum-master-hpc cost rules.
set -o pipefail
cd /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM || exit 2
export JAX_PLATFORMS=cpu
echo "=== branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) on $(hostname); cpus=$SLURM_CPUS_ON_NODE ==="
echo "--- ruff ---";  .venv/bin/ruff check . ;            echo "ruff_exit=$?"
echo "--- black ---"; .venv/bin/black --check . 2>&1 | tail -3 ; echo "black_exit=${PIPESTATUS[0]}"
echo "--- mypy ---";  .venv/bin/mypy src tests 2>&1 | tail -3 ; echo "mypy_exit=${PIPESTATUS[0]}"
echo "--- FULL pytest (incl slow), serial ---"
.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -30 ; echo "pytest_exit=${PIPESTATUS[0]}"
echo "=== DONE $(date) ==="
