#!/bin/bash
#SBATCH -A co_carleton
#SBATCH -p savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -J emu-mc184
#SBATCH -o /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM/.coder/mc184_%j.log
# Issue #184 evidence run: wild-bootstrap calibration at theta_hat, R=500 B=999.
set -o pipefail
cd /global/scratch/fsa/fc_jevons/ligon/mirrors/Emu-GMM || exit 2
export JAX_PLATFORMS=cpu
echo "=== branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) on $(hostname) ==="
.venv/bin/python .coder/mc_wildboot_184.py 500 999
echo "=== DONE $(date) exit=$? ==="
