#!/usr/bin/env bash
#SBATCH --job-name=evolve_gmm_config
#SBATCH --account=co_carleton
#SBATCH --partition=savio4_htc
#SBATCH --qos=carleton_htc4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/evolve_gmm_%j.log
#
# AlphaEvolve run for the emu-gmm estimator-config experiment.
# Submit FROM THE RUN CLONE (shared filesystem; node-local paths die
# with the allocation -- AggLC job 35591011 lesson):
#
#   cd $SUE_SCRATCH/emu_gmm_evolve/repo
#   sbatch [--dependency=afterany:<jobid>] evolve/sbatch_evolve.sh
#
# The dependency gates the run on any in-flight sibling AlphaEvolve run
# (shared per-project API quota).  Registered run size 80/80
# (pre-registration ratified 2026-07-11; README.org).  Concurrency and
# affinity-slot width are OPERATIONAL knobs.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "$REPO_ROOT" in
    /local/*|/tmp/*)
        echo "ERROR: repo at $REPO_ROOT is node-local; submit from a" \
             "shared-filesystem clone" >&2
        exit 2 ;;
esac

AE_PY=${AE_PY:-/global/scratch/users/ligon/tmp/AlphaEvolve/ae_env/bin/python}
AE_REPO=${AE_REPO:-/global/scratch/users/ligon/tmp/alphaevolve-on-googlecloud}
ENV_FILE=${AE_ENV_FILE:-$REPO_ROOT/evolve/.env}

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found; create the project-local .env" \
         "(PROJECT_ID, GE_APP_ID=emu-gmm, ...) before spending API budget" >&2
    exit 2
fi

# Registered run size (pre-registration, ratified 2026-07-11); override
# via environment only for non-registered shakedowns.
: "${EVOLVE_MAX_GENERATED:=80}"
: "${EVOLVE_MAX_EVALUATED:=80}"
: "${EVOLVE_CONCURRENCY:=16}"
: "${EVOLVE_EVAL_THREADS:=2}"
: "${EVOLVE_EVAL_TIMEOUT:=900}"

exec env PYTHONPATH="$AE_REPO/src" AE_ENV_FILE="$ENV_FILE" \
    EVOLVE_MAX_GENERATED="$EVOLVE_MAX_GENERATED" \
    EVOLVE_MAX_EVALUATED="$EVOLVE_MAX_EVALUATED" \
    EVOLVE_CONCURRENCY="$EVOLVE_CONCURRENCY" \
    EVOLVE_EVAL_THREADS="$EVOLVE_EVAL_THREADS" \
    EVOLVE_EVAL_TIMEOUT="$EVOLVE_EVAL_TIMEOUT" \
    ${EVOLVE_INITIAL:+EVOLVE_INITIAL="$EVOLVE_INITIAL"} \
    ${EVOLVE_SEED_EXPECT:+EVOLVE_SEED_EXPECT="$EVOLVE_SEED_EXPECT"} \
    ${EVOLVE_NUM_SAMPLERS:+EVOLVE_NUM_SAMPLERS="$EVOLVE_NUM_SAMPLERS"} \
    ${EVOLVE_NUM_EVALUATORS:+EVOLVE_NUM_EVALUATORS="$EVOLVE_NUM_EVALUATORS"} \
    "$AE_PY" "$REPO_ROOT/evolve/run_experiment.py"
