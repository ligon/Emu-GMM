"""AlphaEvolve GA adapter for the emu-gmm estimator-config experiment.

Runs under the ae_env python (google client libs; no jax needed here):

    /global/scratch/users/ligon/tmp/AlphaEvolve/ae_env/bin/python \
        evolve/run_experiment.py [--dry-run]

The EVALUATOR (evaluator_gmm.py) runs locally under the emu-gmm venv
(EVOLVE_EMU_PY) in a subprocess per candidate, with a wall-clock
timeout AND a per-evaluation CPU-affinity slot: concurrent JAX
subprocesses on shared cores oversubscribe XLA's thread pools (the
64-core JIT-mmap hazard, CLAUDE.md), so each evaluation is pinned to a
disjoint core group via taskset; XLA sizes its pools to the
affinity-visible count.  Candidate code text is the only thing that
leaves the cluster; fixtures and scores are computed here.

Configuration comes from the environment / .env (PROJECT_ID, GE_APP_ID,
... see alphaevolve-on-googlecloud/example.env), plus:

    EVOLVE_EMU_PY         evaluator python (default: the scratch-mirror venv)
    EVOLVE_ORACLE_BUDGET  probe calls per evaluation (default 8; REGISTERED)
    EVOLVE_EVAL_TIMEOUT   seconds per candidate evaluation (default 900)
    EVOLVE_MAX_GENERATED / EVOLVE_MAX_EVALUATED  run size (default 80/80;
                          REGISTERED for the 2026-07-11 run)
    EVOLVE_CONCURRENCY    generation concurrency (default 16; OPERATIONAL)
    EVOLVE_EVAL_THREADS   cores per evaluation slot (default 2; OPERATIONAL)
    EVOLVE_RESULTS_DIR    per-candidate JSON records
                          (default $SUE_SCRATCH/emu_gmm_evolve/runs)
    AE_ENV_FILE           .env path (default: .env next to this file)

Pre-registration: README.org in this directory (ratified 2026-07-11;
sha256 freeze recorded there).  Do not change the fitness fixtures,
oracle budget, or scoring config mid-run; operational knobs
(concurrency, threads, timeouts) may be tuned.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

_DEFAULT_EMU_PY = ("/global/scratch/fsa/fc_jevons/ligon/mirrors/"
                   "Emu-GMM/.venv/bin/python")
EMU_PY = os.getenv("EVOLVE_EMU_PY", _DEFAULT_EMU_PY)
EXPERIMENT_NAME = os.getenv("EVOLVE_EXPERIMENT", "gmm_config")
# Registered per-experiment defaults (pre-registration): run 1 probes
# 8 x 6 reps and evaluates in ~2 min; run 2 probes 6 x 12 reps with
# per-rep anchoring and evaluates in ~20-25 min.
_IS_RIDGE = EXPERIMENT_NAME == "ridge_config"
ORACLE_BUDGET = int(os.getenv("EVOLVE_ORACLE_BUDGET", "6" if _IS_RIDGE else "8"))
EVAL_TIMEOUT = int(os.getenv("EVOLVE_EVAL_TIMEOUT", "2700" if _IS_RIDGE else "900"))
MAX_GENERATED = int(os.getenv("EVOLVE_MAX_GENERATED", "80"))
MAX_EVALUATED = int(os.getenv("EVOLVE_MAX_EVALUATED", "80"))
EVAL_THREADS = int(os.getenv("EVOLVE_EVAL_THREADS", "2"))
RESULTS_DIR = Path(os.getenv(
    "EVOLVE_RESULTS_DIR",
    os.path.join(os.getenv("SUE_SCRATCH",
                           "/global/scratch/fsa/fc_jevons/ligon/"
                           "sue-scratch"),
                 "emu_gmm_evolve", "runs")))

EVALUATION_METRIC = "score"
FAIL_SCORE = -1.0

PROBLEM_GMM_CONFIG = f"""\
Evolve propose_config(ctx) in the given Python file.  Context: a GMM
estimator for a multi-asset consumption Euler economy with EXACT known
truth (beta = 0.96, gamma = 2.0) is configured by a whitelisted dict:
weighting in {{"cue", "identity", "iterated"}}; weighting_iterations
(int, 1..30) and weighting_tol (1e-10..1e-2) for iterated GMM; and
Levenberg-Marquardt optimizer knobs rtol, atol (1e-12..1e-4) and
max_steps (int, 10..400).  The baseline is CUE weighting with LM
defaults.  GOAL: return a config with LOWER exact-truth recovery RMSE
over common-random-number replicates on two fitness fixtures (n=800
and n=200 observations; 24 replicates each).  Score = 1 -
rmse/baseline_rmse averaged over the fixtures (0 = baseline, higher is
better, -1 = failure), GATED on 100 percent convergence and mean
optimizer iterations <= 3x baseline -- accuracy may not be bought with
unbounded compute.  A budgeted probe oracle ({ORACLE_BUDGET} calls, 6
replicates each on the n=800 fixture) lets you measure a candidate
config before committing; respect it via
ctx['probe_calls_remaining']().  Statistical hints: CUE (continuously
updated GMM) is asymptotically efficient but can be erratic in small
samples (the n=200 fixture); iterated GMM trades stability against
efficiency through its iteration count and tolerance; optimizer
tolerances looser than the statistical noise floor cost nothing,
tighter ones waste iterations against the compute gate.  Return DATA
(the config dict), never code; unknown keys or out-of-bounds values
score -1.  Use no randomness.  Docstrings in the file state the full
contract.
"""

PROBLEM_RIDGE_CONFIG = f"""\
Evolve propose_config(ctx) in the given Python file.  Context: a GMM
estimator for a stratified PSU-randomized IV model with EXACT known
truth, genuine cluster-level missingness, and a near-collinear
instrument (M=5 moments, K=2 parameters), estimated with a
design-aware covariance whose assembled V is NOT PSD-by-construction:
on a third to half of replicates an adaptive Tikhonov ridge must
repair an indefinite or ill-conditioned V.  The config dict is
whitelisted: weighting in {{"cue", "identity", "iterated"}};
weighting_iterations (int, 1..30) and weighting_tol (1e-10..1e-2);
optimizer knobs rtol, atol (1e-12..1e-4) and max_steps (int, 10..400);
and kappa_target (1e2..1e12), the ridge repair's condition-number
target (default 1e6).  The all-defaults baseline is measurably
MIS-CALIBRATED here: ~25-30 percent of replicates fail to converge,
more have NaN analytic SEs (indefinite sandwich meat), and effective
CI coverage sits far below the nominal 95 percent.  GOAL: minimize
the calibration error err = mean_k |cov_eff_k - 0.95| + |rej_eff -
0.05| over CRN replicates on two fitness fixtures, where EVERY
registered replicate is in the denominator -- a replicate that fails
to converge or has NaN SEs counts as NOT covered, and one that fails
or has a NaN J p-value counts as a REJECTION, so breaking hard
replicates always hurts.  Score = 1 - err/err_baseline averaged over
the fixtures (0 = baseline, higher is better, -1 = failure), GATED on
n_used >= baseline n_used and mean optimizer iterations <= 3x
baseline.  A budgeted probe oracle ({ORACLE_BUDGET} calls, 12 replicates
each on the first fitness fixture) lets you measure a candidate
before committing; respect it via ctx['probe_calls_remaining']().
Statistical hints: kappa_target trades repair against distortion (too
high leaves V near-singular and kills convergence; too low distorts
the criterion and the J distribution); the ridge anchors per
replicate at theta_init then freezes; CUE re-evaluates the
regularised V along the path and can be erratic when V is barely
repaired, iterated GMM is steadier, identity ignores V for point
estimation (maximal convergence robustness) but its SEs still
sandwich the raw meat so coverage is not automatically better.
Return DATA (the config dict), never code; unknown keys or
out-of-bounds values score -1.  Use no randomness.  Docstrings in the
file state the full contract.
"""

EXPERIMENTS = {
    "gmm_config": dict(
        evaluator="evaluator_gmm.py", initial="initial_config.py",
        title="emu-gmm estimator configuration (run 1)",
        problem=PROBLEM_GMM_CONFIG,
        extra_args=["--oracle-budget", str(ORACLE_BUDGET)]),
    "ridge_config": dict(
        evaluator="evaluator_ridge.py", initial="initial_config_ridge.py",
        title="emu-gmm ridge/weighting calibration, binding regime (run 2)",
        problem=PROBLEM_RIDGE_CONFIG,
        extra_args=["--oracle-budget", str(ORACLE_BUDGET)]),
}
EXPERIMENT = EXPERIMENTS[EXPERIMENT_NAME]

# ---------------------------------------------------------------------------
# CPU-affinity slots: pin each concurrent evaluation to a disjoint core
# group so XLA thread pools size to the slot, not the node (CLAUDE.md
# "64-core JIT-mmap hazard"; validated mechanism: taskset + affinity).
# ---------------------------------------------------------------------------
_SLOTS: "queue.Queue[str]" = queue.Queue()


def _init_slots():
    cores = sorted(os.sched_getaffinity(0))
    t = max(1, EVAL_THREADS)
    groups = [cores[i:i + t] for i in range(0, len(cores) - t + 1, t)]
    if not groups:
        groups = [cores]
    for g in groups:
        _SLOTS.put(",".join(str(c) for c in g))
    logger.info("affinity slots (%d cores, %d per eval): %d concurrent",
                len(cores), t, len(groups))
    return len(groups)


def evaluate_program(program_candidate) -> dict:
    """AlphaEvolve evaluation callback: files in, scores out (local run)."""
    files = program_candidate.get("content", {}).get("files", [])
    if not files:
        return {"scores": {"scores": [
            {"metric": EVALUATION_METRIC, "score": FAIL_SCORE}]},
            "artifacts": {"error": "no files in candidate"}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slot = _SLOTS.get()  # blocks until a core group frees up
    try:
        with tempfile.TemporaryDirectory(prefix="ae_cand_") as td:
            cand = Path(td) / "candidate.py"
            cand.write_text(files[0].get("content", ""))
            cmd = (["taskset", "-c", slot,
                    EMU_PY, str(THIS_DIR / EXPERIMENT["evaluator"]),
                    "--candidate", str(cand)] + EXPERIMENT["extra_args"])
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=EVAL_TIMEOUT, cwd=str(THIS_DIR))
            except subprocess.TimeoutExpired:
                return {"scores": {"scores": [
                    {"metric": EVALUATION_METRIC, "score": FAIL_SCORE}]},
                    "artifacts": {"error": "evaluation timed out after "
                                           f"{EVAL_TIMEOUT}s"}}
    finally:
        _SLOTS.put(slot)
    if proc.returncode != 0:
        return {"scores": {"scores": [
            {"metric": EVALUATION_METRIC, "score": FAIL_SCORE}]},
            "artifacts": {"error": f"evaluator exit {proc.returncode}",
                "stderr": proc.stderr[-2000:]}}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"scores": {"scores": [
            {"metric": EVALUATION_METRIC, "score": FAIL_SCORE}]},
            "artifacts": {"error": "unparseable evaluator output",
                          "stdout": proc.stdout[-2000:]}}
    # Persist the full record next to the run (audit trail).
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
    (RESULTS_DIR / f"cand-{stamp}.json").write_text(
        json.dumps({"result": result,
                    "candidate": files[0].get("content", "")}, indent=2))
    artifacts = {"fixtures": json.dumps(result.get("fixtures", []))[:5000]}
    if "config" in result:
        artifacts["config"] = json.dumps(result["config"])[:800]
    if result.get("status") not in (None, "ok"):
        artifacts["status"] = str(result.get("status"))[:200]
    return {
        "scores": {"scores": [
            {"metric": EVALUATION_METRIC, "score": result["score"]},
            {"metric": "min_fixture_score",
             "score": result["min_fixture_score"]},
        ]},
        "artifacts": artifacts,
    }


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("experiment: %s (%s)", EXPERIMENT_NAME,
                EXPERIMENT["evaluator"])
    n_slots = _init_slots()
    initial_name = os.getenv("EVOLVE_INITIAL", EXPERIMENT["initial"])
    logger.info("seed file: %s", initial_name)
    initial_src = (THIS_DIR / initial_name).read_text()

    # Score the seed locally first: its real score seeds the experiment,
    # and a broken harness fails HERE, before any API call.
    seed_eval = evaluate_program(
        {"content": {"files": [{"path": initial_name,
                                "content": initial_src}]}})
    seed_score = seed_eval["scores"]["scores"][0]["score"]
    logger.info("seed score: %s (expect ~0.0)", seed_score)
    if "--dry-run" in sys.argv:
        print(json.dumps(seed_eval, indent=2))
        return

    seed_expect = float(os.getenv("EVOLVE_SEED_EXPECT", "0.0"))
    if abs(seed_score - seed_expect) > 0.05:
        raise SystemExit(
            f"seed score {seed_score} differs from its registered value {seed_expect} by > 0.05 "
            "-- fixtures/baseline out of sync (or EVOLVE_SEED_EXPECT not "
            "set for a non-baseline seed); rebuild/re-register before "
            "spending API budget")

    from dotenv import load_dotenv
    # Explicit path: the default find_dotenv would search upward from THIS
    # directory and never see a project-local .env.
    load_dotenv(os.getenv("AE_ENV_FILE", str(THIS_DIR / ".env")))
    import asyncio

    import nest_asyncio
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.controller import run_controller_loop
    from alpha_evolve.experiment import AlphaEvolveExperiment
    from alpha_evolve.visualization import get_score

    client = AlphaEvolveClient(
        project_id=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION", "global"),
        collection=os.getenv("COLLECTION", "default_collection"),
        engine=os.getenv("GE_APP_ID"),
        assistant=os.getenv("ASSISTANT", "default_assistant"),
        base_url=os.getenv("BASE_URL", "discoveryengine.googleapis.com"),
    )
    # parallel_evaluation: without it a SYNC evaluator blocks the async
    # loop and evaluations serialize regardless of num_evaluators
    # (measured on the AggLC pilot's run 1).  Our evaluator is an
    # isolated deterministic subprocess in its own affinity slot, so
    # thread-dispatch cannot affect scores.
    experiment = AlphaEvolveExperiment(client, evaluate_program,
                                       MAX_EVALUATED,
                                       parallel_evaluation=True)
    experiment.create_experiment({
        "title": EXPERIMENT["title"],
        "problem_description": EXPERIMENT["problem"],
        "program_language": "python",
        "run_settings": {
            "max_programs": MAX_GENERATED,
            # Request-level, not project-quota (AggLC measurement
            # 2026-07-11: 16 -> ~6-8 gen/min vs ~2.5/min at 4).
            "concurrency": int(os.getenv("EVOLVE_CONCURRENCY", "16")),
        },
    })
    experiment.create_initial_program({
        "content": {"files": [
            {"path": initial_name, "content": initial_src}]},
        "evaluation": {"scores": {"scores": [
            {"metric": EVALUATION_METRIC, "score": seed_score}]}},
    })
    experiment.start_experiment()
    nest_asyncio.apply()
    asyncio.run(run_controller_loop(
        experiment,
        num_samplers=int(os.getenv("EVOLVE_NUM_SAMPLERS", "4")),
        num_evaluators=int(os.getenv("EVOLVE_NUM_EVALUATORS",
                                     str(n_slots)))))

    response = experiment.list_programs(params={"order_by": "score desc"})
    programs = (response or {}).get("alphaEvolvePrograms", [])
    programs.sort(key=lambda p: get_score(p, EVALUATION_METRIC),
                  reverse=True)
    logger.info("top programs:")
    for i, prog in enumerate(programs[:5]):
        logger.info("rank %d: %s score=%s", i + 1,
                    prog.get("name", "?"),
                    get_score(prog, EVALUATION_METRIC))


if __name__ == "__main__":
    main()
