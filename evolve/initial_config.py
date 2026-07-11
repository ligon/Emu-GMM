"""Seed candidate: the all-defaults configuration (scores exactly 0).

CONTRACT (fixed; the evaluator enforces it -- do not change):

``propose_config(ctx)`` receives read-only problem data

    fixtures : {name: {n_sim, baseline: {rmse, mean_iters,
               converged_frac}}} for the fitness fixtures
    theta_true : the exact ground truth (beta, gamma)
    bounds : the whitelist -- keys and admissible ranges
    defaults : the registered default configuration

and two callables

    probe(config) -> {rmse, mean_iters, converged_frac}
        A budgeted probe: the candidate config estimated on 6 CRN
        replicates of the first fitness fixture (at most a fixed
        number of calls; probe_calls_remaining() says how many are
        left).  Exceeding the budget aborts the evaluation.
    probe_calls_remaining() -> int

and must return a config dict using ONLY whitelisted keys within
bounds: weighting in {"cue", "identity", "iterated"};
weighting_iterations (int), weighting_tol, rtol, atol, max_steps.
The evaluator re-runs the returned config at the registered replicate
counts on ALL fitness fixtures; score = 1 - rmse/baseline_rmse per
fixture (mean across fixtures), gated on 100 percent convergence and
mean iterations <= 3x baseline.  Determinism: no randomness.

GOAL: beat the default (CUE weighting + LM defaults) on exact-truth
recovery RMSE without buying accuracy through unbounded compute.
Statistical hints: CUE is asymptotically efficient but can be erratic
in small samples (the n=200 fixture); iterated GMM's iteration count
and tolerance trade stability against efficiency; optimizer
tolerances looser than the statistical noise floor waste nothing,
tighter ones waste compute against the iteration gate.
"""


# EVOLVE-BLOCK-START
def propose_config(ctx):
    """Baseline: the registered defaults, unchanged."""
    return {}


# EVOLVE-BLOCK-END
