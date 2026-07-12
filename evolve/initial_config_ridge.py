"""Seed candidate for run 2: the all-defaults configuration (scores 0).

CONTRACT (fixed; the evaluator enforces it -- do not change):

``propose_config(ctx)`` receives read-only problem data

    fixtures : {name: {n_obs, baseline: {err, cov_eff, rej_eff,
               n_reps, n_used, n_valid_se, mean_iters,
               binding_frequency, meat_indefinite_frac}}} for the
               fitness fixtures
    theta_true : the exact ground truth (beta0, beta1)
    bounds : the whitelist -- keys and admissible ranges
    defaults : the registered default configuration

and two callables

    probe(config) -> the stats dict above, for the candidate config on
        12 CRN replicates of the first fitness fixture (at most a
        fixed number of calls; probe_calls_remaining() says how many
        are left).  Exceeding the budget aborts the evaluation.
    probe_calls_remaining() -> int

and must return a config dict using ONLY whitelisted keys within
bounds: weighting in {"cue", "identity", "iterated"};
weighting_iterations (int), weighting_tol, rtol, atol, max_steps
(int), kappa_target.  Determinism: no randomness.

SETTING: a stratified PSU-randomized IV model (M=5 moments, K=2
parameters) with genuine cluster-level missingness and a
near-collinear instrument, estimated with a design-aware covariance
whose assembled V is NOT PSD-by-construction -- on a third to half of
replicates the adaptive Tikhonov ridge must repair an indefinite or
ill-conditioned V (kappa_target is the repair's condition-number
target).  In this regime the all-defaults baseline is measurably
mis-calibrated: it fails to converge on ~25-30 percent of replicates,
its analytic SEs are NaN on more (indefinite sandwich meat), and its
effective CI coverage sits far below the nominal 95 percent.

GOAL: minimize the fixture's calibration error

    err = mean_k |cov_eff_k - 0.95| + |rej_eff - 0.05|

where EVERY registered replicate is in the denominator: a replicate
that fails to converge or has a NaN SE counts as NOT covered, and one
that fails or has a NaN J p-value counts as a REJECTION.  Breaking
hard replicates therefore always hurts.  Score per fixture = 1 -
err/err_baseline (0 = baseline, higher better, -1 = failure), GATED on
n_used >= baseline n_used and mean iterations <= 3x baseline.

Statistical hints: kappa_target trades repair against distortion --
too high (weak ridge) leaves V near-singular, hurting convergence and
the weighting; too low (aggressive ridge) distorts the criterion and
the J distribution.  The ridge anchors per replicate on V at
theta_init (truth), then freezes.  Weighting interacts with the
ridge: CUE re-evaluates the regularised V along the optimization path
and can be erratic when V is barely repaired; iterated GMM freezes
between iterations and is steadier; identity ignores V for point
estimation entirely (maximal convergence robustness) but its
asymptotic covariance still sandwiches the raw meat, so its SEs and
coverage are NOT automatically better.  Optimizer tolerances looser
than the noise floor cost nothing; tighter ones waste the iteration
budget.
"""


# EVOLVE-BLOCK-START
def propose_config(ctx):
    """Baseline: the registered defaults, unchanged."""
    return {}


# EVOLVE-BLOCK-END
