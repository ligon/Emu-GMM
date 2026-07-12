"""EXPLORATORY reference arms for the run-2 report (NOT registered).

EL's observation (2026-07-11, mid-run-2): standard two-step GMM should
be "in the race".  It nominally is -- iterated weighting with
weighting_iterations in {1, 2} -- but emu-gmm flags fixed-k schedules
non-converged when the theta-sequence hasn't met weighting_tol
(estimator.py:1254; filed as #201), and the registered adversarial
accounting turns that flag into "not covered + rejected".  This script
measures named reference configs on the registered fitness fixtures
under BOTH accountings:

- err_registered: the frozen run-2 metric (flag-respecting).
- err_practitioner: ignores the converged flag entirely -- every rep's
  point estimate and SEs are used as a textbook two-step practitioner
  would (they have no iteration flag to consult); NaN SEs still count
  against coverage, and a NaN J p-value still counts as a rejection.
  CAVEAT: without #201's surfaced iterated_status this cannot separate
  inner-LM failures from schedule exhaustion, so err_practitioner is a
  LOWER bound on the flag artifact's size, not a clean decomposition.

Usage:
    .venv/bin/python evolve/reference_arms.py [fixture ...]
"""

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (
    str(_THIS_DIR),
    str(_REPO_ROOT / "src"),
    str(_REPO_ROOT / "scripts" / "validation"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import ladder_mc as lm  # noqa: E402
import numpy as np  # noqa: E402
from emu_gmm import estimate  # noqa: E402

import evaluator_ridge as ev  # noqa: E402
import fixture_binding as fx  # noqa: E402

REFERENCE_CONFIGS = {
    # Textbook-ish two-step: one V-refresh after the V(theta_init) solve.
    "two_step": {
        "weighting": "iterated",
        "weighting_iterations": 2,
        "weighting_tol": 1e-2,
    },
    # One-step at the anchored V(theta_init) (theta_init = truth here).
    "one_step_anchored": {
        "weighting": "iterated",
        "weighting_iterations": 1,
        "weighting_tol": 1e-2,
    },
    # Family anchors for context.
    "identity": {"weighting": "identity"},
    "cue_default": {},
}


def run_both_accountings(objects, fixture):
    """Like fixture_binding.run_replicates, but tallies both accountings."""
    spec, design = fx._design_for(fixture["spec"])
    cov = lm.covariance_arm(design, "design_aware")
    model, moment_names = lm.model_for(spec)
    master = jax.random.PRNGKey(fixture["key"])
    n_reps = int(fixture["n_reps"])
    truth = np.array([fx.TRUTH.beta0, fx.TRUTH.beta1])

    K = len(fx.PARAM_NAMES)
    cov_reg = np.zeros(K, dtype=int)
    cov_prac = np.zeros(K, dtype=int)
    rej_reg = 0
    rej_prac = 0
    n_used = 0

    for r in range(n_reps):
        measure = lm.draw_measure(design, jax.random.fold_in(master, r))
        res = estimate(
            model=model,
            measure=measure,
            covariance=cov,
            weighting=objects.get("weighting"),
            regularization=objects.get("regularization"),
            optimizer=objects.get("optimizer"),
            theta_init=fx.TRUTH,
            moment_names=moment_names,
        )
        conv = bool(res.converged)
        law = res.asymptotic()
        th = np.array([float(res.theta_hat.beta0), float(res.theta_hat.beta1)])
        se = np.asarray(law.se(), dtype=float).reshape(-1)
        jp = float(law.J_pvalue)

        covered_k = [
            bool(np.isfinite(se[k]) and abs(th[k] - truth[k]) <= fx._Z * se[k])
            for k in range(K)
        ]
        rejected = (not np.isfinite(jp)) or jp < fx.ALPHA

        # practitioner accounting: flag-blind
        for k in range(K):
            cov_prac[k] += covered_k[k]
        rej_prac += rejected
        # registered accounting: flag-respecting
        if conv:
            n_used += 1
            for k in range(K):
                cov_reg[k] += covered_k[k]
            rej_reg += rejected
        else:
            rej_reg += 1
        if (r + 1) % 25 == 0:
            jax.clear_caches()

    def err(cov_counts, rej_count):
        c = cov_counts / float(n_reps)
        return float(np.mean(np.abs(c - fx.LEVEL)) + abs(rej_count / n_reps - fx.ALPHA))

    return dict(
        n_reps=n_reps,
        n_used=int(n_used),
        err_registered=err(cov_reg, rej_reg),
        cov_eff_registered=[float(c) for c in cov_reg / float(n_reps)],
        rej_eff_registered=float(rej_reg / n_reps),
        err_practitioner=err(cov_prac, rej_prac),
        cov_eff_practitioner=[float(c) for c in cov_prac / float(n_reps)],
        rej_eff_practitioner=float(rej_prac / n_reps),
    )


def main(argv):
    fixtures = argv[1:] or [n for n, s in fx.REGISTRY.items() if s["role"] == "fitness"]
    out = {}
    for label, cfg in REFERENCE_CONFIGS.items():
        objects = ev.build_objects(ev.validate_config(dict(cfg)))
        out[label] = {}
        for name in fixtures:
            base = fx.build_baseline(name)["baseline"]
            stats = run_both_accountings(objects, fx.REGISTRY[name])
            stats["baseline_err"] = base["err"]
            stats["score_registered"] = 1.0 - stats["err_registered"] / base["err"]
            stats["score_practitioner"] = 1.0 - stats["err_practitioner"] / base["err"]
            out[label][name] = stats
            print(
                f"{label:18s} {name:22s} used {stats['n_used']:2d}/{stats['n_reps']}"
                f"  err_reg {stats['err_registered']:.4f}"
                f" (score {stats['score_registered']:+.3f})"
                f"  err_prac {stats['err_practitioner']:.4f}"
                f" (score {stats['score_practitioner']:+.3f})",
                flush=True,
            )
    Path("/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve").joinpath(
        "reference_arms_run2.json"
    ).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
