"""Known-truth BINDING-REGIME fixtures for the run-2 evolve harness.

A fixture is a frozen (DesignSpec kwargs, master key, n_reps) triple on
the #130 ladder_mc DGP family (stratified PSU-randomized IV with
genuine PSU-level missingness and, on the binding arms, a
near-collinear fifth moment), estimated with the DESIGN-AWARE
covariance -- the regime where ``DiagonalTikhonov`` actually binds
(#130 pilot: eps-collinearity alone leaves V ill-conditioned but PD;
the design-aware glue binds on ~30-50 percent of datasets in few-PSU
designs).  Ground truth is exact by construction (BETA_TRUE in
ladder_mc; every moment condition holds at the truth).

Anchoring is PER-REP (bare ``estimate()`` per replicate, emu-gmm #142):
``build_estimator``'s factory reuse would freeze replicate 0's
``tau_anchor`` across the study, making regularizer knobs meaningless
to evaluate.  Replicate ``r`` draws with ``fold_in(PRNGKey(key), r)``
(the package CRN contract), so fixtures are fully deterministic.

The registered per-fixture calibration error (the run-2 score's
ingredient; ratified 2026-07-11) uses ADVERSARIAL accounting -- every
registered replicate is in the denominator, so a candidate cannot
improve by breaking hard replicates:

    cov_eff[k] = #{reps: converged AND finite se_k AND |theta_k -
                  truth_k| <= z_.975 * se_k} / n_reps
    rej_eff    = #{reps: NOT converged OR NOT finite J_pvalue OR
                  J_pvalue < 0.05} / n_reps
    err        = mean_k |cov_eff[k] - 0.95| + |rej_eff - 0.05|

Pattern otherwise mirrors fixture_euler.py (run 1).
"""

import json
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts" / "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import ladder_mc as lm  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
from emu_gmm import estimate  # noqa: E402

TRUTH = lm.BETA_TRUE  # Beta(beta0, beta1); exact by construction
PARAM_NAMES = ("beta0", "beta1")
LEVEL = 0.95
ALPHA = 0.05
_Z = float(scipy.stats.norm.ppf(0.5 + LEVEL / 2.0))  # matches studies.coverage

REGISTRY = {
    # Fitness fixtures (both enter every candidate's score; 96 reps
    # ratified 2026-07-11).
    "fb_few_eps_k31": dict(
        spec=dict(n_strata=3, psu_per_cell=2, psu_size=50, collinear_eps=1e-2),
        key=31,
        n_reps=96,
        role="fitness",
    ),
    "fb_strathet_eps_k32": dict(
        spec=dict(
            n_strata=6,
            psu_per_cell=2,
            psu_size=25,
            p_x=0.7,
            p_w=0.5,
            sigma_strat=0.7,
            collinear_eps=2e-2,
        ),
        key=32,
        n_reps=96,
        role="fitness",
    ),
    # Hold-outs: NEVER scored during evolution.  fb_mid is an unseen
    # binding spec (every knob and the key differ from both fitness
    # fixtures); fb_benign is the do-no-harm arm (no collinear moment,
    # comfortable PSU counts, binding ~ 0).
    "fb_mid_eps_k41": dict(
        spec=dict(
            n_strata=4,
            psu_per_cell=3,
            psu_size=30,
            p_x=0.8,
            p_w=0.6,
            sigma_strat=0.35,
            collinear_eps=1.5e-2,
        ),
        key=41,
        n_reps=96,
        role="holdout",
    ),
    "fb_benign_k42": dict(
        spec=dict(n_strata=6, psu_per_cell=4, psu_size=25),
        key=42,
        n_reps=96,
        role="holdout",
    ),
}


def cache_dir():
    d = (
        Path(
            os.environ.get(
                "EMU_EVOLVE_CACHE",
                "/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/"
                "fixtures",
            )
        )
        / "binding"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _design_for(spec_kwargs):
    spec = lm.DesignSpec(**spec_kwargs)
    return spec, lm.make_design(spec)


def run_replicates(config_objects, fixture):
    """Estimate every CRN replicate with PER-REP ridge anchoring (#142).

    ``config_objects`` is the dict produced by
    evaluator_ridge.build_objects: keys ``weighting`` /
    ``regularization`` / ``optimizer``, each possibly None (= package
    default).  Returns the registered stats dict.
    """
    spec, design = _design_for(fixture["spec"])
    cov = lm.covariance_arm(design, "design_aware")
    model, moment_names = lm.model_for(spec)
    master = jax.random.PRNGKey(fixture["key"])
    n_reps = int(fixture["n_reps"])

    truth = np.array([TRUTH.beta0, TRUTH.beta1])
    covered = np.zeros(len(PARAM_NAMES), dtype=int)
    n_valid_se = np.zeros(len(PARAM_NAMES), dtype=int)
    rejections = 0  # adversarial: not-converged / NaN-J count as rejected
    n_used = 0
    iters_used = []
    binding = 0
    meat_indef = 0

    for r in range(n_reps):
        measure = lm.draw_measure(design, jax.random.fold_in(master, r))
        res = estimate(
            model=model,
            measure=measure,
            covariance=cov,
            weighting=config_objects.get("weighting"),
            regularization=config_objects.get("regularization"),
            optimizer=config_objects.get("optimizer"),
            theta_init=TRUTH,
            moment_names=moment_names,
        )
        conv = bool(res.converged)
        if not conv:
            rejections += 1  # a broken rep is adversarially a rejection
        else:
            n_used += 1
            iters_used.append(int(res.iterations))
            law = res.asymptotic()
            th = np.array([float(res.theta_hat.beta0), float(res.theta_hat.beta1)])
            se = np.asarray(law.se(), dtype=float).reshape(-1)
            jp = float(law.J_pvalue)
            for k in range(len(PARAM_NAMES)):
                if np.isfinite(se[k]):
                    n_valid_se[k] += 1
                    if abs(th[k] - truth[k]) <= _Z * se[k]:
                        covered[k] += 1
            if (not np.isfinite(jp)) or jp < ALPHA:
                rejections += 1
            d = res.diagnostics
            binding += int(bool(getattr(d, "binding_ridge", False)))
            meat_indef += int(not np.all(np.isfinite(se)))
        # Bare estimate() accumulates write-only traces (~14 MB/call,
        # #139); clearing every rep triples wall-clock (measured in the
        # #130 study) -- clear every 25 reps, as ladder_mc does.
        if (r + 1) % 25 == 0:
            jax.clear_caches()

    cov_eff = covered / float(n_reps)
    rej_eff = rejections / float(n_reps)
    err = float(np.mean(np.abs(cov_eff - LEVEL)) + abs(rej_eff - ALPHA))
    return dict(
        err=err,
        cov_eff=[float(c) for c in cov_eff],
        rej_eff=float(rej_eff),
        n_reps=n_reps,
        n_used=int(n_used),
        n_valid_se=[int(v) for v in n_valid_se],
        mean_iters=(float(np.mean(iters_used)) if iters_used else float("nan")),
        binding_frequency=(binding / float(n_used) if n_used else float("nan")),
        meat_indefinite_frac=(meat_indef / float(n_used) if n_used else float("nan")),
    )


def build_baseline(name, force=False):
    """Registered baseline = all-defaults config (CUE + default
    DiagonalTikhonov + optimistix_lm defaults)."""
    path = cache_dir() / (name + ".json")
    if path.exists() and not force:
        return json.loads(path.read_text())
    fixture = REGISTRY[name]
    stats = run_replicates({}, fixture)
    spec, _ = _design_for(fixture["spec"])
    meta = dict(
        name=name,
        fixture=fixture,
        n_obs=spec.n_obs,
        baseline=stats,
        theta_true=dict(beta0=float(TRUTH.beta0), beta1=float(TRUTH.beta1)),
    )
    path.write_text(json.dumps(meta, indent=2))
    return meta


def main(argv):
    if len(argv) >= 2 and argv[1] == "build":
        names = argv[2:] or list(REGISTRY)
        for name in names:
            print("building baseline", name, "...", flush=True)
            print(json.dumps(build_baseline(name), indent=2), flush=True)
        return 0
    print("usage: fixture_binding.py build [name ...]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
