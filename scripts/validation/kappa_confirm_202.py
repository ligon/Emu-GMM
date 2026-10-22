"""Confirmatory measurement for issue #202 (kappa_target default review).

Independent fixture family per the #202 protocol comment (2026-07-12,
BINDING): the bundled Euler DGP on the EMPIRICAL path with per-moment
MCAR masks and ``IIDCovariance`` -- the available-pairs conditioning
risk of CLAUDE.md commitment 3 / issue #111, distinct from the
design-aware-glue indefiniteness that produced the run-2 evidence.

Protocol (pre-declared; do not deviate silently):

* Fixture selection is BASELINE-ONLY (all-defaults, kappa_target=1e6).
  Titrate ``(n_sim, per-moment mask probabilities, key)`` until one
  BINDING fixture (binding_frequency >= 0.15, converged_frac >= 0.5)
  and one BENIGN fixture (binding <= 0.05) are found; freeze both
  (the ``REGISTRY`` below) before any non-default kappa is run.
* Comparison: kappa_target in {1e2, 1e3, 1e6}, CRN-paired (same
  ``fold_in`` keys) on both frozen fixtures at 96 reps.
* Metric: the run-2 adversarial calibration error, copied EXACTLY from
  ``evolve/fixture_binding.py`` (branch evolve-harness):

      cov_eff[k] = #{reps: converged AND finite se_k AND
                     |theta_k - truth_k| <= z_.975 * se_k} / n_reps
      rej_eff    = #{reps: NOT converged OR NOT finite J_pvalue OR
                     J_pvalue < 0.05} / n_reps
      err        = mean_k |cov_eff[k] - 0.95| + |rej_eff - 0.05|

  Every registered replicate is in the denominator; a candidate cannot
  improve by breaking hard replicates.  ``binding_frequency`` follows
  the same source: #{converged reps with binding_ridge} / n_used.

Anchoring is PER-REP (bare ``estimate()`` per replicate, emu-gmm #142):
a ``build_estimator`` factory would freeze replicate 0's ``tau_anchor``
across the study, making regularizer knobs meaningless to evaluate.
``jax.clear_caches()`` every 25 reps (the ~14 MB/call bare-estimate
leak, #139).

Usage
-----
    python scripts/validation/kappa_confirm_202.py probe N P0,P1,P2 KEY [REPS] [BLOCK]
        Titration probe: baseline (defaults, kappa=1e6) only.
    python scripts/validation/kappa_confirm_202.py final [NAME ...]
        96-rep baseline on the frozen REGISTRY fixtures; saves JSON.
    python scripts/validation/kappa_confirm_202.py compare
        The 3x2 kappa comparison on the frozen fixtures; saves JSON
        per (kappa, fixture) plus summary.json with the criteria.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
from emu_gmm import estimate  # noqa: E402
from emu_gmm.covariance import IIDCovariance  # noqa: E402
from emu_gmm.examples.euler import (  # noqa: E402
    BETA_TRUE,
    GAMMA_TRUE,
    N_ASSETS,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import EmpiricalMeasure  # noqa: E402
from emu_gmm.regularization import DiagonalTikhonov  # noqa: E402

TRUTH = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
PARAM_NAMES = ("beta", "gamma")
M = N_ASSETS  # 3 moments, K = 2 -> one over-identifying restriction
LEVEL = 0.95
ALPHA = 0.05
_Z = float(scipy.stats.norm.ppf(0.5 + LEVEL / 2.0))  # matches studies.coverage

KAPPA_GRID = (1.0e2, 1.0e3, 1.0e6)  # 1e3 is the default candidate; 1e2 dose-response
N_REPS_FINAL = 96
N_REPS_PROBE = 24

OUT_DIR = Path(
    "/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/kappa_confirm_202"
)

# ---------------------------------------------------------------------------
# Frozen fixtures -- frozen per #202 protocol (baseline-only titration,
# 2026-07-12; see the titration history in the issue thread / summary.json).
# Selection statistics were computed at kappa defaults (1e6) ONLY, first on
# 24-rep probes and then confirmed on the 96-rep baseline below, before any
# other kappa was run.
#
# Titration note: in the moderate-mask regime the protocol comment suggests
# (n ~ 100-200, p like (1.0, 0.6, 0.5) or lower), kappa(V) tops out near
# 2.7e3 and binding is identically zero -- the IID pairwise-overlap V is a
# sum of outer products (PSD by construction), so DiagonalTikhonov binds on
# this family only through extreme conditioning, which needs extreme
# per-moment support disparity.  The binding fixture is therefore the
# rare-moment regime: two fully-observed moments plus one moment supported
# by a handful of observations (E[N_3] = n * p_3 = 3.2), which puts
# kappa(V) right at the 1e6 cliff (probe median 9.7e5).
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    "euler_binding": dict(
        n_sim=8000,
        p=(1.0, 1.0, 0.0004),
        key=51,
        block=None,  # per-observation MCAR
        n_reps=N_REPS_FINAL,
        role="binding",
    ),
    "euler_benign": dict(
        n_sim=800,
        p=(1.0, 0.9, 0.85),
        key=52,
        block=None,  # per-observation MCAR
        n_reps=N_REPS_FINAL,
        role="benign",
    ),
}


def draw_measure(spec: dict, master: jax.Array, r: int) -> EmpiricalMeasure:
    """CRN replicate ``r``: Euler draw + per-moment Bernoulli MCAR mask.

    ``fold_in(master, r)`` is the package CRN contract; the rep key is
    split into a data key and a mask key so the kappa arms see identical
    (data, mask) pairs.  ``block=B`` switches to block-level missingness
    (contiguous blocks of ``B`` observations share one Bernoulli draw
    per moment) -- the protocol's fallback if per-observation MCAR
    cannot reach the binding threshold.
    """
    n_sim = int(spec["n_sim"])
    p = jnp.asarray(spec["p"], dtype=jnp.float64)
    key_r = jax.random.fold_in(master, r)
    k_data, k_mask = jax.random.split(key_r)
    data = euler_sampler_factory(n_sim)(k_data, TRUTH)
    block = spec.get("block")
    if block is None:
        u = jax.random.uniform(k_mask, (n_sim, M))
    else:
        b = int(block)
        n_blocks = -(-n_sim // b)  # ceil
        u_blocks = jax.random.uniform(k_mask, (n_blocks, M))
        u = jnp.repeat(u_blocks, b, axis=0)[:n_sim]
    mask = (u < p[None, :]).astype(jnp.float64)
    return EmpiricalMeasure(x=data, mask=mask, weights=jnp.ones(n_sim))


def run_replicates(spec: dict, kappa: float | None, n_reps: int) -> dict:
    """Estimate every CRN replicate with PER-REP ridge anchoring.

    ``kappa=None`` means all-defaults (``regularization=None``, i.e.
    ``DiagonalTikhonov(kappa_target=1e6)``); a float builds the explicit
    ``DiagonalTikhonov(kappa_target=kappa)``.  Accounting mirrors
    ``evolve/fixture_binding.run_replicates`` exactly; a replicate that
    raises is counted as non-converged (adversarially: a rejection).
    """
    reg = None if kappa is None else DiagonalTikhonov(kappa_target=float(kappa))
    master = jax.random.PRNGKey(int(spec["key"]))
    truth = np.array([BETA_TRUE, GAMMA_TRUE])

    covered = np.zeros(len(PARAM_NAMES), dtype=int)
    n_valid_se = np.zeros(len(PARAM_NAMES), dtype=int)
    rejections = 0  # adversarial: not-converged / NaN-J count as rejected
    n_used = 0
    iters_used: list[int] = []
    binding = 0
    n_errors = 0
    records: list[dict] = []

    t0 = time.time()
    for r in range(n_reps):
        measure = draw_measure(spec, master, r)
        rec: dict = {"rep": r}
        try:
            res = estimate(
                model=euler_residual,
                measure=measure,
                covariance=IIDCovariance(),
                weighting=None,  # CUE default
                regularization=reg,
                optimizer=None,  # LM defaults
                theta_init=TRUTH,
            )
        except Exception as exc:  # zero-support masks etc.: a broken rep
            n_errors += 1
            rejections += 1
            rec.update(converged=False, error=f"{type(exc).__name__}: {exc}")
            records.append(rec)
            if (r + 1) % 25 == 0:
                jax.clear_caches()
            continue
        conv = bool(res.converged)
        d = res.diagnostics
        rec.update(
            converged=conv,
            iterations=int(res.iterations),
            beta=float(res.theta_hat.beta),
            gamma=float(res.theta_hat.gamma),
            binding_ridge=bool(np.asarray(d.binding_ridge)),
            tau_realised=float(d.tau_realised),
            kappa_V=float(d.kappa_V),
        )
        if not conv:
            rejections += 1  # a broken rep is adversarially a rejection
        else:
            n_used += 1
            iters_used.append(int(res.iterations))
            law = res.asymptotic()
            th = np.array([float(res.theta_hat.beta), float(res.theta_hat.gamma)])
            se = np.asarray(law.se(), dtype=float).reshape(-1)
            jp = float(law.J_pvalue)
            rec.update(se_beta=float(se[0]), se_gamma=float(se[1]), J_pvalue=jp)
            for k in range(len(PARAM_NAMES)):
                if np.isfinite(se[k]):
                    n_valid_se[k] += 1
                    if abs(th[k] - truth[k]) <= _Z * se[k]:
                        covered[k] += 1
            if (not np.isfinite(jp)) or jp < ALPHA:
                rejections += 1
            binding += int(bool(np.asarray(d.binding_ridge)))
        records.append(rec)
        # Bare estimate() accumulates write-only traces (~14 MB/call, #139);
        # clear every 25 reps, as fixture_binding.py does.
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
        converged_frac=n_used / float(n_reps),
        n_valid_se=[int(v) for v in n_valid_se],
        n_errors=int(n_errors),
        mean_iters=(float(np.mean(iters_used)) if iters_used else float("nan")),
        binding_frequency=(binding / float(n_used) if n_used else float("nan")),
        wall_seconds=float(time.time() - t0),
        records=records,
    )


def _fmt(stats: dict) -> str:
    return (
        f"err {stats['err']:.4f}  cov_eff {stats['cov_eff']}"
        f"  rej_eff {stats['rej_eff']:.3f}"
        f"  used {stats['n_used']}/{stats['n_reps']}"
        f"  binding {stats['binding_frequency']:.3f}"
        f"  mean_iters {stats['mean_iters']:.1f}"
        f"  errors {stats['n_errors']}"
        f"  ({stats['wall_seconds']:.0f}s)"
    )


def _save(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def cmd_probe(argv: list[str]) -> int:
    """Titration probe. DEFAULTS ONLY (anti-peeking rule): kappa=1e6."""
    n_sim = int(argv[0])
    p = tuple(float(x) for x in argv[1].split(","))
    key = int(argv[2])
    n_reps = int(argv[3]) if len(argv) > 3 else N_REPS_PROBE
    block = int(argv[4]) if len(argv) > 4 else None
    spec = dict(n_sim=n_sim, p=p, key=key, block=block)
    stats = run_replicates(spec, kappa=None, n_reps=n_reps)
    taus = [r["tau_realised"] for r in stats["records"] if "tau_realised" in r]
    kappas = [r["kappa_V"] for r in stats["records"] if "kappa_V" in r]
    print(f"probe n_sim={n_sim} p={p} key={key} block={block} reps={n_reps}")
    print("  " + _fmt(stats))
    if kappas:
        q = np.percentile(kappas, [50, 90, 100])
        print(
            f"  kappa_V p50/p90/max: {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}"
            f"   tau>0 frac: {np.mean(np.asarray(taus) > 0):.2f}"
            f"   tau max: {max(taus):.3g}"
        )
    return 0


def cmd_final(argv: list[str]) -> int:
    """96-rep baseline (defaults) on the frozen fixtures; saves JSON."""
    names = argv or list(REGISTRY)
    for name in names:
        spec = REGISTRY[name]
        stats = run_replicates(spec, kappa=None, n_reps=int(spec["n_reps"]))
        payload = dict(fixture=name, spec=_spec_json(spec), kappa="default(1e6)")
        payload.update(stats)
        path = _save(f"baseline_{name}", payload)
        print(f"final {name}: {_fmt(stats)}\n  -> {path}")
    return 0


def _spec_json(spec: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}


def _strip(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "records"}


def cmd_compare(_argv: list[str]) -> int:
    """kappa in {1e2, 1e3, 1e6} x frozen fixtures, CRN-paired, 96 reps."""
    results: dict[str, dict[str, dict]] = {}
    for name, spec in REGISTRY.items():
        results[name] = {}
        for kappa in KAPPA_GRID:
            tag = f"{int(kappa):.0e}".replace("+0", "").replace("+", "")
            stats = run_replicates(spec, kappa=kappa, n_reps=int(spec["n_reps"]))
            payload = dict(fixture=name, spec=_spec_json(spec), kappa_target=kappa)
            payload.update(stats)
            _save(f"compare_{name}_kappa{tag}", payload)
            results[name][f"{kappa:.0e}"] = _strip(stats)
            print(f"{name} kappa={kappa:.0e}: {_fmt(stats)}")

    bind = results["euler_binding"]
    ben = results["euler_benign"]
    b16, b13 = bind["1e+06"], bind["1e+03"]
    n16, n13 = ben["1e+06"], ben["1e+03"]
    # Criterion 1 (binding fixture): kappa=1e3 err <= 0.9 * baseline err
    # AND n_used >= baseline n_used.
    crit1 = bool(b13["err"] <= 0.9 * b16["err"] and b13["n_used"] >= b16["n_used"])
    # Criterion 2 (benign fixture): kappa=1e3 err <= baseline err + 0.02
    # AND n_used >= baseline n_used - 1.
    crit2 = bool(n13["err"] <= n16["err"] + 0.02 and n13["n_used"] >= n16["n_used"] - 1)
    summary = dict(
        registry={n: _spec_json(s) for n, s in REGISTRY.items()},
        results=results,
        criteria=dict(
            criterion_1_binding=dict(
                passed=crit1,
                err_1e3=b13["err"],
                err_baseline=b16["err"],
                err_bar=0.9 * b16["err"],
                n_used_1e3=b13["n_used"],
                n_used_baseline=b16["n_used"],
            ),
            criterion_2_benign=dict(
                passed=crit2,
                err_1e3=n13["err"],
                err_baseline=n16["err"],
                err_bar=n16["err"] + 0.02,
                n_used_1e3=n13["n_used"],
                n_used_baseline=n16["n_used"],
            ),
            criterion_3_make_check="out of scope for this pass (later PR)",
        ),
    )
    path = _save("summary", summary)
    print(f"criterion 1 (binding): {'PASS' if crit1 else 'FAIL'}")
    print(f"criterion 2 (benign):  {'PASS' if crit2 else 'FAIL'}")
    print(f"-> {path}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "probe":
        return cmd_probe(argv[2:])
    if len(argv) >= 2 and argv[1] == "final":
        return cmd_final(argv[2:])
    if len(argv) >= 2 and argv[1] == "compare":
        return cmd_compare(argv[2:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
