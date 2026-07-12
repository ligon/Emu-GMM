"""#205 measurement: how often do POST-ESTIMATION ridge bisections saturate?

Question (decides whether PR #208's ``tau_saturated`` flag needs surfacing
beyond the estimator anchor): conditional on the FIT's anchor bisection
being clean, how often do the independent ``regularization.apply()``
bisections inside the inference helpers saturate at the SHIPPING default
``DiagonalTikhonov()`` (kappa_target=1e6)?

The apply() sites measured (read from the code on branch
feat/205-tau-saturated):

a. ANCHOR --- ``estimate()``'s anchor-once bisection at ``theta_init``
   (estimator.py ~613); read off ``result.diagnostics.tau_saturated``.
b. THETA-HAT SITE --- ``j_test`` (j_test.py:186) and
   ``moment_wild_bootstrap`` (wild_bootstrap.py:409) both repair
   ``V(theta_point)`` the SAME way before whitening:
   ``V_raw = covariance.covariance(model, theta, measure)`` then
   ``regularization.apply(V_raw)`` (j_test at ``theta_null``, the wild
   bootstrap ONCE at ``theta_hat`` --- per-draw work is Cholesky solves
   against the one repaired factor, no per-draw bisection). Measured here
   at ``theta_hat``: build V_raw exactly as those two lines do and record
   ``reg.tau_saturated(V_raw, tau)``.
c. K-GRID SITE --- the confidence-set inversion pattern:
   ``k_confidence_set`` calls ``k_statistic`` per grid point, and
   ``_k_statistic_arrays`` (k_statistic.py:555-556) runs
   ``V = covariance.covariance(model, theta_0, measure);
   V_star, _tau = regularization.apply(V)`` --- ONE fresh bisection per
   null point. Measured on a 16-point grid of nulls around the fixture
   truth: per-parameter axis offsets {+-1, +-2, +-4} x scale_k (12
   points) plus the 4 corners of the +-4 box. ``scale_k`` is the MC sd
   of ``theta_hat`` over this run's converged, non-zero-support reps
   (for the euler fixtures this reproduces the cached #202 baseline sds
   --- printed for cross-check). "Edge" points are those with
   max |multiplier| == 4 (the 4 axis extremes + 4 corners = 8 of 16).
d. CLUSTER_BOOTSTRAP SITE --- ``cluster_bootstrap`` REFITS via bare
   ``estimate()`` per draw (cluster_bootstrap.py; no per-draw diagnostics
   on ``ClusterBootstrapResult``), so every draw runs its own anchor
   bisection on the resampled world's V. Measured by mirroring its loop
   with its own private helpers (``_cluster_row_indices`` /
   ``_resample_one``, same ``randint`` draw mechanics) and reading each
   refit's ``diagnostics.tau_saturated``. Applicability caveats:
   ``cluster_bootstrap`` is typed to ``ClusteredCovariance``; the euler
   fixtures use ``IIDCovariance``, so the closest faithful application is
   SINGLETON clusters (IID row bootstrap) --- the refit covariance is then
   ``ClusteredCovariance(n_clusters=N)`` rather than ``IIDCovariance``
   (documented family substitution). The fb (design-aware) fixtures are
   SKIPPED: ``DesignAwareCovariance`` is not a ``ClusteredCovariance``,
   and PSU resampling breaks the (stratum x arm) layout its V_TT needs
   --- the helper cannot refit the fixture's covariance family at all.

(No independent bisection exists in ``identification.py`` --- it reads
``result.gn_hessian`` precisely to AVOID re-anchoring a different tau,
per the PR #178 review note at identification.py:429.)

Fixtures (frozen elsewhere; specs copied verbatim, provenance in
comments): the #202 euler-mask pair (branch confirm/202-kappa-euler-mask,
``scripts/validation/kappa_confirm_202.py`` REGISTRY) and the run-2
binding fitness pair (``evolve/fixture_binding.py`` REGISTRY, ladder_mc
design-aware regime). 96 CRN reps each, PER-REP anchoring (bare
``estimate()``, emu-gmm #142), ``theta_init = truth``, ALL at default
``DiagonalTikhonov()``. ``jax.clear_caches()`` every 25 reps/draws (the
~14 MB/call bare-estimate leak, #139).

Denominator discipline (#140 house style): every frequency is reported
with its denominator; zero-support reps (any ``N_j == 0``, the known
anchor-saturation cases) are tracked separately everywhere; the headline
per site is conditional-on-anchor-clean.

Usage
-----
    python scripts/validation/site_saturation_205.py main [FIXTURE ...]
        Sites a+b+c on all (or named) fixtures; JSON per fixture.
    python scripts/validation/site_saturation_205.py boot [FIXTURE ...]
        Site d on the euler fixtures (first 24 reps, B=48 draws).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from emu_gmm import estimate  # noqa: E402
from emu_gmm.covariance import ClusteredCovariance, IIDCovariance  # noqa: E402
from emu_gmm.examples.euler import (  # noqa: E402
    BETA_TRUE,
    GAMMA_TRUE,
    N_ASSETS,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.inference.cluster_bootstrap import (  # noqa: E402
    _cluster_row_indices,
    _resample_one,
)
from emu_gmm.measures import EmpiricalMeasure  # noqa: E402
from emu_gmm.regularization import DiagonalTikhonov  # noqa: E402

import ladder_mc as lm  # noqa: E402  (scripts/validation on sys.path)

OUT_DIR = Path(
    "/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/"
    "site_saturation_205"
)

N_REPS = 96
GRID_AXIS_MULTS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
GRID_CORNER_MULT = 4.0
BOOT_N_PARENT_REPS = 24
BOOT_N_DRAWS = 48

# Frozen fixture registry. euler_* copied verbatim from
# confirm/202-kappa-euler-mask:scripts/validation/kappa_confirm_202.py
# (frozen 2026-07-12, baseline-only titration); fb_* copied verbatim from
# evolve/fixture_binding.py (ratified 2026-07-11).
REGISTRY: dict[str, dict] = {
    "euler_binding": dict(
        family="euler", n_sim=8000, p=(1.0, 1.0, 0.0004), key=51
    ),
    "euler_benign": dict(family="euler", n_sim=800, p=(1.0, 0.9, 0.85), key=52),
    "fb_few_eps_k31": dict(
        family="fb",
        spec=dict(n_strata=3, psu_per_cell=2, psu_size=50, collinear_eps=1e-2),
        key=31,
    ),
    "fb_strathet_eps_k32": dict(
        family="fb",
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
    ),
}

# Cached #202 baseline MC sds (converged reps of the identical CRN run;
# kappa_confirm_202 baseline_euler_*.json) --- printed as a cross-check
# against this run's own scale computation.
CACHED_EULER_SD = {
    "euler_binding": (0.004659165273973566, 0.241472090336274),
    "euler_benign": (0.01866499874215317, 0.803817530159308),
}

EULER_TRUTH = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
FB_TRUTH = lm.BETA_TRUE  # Beta(beta0=0.5, beta1=1.0)


def _plain(a):
    return a.array if hasattr(a, "array") else jnp.asarray(a)


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------


class Fixture:
    """Uniform surface over the two fixture families."""

    def __init__(self, name: str):
        self.name = name
        spec = REGISTRY[name]
        self.family = spec["family"]
        self.key = int(spec["key"])
        self.master = jax.random.PRNGKey(self.key)
        if self.family == "euler":
            self.n_sim = int(spec["n_sim"])
            self.p = tuple(float(x) for x in spec["p"])
            self.model = euler_residual
            self.moment_names = None
            self.covariance = IIDCovariance()
            self.truth = EULER_TRUTH
            self.param_names = ("beta", "gamma")
            self.truth_vec = np.array([BETA_TRUE, GAMMA_TRUE])
        else:
            self.spec = lm.DesignSpec(**spec["spec"])
            self.design = lm.make_design(self.spec)
            self.covariance = lm.covariance_arm(self.design, "design_aware")
            self.model, self.moment_names = lm.model_for(self.spec)
            self.truth = FB_TRUTH
            self.param_names = ("beta0", "beta1")
            self.truth_vec = np.array([FB_TRUTH.beta0, FB_TRUTH.beta1])

    def draw_measure(self, r: int) -> EmpiricalMeasure:
        key_r = jax.random.fold_in(self.master, r)
        if self.family == "euler":
            # Verbatim kappa_confirm_202.draw_measure (per-observation MCAR).
            k_data, k_mask = jax.random.split(key_r)
            data = euler_sampler_factory(self.n_sim)(k_data, EULER_TRUTH)
            u = jax.random.uniform(k_mask, (self.n_sim, N_ASSETS))
            mask = (u < jnp.asarray(self.p)[None, :]).astype(jnp.float64)
            return EmpiricalMeasure(
                x=data, mask=mask, weights=jnp.ones(self.n_sim)
            )
        return lm.draw_measure(self.design, key_r)

    def theta_at(self, vec: np.ndarray):
        if self.family == "euler":
            return EulerParams(beta=float(vec[0]), gamma=float(vec[1]))
        return lm.Beta(beta0=float(vec[0]), beta1=float(vec[1]))

    def theta_vec(self, theta) -> np.ndarray:
        return np.array([float(getattr(theta, n)) for n in self.param_names])

    def estimate(self, measure, theta_init=None):
        return estimate(
            model=self.model,
            measure=measure,
            covariance=self.covariance,
            weighting=None,  # CUE default
            regularization=None,  # DiagonalTikhonov() default, kappa 1e6
            optimizer=None,  # LM defaults
            theta_init=self.truth if theta_init is None else theta_init,
            moment_names=self.moment_names,
        )


def _grid_points(scale: np.ndarray) -> list[tuple[tuple[float, float], bool]]:
    """(multiplier pair, is_edge) for the 16-point grid around truth."""
    pts: list[tuple[tuple[float, float], bool]] = []
    for k in range(2):
        for mult in GRID_AXIS_MULTS:
            m = [0.0, 0.0]
            m[k] = mult
            pts.append(((m[0], m[1]), abs(mult) == GRID_CORNER_MULT))
    for s0 in (-GRID_CORNER_MULT, GRID_CORNER_MULT):
        for s1 in (-GRID_CORNER_MULT, GRID_CORNER_MULT):
            pts.append(((s0, s1), True))
    return pts


# ---------------------------------------------------------------------------
# Sites a + b + c
# ---------------------------------------------------------------------------


def run_main(fixture_name: str) -> dict:
    fx = Fixture(fixture_name)
    reg = DiagonalTikhonov()  # shipping default, kappa_target=1e6
    records: list[dict] = []
    t0 = time.time()

    # --- Pass 1: fit (site a) + theta-hat site (site b) -------------------
    for r in range(N_REPS):
        measure = fx.draw_measure(r)
        n_j = np.asarray(measure.mask).sum(axis=0)
        rec: dict = {
            "rep": r,
            "n_j": [float(x) for x in n_j],
            "zero_support": bool((n_j == 0).any()),
        }
        try:
            res = fx.estimate(measure)
        except Exception as exc:  # zero-support etc.: a broken rep
            rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            if (r + 1) % 25 == 0:
                jax.clear_caches()
            continue
        d = res.diagnostics
        rec.update(
            converged=bool(res.converged),
            iterations=int(res.iterations),
            theta_hat=[float(x) for x in fx.theta_vec(res.theta_hat)],
            anchor_binding=bool(np.asarray(d.binding_ridge)),
            anchor_tau=float(d.tau_realised),
            anchor_kappa_V=float(d.kappa_V),
            anchor_sat=bool(np.asarray(d.tau_saturated)),
        )
        # Site b: the j_test:186 / wild_bootstrap:409 construction at
        # theta_hat --- verified identical in both helpers (V from
        # covariance.covariance at the evaluation point, then apply()).
        V_raw = _plain(fx.covariance.covariance(fx.model, res.theta_hat, measure))
        _V_star, tau_hat = reg.apply(V_raw)
        rec.update(
            hat_tau=float(tau_hat),
            hat_sat=bool(np.asarray(reg.tau_saturated(V_raw, tau_hat))),
        )
        records.append(rec)
        if (r + 1) % 25 == 0:
            jax.clear_caches()
        print(
            f"  [{fixture_name}] rep {r}: conv={rec.get('converged')} "
            f"anchor_sat={rec.get('anchor_sat')} hat_sat={rec.get('hat_sat')} "
            f"zero_support={rec['zero_support']} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # --- Grid scale: MC sd of theta_hat over converged, non-zero-support
    # reps of THIS run (documented choice; euler cross-check printed). ----
    ths = np.array(
        [
            rec["theta_hat"]
            for rec in records
            if rec.get("converged") and not rec["zero_support"]
        ]
    )
    scale = ths.std(axis=0)
    scale_source = "mc_sd_this_run_converged_nonzero_support"
    if len(ths) < 8 or not np.all(np.isfinite(scale)) or np.any(scale == 0):
        cached = CACHED_EULER_SD.get(fixture_name)
        if cached is None:
            raise RuntimeError(
                f"{fixture_name}: cannot form a grid scale "
                f"({len(ths)} usable reps)"
            )
        scale = np.asarray(cached)
        scale_source = "cached_202_baseline_sd_fallback"
    if fixture_name in CACHED_EULER_SD:
        print(
            f"  [{fixture_name}] scale cross-check: this run "
            f"{scale.tolist()} vs cached #202 baseline "
            f"{list(CACHED_EULER_SD[fixture_name])}",
            flush=True,
        )

    # --- Pass 2: K-grid site (site c), measures regenerated by CRN. -------
    grid = _grid_points(scale)
    t1 = time.time()
    for r, rec in enumerate(records):
        measure = fx.draw_measure(r)
        pts = []
        for (m0, m1), edge in grid:
            theta_0 = fx.theta_at(
                fx.truth_vec + np.array([m0, m1]) * scale
            )
            # Verbatim the k_statistic.py:555-556 repair.
            V = _plain(fx.covariance.covariance(fx.model, theta_0, measure))
            _V_star, tau = reg.apply(V)
            sat = bool(np.asarray(reg.tau_saturated(V, tau)))
            pts.append(
                dict(
                    mult=[m0, m1],
                    tau=float(tau),
                    sat=sat,
                    edge=bool(edge),
                )
            )
        n_sat = sum(p["sat"] for p in pts)
        rec["grid"] = dict(
            points=pts,
            any_sat=bool(n_sat > 0),
            n_sat=int(n_sat),
            n_points=len(pts),
            frac_sat=n_sat / len(pts),
            n_edge_sat=int(sum(p["sat"] and p["edge"] for p in pts)),
            n_interior_sat=int(sum(p["sat"] and not p["edge"] for p in pts)),
        )
        if (r + 1) % 25 == 0:
            jax.clear_caches()
        print(
            f"  [{fixture_name}] grid rep {r}: any_sat={rec['grid']['any_sat']} "
            f"frac={rec['grid']['frac_sat']:.2f} ({time.time() - t1:.0f}s)",
            flush=True,
        )

    summary = _summarize_main(records)
    payload = dict(
        fixture=fixture_name,
        registry_spec={
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in REGISTRY[fixture_name].items()
        },
        kappa_target=1e6,
        n_reps=N_REPS,
        grid_scale=[float(s) for s in scale],
        grid_scale_source=scale_source,
        grid_mults_axis=list(GRID_AXIS_MULTS),
        grid_corner_mult=GRID_CORNER_MULT,
        wall_seconds=float(time.time() - t0),
        summary=summary,
        records=records,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"main_{fixture_name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"[{fixture_name}] main -> {path}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    return payload


def _freq(num: int, den: int) -> dict:
    return dict(n=int(num), den=int(den), frac=(num / den if den else None))


def _summarize_main(records: list[dict]) -> dict:
    n = len(records)
    errors = [r for r in records if "error" in r]
    fitted = [r for r in records if "error" not in r]
    zs = [r for r in records if r["zero_support"]]
    nzs_fitted = [r for r in fitted if not r["zero_support"]]
    zs_fitted = [r for r in fitted if r["zero_support"]]
    conv = [r for r in fitted if r["converged"]]
    # Anchor-clean: fit ran and its anchor bisection did NOT saturate.
    clean = [r for r in fitted if not r["anchor_sat"]]
    clean_conv = [r for r in clean if r["converged"]]

    def _site_b(rs):
        return _freq(sum(r["hat_sat"] for r in rs), len(rs))

    def _site_c_any(rs):
        rs = [r for r in rs if "grid" in r]
        return _freq(sum(r["grid"]["any_sat"] for r in rs), len(rs))

    grid_clean = [r for r in clean if "grid" in r]
    grid_clean_sat = [r for r in grid_clean if r["grid"]["any_sat"]]
    return dict(
        n_reps=n,
        n_error=len(errors),
        n_zero_support=len(zs),
        zero_support_reps=[r["rep"] for r in zs],
        n_converged=_freq(len(conv), len(fitted)),
        anchor_sat_overall=_freq(sum(r["anchor_sat"] for r in fitted), len(fitted)),
        anchor_sat_zero_support=_freq(
            sum(r["anchor_sat"] for r in zs_fitted), len(zs_fitted)
        ),
        anchor_sat_nonzero_support=_freq(
            sum(r["anchor_sat"] for r in nzs_fitted), len(nzs_fitted)
        ),
        n_anchor_clean=len(clean),
        hat_sat_overall=_site_b(fitted),
        hat_sat_given_anchor_clean=_site_b(clean),
        hat_sat_given_anchor_clean_and_converged=_site_b(clean_conv),
        hat_sat_zero_support=_site_b(zs_fitted),
        grid_any_sat_overall=_site_c_any(fitted),
        grid_any_sat_given_anchor_clean=_site_c_any(clean),
        grid_any_sat_zero_support=_site_c_any(zs_fitted),
        grid_mean_frac_sat_given_anchor_clean=(
            float(np.mean([r["grid"]["frac_sat"] for r in grid_clean]))
            if grid_clean
            else None
        ),
        grid_sat_edge_only_reps_given_anchor_clean=_freq(
            sum(
                r["grid"]["n_interior_sat"] == 0 and r["grid"]["n_sat"] > 0
                for r in grid_clean
            ),
            len(grid_clean_sat),
        ),
    )


# ---------------------------------------------------------------------------
# Site d: cluster_bootstrap per-draw anchor bisections
# ---------------------------------------------------------------------------


def run_boot(fixture_name: str) -> dict:
    fx = Fixture(fixture_name)
    if fx.family != "euler":
        # Documented skip: DesignAwareCovariance is not a
        # ClusteredCovariance; cluster_bootstrap cannot resample or refit
        # the fixture's covariance family (PSU resampling breaks the
        # (stratum x arm) layout V_TT requires).
        payload = dict(
            fixture=fixture_name,
            skipped=True,
            reason=(
                "cluster_bootstrap is typed to ClusteredCovariance; the "
                "fixture's DesignAwareCovariance cannot be resampled/refit "
                "by it (PSU resampling breaks the stratified design layout)."
            ),
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"boot_{fixture_name}.json"
        path.write_text(json.dumps(payload, indent=1))
        print(f"[{fixture_name}] boot SKIPPED -> {path}", flush=True)
        return payload

    t0 = time.time()
    parent_records: list[dict] = []
    for r in range(BOOT_N_PARENT_REPS):
        measure = fx.draw_measure(r)
        n = int(np.asarray(measure.mask).shape[0])
        n_j = np.asarray(measure.mask).sum(axis=0)
        # Parent anchor status (same CRN rep as the main pass).
        try:
            res = fx.estimate(measure)
            parent_anchor_sat = bool(np.asarray(res.diagnostics.tau_saturated))
            parent_converged = bool(res.converged)
        except Exception as exc:
            parent_anchor_sat = None
            parent_converged = False
            print(f"  parent rep {r} fit error: {exc}", flush=True)
        prec: dict = dict(
            rep=r,
            n_j=[float(x) for x in n_j],
            zero_support=bool((n_j == 0).any()),
            parent_anchor_sat=parent_anchor_sat,
            parent_converged=parent_converged,
            draws=[],
        )
        # Mirror cluster_bootstrap's machinery with SINGLETON clusters
        # (IID row bootstrap; refit covariance is then
        # ClusteredCovariance(n_clusters=N) --- see module docstring).
        rows = _cluster_row_indices(np.arange(n, dtype=np.int64), n)
        boot_key = jax.random.fold_in(jax.random.PRNGKey(fx.key), 10_000 + r)
        keys = jax.random.split(boot_key, BOOT_N_DRAWS)
        for b in range(BOOT_N_DRAWS):
            if b > 0 and b % 25 == 0:
                jax.clear_caches()
            drawn = np.asarray(
                jax.random.randint(keys[b], shape=(n,), minval=0, maxval=n)
            )
            boot_measure, boot_cov = _resample_one(measure, rows, drawn)
            bn_j = np.asarray(boot_measure.mask).sum(axis=0)
            drec: dict = dict(
                draw=b, boot_zero_support=bool((bn_j == 0).any())
            )
            try:
                bres = estimate(
                    model=fx.model,
                    measure=boot_measure,
                    covariance=boot_cov,
                    weighting=None,
                    regularization=None,
                    optimizer=None,
                    theta_init=fx.truth,
                )
            except Exception as exc:
                drec["error"] = f"{type(exc).__name__}: {exc}"
                prec["draws"].append(drec)
                continue
            bd = bres.diagnostics
            drec.update(
                converged=bool(bres.converged),
                anchor_sat=bool(np.asarray(bd.tau_saturated)),
                anchor_binding=bool(np.asarray(bd.binding_ridge)),
                anchor_tau=float(bd.tau_realised),
            )
            prec["draws"].append(drec)
        n_sat = sum(d.get("anchor_sat", False) for d in prec["draws"])
        prec["n_draws_sat"] = int(n_sat)
        parent_records.append(prec)
        jax.clear_caches()
        print(
            f"  [{fixture_name}] boot parent rep {r}: "
            f"{n_sat}/{BOOT_N_DRAWS} draws saturated "
            f"(parent_sat={parent_anchor_sat}, "
            f"zero_support={prec['zero_support']}) "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    summary = _summarize_boot(parent_records)
    payload = dict(
        fixture=fixture_name,
        kappa_target=1e6,
        n_parent_reps=BOOT_N_PARENT_REPS,
        n_draws=BOOT_N_DRAWS,
        covariance_note=(
            "singleton-cluster IID row bootstrap; refit covariance is "
            "ClusteredCovariance(n_clusters=N), a documented substitution "
            "for the fixture's IIDCovariance (cluster_bootstrap's typed "
            "requirement)"
        ),
        wall_seconds=float(time.time() - t0),
        summary=summary,
        records=parent_records,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"boot_{fixture_name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"[{fixture_name}] boot -> {path}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    return payload


def _summarize_boot(parent_records: list[dict]) -> dict:
    all_draws = [d for p in parent_records for d in p["draws"]]
    ok = [d for d in all_draws if "error" not in d]
    clean_parents = [p for p in parent_records if p["parent_anchor_sat"] is False]
    clean_draws = [d for p in clean_parents for d in p["draws"] if "error" not in d]
    zs_draws = [d for d in ok if d["boot_zero_support"]]
    nzs_clean = [d for d in clean_draws if not d["boot_zero_support"]]
    return dict(
        n_parents=len(parent_records),
        n_parents_anchor_clean=len(clean_parents),
        n_draws_total=len(all_draws),
        n_draw_errors=len(all_draws) - len(ok),
        draw_sat_overall=_freq(sum(d["anchor_sat"] for d in ok), len(ok)),
        draw_sat_given_parent_anchor_clean=_freq(
            sum(d["anchor_sat"] for d in clean_draws), len(clean_draws)
        ),
        draw_sat_given_parent_clean_and_boot_nonzero_support=_freq(
            sum(d["anchor_sat"] for d in nzs_clean), len(nzs_clean)
        ),
        draw_boot_zero_support=_freq(len(zs_draws), len(ok)),
        draw_sat_among_boot_zero_support=_freq(
            sum(d["anchor_sat"] for d in zs_draws), len(zs_draws)
        ),
        per_parent_sat_counts={
            str(p["rep"]): p["n_draws_sat"] for p in parent_records
        },
    )


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "main":
        names = argv[2:] or list(REGISTRY)
        for name in names:
            print(f"=== main pass: {name} ===", flush=True)
            run_main(name)
        return 0
    if len(argv) >= 2 and argv[1] == "boot":
        names = argv[2:] or list(REGISTRY)
        for name in names:
            print(f"=== boot pass: {name} ===", flush=True)
            run_boot(name)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
