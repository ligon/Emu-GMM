#!/usr/bin/env python
"""Discriminating DGP family + study runner for the #130 ladder MC report.

The Arc-2 exit criterion (issue #130): a committed, reproducible
repeated-sampling validation of the design-awareness covariance ladder
-- size, coverage, and J-calibration under GENUINE missingness -- driven
by ``emu_gmm.studies``.

Fixture-engineering requirements (the recorded #130 lessons):

- **Discriminate or it doesn't count.** Every study must include an arm
  where the *wrong* strategy is detectably mis-calibrated at the chosen
  sample sizes, alongside the arm whose calibration is being claimed.
  A study in which all arms look fine cannot fail, so it validates
  nothing (the Euler-DGP ~99.996%-efficiency lesson).
- **Genuine missingness everywhere.** Balanced ``mask=ones`` designs are
  blind to the per-coordinate N_j bookkeeping (CLAUDE.md commitment 9);
  the family below has unequal N_j (and PSU-level observability, so the
  pair-specific H_{c,jk} machinery of commitment 10 is exercised,
  including cells at the H < 2 boundary).
- **CRN across arms.** Covariance/weighting arms share the master key
  and the dgp callable, so arm contrasts are common-random-number
  comparisons (the documented ``replicate`` contract).

The DGP family
--------------

A stratified, PSU-randomized experiment with intracluster correlation
and per-moment observability:

- ``S`` strata x 2 arms; each (stratum x arm) cell holds ``G_c`` PSUs of
  ``m`` units. Treatment ``T`` is assigned at the PSU level within
  stratum (the Athey-Imbens fine-stratification structure that
  ``StratifiedCovariance`` targets).
- Outcome: ``y = beta0 + beta1*T + e``, with
  ``e = sqrt(rho)*u_g + sqrt(1-rho)*eps_i`` (PSU random effect ->
  intracluster correlation ``rho``; what separates IID from clustered
  inference) plus an optional stratum effect ``sigma_s * v_s`` (absorbed
  by the within-cell centering of the design ladder, not by IID/plain-
  clustered inference -- the V_TT vs V_SS distinction).
- Instruments ``z = (1, T, x, w)`` against the residual
  ``r = y - beta0 - beta1*T`` give ``M = 4`` moments, ``K = 2``,
  ``J_dof = 2``. ``x`` and ``w`` are exogenous covariates with their own
  PSU components (so the x/w moment rows are cluster-correlated too).
- **Missingness**: the ``x`` moment is observed for a PSU with
  probability ``p_x``, the ``w`` moment with probability ``p_w``
  (PSU-level draws -> whole clusters drop out of a moment, driving
  unequal N_j, pair-specific overlap, and H_{c,jk} < 2 cells when
  ``G_c`` is small). The first two moments are always observed.
- **Misspecification knob**: ``delta`` adds ``delta * w`` to ``y``,
  violating only the over-identifying ``E[r w] = 0`` moment
  (``w`` independent of ``(1, T, x)``), so theta stays consistently
  estimable while J gains a detectable signal -- the power /
  misspecified-alternative arm for the CU+LM robustness column.

Usage::

    .venv/bin/python scripts/validation/ladder_mc.py --study size_iid_vs_cluster \
        --n-reps 500 --out docs/validation/data

Each study prints an org-mode table fragment and (with --out) writes the
per-replicate records to CSV next to a .json sidecar carrying the seed,
the spec, and the runtime -- the reproducibility contract for the
committed report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    IIDCovariance,
    StratifiedCovariance,
    build_estimator,
    estimate,
)
from emu_gmm.studies import (
    MCRecords,
    StudyResult,
    bias_sd,
    coverage,
    j_calibration,
    monte_carlo_study,
    size_power,
    tau_binding,
)

# ---------------------------------------------------------------------------
# Parameters and moment function
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class Beta:
    """Intercept + treatment effect; flat R^2 (v1 path)."""

    beta0: float
    beta1: float


BETA_TRUE = Beta(beta0=0.5, beta1=1.0)

# Column layout of the observation matrix (one row per unit).
COL_Y, COL_T, COL_X, COL_W = 0, 1, 2, 3
N_MOMENTS = 4  # r * (1, T, x, w)
MOMENT_NAMES = ("r_const", "r_T", "r_x", "r_w")


def psi(x, theta):
    """Per-observation moment vector ``r * (1, T, x, w)``; (M,) = (4,)."""
    y, T, xv, w = x[COL_Y], x[COL_T], x[COL_X], x[COL_W]
    r = y - theta.beta0 - theta.beta1 * T
    return jnp.array([r, r * T, r * xv, r * w])


# ---------------------------------------------------------------------------
# The DGP family
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DesignSpec:
    """Knobs of the discriminating DGP family (module docstring).

    ``sigma_strat`` scales a FIXED, demeaned stratum profile (generated
    once per design from a fixed seed, NOT redrawn per replicate). A
    per-replicate stratum random effect would put between-stratum
    variance into the repeated-sampling law of every non-contrast
    moment, which no within-cell-centered (or PSU-clustered) estimator
    targets -- the pilot measured intercept coverage of 0.17-0.73
    across ALL arms under that misdesign. With fixed demeaned effects
    the moment conditions hold exactly (the profile sums to zero and is
    balanced across arms), the within-cell-centered stratified variance
    is calibrated, and the UNCENTERED PSU-clustered variance pools
    strata with different means and is conservative -- which is the
    discriminating contrast the stratified study claims.
    """

    n_strata: int = 10
    psu_per_cell: int = 5  # G_c per (stratum x arm) cell
    psu_size: int = 20  # units per PSU
    rho: float = 0.5  # intracluster correlation of e at PSU level
    sigma_strat: float = 0.0  # scale of the FIXED demeaned stratum profile
    x_psu_load: float = 0.7  # PSU component loading in x and w
    p_x: float = 0.9  # PSU-level observability of the x moment
    p_w: float = 0.5  # PSU-level observability of the w moment
    delta: float = 0.0  # misspecification: y += delta * w

    @property
    def n_cells(self) -> int:
        return self.n_strata * 2

    @property
    def n_psu(self) -> int:
        return self.n_cells * self.psu_per_cell

    @property
    def n_obs(self) -> int:
        return self.n_psu * self.psu_size


@dataclasses.dataclass(frozen=True)
class Design:
    """A realised design: static ids + the measure-building closure."""

    spec: DesignSpec
    psu_ids: np.ndarray  # (N,) dense PSU codes
    cell_ids: np.ndarray  # (N,) dense (stratum x arm) codes
    stratum_ids: np.ndarray  # (N,) dense stratum codes
    T: np.ndarray  # (N,) treatment indicator (0/1)
    v_strat: np.ndarray  # (S,) FIXED demeaned stratum profile (see DesignSpec)


def make_design(spec: DesignSpec) -> Design:
    """Lay out the (stratum, arm, PSU, unit) grid and PSU randomization.

    The layout is DETERMINISTIC (balanced assignment: within each
    stratum, the first ``psu_per_cell`` PSUs are treated). Randomness
    lives entirely in :func:`draw_measure`, keyed per replicate; the
    design ids are static so every covariance arm shares them.
    """
    rows_stratum, rows_cell, rows_psu, rows_T = [], [], [], []
    psu = 0
    for s in range(spec.n_strata):
        for arm in (1, 0):
            cell = s * 2 + (1 - arm)  # treated cell first within stratum
            for _g in range(spec.psu_per_cell):
                rows_stratum += [s] * spec.psu_size
                rows_cell += [cell] * spec.psu_size
                rows_psu += [psu] * spec.psu_size
                rows_T += [arm] * spec.psu_size
                psu += 1
    # FIXED demeaned stratum profile (seed independent of the replicate
    # stream; see DesignSpec docstring for why these must not be
    # redrawn per replicate). Demeaning makes E[r * z] = 0 exact.
    rng = np.random.default_rng(777)
    v = rng.standard_normal(spec.n_strata)
    v = v - v.mean()
    return Design(
        spec=spec,
        psu_ids=np.asarray(rows_psu, dtype=np.int64),
        cell_ids=np.asarray(rows_cell, dtype=np.int64),
        stratum_ids=np.asarray(rows_stratum, dtype=np.int64),
        T=np.asarray(rows_T, dtype=np.float64),
        v_strat=v,
    )


def draw_measure(design: Design, key: jax.Array) -> EmpiricalMeasure:
    """One replicate's data + observability -> EmpiricalMeasure.

    All randomness flows from ``key`` (jax PRNG; fold_in-split into
    independent streams), so the ``replicate`` CRN contract holds across
    covariance arms by construction.
    """
    spec = design.spec
    n, n_psu = spec.n_obs, spec.n_psu

    k_u, k_eps, k_vx, k_vw, k_x, k_w, k_dx, k_dw = jax.random.split(key, 8)

    u_psu = jax.random.normal(k_u, (n_psu,))  # outcome PSU effect
    vx_psu = jax.random.normal(k_vx, (n_psu,))  # x PSU component
    vw_psu = jax.random.normal(k_vw, (n_psu,))  # w PSU component
    eps = jax.random.normal(k_eps, (n,))
    x_i = jax.random.normal(k_x, (n,))
    w_i = jax.random.normal(k_w, (n,))

    psu = jnp.asarray(design.psu_ids)
    strat = jnp.asarray(design.stratum_ids)
    T = jnp.asarray(design.T)
    v_strat = jnp.asarray(design.v_strat)  # FIXED demeaned profile

    x = spec.x_psu_load * vx_psu[psu] + x_i
    w = spec.x_psu_load * vw_psu[psu] + w_i
    e = (
        jnp.sqrt(spec.rho) * u_psu[psu]
        + jnp.sqrt(1.0 - spec.rho) * eps
        + spec.sigma_strat * v_strat[strat]
    )
    y = BETA_TRUE.beta0 + BETA_TRUE.beta1 * T + e + spec.delta * w

    data = jnp.stack([y, T, x, w], axis=1)  # (N, 4)

    # PSU-level observability of the x / w moments (genuine, cluster-
    # correlated missingness; commitment 9/10 regime).
    dx_psu = (jax.random.uniform(k_dx, (n_psu,)) < spec.p_x).astype(jnp.float64)
    dw_psu = (jax.random.uniform(k_dw, (n_psu,)) < spec.p_w).astype(jnp.float64)
    ones = jnp.ones((n,))
    mask = jnp.stack([ones, ones, dx_psu[psu], dw_psu[psu]], axis=1)  # (N, 4)

    return EmpiricalMeasure(x=data, mask=mask, weights=jnp.ones(n))


# ---------------------------------------------------------------------------
# Covariance arms
# ---------------------------------------------------------------------------


def covariance_arm(design: Design, arm: str) -> Any:
    """Build the covariance strategy for a named arm."""
    psu = jnp.asarray(design.psu_ids, dtype=jnp.float64)
    cell = jnp.asarray(design.cell_ids, dtype=jnp.float64)
    strat = jnp.asarray(design.stratum_ids, dtype=jnp.float64)
    spec = design.spec

    if arm == "iid":
        return IIDCovariance()
    if arm == "cluster_psu":
        return ClusteredCovariance(cluster_ids=psu, n_clusters=spec.n_psu)
    if arm == "cluster_psu_dof":
        return ClusteredCovariance(
            cluster_ids=psu, n_clusters=spec.n_psu, dof_correction=True
        )
    if arm in ("stratified", "stratified_fpc"):
        return StratifiedCovariance(
            psu_ids=psu,
            cell_ids=cell,
            stratum_ids=strat,
            n_psu=spec.n_psu,
            n_cells=spec.n_cells,
            n_strata=spec.n_strata,
            fpc=(arm == "stratified_fpc"),
        )
    if arm in ("design_aware", "design_aware_uncoupled"):
        design_block = StratifiedCovariance(
            psu_ids=psu,
            cell_ids=cell,
            stratum_ids=strat,
            n_psu=spec.n_psu,
            n_cells=spec.n_cells,
            n_strata=spec.n_strata,
            fpc=False,
        )
        sampling = ClusteredCovariance(
            cluster_ids=psu, n_clusters=spec.n_psu, dof_correction=True
        )
        # Design moments = those instrumented by the RANDOMIZED variable:
        # r*T. The constant / x / w moments are sampling-side.
        return DesignAwareCovariance.from_design_mask(
            design=design_block,
            sampling=sampling,
            design_moment_mask=jnp.array([0.0, 1.0, 0.0, 0.0]),
            couple=(arm == "design_aware"),
        )
    raise ValueError(f"unknown covariance arm {arm!r}")


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArmResult:
    name: str
    study: StudyResult
    seconds: float


def run_arm(
    design: Design,
    arm: str,
    *,
    n_reps: int,
    seed: int,
    theta_init: Beta = BETA_TRUE,
) -> ArmResult:
    """Run one covariance arm: build_estimator + monte_carlo_study (CRN).

    Calibration arms start at the TRUTH (documented choice): these
    studies measure inference at the optimum, and the anchor-once-then-
    freeze ridge policy anchors on ``V(theta_init)`` -- a far start
    anchors the ridge on a wildly mis-scaled V (the pilot measured the
    coupled design-aware arm anchoring indefinite, tau binding on every
    rep, and 57% non-convergence from a (0, 0) start). Optimization
    robustness from far starts is the misspec study's job, which passes
    its own ``theta_init`` explicitly.
    """
    key = jax.random.PRNGKey(seed)
    template = draw_measure(design, jax.random.fold_in(key, 0))
    run = build_estimator(
        psi,
        measure=template,
        covariance=covariance_arm(design, arm),
        parameters=theta_init,
        moment_names=MOMENT_NAMES,
    )
    t0 = time.time()
    study = monte_carlo_study(
        run,
        lambda rep_key: draw_measure(design, rep_key),
        n_reps=n_reps,
        key=key,
        theta_init=theta_init,
        theta0=BETA_TRUE,
    )
    return ArmResult(name=arm, study=study, seconds=time.time() - t0)


def run_arm_per_rep_anchor(
    design: Design,
    arm: str,
    *,
    n_reps: int,
    seed: int,
    theta_init: Beta = BETA_TRUE,
) -> ArmResult:
    """Like :func:`run_arm`, but each replicate anchors its OWN ridge.

    Calls the bare ``estimate()`` per replicate (the slow path) so the
    anchor-once-then-freeze policy applies per dataset, as it would for
    a real user -- ``build_estimator``'s factory reuse freezes rep 0's
    ``tau_anchor`` (and hence ``binding_ridge``) across the whole study,
    which degenerates the #130 tau-binding column to {0, 1} per arm and
    lets one unlucky/lucky template draw poison/mask an entire arm
    (emu-gmm #142). Used by the studies where the anchor regime is
    exactly what is being measured (design_aware, ridge_binding).

    The CRN contract matches :func:`emu_gmm.studies.replicate`:
    replicate ``r`` draws with ``fold_in(PRNGKey(seed), r)``.
    """
    import jax.tree_util as jtu

    key = jax.random.PRNGKey(seed)
    cov = covariance_arm(design, arm)
    t0 = time.time()
    recs = []
    for r_i in range(n_reps):
        measure = draw_measure(design, jax.random.fold_in(key, r_i))
        res = estimate(
            model=psi,
            measure=measure,
            covariance=cov,
            theta_init=theta_init,
            moment_names=MOMENT_NAMES,
        )
        recs.append(res.record())
    stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *recs)
    records = MCRecords(records=stacked, key=jnp.asarray(key), n_reps=n_reps)
    study = StudyResult(
        records=records,
        bias_sd=bias_sd(records, BETA_TRUE),
        coverage=coverage(records, BETA_TRUE, level=0.95),
        size_power=size_power(records, alpha=(0.01, 0.05, 0.10)),
        tau_binding=tau_binding(records),
        j_calibration=j_calibration(records),
    )
    return ArmResult(name=arm, study=study, seconds=time.time() - t0)


def org_row(label: str, r: ArmResult, alpha_idx: int = 1) -> str:
    """One org-table row: size/coverage/diagnostics for an arm."""
    s = r.study
    cov = s.coverage.coverage  # per-coordinate
    sp = s.size_power
    rej_nom = sp.reject_nominal[alpha_idx]  # at alpha[alpha_idx] (0.05)
    rej_adj = sp.reject_adjusted[alpha_idx]
    bias = s.bias_sd.bias
    se_ratio = s.bias_sd.se_ratio
    tb = s.tau_binding
    ks = s.j_calibration.max_abs_deviation
    return (
        f"| {label} | {s.n_used}/{s.n_reps} | "
        f"{rej_nom:.3f} | {rej_adj:.3f} | {ks:.3f} | "
        f"{cov[0]:.3f} / {cov[1]:.3f} | "
        f"{bias[0]:+.4f} / {bias[1]:+.4f} | "
        f"{se_ratio[0]:.3f} / {se_ratio[1]:.3f} | "
        f"{tb.binding_frequency:.3f} | {r.seconds:.0f}s |"
    )


ORG_HEADER = (
    "| arm | used | J rej@5% | Jadj rej@5% | J-KS | cover b0/b1 | bias b0/b1 "
    "| SE/SD b0/b1 | tau-bind | time |\n|-"
)


# ---------------------------------------------------------------------------
# Studies (the #130 checklist; each names its discrimination claim)
# ---------------------------------------------------------------------------


def study_size_iid_vs_cluster(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 1: J size + Wald coverage, IID vs ClusteredCovariance.

    Discrimination claim: under rho=0.5 with PSUs of 20, IIDCovariance
    must detectably over-reject / under-cover (design effect ~ 10),
    while cluster_psu is calibrated. The rho=0 control shows both
    calibrated (so the contrast is attributable to the clustering).
    """
    out: dict[str, ArmResult] = {}
    clustered = make_design(DesignSpec(rho=0.5))
    independent = make_design(DesignSpec(rho=0.0))
    for arm in ("iid", "cluster_psu", "cluster_psu_dof"):
        out[f"rho0.5/{arm}"] = run_arm(clustered, arm, n_reps=n_reps, seed=seed)
    for arm in ("iid", "cluster_psu"):
        out[f"rho0/{arm}"] = run_arm(independent, arm, n_reps=n_reps, seed=seed)
    return out


def study_stratified(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 2: StratifiedCovariance across PSU counts + fpc + H boundary.

    psu_per_cell=2 puts the w-moment pairs at the H_{c,jk} < 2 boundary
    in a substantial share of cells (p_w=0.5 PSU-level observability).
    """
    out: dict[str, ArmResult] = {}
    for g in (2, 5, 20):
        d = make_design(DesignSpec(psu_per_cell=g, sigma_strat=0.7))
        for arm in ("stratified", "stratified_fpc", "cluster_psu_dof"):
            out[f"G{g}/{arm}"] = run_arm(d, arm, n_reps=n_reps, seed=seed)
    return out


def study_design_aware(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 3: V_TS materiality -- couple=True vs False, in distribution.

    Per-rep anchoring (emu-gmm #142): the coupled assembly's
    conditioning varies per dataset; a factory-shared anchor lets one
    template draw distort the whole arm (the pilot's tau-bind = 1.000 /
    27% non-convergence artifact).
    """
    out: dict[str, ArmResult] = {}
    d = make_design(DesignSpec(psu_per_cell=5, sigma_strat=0.7))
    for arm in ("design_aware", "design_aware_uncoupled", "stratified"):
        out[f"{arm}"] = run_arm_per_rep_anchor(d, arm, n_reps=n_reps, seed=seed)
    return out


def study_misspec_power(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 4: J power + CU/LM robustness under misspecified alternatives.

    delta in {0, 0.1, 0.3}: the null arm pins size; the alternatives
    give the J power curve and the convergence-failure / step-count
    evidence the #9 reopen-decision needs. All arms deliberately start
    FAR from the truth (theta_init = (0, 0)) -- this study, unlike the
    calibration studies, measures the CU+LM *optimization* under both
    regimes, with the null arm as the CRN baseline.

    Step counts are collected by a thin wrapper around the same CRN
    loop (FitRecord deliberately does not carry iterations; see #125).
    """
    far_start = Beta(beta0=0.0, beta1=0.0)
    out: dict[str, ArmResult] = {}
    for delta in (0.0, 0.1, 0.3):
        d = make_design(DesignSpec(delta=delta))
        out[f"delta{delta}/cluster_psu"] = run_arm(
            d, "cluster_psu", n_reps=n_reps, seed=seed, theta_init=far_start
        )
    return out


def study_ridge_binding(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 5: regularised-J calibration where the ridge actually binds.

    Engineered binding-but-healthy regime: a small design (4 strata x 2
    arms x 2 PSUs = 16 PSUs) with sparse PSU-level observability of the
    w moment (p_w = 0.4) puts a large share of cells at the
    H_{c,jk} < 2 boundary, zeroing stratified-V entries and driving the
    signed-spectrum feasibility test into the binding regime -- while
    estimation stays at the truth-anchored healthy start. The
    J_pvalue vs J_pvalue_adjusted divergence here is the Davies/Imhof
    decision evidence (#130 item 4): if the adjusted p-value calibrates
    (J-KS small for the adjusted column where binding fires), the
    Welch-Satterthwaite approximation suffices.
    """
    out: dict[str, ArmResult] = {}
    # p_w = 0.7 (not lower): with 2-PSU cells the chance a cell supports
    # the (w, w) pair is p_w^2 = 0.49; across 12 cells the probability
    # that NO cell supports it -- a fully-zero V row that tau*diag(V)
    # cannot repair, killing the rep -- is ~2e-4. The binding regime
    # wanted here is NEAR-singular (H-boundary zero cells + noisy 1-dof
    # between-PSU estimates -> indefinite/ill-conditioned V), not
    # exactly-singular. Per-rep anchoring (#142) so binding_ridge is a
    # per-dataset event, which is the whole point of the column.
    d = make_design(
        DesignSpec(n_strata=6, psu_per_cell=2, p_x=0.7, p_w=0.7, sigma_strat=0.7)
    )
    for arm in ("stratified", "cluster_psu_dof"):
        out[f"binding/{arm}"] = run_arm_per_rep_anchor(d, arm, n_reps=n_reps, seed=seed)
    return out


def study_misspec_steps(n_reps: int, seed: int) -> dict[str, ArmResult]:
    """Study 4b: per-arm step-count / convergence table (CU+LM column).

    Reuses the CRN loop of :func:`replicate` inline so the per-rep
    ``EstimationResult.iterations`` can be collected alongside the
    records (FitRecord deliberately omits it). Printed as its own
    table by main().
    """
    far_start = Beta(beta0=0.0, beta1=0.0)
    out: dict[str, ArmResult] = {}
    print("\n| arm | conv | iters p50/p90/max |")
    print("|-")
    for delta in (0.0, 0.1, 0.3):
        d = make_design(DesignSpec(delta=delta))
        key = jax.random.PRNGKey(seed)
        template = draw_measure(d, jax.random.fold_in(key, 0))
        run = build_estimator(
            psi,
            measure=template,
            covariance=covariance_arm(d, "cluster_psu"),
            parameters=far_start,
            moment_names=MOMENT_NAMES,
        )
        iters, conv = [], []
        for r_i in range(n_reps):
            res = run(far_start, draw_measure(d, jax.random.fold_in(key, r_i)))
            iters.append(int(np.asarray(res.iterations)))
            conv.append(bool(res.converged))
        it = np.asarray(iters, dtype=float)
        print(
            f"| delta{delta} | {np.mean(conv):.3f} | "
            f"{np.percentile(it, 50):.0f}/{np.percentile(it, 90):.0f}"
            f"/{it.max():.0f} |"
        )
    return out


STUDIES = {
    "size_iid_vs_cluster": study_size_iid_vs_cluster,
    "stratified": study_stratified,
    "design_aware": study_design_aware,
    "misspec_power": study_misspec_power,
    "misspec_steps": study_misspec_steps,
    "ridge_binding": study_ridge_binding,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=sorted(STUDIES), required=True)
    ap.add_argument("--n-reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for per-replicate CSV + provenance JSON",
    )
    args = ap.parse_args()

    t0 = time.time()
    arms = STUDIES[args.study](args.n_reps, args.seed)

    print(f"\n* {args.study}  (n_reps={args.n_reps}, seed={args.seed})\n")
    print(ORG_HEADER)
    for label, r in arms.items():
        print(org_row(label, r))
    print(f"\ntotal: {time.time() - t0:.0f}s")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for label, r in arms.items():
            stem = f"{args.study}__{label.replace('/', '_')}"
            df = r.study.records.to_pandas()
            df.to_csv(args.out / f"{stem}.csv", index=False)
        (args.out / f"{args.study}.json").write_text(
            json.dumps(
                {
                    "study": args.study,
                    "n_reps": args.n_reps,
                    "seed": args.seed,
                    "arms": list(arms),
                    "command": (
                        f"python scripts/validation/ladder_mc.py "
                        f"--study {args.study} --n-reps {args.n_reps} "
                        f"--seed {args.seed}"
                    ),
                    "seconds": round(time.time() - t0, 1),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote records to {args.out}")


if __name__ == "__main__":
    main()
