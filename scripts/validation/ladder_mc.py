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
    crn_pair,
    event_share,
    given,
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

# Column layout of the observation matrix (one row per unit). When the
# binding study's near-collinear instrument is enabled
# (DesignSpec.collinear_eps), a fifth column z5 = x + eps * eta is
# appended and psi5 adds the fifth moment r * z5.
COL_Y, COL_T, COL_X, COL_W, COL_Z5 = 0, 1, 2, 3, 4
N_MOMENTS = 4  # r * (1, T, x, w)
MOMENT_NAMES = ("r_const", "r_T", "r_x", "r_w")
MOMENT_NAMES5 = MOMENT_NAMES + ("r_z5",)


def psi(x, theta):
    """Per-observation moment vector ``r * (1, T, x, w)``; (M,) = (4,)."""
    y, T, xv, w = x[COL_Y], x[COL_T], x[COL_X], x[COL_W]
    r = y - theta.beta0 - theta.beta1 * T
    return jnp.array([r, r * T, r * xv, r * w])


def psi5(x, theta):
    """psi plus the near-collinear fifth moment ``r * z5``; (M,) = (5,).

    ``z5 = x + eps * eta`` makes moments 3 and 5 nearly collinear with
    tunable eps, driving V toward the binding regime of
    :class:`DiagonalTikhonov` (small/negative lambda_min through noisy
    few-PSU between-cell estimates of a near-singular pair) without
    touching the identified parameters.
    """
    y, T, xv, w, z5 = x[COL_Y], x[COL_T], x[COL_X], x[COL_W], x[COL_Z5]
    r = y - theta.beta0 - theta.beta1 * T
    return jnp.array([r, r * T, r * xv, r * w, r * z5])


def model_for(spec: "DesignSpec"):
    """The (psi, moment_names) pair matching a spec's moment count."""
    if spec.collinear_eps is not None:
        return psi5, MOMENT_NAMES5
    return psi, MOMENT_NAMES


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
    #: When set, append the near-collinear instrument z5 = x + eps * eta
    #: (fifth moment r * z5, observability = the x moment's d_x). The
    #: binding-study knob; None = the 4-moment family.
    collinear_eps: float | None = None

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

    # PSU-level observability of the x / w moments (genuine, cluster-
    # correlated missingness; commitment 9/10 regime).
    dx_psu = (jax.random.uniform(k_dx, (n_psu,)) < spec.p_x).astype(jnp.float64)
    dw_psu = (jax.random.uniform(k_dw, (n_psu,)) < spec.p_w).astype(jnp.float64)
    ones = jnp.ones((n,))

    if spec.collinear_eps is not None:
        k_eta = jax.random.fold_in(k_x, 99)
        eta = jax.random.normal(k_eta, (n,))
        z5 = x + spec.collinear_eps * eta
        data = jnp.stack([y, T, x, w, z5], axis=1)  # (N, 5)
        mask = jnp.stack(
            [ones, ones, dx_psu[psu], dw_psu[psu], dx_psu[psu]], axis=1
        )  # (N, 5): z5 shares the x moment's observability
    else:
        data = jnp.stack([y, T, x, w], axis=1)  # (N, 4)
        mask = jnp.stack([ones, ones, dx_psu[psu], dw_psu[psu]], axis=1)

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
    if arm in ("design_aware", "design_aware_uncoupled", "design_aware_nodof"):
        design_block = StratifiedCovariance(
            psu_ids=psu,
            cell_ids=cell,
            stratum_ids=strat,
            n_psu=spec.n_psu,
            n_cells=spec.n_cells,
            n_strata=spec.n_strata,
            fpc=False,
        )
        # The #119 dof-inheritance ablation (review fix 3): the sampling
        # strategy's dof_correction is the knob the cross pass inherits;
        # design_aware_nodof switches it off.
        sampling = ClusteredCovariance(
            cluster_ids=psu,
            n_clusters=spec.n_psu,
            dof_correction=(arm != "design_aware_nodof"),
        )
        # Design moments = those instrumented by the RANDOMIZED variable:
        # r*T. The constant / x / w (and z5) moments are sampling-side.
        n_mom = 5 if spec.collinear_eps is not None else 4
        dmask = jnp.zeros(n_mom).at[1].set(1.0)
        return DesignAwareCovariance.from_design_mask(
            design=design_block,
            sampling=sampling,
            design_moment_mask=dmask,
            couple=(arm != "design_aware_uncoupled"),
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
    spec: DesignSpec
    theta_init: Beta
    anchor: str  # 'factory' (build_estimator reuse) | 'per-rep' (#142)

    def validity(self) -> dict[str, int]:
        """The #140 denominators behind every printed rate (fix 5)."""
        s = self.study
        return {
            "n_used": int(s.n_used),
            "n_excluded": int(s.n_excluded),
            "n_valid_se_min": (
                int(np.min(s.coverage.n_valid_se))
                if np.size(s.coverage.n_valid_se)
                else 0
            ),
            "n_valid_J_nominal": int(s.size_power.n_valid_nominal),
            "n_valid_J_adjusted": int(s.size_power.n_valid_adjusted),
        }

    def has_hidden_invalidity(self) -> bool:
        v = self.validity()
        return v["n_valid_se_min"] < v["n_used"] or v["n_valid_J_nominal"] < v["n_used"]


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
    model, moment_names = model_for(design.spec)
    template = draw_measure(design, jax.random.fold_in(key, 0))
    run = build_estimator(
        model,
        measure=template,
        covariance=covariance_arm(design, arm),
        parameters=theta_init,
        moment_names=moment_names,
    )
    t0 = time.time()
    study = monte_carlo_study(
        run,
        lambda rep_key: draw_measure(design, rep_key),
        n_reps=n_reps,
        key=key,
        theta_init=theta_init,
        theta0=BETA_TRUE,
        # CRN-coupling token: arms drawn from the same (design, seed) share
        # the per-rep data stream (draw_measure depends only on design + key);
        # the covariance arm is applied AFTER the draw. crn_pair verifies this.
        coupling_id=(repr(design.spec), seed),
    )
    return ArmResult(
        name=arm,
        study=study,
        seconds=time.time() - t0,
        spec=design.spec,
        theta_init=theta_init,
        anchor="factory",
    )


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
    model, moment_names = model_for(design.spec)
    t0 = time.time()
    recs = []
    for r_i in range(n_reps):
        measure = draw_measure(design, jax.random.fold_in(key, r_i))
        res = estimate(
            model=model,
            measure=measure,
            covariance=cov,
            theta_init=theta_init,
            moment_names=moment_names,
        )
        recs.append(res.record())
        # Bare estimate() builds fresh closures per call, so JAX's global
        # caches accumulate write-only traces (~14 MB/call measured; the
        # un-mitigated leak OOM-killed this study's first full run at
        # 9.4 GB -- #139 merge-verification thread). But clear_caches()
        # ALSO drops XLA's low-level kernel compilations, which ARE
        # re-hit across reps: clearing every rep tripled wall-clock
        # (~7.4 s/rep vs ~2.6; measured on the second full run's
        # design_aware study). Clear every 25 reps: memory swing
        # ~25 x 14 MB = 350 MB, recompilation amortized to ~6%.
        if (r_i + 1) % 25 == 0:
            jax.clear_caches()
    stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *recs)
    records = MCRecords(
        records=stacked,
        key=jnp.asarray(key),
        n_reps=n_reps,
        coupling_id=(repr(design.spec), seed),
    )
    study = StudyResult(
        records=records,
        bias_sd=bias_sd(records, BETA_TRUE),
        coverage=coverage(records, BETA_TRUE, level=0.95),
        size_power=size_power(records, alpha=(0.01, 0.05, 0.10)),
        tau_binding=tau_binding(records),
        j_calibration=j_calibration(records),
    )
    return ArmResult(
        name=arm,
        study=study,
        seconds=time.time() - t0,
        spec=design.spec,
        theta_init=theta_init,
        anchor="per-rep",
    )


def org_row(label: str, r: ArmResult, alpha_idx: int = 1) -> str:
    """One org-table row: size/coverage/diagnostics for an arm.

    The ``valid`` cell carries the #140 denominators behind the rates
    (fix 5 of the adversarial review): ``<min n_valid_se>se/<n_valid_J>J``
    with a ``!`` marker whenever any denominator is silently below
    ``n_used`` (e.g. converged reps that emitted NaN SEs under the #138
    indefinite-meat policy). Note the SE/SD column mixes denominators
    by construction: mean SE over finite-SE reps vs MC SD over all used
    reps.
    """
    s = r.study
    cov = s.coverage.coverage  # per-coordinate
    sp = s.size_power
    rej_nom = sp.reject_nominal[alpha_idx]  # at alpha[alpha_idx] (0.05)
    rej_adj = sp.reject_adjusted[alpha_idx]
    bias = s.bias_sd.bias
    se_ratio = s.bias_sd.se_ratio
    tb = s.tau_binding
    ks = s.j_calibration.max_abs_deviation
    v = r.validity()
    flag = " !" if r.has_hidden_invalidity() else ""
    valid_cell = f"{v['n_valid_se_min']}se/{v['n_valid_J_nominal']}J{flag}"
    return (
        f"| {label} | {s.n_used}/{s.n_reps} | {valid_cell} | "
        f"{rej_nom:.3f} | {rej_adj:.3f} | {ks:.3f} | "
        f"{cov[0]:.3f} / {cov[1]:.3f} | "
        f"{bias[0]:+.4f} / {bias[1]:+.4f} | "
        f"{se_ratio[0]:.3f} / {se_ratio[1]:.3f} | "
        f"{tb.binding_frequency:.3f} | {r.seconds:.0f}s |"
    )


ORG_HEADER = (
    "| arm | used | valid | J rej@5% | Jadj rej@5% | J-KS | cover b0/b1 "
    "| bias b0/b1 | SE/SD b0/b1 | tau-bind | time |\n|-"
)


# ---------------------------------------------------------------------------
# Studies (the #130 checklist; each names its discrimination claim)
# ---------------------------------------------------------------------------


# --- CRN-paired and host-side readouts (adversarial-review fixes 1-4) ---


def _b1_cover_indicator(r: ArmResult) -> np.ndarray:
    """Per-rep b1 Wald-coverage indicator (NaN where invalid)."""
    rec = r.study.records.records
    theta = np.asarray(rec.theta_flat)[:, 1]
    se = np.asarray(rec.se)[:, 1]
    used = np.asarray(rec.converged) > 0
    ind = np.abs(theta - BETA_TRUE.beta1) <= 1.959963984540054 * se
    out = np.where(used & np.isfinite(se), ind.astype(float), np.nan)
    return out


def paired_dof_readout(a: ArmResult, b: ArmResult, label: str) -> None:
    """CRN-paired contrast for the dof_correction pair (review fix 2).

    Marginal rates cannot resolve the G/(G-1) correction at feasible
    n_reps; under CRN the per-rep differences can: the dof arm's wider
    CIs flip coverage ONE-directionally, so the flip counts are a sign
    test. Also reports the mean paired J difference.

    Routed through :func:`emu_gmm.studies.crn_pair` (#167), which verifies
    the two arms share a CRN stream (matching ``coupling_id``) before
    zipping -- a guard against silently pairing un-coupled draws.
    """
    cp = crn_pair(a.study.records, b.study.records)  # refuses if not CRN-coupled
    ca, cb = _b1_cover_indicator(a), _b1_cover_indicator(b)
    both = cp.both_finite(ca, cb)
    fl = cp.flips(ca, cb, where=both)  # gain = dof covers & base not
    Ja = np.asarray(a.study.records.records.J_stat)
    Jb = np.asarray(b.study.records.records.J_stat)
    mean_dj = cp.mean_paired_diff(Ja, Jb, where=both)
    print(
        f"|paired {label} | n={fl.n_both} | b1-cover flips "
        f"+{fl.gain}/-{fl.lose} (dof gains/loses) | mean dJ {mean_dj:+.4f} |"
    )


def jadj_readout(r: ArmResult, label: str) -> None:
    """ECDF deciles of J_pvalue vs J_pvalue_adjusted, by binding (fix 1iii).

    The Davies/Imhof evidence: among binding reps, the adjusted column
    should be ~U(0,1) (small KS) where the nominal one is not. Refuse
    to read the contrast off an arm whose binding frequency is ~0.
    """
    mc = r.study.records
    rec = mc.records
    used = np.asarray(rec.converged) > 0
    deciles = np.arange(1, 10) / 10.0

    def _ks(p: np.ndarray) -> float:
        p = p[np.isfinite(p)]
        if p.size < 5:
            return float("nan")
        ecdf = np.array([(p <= d).mean() for d in deciles])
        return float(np.max(np.abs(ecdf - deciles)))

    # The binding subpopulation via given() (#167); event_share() supplies the
    # loud counts (selected & converged vs total converged). The within-subset
    # nominal-vs-adjusted p-value contrast is the *blessed* conditional query:
    # a within-selection calibration contrast, NOT a coverage claim -- so it
    # acknowledges the selection-conditional gate (#167 Section 6 Q1).
    cond = given(mc, "binding_ridge", acknowledge_conditional=True)
    share = event_share(mc, "binding_ridge")
    bconv = np.asarray(cond.converged) > 0.5  # binding AND converged
    p_nom = np.asarray(rec.J_pvalue)
    p_adj = np.asarray(rec.J_pvalue_adjusted)
    pnb = np.asarray(cond.J_pvalue)[bconv]
    pab = np.asarray(cond.J_pvalue_adjusted)[bconv]
    nb = share.n_selected_converged
    print(
        f"|jadj {label} | binding {nb}/{share.n_total_converged} | "
        f"KS nom/adj (all): {_ks(p_nom[used]):.3f}/{_ks(p_adj[used]):.3f} | "
        f"KS nom/adj (binding): {_ks(pnb):.3f}/{_ks(pab):.3f} |"
        + (
            f"  [only {nb} binding events (<25): contrast NOT readable]"
            if nb < 25
            else ""
        )
    )


def h_boundary_readout(design: Design, n_reps: int, seed: int, label: str) -> None:
    """Mask-only CRN pass: realised H_{c,jk} < 2 share + zero-row events.

    Review fix 4: masks are not persisted, so the realised boundary
    share must be computed during the run, with the SAME fold_in stream
    as the estimation pass. Reports the share of (cell, pair) entries
    with H < 2 for the (w, w) pair and the count of reps whose V_ww row
    is all-zero (every cell below the boundary -- the unrepairable
    event, named and counted per #140 rather than folded into rates).
    """
    spec = design.spec
    key = jax.random.PRNGKey(seed)
    psu_of_cell = np.asarray(
        [np.unique(design.psu_ids[design.cell_ids == c]) for c in range(spec.n_cells)]
    )
    shares, zero_rows = [], 0
    for r_i in range(n_reps):
        rep_key = jax.random.fold_in(key, r_i)
        # Reproduce ONLY the k_dw stream of draw_measure (same split).
        keys = jax.random.split(rep_key, 8)
        k_dw = keys[7]
        dw_psu = np.asarray(jax.random.uniform(k_dw, (spec.n_psu,)) < spec.p_w).astype(
            float
        )
        h_ww = np.array([dw_psu[p].sum() for p in psu_of_cell])
        shares.append(float((h_ww < 2).mean()))
        zero_rows += int((h_ww < 2).all())
    print(
        f"|H-boundary {label} | mean share H_ww<2: {np.mean(shares):.3f} | "
        f"all-zero-V_ww-row reps: {zero_rows}/{n_reps} |"
    )


# --- Studies ---


def study_size_iid_vs_cluster(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 1: J size + Wald coverage, IID vs ClusteredCovariance.

    Discrimination claims: (a) under rho=0.5 with 20-unit PSUs,
    IIDCovariance must detectably over-reject / under-cover (design
    effect ~ 10) while cluster_psu is calibrated; rho=0 control shows
    both calibrated. (b) FEW-CLUSTER regime (review fix 2): at G = 12
    PSUs (3 strata x 2 arms x 2 PSUs of 50), the dof_correction
    contrast is discriminable -- G_jk/(G_jk-1) = 12/11 in variance
    (less for the masked w pairs), read via the CRN-paired flip counts
    (one-directional under CRN; a handful of one-sided flips at
    n_reps=500 is decisive where marginal rates are not). The G=100
    rho0.5 trio is the SCALE-REGIME row: its dof pair differs by a
    deterministic factor 100/99 (SE factor 1.005) and is NOT dof
    evidence.
    """
    out_arms: dict[str, ArmResult] = {}
    clustered = make_design(DesignSpec(rho=0.5))
    independent = make_design(DesignSpec(rho=0.0))
    few = make_design(DesignSpec(n_strata=3, psu_per_cell=2, psu_size=50))
    for arm in ("iid", "cluster_psu", "cluster_psu_dof"):
        out_arms[f"rho0.5/{arm}"] = run_arm(clustered, arm, n_reps=n_reps, seed=seed)
    for arm in ("iid", "cluster_psu"):
        out_arms[f"rho0/{arm}"] = run_arm(independent, arm, n_reps=n_reps, seed=seed)
    for arm in ("iid", "cluster_psu", "cluster_psu_dof"):
        out_arms[f"fewG12/{arm}"] = run_arm(few, arm, n_reps=n_reps, seed=seed)
    paired_dof_readout(
        out_arms["fewG12/cluster_psu"], out_arms["fewG12/cluster_psu_dof"], "fewG12"
    )
    paired_dof_readout(
        out_arms["rho0.5/cluster_psu"], out_arms["rho0.5/cluster_psu_dof"], "G100"
    )
    return out_arms


def study_stratified(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 2: StratifiedCovariance across PSU counts + fpc + H boundary.

    Claims (pre-registered per review fixes 4 and 7):

    - Calibration: stratified (fpc=False) calibrated at G in {5, 20};
      cluster_psu_dof conservative (uncentered pooling of the fixed
      demeaned stratum means).
    - fpc arm (GUARANTEED-FAIL, pre-registered reproduction targets,
      review fix 7): all PSUs assigned -> _fpc_factor = 1 - g/2g = 0.5
      in every cell, so stratified_fpc is exactly 0.5 * V_stratified
      under this superpopulation DGP. Predictions the full run must
      REPRODUCE (else the row indicts _fpc_factor): CRN-paired SE ratio
      0.7071 vs the fpc=False arm; b1 coverage ~ 0.834
      (= P(|Z| < 1.96 * 0.7071)); J rej@5% ~ 0.224 (= P(chi2_2 >
      5.991 * 0.5)). The arm is an estimand-mismatch check, NOT a
      calibration rung.
    - H boundary (review fix 4): at G=2 / p_w=0.5, P(H_{c,ww} < 2) =
      1 - p_w^2 = 0.75 per cell; StratifiedCovariance zeroes those
      cells while their units still count in N_w, so E[V_ww] ~ 0.5 *
      truth and the row MUST over-reject J -- the predicted ~2x V_ww
      truncation is the reproduction target, and the realised boundary
      share is printed by the mask-only CRN readout. The G2hi
      (p_w=0.85) arm is the minority-boundary rung where calibration
      claims stay meaningful. All-zero-V_ww-row reps (~0.75^20 ~
      0.003/rep at G=2) are counted as a named event.
    """
    out_arms: dict[str, ArmResult] = {}
    for g in (2, 5, 20):
        d = make_design(DesignSpec(psu_per_cell=g, sigma_strat=0.7))
        for arm in ("stratified", "stratified_fpc", "cluster_psu_dof"):
            out_arms[f"G{g}/{arm}"] = run_arm(d, arm, n_reps=n_reps, seed=seed)
        h_boundary_readout(d, n_reps, seed, f"G{g}/p_w0.5")
    d_hi = make_design(DesignSpec(psu_per_cell=2, p_w=0.85, sigma_strat=0.7))
    for arm in ("stratified", "cluster_psu_dof"):
        out_arms[f"G2hi/{arm}"] = run_arm(d_hi, arm, n_reps=n_reps, seed=seed)
    h_boundary_readout(d_hi, n_reps, seed, "G2/p_w0.85")
    return out_arms


def study_design_aware(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 3: DesignAware -- coupling and dof inheritance (rescoped).

    Rescoped claims (review fix 3): under the FIXED demeaned v_strat
    profile all per-rep randomness is independent across PSUs, so the
    true cross-ARM component of V_TS is exactly zero here; the
    couple=True/False contrast exercises the MECHANICAL same-unit
    coupling only (m_T is a sub-sum of m_const; Var(b1) inflates ~3x
    under couple=False). A genuine cross-arm-shock fixture is recorded
    follow-up work, not claimed by this study.

    Arms: the psu_per_cell=5 trio (per-rep anchored, #142); a
    sigma_strat=0 contamination check (fix 3iv: if coupled-arm b1
    coverage normalizes without the stratum profile, the pilot's b1
    under-coverage is profile contamination of the uncentered
    V_SS/V_TS blocks glued to centered V_TT); and the #119
    dof-inheritance ablation in a FEW-cluster sampling regime (G=12,
    where G_jk/(G_jk-1) is discriminable) -- design_aware (sampling
    dof_correction=True, inherited by the cross pass) vs
    design_aware_nodof (False).
    """
    out_arms: dict[str, ArmResult] = {}
    d = make_design(DesignSpec(psu_per_cell=5, sigma_strat=0.7))
    for arm in ("design_aware", "design_aware_uncoupled", "stratified"):
        out_arms[f"{arm}"] = run_arm_per_rep_anchor(d, arm, n_reps=n_reps, seed=seed)
    d0 = make_design(DesignSpec(psu_per_cell=5, sigma_strat=0.0))
    out_arms["nostrat/design_aware"] = run_arm_per_rep_anchor(
        d0, "design_aware", n_reps=n_reps, seed=seed
    )
    d_few = make_design(DesignSpec(n_strata=3, psu_per_cell=2, psu_size=50))
    for arm in ("design_aware", "design_aware_nodof"):
        out_arms[f"fewG12/{arm}"] = run_arm_per_rep_anchor(
            d_few, arm, n_reps=n_reps, seed=seed
        )
    paired_dof_readout(
        out_arms["fewG12/design_aware_nodof"],
        out_arms["fewG12/design_aware"],
        "fewG12-#119",
    )
    jadj_readout(out_arms["design_aware"], "design_aware")
    return out_arms


def study_misspec_power(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 4: J power + CU/LM robustness -- the FLAT-PATH canary.

    delta in {0, 0.1, 0.3}: the null arm pins size; the alternatives
    give the J power curve plus convergence-failure evidence. All arms
    deliberately start FAR from the truth (theta_init = (0, 0)) with
    the null arm as the CRN baseline.

    Scope (review fix 8): this is the owner-approved CU+LM robustness
    column on the FLAT (Euclidean) path -- CU's V(theta) channel makes
    even this affine-psi criterion a genuine ratio-of-quadratics with
    far-start failure modes, so the step-count table is a valid graded
    discriminator. It is NOT by itself evidence to reopen or close #9:
    #9 was closed on direct PSDFixedRank-quotient verification, which
    this fixture cannot probe. Quotient-geometry evidence would need a
    manifold fixture in a separate study.
    """
    far_start = Beta(beta0=0.0, beta1=0.0)
    out_arms: dict[str, ArmResult] = {}
    for delta in (0.0, 0.1, 0.3):
        d = make_design(DesignSpec(delta=delta))
        out_arms[f"delta{delta}/cluster_psu"] = run_arm(
            d, "cluster_psu", n_reps=n_reps, seed=seed, theta_init=far_start
        )
    return out_arms


def study_ridge_binding(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 5: regularised-J calibration where the ridge ACTUALLY binds.

    The review's blocker fix 1: the previous spec (this docstring now
    matches the code -- fix 1iv) never tripped the signed-spectrum
    feasibility test (measured anchors: kappa ~ 1e2 vs kappa_target
    1e10), so J_pvalue_adjusted == J_pvalue identically and checklist
    item 4 was empty. Binding is now ENGINEERED via the near-collinear
    fifth moment r * z5, z5 = x + eps * eta (DesignSpec.collinear_eps):
    the (x, z5) pair drives lambda_min(V) toward 0 at rate eps^2, and
    the noisy few-PSU between-cell estimates push it negative --
    binding through the indefiniteness channel of the #111 signed test.
    PILOT REVISION (30-rep evidence): eps-collinearity alone drives V
    ill-conditioned but PD -- kappa never crosses the 1e10 feasibility
    target and binding stays at exactly 0 -- while the DESIGN-AWARE
    glue (indefinite V_TT + V_TS assembly, the documented
    not-PSD-by-construction case) binds on 28-36% of datasets in the
    few-cluster designs. The binding arms are therefore design-aware;
    the stratified+eps arm is RETAINED as the feasibility-blind-spot
    row: it over-rejects J (pilot 0.276) with binding == 0 -- the
    signed-spectrum test passing a degraded-but-PD V is direct #134
    evidence. The item-4 reading is GATED on the measured binding
    frequency (fix 1ii -- jadj_readout refuses arms with ~0 binding
    events) and reported unconditionally AND conditional on binding
    (fix 1iii). Per-rep anchoring throughout (#142).
    """
    out_arms: dict[str, ArmResult] = {}
    few = DesignSpec(n_strata=3, psu_per_cell=2, psu_size=50)
    arms: list[tuple[str, DesignSpec, str]] = [
        # (label, spec, covariance arm)
        ("da_fewG12", few, "design_aware"),
        (
            "da_fewG12_eps1e-2",
            dataclasses.replace(few, collinear_eps=1e-2),
            "design_aware",
        ),
        (
            "da_G5",
            DesignSpec(psu_per_cell=5, sigma_strat=0.7),
            "design_aware",
        ),
        (
            "strat_eps3e-2",  # feasibility-blind-spot row (binding ~ 0)
            DesignSpec(
                n_strata=6,
                psu_per_cell=2,
                p_x=0.7,
                p_w=0.7,
                sigma_strat=0.7,
                collinear_eps=3e-2,
            ),
            "stratified",
        ),
        (
            "noeps_strat",  # no-binding baseline
            DesignSpec(n_strata=6, psu_per_cell=2, p_x=0.7, p_w=0.7, sigma_strat=0.7),
            "stratified",
        ),
    ]
    for label, spec, cov_arm in arms:
        d = make_design(spec)
        out_arms[label] = run_arm_per_rep_anchor(d, cov_arm, n_reps=n_reps, seed=seed)
        jadj_readout(out_arms[label], label)
    return out_arms


def study_misspec_steps(
    n_reps: int, seed: int, out: Path | None = None
) -> dict[str, ArmResult]:
    """Study 4b: per-arm step-count / convergence table (CU+LM column).

    Reuses the CRN loop of :func:`replicate` inline so the per-rep
    ``EstimationResult.iterations`` can be collected alongside the
    records (FitRecord deliberately omits it; see #125). Persists the
    per-rep iterations/convergence to CSV when --out is given (review
    fix 6iv -- the #9-column evidence must not live only in stdout).
    """
    far_start = Beta(beta0=0.0, beta1=0.0)
    rows: list[dict[str, Any]] = []
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
            rows.append(
                {
                    "delta": delta,
                    "rep": r_i,
                    "iterations": iters[-1],
                    "converged": conv[-1],
                }
            )
        it = np.asarray(iters, dtype=float)
        print(
            f"| delta{delta} | {np.mean(conv):.3f} | "
            f"{np.percentile(it, 50):.0f}/{np.percentile(it, 90):.0f}"
            f"/{it.max():.0f} |"
        )
    if out is not None:
        import pandas as pd

        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "misspec_steps__iterations.csv", index=False)
    return {}


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


def _environment_provenance() -> dict[str, Any]:
    """Pin what a seed alone does not (review fix 6ii).

    jax PRNG output is version- and impl-sensitive, so the committed
    records are reproducible only against these pins.
    """
    import subprocess

    import emu_gmm

    repo = Path(__file__).resolve().parents[2]

    def _git(*a: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *a],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance best-effort
            return "unknown"

    try:
        prng_impl = str(jax.config.jax_default_prng_impl)
    except Exception:  # pragma: no cover
        prng_impl = "unknown"
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": "yes" if _git("status", "--porcelain", "-uno") else "no",
        "emu_gmm_version": getattr(emu_gmm, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": prng_impl,
        "beta_true": (
            dataclasses.asdict(BETA_TRUE)
            if dataclasses.is_dataclass(BETA_TRUE)
            else {"beta0": BETA_TRUE.beta0, "beta1": BETA_TRUE.beta1}
        ),
        "stratum_profile_seed": 777,
    }


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
    arms = STUDIES[args.study](args.n_reps, args.seed, out=args.out)

    org_lines = [f"\n* {args.study}  (n_reps={args.n_reps}, seed={args.seed})\n"]
    org_lines.append(ORG_HEADER)
    for label, r in arms.items():
        org_lines.append(org_row(label, r))
        if r.has_hidden_invalidity():
            v = r.validity()
            print(
                f"WARNING: arm {label} has hidden invalidity -- "
                f"n_valid {v} < n_used {v['n_used']}; rates above use "
                f"shrunken denominators (#140)."
            )
    fragment = "\n".join(org_lines)
    print(fragment)
    print(f"\ntotal: {time.time() - t0:.0f}s")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for label, r in arms.items():
            stem = f"{args.study}__{label.replace('/', '_')}"
            df = r.study.records.to_pandas()
            df.to_csv(args.out / f"{stem}.csv", index=False)
        # The org fragment is part of the committed artifact set
        # (review fix 6iii), not just stdout.
        (args.out / f"{args.study}.org").write_text(fragment + "\n")
        (args.out / f"{args.study}.json").write_text(
            json.dumps(
                {
                    "study": args.study,
                    "n_reps": args.n_reps,
                    "seed": args.seed,
                    "environment": _environment_provenance(),
                    "arms": {
                        label: {
                            "spec": dataclasses.asdict(r.spec),
                            "theta_init": {
                                "beta0": r.theta_init.beta0,
                                "beta1": r.theta_init.beta1,
                            },
                            "anchor": r.anchor,
                            "validity": r.validity(),
                            "seconds": round(r.seconds, 1),
                        }
                        for label, r in arms.items()
                    },
                    "command": (
                        f"python scripts/validation/ladder_mc.py "
                        f"--study {args.study} --n-reps {args.n_reps} "
                        f"--seed {args.seed} --out {args.out}"
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
