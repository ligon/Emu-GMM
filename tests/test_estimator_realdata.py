"""Real-data acceptance test on the empirical path (issue #128).

The fourth acceptance test: where ``test_estimator{,_analytical,_empirical}``
drive the pipeline against synthetic draws, this one drives it against a
frozen extract of *real survey data* — the Seasonality project's
Hansen–Singleton-style Euler test on Malawian household panel data.

Data
----
``tests/data/seasonality_euler_extract.npz`` — a derived extract of
``Seasonality/src/euler_data.py::euler_observation(stock_indicator=2)``
(one row per ``(i, t)`` cell with at least one crop held; transformed
m / delta / R / z columns, no household identifiers). Provenance —
source commit, generation command, content hash, per-column checksums —
in the JSON sidecar next to it; regeneration via
``scripts/freeze_seasonality_extract.py`` (requires the private
Seasonality data tree). Owner approved in-repo storage 2026-06-10.

Why this data exercises the empirical path's distinguishing features:

- *Genuine per-coordinate missingness*: moment ``(crop j, instrument k)``
  is observable for a row iff the household held crop ``j`` at ``t-1``,
  so the per-coordinate counts ``N_j`` range from 258 (groundnut) to
  2934 (maize) — the unequal-``N_j`` regime of CLAUDE.md commitment 9
  on real data rather than a constructed mask.
- *Design cluster structure*: 30 strata-of-six (Athey–Imbens fine
  stratification) x 3 arms = 90 ``(stratum x arm)`` cells over 179
  randomization groups (PSUs), driving both ``ClusteredCovariance``
  and the three-level ``StratifiedCovariance`` design sandwich.
- *A weakly identified scalar on a manifold*: ``sigma > 0`` via the
  ``Positive`` manifold (RiemannianLM auto-dispatch). Under the
  coarser covariance levels the CUE optimum is *interior*; under the
  design covariance the real-data optimum collapses to the boundary
  ``sigma -> 0+`` (the criterion's limit there is the log-residual
  specification) — both regimes are pinned below.

The moment function replicates the consumer's
``Seasonality/src/euler_gmm.py::make_moment_function`` (the row-id
``mbar`` closure) so the pins double as a consumer-contract check:

- ``stratum_arm``: sigma_hat 1.2252, SE 0.20 — the consumer's recorded
  PREFERRED estimate (``euler_test.py::SIGMA_PREFERRED``); J matches
  ``data/euler_real_cluster_compare.csv`` (J=6.574, p=0.8325).
- ``design``: sigma_hat 0.0 (boundary) — matches the same CSV.

Pins recorded 2026-06-10 on x86-64 CPU float64 (Seasonality commit
c5a3f4a63a4c42404e2f838533cbc03057959c6e). Tolerances are the contract:
do not loosen them to make a regression pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ClusteredCovariance,
    ContinuouslyUpdated,
    DiagonalTikhonov,
    EmpiricalMeasure,
    StratifiedCovariance,
    estimate,
)
from emu_gmm.manifolds import Positive, riemannian_lm
from emu_gmm.types import EstimationResult
from jax.ops import segment_sum

# Real-data acceptance test: marked slow so the every-push quick gate
# (`pytest -m "not slow"`) skips it; it still runs in the full-suite matrix
# and nightly. (#154)
pytestmark = pytest.mark.slow

DATA_DIR = Path(__file__).parent / "data"
NPZ_PATH = DATA_DIR / "seasonality_euler_extract.npz"
PROVENANCE_PATH = DATA_DIR / "seasonality_euler_extract.json"

#: The published Treatment spec: randomized transfer-arm dummies as
#: instruments, crossed with the 6 crop residuals -> M = 12 moments.
INSTRUMENTS = ("cash", "kind")

# ── Pinned values (the acceptance contract) ─────────────────────────
# Published preferred spec: CUE, Positive(sigma), ClusteredCovariance
# at the (stratum x arm) cell level (G = 90).
SIGMA_HAT_PUBLISHED = 1.225197217084243
SIGMA_SE_PUBLISHED = 0.20059258207328795
J_STAT_PUBLISHED = 6.573933050290934
J_PVALUE_PUBLISHED = 0.8324639536151227

# Design-based spec: StratifiedCovariance over PSU-in-cell-in-stratum.
# The real-data CUE optimum collapses to the boundary sigma -> 0+; the
# J pin is the criterion's stable log-residual limit there.
#
# Re-blessed for #151. The previous pins (J=5.437512, p_adjusted=0.908143)
# were an ORPHANED transcription -- no current code path reproduces them.
# emu-gmm's StratifiedCovariance and the consumer's current 'design' spec
# (which is built on that same emu-gmm covariance and reads J off the
# EstimationResult) both yield J=7.0600 on this extract, independently
# re-derived from first principles as m_bar' V_X^{-1} m_bar = 7.058. The
# 5.44 was a stale pre-migration value hidden by the suite's accumulation
# crash. See issue #151 for the full investigation.
# Boundary-limit design J / adjusted p-value, evaluated at a FIXED in-zone
# sigma (#159). Under the design (CUE) spec the optimum collapses to
# sigma -> 0+, where (a) the criterion is flat in sigma -- so the optimiser's
# exact stopping sigma is float-rounding noise -- and (b) the consumer's
# ``m**sigma`` moment loses precision below ~1e-11 via catastrophic
# cancellation (ligon/Emu-GMM#159, fixed upstream in TaimakaSeasonality#27).
# Pinning J off the optimiser's stopping point is therefore platform-fragile
# (J=5.44 on one OpenBLAS box vs ~7.08 on CI). We instead pin the boundary
# limit at ``SIGMA_BOUNDARY_EVAL`` -- firmly above the cancellation cliff,
# where J is deterministic and platform-stable (it varies < 1e-5 across
# sigma in [1e-7, 1e-5]). These values are J(sigma=1e-6); tolerances are
# tightened accordingly (rel 1e-2 -> 1e-4).
SIGMA_BOUNDARY_EVAL = 1e-6
J_STAT_DESIGN = 7.0867474141
J_PVALUE_ADJ_DESIGN = 0.7920156209

# Data-integrity pins (exact properties of the frozen extract).
N_ROWS = 3422
N_T = 8
CROPS = ("beans", "gcorn", "groundnut", "maize", "millet", "rice")
N_J = {
    "beans": 2127,
    "gcorn": 1571,
    "groundnut": 258,
    "maize": 2934,
    "millet": 2193,
    "rice": 711,
}
N_STRATA = 30
N_CELLS = 90
N_PSU_IN_SAMPLE = 179  # of 189 coded PSUs, those reaching the moment matrix


@jdc.pytree_dataclass
class EulerSigma:
    """Scalar elasticity ``sigma > 0`` on the Positive manifold.

    Mirrors the consumer's ``EulerParamsPositive``: the annotation routes
    ``estimate(..., optimizer=None)`` to the RiemannianLM auto-dispatch,
    whose exponential retraction keeps ``sigma`` strictly positive even
    when the optimum sits at the boundary ``sigma -> 0+``.
    """

    sigma: jnp.ndarray

    __emu_manifolds__ = {"sigma": Positive()}


# ── Consumer-contract moment function ────────────────────────────────


def make_psi(obs_aug, cols, crops, instruments, n_t):
    """Per-observation Euler moment, replicating the consumer's closure.

    ``psi_(j,k)(x_i, sigma) = (m_i^sigma/|sigma| - mbar_{t(i),j}) R_{t(i),j}
    delta_{i,j} z_{i,k}`` with the within-``(t, j)`` mean ``mbar`` taken
    over observers of crop ``j`` (a sample-level reduction, evaluated by
    a ``segment_sum`` closure over the full sample and re-broadcast via a
    trailing row-id column; see emu-gmm issue #113 for the first-class
    feature this works around, and Seasonality's
    ``euler_gmm.make_moment_function`` for the original).
    """
    m_idx = cols["m"]
    t_idx = cols["t_code"]
    delta_idxs = tuple(cols[f"delta_{c}"] for c in crops)
    R_idxs = tuple(cols[f"R_{c}"] for c in crops)
    z_idxs = tuple(cols[c] for c in instruments)

    X_full = jnp.asarray(obs_aug)
    deltas_full = jnp.stack([X_full[:, j] for j in delta_idxs], axis=1)
    t_code_full = X_full[:, t_idx].astype(jnp.int32)
    m_full = X_full[:, m_idx]
    row_id_idx = X_full.shape[1] - 1

    def _mbar_at_rows(sigma):
        m_sigma = (m_full**sigma) / jnp.abs(sigma)
        m_sigma_safe = jnp.where(jnp.isnan(m_sigma), 0.0, m_sigma)
        num = segment_sum(
            m_sigma_safe[:, None] * deltas_full, t_code_full, num_segments=n_t
        )
        den = segment_sum(deltas_full, t_code_full, num_segments=n_t)
        mbar = num / jnp.where(den > 0, den, 1.0)
        return mbar[t_code_full]

    def psi(x, theta):
        sigma = theta.sigma
        m = x[m_idx]
        deltas = jnp.array([x[j] for j in delta_idxs])
        Rs = jnp.array([x[j] for j in R_idxs])
        zs = jnp.array([x[j] for j in z_idxs])
        m_sigma = (m**sigma) / jnp.abs(sigma)
        m_sigma_safe = jnp.where(jnp.isnan(m_sigma), 0.0, m_sigma)
        row_id = x[row_id_idx].astype(jnp.int32)
        mbar_row = _mbar_at_rows(sigma)[row_id]
        u = (m_sigma_safe - mbar_row) * Rs * deltas
        moments = u[:, None] * zs[None, :]
        return moments.reshape(len(crops) * len(zs))

    return psi


def make_mask(obs, cols, crops, instruments):
    """Observability mask: moment ``(j, k)`` observed iff ``delta_j = 1``."""
    n = obs.shape[0]
    deltas = np.stack([obs[:, cols[f"delta_{c}"]] for c in crops], axis=1).astype(
        np.float64
    )
    mask = np.broadcast_to(deltas[:, :, None], (n, len(crops), len(instruments)))
    return mask.reshape(n, len(crops) * len(instruments)).copy()


def _dense(raw):
    """Contiguous int codes in ``[0, G)`` plus the count ``G``."""
    _, codes = np.unique(raw, return_inverse=True)
    codes = codes.astype(np.int64)
    return codes, int(codes.max()) + 1


# ── Fixtures (one estimation per covariance spec, module-scoped) ────


@pytest.fixture(scope="module")
def extract():
    with np.load(NPZ_PATH) as z:
        d = {k: z[k] for k in z.files}
    d["column_index"] = {str(name): j for j, name in enumerate(d["column_names"])}
    d["crop_list"] = [str(c) for c in d["crops"]]
    return d


@pytest.fixture(scope="module")
def bundle(extract):
    """Measure + psi + dense design codes, shared by both specs."""
    obs = extract["observation"]
    cols = extract["column_index"]
    crops = extract["crop_list"]
    n = obs.shape[0]
    n_t = int(extract["t_codes"].max()) + 1

    obs_aug = np.column_stack([obs, np.arange(n, dtype=np.float64)])
    psi = make_psi(obs_aug, cols, crops, INSTRUMENTS, n_t)
    mask = make_mask(obs, cols, crops, INSTRUMENTS)
    measure = EmpiricalMeasure(
        x=jnp.asarray(obs_aug),
        mask=jnp.asarray(mask),
        weights=jnp.ones(n),
    )

    cid = extract["cluster_ids"].astype(np.int64)
    arm = extract["arm_codes"].astype(np.int64)
    n_arms = int(arm.max()) + 1
    cid_codes, n_strata = _dense(cid)
    psu_codes, n_psu = _dense(extract["psu_ids"])
    cell_codes, n_cells = _dense(cid * n_arms + arm)
    return {
        "psi": psi,
        "measure": measure,
        "cid_codes": cid_codes,
        "n_strata": n_strata,
        "psu_codes": psu_codes,
        "n_psu": n_psu,
        "cell_codes": cell_codes,
        "n_cells": n_cells,
    }


def _estimate(
    bundle, covariance, *, optimizer=None, init_sigma=1.0
) -> EstimationResult:
    # optimizer=None auto-dispatches to RiemannianLM (sigma > 0, the actual
    # optimise). Passing riemannian_lm(max_steps=0) instead *evaluates* the
    # criterion at the fixed ``init_sigma`` without moving the iterate -- used
    # to pin the boundary-limit J at a stable in-zone sigma (#159).
    return estimate(
        model=bundle["psi"],
        measure=bundle["measure"],
        covariance=covariance,
        weighting=ContinuouslyUpdated(),
        regularization=DiagonalTikhonov(kappa_target=1e10),
        optimizer=optimizer,
        theta_init=EulerSigma(sigma=jnp.asarray(init_sigma)),
    )


@pytest.fixture(scope="module")
def fit_published(bundle) -> EstimationResult:
    """Published preferred spec: (stratum x arm)-clustered CUE."""
    covariance = ClusteredCovariance(
        cluster_ids=jnp.asarray(bundle["cell_codes"], dtype=jnp.float64),
        n_clusters=bundle["n_cells"],
    )
    return _estimate(bundle, covariance)


def _design_covariance(bundle) -> StratifiedCovariance:
    """Three-level StratifiedCovariance (PSU-in-cell-in-stratum, fpc off)."""
    return StratifiedCovariance(
        psu_ids=jnp.asarray(bundle["psu_codes"], dtype=jnp.float64),
        cell_ids=jnp.asarray(bundle["cell_codes"], dtype=jnp.float64),
        stratum_ids=jnp.asarray(bundle["cid_codes"], dtype=jnp.float64),
        n_psu=bundle["n_psu"],
        n_cells=bundle["n_cells"],
        n_strata=bundle["n_strata"],
        fpc=False,  # superpopulation estimand; matches the consumer
    )


@pytest.fixture(scope="module")
def fit_design(bundle) -> EstimationResult:
    """Design-based spec: the actual CUE optimise (collapses to sigma -> 0)."""
    return _estimate(bundle, _design_covariance(bundle))


@pytest.fixture(scope="module")
def fit_design_boundary_limit(bundle) -> EstimationResult:
    """Design J evaluated at a FIXED in-zone sigma (#159 robustness).

    ``riemannian_lm(max_steps=0)`` evaluates the criterion at
    ``SIGMA_BOUNDARY_EVAL`` without moving the iterate -- deterministic and
    platform-stable, unlike J read off the optimiser's flat-region,
    cancellation-prone stopping point.
    """
    return _estimate(
        bundle,
        _design_covariance(bundle),
        optimizer=riemannian_lm(max_steps=0),
        init_sigma=SIGMA_BOUNDARY_EVAL,
    )


# ── Extract integrity (cheap; no JAX compilation) ────────────────────


class TestExtractIntegrity:
    """The frozen extract is exactly the artifact the pins were taken on."""

    def test_npz_matches_recorded_sha256(self):
        sidecar = json.loads(PROVENANCE_PATH.read_text())
        actual = hashlib.sha256(NPZ_PATH.read_bytes()).hexdigest()
        assert actual == sidecar["sha256"], (
            "tests/data/seasonality_euler_extract.npz does not match the "
            "hash recorded in its provenance sidecar; if the extract was "
            "deliberately regenerated, re-pin the acceptance values and "
            "update the sidecar together."
        )

    def test_shape_and_design_counts(self, extract, bundle):
        assert extract["observation"].shape[0] == N_ROWS
        assert tuple(extract["crop_list"]) == CROPS
        assert int(extract["t_codes"].max()) + 1 == N_T
        assert bundle["n_strata"] == N_STRATA
        assert bundle["n_cells"] == N_CELLS
        assert bundle["n_psu"] == N_PSU_IN_SAMPLE

    def test_genuine_unequal_observability(self, extract):
        """Per-crop N_j pins; the unequal-N_j regime is real, not built."""
        obs = extract["observation"]
        cols = extract["column_index"]
        n_j = {c: int(obs[:, cols[f"delta_{c}"]].sum()) for c in extract["crop_list"]}
        assert n_j == N_J
        assert len(set(n_j.values())) > 1  # genuinely unequal

    def test_columns_finite_where_observed(self, extract):
        """m > 0 everywhere; R_j finite wherever delta_j = 1; z finite."""
        obs = extract["observation"]
        cols = extract["column_index"]
        m = obs[:, cols["m"]]
        assert np.isfinite(m).all() and (m > 0).all()
        for c in extract["crop_list"]:
            delta = obs[:, cols[f"delta_{c}"]]
            R = obs[:, cols[f"R_{c}"]]
            assert np.isfinite(R[delta == 1]).all()
        for z in INSTRUMENTS:
            assert np.isfinite(obs[:, cols[z]]).all()


# ── The published preferred spec (interior optimum) ──────────────────


class TestPublishedTreatmentSpec:
    """Treatment CUE at the (stratum x arm) cell level: G = 90.

    The pins cross-validate against the consumer's recorded values
    (sigma_hat = 1.2252, SE = 0.20; J = 6.574, p = 0.8325 in
    ``euler_real_cluster_compare.csv``) — this test reproducing them
    from the frozen extract is the consumer-contract half of #128.
    """

    def test_sigma_hat(self, fit_published):
        sigma = float(fit_published.theta_hat.sigma)
        assert sigma == pytest.approx(SIGMA_HAT_PUBLISHED, rel=1e-3)

    def test_sigma_se(self, fit_published):
        se = float(np.asarray(fit_published.standard_errors.array)[0])
        assert se == pytest.approx(SIGMA_SE_PUBLISHED, rel=1e-3)

    def test_J(self, fit_published):
        assert float(fit_published.J_stat) == pytest.approx(J_STAT_PUBLISHED, rel=1e-3)
        assert fit_published.J_dof == 11  # M=12, K=1
        assert float(fit_published.J_pvalue) == pytest.approx(
            J_PVALUE_PUBLISHED, abs=2e-3
        )

    def test_converged_interior_unregularized(self, fit_published):
        assert bool(fit_published.converged)
        # Interior optimum, ridge not binding: the chi-squared J applies.
        assert not bool(fit_published.diagnostics.binding_ridge)
        assert float(fit_published.diagnostics.tau_realised) == 0.0


# ── The design-based spec (boundary regime) ──────────────────────────


class TestDesignSpec:
    """StratifiedCovariance over PSU-in-cell-in-stratum (179/90/30).

    On the real data the CUE optimum under the design covariance collapses to
    the boundary ``sigma -> 0+`` (the weak-identification regime; the
    criterion's limit is the log-residual specification). Two separable
    claims, deliberately tested off *different* fits (#159):

    * the *actual optimise* (``fit_design``) collapses to the boundary and the
      ``Positive`` manifold keeps the iterate strictly positive -- but its
      exact stopping sigma is float-rounding noise (the criterion is flat
      there) so we do NOT pin J off it;
    * the *boundary-limit J* is pinned off ``fit_design_boundary_limit``, which
      evaluates the criterion at a fixed in-zone sigma where it is
      deterministic and platform-stable (above the ``m**sigma`` cancellation
      cliff; see SIGMA_BOUNDARY_EVAL and #159).
    """

    def test_boundary_collapse_stays_positive(self, fit_design):
        # The Positive-manifold guarantee on the real optimise: it collapses to
        # the sigma -> 0 boundary and never crosses zero. (Robust; the exact
        # sigma is platform-dependent and is NOT asserted -- see #159.)
        sigma = float(fit_design.theta_hat.sigma)
        assert 0.0 < sigma < 1e-6
        assert bool(fit_design.converged)

    def test_J_at_boundary_limit(self, fit_design_boundary_limit):
        # Boundary-limit J at a FIXED in-zone sigma (#159): deterministic and
        # platform-stable, so tightened from rel=1e-2 to rel=1e-4. The old pin
        # read J off the optimiser's flat-region, catastrophic-cancellation-
        # prone stopping point (J=5.44 on one OpenBLAS box vs ~7.08 on CI).
        assert float(fit_design_boundary_limit.J_stat) == pytest.approx(
            J_STAT_DESIGN, rel=1e-4
        )
        assert fit_design_boundary_limit.J_dof == 11

    def test_adjusted_pvalue(self, fit_design_boundary_limit):
        # The consumer's 'design' spec reads this p_adjusted off the result;
        # now pinned at the stable boundary limit (#159). The old 0.9081 was a
        # stale pre-migration orphan (#151).
        assert float(fit_design_boundary_limit.J_pvalue_adjusted) == pytest.approx(
            J_PVALUE_ADJ_DESIGN, abs=2e-4
        )

    def test_eval_sigma_is_in_the_stable_zone(self, bundle, fit_design_boundary_limit):
        # Guard the #159 fix: SIGMA_BOUNDARY_EVAL must sit in the cancellation-
        # free zone, where J is flat. If a future change pushed the cliff up
        # into it, J would diverge from J at a neighbouring in-zone sigma and
        # this fails loudly -- a clear signal, not a silently-wrong pin.
        j_neighbor = float(
            _estimate(
                bundle,
                _design_covariance(bundle),
                optimizer=riemannian_lm(max_steps=0),
                init_sigma=SIGMA_BOUNDARY_EVAL / 10.0,
            ).J_stat
        )
        assert float(fit_design_boundary_limit.J_stat) == pytest.approx(
            j_neighbor, rel=1e-3
        )
