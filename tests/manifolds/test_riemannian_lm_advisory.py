"""#152 advisory: riemannian_lm stall -> horizontal-Hessian probe -> warning.

When a gauge-bearing solve converges, ``riemannian_lm`` probes the smallest
eigenvalue of the horizontal true Hessian at ``theta_hat`` (the ``k(k-1)/2``
gauge directions are projected to ~0) and, if it is indefinite, warns that the
criterion looks non-convex (a saddle) and suggests ``riemannian_tr()``, setting
``OptimizerInfo.stalled_indefinite`` / ``.min_curvature``.

The warning fires ONLY at a genuine stationary point (the exact ``grad_ok``
test, recomputed at ``theta_hat``) with an indefinite Hessian. A #156 ftol
(cost-stagnation) stop is NOT a stationary point -- the iterate drifts at a
large gradient on a correct estimate -- so it is excluded (no false positive on
the common CU+clustered pattern).

These guard:
  - the curvature detector ``_min_horizontal_curvature`` on a KNOWN indefinite
    vs PSD Riemannian Hessian;
  - the end-to-end warning at a CONSTRUCTED saddle (LM started at a critical
    point whose horizontal Hessian is indefinite);
  - silence at a genuine minimum (the probe runs, finds PSD, does not warn);
  - silence at an ftol-drift stop (NOT stationary -> probe skipped, fields None);
  - the ``advise_nonconvex=False`` kill-switch (no probe, fields ``None``).

Finding (2026-06-17): on gauge-invariant ``PSDFixedRank`` fixtures
``riemannian_lm`` reaches local MINIMA, not saddles -- even on deliberately
non-convex fixtures where RTR's tCG fires negative curvature *transiently on the
path* (see ``test_rtr_pymanopt_parity.TestNonConvexMetaGate``). The warning
therefore fires only at a genuinely indefinite *stationary point*, which here is
constructed explicitly via a left-null-space residual offset. The advisory is a
correct, cheap safety net for consumer non-convex criteria (#152); it is silent
on the in-repo fixtures because they converge to minima.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.covariance import ClusteredCovariance, SyntheticCovariance
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import _min_horizontal_curvature, riemannian_lm
from emu_gmm.measures import EmpiricalMeasure, SyntheticMeasure
from emu_gmm.weighting import Identity

jax.config.update("jax_enable_x64", True)

_N, _K = 5, 2
_TRIU = jnp.array(np.triu_indices(_N)).T


@jdc.pytree_dataclass
class _P:
    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _mk(Y, phi) -> _P:
    return _P(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(_N, _K)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _triu(G):
    return G[_TRIU[:, 0], _TRIU[:, 1]]


def _frozen_measure(target):
    """A noise-free synthetic measure whose empirical moment is exactly target."""
    return SyntheticMeasure(
        key=jax.random.PRNGKey(0),
        n_sim=1,
        sampler=lambda key, theta: jnp.asarray(target)[None, :],
    )


def _nonconvex_warning(records):
    return [str(r.message) for r in records if "non-convex" in str(r.message)]


# ---------------------------------------------------------------------------
# 1. The curvature detector in isolation.
# ---------------------------------------------------------------------------
class TestCurvatureProbeDetector:
    """``_min_horizontal_curvature`` flags negative curvature and only that.

    Constructed so the ambient gradient at the eval point is zero (the residual
    Jacobian vanishes there), so the retraction correction drops out and the
    Riemannian Hessian equals the projected ambient Hessian ``c0 * Q``. With
    ``Q`` negative on the ``Y`` block, the horizontal eigenvalues are ``-c0``
    while the gauge direction is projected to ~0 (it does NOT masquerade as
    curvature). The PSD control flips ``Q`` positive.
    """

    def _flat_star(self):
        Ystar = jnp.asarray(np.random.default_rng(1).normal(size=(_N, _K)))
        theta = _mk(Ystar, 0.5)
        spec = manifold_spec_from_params(theta)
        flat_star = jnp.concatenate([Ystar.reshape(-1), jnp.array([0.5])])
        return spec, flat_star

    def test_indefinite_hessian_detected(self):
        spec, flat_star = self._flat_star()
        c0 = 3.0
        qdiag = jnp.concatenate([-jnp.ones(_N * _K), jnp.array([1.0])])

        def residual(flat):
            d = flat - flat_star
            return jnp.array([c0 + 0.5 * jnp.sum(qdiag * d * d)])

        lam_min, lam_max = _min_horizontal_curvature(residual, spec, flat_star)
        assert lam_min == pytest.approx(-c0, abs=1e-6)
        assert lam_max == pytest.approx(c0, abs=1e-6)

    def test_psd_hessian_stays_nonnegative(self):
        spec, flat_star = self._flat_star()
        c0 = 3.0

        def residual_psd(flat):
            d = flat - flat_star
            return jnp.array([c0 + 0.5 * jnp.sum(d * d)])

        lam_min, lam_max = _min_horizontal_curvature(residual_psd, spec, flat_star)
        assert lam_min > -1e-6  # gauge directions ~0, horizontal directions +c0
        assert lam_max == pytest.approx(c0, abs=1e-6)


# ---------------------------------------------------------------------------
# A constructed saddle: LM started at a critical point with indefinite Hessian.
# M=12 (>= K_id=10) small linear fillers identify the problem and are exactly
# zero at flat*; one indefinite quadratic supplies the negative curvature. At
# flat* the horizontal gradient is zero, so LM certifies in one step.
# ---------------------------------------------------------------------------
_EPS, _C0 = 1e-2, 3.0
_QDIAG = jnp.concatenate([-jnp.ones(_N * _K), jnp.array([1.0])])
_YSTAR = jnp.asarray(np.random.default_rng(1).normal(size=(_N, _K)))
_PHISTAR = 0.5
_FLAT_STAR = jnp.concatenate([_YSTAR.reshape(-1), jnp.array([_PHISTAR])])
_KDIM = _N * _K + 1


def _saddle_model(x, theta):
    flat = jnp.concatenate([theta.Y.array.reshape(-1), theta.phi.array])
    d = flat - _FLAT_STAR
    lin = _EPS * d  # K linear fillers, zero at flat*, full horizontal rank
    quad = _C0 + 0.5 * jnp.sum(_QDIAG * d * d)  # indefinite quadratic
    return jnp.concatenate([lin, jnp.array([quad])]) - x  # M = K + 1 = 12


def _run_saddle(**lm_kw):
    measure = _frozen_measure(jnp.zeros(_KDIM + 1))
    with warnings.catch_warnings(record=True) as recs:
        warnings.simplefilter("always")
        res = estimate(
            model=_saddle_model,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=Identity(),
            optimizer=riemannian_lm(max_steps=400, **lm_kw),
            theta_init=_mk(_YSTAR, _PHISTAR),
        )
    return res, _nonconvex_warning(recs)


@pytest.mark.slow  # estimate()-based; full-suite/nightly gate (#152)
class TestAdvisoryWarnsAtIndefiniteStationaryPoint:
    def test_warning_fires_and_fields_set(self):
        res, warned = _run_saddle()
        info = res.diagnostics.optimizer_info
        assert bool(res.converged) is True
        assert info.stalled_indefinite is True
        assert float(info.min_curvature) < -1e-6
        assert len(warned) == 1
        assert "riemannian_tr()" in warned[0]

    def test_advise_nonconvex_false_silences_and_skips_probe(self):
        res, warned = _run_saddle(advise_nonconvex=False)
        info = res.diagnostics.optimizer_info
        assert bool(res.converged) is True
        assert len(warned) == 0
        # The probe is skipped entirely: both fields stay None.
        assert info.stalled_indefinite is None
        assert info.min_curvature is None


# ---------------------------------------------------------------------------
# Silence at a genuine minimum: a convex (linear-in-Gamma) gauge-invariant
# solve from near the truth converges via grad_ok to a PSD-Hessian minimum, so
# the probe RUNS (min_curvature is set, proving it is not None by accident) but
# finds no negative curvature and does NOT warn.
# ---------------------------------------------------------------------------
def _convex_model(x, theta):
    Y = theta.Y.array
    phi = theta.phi.array[0]
    return jnp.concatenate([_triu(Y @ Y.T), jnp.reshape(phi, (1,))]) - x


@pytest.mark.slow  # estimate()-based; full-suite/nightly gate (#152)
class TestAdvisorySilentAtMinimum:
    def test_convex_minimum_probed_but_no_warning(self):
        rng = np.random.default_rng(2)
        A = jnp.asarray(rng.normal(size=(_N, _K)))
        target = jnp.concatenate([_triu(A @ A.T), jnp.array([0.7])])
        measure = _frozen_measure(target)
        Y0 = jnp.asarray(A + 0.05 * rng.normal(size=(_N, _K)))
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            res = estimate(
                model=_convex_model,
                measure=measure,
                covariance=SyntheticCovariance(),
                weighting=Identity(),
                optimizer=riemannian_lm(max_steps=400),
                theta_init=_mk(Y0, 0.65),
            )
        info = res.diagnostics.optimizer_info
        assert bool(res.converged) is True
        # The probe ran (a gauge-bearing converged solve) ...
        assert info.min_curvature is not None
        # ... and found a PSD minimum, so it did not flag indefiniteness.
        assert info.stalled_indefinite is False
        assert len(_nonconvex_warning(recs)) == 0


class TestAdvisorySilentAtFtolDriftStop:
    """A #156 CU+clustered solve certifies via cost stagnation while the
    horizontal gradient is still large (~1e-1): it DRIFTS at constant cost, so it
    is NOT a stationary point. Its Hessian is trivially indefinite, but probing
    it would warn on a CORRECT estimate (a false positive on the common
    CU+clustered pattern). The stationarity gate excludes it: no warning, and the
    advisory fields stay None.
    """

    @pytest.mark.slow
    def test_ftol_drift_is_not_probed_or_warned(self):
        n4, k2 = 4, 2
        triu4 = jnp.array(np.triu_indices(n4)).T
        m_dim = n4 * (n4 + 1) // 2 + 1

        @jdc.pytree_dataclass
        class _P4:
            A: ManifoldLeaf
            phi: ManifoldLeaf

        def model(x, th):
            A = th.A.array
            phi = th.phi.array[0]
            g = (A @ A.T)[triu4[:, 0], triu4[:, 1]]
            return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x

        rng = np.random.default_rng(3)
        A = jnp.asarray(rng.normal(size=(n4, k2)))
        tgt = jnp.concatenate([(A @ A.T)[triu4[:, 0], triu4[:, 1]], jnp.array([0.7])])
        nc, opc = 6, 40
        n = nc * opc
        x = np.asarray(tgt)[None, :] + 0.1 * rng.standard_normal((n, m_dim))
        meas = EmpiricalMeasure(
            x=jnp.asarray(x), mask=jnp.ones((n, m_dim)), weights=jnp.ones(n)
        )
        cid = jnp.repeat(jnp.arange(nc, dtype=jnp.float64), opc)
        cov = ClusteredCovariance(cluster_ids=cid, n_clusters=nc)
        theta = _P4(
            A=ManifoldLeaf(jnp.asarray(A), PSDFixedRank(n4, k2)),
            phi=ManifoldLeaf(jnp.array([0.7]), Euclidean(1)),
        )
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            res = estimate(  # default weighting is continuously-updated
                model=model, measure=meas, covariance=cov, theta_init=theta
            )
        info = res.diagnostics.optimizer_info
        assert bool(res.converged) is True
        # NOT a stationary point (ftol drift) -> probe skipped, fields None,
        # nothing warned, even though the drift-point Hessian is indefinite.
        assert info.stalled_indefinite is None
        assert info.min_curvature is None
        assert len(_nonconvex_warning(recs)) == 0
