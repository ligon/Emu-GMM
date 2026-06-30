r"""Tests for :func:`emu_gmm.inference.profiled_k_confidence_set` (#176).

Coverage (the issue's acceptance criteria):

(a) Matches a hand-rolled re-optimising ``theta_builder`` — the documented
    workaround — on a Euclidean nuisance, exactly (same p-value per grid point).

(b) Composes with a manifold nuisance: profiling a Euclidean interest field
    while a ``PSDFixedRank`` ``Gamma`` factor is the re-optimised nuisance
    matches a hand-rolled re-estimate of the manifold factor, with the gauge
    handled by the existing quotient-aware ``estimate`` path (finite p-values,
    a clean interval).

(c) Weak vs strong: the profiled set over a weakly-identified coordinate is
    unbounded (open at a grid edge) while a strongly-identified coordinate's
    set is a bounded interval.

(d) The ``k_confidence_set(..., profile=[...])`` hook delegates identically;
    validation of the profile spec and the misuse guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    IIDCovariance,
    estimate,
    k_confidence_set,
    k_statistic,
    profiled_k_confidence_set,
)
from emu_gmm.manifolds import ManifoldLeaf, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.optimizer import optimistix_lm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Weak-instrument fixture: theta_s strong, theta_w weak.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class IVParams:
    theta_s: float
    theta_w: float


@jdc.pytree_dataclass
class _SOnly:
    theta_s: float


def _iv_model(x, theta):
    y = x[0]
    a = x[1]
    b = x[2]
    z = x[3:6]
    return z * (y - theta.theta_s * a - theta.theta_w * b)


def _fit_weak_iv(seed: int = 1, n: int = 3000):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    pi_a = np.array([1.4, 1.2, 1.1])  # strong
    pi_b = np.array([0.015, 0.012, 0.010])  # genuinely weak
    a = Z @ pi_a + rng.normal(size=n) * 0.4
    b = Z @ pi_b + rng.normal(size=n) * 0.4
    y = 1.5 * a - 0.7 * b + rng.normal(size=n) * 0.3
    X = np.column_stack([y, a, b, Z])
    measure = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    result = estimate(
        _iv_model,
        measure,
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        theta_init=IVParams(theta_s=1.5, theta_w=-0.7),
    )
    return result


def _fit_strong_iv(seed: int = 2, n: int = 3000):
    """Both regressors strongly instrumented — both coordinates well-identified."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    pi_a = np.array([1.4, 1.2, 1.1])
    pi_b = np.array([1.0, 0.9, 1.1])  # strong (cf. the weak fixture)
    a = Z @ pi_a + rng.normal(size=n) * 0.4
    b = Z @ pi_b + rng.normal(size=n) * 0.4
    y = 1.5 * a - 0.7 * b + rng.normal(size=n) * 0.3
    X = np.column_stack([y, a, b, Z])
    measure = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    return estimate(
        _iv_model,
        measure,
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        theta_init=IVParams(theta_s=1.5, theta_w=-0.7),
    )


# ---------------------------------------------------------------------------
# (a) Matches the hand-rolled re-optimising builder (Euclidean nuisance).
# ---------------------------------------------------------------------------


class TestMatchesHandRolled:
    @pytest.mark.slow
    def test_profiled_equals_reoptimising_theta_builder(self):
        result = _fit_weak_iv()
        ts = float(result.theta_hat.theta_s)
        tw = float(result.theta_hat.theta_w)
        se = np.asarray(result.standard_errors.array)
        measure = result.measure
        grid = np.linspace(tw - 8 * se[1], tw + 8 * se[1], 9)

        def builder(g):
            return IVParams(theta_s=ts, theta_w=g)  # nuisance warm start at ts

        cs = profiled_k_confidence_set(
            builder, grid, measure, IIDCovariance(), _iv_model, profile=["theta_w"]
        )

        # Hand-rolled re-optimising builder (the documented workaround): at each
        # grid value, re-estimate theta_s with theta_w fixed, then k_statistic.
        hand = []
        for g in grid:

            def red(x, s, _g=float(g)):
                return _iv_model(x, IVParams(theta_s=s.theta_s, theta_w=_g))

            inner = estimate(
                red,
                measure,
                covariance=IIDCovariance(),
                weighting=ContinuouslyUpdated(),
                optimizer=optimistix_lm(),
                theta_init=_SOnly(theta_s=ts),
            )
            full = IVParams(theta_s=inner.theta_hat.theta_s, theta_w=float(g))
            hand.append(
                float(k_statistic(full, measure, IIDCovariance(), _iv_model).p_K)
            )
            jax.clear_caches()

        np.testing.assert_allclose(cs.p_grid, np.array(hand), rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    def test_hook_delegates_identically(self):
        result = _fit_weak_iv()
        ts = float(result.theta_hat.theta_s)
        tw = float(result.theta_hat.theta_w)
        se = np.asarray(result.standard_errors.array)
        grid = np.linspace(tw - 6 * se[1], tw + 6 * se[1], 5)

        def builder(g):
            return IVParams(theta_s=ts, theta_w=g)

        direct = profiled_k_confidence_set(
            builder,
            grid,
            result.measure,
            IIDCovariance(),
            _iv_model,
            profile=["theta_w"],
        )
        via_hook = k_confidence_set(
            builder,
            grid,
            result.measure,
            IIDCovariance(),
            _iv_model,
            profile=["theta_w"],
        )
        np.testing.assert_array_equal(direct.p_grid, via_hook.p_grid)
        assert direct.topology == via_hook.topology


# ---------------------------------------------------------------------------
# (c) Weak coordinate -> unbounded; strong coordinate -> bounded interval.
# ---------------------------------------------------------------------------


class TestWeakVsStrong:
    @pytest.mark.slow
    def test_weak_interest_profiled_set_is_unbounded(self):
        # Weak fixture: profile the WEAK theta_w (re-optimise the strong
        # theta_s). The criterion stays flat across a wide window, so the
        # identification-robust set runs off a grid edge (open / unbounded) —
        # the honest "unbounded is the answer" the issue motivates.
        result = _fit_weak_iv()
        ts = float(result.theta_hat.theta_s)
        tw = float(result.theta_hat.theta_w)
        se = np.asarray(result.standard_errors.array)
        gw = np.linspace(tw - 10 * se[1], tw + 10 * se[1], 13)
        weak = profiled_k_confidence_set(
            lambda g: IVParams(theta_s=ts, theta_w=g),
            gw,
            result.measure,
            IIDCovariance(),
            _iv_model,
            profile=["theta_w"],
        )
        assert weak.open_left or weak.open_right

    @pytest.mark.slow
    def test_strong_interest_profiled_set_is_bounded(self):
        # Both-strong fixture: profile the strong theta_s (re-optimise the
        # strong theta_w nuisance). With the nuisance well-identified the
        # concentrated set is a bounded interval strictly inside the window.
        # (A weak nuisance would legitimately widen even a strong interest's
        # set — which is exactly why this claim needs a strong nuisance.)
        result = _fit_strong_iv()
        ts = float(result.theta_hat.theta_s)
        tw = float(result.theta_hat.theta_w)
        se = np.asarray(result.standard_errors.array)
        gs = np.linspace(ts - 8 * se[0], ts + 8 * se[0], 13)
        strong = profiled_k_confidence_set(
            lambda g: IVParams(theta_s=g, theta_w=tw),
            gs,
            result.measure,
            IIDCovariance(),
            _iv_model,
            profile=["theta_s"],
        )
        assert not strong.open_left and not strong.open_right
        assert strong.topology in ("interval", "disconnected")


# ---------------------------------------------------------------------------
# (b) Manifold nuisance: profile a Euclidean field, re-optimise a PSDFixedRank
#     Gamma factor. Gauge handled by the existing quotient-aware estimate path.
# ---------------------------------------------------------------------------


def _load_phase4_fixture():
    manifolds_dir = Path(__file__).resolve().parents[1] / "manifolds"
    if str(manifolds_dir) not in sys.path:
        sys.path.insert(0, str(manifolds_dir))
    import test_estimator_inference_phase4 as ph4

    return ph4


class TestManifoldNuisance:
    @pytest.mark.slow
    def test_profile_euclidean_reoptimise_psd_matches_hand_rolled(self):
        ph4 = _load_phase4_fixture()
        k = 2
        result, _spec, _M, _ = ph4._run_estimate(k, seed=300)
        measure, cov = result.measure, result.covariance
        phi_hat = float(result.theta_hat.phi.array[0])
        Y_hat = jnp.asarray(np.asarray(result.theta_hat.Y.array))

        def builder(g):
            return ph4._make_params(Y_hat, g, k)  # phi = g, Y warm-started

        grid = np.linspace(phi_hat - 0.4, phi_hat + 0.4, 5)
        cs = profiled_k_confidence_set(
            builder, grid, measure, cov, ph4._model, profile=["phi"]
        )
        # Gauge handled: every grid point yields a finite p-value (no inf/nan
        # from the PSDFixedRank gauge directions).
        assert np.all(np.isfinite(cs.p_grid))

        # Hand-rolled: re-estimate the manifold Y with phi fixed (the nuisance
        # optimiser auto-dispatches to riemannian_lm for the manifold leaf).
        @jdc.pytree_dataclass
        class _YOnly:
            Y: object

        hand = []
        for g in grid:

            def red(x, yo, _g=float(g)):
                return ph4._model(x, ph4._make_params(yo.Y.array, _g, k))

            inner = estimate(
                red,
                measure,
                covariance=cov,
                optimizer=riemannian_lm(max_steps=400),
                theta_init=_YOnly(Y=ManifoldLeaf(Y_hat, PSDFixedRank(ph4.N, k))),
            )
            full = ph4._make_params(inner.theta_hat.Y.array, float(g), k)
            hand.append(float(k_statistic(full, measure, cov, ph4._model).p_K))
            jax.clear_caches()

        np.testing.assert_allclose(cs.p_grid, np.array(hand), rtol=1e-8, atol=1e-8)

    @pytest.mark.slow
    def test_phi_is_identified_bounded_interval(self):
        ph4 = _load_phase4_fixture()
        k = 2
        result, _spec, _M, _ = ph4._run_estimate(k, seed=301)
        phi_hat = float(result.theta_hat.phi.array[0])
        Y_hat = jnp.asarray(np.asarray(result.theta_hat.Y.array))
        grid = np.linspace(phi_hat - 0.5, phi_hat + 0.5, 7)
        cs = profiled_k_confidence_set(
            lambda g: ph4._make_params(Y_hat, g, k),
            grid,
            result.measure,
            result.covariance,
            ph4._model,
            profile=["phi"],
        )
        # phi is strongly identified: a bounded interval inside the window.
        assert cs.topology == "interval"
        assert not cs.open_left and not cs.open_right


# ---------------------------------------------------------------------------
# (d) Validation + misuse guards.
# ---------------------------------------------------------------------------


class TestValidationAndApi:
    # The validation guards fire BEFORE any inner estimate (in _ProfileReducer
    # construction / the k_confidence_set hook guard), so a tiny measure and a
    # fixed builder suffice — no fit needed, keeping these in the fast gate.
    _MEASURE = EmpiricalMeasure.from_arrays(
        jnp.asarray(np.random.default_rng(0).normal(size=(50, 6))), M=3
    )
    _GRID = np.linspace(-1.0, 1.0, 4)

    @staticmethod
    def _builder(g):
        return IVParams(theta_s=0.0, theta_w=g)

    def test_unknown_profile_field(self):
        with pytest.raises(ValueError, match="not"):
            profiled_k_confidence_set(
                self._builder,
                self._GRID,
                self._MEASURE,
                IIDCovariance(),
                _iv_model,
                profile=["nope"],
            )

    def test_all_fields_fixed(self):
        with pytest.raises(ValueError, match="nothing to optimise"):
            profiled_k_confidence_set(
                self._builder,
                self._GRID,
                self._MEASURE,
                IIDCovariance(),
                _iv_model,
                profile=["theta_s", "theta_w"],
            )

    def test_empty_profile(self):
        with pytest.raises(ValueError, match=">= 1 field"):
            profiled_k_confidence_set(
                self._builder,
                self._GRID,
                self._MEASURE,
                IIDCovariance(),
                _iv_model,
                profile=[],
            )

    def test_nuisance_optimizer_without_profile_is_rejected(self):
        with pytest.raises(ValueError, match="profiled path"):
            k_confidence_set(
                self._builder,
                self._GRID,
                self._MEASURE,
                IIDCovariance(),
                _iv_model,
                nuisance_optimizer=optimistix_lm(),
            )

    @pytest.mark.slow
    def test_return_profiled_points(self):
        result = _fit_weak_iv()
        ts = float(result.theta_hat.theta_s)
        tw = float(result.theta_hat.theta_w)
        grid = np.linspace(tw - 1, tw + 1, 5)
        cs, pts = profiled_k_confidence_set(
            lambda g: IVParams(theta_s=ts, theta_w=g),
            grid,
            result.measure,
            IIDCovariance(),
            _iv_model,
            profile=["theta_w"],
            return_profiled_points=True,
        )
        assert len(pts) == len(grid)
        # Each profiled point holds theta_w at its grid value (the fixed field).
        for g, pt in zip(grid, pts, strict=True):
            np.testing.assert_allclose(float(pt.theta_w), float(g), rtol=1e-10)
