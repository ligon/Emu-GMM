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
    build_estimator,
    estimate,
    k_confidence_set,
    k_statistic,
    profiled_k_confidence_set,
)
from emu_gmm.manifolds import ManifoldLeaf, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.studies import monte_carlo_study

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
        # #179: the profiled set re-references the concentrated K to the
        # SUBVECTOR dof (here dim(interest) == 1), so the hand-rolled reference
        # applies the same chi^2_1 rescoring to K. The K *statistic values* are
        # what the re-optimisation machinery must reproduce; only the reference
        # distribution (df=1, not the full-vector df=2=p_id) changes.
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
            ks = k_statistic(full, measure, IIDCovariance(), _iv_model)
            hand.append(float(jax.scipy.stats.chi2.sf(ks.K, 1)))
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


# ---------------------------------------------------------------------------
# (e) Subvector calibration Monte Carlo (#179), via the studies harness.
#
# The studies harness owns the draws AND the statistic: `monte_carlo_study`
# draws + fits the restricted (nuisance-only) model, its native `size_power`
# checks the restricted over-id J (chi^2_{M - dim_nuisance}), and the new
# per-draw `statistics=` channel collects the package's own subvector-K
# p-value (k_statistic(interest=[...])) at the true null -- no hand-rolled
# replication loop, no measure regeneration, no re-derived p-value. Under
# strong nuisance ID the subvector K (df = dim(interest)) is much closer to
# nominal than the pre-#179 full-vector dof, which under-rejects (conservative).
#
# The weak-nuisance degradation is the strong_id precondition (documented on
# profiled_k_confidence_set / k_statistic(interest=)); it is not gated here --
# the strong-vs-weak size gap is within MC noise at feasible rep counts, so a
# tight assertion would be flaky. The _mc_dgp(strong_nuisance=False) arm stays
# available for ad-hoc study.
#
# Interest is theta_s (always strongly instrumented); the profiled-out nuisance
# theta_w is strongly or weakly instrumented. True null (theta_s, theta_w) =
# (_B0, _G0). M = 3 moments, 2 params: dim(interest) = 1, full p_id = 2.
# ---------------------------------------------------------------------------

_MC_KEY = jax.random.PRNGKey(20260630)
_B0, _G0 = 1.5, -0.7


@jdc.pytree_dataclass
class _WOnly:
    """Nuisance-only reduced parameter (theta_s pinned at the truth _B0)."""

    theta_w: float


def _mc_dgp(strong_nuisance: bool, n: int = 1500):
    pi_a = jnp.array([1.4, 1.2, 1.1])  # interest theta_s: always strong instruments
    pi_b = (
        jnp.array([1.0, 0.9, 1.1])  # nuisance strongly instrumented
        if strong_nuisance
        else jnp.array([0.03, 0.025, 0.035])  # nuisance genuinely weak
    )

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        Z = jax.random.normal(k1, (n, 3))
        a = Z @ pi_a + 0.4 * jax.random.normal(k2, (n,))
        b = Z @ pi_b + 0.4 * jax.random.normal(k3, (n,))
        y = _B0 * a + _G0 * b + 0.3 * jax.random.normal(k4, (n,))
        X = jnp.column_stack([y[:, None], a[:, None], b[:, None], Z])
        return EmpiricalMeasure.from_arrays(X, M=3)

    return dgp


def _restricted_model(x, nu):
    # Nuisance-only reduced model: interest theta_s pinned at the truth _B0.
    return _iv_model(x, IVParams(theta_s=_B0, theta_w=nu.theta_w))


def _full_at(result):
    # Reconstruct the full null theta from the concentrated nuisance fit.
    return IVParams(theta_s=_B0, theta_w=result.theta_hat.theta_w)


def _p_K_sub(result, measure):
    # The package's OWN subvector-K p-value at the concentrated point.
    return k_statistic(
        _full_at(result), measure, IIDCovariance(), _iv_model, interest=["theta_s"]
    ).p_K


def _p_K_full(result, measure):
    # The pre-#179 full-vector reference (no `interest=`), for contrast.
    return k_statistic(_full_at(result), measure, IIDCovariance(), _iv_model).p_K


def _run_subvector_study(strong_nuisance: bool, n_reps: int = 300):
    dgp = _mc_dgp(strong_nuisance)
    run = build_estimator(  # factory => the no-leak cached path (CLAUDE.md)
        _restricted_model,
        measure=dgp(_MC_KEY),
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        parameters=_WOnly(theta_w=_G0),
    )
    return monte_carlo_study(
        run,
        dgp,
        n_reps=n_reps,
        key=_MC_KEY,
        theta_init=_WOnly(theta_w=_G0),
        theta0=_WOnly(theta_w=_G0),
        statistics={"p_K_sub": _p_K_sub, "p_K_full": _p_K_full},
    )


def _size(study, name: str, alpha: float = 0.05) -> float:
    p = np.asarray(study.records.extra[name])
    p = p[np.isfinite(p)]
    return float((p < alpha).mean())


class TestSubvectorCalibrationMC:
    @pytest.mark.slow
    def test_strong_nuisance_subvector_nominal_fullvector_conservative(self):
        # The core #179 claim, demonstrated through the studies harness: under
        # strong nuisance ID the subvector dof (df = dim(interest) = 1) is much
        # closer to nominal 5% size than the pre-#179 full-vector dof
        # (df = p_id = 2), which systematically under-rejects (conservative,
        # over-wide sets). Asserted as seed-robust *inequalities* rather than a
        # tight band on a 300-rep size estimate.
        study = _run_subvector_study(strong_nuisance=True)
        size_sub = _size(study, "p_K_sub")
        size_full = _size(study, "p_K_full")
        # The fix: the subvector reference is strictly closer to nominal...
        assert abs(size_sub - 0.05) < abs(size_full - 0.05)
        # ...the full-vector dof is strictly more conservative...
        assert size_full < size_sub
        # ...clearly below nominal (it under-rejects)...
        assert size_full < 0.035
        # ...and the subvector reference does not over-reject.
        assert size_sub <= 0.10
        # Native-harness leg: the restricted over-id J (chi^2_{M - dim_nuisance})
        # stays calibrated, read off the same records by size_power.
        ar = float(np.asarray(study.size_power.reject_nominal)[1])  # alpha=0.05
        assert 0.02 <= ar <= 0.10
        # The custom statistic also materializes in to_pandas (the #179 channel).
        assert "p_K_sub" in study.records.to_pandas().columns
