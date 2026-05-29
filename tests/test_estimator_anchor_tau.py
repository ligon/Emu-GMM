"""Anchor-once-then-freeze tau policy regression tests.

The framework's commitment 3 in CLAUDE.md and design.org §5 say tau is
chosen once at an anchor point and held fixed thereafter, so the
residual surface stays C^1 in theta. The previous implementation called
``regularization.apply(V(theta))`` inside the residual closure, and the
``jnp.where(kappa <= kappa_target, 0.0, tau_search)`` short-circuit
made tau discontinuous wherever kappa(V(theta)) crossed kappa_target.

These tests probe that:

1. The residual ``residual_fn(theta)`` is smooth in theta even when
   theta is varied so that kappa(V(theta)) crosses kappa_target. We
   sweep a 1-D ray through theta-space and check the gradient norm
   doesn't spike.

2. The reported diagnostic ``tau_realised`` matches the anchored value
   computed at theta_init, not a tau re-chosen at theta_hat.

Reference: docs/reviews/v1x-math-correctness.org [HIGH] finding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    ContinuouslyUpdated,
    DiagonalTikhonov,
    Identity,
    estimate,
)
from emu_gmm.optimizer import scipy_lm


@jdc.pytree_dataclass
class _Params1D:
    """Single scalar parameter for the smoothness probe."""

    a: float


def _dummy_model(x, theta):
    """Placeholder ``StructuralModel`` used by AnalyticalMeasure /
    AnalyticalCovariance — both ignore the model argument."""
    del x, theta
    return jnp.zeros((2,))


def _moment_fn(model, theta: _Params1D):
    """Closed-form (M=2) moment vector that vanishes at a = 0.

    Two moments scaled so that ``a`` drives the system away from the
    optimum and the population variance is well behaved at a=0.
    """
    del model
    return jnp.array([theta.a, 0.5 * theta.a])


def _make_V_with_kappa_crossing(target_kappa: float):
    """Return a covariance_fn whose kappa(V(theta.a)) sweeps across
    ``target_kappa`` as ``theta.a`` varies on a smooth 1-D ray.

    Concretely, we build V(a) = Q diag(1, sigma(a)^2) Q' with Q a fixed
    rotation; sigma(a) is a smooth monotone function of a, and at a=0
    sigma is small enough that the smaller eigenvalue is well below
    1/target_kappa, while at a=1 it is comfortably above.
    """
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((2, 2))
    Q_np, _ = np.linalg.qr(A)
    Q = jnp.asarray(Q_np)

    def covariance_fn(model, theta: _Params1D):
        del model
        # sigma(a) sweeps smoothly through a regime where the resulting
        # condition number crosses target_kappa.
        # eigenvalue1 = 1.0; eigenvalue2 = 1e-3 * exp(8 * a) (so at a=0,
        # kappa = 1e3 < target; at a=0.5, kappa ~ 1e3 * exp(-4) = ~18;
        # we pick the multiplier so that target=1e2 is crossed).
        lam2 = 1.0e-4 * jnp.exp(8.0 * theta.a)
        D = jnp.diag(jnp.array([1.0, lam2]))
        return Q @ D @ Q.T

    return covariance_fn


class TestAnchorTauSmoothness:
    """Validate that the residual is smooth in theta across kappa_target."""

    def _build_residual_closure(self, kappa_target: float):
        """Construct the same residual closure ``estimate`` uses.

        Returns ``(residual_fn, theta_init_flat, tau_anchor)``.
        """
        from emu_gmm._internal import params as pm
        from emu_gmm.regularization import DiagonalTikhonov

        cov_fn = _make_V_with_kappa_crossing(kappa_target)
        measure = AnalyticalMeasure(expectation_fn=_moment_fn)
        covariance = AnalyticalCovariance(covariance_fn=cov_fn)
        weighting = ContinuouslyUpdated()
        regularization = DiagonalTikhonov(kappa_target=kappa_target)

        theta_init = _Params1D(a=0.0)
        theta_init_flat, treedef = pm.flatten_params(theta_init)

        V0 = covariance.covariance(_dummy_model, theta_init, measure)
        _V0_star, tau_anchor = regularization.apply(V0)
        tau_anchor = jnp.asarray(tau_anchor)

        def residual_fn(theta_flat):
            theta = pm.unflatten_params(theta_flat, treedef)
            m = measure.expectation(_dummy_model, theta)
            V = covariance.covariance(_dummy_model, theta, measure)
            V_star = regularization.apply_fixed_tau(V, tau_anchor)
            return weighting.whitening_residual(m, V_star, theta)

        return residual_fn, theta_init_flat, tau_anchor

    def test_kappa_actually_crosses_target_in_sweep(self):
        """Sanity: the constructed covariance_fn really does have
        kappa(V(theta)) bracketing kappa_target somewhere on the sweep.
        Otherwise the smoothness test below is vacuous.
        """
        kappa_target = 1.0e2
        cov_fn = _make_V_with_kappa_crossing(kappa_target)
        a_grid = jnp.linspace(0.0, 1.0, 41)
        kappas = jnp.array(
            [
                float(jnp.linalg.cond(cov_fn(_dummy_model, _Params1D(a=float(a)))))
                for a in a_grid
            ]
        )
        assert jnp.min(kappas) < kappa_target
        assert jnp.max(kappas) > kappa_target

    def test_residual_is_C1_across_kappa_threshold(self):
        """The residual ``y(theta)`` is differentiable across the
        kappa_target threshold under the anchor-tau policy.

        With the old behaviour (tau recomputed per theta), the
        ``jnp.where`` short-circuit produced a step in tau where kappa
        crossed kappa_target, propagating into a step in y and a delta
        in the gradient. With anchor-tau, no such step exists.
        """
        kappa_target = 1.0e2
        residual_fn, _theta_init_flat, _ = self._build_residual_closure(kappa_target)

        # Build the scalar objective q(a) = 0.5 ||y(a)||^2 and check it
        # has a finite, bounded second derivative everywhere on the sweep.
        # A discontinuous tau would produce an unbounded second
        # difference at the threshold.
        def q(a_scalar):
            return 0.5 * jnp.sum(residual_fn(jnp.asarray([a_scalar])) ** 2)

        q_grad = jax.grad(q)

        a_grid = jnp.linspace(0.0, 1.0, 201)
        grads = jnp.array([float(q_grad(float(a))) for a in a_grid])

        # First-difference of the gradient. With anchor-tau the
        # objective is smooth, so successive grad-differences should be
        # bounded; without it, one entry would be order of magnitude
        # larger than the rest.
        diffs = jnp.abs(jnp.diff(grads))
        median_diff = float(jnp.median(diffs))
        max_diff = float(jnp.max(diffs))
        # Allow some headroom for the smooth nonlinearity itself; the
        # diagnostic is "no anomalous spike".
        assert max_diff < 50.0 * max(median_diff, 1e-10), (
            f"Anchor-tau residual is not smooth: max grad-diff {max_diff:.3e} "
            f">> median {median_diff:.3e}"
        )

    def test_tau_is_anchored_and_not_recomputed(self):
        """Diagnostics' tau_realised equals tau computed at theta_init,
        regardless of where theta_hat lands.
        """
        kappa_target = 1.0e2
        cov_fn = _make_V_with_kappa_crossing(kappa_target)

        def _model(x, t):
            return jnp.array([0.0])

        measure = AnalyticalMeasure(expectation_fn=_moment_fn)
        covariance = AnalyticalCovariance(covariance_fn=cov_fn)
        regularization = DiagonalTikhonov(kappa_target=kappa_target)

        theta_init = _Params1D(a=0.0)
        V0 = covariance.covariance(_model, theta_init, measure)
        _V0_star, tau_anchor_expected = regularization.apply(V0)

        result = estimate(
            model=_model,
            measure=measure,
            covariance=covariance,
            weighting=Identity(),
            regularization=regularization,
            optimizer=scipy_lm(),
            theta_init=theta_init,
        )

        # The reported tau matches the one anchored at theta_init.
        assert float(result.diagnostics.tau_realised) == pytest.approx(
            float(tau_anchor_expected), rel=1e-8, abs=1e-12
        )


class TestApplyFixedTau:
    """Direct unit tests for ``DiagonalTikhonov.apply_fixed_tau``."""

    def test_returns_v_plus_tau_diag(self):
        reg = DiagonalTikhonov()
        V = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        tau = 0.05
        V_star = reg.apply_fixed_tau(V, tau)
        expected = V + tau * jnp.diag(jnp.diag(V))
        assert jnp.allclose(V_star, expected)

    def test_is_C1_in_tau(self):
        """Differentiable in tau (no jnp.where short-circuit)."""
        reg = DiagonalTikhonov()
        V = jnp.array([[2.0, 1.0], [1.0, 3.0]])

        def f(tau):
            V_star = reg.apply_fixed_tau(V, tau)
            return jnp.sum(V_star)

        # gradient at multiple values should agree with the closed form
        # (which is constant: sum of diag(V) = 5.0).
        for tau_val in [0.0, 1e-6, 1e-3, 1.0]:
            g = float(jax.grad(f)(jnp.asarray(tau_val)))
            assert g == pytest.approx(5.0, rel=1e-8)
