"""Tests for the in-objective theta penalty hook (issue #7).

Covers:

- The :class:`TikhonovPenalty` itself: value and gradient on toy theta.
- ``estimate(..., penalty=None)`` reproduces the v1 result bitwise.
- ``estimate(..., penalty=TikhonovPenalty(c=...))`` pulls ``theta_hat``
  toward the origin in the expected direction and magnitude.
- AD through the penalty composes with AD through CU whitening (the
  combined gradient is finite and consistent with finite differences).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.penalty import PenaltyStrategy, TikhonovPenalty
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated

N_SIM = 5000


# ---------------------------------------------------------------------------
# Unit tests on TikhonovPenalty itself
# ---------------------------------------------------------------------------


class TestTikhonovPenaltyValue:
    """The penalty value matches the closed-form c * ||theta_flat||^2."""

    def test_zero_at_origin(self):
        pen = TikhonovPenalty(c=jnp.asarray(0.5))
        theta = EulerParams(beta=0.0, gamma=0.0)
        assert float(pen.penalty(theta)) == pytest.approx(0.0)

    def test_scales_quadratically(self):
        pen = TikhonovPenalty(c=jnp.asarray(1.0))
        theta = EulerParams(beta=0.9, gamma=2.0)
        expected = 0.9**2 + 2.0**2
        assert float(pen.penalty(theta)) == pytest.approx(expected, rel=1e-12)

    def test_scales_with_c(self):
        pen = TikhonovPenalty(c=jnp.asarray(3.0))
        theta = EulerParams(beta=0.5, gamma=1.5)
        expected = 3.0 * (0.5**2 + 1.5**2)
        assert float(pen.penalty(theta)) == pytest.approx(expected, rel=1e-12)


class TestTikhonovPenaltyGradient:
    """Gradient is 2c*theta, returned in the original pytree shape."""

    def test_gradient_shape_matches_theta(self):
        pen = TikhonovPenalty(c=jnp.asarray(0.5))
        theta = EulerParams(beta=0.9, gamma=2.0)
        g = pen.gradient(theta)
        assert isinstance(g, EulerParams)

    def test_gradient_values(self):
        c = 0.5
        pen = TikhonovPenalty(c=jnp.asarray(c))
        theta = EulerParams(beta=0.9, gamma=2.0)
        g = pen.gradient(theta)
        assert float(g.beta) == pytest.approx(2 * c * 0.9, rel=1e-12)
        assert float(g.gamma) == pytest.approx(2 * c * 2.0, rel=1e-12)


class TestProtocolConformance:
    """TikhonovPenalty satisfies the runtime-checkable PenaltyStrategy."""

    def test_isinstance_protocol(self):
        pen = TikhonovPenalty(c=jnp.asarray(1.0))
        assert isinstance(pen, PenaltyStrategy)


# ---------------------------------------------------------------------------
# Estimator wiring: penalty=None preserves v1 behaviour
# ---------------------------------------------------------------------------


def _make_measure() -> SyntheticMeasure:
    sampler = euler_sampler_factory(N_SIM)
    return SyntheticMeasure(
        key=jax.random.PRNGKey(0),
        n_sim=N_SIM,
        sampler=sampler,
    )


def _run(penalty=None):
    return estimate(
        model=euler_residual,
        measure=_make_measure(),
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        regularization=DiagonalTikhonov(),
        optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        theta_init=EulerParams(beta=0.9, gamma=1.0),
        penalty=penalty,
    )


class TestPenaltyNoneIsV1:
    """penalty=None must reproduce the v1 estimator output bitwise."""

    def test_theta_hat_identical_to_no_penalty_kwarg(self):
        # Reference: call without supplying the penalty kwarg at all.
        ref = estimate(
            model=euler_residual,
            measure=_make_measure(),
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )
        new = _run(penalty=None)
        assert float(new.theta_hat.beta) == float(ref.theta_hat.beta)
        assert float(new.theta_hat.gamma) == float(ref.theta_hat.gamma)
        assert new.J_stat == ref.J_stat
        assert new.J_dof == ref.J_dof


# ---------------------------------------------------------------------------
# Penalty pulls theta_hat toward zero
# ---------------------------------------------------------------------------


class TestPenaltyPullsToZero:
    """Adding a Tikhonov penalty biases theta_hat toward the origin."""

    def test_beta_smaller_with_penalty(self):
        unpen = _run(penalty=None)
        pen = _run(penalty=TikhonovPenalty(c=jnp.asarray(10.0)))
        # beta_unpen ~ 0.96 (truth); the penalty pulls toward 0, so the
        # penalised estimate should be strictly smaller.
        assert float(pen.theta_hat.beta) < float(unpen.theta_hat.beta)

    def test_gamma_smaller_with_penalty(self):
        unpen = _run(penalty=None)
        pen = _run(penalty=TikhonovPenalty(c=jnp.asarray(10.0)))
        # gamma_unpen ~ 2.0; penalty toward 0 ==> smaller penalised gamma.
        assert float(pen.theta_hat.gamma) < float(unpen.theta_hat.gamma)

    def test_huge_penalty_shrinks_substantially(self):
        # With a very large c the penalty dominates and theta_hat should
        # shrink substantially from the unpenalised truth. We test
        # against an intermediate c relative to the unpenalised values:
        # both estimates strictly drop and the shrinkage is monotone in c.
        small = _run(penalty=TikhonovPenalty(c=jnp.asarray(1.0)))
        big = _run(penalty=TikhonovPenalty(c=jnp.asarray(1e6)))
        # Stronger penalty => more shrinkage.
        assert float(big.theta_hat.beta) < float(small.theta_hat.beta)
        # gamma is unbounded; with a strong ridge it must be near 0.
        assert abs(float(big.theta_hat.gamma)) < 0.2
        # beta also shrinks but the data anchor (E[psi]=0 at beta near 0.96)
        # competes with the penalty; with c=1e6 beta should still be well
        # below BETA_TRUE.
        assert float(big.theta_hat.beta) < 0.8 * BETA_TRUE

    def test_zero_c_recovers_unpenalised(self):
        # c = 0 means p(theta) = 0; theta_hat should match the v1 result.
        unpen = _run(penalty=None)
        pen = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.0)))
        assert float(pen.theta_hat.beta) == pytest.approx(
            float(unpen.theta_hat.beta), abs=1e-6
        )
        assert float(pen.theta_hat.gamma) == pytest.approx(
            float(unpen.theta_hat.gamma), abs=1e-6
        )


# ---------------------------------------------------------------------------
# AD through penalty composes with AD through whitening
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _TwoParam:
    a: float
    b: float


def _toy_model(x, theta):
    # Linear-in-theta residual: psi_j = a + b * x_j (two moments).
    return jnp.array([theta.a + theta.b * x[0], theta.a + theta.b * x[1]])


class TestADCompositionWithCU:
    """AD threads cleanly through penalty + CU whitening + covariance."""

    def test_gradient_finite_at_general_theta(self):
        # Build a small synthetic measure with a fixed sample.
        x_sample = jnp.array([[0.5, 1.0], [1.5, 2.0], [-0.5, 0.3]])

        def sampler(key, theta):
            del key, theta
            return x_sample

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=3,
            sampler=sampler,
        )
        cov = SyntheticCovariance()
        reg = DiagonalTikhonov()
        weight = ContinuouslyUpdated()
        pen = TikhonovPenalty(c=jnp.asarray(0.25))

        def half_obj_pen(theta_flat):
            theta = _TwoParam(a=theta_flat[0], b=theta_flat[1])
            m = measure.expectation(_toy_model, theta)
            V = cov.covariance(_toy_model, theta, measure)
            V_star, _ = reg.apply(V)
            y = weight.whitening_residual(m, V_star, theta)
            p = pen.penalty(theta)
            return 0.5 * (jnp.sum(y * y) + p)

        theta0 = jnp.array([0.3, 0.7])
        g = jax.grad(half_obj_pen)(theta0)
        # The combined gradient must be finite at a generic theta.
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_matches_finite_differences(self):
        # The same composed objective, compared to a centred-difference
        # approximation. This is the property that protects against any
        # subtle dropped-gradient error in the residual construction.
        x_sample = jnp.array([[0.5, 1.0], [1.5, 2.0], [-0.5, 0.3]])

        def sampler(key, theta):
            del key, theta
            return x_sample

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=3,
            sampler=sampler,
        )
        cov = SyntheticCovariance()
        reg = DiagonalTikhonov()
        weight = ContinuouslyUpdated()
        pen = TikhonovPenalty(c=jnp.asarray(0.25))

        def half_obj_pen(theta_flat):
            theta = _TwoParam(a=theta_flat[0], b=theta_flat[1])
            m = measure.expectation(_toy_model, theta)
            V = cov.covariance(_toy_model, theta, measure)
            V_star, _ = reg.apply(V)
            y = weight.whitening_residual(m, V_star, theta)
            p = pen.penalty(theta)
            return 0.5 * (jnp.sum(y * y) + p)

        theta0 = jnp.array([0.3, 0.7])
        g_ad = jax.grad(half_obj_pen)(theta0)

        eps = 1e-5
        g_fd = jnp.zeros(2)
        for i in range(2):
            e = jnp.zeros(2).at[i].set(eps)
            f_plus = float(half_obj_pen(theta0 + e))
            f_minus = float(half_obj_pen(theta0 - e))
            g_fd = g_fd.at[i].set((f_plus - f_minus) / (2 * eps))

        assert jnp.allclose(g_ad, g_fd, atol=1e-5, rtol=1e-4)
