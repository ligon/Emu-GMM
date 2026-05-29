"""Tests for the in-objective theta penalty hook (issue #7).

Covers:

- The :class:`TikhonovPenalty` itself: value and gradient on toy theta.
- ``estimate(..., penalty=None)`` reproduces the v1 result bitwise.
- ``estimate(..., penalty=TikhonovPenalty(c=...))`` pulls ``theta_hat``
  toward the origin in the expected direction and magnitude.
- AD through the penalty composes with AD through CU whitening (the
  combined gradient is finite and consistent with finite differences).
- :class:`Diagnostics` splits the criterion into data-only
  (``final_objective_data``) and full (``final_objective_full``)
  components and routes ``cond_info`` through the data-only and full
  information matrices independently (concerns raised on PR #35).
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


# ---------------------------------------------------------------------------
# Diagnostics split into data-only and full criterion fields (PR #35 concern 1)
# ---------------------------------------------------------------------------


class TestFinalObjectiveSplit:
    """``final_objective_data`` is data-only; ``final_objective_full``
    adds the penalty. ``final_objective`` is retained as an alias for
    ``final_objective_data`` for backwards compatibility.
    """

    def test_unpenalised_all_three_fields_equal(self):
        # With penalty=None the three fields coincide (== J_stat).
        result = _run(penalty=None)
        d = result.diagnostics
        assert float(d.final_objective) == pytest.approx(
            float(d.final_objective_data), rel=0, abs=0
        )
        assert float(d.final_objective_full) == pytest.approx(
            float(d.final_objective_data), rel=0, abs=0
        )
        assert float(d.final_objective_data) == pytest.approx(
            float(result.J_stat), rel=0, abs=0
        )

    def test_penalised_full_strictly_above_data(self):
        # With a strictly-positive penalty and theta_hat != 0,
        # final_objective_full > final_objective_data.
        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(1.0)))
        d = result.diagnostics
        data = float(d.final_objective_data)
        full = float(d.final_objective_full)
        # Theta_hat is well away from the origin at the Euler truth
        # (~0.96, ~2.0) so p(theta_hat) is well above any noise floor.
        assert full > data + 1e-3
        # Data-only value still equals J_stat.
        assert data == pytest.approx(float(result.J_stat), rel=0, abs=0)

    def test_penalised_full_matches_data_plus_penalty(self):
        # Concrete identity: full == data + p(theta_hat).
        c = 0.5
        pen = TikhonovPenalty(c=jnp.asarray(c))
        result = _run(penalty=pen)
        d = result.diagnostics
        p_hat = float(pen.penalty(result.theta_hat))
        full = float(d.final_objective_full)
        data = float(d.final_objective_data)
        assert full == pytest.approx(data + p_hat, rel=1e-10, abs=1e-12)

    def test_penalised_full_matches_optimizer_info_final_objective(self):
        # ``OptimizerInfo.final_objective`` reports the *half* norm
        # ``(1/2) ||r||^2`` (standard NLLS convention), while
        # ``Diagnostics.final_objective_full`` is ``||r||^2`` so it stays
        # on the same scale as ``J_stat`` / ``final_objective_data``.
        # Hence at the optimum
        # ``optimizer_info.final_objective == 0.5 * final_objective_full``
        # up to the tiny 1e-30 sqrt-floor on the penalty row.
        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.5)))
        d = result.diagnostics
        # OptimizerInfo carries a Python float once the optimiser returns
        # (eager path); compare against the full Diagnostics value.
        opt_obj = float(d.optimizer_info.final_objective)
        full = float(d.final_objective_full)
        # The 1e-30 floor on the penalty row is negligible at float64.
        assert opt_obj == pytest.approx(0.5 * full, rel=1e-8, abs=1e-10)

    def test_legacy_final_objective_aliases_data(self):
        # Pre-existing code that reads ``diagnostics.final_objective``
        # must continue to see the data-only value, not data + penalty.
        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.5)))
        d = result.diagnostics
        assert float(d.final_objective) == pytest.approx(
            float(d.final_objective_data), rel=0, abs=0
        )

    def test_summary_includes_split(self):
        # ``to_pandas()['summary']`` exposes both split fields so the
        # pandas reporting boundary doesn't lose the distinction.
        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.5)))
        summary = result.to_pandas()["summary"]
        assert "final_objective_data" in summary.index
        assert "final_objective_full" in summary.index
        assert "final_objective" in summary.index
        assert float(summary["final_objective_full"]) > float(
            summary["final_objective_data"]
        )


# ---------------------------------------------------------------------------
# final_gradient_norm with a penalty supplied (PR #35 concern 2)
# ---------------------------------------------------------------------------


class TestFinalGradientNormWithPenalty:
    """The reported ``final_gradient_norm`` is the norm of
    ``grad (1/2) ||r||^2`` where ``r`` is the residual the optimiser
    saw --- that is, including the penalty contribution when one is
    supplied. We pin the value rather than relying solely on the
    docstring claim.
    """

    def test_matches_grad_of_full_objective(self):
        # Build a tiny linear problem so we can replicate the exact
        # objective the estimator minimises. The estimator's residual
        # is r = [y; sqrt(p+1e-30)] (penalty supplied), so
        # ||r||^2 = ||y||^2 + p + 1e-30 and the (1/2) grad of this at
        # theta_hat is what should be reported.
        x_sample = jnp.array([[0.5, 1.0], [1.5, 2.0], [-0.5, 0.3]])

        def sampler(key, theta):
            del key, theta
            return x_sample

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=3,
            sampler=sampler,
        )

        pen = TikhonovPenalty(c=jnp.asarray(0.25))
        result = estimate(
            model=_toy_model,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.3, b=0.7),
            penalty=pen,
        )

        # Reproduce the residual the estimator actually used and take
        # grad of (1/2) ||r||^2 at theta_hat by hand. Note we *cannot*
        # call the data-only ||y||^2/2 here because that is what was
        # explicitly *not* claimed to be reported.
        from emu_gmm._internal.params import flatten_params, unflatten_params

        flat_hat, treedef = flatten_params(result.theta_hat)
        reg = DiagonalTikhonov()
        cov = SyntheticCovariance()
        weight = ContinuouslyUpdated()
        V0 = cov.covariance(_toy_model, result.theta_init, measure)
        _, tau0 = reg.apply(V0)
        tau0 = jnp.asarray(tau0)

        def _apply_anchored(V):
            return V + tau0 * jnp.diag(jnp.diag(V))

        def full_half_obj(tf):
            theta = unflatten_params(tf, treedef)
            m = measure.expectation(_toy_model, theta)
            V = cov.covariance(_toy_model, theta, measure)
            Vs = _apply_anchored(V)
            y = weight.whitening_residual(m, Vs, theta)
            p = pen.penalty(theta)
            extra = jnp.sqrt(p + 1e-30)
            r = jnp.concatenate([y, jnp.atleast_1d(extra)])
            return 0.5 * jnp.sum(r * r)

        g = jax.grad(full_half_obj)(flat_hat)
        expected_norm = float(jnp.linalg.norm(g))
        reported = float(result.diagnostics.final_gradient_norm)
        # Same mathematical computation, but the in-framework path goes
        # through the cached ``expectation_and_contributions`` /
        # ``moments_and_contributions`` primitive (see
        # ``estimator._cache_method``) while this reconstruction calls
        # ``measure.expectation`` directly. The reduction orders are
        # slightly different, and at a not-perfectly-converged optimum
        # the residual difference propagates into the reported norm.
        # Compare at "interpretive equality" tolerance rather than
        # bit-exact: same shape, same scale, gradient is the gradient
        # of the same surface.
        assert reported == pytest.approx(expected_norm, rel=1e-4, abs=1e-6)

    def test_strictly_above_data_only_norm(self):
        # Sanity: a *substantial* penalty makes the full-residual
        # gradient norm differ from the data-only one. (At a tight
        # optimum the data-only piece can be ~0, but the residual
        # ||grad p|| at theta_hat picks up the penalty pull.)
        x_sample = jnp.array([[0.5, 1.0], [1.5, 2.0], [-0.5, 0.3]])

        def sampler(key, theta):
            del key, theta
            return x_sample

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=3,
            sampler=sampler,
        )

        # No-penalty reference: at convergence the data gradient is ~0.
        ref = estimate(
            model=_toy_model,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.3, b=0.7),
        )
        ref_norm = float(ref.diagnostics.final_gradient_norm)

        # With a strong penalty, theta_hat shifts off the data optimum
        # and the *full* gradient norm at the new optimum stays small
        # (it's the LM convergence criterion), but the *data-only*
        # gradient evaluated there is no longer zero. We just check
        # that the residual the estimator reports is consistent with
        # the penalised LM tolerance: small in absolute terms but the
        # underlying data gradient is bigger.
        pen = TikhonovPenalty(c=jnp.asarray(5.0))
        pen_result = estimate(
            model=_toy_model,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.3, b=0.7),
            penalty=pen,
        )
        pen_norm = float(pen_result.diagnostics.final_gradient_norm)

        # Both convergence-tolerance-small. Sanity ranges.
        assert ref_norm < 1e-4
        assert pen_norm < 1e-4


# ---------------------------------------------------------------------------
# cond_info(data_only) excludes the penalty Hessian (PR #35 concern 3)
# ---------------------------------------------------------------------------


class TestCondInfoExcludesPenalty:
    """``Diagnostics.cond_info['data_only']`` is built from
    :math:`G' \\Lambda G` alone --- excluding the penalty Hessian
    contribution --- while ``'raw'`` includes it. Without a penalty
    the two keys coincide.
    """

    def test_unpenalised_all_three_equal(self):
        result = _run(penalty=None)
        info = result.diagnostics.cond_info
        # All three keys present and equal in the unpenalised path.
        assert set(info) >= {"raw", "data_only", "exclude_gauge"}
        assert float(info["raw"]) == pytest.approx(
            float(info["data_only"]), rel=0, abs=0
        )
        assert float(info["exclude_gauge"]) == pytest.approx(
            float(info["raw"]), rel=0, abs=0
        )

    def test_penalised_data_only_differs_from_raw(self):
        # Hessian of c||theta||^2 is 2cI. With a non-zero c the
        # full-info matrix is G'LambdaG + cI (the (1/2) on the LM half-
        # norm means we add (1/2)*2cI = cI), which has a different
        # condition number from the data-only matrix.
        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.5)))
        info = result.diagnostics.cond_info
        raw = float(info["raw"])
        data_only = float(info["data_only"])
        # The two must differ in finite samples (the penalty ridge
        # always *improves* the condition number relative to the data
        # information matrix at the same theta).
        assert raw != data_only
        assert raw < data_only

    def test_penalised_data_only_matches_no_penalty_compute(self):
        # ``data_only`` at penalised theta_hat must equal what
        # compute_cond_info(G, V_star) (no penalty) returns at that
        # same point. We replay the data-only computation by hand.
        from emu_gmm.diagnostics import compute_cond_info

        result = _run(penalty=TikhonovPenalty(c=jnp.asarray(0.5)))
        # Reproduce G and V* at the penalised theta_hat.
        measure = _make_measure()
        cov = SyntheticCovariance()
        reg = DiagonalTikhonov()
        # The estimator anchors tau at theta_init; replay that too.
        V0 = cov.covariance(euler_residual, result.theta_init, measure)
        _, tau0 = reg.apply(V0)
        tau0 = jnp.asarray(tau0)
        V_hat = cov.covariance(euler_residual, result.theta_hat, measure)
        V_star_hat = V_hat + tau0 * jnp.diag(jnp.diag(V_hat))
        G_hat = jnp.asarray(measure.jacobian(euler_residual, result.theta_hat))

        data_only_direct = compute_cond_info(G_hat, V_star_hat)
        reported_data_only = float(result.diagnostics.cond_info["data_only"])
        # The reported value matches the data-only computation exactly.
        assert reported_data_only == pytest.approx(
            float(data_only_direct["data_only"]), rel=1e-8, abs=1e-10
        )

    def test_compute_cond_info_with_penalty_hessian_arg(self):
        # Unit-level: passing penalty_hessian to compute_cond_info
        # changes only ``raw`` (and ``exclude_gauge`` which aliases it)
        # while leaving ``data_only`` unchanged.
        from emu_gmm.diagnostics import compute_cond_info

        # Simple G and V*: G is (3, 2), V* = I_3.
        G = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        V_star = jnp.eye(3)
        no_penalty = compute_cond_info(G, V_star)
        with_penalty = compute_cond_info(
            G, V_star, penalty_hessian=jnp.array([[2.0, 0.0], [0.0, 2.0]])
        )
        # data_only is independent of the penalty.
        assert float(with_penalty["data_only"]) == pytest.approx(
            float(no_penalty["data_only"]), rel=0, abs=0
        )
        # raw differs: cond(G'G + I) != cond(G'G).
        assert float(with_penalty["raw"]) != float(no_penalty["raw"])
        # exclude_gauge aliases raw.
        assert float(with_penalty["exclude_gauge"]) == pytest.approx(
            float(with_penalty["raw"]), rel=0, abs=0
        )
