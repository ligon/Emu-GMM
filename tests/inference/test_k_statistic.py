"""Tests for emu_gmm.inference.k_statistic.

Three cases cover the algebraic identities and the asymptotic distribution:

(a) Just-identified case (M = p): G_tilde is square; col(G_tilde) = R^M;
    P_G = I; K = J; S = 0 exactly.

(b) Over-identified case with the analytical Euler example evaluated at
    (BETA_TRUE, GAMMA_TRUE): the analytical expectation is exactly zero
    at the truth, so m = 0 and consequently K = S = J = 0.

(c) Chi-squared distributional sanity: under the null, with V chosen to
    match the actual sampling distribution, K should be distributed
    chi^2_p. We construct an explicit Gaussian-moment toy where m | H_0
    is exactly N(0, V), and check that the empirical quantiles of K over
    a moderate Monte Carlo replication agree with the chi^2_p quantiles
    to within Monte Carlo error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
import scipy.stats
from emu_gmm.covariance import AnalyticalCovariance
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    N_ASSETS,
    EulerParams,
    euler_analytical_expectation,
    euler_residual,
)
from emu_gmm.inference import KStatisticResult, k_statistic
from emu_gmm.measures import AnalyticalMeasure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_cov_factory(M: int):
    def cov_fn(model, theta):
        del model, theta
        return jnp.eye(M)

    return AnalyticalCovariance(covariance_fn=cov_fn)


def _noop_model(x, theta):
    """StructuralModel placeholder for AnalyticalMeasure-only tests.

    AnalyticalMeasure.expectation_fn ignores the model argument, so a
    callable that returns zeros is sufficient to satisfy the type.
    """
    del x, theta
    return jnp.zeros((1,))


# ---------------------------------------------------------------------------
# (a) Just-identified: M = p -> S = 0, J = K.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _TwoParams:
    a: float
    b: float


def _two_moment_expectation_factory(m_at_theta):
    """Return an AnalyticalMeasure whose expectation is (a, b) - m_at_theta.

    Two parameters (a, b), two moments. The expectation is a linear
    function of theta with full-rank Jacobian (the identity), so the
    just-identified algebra is exact: P_G = I and K equals J.
    """

    def exp_fn(model, theta):
        del model
        return jnp.array([theta.a, theta.b]) - jnp.asarray(m_at_theta)

    return AnalyticalMeasure(expectation_fn=exp_fn)


class TestJustIdentified:
    """M = p: the K-S-J decomposition collapses to K = J, S = 0."""

    def _run(self, theta_0, m_at_theta):
        measure = _two_moment_expectation_factory(m_at_theta)
        cov = _identity_cov_factory(2)
        return k_statistic(theta_0, measure, cov, model=_noop_model)

    def test_returns_k_statistic_result(self):
        r = self._run(_TwoParams(a=1.0, b=2.0), m_at_theta=(0.0, 0.0))
        assert isinstance(r, KStatisticResult)

    def test_S_is_zero(self):
        """Exact identity: with G square and invertible, P_G = I -> S = 0."""
        r = self._run(_TwoParams(a=0.3, b=-0.7), m_at_theta=(0.1, 0.2))
        # Up to floating-point round-off only.
        assert float(r.S) == pytest.approx(0.0, abs=1e-12)

    def test_J_equals_K(self):
        r = self._run(_TwoParams(a=0.3, b=-0.7), m_at_theta=(0.1, 0.2))
        assert float(r.J) == pytest.approx(float(r.K), abs=1e-12)

    def test_df_K_equals_df_J(self):
        r = self._run(_TwoParams(a=0.0, b=0.0), m_at_theta=(0.0, 0.0))
        assert r.df_K == 2
        assert r.df_J == 2
        assert r.df_S == 0

    def test_p_S_is_nan(self):
        """df_S = 0 -> p_S is undefined; we surface NaN rather than 1.0."""
        r = self._run(_TwoParams(a=0.0, b=0.0), m_at_theta=(0.0, 0.0))
        assert jnp.isnan(jnp.asarray(r.p_S))


# ---------------------------------------------------------------------------
# (b) Over-identified, m = 0 at theta_0 -> all stats are zero.
# ---------------------------------------------------------------------------


class TestOverIdentifiedAtTruth:
    """Analytical Euler at (BETA_TRUE, GAMMA_TRUE): m = 0, so K = S = J = 0."""

    def _run(self):
        theta_0 = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
        measure = AnalyticalMeasure(expectation_fn=euler_analytical_expectation)
        cov = _identity_cov_factory(N_ASSETS)
        return k_statistic(theta_0, measure, cov, model=euler_residual)

    def test_J_is_zero(self):
        r = self._run()
        assert float(r.J) < 1e-12

    def test_K_is_zero(self):
        r = self._run()
        assert float(r.K) < 1e-12

    def test_S_is_zero(self):
        r = self._run()
        assert float(r.S) < 1e-12

    def test_df_K_is_p(self):
        """p = 2 structural parameters (beta, gamma)."""
        r = self._run()
        assert r.df_K == 2

    def test_df_J_is_M(self):
        """M = N_ASSETS = 3 moment conditions."""
        r = self._run()
        assert r.df_J == N_ASSETS

    def test_df_S_is_M_minus_p(self):
        r = self._run()
        assert r.df_S == N_ASSETS - 2

    def test_J_equals_K_plus_S(self):
        """Algebraic identity; holds exactly modulo floating-point."""
        r = self._run()
        assert float(r.J) == pytest.approx(float(r.K) + float(r.S), abs=1e-12)


# ---------------------------------------------------------------------------
# (c) Chi-squared distributional sanity.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _LinearScoreParams:
    a: float
    b: float


# The whitened-K/S kernel is exactly what the helper computes once V is
# the identity (so V^star = V and L = I): K = ||Q'm||^2, S = ||m||^2 - K
# with G = [[1,0],[0,1],[0,0]]. We exercise the helper directly *once*
# to fix the algebraic identity, then vmap a stripped-down kernel for
# the distributional sweep so we don't re-trace 600 closures.
_G_3x2 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


def _kernel_KS_from_m(m_realisation):
    """K and S given a single whitened moment draw.

    V = I_M, so the whitening step is the identity; reduce to a thin QR
    on the fixed G and project. Compiled once under jit and vmapped over
    the rep axis for speed.
    """
    Q, _R = jnp.linalg.qr(_G_3x2, mode="reduced")
    proj = Q.T @ m_realisation
    K = jnp.sum(proj * proj)
    J = jnp.sum(m_realisation * m_realisation)
    return K, J - K


_kernel_KS_batched = jax.jit(jax.vmap(_kernel_KS_from_m))


def _gaussian_moment_measure(m_realisation):
    """AnalyticalMeasure whose expectation at theta is ``theta - 0`` shifted
    by a fixed Monte Carlo realisation ``m_realisation``.

    Used in the single-rep sanity test that exercises the full
    :func:`k_statistic` path; the Monte Carlo sweep below bypasses the
    helper and uses the fast vmapped kernel to avoid re-tracing.
    """

    def exp_fn(model, theta):
        del model
        baseline = jnp.array([theta.a, theta.b, 0.0])
        return baseline + jnp.asarray(m_realisation)

    return AnalyticalMeasure(expectation_fn=exp_fn)


class TestChi2DistributionalSanity:
    """K(theta_0) under correctly-specified V is chi^2_p in distribution."""

    N_REPS = 2000
    M = 3
    P = 2

    def _draw_K_sample(self, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Vmapped Monte Carlo K- and S-statistic samples under the null.

        Draws ``N_REPS`` independent moment vectors ``m ~ N(0, I_M)`` and
        evaluates the kernel that the K-statistic helper applies when
        ``V = I`` and ``G`` is the fixed `_G_3x2` defined above. The
        resulting ``K`` samples are i.i.d. :math:`\\chi^2_{df_K}` and
        ``S`` samples are i.i.d. :math:`\\chi^2_{df_S}` under the null.
        """
        key = jax.random.PRNGKey(seed)
        ms = jax.random.normal(key, (self.N_REPS, self.M))
        Ks, Ss = _kernel_KS_batched(ms)
        return Ks, Ss

    def test_kernel_matches_helper(self):
        """Sanity-check: kernel agrees with the public k_statistic on
        one rep with V = I_M. Validates that the speed-optimised loop
        is computing the same quantity as the helper."""
        cov = _identity_cov_factory(self.M)
        theta_0 = _LinearScoreParams(a=0.0, b=0.0)
        m_real = jax.random.normal(jax.random.PRNGKey(99), (self.M,))
        measure = _gaussian_moment_measure(m_real)
        res = k_statistic(theta_0, measure, cov, model=_noop_model)
        K_kernel, S_kernel = _kernel_KS_from_m(m_real)
        assert float(res.K) == pytest.approx(float(K_kernel), abs=1e-12)
        assert float(res.S) == pytest.approx(float(S_kernel), abs=1e-12)

    def test_K_mean_matches_chi2_p(self):
        """E[chi^2_p] = p. With 2000 reps the SE on the mean is
        sqrt(2p/n) ~ 0.045."""
        Ks, _ = self._draw_K_sample(seed=0)
        mean_K = float(jnp.mean(Ks))
        # 4-sigma envelope (~0.18) catches systematic mis-scaling.
        assert mean_K == pytest.approx(self.P, abs=0.2)

    def test_S_mean_matches_chi2_M_minus_p(self):
        """df_S = M - p = 1; E[chi^2_1] = 1."""
        _, Ss = self._draw_K_sample(seed=1)
        mean_S = float(jnp.mean(Ss))
        # SE = sqrt(2/n) ~ 0.032; 4-sigma envelope ~0.13.
        assert mean_S == pytest.approx(self.M - self.P, abs=0.15)

    def test_K_quantile_matches_chi2(self):
        """75th percentile of chi^2_2 is ~2.77; check sample 75th is close."""
        Ks, _ = self._draw_K_sample(seed=2)
        q_emp = float(jnp.quantile(Ks, 0.75))
        q_theo = float(scipy.stats.chi2.ppf(0.75, self.P))
        # Quantile SE shrinks with sqrt(N); 0.3 is permissive.
        assert q_emp == pytest.approx(q_theo, abs=0.3)

    def test_J_equals_K_plus_S_each_rep(self):
        """J = K + S identically; verify on a small sample using the helper."""
        cov = _identity_cov_factory(self.M)
        theta_0 = _LinearScoreParams(a=0.0, b=0.0)
        key = jax.random.PRNGKey(3)
        for r in range(5):
            subkey = jax.random.fold_in(key, r)
            m_real = jax.random.normal(subkey, (self.M,))
            measure = _gaussian_moment_measure(m_real)
            res = k_statistic(theta_0, measure, cov, model=_noop_model)
            assert float(res.J) == pytest.approx(float(res.K) + float(res.S), abs=1e-10)


# ---------------------------------------------------------------------------
# (d) Accepts an EstimationResult as the first argument.
# ---------------------------------------------------------------------------


class TestAcceptsEstimationResult:
    """Verifies the convenience overload: first arg can be an EstimationResult."""

    def test_uses_theta_hat_when_passed_result(self):
        from emu_gmm.estimator import estimate
        from emu_gmm.optimizer import optimistix_lm
        from emu_gmm.regularization import DiagonalTikhonov
        from emu_gmm.weighting import ContinuouslyUpdated

        measure = AnalyticalMeasure(expectation_fn=euler_analytical_expectation)
        cov = _identity_cov_factory(N_ASSETS)
        result = estimate(
            model=euler_residual,
            measure=measure,
            covariance=cov,
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10, max_steps=200),
            theta_init=EulerParams(beta=0.9, gamma=1.5),
        )

        ks_at_hat = k_statistic(result, measure, cov, model=euler_residual)
        # Convergence to the truth -> m_hat ~ 0 -> J ~ 0 -> K, S ~ 0.
        assert float(ks_at_hat.J) < 1e-8
