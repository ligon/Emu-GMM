"""Tests for emu_gmm.inference.k_statistic.

Covers the algebraic identities, the asymptotic distribution, the
Kleibergen D-tilde orthogonalisation property (reparameterisation
invariance of K under linear transformations of m), and the
jit / vmap compatibility of the helper.

(a) Just-identified case (M = p): D_tilde_w is square; col = R^M;
    P_D = I; K = J; S = 0 exactly.

(b) Over-identified case with the analytical Euler example evaluated at
    (BETA_TRUE, GAMMA_TRUE): the analytical expectation is exactly zero
    at the truth, so m = 0 and consequently K = S = J = 0.

(c) Chi-squared distributional sanity: under the null, with V chosen to
    match the actual sampling distribution, K should be distributed
    chi^2_p. The vmapped kernel approximates the asymptotic limit law to
    within Monte Carlo error.

(d) Convenience overload: passing an EstimationResult evaluates at theta_hat.

(e) Kleibergen D-tilde correctness: the K stat is invariant under
    invertible linear reparameterisations of the moment vector
    m -> A m. This is the central property the D-tilde construction
    delivers (the raw-G form is invariant only when A is orthogonal in
    the V metric).

(f) jit / vmap compatibility: k_statistic compiles and vmaps cleanly,
    matching the contract of every other public helper in the framework.
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
    euler_sampler_factory,
)
from emu_gmm.inference import KStatisticResult, k_statistic
from emu_gmm.measures import AnalyticalMeasure, SyntheticMeasure

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


# ---------------------------------------------------------------------------
# (e) Kleibergen D-tilde correctness: K is invariant under invertible linear
# reparameterisations of the moment vector.
# ---------------------------------------------------------------------------


class TestDTildeReparameterisationInvariance:
    """The Kleibergen K-stat with D-tilde is invariant under m -> A m.

    Concretely: replace the moment vector by ``A @ m`` (an invertible
    linear reparameterisation), and the corresponding variance becomes
    ``A V A'``, the Jacobian ``A G``, and the score covariance tensor
    ``Sigma_jm_new[j] = A Sigma_jm[j] A'``. Plugged into

       D_tilde_j = A G_j - A Sigma_jm[j] A' (A V A')^{-1} A m
                = A (G_j - Sigma_jm[j] V^{-1} m)
                = A D_tilde_j_original

    the whitened residual ``L_new^{-1} (A m) = U L^{-1} m`` for some
    orthogonal U (any factor of ``A V A' = (A L)(A L)'`` differs from
    ``A L`` by orthogonal rotation), so K, J, S are all invariant by
    construction.

    The raw-G form would only deliver this invariance when ``A`` is
    orthogonal in the V metric; under a generic ``A`` the raw-G K
    statistic shifts. So this test directly distinguishes the D-tilde
    form from the old raw-G form.
    """

    N_SIM = 5000
    P = 2
    M = 3

    def _build(self, seed: int):
        sampler = euler_sampler_factory(self.N_SIM)
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(seed), n_sim=self.N_SIM, sampler=sampler
        )
        # Identity covariance so the test is about D-tilde, not weighting.
        cov = _identity_cov_factory(N_ASSETS)
        return measure, cov

    def _reparam_setup(self, base_measure, A):
        """Wrap the synthetic measure under m -> A m.

        Achieved by composing the user residual with a (constant) linear
        map: ``A @ psi(x, theta)``. The synthetic measure path computes
        moment / Jacobian contributions automatically by chain rule.
        """
        A_arr = jnp.asarray(A)

        def reparam_residual(x, theta):
            psi = euler_residual(x, theta)
            return A_arr @ psi

        # Per-moment cov also rotates: A I A' = A A'.
        def cov_fn(model, theta):
            del model, theta
            return A_arr @ A_arr.T

        return reparam_residual, AnalyticalCovariance(covariance_fn=cov_fn)

    def test_K_invariant_under_reparameterisation(self):
        """A misspecified theta_0 -> non-zero m; K should agree across A."""
        measure, cov = self._build(seed=42)
        # Deliberately mis-specified theta_0 so m != 0 and the D-tilde
        # correction is non-trivial. (At m = 0 the test is vacuous.)
        theta_0 = EulerParams(beta=0.85, gamma=3.0)
        # Baseline K under the identity reparameterisation.
        res_id = k_statistic(theta_0, measure, cov, model=euler_residual)
        # A non-orthogonal A: a shear + scale that mixes moments.
        A = jnp.array([[1.0, 0.3, -0.2], [0.0, 1.4, 0.5], [0.7, 0.0, 1.1]])
        reparam_residual, cov_A = self._reparam_setup(measure, A)
        res_A = k_statistic(theta_0, measure, cov_A, model=reparam_residual)
        # K and J are coordinate-free under D-tilde + correct V rotation.
        assert float(res_A.K) == pytest.approx(float(res_id.K), rel=1e-6, abs=1e-8)
        assert float(res_A.J) == pytest.approx(float(res_id.J), rel=1e-6, abs=1e-8)
        assert float(res_A.S) == pytest.approx(float(res_id.S), rel=1e-6, abs=1e-8)

    def test_K_plus_S_equals_J(self):
        """Algebraic identity must continue to hold under D-tilde."""
        measure, cov = self._build(seed=7)
        theta_0 = EulerParams(beta=0.85, gamma=3.0)
        res = k_statistic(theta_0, measure, cov, model=euler_residual)
        assert float(res.K) + float(res.S) == pytest.approx(float(res.J), abs=1e-10)


# ---------------------------------------------------------------------------
# (f) jit / vmap compatibility.
# ---------------------------------------------------------------------------


class TestJitVmapCompatibility:
    """k_statistic compiles under jit and vmaps over batches of theta_0.

    The contract is the same as estimate() and J-stat: all array fields
    are 0-d JAX arrays, the dofs are static, and the helper threads
    through the standard tracing boundaries.
    """

    M = 3
    P = 2

    def _setup(self):
        sampler = euler_sampler_factory(500)
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=500, sampler=sampler
        )
        cov = _identity_cov_factory(N_ASSETS)
        return measure, cov

    def test_jit_compiles_and_matches_eager(self):
        measure, cov = self._setup()
        theta_0 = EulerParams(beta=0.92, gamma=2.5)
        eager = k_statistic(theta_0, measure, cov, model=euler_residual)

        def kernel(theta):
            return k_statistic(theta, measure, cov, model=euler_residual)

        # Force a recompile via jit by jit-wrapping the result-producing
        # core (k_statistic returns a pytree dataclass, which jit handles).
        jitted = jax.jit(kernel)
        traced = jitted(theta_0)
        # K, S, J should agree between eager and jitted execution.
        assert float(traced.K) == pytest.approx(float(eager.K), rel=1e-9, abs=1e-12)
        assert float(traced.S) == pytest.approx(float(eager.S), rel=1e-9, abs=1e-12)
        assert float(traced.J) == pytest.approx(float(eager.J), rel=1e-9, abs=1e-12)
        # p-values are now JAX arrays, not Python floats; they should
        # also agree.
        assert float(traced.p_K) == pytest.approx(float(eager.p_K), abs=1e-10)
        assert float(traced.p_J) == pytest.approx(float(eager.p_J), abs=1e-10)

    def test_vmap_over_theta_batch(self):
        """vmap over a batch of theta_0; result fields stack on the leading axis."""
        measure, cov = self._setup()

        # Stack two thetas on the leaves: beta=[0.95, 0.90], gamma=[2.0, 2.5].
        betas = jnp.asarray([0.95, 0.90])
        gammas = jnp.asarray([2.0, 2.5])

        def kernel(beta, gamma):
            theta = EulerParams(beta=beta, gamma=gamma)
            return k_statistic(theta, measure, cov, model=euler_residual)

        batched = jax.vmap(kernel)(betas, gammas)
        assert batched.K.shape == (2,)
        assert batched.S.shape == (2,)
        assert batched.J.shape == (2,)
        # dofs are static — same scalar for every entry of the batch.
        assert batched.df_K == self.P
        assert batched.df_J == self.M

    def test_p_values_are_jax_arrays_not_python_floats(self):
        """The rewrite returns p_K/p_S/p_J as 0-d JAX arrays so vmap works."""
        measure, cov = self._setup()
        theta_0 = EulerParams(beta=0.95, gamma=2.0)
        res = k_statistic(theta_0, measure, cov, model=euler_residual)
        # Each p-value is a 0-d JAX array, not a Python float — required
        # for vmap and jit transparency.
        assert hasattr(res.p_K, "shape")
        assert res.p_K.shape == ()
        assert hasattr(res.p_J, "shape")
        assert res.p_J.shape == ()


# ---------------------------------------------------------------------------
# (g) Under-identified problems are rejected with a clear error.
# ---------------------------------------------------------------------------


class TestUnderIdentifiedRaises:
    """M < p should raise ValueError, not silently return a degenerate result."""

    def test_under_identified_raises(self):
        @jdc.pytree_dataclass
        class _ThreeParams:
            a: float
            b: float
            c: float

        def exp_fn(model, theta):
            del model
            # 2 moments, 3 parameters -> M < p.
            return jnp.array([theta.a + theta.b, theta.b + theta.c])

        measure = AnalyticalMeasure(expectation_fn=exp_fn)
        cov = _identity_cov_factory(2)
        with pytest.raises(ValueError, match="under-identified"):
            k_statistic(
                _ThreeParams(a=0.0, b=0.0, c=0.0),
                measure,
                cov,
                model=_noop_model,
            )
