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
import numpy as np
import pytest
import scipy.stats
from emu_gmm.covariance import AnalyticalCovariance, ClusteredCovariance, IIDCovariance
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
from emu_gmm.measures import AnalyticalMeasure, EmpiricalMeasure, SyntheticMeasure

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
        # AnalyticalMeasure has no per-observation contributions; the
        # strong-ID limit must be opted into explicitly since #41.
        return k_statistic(
            theta_0, measure, cov, model=_noop_model, strong_id_fallback=True
        )

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
        # AnalyticalMeasure -> explicit strong-ID opt-in (#41).
        return k_statistic(
            theta_0, measure, cov, model=euler_residual, strong_id_fallback=True
        )

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
        res = k_statistic(
            theta_0, measure, cov, model=_noop_model, strong_id_fallback=True
        )
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
            res = k_statistic(
                theta_0, measure, cov, model=_noop_model, strong_id_fallback=True
            )
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

        ks_at_hat = k_statistic(
            result, measure, cov, model=euler_residual, strong_id_fallback=True
        )
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


# ---------------------------------------------------------------------------
# (h) Issue #52 regression: K-stat must be chi^2-calibrated under clustered
# dependence when the empirical Sigma_jm estimator is used.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _OneParam:
    theta: float


def _linear_psi(x, params):
    """Three moments of a linear-in-x regression residual.

    ``x = (y, x_val, 1.0)``; the residual is ``r = y - x_val * theta``; the
    moment vector is ``(r * x_val, r * x_val^2, r)``. Over-identified
    (M=3, p=1) so the K/S decomposition has a non-trivial chi^2_1 K-stat
    and chi^2_2 S-stat at the true ``theta``.
    """
    y, xval, _one = x[0], x[1], x[2]
    resid = y - xval * params.theta
    return jnp.array([resid * xval, resid * xval * xval, resid])


def _simulate_clustered_dataset(
    seed: int,
    n_clusters: int,
    cluster_size: int,
    rho: float,
    theta_true: float = 1.0,
):
    """Generate a clustered linear regression dataset.

    Returns ``(measure, cov, theta_true)`` ready for :func:`k_statistic`.
    The residual ``e_i = sqrt(rho) u_{c(i)} + sqrt(1 - rho) eps_i`` carries
    intracluster correlation ``rho``; ``rho = 0`` reduces to IID.
    """
    rng = np.random.default_rng(seed)
    N = n_clusters * cluster_size
    u_cluster = rng.standard_normal(n_clusters)
    u_within = rng.standard_normal(N)
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_size).astype(np.float64)
    e = np.sqrt(rho) * u_cluster[cluster_ids.astype(int)] + np.sqrt(1 - rho) * u_within
    x = rng.standard_normal(N)
    y = x * theta_true + e
    x_data = jnp.asarray(np.column_stack([y, x, np.ones(N)]))
    mask = jnp.ones((N, 3))
    weights = jnp.ones(N)
    measure = EmpiricalMeasure(x=x_data, mask=mask, weights=weights)
    cov = ClusteredCovariance(
        cluster_ids=jnp.asarray(cluster_ids), n_clusters=n_clusters
    )
    return measure, cov, theta_true


class TestClusteredKStatNullCalibration:
    """Regression test for issue #52.

    Under H0 (theta = theta_true) with intracluster-correlated data and a
    correctly specified :class:`ClusteredCovariance`, the K-statistic must
    be chi^2_p in distribution, i.e. ``p_K`` must be ~Uniform(0, 1).

    The pre-fix ``_sigma_jm_from_contributions`` used a ``1/N``-scaled
    cross-covariance that did not match the ``1/(N_m N_k)`` scale of the
    cluster-totals :math:`V`. The result was a systematically conservative
    K-statistic (too few rejections / too high mean ``p_K``). The wf7 MC
    at n_clusters=50 measured mean p_K=0.315 with KS p<1e-4 vs Uniform.

    This test runs a moderate MC sweep (200 reps; ~60 s on CPU). Under
    Uniform(0, 1) the SE on the mean of N=200 i.i.d. samples is
    ``sqrt(1/(12 * 200)) ~= 0.020``. Pre-fix the mean sits ~0.42 (~4 SE
    below 0.5); post-fix the mean sits within +/- 2 SE of 0.5. The
    KS-against-uniform check is the complementary distributional test;
    pre-fix KS p-value is consistently below 0.005, post-fix it is
    comfortably above the ``0.01`` threshold the test uses.
    """

    N_MC = 200
    N_CLUSTERS = 50
    CLUSTER_SIZE = 10
    RHO = 0.6

    def _draw_p_K(self) -> np.ndarray:
        p_K = np.empty(self.N_MC)
        for i in range(self.N_MC):
            measure, cov, theta_true = _simulate_clustered_dataset(
                seed=1000 + i,
                n_clusters=self.N_CLUSTERS,
                cluster_size=self.CLUSTER_SIZE,
                rho=self.RHO,
            )
            res = k_statistic(
                _OneParam(theta=theta_true), measure, cov, model=_linear_psi
            )
            p_K[i] = float(res.p_K)
        return p_K

    def test_p_K_mean_in_uniform_window(self):
        """E[U(0, 1)] = 0.5; with N_MC=200 the SE on the mean is ~0.020.

        Pre-fix this test produces mean p_K ~ 0.42 (~4 SE below 0.5);
        post-fix it falls in [0.45, 0.55] reliably.
        """
        p_K = self._draw_p_K()
        assert np.mean(p_K) == pytest.approx(0.5, abs=0.05)

    def test_p_K_uniform_via_KS(self):
        """KS-test of p_K against Uniform(0, 1).

        Under H0 the KS p-value is itself Uniform(0, 1). The pre-fix code
        produces KS p-values < 0.005 (the p_K distribution is shifted
        toward zero); post-fix the KS p-value is comfortably above the
        ``0.01`` threshold. The threshold is intentionally conservative
        so that nominal Type-I error on the regression test stays well
        below 1%.
        """
        p_K = self._draw_p_K()
        _ks_stat, ks_p = scipy.stats.kstest(p_K, "uniform")
        assert ks_p > 0.01


class TestIIDKStatNullCalibration:
    """The same null-calibration check for the IID path.

    The cluster-totals form collapses to the pairwise-overlap IID form
    when every cluster is a singleton, so :class:`IIDCovariance` users
    benefit from the same scale fix. Pre-fix the IID path was also
    miscalibrated (mean p_K ~ 0.39 in 100-rep MC, ~5 SE below 0.5);
    post-fix it tracks Uniform(0, 1) within ordinary MC noise.
    """

    N_MC = 100
    N_OBS = 300

    def _simulate(self, seed: int):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal(self.N_OBS)
        x = rng.standard_normal(self.N_OBS)
        theta_true = 1.0
        y = x * theta_true + e
        x_data = jnp.asarray(np.column_stack([y, x, np.ones(self.N_OBS)]))
        mask = jnp.ones((self.N_OBS, 3))
        weights = jnp.ones(self.N_OBS)
        measure = EmpiricalMeasure(x=x_data, mask=mask, weights=weights)
        cov = IIDCovariance()
        return measure, cov, theta_true

    def _draw_p_K(self) -> np.ndarray:
        p_K = np.empty(self.N_MC)
        for i in range(self.N_MC):
            measure, cov, theta_true = self._simulate(seed=2000 + i)
            res = k_statistic(
                _OneParam(theta=theta_true), measure, cov, model=_linear_psi
            )
            p_K[i] = float(res.p_K)
        return p_K

    def test_p_K_mean_in_uniform_window(self):
        p_K = self._draw_p_K()
        assert np.mean(p_K) == pytest.approx(0.5, abs=0.05)

    def test_p_K_uniform_via_KS(self):
        """KS-test of p_K against Uniform(0, 1) on the IID path.

        Pre-fix KS p-value < 0.001; post-fix KS p-value > 0.01.
        """
        p_K = self._draw_p_K()
        _ks_stat, ks_p = scipy.stats.kstest(p_K, "uniform")
        assert ks_p > 0.01


class TestSingletonClustersMatchIID:
    """Cluster-totals with one observation per cluster reduces to IID.

    Algebraic identity: with ``cluster_ids = [0, 1, ..., N-1]`` and
    ``n_clusters = N``, the cluster-totals cross-covariance equals the
    pairwise-overlap IID form. Both routes must produce numerically
    identical K, S, J statistics in this limit. This is the canonical
    sanity check that the new ClusteredCovariance dispatch does not
    perturb the IID numerics it was designed to generalise.
    """

    def _build_dataset(self):
        rng = np.random.default_rng(42)
        N = 200
        e = rng.standard_normal(N)
        x = rng.standard_normal(N)
        theta_true = 1.0
        y = x * theta_true + e
        x_data = jnp.asarray(np.column_stack([y, x, np.ones(N)]))
        mask = jnp.ones((N, 3))
        weights = jnp.ones(N)
        measure = EmpiricalMeasure(x=x_data, mask=mask, weights=weights)
        return measure, theta_true, N

    def test_singleton_clusters_match_iid(self):
        measure, theta_true, N = self._build_dataset()
        iid_cov = IIDCovariance()
        # Every observation is its own cluster.
        cluster_ids = jnp.arange(N, dtype=jnp.float64)
        clu_cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=N)
        theta_0 = _OneParam(theta=theta_true)
        # Evaluate at a deliberately offset theta_0 so m != 0 and the
        # Sigma_jm correction is non-trivial; tests are vacuous when m = 0.
        theta_offset = _OneParam(theta=theta_true + 0.2)
        for theta in (theta_0, theta_offset):
            res_iid = k_statistic(theta, measure, iid_cov, model=_linear_psi)
            res_clu = k_statistic(theta, measure, clu_cov, model=_linear_psi)
            assert float(res_clu.K) == pytest.approx(
                float(res_iid.K), rel=1e-9, abs=1e-12
            )
            assert float(res_clu.S) == pytest.approx(
                float(res_iid.S), rel=1e-9, abs=1e-12
            )
            assert float(res_clu.J) == pytest.approx(
                float(res_iid.J), rel=1e-9, abs=1e-12
            )


# ---------------------------------------------------------------------------
# (i) Issue #41 hardening: the strong-ID fallback is loud, score_cov_fn is
# validated and takes precedence, and the design sandwiches are refused
# until a matched cross-covariance form exists.
# ---------------------------------------------------------------------------


class TestStrongIdFallbackIsLoud:
    """No contributions + no score_cov_fn must raise, not silently degrade.

    Before #41 a caller wiring an AnalyticalMeasure through k_statistic
    silently got the raw-G (non-robust) statistic — the exact footgun
    the Seasonality consumer flagged. The fallback is now an explicit
    opt-in.
    """

    def test_raises_without_opt_in(self):
        measure = _two_moment_expectation_factory((0.1, 0.2))
        cov = _identity_cov_factory(2)
        with pytest.raises(ValueError, match="strong_id_fallback"):
            k_statistic(_TwoParams(a=0.3, b=-0.7), measure, cov, model=_noop_model)

    def test_opt_in_equals_zero_score_cov(self):
        """The fallback is exactly the Sigma_jm = 0 statistic."""
        measure = _two_moment_expectation_factory((0.1, 0.2))
        cov = _identity_cov_factory(2)
        theta_0 = _TwoParams(a=0.3, b=-0.7)
        via_flag = k_statistic(
            theta_0, measure, cov, model=_noop_model, strong_id_fallback=True
        )
        via_zeros = k_statistic(
            theta_0,
            measure,
            cov,
            model=_noop_model,
            score_cov_fn=lambda model, theta: jnp.zeros((2, 2, 2)),
        )
        assert float(via_flag.K) == pytest.approx(float(via_zeros.K), abs=0.0)
        assert float(via_flag.S) == pytest.approx(float(via_zeros.S), abs=0.0)


class TestScoreCovFnPath:
    """The level-1 dispatch: user-supplied Sigma_{G,m} (conformance-review gap)."""

    def test_wrong_shape_raises(self):
        measure = _two_moment_expectation_factory((0.1, 0.2))
        cov = _identity_cov_factory(2)
        with pytest.raises(ValueError, match="score_cov_fn returned shape"):
            k_statistic(
                _TwoParams(a=0.3, b=-0.7),
                measure,
                cov,
                model=_noop_model,
                # (p, M) instead of (p, M, M).
                score_cov_fn=lambda model, theta: jnp.zeros((2, 2)),
            )

    def test_takes_precedence_over_contributions(self):
        """With contributions available, an explicit score_cov_fn wins.

        At an offset theta_0 (m != 0) the Sigma_jm correction is
        non-trivial, so the zero-Sigma statistic must differ from the
        contributions-route statistic; and supplying the contributions
        route's own Sigma_jm through score_cov_fn must reproduce the
        default bitwise.
        """
        from emu_gmm.inference.k_statistic import (
            _N_j_from_measure,
            _sigma_jm_iid_from_contributions,
            _to_plain,
        )

        measure, _clu_cov, theta_true = _simulate_clustered_dataset(
            seed=7, n_clusters=40, cluster_size=5, rho=0.0
        )
        cov = IIDCovariance()
        theta_offset = _OneParam(theta=theta_true + 0.3)

        res_default = k_statistic(theta_offset, measure, cov, model=_linear_psi)
        res_zero = k_statistic(
            theta_offset,
            measure,
            cov,
            model=_linear_psi,
            score_cov_fn=lambda model, theta: jnp.zeros((1, 3, 3)),
        )
        assert float(res_zero.K) != pytest.approx(float(res_default.K), rel=1e-6)

        def exact_sigma(model, theta):
            g = _to_plain(measure.moment_contributions(model, theta))
            D = _to_plain(measure.jacobian_contributions(model, theta))
            return _sigma_jm_iid_from_contributions(g, D, _N_j_from_measure(measure))

        res_exact = k_statistic(
            theta_offset, measure, cov, model=_linear_psi, score_cov_fn=exact_sigma
        )
        assert float(res_exact.K) == pytest.approx(float(res_default.K), abs=0.0)
        assert float(res_exact.S) == pytest.approx(float(res_default.S), abs=0.0)


class TestDesignSandwichRefused:
    """StratifiedCovariance / DesignAwareCovariance have no matched
    Sigma_{G,m} form yet; silently using the IID form mis-scales the
    orthogonalisation (#41, surfaced by the Seasonality consumer). The
    dispatch must refuse with actionable guidance.
    """

    def _measure(self):
        measure, _cov, theta_true = _simulate_clustered_dataset(
            seed=11, n_clusters=20, cluster_size=4, rho=0.0
        )
        return measure, _OneParam(theta=theta_true)

    def test_stratified_refused(self):
        from emu_gmm.covariance import StratifiedCovariance

        measure, theta_0 = self._measure()
        n = int(measure.x.shape[0])
        design = StratifiedCovariance(
            psu_ids=jnp.zeros(n),
            cell_ids=jnp.zeros(n),
            stratum_ids=jnp.zeros(n),
            n_psu=1,
            n_cells=1,
            n_strata=1,
        )
        with pytest.raises(ValueError, match="score_cov_fn"):
            k_statistic(theta_0, measure, design, model=_linear_psi, V=jnp.eye(3))

    def test_design_aware_refused(self):
        from emu_gmm.covariance import DesignAwareCovariance, StratifiedCovariance

        measure, theta_0 = self._measure()
        n = int(measure.x.shape[0])
        design = StratifiedCovariance(
            psu_ids=jnp.zeros(n),
            cell_ids=jnp.zeros(n),
            stratum_ids=jnp.zeros(n),
            n_psu=1,
            n_cells=1,
            n_strata=1,
        )
        sampling = ClusteredCovariance(cluster_ids=jnp.zeros(n), n_clusters=1)
        mixed = DesignAwareCovariance.from_design_mask(
            design=design,
            sampling=sampling,
            design_moment_mask=jnp.ones(3),
        )
        with pytest.raises(ValueError, match="score_cov_fn"):
            k_statistic(theta_0, measure, mixed, model=_linear_psi, V=jnp.eye(3))
