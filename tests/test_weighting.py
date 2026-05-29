"""Tests for emu_gmm.weighting."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import types as t
from emu_gmm._internal.cholesky import cholesky
from emu_gmm.weighting import (
    CUE,
    ContinuouslyUpdated,
    Fixed,
    Identity,
    IteratedWeighting,
)


@jdc.pytree_dataclass
class _Scalar:
    theta: float


def _make_pd(seed: int = 0, dim: int = 3) -> jnp.ndarray:
    """Deterministic PD matrix of size ``dim`` x ``dim``."""
    rng = np.random.default_rng(seed=seed)
    A = rng.standard_normal((dim, dim)).astype(np.float64)
    return jnp.asarray(A @ A.T + np.eye(dim))


# ---------------------------------------------------------------------------


class TestIdentity:
    def test_satisfies_protocol(self):
        assert isinstance(Identity(), t.WeightingStrategy)

    def test_returns_m_unchanged(self):
        m = jnp.array([0.5, -0.2, 0.7])
        V = _make_pd()  # ignored
        theta = _Scalar(theta=1.5)
        y = Identity().whitening_residual(m, V, theta)
        assert jnp.allclose(y, m)

    def test_ignores_V(self):
        """Two different V matrices give the same output for the same m."""
        m = jnp.array([1.0, 2.0, 3.0])
        V_a = _make_pd(seed=0)
        V_b = _make_pd(seed=1)
        theta = _Scalar(theta=0.0)
        s = Identity()
        y_a = s.whitening_residual(m, V_a, theta)
        y_b = s.whitening_residual(m, V_b, theta)
        assert jnp.allclose(y_a, y_b)
        assert jnp.allclose(y_a, m)

    def test_is_pytree_with_no_leaves(self):
        leaves, _ = jax.tree_util.tree_flatten(Identity())
        assert leaves == []

    def test_jits(self):
        m = jnp.array([0.5, -0.2, 0.7])
        V = _make_pd()
        theta = _Scalar(theta=1.5)
        strategy = Identity()

        @jax.jit
        def compute(s, mm, vv, tt):
            return s.whitening_residual(mm, vv, tt)

        y_eager = strategy.whitening_residual(m, V, theta)
        y_jit = compute(strategy, m, V, theta)
        assert jnp.allclose(y_eager, y_jit)


# ---------------------------------------------------------------------------


class TestFixed:
    def test_satisfies_protocol(self):
        V0 = _make_pd()
        assert isinstance(Fixed.from_V0(V0), t.WeightingStrategy)

    def test_from_V0_stores_cholesky(self):
        V0 = _make_pd()
        strategy = Fixed.from_V0(V0)
        # Lower-triangular Cholesky factor stored on the dataclass.
        assert jnp.allclose(strategy.L0 @ strategy.L0.T, V0, atol=1e-6)

    def test_direct_construction_with_L0(self):
        V0 = _make_pd()
        L0 = cholesky(V0)
        strategy = Fixed(L0=L0)
        assert jnp.allclose(strategy.L0, L0)

    def test_whitens_to_identity_covariance(self):
        """If x ~ N(0, V0), then L0^{-1} x has identity covariance.

        Test with N = 10_000 samples and ``jnp.cov`` of the whitened
        batch. The empirical covariance should be close to the identity.
        """
        dim = 3
        V0 = _make_pd(seed=2, dim=dim)
        L0 = cholesky(V0)
        # Draw N samples from N(0, V0) by drawing standard normals and
        # applying the Cholesky factor.
        N = 10_000
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, shape=(N, dim))  # (N, dim)
        x = z @ L0.T  # x has covariance V0
        strategy = Fixed(L0=L0)
        theta = _Scalar(theta=0.0)
        # Apply whitening row by row via vmap.
        y_batch = jax.vmap(
            lambda mv: strategy.whitening_residual(mv, jnp.eye(dim), theta)
        )(x)
        # Empirical covariance: rows are samples.
        cov_y = jnp.cov(y_batch.T)
        assert jnp.allclose(cov_y, jnp.eye(dim), atol=0.1)

    def test_V_argument_ignored(self):
        """The V argument to whitening_residual is ignored."""
        V0 = _make_pd(seed=0)
        L0 = cholesky(V0)
        strategy = Fixed(L0=L0)
        m = jnp.array([1.0, 2.0, 3.0])
        theta = _Scalar(theta=0.0)
        y_eye = strategy.whitening_residual(m, jnp.eye(3), theta)
        # Pass a wildly different V; result should be identical.
        V_other = _make_pd(seed=42)
        y_other = strategy.whitening_residual(m, V_other, theta)
        assert jnp.allclose(y_eye, y_other)

    def test_jits(self):
        V0 = _make_pd()
        strategy = Fixed.from_V0(V0)
        m = jnp.array([0.5, -0.2, 0.7])
        theta = _Scalar(theta=0.0)

        @jax.jit
        def compute(s, mm, tt):
            return s.whitening_residual(mm, V0, tt)

        y_eager = strategy.whitening_residual(m, V0, theta)
        y_jit = compute(strategy, m, theta)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)

    # --- Safe-construction guards (Fix 1, v1.x API safety) -----------------

    def test_kwarg_L0_constructs(self):
        """``Fixed(L0=L0)`` (back-compat keyword path) works."""
        L0 = cholesky(_make_pd(seed=4))
        strategy = Fixed(L0=L0)
        assert jnp.allclose(strategy.L0, L0)

    def test_kwarg_V0_constructs(self):
        """``Fixed(V0=V0)`` works and matches ``Fixed.from_V0(V0)``."""
        V0 = _make_pd(seed=5)
        s_kwarg = Fixed(V0=V0)
        s_factory = Fixed.from_V0(V0)
        assert jnp.allclose(s_kwarg.L0, s_factory.L0)
        # And that L0 satisfies L0 L0^T = V0.
        assert jnp.allclose(s_kwarg.L0 @ s_kwarg.L0.T, V0, atol=1e-6)

    def test_from_L0_factory(self):
        """``Fixed.from_L0(L0)`` is equivalent to ``Fixed(L0=L0)``."""
        L0 = cholesky(_make_pd(seed=6))
        s_factory = Fixed.from_L0(L0)
        s_kwarg = Fixed(L0=L0)
        assert jnp.allclose(s_factory.L0, s_kwarg.L0)

    def test_positional_arg_errors_with_clear_message(self):
        """The legacy ``Fixed(L0)`` positional pattern now raises a
        ``TypeError`` mentioning the W-vs-L0 hazard and the safe
        constructors.

        Rationale: a ManifoldGMM user porting ``Fixed(W_hat)`` by analogy
        would otherwise silently store ``W`` as the Cholesky factor and
        get wrong results. The error message must explicitly name both
        ``L0=`` and ``V0=`` and the porting hint.
        """
        L0 = cholesky(_make_pd(seed=7))
        with pytest.raises(TypeError) as exc:
            Fixed(L0)
        msg = str(exc.value)
        assert "positional" in msg
        assert "L0=" in msg
        assert "V0=" in msg
        # Porting hint should reference W (the ManifoldGMM analogue).
        assert "W" in msg

    def test_no_kwargs_errors(self):
        """``Fixed()`` with neither L0 nor V0 raises ``TypeError``."""
        with pytest.raises(TypeError) as exc:
            Fixed()
        msg = str(exc.value)
        assert "L0" in msg and "V0" in msg

    def test_both_kwargs_errors(self):
        """``Fixed(L0=L0, V0=V0)`` raises ``TypeError`` (exclusive)."""
        V0 = _make_pd(seed=8)
        L0 = cholesky(V0)
        with pytest.raises(TypeError) as exc:
            Fixed(L0=L0, V0=V0)
        msg = str(exc.value)
        assert "L0" in msg and "V0" in msg

    def test_from_V0_and_V0_kwarg_match(self):
        """``Fixed.from_V0(V0)`` and ``Fixed(V0=V0)`` produce identical
        internal state (the L0 field).
        """
        V0 = _make_pd(seed=9)
        s_factory = Fixed.from_V0(V0)
        s_kwarg = Fixed(V0=V0)
        assert jnp.allclose(s_factory.L0, s_kwarg.L0)


# ---------------------------------------------------------------------------


class TestContinuouslyUpdated:
    def test_satisfies_protocol(self):
        assert isinstance(ContinuouslyUpdated(), t.WeightingStrategy)

    def test_matches_explicit_cholesky_solve(self):
        """``whitening_residual`` agrees with an explicit
        ``forward_solve(cholesky(V), m)``.
        """
        V = _make_pd(seed=3)
        m = jnp.array([0.5, -0.2, 0.7])
        theta = _Scalar(theta=1.0)
        strategy = ContinuouslyUpdated()
        y = strategy.whitening_residual(m, V, theta)
        # Reference: explicit Cholesky + triangular solve.
        L = cholesky(V)
        ref = jax.scipy.linalg.solve_triangular(L, m, lower=True)
        assert jnp.allclose(y, ref, atol=1e-6)

    def test_ad_through_V_of_theta(self):
        """For ``V(theta) = theta**2 * I``, gradient of
        ``sum(whitening_residual(m, V(theta), theta)**2)`` wrt ``theta``
        should be non-zero, demonstrating that JAX AD threads through
        the Cholesky and the solve.
        """
        m_fixed = jnp.array([1.0, -0.5, 2.0])

        def V_of_theta(t: float) -> jnp.ndarray:
            return jnp.eye(3) * (t**2)

        strategy = ContinuouslyUpdated()

        def loss(theta_scalar: float) -> jnp.ndarray:
            # Wrap theta as a dataclass so the call site mirrors the
            # framework's protocol.
            theta = _Scalar(theta=theta_scalar)
            V = V_of_theta(theta_scalar)
            y = strategy.whitening_residual(m_fixed, V, theta)
            return jnp.sum(y * y)

        # Analytical: y = m / theta (for theta > 0 with V = theta^2 I),
        # so sum(y**2) = ||m||^2 / theta^2; gradient = -2 ||m||^2 / theta^3.
        theta0 = 1.7
        g = jax.grad(loss)(theta0)
        m_sq = float(jnp.sum(m_fixed * m_fixed))
        expected = -2.0 * m_sq / (theta0**3)
        assert float(g) == pytest.approx(expected, rel=1e-4)
        # And it is definitely non-zero (the key claim).
        assert abs(float(g)) > 1e-6

    def test_is_pytree_with_no_leaves(self):
        leaves, _ = jax.tree_util.tree_flatten(ContinuouslyUpdated())
        assert leaves == []

    def test_jits(self):
        V = _make_pd()
        m = jnp.array([0.5, -0.2, 0.7])
        theta = _Scalar(theta=1.0)
        strategy = ContinuouslyUpdated()

        @jax.jit
        def compute(s, mm, vv, tt):
            return s.whitening_residual(mm, vv, tt)

        y_eager = strategy.whitening_residual(m, V, theta)
        y_jit = compute(strategy, m, V, theta)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)

    def test_cue_alias(self):
        """``CUE`` is the econometrics-literature alias for
        ``ContinuouslyUpdated`` (Hansen-Heaton-Yaron 1996).

        It is the *same* class object, not a subclass; ``CUE()`` and
        ``ContinuouslyUpdated()`` are interchangeable at every call
        site.
        """
        assert CUE is ContinuouslyUpdated


# ---------------------------------------------------------------------------


class TestIteratedWeighting:
    """Unit tests for the ``IteratedWeighting`` *strategy object*.

    Estimator-level tests (the actual outer Python loop, convergence
    matching CU, iteration-cap warning) live in
    :class:`TestIteratedWeightingEstimator` below.
    """

    def test_satisfies_protocol(self):
        w = IteratedWeighting(weighting_iterations=5, weighting_tol=1e-6)
        assert isinstance(w, t.WeightingStrategy)

    def test_fields_round_trip(self):
        w = IteratedWeighting(weighting_iterations=7, weighting_tol=1e-3)
        assert w.weighting_iterations == 7
        assert w.weighting_tol == pytest.approx(1e-3)

    def test_validates_iterations(self):
        with pytest.raises(ValueError, match="weighting_iterations"):
            IteratedWeighting(weighting_iterations=0, weighting_tol=1e-6)

    def test_validates_tol(self):
        with pytest.raises(ValueError, match="weighting_tol"):
            IteratedWeighting(weighting_iterations=5, weighting_tol=0.0)

    def test_static_fields_in_pytree(self):
        """``weighting_iterations`` and ``weighting_tol`` are static (no leaves)."""
        w = IteratedWeighting(weighting_iterations=3, weighting_tol=1e-6)
        leaves, _ = jax.tree_util.tree_flatten(w)
        assert leaves == []

    def test_whitening_residual_matches_cu_fallback(self):
        """Direct ``whitening_residual`` call mirrors :class:`ContinuouslyUpdated`."""
        V = _make_pd(seed=4)
        m = jnp.array([0.3, -0.1, 0.4])
        theta = _Scalar(theta=1.2)
        iterated = IteratedWeighting(weighting_iterations=5, weighting_tol=1e-6)
        cu = ContinuouslyUpdated()
        y_it = iterated.whitening_residual(m, V, theta)
        y_cu = cu.whitening_residual(m, V, theta)
        assert jnp.allclose(y_it, y_cu, atol=1e-6)


# ---------------------------------------------------------------------------
#
# Estimator-level tests for the iterated-weighting outer loop.


@jdc.pytree_dataclass
class _MeanModelParams:
    mu: float


def _mean_var_psi(x: jnp.ndarray, p: _MeanModelParams) -> jnp.ndarray:
    """Two moments: (X - mu) and (X^2 - (mu^2 + 1)).

    Truth: mu = ``_MEAN_TRUTH``, X ~ N(mu, 1). Two moments and one
    parameter give a J_dof = 1 over-identified problem, so the
    weighting matrix is non-trivial.
    """
    return jnp.array([x[0] - p.mu, x[0] ** 2 - (p.mu**2 + 1.0)])


_MEAN_TRUTH = 0.5
_N_SIM = 4000


def _make_mean_sampler():
    """Sampler returning N(_MEAN_TRUTH, 1) draws."""

    def sampler(key, p):
        del p  # exogenous DGP
        z = jax.random.normal(key, shape=(_N_SIM, 1))
        return z + _MEAN_TRUTH

    return sampler


def _build_measure():
    from emu_gmm.measures import SyntheticMeasure

    return SyntheticMeasure(
        key=jax.random.PRNGKey(0),
        n_sim=_N_SIM,
        sampler=_make_mean_sampler(),
    )


class TestIteratedWeightingEstimator:
    """Drive :func:`emu_gmm.estimate` with :class:`IteratedWeighting`."""

    def test_converges_in_a_few_iters_on_linear_case(self):
        """Mean-and-variance model: iterated weighting converges quickly.

        With a smooth, well-specified DGP the V-refresh fixed point is
        reached well inside ``weighting_iterations``. We start the
        outer loop already near the fixed point so 1-2 outer steps
        suffice.
        """
        from emu_gmm import (
            IteratedWeighting,
            SyntheticCovariance,
            estimate,
            optimistix_lm,
        )

        result = estimate(
            model=_mean_var_psi,
            measure=_build_measure(),
            covariance=SyntheticCovariance(),
            weighting=IteratedWeighting(weighting_iterations=10, weighting_tol=1e-8),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_MeanModelParams(mu=_MEAN_TRUTH),
        )
        assert result.converged
        assert result.diagnostics.optimizer_info.status == "converged"
        # Recovery is sharp at the truth-initialised problem.
        assert float(result.theta_hat.mu) == pytest.approx(_MEAN_TRUTH, abs=0.05)

    def test_iterated_matches_cu_in_well_specified_case(self):
        """Iterated and CU agree to high precision on a smooth example.

        Both schemes are asymptotically equivalent; at the same data /
        same starting point on a smooth, well-specified model the
        finite-sample point estimates should agree to many digits.
        """
        from emu_gmm import (
            ContinuouslyUpdated,
            IteratedWeighting,
            SyntheticCovariance,
            estimate,
            optimistix_lm,
        )

        measure = _build_measure()
        cov = SyntheticCovariance()
        opt = optimistix_lm(rtol=1e-10, atol=1e-10)
        theta0 = _MeanModelParams(mu=0.0)

        r_iter = estimate(
            model=_mean_var_psi,
            measure=measure,
            covariance=cov,
            weighting=IteratedWeighting(weighting_iterations=20, weighting_tol=1e-10),
            optimizer=opt,
            theta_init=theta0,
        )
        r_cu = estimate(
            model=_mean_var_psi,
            measure=measure,
            covariance=cov,
            weighting=ContinuouslyUpdated(),
            optimizer=opt,
            theta_init=theta0,
        )
        assert r_iter.converged
        assert r_cu.converged
        assert float(r_iter.theta_hat.mu) == pytest.approx(
            float(r_cu.theta_hat.mu), abs=1e-6
        )

    def test_max_iterations_warns_and_does_not_raise(self):
        """Capped iterations: surface a warning, return result with
        ``converged=False``, and *do not* raise."""
        from emu_gmm import IteratedWeighting, SyntheticCovariance, estimate

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimate(
                model=_mean_var_psi,
                measure=_build_measure(),
                covariance=SyntheticCovariance(),
                # weighting_iterations=1 with an impossibly tight tol
                # forces a single outer step that cannot satisfy the
                # exit criterion.
                weighting=IteratedWeighting(
                    weighting_iterations=1,
                    weighting_tol=1e-30,
                ),
                theta_init=_MeanModelParams(mu=0.0),
            )

        # The warning was emitted.
        iterated_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "IteratedWeighting" in str(w.message)
        ]
        assert len(iterated_warnings) == 1
        # Surfaces non-convergence in the result without raising.
        assert result.converged is False
        assert result.diagnostics.optimizer_info.status == "max_iterations"

    def test_outer_steps_counter_advances(self):
        """``OptimizerInfo.steps`` accumulates the inner solves' step counts."""
        from emu_gmm import IteratedWeighting, SyntheticCovariance, estimate

        result = estimate(
            model=_mean_var_psi,
            measure=_build_measure(),
            covariance=SyntheticCovariance(),
            weighting=IteratedWeighting(weighting_iterations=5, weighting_tol=1e-8),
            theta_init=_MeanModelParams(mu=0.0),
        )
        # At least one inner LM step was taken.
        assert result.iterations >= 1

    def test_inner_non_convergence_warns_and_flags_failure(self):
        """Inner LM non-convergence propagates to the outer level.

        Drives the iterated outer loop with an optimiser whose inner
        ``max_steps`` budget is impossibly small (``max_steps=1``) so
        every inner Fixed-weight solve returns
        ``status="max_iterations"``. The outer loop must:

        - emit a :class:`UserWarning` explicitly naming inner-solve
          non-convergence (concern #1 from the PR #34 review);
        - return ``EstimationResult.converged=False`` so downstream code
          that branches on ``result.converged`` does the right thing;
        - tag the outer ``OptimizerInfo.status`` as
          ``"inner_non_convergence"`` so a single, consistent
          convergence flag is visible in diagnostics.
        """
        from emu_gmm import (
            IteratedWeighting,
            SyntheticCovariance,
            estimate,
            optimistix_lm,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimate(
                model=_mean_var_psi,
                measure=_build_measure(),
                covariance=SyntheticCovariance(),
                weighting=IteratedWeighting(
                    weighting_iterations=2,
                    weighting_tol=1e-8,
                ),
                # max_steps=1 starves every inner LM solve: it cannot
                # certify convergence in a single LM step on this
                # smooth-but-not-trivial problem starting from mu=0.0.
                optimizer=optimistix_lm(rtol=1e-12, atol=1e-12, max_steps=1),
                theta_init=_MeanModelParams(mu=0.0),
            )

        inner_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "inner" in str(w.message).lower()
            and "IteratedWeighting" in str(w.message)
        ]
        assert len(inner_warnings) == 1, (
            "Expected exactly one UserWarning about inner-solve "
            f"non-convergence; got {[str(w.message) for w in caught]}"
        )
        assert result.converged is False
        assert result.diagnostics.optimizer_info.status == "inner_non_convergence"

    def test_final_objective_matches_cu_fallback_at_theta_hat(self):
        """``OptimizerInfo.final_objective`` is the user-facing CU value.

        Before the concern #2 fix, ``final_objective`` was the inner
        Fixed-weight objective at the penultimate :math:`V_k`, *not* the
        CU-fallback value at the returned ``theta_hat``. The two agree
        at the V-refresh fixed point but differ when the outer loop is
        capped early.

        This test:

        1. runs the iterated estimator with a tight cap so the outer
           loop terminates early (``weighting_iterations=2``);
        2. recomputes the CU-fallback objective by hand from
           ``result.theta_hat``, ``measure.expectation``,
           ``covariance.covariance``, the anchored ridge, and the
           ``IteratedWeighting.whitening_residual`` (which is the CU
           fallback);
        3. asserts ``result.diagnostics.optimizer_info.final_objective``
           equals that hand-computed value.
        """
        from emu_gmm import (
            DiagonalTikhonov,
            IteratedWeighting,
            SyntheticCovariance,
            estimate,
        )
        from emu_gmm.measures import SyntheticMeasure

        measure = _build_measure()
        cov = SyntheticCovariance()
        reg = DiagonalTikhonov()
        weighting = IteratedWeighting(weighting_iterations=2, weighting_tol=1e-8)

        result = estimate(
            model=_mean_var_psi,
            measure=measure,
            covariance=cov,
            regularization=reg,
            weighting=weighting,
            theta_init=_MeanModelParams(mu=0.0),
        )

        # Recompute the anchored ridge (estimator anchors at theta_init).
        V0 = cov.covariance(_mean_var_psi, _MeanModelParams(mu=0.0), measure)
        _, tau_anchor = reg.apply(V0)
        tau_anchor = jnp.asarray(tau_anchor)

        def apply_anchored(V):
            return V + tau_anchor * jnp.diag(jnp.diag(V))

        m_hat = jnp.asarray(measure.expectation(_mean_var_psi, result.theta_hat))
        V_hat = cov.covariance(_mean_var_psi, result.theta_hat, measure)
        V_star_hat = apply_anchored(V_hat)
        y_hat = weighting.whitening_residual(m_hat, V_star_hat, result.theta_hat)
        expected_final_obj = 0.5 * float(jnp.sum(y_hat * y_hat))

        actual = float(result.diagnostics.optimizer_info.final_objective)
        assert actual == pytest.approx(expected_final_obj, rel=1e-10, abs=1e-12)

        # Keep the SyntheticMeasure import live --- some lint configs flag
        # the conditional import otherwise.
        assert isinstance(measure, SyntheticMeasure)

    def test_weighting_tol_is_rescaled_by_parameter_norm(self):
        """``weighting_tol`` is interpreted as a relative-to-||theta|| test.

        Before the concern #3 fix, ``delta < weighting_tol`` compared
        the raw L2 step against a fixed absolute number, which is
        meaningless when parameter components vary by orders of
        magnitude: at theta ~ 1e6 a 0.1-unit step is tiny, but the
        absolute test would call it large; at theta ~ 1e-6 a 0.1-unit
        step is gigantic but the absolute test on the same tol would
        call it the same.

        This test uses a 2-parameter model where the truth is at very
        different scales (``mu ~ 1e0`` and ``nu ~ 1e6``) and confirms
        that the iterated loop converges to ``status="converged"`` with
        a tol value (``1e-4``) that would be impossible to satisfy in
        absolute terms (the optimum step on the large parameter is
        O(1) at best). Equivalently: the test fails if the loop falls
        through to ``"max_iterations"``.
        """
        from emu_gmm import (
            IteratedWeighting,
            SyntheticCovariance,
            estimate,
            optimistix_lm,
        )

        @jdc.pytree_dataclass
        class _TwoScaleParams:
            mu: float
            nu: float

        # Truth: ``mu_true ~ O(1)``, ``nu_true ~ O(1e6)``; the two
        # parameters thus span six orders of magnitude. Moment vector
        # is two affine residuals so the system is exactly identified
        # (M = K = 2) and any optimiser should reach the truth.
        mu_true = 1.5
        nu_true = 1.5e6

        def two_scale_psi(x, p):
            return jnp.array([x[0] - p.mu, (x[1] - p.nu) / 1e6])

        def sampler(key, p):
            del p
            keys = jax.random.split(key, 2)
            x0 = jax.random.normal(keys[0], shape=(_N_SIM,)) + mu_true
            x1 = jax.random.normal(keys[1], shape=(_N_SIM,)) * 1e3 + nu_true
            return jnp.stack([x0, x1], axis=1)

        from emu_gmm.measures import SyntheticMeasure

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=_N_SIM,
            sampler=sampler,
        )

        # weighting_tol = 1e-4 is a *relative* tolerance once rescaled
        # by max(||theta||, eps). Without rescaling, an absolute
        # threshold of 1e-4 is unreachable on the ``nu`` coordinate
        # where typical LM steps are O(1) due to the 1e6 scale.
        result = estimate(
            model=two_scale_psi,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=IteratedWeighting(weighting_iterations=10, weighting_tol=1e-4),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=_TwoScaleParams(mu=0.0, nu=0.0),
        )
        assert result.converged
        assert result.diagnostics.optimizer_info.status == "converged"
