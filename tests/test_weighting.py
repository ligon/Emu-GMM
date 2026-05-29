"""Tests for emu_gmm.weighting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import types as t
from emu_gmm._internal.cholesky import cholesky
from emu_gmm.weighting import CUE, ContinuouslyUpdated, Fixed, Identity


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
