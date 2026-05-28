"""Tests for emu_gmm._internal.cholesky."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from emu_gmm._internal import cholesky as cho


# Deterministic 5x5 PD matrix used across the suite.
def _make_pd_5() -> jnp.ndarray:
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((5, 5)).astype(np.float64)
    V = A @ A.T + np.eye(5)  # symmetric PD
    return jnp.asarray(V)


class TestCholesky:
    def test_returns_lower_triangular(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        # Above the diagonal: zeros (lower-triangular).
        upper = jnp.triu(L, k=1)
        assert jnp.allclose(upper, 0.0, atol=1e-6)

    def test_factorisation_identity(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        assert jnp.allclose(L @ L.T, V, atol=1e-5)

    def test_matches_scipy(self):
        V = _make_pd_5()
        L_emu = cho.cholesky(V)
        L_sp = scipy.linalg.cholesky(np.asarray(V), lower=True)
        assert jnp.allclose(L_emu, jnp.asarray(L_sp), atol=1e-5)


class TestForwardSolve:
    def test_basic(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        b = jnp.arange(5, dtype=jnp.float32) + 1.0
        y = cho.forward_solve(L, b)
        assert jnp.allclose(L @ y, b, atol=1e-5)

    def test_matches_scipy(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        b = jnp.arange(5, dtype=jnp.float32) + 1.0
        y_emu = cho.forward_solve(L, b)
        y_sp = scipy.linalg.solve_triangular(np.asarray(L), np.asarray(b), lower=True)
        assert jnp.allclose(y_emu, jnp.asarray(y_sp), atol=1e-5)


class TestBackSolve:
    def test_basic(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        b = jnp.arange(5, dtype=jnp.float32) + 1.0
        x = cho.back_solve(L, b)
        assert jnp.allclose(L.T @ x, b, atol=1e-5)

    def test_matches_scipy(self):
        V = _make_pd_5()
        L = cho.cholesky(V)
        b = jnp.arange(5, dtype=jnp.float32) + 1.0
        x_emu = cho.back_solve(L, b)
        x_sp = scipy.linalg.solve_triangular(
            np.asarray(L), np.asarray(b), lower=True, trans="T"
        )
        assert jnp.allclose(x_emu, jnp.asarray(x_sp), atol=1e-5)


class TestWhiten:
    def test_identity_of_squared_norm(self):
        # ||L^{-1} m||^2 should equal m' V^{-1} m.
        V = _make_pd_5()
        m = jnp.array([0.5, -0.2, 0.7, 0.1, -0.3])
        y = cho.whiten(V, m)

        V_inv = jnp.linalg.inv(V)
        expected = m @ V_inv @ m

        assert float(jnp.sum(y * y)) == pytest.approx(float(expected), rel=1e-4)

    def test_zero_vector(self):
        V = _make_pd_5()
        m = jnp.zeros(5)
        y = cho.whiten(V, m)
        assert jnp.allclose(y, 0.0, atol=1e-7)

    def test_gradient_is_V_inv_m(self):
        # d/dm [(1/2) ||L^{-1} m||^2] = V^{-1} m.
        V = _make_pd_5()
        m = jnp.array([0.5, -0.2, 0.7, 0.1, -0.3])

        def half_quad(mv):
            y = cho.whiten(V, mv)
            return 0.5 * jnp.sum(y * y)

        g = jax.grad(half_quad)(m)
        V_inv = jnp.linalg.inv(V)
        expected = V_inv @ m
        assert jnp.allclose(g, expected, atol=1e-4)


class TestQuadraticForm:
    def test_matches_explicit(self):
        V = _make_pd_5()
        m = jnp.array([0.5, -0.2, 0.7, 0.1, -0.3])
        q = cho.quadratic_form(V, m)
        expected = m @ jnp.linalg.inv(V) @ m
        assert float(q) == pytest.approx(float(expected), rel=1e-4)


class TestJitCompatibility:
    def test_whiten_jits(self):
        V = _make_pd_5()
        m = jnp.array([0.5, -0.2, 0.7, 0.1, -0.3])
        y_eager = cho.whiten(V, m)
        y_jit = jax.jit(cho.whiten)(V, m)
        assert jnp.allclose(y_eager, y_jit, atol=1e-6)

    def test_cholesky_vmaps(self):
        # A batch of 4 PD matrices; cholesky should broadcast.
        rng = np.random.default_rng(seed=7)
        batch = []
        for _ in range(4):
            A = rng.standard_normal((3, 3))
            batch.append(A @ A.T + np.eye(3))
        Vs = jnp.stack([jnp.asarray(v) for v in batch])
        Ls = jax.vmap(cho.cholesky)(Vs)
        for i in range(4):
            assert jnp.allclose(Ls[i] @ Ls[i].T, Vs[i], atol=1e-5)
