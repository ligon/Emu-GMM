"""Tests for emu_gmm.manifolds.PSDFixedRank (Phase 2).

Exercises:

- The :class:`ManifoldParam` protocol round-trip.
- Plan §2.1 contract: ``dimension == n*k`` (ambient!),
  ``gauge_dim == k*(k-1)/2``.
- The Kronecker Lyapunov solve matches scipy's
  :func:`scipy.linalg.solve_continuous_lyapunov` at float64.
- Pymanopt parity on a (m, k) x seed grid: projection / retraction /
  riemannian_gradient / distance agree at rtol=1e-9.
- ``projection`` is idempotent.
- Gauge invariance: projection commutes with right-multiplication by
  :math:`Q \\in O(k)` up to expected factors.
- ``jit(projection)`` and ``vmap(projection)`` work (the v1 hot-path
  contract carries over to v2 manifolds).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.manifolds import ManifoldParam, PSDFixedRank
from emu_gmm.manifolds.psd_fixed_rank import _solve_continuous_lyapunov_kron

# Pymanopt parity grid (plan §2 Phase-2 acceptance, reduced to 20 seeds
# from the plan's 100 for unit-test speed; the standalone parity harness
# at docs/reviews/pymanopt_parity_harness.py runs the full 100 grid).
SHAPES = [(3, 1), (3, 2), (5, 1), (5, 2), (5, 3), (8, 2), (8, 3), (10, 3)]
N_SEEDS_PARITY = 20


class TestProtocolSurface:
    def test_satisfies_manifold_param(self):
        assert isinstance(PSDFixedRank(5, 2), ManifoldParam)

    def test_attributes(self):
        m = PSDFixedRank(5, 2)
        assert m.ambient_shape == (5, 2)
        assert m.dimension == 10  # n*k, plan §2.1 ambient storage
        assert m.gauge_dim == 1  # k*(k-1)/2 = 1 for k=2

    def test_gauge_dim_grid(self):
        # k*(k-1)/2 for various k
        assert PSDFixedRank(5, 1).gauge_dim == 0
        assert PSDFixedRank(5, 2).gauge_dim == 1
        assert PSDFixedRank(5, 3).gauge_dim == 3
        assert PSDFixedRank(10, 4).gauge_dim == 6

    def test_rejects_invalid_k(self):
        with pytest.raises(ValueError, match="1 <= k <= n"):
            PSDFixedRank(3, 0)
        with pytest.raises(ValueError, match="1 <= k <= n"):
            PSDFixedRank(3, 4)  # k > n


class TestKroneckerLyapunovSolve:
    """Plan §4 specifies a Kronecker Lyapunov formulation."""

    def test_matches_scipy_solve_continuous_lyapunov(self):
        scipy_linalg = pytest.importorskip("scipy.linalg")
        rng = np.random.default_rng(0)
        for k in (1, 2, 3, 4, 5):
            Y = rng.standard_normal((7, k))
            A = Y.T @ Y
            V = rng.standard_normal((7, k))
            B = Y.T @ V - V.T @ Y  # skew-symmetric
            X_scipy = scipy_linalg.solve_continuous_lyapunov(A, B)
            X_jax = np.asarray(
                _solve_continuous_lyapunov_kron(jnp.asarray(A), jnp.asarray(B))
            )
            np.testing.assert_allclose(X_jax, X_scipy, rtol=1e-10, atol=1e-12)


class TestProjectionAlgebra:
    def test_idempotent(self):
        m = PSDFixedRank(5, 2)
        key = jax.random.PRNGKey(0)
        Y = m.random_point(key)
        V = jax.random.normal(jax.random.PRNGKey(1), (5, 2), dtype=jnp.float64)
        once = m.projection(Y, V)
        twice = m.projection(Y, once)
        np.testing.assert_allclose(np.asarray(once), np.asarray(twice), atol=1e-12)

    def test_zero_input_yields_zero_output(self):
        m = PSDFixedRank(4, 2)
        Y = m.random_point(jax.random.PRNGKey(7))
        zero = jnp.zeros((4, 2), dtype=jnp.float64)
        out = m.projection(Y, zero)
        np.testing.assert_allclose(np.asarray(out), np.zeros((4, 2)), atol=1e-12)


class TestPymanoptParity:
    """Parity tests against the reference pymanopt implementation.

    Gated by ``pytest.importorskip("pymanopt")``. The standalone harness
    at ``docs/reviews/pymanopt_parity_harness.py`` exercises the same
    operations on a larger grid.
    """

    @pytest.mark.parametrize("shape", SHAPES)
    def test_parity_projection_retraction_riemannian_distance(self, shape):
        pytest.importorskip("pymanopt")
        from pymanopt.manifolds import PSDFixedRank as PymanoptPSDFixedRank

        n, k = shape
        pym = PymanoptPSDFixedRank(n, k)
        emu = PSDFixedRank(n, k)

        for seed in range(N_SEEDS_PARITY):
            rng = np.random.default_rng(seed * 9973 + n * 131 + k)
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            X = np.asarray(pym.random_point(), dtype=np.float64)
            Y = np.asarray(pym.random_point(), dtype=np.float64)
            V = np.asarray(pym.random_tangent_vector(X), dtype=np.float64)
            ambient = rng.standard_normal(size=X.shape).astype(np.float64)

            # Projection
            np.testing.assert_allclose(
                np.asarray(emu.projection(X, ambient)),
                pym.projection(X, ambient),
                rtol=1e-9,
                atol=0.0,
            )
            # Retraction
            np.testing.assert_allclose(
                np.asarray(emu.retraction(X, V)),
                pym.retraction(X, V),
                rtol=1e-9,
                atol=0.0,
            )
            # Riemannian gradient (pymanopt: euclidean_to_riemannian_gradient).
            np.testing.assert_allclose(
                np.asarray(emu.riemannian_gradient(X, ambient)),
                pym.euclidean_to_riemannian_gradient(X, ambient),
                rtol=1e-9,
                atol=0.0,
            )
            # Distance
            np.testing.assert_allclose(
                np.asarray(emu.distance(X, Y)),
                pym.dist(X, Y),
                rtol=1e-9,
                atol=0.0,
            )


class TestJITAndVMAP:
    """Hot-path contract: operators must be jit/vmap-friendly."""

    def test_jit_projection(self):
        m = PSDFixedRank(5, 2)
        Y = m.random_point(jax.random.PRNGKey(0))
        V = jax.random.normal(jax.random.PRNGKey(1), (5, 2), dtype=jnp.float64)
        eager = m.projection(Y, V)
        jitted = jax.jit(m.projection)(Y, V)
        np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), atol=1e-12)

    def test_vmap_projection(self):
        m = PSDFixedRank(5, 2)
        batch = 4
        Ys = jax.random.normal(jax.random.PRNGKey(0), (batch, 5, 2), dtype=jnp.float64)
        Vs = jax.random.normal(jax.random.PRNGKey(1), (batch, 5, 2), dtype=jnp.float64)
        vmapped = jax.vmap(m.projection)(Ys, Vs)
        assert vmapped.shape == (batch, 5, 2)
        # And matches per-slice eager computation.
        for i in range(batch):
            np.testing.assert_allclose(
                np.asarray(vmapped[i]),
                np.asarray(m.projection(Ys[i], Vs[i])),
                atol=1e-12,
            )


class TestGaugeInvariance:
    """Projection ``Q``-equivariance: projecting at ``YQ`` then right-mul
    by ``Q.T`` matches projecting at ``Y`` (with the ambient vector
    right-mul'd by ``Q``)."""

    def test_q_equivariance(self):
        m = PSDFixedRank(5, 2)
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((5, 2))
        V = rng.standard_normal((5, 2))
        # Random orthogonal Q in O(2)
        q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        lhs = np.asarray(m.projection(Y @ q, V @ q)) @ q.T
        rhs = np.asarray(m.projection(Y, V))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestLabels:
    def test_basis_names_count(self):
        m = PSDFixedRank(3, 2)
        names = m.tangent_basis_names("L")
        assert len(names) == m.dimension == 6

    def test_basis_names_format_single_digit(self):
        m = PSDFixedRank(3, 2)
        names = m.tangent_basis_names("L")
        assert names == [
            "L_t_00",
            "L_t_01",
            "L_t_10",
            "L_t_11",
            "L_t_20",
            "L_t_21",
        ]

    def test_basis_names_format_double_digit(self):
        m = PSDFixedRank(11, 2)
        names = m.tangent_basis_names("L")
        # Fallback underscore-separated form when indices exceed 9.
        assert names[0] == "L_t_0_0"
        assert names[-1] == "L_t_10_1"
        assert len(names) == 22
