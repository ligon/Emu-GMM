"""Tests for emu_gmm.manifolds.Euclidean (Phase 1).

Exercises:
- The :class:`ManifoldParam` protocol round-trip on ``Euclidean``.
- The plan §2.8 contract: ``Euclidean()`` (scalar) round-trips a 0-d
  scalar; ``Euclidean(1)`` is the (rare) 1-D length-1 alternative.
- Operator algebra: projection is identity, retraction is addition.
- Frobenius distance correctness.
- ``tangent_basis_names`` for scalar / vector / matrix shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.manifolds import Euclidean, ManifoldParam


class TestEuclideanProtocol:
    def test_satisfies_manifold_param_protocol(self):
        assert isinstance(Euclidean(3), ManifoldParam)
        assert isinstance(Euclidean(), ManifoldParam)
        assert isinstance(Euclidean(2, 3), ManifoldParam)

    def test_attributes(self):
        m = Euclidean(3)
        assert m.dimension == 3
        assert m.gauge_dim == 0
        assert m.ambient_shape == (3,)

    def test_scalar_euclidean_is_zero_d(self):
        """Plan §2.8: ``Euclidean()`` has empty shape and dimension == 1."""
        m = Euclidean()
        assert m.ambient_shape == ()
        assert m.dimension == 1
        assert m.gauge_dim == 0

    def test_matrix_dimension(self):
        m = Euclidean(2, 3)
        assert m.dimension == 6
        assert m.ambient_shape == (2, 3)


class TestEuclideanOperators:
    def test_projection_is_identity(self):
        m = Euclidean(4)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        v = jnp.array([0.1, 0.2, 0.3, 0.4])
        # Plan §2.7: projection is idempotent identity for Euclidean.
        assert jnp.allclose(m.projection(x, v), v)
        assert jnp.allclose(m.projection(x, m.projection(x, v)), v)

    def test_retraction_is_addition(self):
        m = Euclidean(3)
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, -0.5, 0.0])
        assert jnp.allclose(m.retraction(x, v), x + v)

    def test_riemannian_gradient_is_identity(self):
        m = Euclidean(2, 2)
        x = jnp.eye(2)
        g = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(m.riemannian_gradient(x, g), g)

    def test_distance_is_frobenius(self):
        m = Euclidean(3)
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 6.0, 3.0])
        # ||a - b|| = sqrt(9 + 16 + 0) = 5
        assert float(m.distance(a, b)) == pytest.approx(5.0)

    def test_random_point_has_right_shape_and_dtype(self):
        m = Euclidean(3, 4)
        key = jax.random.PRNGKey(0)
        p = m.random_point(key)
        assert p.shape == (3, 4)
        assert p.dtype == jnp.float64


class TestEuclideanLabels:
    def test_scalar_label_is_field_name(self):
        """Plan §2.10: ``Euclidean()`` preserves the v1 single-name contract."""
        assert Euclidean().tangent_basis_names("beta") == ["beta"]

    def test_vector_labels(self):
        m = Euclidean(3)
        assert m.tangent_basis_names("mu") == ["mu_t_0", "mu_t_1", "mu_t_2"]

    def test_matrix_labels(self):
        m = Euclidean(2, 2)
        assert m.tangent_basis_names("L") == [
            "L_t_0_0",
            "L_t_0_1",
            "L_t_1_0",
            "L_t_1_1",
        ]


class TestEuclideanHashEquality:
    """Required so ``ManifoldSpec`` can be a frozen dataclass."""

    def test_equality(self):
        assert Euclidean(3) == Euclidean(3)
        assert Euclidean() == Euclidean()
        assert Euclidean(3) != Euclidean(4)
        assert Euclidean() != Euclidean(1)  # plan §2.8: distinct!

    def test_hashable(self):
        # Same shape -> same hash; storable in a set/dict.
        s = {Euclidean(3), Euclidean(3), Euclidean(4)}
        assert len(s) == 2


class TestPymanoptParityIfAvailable:
    """Compare to pymanopt's Euclidean (skip if pymanopt missing)."""

    def test_parity_random_shape_grid(self):
        pytest.importorskip("pymanopt")
        from pymanopt.manifolds import Euclidean as PymanoptEuclidean

        rng = np.random.default_rng(0)
        for shape in [(3,), (5,), (3, 2), (4, 5)]:
            pym = PymanoptEuclidean(*shape)
            emu = Euclidean(*shape)
            x = rng.standard_normal(shape)
            v = rng.standard_normal(shape)
            np.testing.assert_allclose(
                np.asarray(emu.projection(x, v)),
                pym.projection(x, v),
            )
            np.testing.assert_allclose(
                np.asarray(emu.retraction(x, v)),
                pym.retraction(x, v),
            )
