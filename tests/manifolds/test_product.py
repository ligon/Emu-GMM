"""Tests for emu_gmm.manifolds.Product (Phase 3).

Verifies that ``Product`` composes factor manifolds correctly:

- ``dimension`` is the sum of factor dimensions.
- ``gauge_dim`` is the sum of factor gauge_dims.
- Factor-wise operators (projection, retraction, riemannian_gradient,
  distance) delegate correctly.
- ``random_point`` produces tuples of correctly-shaped arrays per factor.
- Nested ``Product(Product(...))`` is rejected at construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.manifolds import Euclidean, Product, PSDFixedRank


class TestProductConstruction:
    def test_dimension_sums(self):
        m = Product(Euclidean(3), PSDFixedRank(5, 2), Euclidean(2))
        assert m.dimension == 3 + 5 * 2 + 2  # 15

    def test_gauge_dim_sums(self):
        m = Product(Euclidean(3), PSDFixedRank(5, 2), PSDFixedRank(4, 3))
        # 0 + k(k-1)/2 for k=2 (= 1) + k(k-1)/2 for k=3 (= 3) = 4
        assert m.gauge_dim == 0 + 1 + 3

    def test_empty_factors_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Product()

    def test_nested_product_rejected(self):
        inner = Product(Euclidean(2), Euclidean(3))
        with pytest.raises(ValueError, match="Nested Product"):
            Product(inner, Euclidean(1))

    def test_ambient_shape_not_defined(self):
        m = Product(Euclidean(3), Euclidean(2))
        with pytest.raises(NotImplementedError, match="per-factor"):
            _ = m.ambient_shape

    def test_factors_attribute_exposes_originals(self):
        e3 = Euclidean(3)
        p52 = PSDFixedRank(5, 2)
        m = Product(e3, p52)
        assert m.factors == (e3, p52)


class TestProductOperators:
    def _build(self):
        return Product(Euclidean(3), PSDFixedRank(5, 2))

    def test_projection_per_factor(self):
        m = self._build()
        p_eucl = jnp.array([1.0, 2.0, 3.0])
        p_psd = PSDFixedRank(5, 2).random_point(jax.random.PRNGKey(0))
        v_eucl = jnp.array([0.1, 0.2, 0.3])
        v_psd = jax.random.normal(jax.random.PRNGKey(1), (5, 2), dtype=jnp.float64)
        result = m.projection((p_eucl, p_psd), (v_eucl, v_psd))
        # Euclidean factor: identity.
        np.testing.assert_allclose(np.asarray(result[0]), np.asarray(v_eucl))
        # PSD factor: matches direct call.
        expected_psd = PSDFixedRank(5, 2).projection(p_psd, v_psd)
        np.testing.assert_allclose(np.asarray(result[1]), np.asarray(expected_psd))

    def test_retraction_per_factor(self):
        m = self._build()
        p = (jnp.zeros(3), jnp.zeros((5, 2)))
        v = (jnp.array([1.0, 2.0, 3.0]), jnp.ones((5, 2)))
        out = m.retraction(p, v)
        np.testing.assert_allclose(np.asarray(out[0]), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(np.asarray(out[1]), np.ones((5, 2)))

    def test_distance_is_root_sum_of_squares(self):
        m = Product(Euclidean(3), Euclidean(2))
        a = (jnp.array([1.0, 2.0, 3.0]), jnp.array([0.0, 0.0]))
        b = (jnp.array([4.0, 6.0, 3.0]), jnp.array([3.0, 4.0]))
        # ||a0-b0|| = 5; ||a1-b1|| = 5; total = sqrt(25 + 25) = sqrt(50).
        out = float(m.distance(a, b))
        assert out == pytest.approx(np.sqrt(50.0))

    def test_random_point_factor_shapes(self):
        m = Product(Euclidean(3), PSDFixedRank(5, 2), Euclidean(2))
        key = jax.random.PRNGKey(123)
        point = m.random_point(key)
        assert isinstance(point, tuple)
        assert len(point) == 3
        assert point[0].shape == (3,)
        assert point[1].shape == (5, 2)
        assert point[2].shape == (2,)


class TestProductHashEquality:
    def test_equality(self):
        m1 = Product(Euclidean(3), PSDFixedRank(5, 2))
        m2 = Product(Euclidean(3), PSDFixedRank(5, 2))
        m3 = Product(Euclidean(3), PSDFixedRank(5, 3))
        assert m1 == m2
        assert m1 != m3

    def test_hashable(self):
        s = {Product(Euclidean(3)), Product(Euclidean(3))}
        assert len(s) == 1


class TestPymanoptParity:
    def test_parity_dimension(self):
        pytest.importorskip("pymanopt")
        from pymanopt.manifolds import Euclidean as PymEucl
        from pymanopt.manifolds import Product as PymProduct
        from pymanopt.manifolds import PSDFixedRank as PymPSD

        m_emu = Product(Euclidean(3), PSDFixedRank(5, 2), Euclidean(2))
        m_pym = PymProduct([PymEucl(3), PymPSD(5, 2), PymEucl(2)])
        # pymanopt reports the quotient dimension; ours is the *ambient*
        # dimension. The difference equals total gauge_dim.
        assert m_pym.dim == m_emu.dimension - m_emu.gauge_dim
