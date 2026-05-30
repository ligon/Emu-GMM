"""Unit tests for the scalar :class:`Positive` manifold.

Checks the affine-invariant metric operators and --- the load-bearing
property --- that the exponential retraction never leaves
:math:`\\mathbb{R}_{>0}` for any tangent step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from emu_gmm.manifolds import Euclidean, ManifoldParam, Positive


class TestPositiveProtocol:
    def test_satisfies_manifold_param(self):
        assert isinstance(Positive(), ManifoldParam)

    def test_dimension_and_gauge(self):
        p = Positive()
        assert p.dimension == 1
        assert p.gauge_dim == 0
        assert p.ambient_shape == ()

    def test_instances_equal_and_hash_equal(self):
        assert Positive() == Positive()
        assert hash(Positive()) == hash(Positive())


class TestPositiveOperators:
    def test_projection_is_identity(self):
        p = Positive()
        v = jnp.asarray(0.7)
        assert float(p.projection(jnp.asarray(1.5), v)) == pytest.approx(0.7)

    def test_retraction_first_order_matches_euclidean(self):
        # R_x(v) = x exp(v/x) ~ x + v for small v.
        p = Positive()
        x = jnp.asarray(2.0)
        v = jnp.asarray(1e-4)
        assert float(p.retraction(x, v)) == pytest.approx(float(x + v), rel=1e-6)

    def test_retraction_never_negative_extreme(self):
        # x exp(v/x) >= 0 for any v; never crosses to negative (unlike a
        # plain Euclidean x + v). For huge negative v it may underflow to
        # exactly 0.0 in float64 --- a float limit, not a sign crossing.
        p = Positive()
        for x0 in (2.0, 0.5, 0.05, 1e-3):
            for v in (-100.0, -1e3, -1e6):
                out = float(p.retraction(jnp.asarray(x0), jnp.asarray(v)))
                assert out >= 0.0

    def test_retraction_strictly_positive_moderate(self):
        # For steps that don't underflow, the result is strictly > 0
        # where a Euclidean retraction x + v would have gone negative.
        p = Positive()
        for x0 in (2.0, 0.5, 0.05):
            for v in (-5.0, -20.0, -1.0):
                out = float(p.retraction(jnp.asarray(x0), jnp.asarray(v)))
                assert out > 0.0

    def test_inner_product_metric(self):
        # g_x(u, v) = u v / x^2.
        p = Positive()
        x, u, v = jnp.asarray(2.0), jnp.asarray(3.0), jnp.asarray(5.0)
        assert float(p.inner_product(x, u, v)) == pytest.approx(3.0 * 5.0 / 4.0)

    def test_norm_is_sqrt_inner(self):
        p = Positive()
        x, v = jnp.asarray(2.0), jnp.asarray(6.0)
        assert float(p.norm(x, v)) == pytest.approx(6.0 / 2.0)

    def test_euclidean_to_riemannian_gradient_scales_by_x_squared(self):
        p = Positive()
        x, g = jnp.asarray(1.5), jnp.asarray(4.0)
        assert float(p.euclidean_to_riemannian_gradient(x, g)) == pytest.approx(
            (1.5**2) * 4.0
        )

    def test_riemannian_gradient_alias(self):
        p = Positive()
        x, g = jnp.asarray(1.5), jnp.asarray(4.0)
        assert float(p.riemannian_gradient(x, g)) == float(
            p.euclidean_to_riemannian_gradient(x, g)
        )

    def test_distance_is_log_difference(self):
        p = Positive()
        a, b = jnp.asarray(1.0), jnp.asarray(jnp.e)
        assert float(p.distance(a, b)) == pytest.approx(1.0)

    def test_random_point_positive_and_float64(self):
        p = Positive()
        x = p.random_point(jax.random.PRNGKey(0))
        assert float(x) > 0.0
        assert x.dtype == jnp.float64

    def test_zero_vector(self):
        p = Positive()
        z = p.zero_vector(jnp.asarray(1.5))
        assert float(z) == 0.0
        assert z.shape == ()

    def test_tangent_basis_names(self):
        assert Positive().tangent_basis_names("sigma") == ["sigma"]


class TestEuclideanAdditiveMethods:
    """The Phase-4 additive methods on the landed Euclidean."""

    def test_inner_product(self):
        e = Euclidean()
        u = jnp.asarray([1.0, 2.0])
        v = jnp.asarray([3.0, 4.0])
        assert float(e.inner_product(None, u, v)) == pytest.approx(11.0)

    def test_norm(self):
        e = Euclidean()
        v = jnp.asarray([3.0, 4.0])
        assert float(e.norm(None, v)) == pytest.approx(5.0)

    def test_euclidean_to_riemannian_gradient_identity(self):
        e = Euclidean()
        g = jnp.asarray(7.0)
        assert float(e.euclidean_to_riemannian_gradient(jnp.asarray(2.0), g)) == 7.0
