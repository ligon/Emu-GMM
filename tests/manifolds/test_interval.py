"""Unit tests for the bounded scalar :class:`Interval` manifold (#152).

The interval analogue of :class:`Positive`: the logit-pullback geometry on
``(lo, hi)`` whose exponential retraction never crosses either bound. Mirrors
``test_positive.py``. Motivation: a compact ``[lo, hi]`` scale parameter
restores the CUE regularity condition (``V_X`` bounded away from singular) that
fails as ``sigma -> 0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal.params import (
    flatten_params_with_spec,
    manifold_spec_from_params,
    unflatten_params,
)
from emu_gmm.manifolds import Interval

jax.config.update("jax_enable_x64", True)


class TestIntervalProtocol:
    def test_shape_dim_gauge(self):
        m = Interval(0.5, 3.0)
        assert m.ambient_shape == ()
        assert m.dimension == 1
        assert m.gauge_dim == 0

    def test_requires_lo_lt_hi(self):
        with pytest.raises(ValueError):
            Interval(2.0, 1.0)
        with pytest.raises(ValueError):
            Interval(1.0, 1.0)

    def test_frozen_hashable_equal(self):
        assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
        assert hash(Interval(0.0, 1.0)) == hash(Interval(0.0, 1.0))
        assert Interval(0.0, 1.0) != Interval(0.0, 2.0)


class TestIntervalOperators:
    LO, HI = 0.5, 4.0

    def test_retraction_never_crosses_bounds(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.7)
        # Never crosses either bound, for ANY v (sigmoid in [0,1] => R in [lo,hi]).
        for v in [-1e4, -100.0, -1.0, 0.0, 1.0, 100.0, 1e4]:
            xn = float(m.retraction(x, jnp.asarray(v)))
            assert self.LO <= xn <= self.HI, (v, xn)
        # Strictly interior for moderate steps (extreme v saturates to the bound
        # in float64, exactly as Positive's exp retraction reaches 0 at v -> -inf).
        for v in [-20.0, -1.0, 1.0, 20.0]:
            xn = float(m.retraction(x, jnp.asarray(v)))
            assert self.LO < xn < self.HI, (v, xn)

    def test_retraction_identity_at_zero(self):
        m = Interval(self.LO, self.HI)
        assert float(m.retraction(jnp.asarray(2.3), jnp.asarray(0.0))) == pytest.approx(
            2.3, abs=1e-12
        )

    def test_retraction_differential_is_identity(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(2.3)
        t = 1e-6
        fd = (
            float(m.retraction(x, jnp.asarray(t)))
            - float(m.retraction(x, jnp.asarray(-t)))
        ) / (2.0 * t)
        assert fd == pytest.approx(1.0, rel=1e-6)
        assert float(m.retraction_differential(x)) == pytest.approx(1.0)

    def test_inner_product_and_norm(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.2)
        phip = (self.HI - self.LO) / ((x - self.LO) * (self.HI - x))
        u, v = jnp.asarray(0.7), jnp.asarray(-0.3)
        assert float(m.inner_product(x, u, v)) == pytest.approx(
            float(phip**2 * u * v), rel=1e-10
        )
        assert float(m.norm(x, u)) == pytest.approx(float(jnp.abs(u) * phip), rel=1e-10)

    def test_gradient_relation(self):
        # g_x(rgrad, v) == egrad * v  for all v.
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.9)
        egrad = jnp.asarray(0.42)
        rgrad = m.euclidean_to_riemannian_gradient(x, egrad)
        for v in [0.5, -1.3, 2.0]:
            vv = jnp.asarray(v)
            assert float(m.inner_product(x, rgrad, vv)) == pytest.approx(
                float(egrad * vv), rel=1e-10
            )

    def test_distance_symmetric_and_zero_diagonal(self):
        m = Interval(self.LO, self.HI)
        a, b = jnp.asarray(1.0), jnp.asarray(3.0)
        assert float(m.distance(a, a)) == pytest.approx(0.0, abs=1e-12)
        assert float(m.distance(a, b)) == pytest.approx(
            float(m.distance(b, a)), rel=1e-12
        )
        assert float(m.distance(a, b)) > 0.0

    def test_retraction_realises_geodesic_distance(self):
        # ||v||_g == distance(x, R_x(v)): the exp map is a unit-speed geodesic.
        m = Interval(self.LO, self.HI)
        x, v = jnp.asarray(1.5), jnp.asarray(0.8)
        xn = m.retraction(x, v)
        assert float(m.distance(x, xn)) == pytest.approx(float(m.norm(x, v)), rel=1e-8)

    def test_random_point_in_bounds(self):
        m = Interval(self.LO, self.HI)
        for s in range(5):
            p = float(m.random_point(jax.random.PRNGKey(s)))
            assert self.LO < p < self.HI


class TestIntervalAsLeaf:
    def test_spec_resolves_interval_and_roundtrips(self):
        @jdc.pytree_dataclass
        class P:
            sigma: jnp.ndarray
            __emu_manifolds__ = {"sigma": Interval(0.5, 4.0)}

        p = P(sigma=jnp.asarray(1.3))
        # The parameter-space declaration layer resolves the annotated manifold.
        spec = manifold_spec_from_params(p)
        assert spec.leaf_specs[0].manifold == Interval(0.5, 4.0)
        assert int(spec.total_gauge_dim) == 0
        flat, treedef, _ = flatten_params_with_spec(p)
        assert int(flat.shape[0]) == 1
        p2 = unflatten_params(flat, treedef, manifold_spec=spec)
        assert float(p2.sigma) == pytest.approx(1.3, abs=1e-12)
