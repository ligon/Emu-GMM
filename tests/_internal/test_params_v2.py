"""Tests for emu_gmm._internal.params v2 surface (plan §3 / Phase 1).

The v1 ``(flat, treedef)`` return of :func:`flatten_params` is
unchanged. The v2 addition is :func:`manifold_spec_from_params`, which
returns a :class:`ManifoldSpec` annotating each leaf. For v1-style
trees this spec carries one ``Euclidean()`` (scalar) per leaf, with
``total_gauge_dim == 0``.

Verifies:

- Bitwise-identical flat array for v1 trees (the §2.8 contract).
- :class:`ManifoldSpec` is hashable (so it can ride as a
  ``jdc.static_field`` or a ``static_argnames`` to :func:`jax.jit`).
- Mixed scalar-leaf trees produce a spec consistent with their PyTree.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds import Euclidean, ManifoldSpec


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


@jdc.pytree_dataclass
class _ThreeField:
    a: float
    b: float
    c: float


class TestV1FlattenBitwiseIdentical:
    """Plan §2.8: a v1-style tree produces flat / treedef identical to v1."""

    def test_two_field_flat(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        flat, treedef = params_mod.flatten_params(p)
        # Exact (bitwise) match to the v1 contract.
        assert jnp.array_equal(flat, jnp.array([0.95, 2.0]))
        # Round-trip.
        restored = params_mod.unflatten_params(flat, treedef)
        assert isinstance(restored, _EulerParams)
        assert float(restored.beta) == pytest.approx(0.95)
        assert float(restored.gamma) == pytest.approx(2.0)

    def test_three_field_flat(self):
        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        flat, _ = params_mod.flatten_params(p)
        assert jnp.array_equal(flat, jnp.array([1.0, 2.0, 3.0]))


class TestManifoldSpecForV1Trees:
    def test_simple_two_field(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)

        # The high-level totals.
        assert spec.total_ambient_dim == 2
        assert spec.total_dimension == 2
        assert spec.total_gauge_dim == 0
        # Two leaves, both Euclidean() (per plan §2.8).
        assert len(spec.leaf_specs) == 2
        for ls in spec.leaf_specs:
            assert ls.ambient_shape == ()
            assert isinstance(ls.manifold, Euclidean)
            assert ls.manifold.ambient_shape == ()
            # Critical contract: NOT Euclidean(1).
            assert ls.manifold != Euclidean(1)

    def test_leaf_offsets_increment(self):
        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        spec = params_mod.manifold_spec_from_params(p)
        offsets = [ls.offset for ls in spec.leaf_specs]
        assert offsets == [0, 1, 2]

    def test_field_names_preserved(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)
        names = [ls.field_name for ls in spec.leaf_specs]
        assert names == ["beta", "gamma"]

    def test_non_dataclass_root_gets_none_field_names(self):
        # A plain tuple of scalars: no dataclass field names available.
        spec = params_mod.manifold_spec_from_params((1.0, 2.0, 3.0))
        assert all(ls.field_name is None for ls in spec.leaf_specs)
        assert spec.total_ambient_dim == 3
        assert spec.total_gauge_dim == 0


class TestManifoldSpecHashable:
    """Required for jit-static usage (plan §2.7)."""

    def test_hashable(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)
        # Hashing should succeed; same params -> same hash.
        h1 = hash(spec)
        spec2 = params_mod.manifold_spec_from_params(_EulerParams(beta=0.5, gamma=0.5))
        h2 = hash(spec2)
        # Values of the *params* don't enter the spec; only shapes/manifolds.
        assert h1 == h2

    def test_distinct_specs_distinct_hashes(self):
        s_two = params_mod.manifold_spec_from_params(_EulerParams(beta=0.0, gamma=0.0))
        s_three = params_mod.manifold_spec_from_params(_ThreeField(a=0.0, b=0.0, c=0.0))
        assert hash(s_two) != hash(s_three)

    def test_usable_as_jit_static_argument(self):
        """ManifoldSpec passes through :func:`jax.jit` as a static argument."""
        from functools import partial

        spec = params_mod.manifold_spec_from_params(_EulerParams(beta=0.95, gamma=2.0))

        @partial(jax.jit, static_argnames=("spec",))
        def f(flat: jnp.ndarray, spec: ManifoldSpec) -> jnp.ndarray:
            # Trivial: use spec to know how long flat should be.
            return jnp.broadcast_to(spec.total_ambient_dim, flat.shape) + flat

        flat = jnp.array([0.95, 2.0])
        out = f(flat, spec)
        assert out.shape == flat.shape
        assert jnp.array_equal(out, flat + 2)


class TestV1AcceptanceUnchanged:
    """Sanity: the v1 estimator path uses flatten_params + treedef and
    must not see any behavioural change. The full v1 acceptance suite
    exercises this end-to-end; here we exercise the smallest contract."""

    def test_grad_through_flatten(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        flat_init, treedef = params_mod.flatten_params(p)

        def loss(flat):
            params = params_mod.unflatten_params(flat, treedef)
            return params.beta**2 + 2 * params.gamma**2

        g = jax.grad(loss)(flat_init)
        assert jnp.allclose(g, jnp.array([1.9, 8.0]))

    def test_dataclass_round_trip_field_count(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)
        assert spec.total_ambient_dim == len(dataclasses.fields(_EulerParams))
