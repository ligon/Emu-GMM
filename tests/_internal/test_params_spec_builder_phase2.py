"""Phase-2 tests: non-scalar :func:`manifold_spec_from_params`.

Manifold epic #12, Phase 2. The spec builder ``estimate()`` calls
(:func:`emu_gmm._internal.params.manifold_spec_from_params`) now accepts
non-scalar leaves expressed as :class:`ManifoldLeaf`-wrapped pytree
leaves (or via an ``__emu_manifolds__`` annotation), populating each
``LeafSpec.offset`` / ``ambient_shape`` from the *actual array shape*
with ``offset += prod(ambient_shape)`` (NOT ``manifold.dimension``).

The load-bearing contract: for the SAME params,
``manifold_spec_from_params(params)`` agrees field-for-field with
``flatten_params_with_spec(params)[2]`` (offsets, ambient_shape,
manifold, dtype, field_name, leaf order, total_ambient_dim,
total_dimension, total_gauge_dim). Both delegate to the shared
``_walk_leaf_specs`` leaf-walk so they cannot drift.

Covers red-team risks R1-R33 (the spec-builder consistency / silent
block-boundary-corruption family).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds import (
    Euclidean,
    ManifoldLeaf,
    PSDFixedRank,
)


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


@jdc.pytree_dataclass
class _ProductParams:
    """A:(5,K) PSDFixedRank matrix leaf + phi:() scalar leaf."""

    A: ManifoldLeaf
    # The fixtures pass a 0-d jnp array (the v1 scalar-leaf form);
    # ``float`` alone made mypy reject the constructor call (#122).
    phi: float | jax.Array


def _make_product(K: int = 2, seed: int = 0) -> _ProductParams:
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((5, K)))
    return _ProductParams(A=ManifoldLeaf(A, PSDFixedRank(5, K)), phi=jnp.asarray(0.37))


def _assert_specs_equal(s_from_params, s_from_flatten) -> None:
    """Field-for-field equality of two ManifoldSpec (R1/R8/R26/R33)."""
    assert s_from_params.total_ambient_dim == s_from_flatten.total_ambient_dim
    assert s_from_params.total_dimension == s_from_flatten.total_dimension
    assert s_from_params.total_gauge_dim == s_from_flatten.total_gauge_dim
    assert len(s_from_params.leaf_specs) == len(s_from_flatten.leaf_specs)
    for a, b in zip(s_from_params.leaf_specs, s_from_flatten.leaf_specs, strict=True):
        assert a.offset == b.offset
        assert a.ambient_shape == b.ambient_shape
        assert a.manifold == b.manifold
        assert a.field_name == b.field_name
        assert a.dtype == b.dtype
    assert s_from_params == s_from_flatten
    assert hash(s_from_params) == hash(s_from_flatten)


class TestAcceptanceProductSpec:
    def test_psd5x2_plus_scalar(self):
        p = _make_product(K=2)
        spec = params_mod.manifold_spec_from_params(p)

        assert spec.total_ambient_dim == 11
        assert spec.total_dimension == 11
        assert spec.total_gauge_dim == 1  # K(K-1)/2 = 1 for K=2

        assert len(spec.leaf_specs) == 2
        a, phi = spec.leaf_specs
        assert a.offset == 0
        assert a.ambient_shape == (5, 2)
        assert isinstance(a.manifold, PSDFixedRank)
        assert a.manifold.gauge_dim == 1
        assert a.field_name == "A"
        assert phi.offset == 10
        assert phi.ambient_shape == ()
        assert isinstance(phi.manifold, Euclidean)
        assert phi.manifold.ambient_shape == ()
        assert phi.field_name == "phi"

    def test_psd5x3_plus_scalar(self):
        p = _make_product(K=3)
        spec = params_mod.manifold_spec_from_params(p)

        assert spec.total_gauge_dim == 3  # K(K-1)/2 = 3 for K=3
        a, phi = spec.leaf_specs
        assert a.ambient_shape == (5, 3)
        assert int(np.prod(a.ambient_shape)) == 15
        assert a.manifold.dimension == 15
        assert spec.total_ambient_dim == 16
        assert spec.total_dimension == 16
        assert phi.offset == 15

    def test_offset_is_prod_ambient_not_manifold_dimension(self):
        @jdc.pytree_dataclass
        class _ThreeLeaf:
            A: ManifoldLeaf
            phi: float
            psi: float

        rng = np.random.default_rng(1)
        A = jnp.asarray(rng.standard_normal((5, 2)))
        p = _ThreeLeaf(
            A=ManifoldLeaf(A, PSDFixedRank(5, 2)),
            phi=jnp.asarray(0.1),
            psi=jnp.asarray(0.2),
        )
        spec = params_mod.manifold_spec_from_params(p)
        offsets = [ls.offset for ls in spec.leaf_specs]
        assert offsets == [0, 10, 11]
        assert spec.total_ambient_dim == 12


class TestSpecBuilderConsistency:
    def test_product_k2_consistent(self):
        p = _make_product(K=2)
        s1 = params_mod.manifold_spec_from_params(p)
        _, _, s2 = params_mod.flatten_params_with_spec(p)
        _assert_specs_equal(s1, s2)

    def test_product_k3_consistent(self):
        p = _make_product(K=3)
        s1 = params_mod.manifold_spec_from_params(p)
        _, _, s2 = params_mod.flatten_params_with_spec(p)
        _assert_specs_equal(s1, s2)

    def test_v1_scalar_tree_consistent(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        s1 = params_mod.manifold_spec_from_params(p)
        _, _, s2 = params_mod.flatten_params_with_spec(p)
        _assert_specs_equal(s1, s2)

    def test_non_dataclass_tuple_consistent(self):
        p = (1.0, 2.0, 3.0)
        s1 = params_mod.manifold_spec_from_params(p)
        _, _, s2 = params_mod.flatten_params_with_spec(p)
        _assert_specs_equal(s1, s2)

    def test_float32_dtype_consistent(self):
        @jdc.pytree_dataclass
        class _Mixed:
            A: ManifoldLeaf
            phi: float

        A = jnp.zeros((5, 2), dtype=jnp.float32)
        p = _Mixed(A=ManifoldLeaf(A, PSDFixedRank(5, 2)), phi=jnp.asarray(0.5))
        s1 = params_mod.manifold_spec_from_params(p)
        _, _, s2 = params_mod.flatten_params_with_spec(p)
        _assert_specs_equal(s1, s2)
        assert s1.leaf_specs[0].dtype == jnp.float32


class TestV1Unchanged:
    def test_all_scalar_totals(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)
        assert spec.total_ambient_dim == 2
        assert spec.total_dimension == 2
        assert spec.total_gauge_dim == 0
        for ls in spec.leaf_specs:
            assert ls.ambient_shape == ()
            assert isinstance(ls.manifold, Euclidean)
            assert ls.manifold.ambient_shape == ()
            assert ls.manifold != Euclidean(1)

    def test_offsets_increment_by_one(self):
        @jdc.pytree_dataclass
        class _ThreeField:
            a: float
            b: float
            c: float

        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        spec = params_mod.manifold_spec_from_params(p)
        assert [ls.offset for ls in spec.leaf_specs] == [0, 1, 2]

    def test_field_count_invariant(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        spec = params_mod.manifold_spec_from_params(p)
        assert spec.total_dimension == len(dataclasses.fields(_EulerParams))


class TestAnnotationPath:
    def test_scalar_positive_annotation_unchanged(self):
        from emu_gmm.manifolds import Positive

        @jdc.pytree_dataclass
        class _Sig:
            sigma: float

        _Sig.__emu_manifolds__ = {"sigma": Positive()}
        p = _Sig(sigma=jnp.asarray(1.5))
        spec = params_mod.manifold_spec_from_params(p)
        # manifold_spec_from_params honours __emu_manifolds__ (the v1-lite
        # scalar-Positive path estimate() drives, e.g.
        # tests/test_estimator_positive.py). flatten_params_with_spec
        # deliberately does NOT consult the annotation -- a bare scalar
        # leaf there is Euclidean() -- so the two builders intentionally
        # differ ONLY on this annotation path, which never pairs the
        # scalar-Positive tree with a flatten round-trip. The
        # ManifoldLeaf-wrapped path (TestSpecBuilderConsistency) is the one
        # where field-for-field equality is contractual.
        assert spec.leaf_specs[0].ambient_shape == ()
        assert isinstance(spec.leaf_specs[0].manifold, Positive)
        assert spec.leaf_specs[0].field_name == "sigma"
        assert spec.total_ambient_dim == 1
        assert spec.total_dimension == 1
        assert spec.total_gauge_dim == 0


class TestGuards:
    def test_bare_non_scalar_leaf_rejected(self):
        @jdc.pytree_dataclass
        class _Bad:
            A: jnp.ndarray
            phi: float

        p = _Bad(A=jnp.zeros((5, 2)), phi=jnp.asarray(0.1))
        with pytest.raises(ValueError, match="non-scalar"):
            params_mod.manifold_spec_from_params(p)

    def test_array_shape_manifold_mismatch_rejected(self):
        @jdc.pytree_dataclass
        class _Mismatch:
            A: ManifoldLeaf
            phi: float

        leaf = ManifoldLeaf(jnp.zeros((5, 3)), PSDFixedRank(5, 2))
        p = _Mismatch(A=leaf, phi=jnp.asarray(0.1))
        with pytest.raises(ValueError, match="ambient_shape"):
            params_mod.manifold_spec_from_params(p)

    def test_block_boundary_invariant_holds(self):
        for K in (2, 3, 4):
            spec = params_mod.manifold_spec_from_params(_make_product(K=K))
            total = sum(int(np.prod(ls.ambient_shape)) for ls in spec.leaf_specs)
            assert total == spec.total_ambient_dim == spec.total_dimension


class TestHashable:
    def test_product_spec_hashable(self):
        spec = params_mod.manifold_spec_from_params(_make_product(K=2))
        hash(spec)
        spec2 = params_mod.manifold_spec_from_params(_make_product(K=2, seed=99))
        assert hash(spec) == hash(spec2)

    def test_distinct_K_distinct_hash(self):
        s2 = params_mod.manifold_spec_from_params(_make_product(K=2))
        s3 = params_mod.manifold_spec_from_params(_make_product(K=3))
        assert hash(s2) != hash(s3)


class TestRoundTrip:
    def test_spec_drives_unflatten(self):
        p = _make_product(K=2)
        flat, treedef, _ = params_mod.flatten_params_with_spec(p)
        spec = params_mod.manifold_spec_from_params(p)
        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert isinstance(restored, _ProductParams)
        assert jnp.allclose(restored.A.array, p.A.array)
        assert jnp.allclose(jnp.asarray(restored.phi), jnp.asarray(p.phi))
        assert restored.A.array.shape == (5, 2)
