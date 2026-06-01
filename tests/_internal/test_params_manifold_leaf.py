"""Phase-1 tests for the manifold-aware flatten core (manifold epic #12).

Covers:

* ``ManifoldLeaf`` PyTree-node contract (single array child; manifold +
  dtype in static, hashable aux_data; immutability; jit cache stability).
* ``flatten_params_with_spec`` 3-tuple: ambient-block concatenation,
  spec construction, the block-width invariant.
* ``unflatten_params`` manifold-aware round-trip preserving exact shape
  AND dtype for scalar AND non-scalar leaves.
* v1 back-compat: ``flatten_params`` stays a 2-tuple, scalar-only,
  bitwise identical, and equals ``flatten_params_with_spec(...)[:2]`` for
  all-scalar trees.

These tests address the red-team risks R1-R37 named in the task spec.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds import (
    Euclidean,
    ManifoldLeaf,
    Product,
    PSDFixedRank,
)


@jdc.pytree_dataclass
class _ScalarPair:
    beta: float
    gamma: float


@jdc.pytree_dataclass
class _ProductParams:
    """A:(5,K) PSDFixedRank matrix leaf + phi:() scalar leaf."""

    A: ManifoldLeaf
    phi: float


def _make_product(K: int = 2, seed: int = 0) -> _ProductParams:
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((5, K)))
    return _ProductParams(A=ManifoldLeaf(A, PSDFixedRank(5, K)), phi=jnp.asarray(0.37))


# ---------------------------------------------------------------------------
# ManifoldLeaf PyTree contract (R6, R7, R18, R19, R27, R30).
# ---------------------------------------------------------------------------
class TestManifoldLeafPyTree:
    def test_exactly_one_array_child(self):
        # R6: the manifold must NOT be a leaf; only the array is.
        leaf = ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2))
        children = jax.tree_util.tree_leaves(leaf)
        assert len(children) == 1
        assert jnp.asarray(children[0]).shape == (5, 2)

    def test_aux_data_is_static_hashable(self):
        # R7/R27: aux_data carries (manifold, dtype) -- hashable, no arrays.
        leaf = ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2))
        children, aux = leaf.tree_flatten()
        assert len(children) == 1
        manifold, dtype = aux
        assert isinstance(manifold, PSDFixedRank)
        # aux_data must hash (jit cache key).
        hash(aux)
        # No jax array smuggled into aux_data.
        assert not isinstance(manifold, jnp.ndarray)

    def test_structurally_identical_leaves_same_treedef(self):
        # R18/R19: identical structure -> identical treedef (cache hit).
        a = ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2))
        b = ManifoldLeaf(jnp.ones((5, 2)), PSDFixedRank(5, 2))
        _, td_a = jax.tree_util.tree_flatten(a)
        _, td_b = jax.tree_util.tree_flatten(b)
        assert td_a == td_b

    def test_immutable(self):
        # R30: mutation must raise.
        leaf = ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2))
        with pytest.raises(AttributeError):
            leaf.array = jnp.ones((5, 2))

    def test_rejects_non_manifold(self):
        with pytest.raises(TypeError):
            ManifoldLeaf(jnp.zeros((5, 2)), object())

    def test_round_trips_through_jit_without_recompile(self):
        # R7/R18/R19: same structure must not trigger recompilation.
        calls = {"n": 0}

        @jax.jit
        def f(leaf: ManifoldLeaf) -> jnp.ndarray:
            calls["n"] += 1  # only runs during tracing
            return jnp.sum(leaf.array) + leaf.array.shape[0]

        a = ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2))
        b = ManifoldLeaf(jnp.ones((5, 2)), PSDFixedRank(5, 2))
        f(a)
        f(b)
        assert calls["n"] == 1  # single trace -> cache hit on second call

    def test_dtype_recorded_in_aux(self):
        leaf = ManifoldLeaf(jnp.zeros((5, 2), dtype=jnp.float32), PSDFixedRank(5, 2))
        _, aux = leaf.tree_flatten()
        _, dtype = aux
        assert dtype == jnp.float32


# ---------------------------------------------------------------------------
# v1 back-compat (R1, R2, R3, R15, R16).
# ---------------------------------------------------------------------------
class TestV1BackCompat:
    def test_flatten_params_still_2_tuple(self):
        p = _ScalarPair(beta=0.95, gamma=2.0)
        out = params_mod.flatten_params(p)
        assert isinstance(out, tuple) and len(out) == 2

    def test_flatten_params_rejects_non_scalar_unchanged(self):
        # R15: v1 flatten stays scalar-only.
        @jdc.pytree_dataclass
        class Bad:
            vec: jnp.ndarray

        bad = Bad(vec=jnp.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="0-d scalars"):
            params_mod.flatten_params(bad)

    def test_with_spec_matches_v1_first_two_for_scalar_tree(self):
        # R2/R3: bitwise-identical flat + same treedef for scalar trees.
        p = _ScalarPair(beta=0.95, gamma=2.0)
        flat_v1, treedef_v1 = params_mod.flatten_params(p)
        flat_v2, treedef_v2, spec = params_mod.flatten_params_with_spec(p)
        assert jnp.array_equal(flat_v1, flat_v2)
        assert flat_v1.dtype == flat_v2.dtype
        assert flat_v1.shape == flat_v2.shape == (2,)
        assert treedef_v1 == treedef_v2
        assert spec.total_ambient_dim == 2
        assert spec.total_gauge_dim == 0

    def test_unflatten_v1_signature_unchanged(self):
        # R4/R10: 2-arg unflatten (manifold_spec=None) is the v1 path.
        p = _ScalarPair(beta=0.95, gamma=2.0)
        flat, treedef = params_mod.flatten_params(p)
        restored = params_mod.unflatten_params(flat, treedef)
        assert isinstance(restored, _ScalarPair)
        # Scalar leaves remain 0-d (R10/R33).
        assert jnp.asarray(restored.beta).ndim == 0
        assert float(restored.beta) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Manifold-aware flatten / unflatten round-trip (THE acceptance, R5, R8,
# R9, R20, R21, R23, R34, R37).
# ---------------------------------------------------------------------------
class TestProductRoundTrip:
    def test_acceptance_shape_and_dtype(self):
        # The Phase-1 gate: (A:(5,2), phi:()) round-trips with exact
        # shape AND dtype per leaf.
        p = _make_product(K=2)
        A_orig = p.A.array
        phi_orig = jnp.asarray(p.phi)

        flat, treedef, spec = params_mod.flatten_params_with_spec(p)

        # Buffer width = 5*2 + 1 = 11 (R23).
        assert flat.shape == (11,)
        assert spec.total_ambient_dim == 11
        assert spec.total_dimension == 11  # ambient storage convention
        assert spec.total_gauge_dim == 2 * (2 - 1) // 2  # == 1

        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert isinstance(restored, _ProductParams)

        A_back = restored.A.array
        phi_back = jnp.asarray(restored.phi)

        assert A_back.shape == (5, 2)
        assert A_back.dtype == A_orig.dtype
        assert jnp.array_equal(A_back, A_orig)

        assert phi_back.shape == ()
        assert phi_back.dtype == phi_orig.dtype
        assert float(phi_back) == pytest.approx(float(phi_orig))

        # The ManifoldParam survives the round-trip.
        assert restored.A.manifold == PSDFixedRank(5, 2)

    def test_K3_round_trip(self):
        p = _make_product(K=3, seed=7)
        flat, treedef, spec = params_mod.flatten_params_with_spec(p)
        assert flat.shape == (5 * 3 + 1,)
        assert spec.total_gauge_dim == 3 * (3 - 1) // 2  # == 3
        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert jnp.array_equal(restored.A.array, p.A.array)

    def test_row_major_order_preserved(self):
        # R20: non-square block must reconstruct element-for-element
        # (no transpose). Use a deterministic, distinct-entry matrix.
        A = jnp.arange(10.0).reshape((5, 2))
        p = _ProductParams(A=ManifoldLeaf(A, PSDFixedRank(5, 2)), phi=jnp.asarray(1.0))
        flat, treedef, spec = params_mod.flatten_params_with_spec(p)
        # First block is A ravelled C-order: [0,1,2,...,9].
        assert jnp.array_equal(flat[:10], jnp.arange(10.0))
        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert jnp.array_equal(restored.A.array, A)

    def test_block_width_invariant(self):
        # R5/R9/R23: sum(prod(ambient_shape)) == len(flat).
        p = _make_product(K=2)
        flat, _, spec = params_mod.flatten_params_with_spec(p)
        total = sum(int(np.prod(ls.ambient_shape)) for ls in spec.leaf_specs)
        assert total == int(flat.shape[0])

    def test_leaf_spec_order_and_offsets(self):
        # R21: leaf_specs in PyTree-leaf-walk order; offsets tile buffer.
        p = _make_product(K=2)
        _, _, spec = params_mod.flatten_params_with_spec(p)
        names = [ls.field_name for ls in spec.leaf_specs]
        assert names == ["A", "phi"]
        offsets = [ls.offset for ls in spec.leaf_specs]
        shapes = [ls.ambient_shape for ls in spec.leaf_specs]
        assert offsets == [0, 10]
        assert shapes == [(5, 2), ()]

    def test_mixed_dtype_round_trip(self):
        # R8/R12: a float32 matrix leaf + float64 scalar round-trips with
        # each leaf's exact dtype, even though the flat buffer promotes.
        A = jnp.asarray(np.arange(10.0).reshape((5, 2)), dtype=jnp.float32)
        phi = jnp.asarray(0.5, dtype=jnp.float64)
        p = _ProductParams(A=ManifoldLeaf(A, PSDFixedRank(5, 2)), phi=phi)
        flat, treedef, spec = params_mod.flatten_params_with_spec(p)
        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert restored.A.array.dtype == jnp.float32
        assert jnp.asarray(restored.phi).dtype == jnp.float64
        assert jnp.array_equal(restored.A.array, A)

    def test_idempotent(self):
        # R24: pure / idempotent.
        p = _make_product(K=2)
        f1, t1, s1 = params_mod.flatten_params_with_spec(p)
        f2, t2, s2 = params_mod.flatten_params_with_spec(p)
        assert jnp.array_equal(f1, f2)
        assert t1 == t2
        assert s1 == s2

    def test_rejects_unwrapped_non_scalar(self):
        # R29: a bare matrix leaf (not wrapped) is rejected loudly.
        @jdc.pytree_dataclass
        class BadNonScalar:
            mat: jnp.ndarray

        bad = BadNonScalar(mat=jnp.ones((3, 2)))
        with pytest.raises(ValueError, match="ManifoldLeaf"):
            params_mod.flatten_params_with_spec(bad)

    def test_unflatten_length_mismatch_raises(self):
        # R34: a flat buffer that doesn't match the spec is rejected.
        p = _make_product(K=2)
        _, treedef, spec = params_mod.flatten_params_with_spec(p)
        with pytest.raises(ValueError, match="elements"):
            params_mod.unflatten_params(jnp.zeros(7), treedef, spec)

    def test_grad_through_manifold_unflatten(self):
        # The optimiser differentiates a loss as a function of flat.
        p = _make_product(K=2)
        flat, treedef, spec = params_mod.flatten_params_with_spec(p)

        def loss(f):
            params = params_mod.unflatten_params(f, treedef, spec)
            return jnp.sum(params.A.array**2) + jnp.asarray(params.phi) ** 2

        g = jax.grad(loss)(flat)
        # d/dA = 2A, d/dphi = 2 phi.
        expected = jnp.concatenate(
            [
                2.0 * jnp.reshape(p.A.array, (-1,)),
                2.0 * jnp.reshape(jnp.asarray(p.phi), (1,)),
            ]
        )
        assert jnp.allclose(g, expected)


# ---------------------------------------------------------------------------
# Product-of-Euclidean (vector) leaf, to exercise the d>1 Euclidean path.
# ---------------------------------------------------------------------------
class TestEuclideanVectorLeaf:
    def test_vector_leaf_round_trip(self):
        @jdc.pytree_dataclass
        class P:
            v: ManifoldLeaf
            s: float

        v = jnp.asarray([1.0, 2.0, 3.0])
        p = P(v=ManifoldLeaf(v, Euclidean(3)), s=jnp.asarray(9.0))
        flat, treedef, spec = params_mod.flatten_params_with_spec(p)
        assert flat.shape == (4,)
        assert spec.total_gauge_dim == 0
        restored = params_mod.unflatten_params(flat, treedef, spec)
        assert jnp.array_equal(restored.v.array, v)
        assert restored.v.array.shape == (3,)
        assert float(restored.s) == pytest.approx(9.0)


def test_product_manifold_constructs_for_spec():
    # Sanity that the Product manifold the slice targets composes the
    # factor gauge dims the spec sums independently.
    prod = Product(PSDFixedRank(5, 2), Euclidean())
    assert prod.gauge_dim == 1
    assert prod.dimension == 11
