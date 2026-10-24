"""Registered ownership regressions for ledger Section 23 / issue #191."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import EmpiricalMeasure, IIDCovariance, build_estimator, estimate
from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, Positive, PSDFixedRank


@jdc.pytree_dataclass
class _StaticContainer:
    config: jdc.Static[str]
    sigma: float
    offsets: tuple[float, float]

    __emu_manifolds__ = {"sigma": Positive()}


def test_static_and_container_fields_do_not_move_legacy_geometry():
    theta = _StaticContainer("config", 2.0, (-3.0, 4.0))
    spec = params_mod.manifold_spec_from_params(theta)
    assert [leaf.field_name for leaf in spec.leaf_specs] == ["sigma", None, None]
    assert [type(leaf.manifold) for leaf in spec.leaf_specs] == [
        Positive,
        Euclidean,
        Euclidean,
    ]
    flat, _, flat_spec = params_mod.flatten_params_with_spec(theta)
    np.testing.assert_array_equal(flat, [2.0, -3.0, 4.0])
    assert [leaf.field_name for leaf in flat_spec.leaf_specs] == ["sigma", None, None]
    # Legacy annotations intentionally do not affect the ambient flatten codec.
    assert all(isinstance(leaf.manifold, Euclidean) for leaf in flat_spec.leaf_specs)
    with pytest.raises(NotImplementedError, match="Nested"):
        params_mod.param_names(theta)


def test_direct_dynamic_names_exclude_static_fields():
    @jdc.pytree_dataclass
    class Flat:
        config: jdc.Static[str]
        sigma: float
        other: float

        __emu_manifolds__ = {"sigma": Positive()}

    # Equal numeric values cannot establish field ownership by themselves.
    theta = Flat("config", 1.0, 1.0)
    spec = params_mod.manifold_spec_from_params(theta)
    assert params_mod.param_names(theta) == ["sigma", "other"]
    assert [leaf.field_name for leaf in spec.leaf_specs] == ["sigma", "other"]
    assert isinstance(spec.leaf_specs[0].manifold, Positive)
    assert isinstance(spec.leaf_specs[1].manifold, Euclidean)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, slots=True)
class _Reordered:
    left: Any
    right: Any
    config: Any

    __emu_manifolds__ = {"left": Positive()}

    def __post_init__(self):
        if not all(
            isinstance(value, float | jax.Array | jax.core.Tracer)
            for value in (self.left, self.right)
        ):
            raise TypeError("constructor requires numeric parameters")

    def tree_flatten(self):
        return (self.right, self.left), (self.config,)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(left=children[1], right=children[0], config=auxiliary[0])


@pytest.mark.parametrize("left,right", [(1.0, 2.0), (1.0, 1.0)])
def test_foreign_slots_registration_owns_its_actual_order(left, right):
    theta = _Reordered(left, right, 1.0)
    spec = params_mod.manifold_spec_from_params(theta)
    flat, treedef, flat_spec = params_mod.flatten_params_with_spec(theta)
    assert params_mod.param_names(theta) == ["right", "left"]
    assert [leaf.field_name for leaf in spec.leaf_specs] == ["right", "left"]
    assert [leaf.field_name for leaf in flat_spec.leaf_specs] == ["right", "left"]
    assert isinstance(spec.leaf_specs[0].manifold, Euclidean)
    assert isinstance(spec.leaf_specs[1].manifold, Positive)
    np.testing.assert_array_equal(flat, [right, left])
    restored = params_mod.unflatten_params(flat, treedef, flat_spec)
    assert restored.config == theta.config
    assert float(restored.left) == left
    assert float(restored.right) == right


def test_legacy_annotation_on_container_refuses_even_one_descendant():
    @jdc.pytree_dataclass
    class Nested:
        block: tuple[float]

        __emu_manifolds__ = {"block": Positive()}

    theta = Nested((1.0,))
    with pytest.raises(ValueError, match="directly registered top-level"):
        params_mod.manifold_spec_from_params(theta)
    _, _, flat_spec = params_mod.flatten_params_with_spec(theta)
    assert flat_spec.leaf_specs[0].field_name is None
    assert isinstance(flat_spec.leaf_specs[0].manifold, Euclidean)


def test_native_manifold_containers_keep_geometry_and_stay_unnamed():
    @jdc.pytree_dataclass
    class Native:
        config: jdc.Static[str]
        block: dict

    theta = Native(
        "config",
        {
            "A": ManifoldLeaf(jnp.array([[1.0], [2.0]]), PSDFixedRank(2, 1)),
            "sigma": ManifoldLeaf(jnp.array(3.0), Positive()),
        },
    )
    spec = params_mod.manifold_spec_from_params(theta)
    flat, treedef, flat_spec = params_mod.flatten_params_with_spec(theta)
    assert spec == flat_spec
    assert all(leaf.field_name is None for leaf in spec.leaf_specs)
    assert [type(leaf.manifold) for leaf in spec.leaf_specs] == [PSDFixedRank, Positive]
    restored = params_mod.unflatten_params(flat, treedef, spec)
    assert restored.config == "config"
    np.testing.assert_array_equal(restored.block["A"].array, theta.block["A"].array)


@jdc.pytree_dataclass
class _Checked:
    block: ManifoldLeaf

    def __post_init__(self):
        if not isinstance(self.block, ManifoldLeaf):
            raise RuntimeError("constructor requires a native ManifoldLeaf")


def test_constructor_invariants_are_not_bypassed_by_naming_probe():
    theta = _Checked(ManifoldLeaf(jnp.array(2.0), Positive()))
    spec = params_mod.manifold_spec_from_params(theta)
    assert isinstance(spec.leaf_specs[0].manifold, Positive)
    assert spec.leaf_specs[0].field_name == "block"
    _, _, flat_spec = params_mod.flatten_params_with_spec(theta)
    assert flat_spec == spec
    assert params_mod.param_names(theta) == ["block"]

    # Actual registered reconstruction still supplies a wrapper, not a token;
    # the constructor accepts JAX's real traced/placeholder round trip.
    @jax.jit
    def flattened(point):
        return params_mod.flatten_params_with_spec(point)[0]

    np.testing.assert_array_equal(flattened(theta), [2.0])
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), theta)
    np.testing.assert_array_equal(jax.jit(jax.vmap(flattened))(batched), [[2.0], [2.0]])
    with pytest.raises(RuntimeError, match="requires a native"):
        _Checked(object())


def test_legacy_annotation_survives_constructor_checked_field():
    @jdc.pytree_dataclass
    class CheckedScalar:
        sigma: Any

        __emu_manifolds__ = {"sigma": Positive()}

        def __post_init__(self):
            if not isinstance(self.sigma, float | jax.Array | jax.core.Tracer):
                raise RuntimeError("constructor requires numeric sigma")

    theta = CheckedScalar(2.0)
    spec = params_mod.manifold_spec_from_params(theta)
    assert isinstance(spec.leaf_specs[0].manifold, Positive)
    assert spec.leaf_specs[0].field_name == "sigma"
    assert params_mod.param_names(theta) == ["sigma"]


@jdc.pytree_dataclass
class _CheckedScalars:
    a: Any
    b: Any

    def __post_init__(self):
        if not all(
            isinstance(value, float | jax.Array | jax.core.Tracer)
            for value in (self.a, self.b)
        ):
            raise TypeError("constructor requires numeric parameters")


@pytest.mark.parametrize("initial", [(0.0, 0.0), (0.25, 0.75)])
def test_checked_scalar_estimate_and_factory_remain_supported(initial):
    # Eight cube vertices give centered, independent empirical noise with a
    # positive covariance. All three moment means vanish at (1.5, -0.5).
    noise = np.array(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    )
    measure = EmpiricalMeasure.from_arrays(noise + np.array([1.5, -0.5, 1.0]))
    theta = _CheckedScalars(*initial)

    def model(row, point):
        return row - jnp.array([point.a, point.b, point.a + point.b])

    result = estimate(model, measure, covariance=IIDCovariance(), parameters=theta)
    run = build_estimator(
        model, measure=measure, covariance=IIDCovariance(), parameters=theta
    )
    reused = run(result.theta_hat, measure)
    for fitted in (result, reused):
        assert bool(fitted.converged)
        np.testing.assert_allclose(
            [fitted.theta_hat.a, fitted.theta_hat.b], [1.5, -0.5], atol=1e-8
        )
        assert params_mod.param_names(fitted.theta_hat) == ["a", "b"]


def test_checked_native_wrapper_estimate_and_factory_remain_supported():
    measure = EmpiricalMeasure.from_arrays(np.array([[1.0], [2.0], [3.0], [4.0]]))
    theta = _Checked(ManifoldLeaf(jnp.array(1.0), Positive()))

    def model(row, point):
        return jnp.array([row[0] - point.block.array])

    result = estimate(model, measure, covariance=IIDCovariance(), parameters=theta)
    run = build_estimator(
        model, measure=measure, covariance=IIDCovariance(), parameters=theta
    )
    reused = run(result.theta_hat, measure)
    for fitted in (result, reused):
        assert bool(fitted.converged)
        assert float(fitted.theta_hat.block.array) == pytest.approx(2.5, abs=1e-8)
        assert params_mod.param_names(fitted.theta_hat) == ["block"]


def test_checked_numeric_probe_preserves_jit_vmap_ad_and_original_leaves():
    theta = _CheckedScalars(jnp.array(1.0, dtype=jnp.float32), jnp.array(1.0))
    original = jax.tree.leaves(theta)
    leaves, names = params_mod._parameter_leaves_and_names(theta)
    assert names == ["a", "b"]
    assert all(after is before for after, before in zip(leaves, original, strict=True))

    def flattened(point):
        assert params_mod.param_names(point) == ["a", "b"]
        return params_mod.flatten_params_for_ad(point)[0]

    np.testing.assert_array_equal(jax.jit(flattened)(theta), [1.0, 1.0])
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), theta)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(flattened))(batched), np.ones((2, 2))
    )
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(flattened(p) ** 2)))(theta)
    np.testing.assert_array_equal(jax.tree.leaves(gradient), [2.0, 2.0])
    assert theta.a is original[0] and theta.b is original[1]
    assert theta.a.dtype == jnp.float32


def test_constructor_that_erases_probe_identity_still_refuses_legacy_binding():
    @jdc.pytree_dataclass
    class CopiesInput:
        sigma: Any

        __emu_manifolds__ = {"sigma": Positive()}

        def __post_init__(self):
            object.__setattr__(self, "sigma", jnp.array(self.sigma, copy=True))

    theta = CopiesInput(2.0)
    with pytest.raises(ValueError, match="cannot bind"):
        params_mod.manifold_spec_from_params(theta)


def test_registered_ownership_is_static_under_jit_and_vmap():
    theta = _StaticContainer("config", 2.0, (-3.0, 4.0))

    def inspect(point):
        spec = params_mod.manifold_spec_from_params(point)
        return jnp.array(
            [isinstance(leaf.manifold, Positive) for leaf in spec.leaf_specs]
        )

    np.testing.assert_array_equal(jax.jit(inspect)(theta), [True, False, False])
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), theta)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(inspect))(batched),
        [[True, False, False], [True, False, False]],
    )
