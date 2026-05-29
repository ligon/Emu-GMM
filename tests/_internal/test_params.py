"""Tests for emu_gmm._internal.params."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal import params as params_mod


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


@jdc.pytree_dataclass
class _ThreeField:
    a: float
    b: float
    c: float


@jdc.pytree_dataclass
class _NestedOuter:
    inner: _EulerParams
    scale: float


class TestFlattenParams:
    def test_simple_flatten(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        flat, treedef = params_mod.flatten_params(p)
        assert flat.shape == (2,)
        # PyTree-traversal order for jdc dataclasses is field-declaration order.
        assert float(flat[0]) == pytest.approx(0.95)
        assert float(flat[1]) == pytest.approx(2.0)

    def test_three_field(self):
        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        flat, _ = params_mod.flatten_params(p)
        assert flat.shape == (3,)
        assert jnp.allclose(flat, jnp.array([1.0, 2.0, 3.0]))

    def test_jax_array_inputs(self):
        # Scalar JAX arrays should also work.
        p = _EulerParams(beta=jnp.asarray(0.95), gamma=jnp.asarray(2.0))
        flat, _ = params_mod.flatten_params(p)
        assert flat.shape == (2,)

    def test_rejects_non_scalar_leaf(self):
        # A leaf with shape (3,) should be rejected.
        @jdc.pytree_dataclass
        class Bad:
            vec: jnp.ndarray

        bad = Bad(vec=jnp.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="0-d scalars"):
            params_mod.flatten_params(bad)


class TestUnflattenParams:
    def test_round_trip_simple(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        flat, treedef = params_mod.flatten_params(p)
        restored = params_mod.unflatten_params(flat, treedef)
        assert isinstance(restored, _EulerParams)
        assert float(restored.beta) == pytest.approx(0.95)
        assert float(restored.gamma) == pytest.approx(2.0)

    def test_round_trip_three_field(self):
        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        flat, treedef = params_mod.flatten_params(p)
        restored = params_mod.unflatten_params(flat, treedef)
        assert isinstance(restored, _ThreeField)
        assert float(restored.a) == pytest.approx(1.0)
        assert float(restored.b) == pytest.approx(2.0)
        assert float(restored.c) == pytest.approx(3.0)

    def test_round_trip_modified_values(self):
        # The point of the flat representation: optimisers update it
        # and we reconstruct with new values.
        p = _EulerParams(beta=0.0, gamma=0.0)
        _, treedef = params_mod.flatten_params(p)
        new_flat = jnp.array([0.97, 1.5])
        restored = params_mod.unflatten_params(new_flat, treedef)
        assert float(restored.beta) == pytest.approx(0.97)
        assert float(restored.gamma) == pytest.approx(1.5)

    def test_rejects_non_1d(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        _, treedef = params_mod.flatten_params(p)
        with pytest.raises(ValueError, match="must be 1-D"):
            params_mod.unflatten_params(jnp.zeros((2, 2)), treedef)

    def test_rejects_wrong_length(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        _, treedef = params_mod.flatten_params(p)
        with pytest.raises(ValueError, match="expects 2 leaves"):
            params_mod.unflatten_params(jnp.zeros(3), treedef)


class TestParamNames:
    def test_simple(self):
        p = _EulerParams(beta=0.95, gamma=2.0)
        assert params_mod.param_names(p) == ["beta", "gamma"]

    def test_declaration_order(self):
        p = _ThreeField(a=1.0, b=2.0, c=3.0)
        assert params_mod.param_names(p) == ["a", "b", "c"]

    def test_matches_flatten_leaf_order(self):
        # The promise of param_names: its order matches flatten_params'
        # leaf order. Verify on a known case.
        p = _EulerParams(beta=0.95, gamma=2.0)
        names = params_mod.param_names(p)
        flat, _ = params_mod.flatten_params(p)
        # Reconstruct a dict {name: value} and verify it matches the
        # original (under float32 precision).
        observed = {name: float(flat[i]) for i, name in enumerate(names)}
        assert observed["beta"] == pytest.approx(0.95)
        assert observed["gamma"] == pytest.approx(2.0)
        assert list(observed.keys()) == ["beta", "gamma"]

    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError, match="dataclass instance"):
            params_mod.param_names({"beta": 0.95})  # plain dict

    def test_rejects_nested_dataclass(self):
        nested = _NestedOuter(inner=_EulerParams(beta=0.95, gamma=2.0), scale=1.0)
        with pytest.raises(NotImplementedError, match="Nested"):
            params_mod.param_names(nested)


class TestJaxCompatibility:
    """Confirm flatten/unflatten work inside jit'd functions."""

    def test_can_jit(self):
        p_init = _EulerParams(beta=0.95, gamma=2.0)
        _, treedef = params_mod.flatten_params(p_init)

        @jax.jit
        def double_then_reconstruct(flat):
            return params_mod.unflatten_params(flat * 2.0, treedef)

        flat_init, _ = params_mod.flatten_params(p_init)
        doubled = double_then_reconstruct(flat_init)
        assert float(doubled.beta) == pytest.approx(1.9)
        assert float(doubled.gamma) == pytest.approx(4.0)

    def test_can_grad(self):
        # A scalar loss as a function of the flat param vector should
        # be differentiable.
        p = _EulerParams(beta=0.95, gamma=2.0)
        flat_init, treedef = params_mod.flatten_params(p)

        def loss(flat):
            params = params_mod.unflatten_params(flat, treedef)
            return params.beta**2 + 2 * params.gamma**2

        g = jax.grad(loss)(flat_init)
        # d/d_beta = 2 beta = 1.9; d/d_gamma = 4 gamma = 8.0
        assert jnp.allclose(g, jnp.array([1.9, 8.0]))
