"""Positive trial-domain acceptance and damping-search guards (#180)."""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, Positive, riemannian_lm


@jdc.pytree_dataclass
class _Legacy:
    x: jnp.ndarray
    __emu_manifolds__ = {"x": Positive()}


@jdc.pytree_dataclass
class _Wrapped:
    x: ManifoldLeaf


@jdc.pytree_dataclass
class _Mixed:
    x: ManifoldLeaf
    y: ManifoldLeaf


class _PositiveBlock(Positive):
    """Explicit vector subclass exercises the non-scalar static-plan branch."""

    @property
    def ambient_shape(self):
        return (2,)

    @property
    def dimension(self):
        return 2

    def inner_product(self, point, u, v):
        return jnp.sum(u * v / point**2)


def _point(kind, initial=1.0):
    if kind == "legacy":
        return _Legacy(jnp.asarray(initial))
    if kind == "scalar":
        return _Wrapped(ManifoldLeaf(jnp.asarray(initial), Positive()))
    if kind == "block":
        return _Wrapped(ManifoldLeaf(jnp.asarray([initial, 2.0]), _PositiveBlock()))
    return _Mixed(
        ManifoldLeaf(jnp.asarray(initial), Positive()),
        ManifoldLeaf(jnp.asarray(-2.0), Euclidean()),
    )


def _value(theta):
    return jnp.asarray(getattr(theta.x, "array", theta.x))


def _kernel(flat, target):
    return jnp.atleast_1d(jnp.arctan(1000 * (flat[0] - target)))


@pytest.mark.parametrize("kind", ["legacy", "scalar", "block", "mixed"])
def test_decreasing_cost_does_not_accept_underflowed_positive_trial(kind):
    initial = _point(kind)
    spec = manifold_spec_from_params(initial)
    result, info = riemannian_lm(max_steps=1)(
        lambda flat: _kernel(flat, 0.001), initial, spec
    )
    # The unconstrained first LM step makes exp(d/x) underflow. Its finite
    # residual improves, but zero is outside Positive and must be rejected.
    np.testing.assert_array_equal(_value(result), _value(initial))
    assert np.isfinite(_value(result)).all() and (_value(result) > 0).all()
    assert int(info.steps) == 1
    assert not bool(info.done) and not bool(info.stagnated)
    assert info.status == "max_iterations"
    if kind == "mixed":
        np.testing.assert_array_equal(result.y.array, -2.0)


def test_finite_decreasing_residual_does_not_accept_positive_overflow():
    initial = _point("scalar", initial=0.001)
    result, info = riemannian_lm(max_steps=1)(
        lambda flat: jnp.atleast_1d(jnp.arctan(1000 * (flat[0] - 1)) - 0.5),
        initial,
        manifold_spec_from_params(initial),
    )
    np.testing.assert_array_equal(_value(result), _value(initial))
    assert not bool(info.done) and not bool(info.stagnated)


def test_rejected_zero_trial_continues_the_existing_damping_search():
    initial = _point("scalar")
    result, info = riemannian_lm(max_steps=2)(
        lambda flat: _kernel(flat, 0.001), initial, manifold_spec_from_params(initial)
    )
    np.testing.assert_array_equal(_value(result), _value(initial))
    assert int(info.steps) == 2
    assert not bool(info.done) and not bool(info.stagnated)


def test_healthy_interior_root_still_recovers():
    initial = _point("scalar", initial=0.01)
    result, info = riemannian_lm(rtol=1e-10, atol=1e-10, max_steps=200)(
        lambda flat: _kernel(flat, 0.001), initial, manifold_spec_from_params(initial)
    )
    assert bool(info.done) and not bool(info.stagnated)
    assert np.isfinite(_value(result)) and _value(result) > 0
    np.testing.assert_allclose(_value(result), 0.001, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("max_steps", [1, 2, 20])
def test_traced_args_preserve_rejected_trial_and_legacy_parity(max_steps):
    initial = _point("scalar")
    spec = manifold_spec_from_params(initial)
    solver = riemannian_lm(max_steps=max_steps)
    old, old_info = solver(lambda flat: _kernel(flat, 0.001), initial, spec)
    traced, traced_info = jax.jit(
        lambda target: solver(_kernel, initial, spec, args=target)
    )(jnp.asarray(0.001))
    np.testing.assert_array_equal(_value(traced), _value(old))
    np.testing.assert_array_equal(traced_info.done, old_info.done)
    np.testing.assert_array_equal(traced_info.stagnated, old_info.stagnated)
    np.testing.assert_array_equal(traced_info.steps, old_info.steps)
    assert np.isfinite(_value(traced)) and _value(traced) > 0
