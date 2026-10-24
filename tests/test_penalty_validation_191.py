"""Evaluated native coefficients fail loudly without restricting AD carriers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import EmpiricalMeasure, IIDCovariance, TikhonovPenalty, estimate

_BAD_VALUES = [-1.0, float("nan"), float("inf"), -float("inf")]


def _evaluate(c):
    return TikhonovPenalty(c=c).penalty((jnp.array(2.0),))


@pytest.mark.parametrize("c", _BAD_VALUES)
def test_bad_concrete_coefficient_raises_on_evaluation(c):
    penalty = TikhonovPenalty(c=c)
    with pytest.raises(ValueError, match="finite and non-negative"):
        penalty.penalty((jnp.array(2.0),))


@pytest.mark.parametrize("c", _BAD_VALUES)
@pytest.mark.parametrize("transform", ["jit", "vmap", "jit_vmap"])
def test_bad_traced_coefficient_raises_instead_of_returning_nan(c, transform):
    if transform == "jit":
        run = jax.jit(_evaluate)
        coefficient = jnp.array(c)
    else:
        run = jax.vmap(_evaluate)
        if transform == "jit_vmap":
            run = jax.jit(run)
        coefficient = jnp.array([1.0, c, 2.0])
    error = jax.errors.JaxRuntimeError if "jit" in transform else ValueError
    with pytest.raises(error, match="TikhonovPenalty.c"):
        jax.block_until_ready(run(coefficient))


@pytest.mark.parametrize("differentiate", [False, True])
def test_bad_coefficient_still_raises_when_penalty_arithmetic_is_zero(differentiate):
    def at_origin(c):
        return TikhonovPenalty(c=c).penalty((jnp.array(0.0),))

    run = jax.jit(jax.grad(at_origin) if differentiate else at_origin)
    with pytest.raises(jax.errors.JaxRuntimeError, match="non-negative"):
        jax.block_until_ready(run(jnp.array(-1.0)))


@pytest.mark.parametrize("c", [np.array([1.0]), np.array(1.0 + 0.0j), True, "1", None])
def test_shape_and_real_dtype_are_validated_eagerly(c):
    with pytest.raises(ValueError, match="real numeric scalar"):
        _evaluate(c)


@pytest.mark.parametrize("c", [jnp.array([1.0]), jnp.array(1.0j), jnp.array(True)])
def test_shape_and_dtype_guards_are_static_under_jit(c):
    with pytest.raises(ValueError, match="real numeric scalar"):
        jax.jit(_evaluate)(c)


def test_valid_dynamic_sweeps_share_one_trace_and_preserve_values():
    traces = []

    @jax.jit
    def run(c):
        traces.append(True)
        return _evaluate(c)

    for c in [0.0, 0.5, 1.0, 3.0]:
        assert float(run(jnp.array(c))) == 4.0 * c
    assert len(traces) == 1
    values = jax.jit(jax.vmap(_evaluate))(jnp.array([0.0, 0.5, 3.0]))
    np.testing.assert_array_equal(values, [0.0, 2.0, 12.0])


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.float64, jnp.bfloat16])
def test_real_numeric_scalar_dtypes_are_supported(dtype):
    coefficient = jnp.array(2, dtype=dtype)
    assert float(_evaluate(coefficient)) == 8.0
    assert float(jax.jit(_evaluate)(coefficient)) == 8.0


def test_gradients_with_respect_to_theta_and_coefficient_stay_native():
    def value(c, theta):
        return TikhonovPenalty(c=c).penalty((theta[0], theta[1]))

    c = jnp.array(0.5)
    theta = jnp.array([2.0, 3.0])
    grad_c, grad_theta = jax.jit(jax.grad(value, argnums=(0, 1)))(c, theta)
    assert float(grad_c) == 13.0
    np.testing.assert_array_equal(grad_theta, 2 * c * theta)
    assert float(jax.jit(jax.grad(jax.grad(lambda c: value(c, theta))))(c)) == 0.0


@pytest.mark.parametrize("traced", [False, True])
def test_negative_cotangent_can_be_reconstructed_without_evaluation(traced):
    primal = TikhonovPenalty(c=jnp.array(1.0))

    def negative_value(penalty):
        return -penalty.penalty((jnp.array(2.0),))

    differentiate = jax.grad(negative_value)
    if traced:
        differentiate = jax.jit(differentiate)
    cotangent = differentiate(primal)
    assert float(cotangent.c) == -4.0
    assert float(primal.penalty((jnp.array(2.0),))) == 4.0
    with pytest.raises(ValueError, match="non-negative"):
        cotangent.penalty((jnp.array(2.0),))


@pytest.mark.parametrize("c", [1.0, np.array(1.0), jnp.array(1.0)])
def test_fixed_captured_coefficient_adds_no_debug_callback(c):
    penalty = TikhonovPenalty(c=c)
    closed = jax.make_jaxpr(lambda theta: penalty.penalty((theta,)))(jnp.array(2.0))
    assert not any(eqn.primitive.name == "debug_callback" for eqn in closed.jaxpr.eqns)
    assert not closed.jaxpr.effects


def test_dynamic_callbacks_run_in_scalar_batch_order(monkeypatch):
    import emu_gmm.penalty as penalty_module

    observed = []
    validate = penalty_module._check_coefficient_value

    def record_and_validate(c):
        observed.append(float(c))
        validate(c)

    monkeypatch.setattr(penalty_module, "_check_coefficient_value", record_and_validate)
    run = jax.jit(jax.vmap(lambda c: _evaluate(c)))
    jax.block_until_ready(run(jnp.array([1.0, 2.0, 3.0])))
    jax.effects_barrier()
    assert observed == [1.0, 2.0, 3.0]


def test_abstract_and_placeholder_carriers_do_not_trigger_value_conversion():
    _, treedef = jax.tree_util.tree_flatten(TikhonovPenalty(c=jnp.array(1.0)))
    placeholder = object()
    assert jax.tree_util.tree_unflatten(treedef, [placeholder]).c is placeholder
    abstract = TikhonovPenalty(c=jax.ShapeDtypeStruct((), jnp.float64))
    shape = jax.eval_shape(lambda p: p.penalty((jnp.array(2.0),)), abstract)
    assert shape.shape == ()
    assert shape.dtype == jnp.float64


def test_invalid_fixed_coefficient_is_reported_by_eager_estimate():
    @jdc.pytree_dataclass
    class Theta:
        value: float

    measure = EmpiricalMeasure.from_arrays(np.array([[1.0], [2.0], [3.0]]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        estimate(
            lambda row, theta: jnp.array([row[0] - theta.value]),
            measure,
            covariance=IIDCovariance(),
            parameters=Theta(2.0),
            penalty=TikhonovPenalty(c=jnp.array(-1.0)),
        )
