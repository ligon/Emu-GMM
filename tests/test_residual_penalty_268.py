"""Known finite quadratic optimum and optimization-only penalty rows (#268)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    Euclidean,
    Fixed,
    ManifoldLeaf,
    ParameterSpace,
    ResidualPenalty,
    TikhonovPenalty,
    build_estimator,
    estimate,
    on,
    riemannian_lm,
)
from emu_gmm.penalty import PenaltyStrategy


class Point(ParameterSpace):
    x: jax.Array = on(Euclidean(1), default=jnp.array([0.0]))
    s: jax.Array = on(Euclidean(2), default=jnp.array([0.5, 1.0]))


def _point(values):
    return Point(
        x=ManifoldLeaf(jnp.asarray(values[:1]), Euclidean(1)),
        s=ManifoldLeaf(jnp.asarray(values[1:]), Euclidean(2)),
    )


def _flat(theta):
    return jnp.concatenate([theta.x.array, theta.s.array])


def _base(values, eps=0.01, delta=0.01, target=1.0):
    x, y, z = values
    return jnp.array([eps * x + y - target, z - target, delta * y])


def _full_objective(values, eps=0.01, delta=0.01, target=1.0):
    return jnp.sum(_base(values, eps, delta, target) ** 2) + jnp.sum(values[1:] ** 2)


def _problem(eps=0.01, delta=0.01, target=1.0):
    def model(row, theta):
        del row
        return _base(_flat(theta), eps, delta, target)

    measure = AnalyticalMeasure(expectation_fn=lambda model, theta: model(None, theta))
    covariance = AnalyticalCovariance(covariance_fn=lambda model, theta: jnp.eye(3))
    return model, measure, covariance


def _fit(penalty, *, eps=0.01, delta=0.01, zero=False, cap=50):
    model, measure, covariance = _problem(eps, delta, target=0.0 if zero else 1.0)
    return estimate(
        model,
        measure,
        parameters=_point(jnp.zeros(3)) if zero else Point.point(),
        covariance=covariance,
        weighting=Fixed(L0=jnp.eye(3)),
        penalty=penalty,
        optimizer=riemannian_lm(rtol=1e-9, atol=1e-9, max_steps=cap),
    )


@pytest.mark.parametrize(
    "eps,delta", [(0.01, 0.01), (0.01, 0.1), (0.1, 0.01), (0.1, 0.1)]
)
def test_reported_quadratic_converges_and_keeps_data_inference_separate(eps, delta):
    penalty = ResidualPenalty(lambda theta: theta.s.array)
    result = _fit(penalty, eps=eps, delta=delta)
    values = _flat(result.theta_hat)
    assert result.converged
    assert result.iterations < 40
    np.testing.assert_allclose(values, [1 / eps, 0.0, 0.5], atol=1e-6, rtol=0)
    expected_gradient = jax.grad(lambda v: _full_objective(v, eps, delta) / 2)(values)
    np.testing.assert_allclose(
        result.gradient, expected_gradient, atol=1e-13, rtol=1e-11
    )
    assert np.linalg.norm(result.gradient) < 1e-8
    assert float(result.diagnostics.final_objective_full) == pytest.approx(
        0.5, abs=1e-12
    )
    assert float(result.objective_value) == pytest.approx(0.25, abs=1e-9)
    assert float(result.diagnostics.final_objective_data) == float(
        result.objective_value
    )
    assert float(result.diagnostics.optimizer_info.final_objective) == pytest.approx(
        0.25
    )
    assert (
        result.n_overid == 0
    )  # Three data moments, three parameters, no penalty rows.
    np.testing.assert_array_equal(result.moment_covariance, np.eye(3))
    G = np.array([[eps, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, delta, 0.0]])
    np.testing.assert_array_equal(result.moment_jacobian, G)
    np.testing.assert_array_equal(result.gn_hessian, G.T @ G)
    assert result.diagnostics.moment_residual.array.shape == (3,)
    assert isinstance(penalty, PenaltyStrategy)


@pytest.mark.parametrize(
    "values", [[0.0, 0.5, 1.0], [100.0, 0.0, 0.5], [2.0, -0.2, 0.3]]
)
def test_scalar_and_residual_objectives_and_gradients_agree(values):
    values = jnp.array(values)
    penalty = ResidualPenalty(lambda theta: theta.s.array)

    def actual(v):
        return jnp.sum(_base(v) ** 2) + penalty.penalty(_point(v))

    assert float(actual(values)) == pytest.approx(
        float(_full_objective(values)), abs=1e-15
    )
    np.testing.assert_allclose(
        jax.grad(actual)(values), jax.grad(_full_objective)(values)
    )


class ScalarPenalty:
    def penalty(self, theta):
        return jnp.sum(theta.s.array**2)

    def gradient(self, theta):
        return jax.grad(self.penalty)(theta)


def test_scalar_only_protocol_retains_correct_objective_gradient_and_failure_status():
    result = _fit(ScalarPenalty(), cap=50)
    assert not result.converged
    assert result.iterations == 50
    values = _flat(result.theta_hat)
    np.testing.assert_allclose(
        result.gradient,
        jax.grad(lambda v: _full_objective(v) / 2)(values),
        rtol=1e-11,
        atol=1e-13,
    )
    assert float(result.diagnostics.final_objective_full) == pytest.approx(
        float(_full_objective(values)), rel=1e-13
    )


def test_zero_penalty_optimum_has_finite_derivatives_and_converges_in_one_step():
    result = _fit(ResidualPenalty(lambda theta: theta.s.array), zero=True)
    assert result.converged
    assert result.iterations == 1
    np.testing.assert_array_equal(_flat(result.theta_hat), np.zeros(3))
    np.testing.assert_array_equal(result.gradient, np.zeros(3))
    assert float(result.diagnostics.final_objective_full) == 0.0


@pytest.mark.parametrize("c", [0.0, 0.5, 3.0])
@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
def test_residual_ridge_keeps_full_curvature_including_at_zero(c, values):
    penalty = ResidualPenalty(lambda v: jnp.sqrt(c) * v)
    values = jnp.array(values)
    residuals = jax.jit(penalty.residuals)
    J = jax.jacfwd(residuals)(values)
    np.testing.assert_allclose(J.T @ J, c * np.eye(3), atol=1e-15)
    assert np.all(np.isfinite(J))
    assert float(jnp.sum(residuals(values) ** 2)) == pytest.approx(
        c * float(jnp.sum(values**2))
    )
    np.testing.assert_allclose(jax.grad(penalty.penalty)(values), 2 * c * values)


@pytest.mark.parametrize(
    "value", [jnp.ones((1, 2)), jnp.array(1.0), jnp.array([1.0j]), jnp.array([True])]
)
def test_residual_contract_rejects_wrong_shape_or_nonreal_values(value):
    penalty = ResidualPenalty(lambda theta: value)
    with pytest.raises(ValueError, match="one-dimensional real array"):
        jax.jit(penalty.penalty)(jnp.zeros(2))


def test_empty_residual_penalty_is_zero():
    penalty = ResidualPenalty(lambda theta: jnp.empty(0))
    assert float(penalty.penalty(jnp.ones(2))) == 0.0
    np.testing.assert_array_equal(penalty.gradient(jnp.ones(2)), np.zeros(2))


def test_native_factory_accepts_residual_penalty_under_jit_and_vmap():
    model, measure, covariance = _problem()
    run = build_estimator(
        model,
        measure=measure,
        parameters=Point.point(),
        covariance=covariance,
        weighting=Fixed(L0=jnp.eye(3)),
        penalty=ResidualPenalty(lambda theta: theta.s.array),
        optimizer=riemannian_lm(rtol=1e-9, atol=1e-9, max_steps=50),
    )

    def solve(values):
        result = run(_point(values), measure)
        return _flat(result.theta_hat), result.converged

    points, converged = jax.jit(jax.vmap(solve))(
        jnp.array([[0.0, 0.5, 1.0], [1.0, -0.2, 0.3]])
    )
    np.testing.assert_allclose(points, [[100.0, 0.0, 0.5]] * 2, atol=1e-6, rtol=0)
    assert np.all(converged)


@jdc.pytree_dataclass
class ScalarPoint:
    x: float
    y: float
    z: float


def test_affine_residual_penalty_works_with_declared_linear_solver():
    target = jnp.array([2.0, -1.0, 0.5])

    def model(row, theta):
        del row
        return jnp.array([theta.x, theta.y, theta.z]) - target

    result = estimate(
        model,
        AnalyticalMeasure(expectation_fn=lambda model, theta: model(None, theta)),
        parameters=ScalarPoint(-0.5, 2.0, 1.0),
        covariance=AnalyticalCovariance(covariance_fn=lambda model, theta: jnp.eye(3)),
        weighting=Fixed(L0=jnp.eye(3)),
        penalty=ResidualPenalty(
            lambda theta: jnp.sqrt(0.5) * jnp.array([theta.x, theta.y, theta.z])
        ),
        linear=True,
    )
    assert result.converged
    np.testing.assert_allclose(
        [result.theta_hat.x, result.theta_hat.y, result.theta_hat.z],
        target / 1.5,
        rtol=1e-12,
        atol=1e-12,
    )


def test_scalar_override_of_native_ridge_does_not_inherit_wrong_residuals():
    class LegacyRidge(TikhonovPenalty):
        def penalty(self, theta):
            return jnp.sum(theta.s.array**2)

    scalar = _fit(ScalarPenalty(), cap=1)
    inherited = _fit(LegacyRidge(c=jnp.array(20.0)), cap=1)
    np.testing.assert_array_equal(_flat(inherited.theta_hat), _flat(scalar.theta_hat))
    assert float(inherited.diagnostics.final_objective_full) == float(
        scalar.diagnostics.final_objective_full
    )
