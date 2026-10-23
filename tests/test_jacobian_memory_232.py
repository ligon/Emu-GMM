"""Numerical and API checks for the opt-in bounded Jacobian path (#232)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import build_estimator, estimate
from emu_gmm.covariance import AnalyticalCovariance, IIDCovariance, SyntheticCovariance
from emu_gmm.manifolds import Positive
from emu_gmm.measures import AnalyticalMeasure, EmpiricalMeasure, SyntheticMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.penalty import TikhonovPenalty


@jdc.pytree_dataclass
class _Theta:
    a: float
    b: float


def _moments(_model, theta):
    return jnp.array([theta.a**2 - 1.0, theta.b - 2.0, theta.a + theta.b - 3.0])


def _covariance(_model, _theta):
    return jnp.eye(3)


def test_bounded_estimate_matches_default_on_nonlinear_moments():
    measure = AnalyticalMeasure(expectation_fn=_moments)
    covariance = AnalyticalCovariance(covariance_fn=_covariance)
    kw = dict(measure=measure, covariance=covariance, theta_init=_Theta(0.7, 1.7))
    standard = estimate(lambda x, theta: x, **kw)
    bounded = estimate(lambda x, theta: x, jacobian_memory="bounded", **kw)
    assert bounded.converged
    assert np.allclose(
        [bounded.theta_hat.a, bounded.theta_hat.b],
        [standard.theta_hat.a, standard.theta_hat.b],
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.allclose(bounded.moment_jacobian, standard.moment_jacobian, rtol=1e-9)
    assert float(bounded.objective_value) == pytest.approx(
        float(standard.objective_value), abs=1e-8
    )


def test_bounded_keeps_supplied_analytical_jacobian():
    def user_jacobian(_model, theta):
        return jnp.array([[2 * theta.a, 0.0], [0.0, 1.0], [1.0, 1.0]])

    measure = AnalyticalMeasure(expectation_fn=_moments, jacobian_fn=user_jacobian)
    result = estimate(
        lambda x, theta: x,
        measure=measure,
        covariance=AnalyticalCovariance(covariance_fn=_covariance),
        theta_init=_Theta(0.7, 1.7),
        jacobian_memory="bounded",
    )
    expected = user_jacobian(None, result.theta_hat)
    assert np.array_equal(np.asarray(result.moment_jacobian), np.asarray(expected))


def test_bounded_factory_accepts_fresh_same_structure_measure():
    calls = {"n": 0}

    def counting_moments(model, theta):
        calls["n"] += 1
        return _moments(model, theta)

    measure = AnalyticalMeasure(expectation_fn=counting_moments)
    run = build_estimator(
        lambda x, theta: x,
        measure=measure,
        covariance=AnalyticalCovariance(covariance_fn=_covariance),
        theta_init=_Theta(0.7, 1.7),
        jacobian_memory="bounded",
    )
    one = run(_Theta(0.7, 1.7), measure)
    first_calls = calls["n"]
    two = run(_Theta(0.8, 1.8), AnalyticalMeasure(expectation_fn=counting_moments))
    assert np.allclose(
        [one.theta_hat.a, one.theta_hat.b],
        [two.theta_hat.a, two.theta_hat.b],
        atol=1e-7,
    )
    assert calls["n"] == first_calls


def test_bounded_ragged_cu_with_penalty_matches_default():
    rng = np.random.default_rng(232)
    x = rng.normal(loc=[1.0, 2.0, 3.0], scale=0.3, size=(96, 3))
    mask = np.ones((96, 3))
    mask[::3, 1] = 0.0
    mask[::2, 2] = 0.0
    x[mask == 0] = np.nan
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.ones(96)
    )

    def model(row, theta):
        return jnp.array(
            [row[0] - theta.a, row[1] - theta.b, row[2] - theta.a - theta.b]
        )

    kw = dict(
        measure=measure,
        covariance=IIDCovariance(),
        theta_init=_Theta(0.7, 1.7),
        penalty=TikhonovPenalty(c=jnp.asarray(0.01)),
    )
    standard = estimate(model, **kw)
    bounded = estimate(model, jacobian_memory="bounded", **kw)
    assert bounded.converged
    assert np.allclose(
        [bounded.theta_hat.a, bounded.theta_hat.b],
        [standard.theta_hat.a, standard.theta_hat.b],
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.allclose(bounded.moment_jacobian, standard.moment_jacobian, rtol=1e-9)
    assert float(bounded.objective_value) == pytest.approx(
        float(standard.objective_value), rel=1e-5, abs=1e-7
    )


def test_bounded_synthetic_keeps_crn_draws():
    def sampler(key, theta):
        del theta
        return jax.random.normal(key, (128, 2))

    def model(row, theta):
        return jnp.array(
            [
                theta.a - row[0],
                theta.b * row[1] ** 2 - 1.0,
                theta.a + theta.b - row[0] - row[1] ** 2,
            ]
        )

    measure = SyntheticMeasure(key=jax.random.PRNGKey(232), n_sim=128, sampler=sampler)
    kw = dict(
        measure=measure,
        covariance=SyntheticCovariance(),
        theta_init=_Theta(0.2, 0.7),
    )
    standard = estimate(model, **kw)
    bounded = estimate(model, jacobian_memory="bounded", **kw)
    assert bounded.converged
    assert np.allclose(
        [bounded.theta_hat.a, bounded.theta_hat.b],
        [standard.theta_hat.a, standard.theta_hat.b],
        atol=1e-6,
    )
    assert np.allclose(bounded.moment_jacobian, standard.moment_jacobian, rtol=1e-9)


@jdc.pytree_dataclass
class _Scale:
    sigma: jnp.ndarray

    __emu_manifolds__ = {"sigma": Positive()}


def _scale_model(x, theta):
    return jnp.array([x[0] ** 2 - theta.sigma**2, x[0] ** 4 - 3 * theta.sigma**4])


def test_bounded_positive_manifold_matches_default():
    draws = np.random.default_rng(232).normal(0.0, 1.5, size=(128, 1))
    measure = EmpiricalMeasure(
        x=jnp.asarray(draws), mask=jnp.ones((128, 2)), weights=jnp.ones(128)
    )
    kw = dict(
        measure=measure,
        covariance=IIDCovariance(),
        theta_init=_Scale(sigma=jnp.asarray(1.0)),
    )
    standard = estimate(_scale_model, **kw)
    bounded = estimate(_scale_model, jacobian_memory="bounded", **kw)
    assert bounded.converged
    assert float(bounded.theta_hat.sigma) == pytest.approx(
        float(standard.theta_hat.sigma), rel=1e-6
    )
    assert np.allclose(bounded.moment_jacobian, standard.moment_jacobian, rtol=1e-8)


def test_unknown_or_unsupported_bounded_options_refused():
    with pytest.raises(ValueError, match="jacobian_memory"):
        optimistix_lm(jacobian_memory="unknown")

    measure = AnalyticalMeasure(expectation_fn=_moments)
    kw = dict(
        measure=measure,
        covariance=AnalyticalCovariance(covariance_fn=_covariance),
        theta_init=_Theta(0.7, 1.7),
    )
    with pytest.raises(ValueError, match="jacobian_memory"):
        build_estimator(lambda x, theta: x, jacobian_memory="unknown", **kw)
    with pytest.raises(TypeError, match="requires optimistix_lm"):
        build_estimator(
            lambda x, theta: x,
            jacobian_memory="bounded",
            optimizer=lambda fn, x: (x, None),
            **kw,
        )
    run = build_estimator(lambda x, theta: x, jacobian_memory="bounded", **kw)
    with pytest.raises(TypeError, match="requires a built-in Measure"):
        run(_Theta(0.7, 1.7), object())


def test_lsmr_reports_a_finite_rank_poor_solution():
    def residual(x):
        error = x[0] + x[1] - 1.0
        return jnp.array([error, 2.0 * error])

    solution, info = optimistix_lm(jacobian_memory="bounded")(residual, jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(solution)))
    assert float(info.final_objective) < 1e-10
    assert bool(info.done)


def test_bounded_lm_jit_vmap_with_traced_args():
    optimizer = optimistix_lm(jacobian_memory="bounded")

    def residual(x, target):
        return jnp.array([x[0] - target, x[1] ** 2 - 1.0])

    def fit(target):
        solution, info = optimizer(residual, jnp.array([0.0, 0.8]), args=target)
        return solution, info.done

    solutions, done = jax.jit(jax.vmap(fit))(jnp.array([-0.5, 0.0, 0.5]))
    assert bool(jnp.all(done))
    assert np.allclose(np.asarray(solutions[:, 0]), [-0.5, 0.0, 0.5], atol=1e-6)
    assert np.allclose(np.asarray(solutions[:, 1]), 1.0, atol=1e-6)
