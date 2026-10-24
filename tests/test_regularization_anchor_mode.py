"""Native ridge mode anchoring and scale-relative detection (ledger 24)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    DiagonalTikhonov,
    EmpiricalMeasure,
    build_estimator,
)
from emu_gmm.regularization import _DIAG_RTOL, _is_diagonal, _pd_floor
from emu_gmm.studies import replicate
from emu_gmm.types import OptimizerInfo


@jdc.pytree_dataclass
class _Theta:
    a: float


def _model(x, theta):
    del x
    return jnp.array([theta.a, 0.5 * theta.a])


class _CaptureOptimizer:
    def __call__(self, residual_fn, theta_init, *, args=None):
        self.residual = (
            (lambda theta: residual_fn(theta))
            if args is None
            else (lambda theta: residual_fn(theta, args))
        )
        return theta_init, OptimizerInfo(
            steps=0,
            final_objective=jnp.sum(self.residual(theta_init) ** 2),
            status="converged",
            backend="residual_probe",
            done=True,
        )


@pytest.mark.parametrize("scale", [1e-200, 1e-18, 1.0, 1e18, 1e300])
def test_detector_is_relative_for_correlated_diagonal_and_zero_matrices(scale):
    correlated = scale * jnp.array([[1.0, 0.2], [0.2, 2.0]])
    diagonal = scale * jnp.diag(jnp.array([1.0, 2.0]))
    assert not bool(_is_diagonal(correlated))
    assert bool(_is_diagonal(diagonal))
    assert bool(_is_diagonal(jnp.zeros((2, 2))))
    assert not bool(_is_diagonal(scale * jnp.array([[0.0, 1.0], [1.0, 0.0]])))
    assert bool(jax.jit(_is_diagonal)(diagonal))
    assert not bool(jax.jit(_is_diagonal)(correlated))


@pytest.mark.parametrize("scale", [1e-18, 1.0, 1e18])
def test_relative_threshold_and_explicit_modes_use_existing_ridge_formulas(scale):
    reg = DiagonalTikhonov()
    V = scale * jnp.array([[1.0, 0.2], [0.2, 2.0]])
    tau = jnp.asarray(0.1)
    canonical = V + tau * jnp.diag(jnp.diag(V))
    diagonal = V + tau * (jnp.trace(V) / 2) * jnp.eye(2)
    np.testing.assert_allclose(
        reg.apply_fixed_tau(V, tau), canonical, atol=0, rtol=1e-14
    )
    np.testing.assert_allclose(
        reg.apply_fixed_tau(V, tau, diagonal_mode=False), canonical, atol=0, rtol=1e-14
    )
    np.testing.assert_allclose(
        jax.jit(lambda v: reg.apply_fixed_tau(v, tau, diagonal_mode=jnp.asarray(True)))(
            V
        ),
        diagonal,
        atol=0,
        rtol=1e-14,
    )
    threshold = _DIAG_RTOL * 1.5 * scale
    assert bool(
        _is_diagonal(
            scale * jnp.diag(jnp.array([1.0, 2.0]))
            + jnp.array([[0.0, 0.9 * threshold], [0.9 * threshold, 0.0]])
        )
    )
    assert not bool(
        _is_diagonal(
            scale * jnp.diag(jnp.array([1.0, 2.0]))
            + jnp.array([[0.0, 1.1 * threshold], [1.1 * threshold, 0.0]])
        )
    )


def test_direct_apply_and_saturation_keep_existing_absolute_pd_floor():
    reg = DiagonalTikhonov(kappa_target=1e3)
    for V in (
        jnp.diag(jnp.array([1.0, 1e-8])),
        jnp.array([[1.0, 0.3], [0.3, 2.0]]),
        1e-18 * jnp.array([[1.0, 0.3], [0.3, 2.0]]),
        jnp.zeros((2, 2)),
    ):
        star, tau = reg.apply(V)
        np.testing.assert_allclose(
            star, reg.apply_fixed_tau(V, tau), atol=0, rtol=1e-14
        )
    assert float(_pd_floor(1e-18 * jnp.eye(2))) == pytest.approx(1e-12)
    assert bool(reg.tau_saturated(jnp.zeros((2, 2)), reg.apply(jnp.zeros((2, 2)))[1]))


@pytest.mark.parametrize("mode", [jnp.array([True, False]), 1, 0.5])
def test_explicit_mode_requires_scalar_boolean(mode):
    with pytest.raises(ValueError, match="scalar boolean"):
        DiagonalTikhonov().apply_fixed_tau(
            jnp.eye(2), jnp.asarray(0.1), diagonal_mode=mode
        )


def _threshold_covariance(model, theta):
    del model
    off = theta.a * _DIAG_RTOL
    return jnp.array([[1.0, off], [off, 1e-8]])


@pytest.mark.parametrize("anchor", [0.0, 1.0])
def test_actual_estimator_residual_is_smooth_across_diagonal_threshold(anchor):
    measure = AnalyticalMeasure(expectation_fn=lambda model, theta: _model(None, theta))
    covariance = AnalyticalCovariance(covariance_fn=_threshold_covariance)
    reg = DiagonalTikhonov(kappa_target=100.0)
    probe = _CaptureOptimizer()
    theta0 = _Theta(anchor)
    run = build_estimator(
        _model,
        measure=measure,
        covariance=covariance,
        regularization=reg,
        optimizer=probe,
        parameters=theta0,
    )
    run(theta0, measure)
    V0 = _threshold_covariance(None, theta0)
    _, tau = reg.apply(V0)
    mode = _is_diagonal(V0)
    assert bool(mode) == (anchor == 0.0)

    def reference(a):
        theta = _Theta(a)
        V = reg.apply_fixed_tau(
            _threshold_covariance(None, theta), tau, diagonal_mode=mode
        )
        return jax.scipy.linalg.solve_triangular(
            jnp.linalg.cholesky(V), _model(None, theta), lower=True
        )

    def objective(a):
        return 0.5 * jnp.sum(probe.residual(jnp.asarray([a])) ** 2)

    crossing = 0.5 * (1 + 1e-8)
    for a in (crossing - 1e-4, crossing, crossing + 1e-4):
        np.testing.assert_allclose(
            probe.residual(jnp.asarray([a])), reference(a), atol=1e-10, rtol=1e-12
        )
        step = 1e-6
        finite_difference = (objective(a + step) - objective(a - step)) / (2 * step)
        np.testing.assert_allclose(
            jax.grad(objective)(a), finite_difference, atol=1e-6, rtol=1e-7
        )
    left, right = float(objective(crossing - 1e-7)), float(objective(crossing + 1e-7))
    assert abs(right - left) < 1e-5 * max(1.0, abs(left))


class _DatasetCovariance:
    def covariance(self, model, theta, measure):
        del model, theta
        regime = measure.x[0, 0]
        return jnp.array(
            [[1.0, 0.69 * regime], [0.69 * regime, 1e-8 + (0.5 - 1e-8) * regime]]
        )


def _dataset(regime):
    return EmpiricalMeasure(
        x=jnp.full((2, 1), regime), mask=jnp.ones((2, 2)), weights=jnp.ones(2)
    )


def test_factory_template_mode_and_per_rep_fresh_modes_follow_existing_anchor_policy():
    reg = DiagonalTikhonov(kappa_target=10.0)
    covariance = _DatasetCovariance()
    theta = _Theta(0.0)
    template = _dataset(0.0)
    run = build_estimator(
        _model,
        measure=template,
        covariance=covariance,
        regularization=reg,
        optimizer=_CaptureOptimizer(),
        parameters=theta,
    )
    assert run._emu_gmm_factory_spec["regularization"] is reg
    _, template_tau = reg.apply(covariance.covariance(_model, theta, template))
    datasets = [_dataset(0.0), _dataset(1.0)]
    for dataset in datasets:
        result = run(theta, dataset)
        V = covariance.covariance(_model, theta, dataset)
        expected = reg.apply_fixed_tau(V, template_tau, diagonal_mode=True)
        np.testing.assert_allclose(
            result.weighting_matrix, jnp.linalg.inv(expected), atol=1e-10, rtol=1e-12
        )

    # The driver uses its existing factory spec to rebuild each fit, not a
    # wrapped native regularizer that has lost the per-fit mode decision.
    index = {"next": 0}

    def dgp(key):
        del key
        value = datasets[index["next"]]
        index["next"] += 1
        return value

    records = replicate(
        run,
        dgp,
        n_reps=2,
        key=jax.random.PRNGKey(24),
        theta_init=theta,
        anchor_per_rep=True,
        statistics={"weight": lambda result, _: result.weighting_matrix},
    )
    for i, dataset in enumerate(datasets):
        expected, tau = reg.apply(covariance.covariance(_model, theta, dataset))
        np.testing.assert_allclose(
            records.extra["weight"][i], jnp.linalg.inv(expected), atol=1e-10, rtol=1e-12
        )
        np.testing.assert_allclose(records.records.tau_realised[i], tau, atol=1e-12)


@pytest.mark.parametrize("subclass", [False, True])
def test_custom_regularizer_two_argument_override_is_not_bypassed(subclass):
    seen = []
    if subclass:

        class Custom(DiagonalTikhonov):
            def apply_fixed_tau(self, V, tau):
                seen.append(V.shape)
                return V + tau * jnp.eye(V.shape[0])

        reg = Custom(kappa_target=10.0)
    else:

        class Custom:
            def apply(self, V):
                return V + 0.2 * jnp.eye(V.shape[0]), jnp.asarray(0.2)

            def apply_fixed_tau(self, V, tau):
                seen.append(V.shape)
                return V + tau * jnp.eye(V.shape[0])

        reg = Custom()
    measure = _dataset(0.0)
    covariance = _DatasetCovariance()
    theta = _Theta(0.0)
    run = build_estimator(
        _model,
        measure=measure,
        covariance=covariance,
        regularization=reg,
        optimizer=_CaptureOptimizer(),
        parameters=theta,
    )
    result = run(theta, measure)
    _, tau = reg.apply(covariance.covariance(_model, theta, measure))
    expected = covariance.covariance(_model, theta, measure) + tau * jnp.eye(2)
    np.testing.assert_allclose(
        result.weighting_matrix, jnp.linalg.inv(expected), atol=1e-10
    )
    assert seen
