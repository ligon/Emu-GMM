"""Public native factory reuse and geometry guards (ledger Section 26)."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import EmpiricalMeasure, IIDCovariance, build_estimator
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, Positive, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.riemannian_tr import riemannian_tr
from emu_gmm.types import OptimizerInfo
from emu_gmm.weighting import ContinuouslyUpdated, Fixed, Identity, IteratedWeighting


@jdc.pytree_dataclass
class Point:
    factor: ManifoldLeaf
    level: ManifoldLeaf


_A = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
_TRIU = np.triu_indices(3)


def _point():
    return Point(
        ManifoldLeaf(1.01 * _A, PSDFixedRank(3, 2)),
        ManifoldLeaf(jnp.array([1.9]), Euclidean(1)),
    )


def _measure(seed, n=24):
    noise = np.random.default_rng(seed).normal(size=(n, 7))
    noise -= noise.mean(axis=0)
    target = np.r_[np.asarray(_A @ _A.T)[_TRIU], 2.0 + 0.01 * seed]
    return EmpiricalMeasure.from_arrays(target + 0.02 * noise)


class CountingModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, row, point):
        self.calls += 1
        gamma = point.factor.array @ point.factor.array.T
        return jnp.r_[gamma[_TRIU], point.level.array] - row


@pytest.mark.parametrize(
    "optimizer", [riemannian_lm(max_steps=80), riemannian_tr(max_steps=80)]
)
@pytest.mark.parametrize(
    "weighting",
    [
        ContinuouslyUpdated(),
        Identity(),
        Fixed(L0=jnp.eye(7)),
        IteratedWeighting(weighting_iterations=3),
    ],
)
def test_public_factory_reuses_native_solve_and_inference(
    optimizer, weighting, lm_advisory_trace_counts
):
    model = CountingModel()
    observed = lm_advisory_trace_counts(lambda: (model.calls,))
    run = build_estimator(
        model,
        measure=_measure(0),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=optimizer,
        weighting=weighting,
    )
    first = run(_point(), _measure(1))
    assert bool(first.converged)
    calls, _ = observed()
    second = run(_point(), _measure(2))
    assert bool(second.converged)
    assert observed()[0] == calls
    assert observed()[1] <= 1
    third = run(_point(), _measure(3))
    assert bool(third.converged)
    assert observed()[0] == calls
    assert observed()[1] <= 1
    for seed, result in ((1, first), (2, second)):
        np.testing.assert_allclose(
            result.theta_hat.level.array, [2 + 0.01 * seed], atol=1e-7
        )
        np.testing.assert_allclose(
            result.theta_hat.factor.array @ result.theta_hat.factor.array.T,
            _A @ _A.T,
            atol=1e-7,
        )
    assert not np.array_equal(first.moment_covariance, second.moment_covariance)


def test_native_factory_new_shape_compiles_once(lm_advisory_trace_counts):
    model = CountingModel()
    observed = lm_advisory_trace_counts(lambda: (model.calls,))
    run = build_estimator(
        model,
        measure=_measure(0),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=riemannian_lm(max_steps=80),
    )
    run(_point(), _measure(1))
    old = observed()[0][0]
    run(_point(), _measure(2, n=32))
    new = observed()[0][0]
    assert new > old
    run(_point(), _measure(3, n=32))
    assert observed()[0][0] == new
    assert observed()[1] <= 2  # at most one curvature trace per row-count shape


@jdc.pytree_dataclass
class ScalarPositive:
    sigma: float
    __emu_manifolds__ = {"sigma": Positive()}


def test_iterated_scalar_positive_retains_actual_geometry(monkeypatch):
    import emu_gmm.weighting as weighting_mod

    seen = []
    original = weighting_mod._gauge_invariant_signature

    def record(point):
        seen.append(point)
        return original(point)

    monkeypatch.setattr(weighting_mod, "_gauge_invariant_signature", record)

    def model(row, theta):
        return jnp.array([theta.sigma - row[0]])

    measure = EmpiricalMeasure.from_arrays(np.array([[1.0], [2.0], [3.0], [4.0]]))
    # Negative Euclidean Newton steps are not relevant to this fixture; the
    # actual geometric contract is recorded at the native call boundary below.
    from emu_gmm.manifolds.riemannian_lm import _RiemannianLM

    call = _RiemannianLM.__call__
    geometries = []

    def checked(self, residual_fn, theta_init, manifold_spec, *, args=None):
        geometries.append(manifold_spec)
        return call(self, residual_fn, theta_init, manifold_spec, args=args)

    monkeypatch.setattr(_RiemannianLM, "__call__", checked)
    run = build_estimator(
        model,
        measure=measure,
        covariance=IIDCovariance(),
        parameters=ScalarPositive(1.0),
        weighting=IteratedWeighting(weighting_iterations=3),
        optimizer=riemannian_lm(max_steps=80),
    )
    result = run(ScalarPositive(1.0), measure)
    assert bool(result.converged)
    assert all(isinstance(spec.leaf_specs[0].manifold, Positive) for spec in geometries)
    assert seen  # v2 args still uses the gauge-invariant/pytree stopping path
    assert float(result.theta_hat.sigma) == pytest.approx(2.5, abs=1e-8)


def test_transformed_native_factory_does_not_run_eager_advisory():
    run = build_estimator(
        CountingModel(),
        measure=_measure(0),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=riemannian_lm(max_steps=80),
    )

    def fit(measure):
        result = run(_point(), measure)
        assert result.diagnostics.optimizer_info.min_curvature is None
        return result.converged, result.theta_hat.level.array

    batch = jax.tree.map(lambda *xs: jnp.stack(xs), _measure(1), _measure(2))
    with warnings.catch_warnings(record=True) as records:
        done, levels = jax.jit(jax.vmap(fit))(batch)
    assert bool(jnp.all(done))
    np.testing.assert_allclose(levels[:, 0], [2.01, 2.02], atol=1e-7)
    assert not any("non-convex" in str(record.message) for record in records)


def test_unregistered_third_party_measure_retains_legacy_fallback():
    class MeasureWrapper:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

    model = CountingModel()
    run = build_estimator(
        model,
        measure=MeasureWrapper(_measure(0)),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=riemannian_lm(max_steps=80),
    )
    result = run(_point(), MeasureWrapper(_measure(1)))
    assert bool(result.converged)
    np.testing.assert_allclose(result.theta_hat.level.array, [2.01], atol=1e-7)


def test_third_party_manifold_optimizer_keeps_legacy_call():
    class CustomOptimizer:
        def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
            assert args is None
            return riemannian_lm(max_steps=80)(residual_fn, theta_init, manifold_spec)

    run = build_estimator(
        CountingModel(),
        measure=_measure(0),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=CustomOptimizer(),
    )
    result = run(_point(), _measure(1))
    assert bool(result.converged)


def test_iterated_args_stops_on_gamma_despite_rotated_factor(monkeypatch):
    from emu_gmm.manifolds.riemannian_lm import _RiemannianLM

    calls = []

    def rotate(self, residual_fn, theta_init, manifold_spec, *, args=None):
        assert args is not None
        calls.append(manifold_spec)
        rotated = Point(
            ManifoldLeaf(
                theta_init.factor.array @ jnp.array([[0.0, -1.0], [1.0, 0.0]]),
                theta_init.factor.manifold,
            ),
            theta_init.level,
        )
        return rotated, OptimizerInfo(
            steps=jnp.array(1),
            final_objective=jnp.array(0.0),
            status="converged",
            backend="test",
            done=jnp.array(True),
        )

    monkeypatch.setattr(_RiemannianLM, "__call__", rotate)
    run = build_estimator(
        CountingModel(),
        measure=_measure(0),
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=riemannian_lm(),
        weighting=IteratedWeighting(weighting_iterations=3),
    )
    result = run(_point(), _measure(1))
    assert bool(result.converged)
    assert len(calls) == 1


def test_eager_args_advisory_warns_and_reuses_probe():
    class Saddle:
        calls = 0

        def __call__(self, row, point):
            self.calls += 1
            d = jnp.r_[point.factor.array.ravel() - _A.ravel(), point.level.array - 2.0]
            quad = 3.0 + 0.5 * jnp.sum(jnp.r_[-jnp.ones(6), 1.0] * d**2)
            return jnp.r_[0.01 * d, quad] - row

    model = Saddle()

    def measure(seed):
        noise = np.random.default_rng(seed).normal(size=(24, 8))
        return EmpiricalMeasure.from_arrays(0.01 * (noise - noise.mean(axis=0)))

    point = Point(
        ManifoldLeaf(_A, PSDFixedRank(3, 2)),
        ManifoldLeaf(jnp.array([2.0]), Euclidean(1)),
    )
    run = build_estimator(
        model,
        measure=measure(0),
        covariance=IIDCovariance(),
        parameters=point,
        optimizer=riemannian_lm(max_steps=10),
        weighting=Identity(),
    )
    with pytest.warns(UserWarning, match="non-convex"):
        first = run(point, measure(1))
    calls = model.calls
    with pytest.warns(UserWarning, match="non-convex"):
        second = run(point, measure(2))
    assert model.calls == calls
    for result in (first, second):
        info = result.diagnostics.optimizer_info
        assert bool(result.converged)
        assert info.stalled_indefinite is True
        assert float(info.min_curvature) < -1.0


@pytest.mark.parametrize(
    "optimizer", [riemannian_lm(max_steps=80), riemannian_tr(max_steps=80)]
)
@pytest.mark.parametrize(
    "weighting", [ContinuouslyUpdated(), IteratedWeighting(weighting_iterations=3)]
)
def test_public_native_args_matches_legacy_manifold_fit(optimizer, weighting):
    class Legacy:
        def __call__(self, residual_fn, theta_init, manifold_spec):
            return optimizer(residual_fn, theta_init, manifold_spec)

    results = []
    for solver in (optimizer, Legacy()):
        run = build_estimator(
            CountingModel(),
            measure=_measure(0),
            covariance=IIDCovariance(),
            parameters=_point(),
            optimizer=solver,
            weighting=weighting,
        )
        results.append(run(_point(), _measure(1)))
    native, legacy = results
    assert bool(native.converged) and bool(legacy.converged)
    assert native.labels == legacy.labels
    for name in (
        "objective_value",
        "moment_jacobian",
        "moment_covariance",
        "weighting_matrix",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(legacy, name), rtol=1e-7, atol=1e-9
        )
    np.testing.assert_allclose(
        native.theta_hat.factor.array @ native.theta_hat.factor.array.T,
        legacy.theta_hat.factor.array @ legacy.theta_hat.factor.array.T,
        atol=1e-8,
    )
