"""One-shot compatibility and native factory reuse (ledger26 refinement)."""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    Identity,
    IIDCovariance,
    IteratedWeighting,
    build_estimator,
    estimate,
    linear_solver,
)
from emu_gmm.manifolds import Positive, riemannian_lm, riemannian_tr
from emu_gmm.manifolds.riemannian_lm import _RiemannianLM
from emu_gmm.manifolds.riemannian_tr import _RiemannianTR


@jdc.pytree_dataclass
class _Positive:
    sigma: float
    __emu_manifolds__ = {"sigma": Positive()}


def _mean(row, theta):
    return jnp.array([theta.sigma - row[0]])


@pytest.mark.parametrize("solver", ["lm", "tr"])
@pytest.mark.parametrize(
    "weighting", [ContinuouslyUpdated(), IteratedWeighting(weighting_iterations=3)]
)
def test_one_shot_native_preserves_legacy_route_and_result(
    solver, weighting, monkeypatch
):
    """Both native one-shot gates preserve the old optimizer call boundary."""
    cls = _RiemannianLM if solver == "lm" else _RiemannianTR
    native = riemannian_lm() if solver == "lm" else riemannian_tr()
    original = cls.__call__
    calls = []

    def observed(self, residual_fn, theta_init, manifold_spec, *, args=None):
        calls.append(args)
        assert isinstance(manifold_spec.leaf_specs[0].manifold, Positive)
        return original(self, residual_fn, theta_init, manifold_spec, args=args)

    monkeypatch.setattr(cls, "__call__", observed)

    class Legacy:
        def __call__(self, residual_fn, theta_init, manifold_spec):
            return native(residual_fn, theta_init, manifold_spec)

    measure = EmpiricalMeasure.from_arrays(np.array([[1.0], [2.0], [3.0], [4.0]]))
    kwargs = dict(
        covariance=IIDCovariance(),
        parameters=_Positive(1.0),
        weighting=weighting,
    )
    # Exercise auto-dispatch for the default LM as well as explicit native TR.
    actual = estimate(
        _mean, measure, optimizer=None if solver == "lm" else native, **kwargs
    )
    assert calls and all(arg is None for arg in calls)
    calls.clear()
    legacy = estimate(_mean, measure, optimizer=Legacy(), **kwargs)
    assert calls and all(arg is None for arg in calls)
    assert bool(actual.converged) and bool(legacy.converged)
    assert float(actual.theta_hat.sigma) == pytest.approx(2.5, abs=1e-7)
    np.testing.assert_array_equal(actual.theta_hat.sigma, legacy.theta_hat.sigma)
    assert int(actual.iterations) == int(legacy.iterations)
    assert (
        actual.diagnostics.optimizer_info.status
        == legacy.diagnostics.optimizer_info.status
    )
    for field in (
        "objective_value",
        "gradient",
        "moment_jacobian",
        "moment_covariance",
        "weighting_matrix",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(legacy, field))


class _ReducingPositive:
    """A theta-dependent global mean with independently known positive root."""

    def __init__(self):
        self.aux_calls = self.psi_calls = 0

    def aux(self, measure, theta):
        self.aux_calls += 1
        exposure = measure.mask[:, 0] * measure.weights
        xmean = jnp.sum(exposure * measure.x[:, 0]) / jnp.sum(exposure)
        return theta.sigma**2 * xmean

    def psi(self, row, theta, aux):
        del theta
        self.psi_calls += 1
        return jnp.array([aux - row[1]])


def _positive_measure(seed, root):
    x = np.linspace(0.7, 1.6, 8) + 0.02 * seed
    weights = 1.0 + 0.1 * seed * np.arange(8)
    noise = np.sin(np.arange(8) + seed)
    noise -= np.average(noise, weights=weights)
    # E_w[Y] = root**2 * E_w[X], even as outcomes and masses change.
    y = root**2 * x + 0.1 * noise
    return EmpiricalMeasure(
        x=jnp.asarray(np.column_stack([x, y])),
        mask=jnp.ones((8, 1)),
        weights=jnp.asarray(weights),
    )


@pytest.mark.parametrize("solver", ["lm", "tr"])
def test_positive_scalar_factory_reuses_aux_and_solve_for_fresh_measures(
    solver, monkeypatch
):
    cls = _RiemannianLM if solver == "lm" else _RiemannianTR
    native = riemannian_lm() if solver == "lm" else riemannian_tr()
    original = cls.__call__
    passed_measures = []

    def observed(self, residual_fn, theta_init, manifold_spec, *, args=None):
        passed_measures.append(args)
        assert isinstance(manifold_spec.leaf_specs[0].manifold, Positive)
        return original(self, residual_fn, theta_init, manifold_spec, args=args)

    monkeypatch.setattr(cls, "__call__", observed)
    model = _ReducingPositive()
    run = build_estimator(
        model,
        measure=_positive_measure(0, 1.3),
        covariance=IIDCovariance(),
        parameters=_Positive(1.2),
        optimizer=native,
    )
    calls = None
    fitted = []
    for seed, root in ((1, 1.4), (2, 1.8), (3, 2.2)):
        measure = _positive_measure(seed, root)
        result = run(_Positive(1.2), measure)
        assert bool(result.converged)
        assert passed_measures[-1] is measure
        assert float(result.theta_hat.sigma) == pytest.approx(root, abs=1e-7)
        fitted.append(float(result.theta_hat.sigma))
        counts = (model.aux_calls, model.psi_calls)
        if calls is None:
            assert all(count > 0 for count in counts)
            calls = counts
        else:
            assert counts == calls
    assert len(passed_measures) == 3
    assert np.all(np.diff(fitted) > 0.3)


def test_one_shot_preserves_existing_v1_args_capability():
    @jdc.pytree_dataclass
    class Euclidean:
        sigma: float

    class Solver:
        def __init__(self):
            self.args = []

        def __call__(self, residual_fn, theta_init, *, args=None):
            self.args.append(args)
            return linear_solver(verify=False)(residual_fn, theta_init, args=args)

    measure = EmpiricalMeasure.from_arrays(np.array([[1.0], [2.0], [3.0], [4.0]]))
    solver = Solver()
    result = estimate(
        _mean,
        measure,
        covariance=IIDCovariance(),
        parameters=Euclidean(1.0),
        optimizer=solver,
        weighting=Identity(),
    )
    assert bool(result.converged)
    assert len(solver.args) == 1 and solver.args[0] is measure
    assert float(result.theta_hat.sigma) == pytest.approx(2.5, abs=1e-14)
