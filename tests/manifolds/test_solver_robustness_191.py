"""Direct, bounded regressions for ledger Section 20 / issue #191.

Reproduce with PYTHONPATH=src python -m pytest
tests/manifolds/test_solver_robustness_191.py. Fixtures are deterministic,
float64, and use only native solvers (no estimation or calibration loop).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm._internal.params import flatten_params_with_spec
from emu_gmm.manifolds import ManifoldLeaf, Positive, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import _min_horizontal_curvature, riemannian_lm
from emu_gmm.manifolds.riemannian_tr import (
    _build_plan,
    _psd_factors_full_rank,
    _riemannian_hvp,
    riemannian_tr,
)


def _factor(scale=1.0):
    return scale * jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


def _point(factor):
    theta = (ManifoldLeaf(factor, PSDFixedRank(3, 2)),)
    flat, _, spec = flatten_params_with_spec(theta)
    return theta, flat, spec


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_rank_guard_is_relative_and_rejects_actual_rank_loss(scale):
    _, flat, spec = _point(_factor(scale))
    plan = _build_plan(spec, flat.size)
    assert bool(_psd_factors_full_rank(plan, flat))
    singular = flat.reshape(3, 2).at[:, 1].set(0.0).ravel()
    assert not bool(_psd_factors_full_rank(plan, singular))


def test_rank_guard_checks_each_factor_at_its_own_scale():
    theta = (
        ManifoldLeaf(_factor(1e12), PSDFixedRank(3, 2)),
        ManifoldLeaf(_factor(1e-12), PSDFixedRank(3, 2)),
    )
    flat, _, spec = flatten_params_with_spec(theta)
    plan = _build_plan(spec, flat.size)
    assert bool(_psd_factors_full_rank(plan, flat))
    assert not bool(_psd_factors_full_rank(plan, flat.at[6:].set(0.0)))


def test_rank_guard_handles_cancellation_in_gram_eigenvalues():
    # The columns differ by 1e-8, resolved in singular values even though the
    # Gram diagonal rounds 1 + 1e-16 to 1 and loses the smaller eigenvalue.
    factor = jnp.array([[1.0, 1.0], [0.0, 1e-8], [0.0, 0.0]])
    _, flat, spec = _point(factor)
    assert bool(_psd_factors_full_rank(_build_plan(spec, flat.size), flat))


def test_small_full_rank_factor_moves_instead_of_false_convergence():
    # Before #191 all proposals failed the absolute Gram cutoff, and the
    # unchanged starting point was certified after four radius decreases.
    scale = 1e-8
    factor = _factor(scale)
    theta, _, spec = _point(factor)
    target = 2 * factor @ factor.T

    def residual(flat):
        a = flat.reshape(3, 2)
        return ((a @ a.T - target) / scale**2).ravel()

    fitted, info = riemannian_tr(max_steps=12, init_radius=1e-10, max_radius=1e-8)(
        residual, theta, spec
    )
    assert bool(info.extra["proposed_full_rank"][0])
    assert bool(info.extra["accepted"][0])
    assert not np.array_equal(fitted[0].array, factor)
    assert float(info.final_objective) < 1.0  # initial objective is exactly 1


@pytest.mark.parametrize("transform", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("invalid", [False, True])
def test_rejected_radius_collapse_is_failure(transform, invalid):
    # An invalid factor or a rejected nonlinear proposal does not establish
    # stationarity. Both have nonzero horizontal gradient at the stopped point.
    factor = _factor().at[1, 1].set(1e-16) if invalid else _factor()
    theta, flat, spec = _point(factor)

    def residual(flat):
        a = flat.reshape(3, 2)
        g00 = (a @ a.T)[0, 0]
        if invalid:
            return jnp.array([g00 - 2.0])
        s = 1e14 * (g00 - 1.0)
        return jnp.array([s - 1.0 + 100 * s**3])

    solver = riemannian_tr(max_steps=12, init_radius=1e-10)

    def run(theta):
        fitted, info = solver(residual, theta, spec)
        return (
            fitted[0].array,
            info.done,
            info.stagnated,
            info.steps,
            info.final_gradient_norm,
            info.extra["accepted"],
            info.tr_trace.tcg_stop,
        )

    if transform == "vmap":
        batched = jax.tree.map(lambda x: jnp.stack([x, x]), theta)
        outputs = jax.jit(jax.vmap(run))(batched)
        outputs = tuple(x[0] for x in outputs)
    else:
        outputs = jax.jit(run)(theta) if transform == "jit" else run(theta)
    fitted, done, stagnated, steps, gradient, accepted, codes = outputs
    assert not bool(done)
    assert bool(stagnated)
    assert int(steps) < 12
    assert float(gradient) > 1.0
    np.testing.assert_array_equal(fitted.ravel(), flat)
    assert not np.any(np.asarray(accepted)[: int(steps)])
    assert np.all(np.asarray(codes)[int(steps) :] == -1)
    if transform == "eager":
        _, info = solver(residual, theta, spec)
        assert info.status == "stagnated"


def test_actual_rank_dropping_proposal_is_rejected():
    theta, _, spec = _point(_factor())

    def residual(flat):
        a = flat.reshape(3, 2)
        gamma = a @ a.T
        # The first objective term is 0.5 ||a[0]||^2; its unit-radius
        # Newton proposal annihilates the first row and loses one column rank.
        return jnp.array([jnp.linalg.norm(a[0]), gamma[1, 1] - 1, gamma[0, 1]])

    fitted, info = riemannian_tr(max_steps=1, init_radius=1.0)(residual, theta, spec)
    assert not bool(info.extra["proposed_full_rank"][0])
    assert not bool(info.extra["accepted"][0])
    assert not bool(info.done)
    np.testing.assert_array_equal(fitted[0].array, theta[0].array)


@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_residual_scaling_cannot_certify_material_model_gain(scale):
    theta, flat, spec = _point(_factor())

    def residual(x):
        a = x.reshape(3, 2)
        s = 1e14 * ((a @ a.T)[0, 0] - 1.0)
        return scale * jnp.array([s - 1.0 + 100 * s**3])

    fitted, info = riemannian_tr(max_steps=12, init_radius=1e-10)(residual, theta, spec)
    assert not bool(info.done)
    assert bool(info.stagnated)
    np.testing.assert_array_equal(fitted[0].array.ravel(), flat)
    last = int(info.steps) - 1
    # The scaled objective is tiny, but its predicted relative gain is material.
    assert float(info.tr_trace._d["rhoden"][last]) > (
        np.finfo(float).eps * float(info.final_objective)
    )


@pytest.mark.parametrize("transform", ["eager", "jit", "vmap"])
def test_tiny_model_gain_does_not_certify_large_actual_increase(transform):
    theta, flat, spec = _point(_factor())

    def residual(x):
        a = x.reshape(3, 2)
        d = (a @ a.T)[0, 0] - 1.0
        return jnp.array([-1.0 + 5e-6 * d + 1e40 * d**3])

    solver = riemannian_tr(max_steps=12, init_radius=1e-10)

    def run(theta):
        fitted, info = solver(residual, theta, spec)
        last = info.steps - 1
        return (
            fitted[0].array,
            info.done,
            info.stagnated,
            info.final_objective,
            info.tr_trace._d["rhoden"][last],
            info.tr_trace._d["rhonum"][last],
        )

    if transform == "vmap":
        batched = jax.tree.map(lambda x: jnp.stack([x, x]), theta)
        outputs = tuple(x[0] for x in jax.jit(jax.vmap(run))(batched))
    else:
        outputs = jax.jit(run)(theta) if transform == "jit" else run(theta)
    fitted, done, stagnated, cost, predicted, actual = outputs
    assert not bool(done)
    assert bool(stagnated)
    assert 0 <= float(predicted) <= np.finfo(float).eps * float(cost)
    assert abs(float(actual)) > 1e-8 * float(cost)
    np.testing.assert_array_equal(fitted.ravel(), flat)


@pytest.mark.parametrize("ftol, expected", [(1e-8, True), (1e-14, False), (0.0, False)])
def test_rejected_cost_floor_respects_existing_ftol(ftol, expected):
    theta, _, spec = _point(_factor())

    def residual(x):
        a = x.reshape(3, 2)
        d = (a @ a.T)[0, 0] - 1.0
        # Deterministic evaluation noise emulates a tiny CU cost increase;
        # unlike the large nonlinear mismatch, it meets the default ftol.
        jitter = jnp.where(d == 0.0, 0.0, 1e-12)
        return jnp.array([-1.0 + 5e-6 * d - jitter])

    _, info = riemannian_tr(max_steps=12, init_radius=1e-10, ftol=ftol)(
        residual, theta, spec
    )
    assert bool(info.done) is expected
    assert bool(info.stagnated) is not expected
    assert not np.any(np.asarray(info.tr_trace._d["accepted"])[: int(info.steps)])


def test_frobenius_advisory_eigenvalues_are_unchanged():
    _, flat, spec = _point(_factor())

    def residual(x):
        return jnp.array([3.0 + jnp.sum((x - flat) ** 2)])

    eye = jnp.eye(flat.size)
    old = jnp.stack(
        [_riemannian_hvp(residual, flat, spec, eye[:, j]) for j in range(flat.size)],
        axis=1,
    )
    expected = jnp.linalg.eigvalsh(0.5 * (old + old.T))
    observed = _min_horizontal_curvature(residual, spec, flat)
    np.testing.assert_array_equal(observed, expected[jnp.array([0, -1])])


def _mixed_point_and_residual(*, saddle=False):
    theta = (
        ManifoldLeaf(_factor(), PSDFixedRank(3, 2)),
        ManifoldLeaf(jnp.array(10.0), Positive()),
    )
    flat, _, spec = flatten_params_with_spec(theta)

    def residual(x):
        a = x[:6].reshape(3, 2)
        gamma = a @ a.T
        s = x[6] - 10.0
        linear = jnp.array(
            [
                gamma[0, 0] - 1.0 + 0.5 * s,
                gamma[1, 1] - 1.0,
                gamma[0, 1],
                gamma[0, 2],
                gamma[1, 2],
                jnp.sqrt(0.75) * s,
            ]
        )
        if saddle:
            # Nonzero residual with negative second derivative at the same
            # stationary point supplies genuine horizontal negative curvature.
            return jnp.concatenate([linear, jnp.array([3.0 - s**2])])
        return linear

    return theta, flat, spec, residual


@pytest.mark.parametrize("saddle", [False, True])
def test_mixed_positive_advisory_distinguishes_minimum_and_saddle(saddle):
    theta, flat, spec, residual = _mixed_point_and_residual(saddle=saddle)
    min_curvature, _ = _min_horizontal_curvature(residual, spec, flat)
    # Gauge direction is zero. A least-squares zero-residual minimum cannot
    # have negative curvature; the saddle has a strictly negative scale mode.
    assert (min_curvature < -1e-6) == saddle
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = riemannian_lm(max_steps=3)(residual, theta, spec)
    assert info.status == "converged"
    assert info.stalled_indefinite is saddle
    advisory = [w for w in caught if "non-convex" in str(w.message)]
    assert bool(advisory) == saddle
