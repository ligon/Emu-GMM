"""Projected PCG accuracy, trust geometry and work counts (issue #166)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import Euclidean, ManifoldLeaf, PSDFixedRank, riemannian_tr
from emu_gmm._internal.params import flatten_params_with_spec
from emu_gmm.manifolds.riemannian_tr import (
    _build_plan,
    _gn_diagonal_preconditioner,
    _project_flat,
    _tcg_core,
)


@jdc.pytree_dataclass
class VectorParams:
    x: ManifoldLeaf


@jdc.pytree_dataclass
class FactorParams:
    Y: ManifoldLeaf


def euclidean_plan(n):
    point, _, spec = flatten_params_with_spec(
        VectorParams(ManifoldLeaf(jnp.zeros(n), Euclidean(n)))
    )
    return point, _build_plan(spec, n)


def solve_inner(J, g, radius, *, min_inner=0, diagonal=True, max_inner=None):
    J, g = jnp.asarray(J), jnp.asarray(g)
    point, plan = euclidean_plan(len(g))
    precon = _gn_diagonal_preconditioner(J, plan, point) if diagonal else None
    return _tcg_core(
        lambda v: J.T @ (J @ v),
        g,
        plan,
        point,
        jnp.asarray(radius),
        theta=1.0,
        kappa=1e-10,
        min_inner=min_inner,
        max_tcg_steps=max_inner or len(g),
        precon=precon,
    )


def test_exact_spd_solution_stops_before_minimum_without_false_negative_curvature():
    step, Hstep, info = solve_inner(np.diag([1.0, 2.0]), [1.0, 4.0], 10.0, min_inner=1)
    np.testing.assert_array_equal(step, [-1.0, -1.0])
    np.testing.assert_array_equal(Hstep, [-1.0, -4.0])
    assert info["stop_reason"] == "reached_target_linear"
    assert int(info["num_inner"]) == 1


def test_zero_gradient_does_no_inner_work():
    step, _, info = solve_inner(np.eye(2), [0.0, 0.0], 1.0, min_inner=2)
    np.testing.assert_array_equal(step, [0.0, 0.0])
    assert int(info["num_inner"]) == 0
    assert info["stop_reason"] == "reached_target_superlinear"


def test_large_finite_gradient_retains_finite_physical_boundary_step():
    J, g = np.diag([1.0, 1e-8]), [0.0, 1e140]
    step, _, info = solve_inner(J, g, 1.0)
    reference, _, _ = solve_inner(J, g, 1.0, diagonal=False)
    np.testing.assert_array_equal(step, reference)
    np.testing.assert_array_equal(step, [0.0, -1.0])
    assert info["stop_reason"] == "exceeded_tr"


def test_tiny_finite_gradient_does_not_underflow_direction_curvature():
    J, g = np.diag([1.0, 1e-8]), [1e-140, 0.0]
    step, _, info = solve_inner(J, g, 1.0)
    reference, _, _ = solve_inner(J, g, 1.0, diagonal=False)
    np.testing.assert_allclose(step, reference, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(step, [-1e-140, 0.0], rtol=1e-12, atol=0.0)
    assert info["stop_reason"] == "reached_target_superlinear"


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_projected_inverse_is_positive_self_adjoint_and_scale_independent(scale):
    rng = np.random.default_rng(166)
    Y = jnp.asarray(rng.normal(size=(5, 2)))
    point, _, spec = flatten_params_with_spec(
        FactorParams(ManifoldLeaf(Y, PSDFixedRank(5, 2)))
    )
    plan = _build_plan(spec, len(point))
    J = jnp.asarray(rng.normal(size=(14, 10)) * np.logspace(-3, 3, 10))
    precon = _gn_diagonal_preconditioner(scale * J, plan, point)
    baseline = _gn_diagonal_preconditioner(J, plan, point)
    u, v = (jnp.asarray(rng.normal(size=10)) for _ in range(2))
    Pu, Pv = precon(u), precon(v)
    np.testing.assert_allclose(Pu, baseline(u), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(Pu, _project_flat(plan, point, Pu), atol=1e-12)
    np.testing.assert_allclose(u @ Pv, v @ Pu, rtol=2e-12, atol=1e-14)
    assert float(u @ Pu) > 0.0
    omega = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(precon((Y @ omega).ravel()), 0.0, atol=1e-12)


def test_zero_and_weak_columns_give_a_finite_positive_inverse():
    point, plan = euclidean_plan(3)
    for J in [jnp.zeros((3, 3)), jnp.diag(jnp.array([1.0, 0.0, 1e-150]))]:
        precon = _gn_diagonal_preconditioner(J, plan, point)
        matrix = jax.jacfwd(precon)(point)
        assert bool(jnp.all(jnp.isfinite(matrix)))
        assert float(jnp.linalg.eigvalsh(matrix)[0]) > 0.0
        assert float(jnp.linalg.eigvalsh(matrix)[-1]) <= 1.0
    np.testing.assert_array_equal(
        _gn_diagonal_preconditioner(jnp.zeros((2, 3)), plan, point)(jnp.ones(3)),
        jnp.ones(3),
    )


def test_nonfinite_derivatives_are_not_silently_replaced_by_identity():
    point, plan = euclidean_plan(2)
    precon = _gn_diagonal_preconditioner(jnp.array([[1.0, jnp.nan]]), plan, point)
    assert bool(jnp.any(~jnp.isfinite(precon(jnp.ones(2)))))


@pytest.mark.parametrize("scale", [1e-140, 1.0, 1e140])
def test_boundary_after_multiple_iterations_uses_original_metric(scale):
    H = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    g = np.array([1.0, 2.0, 3.0])
    J = np.linalg.cholesky(H).T
    inverse = 1.0 / np.diag(H)
    z, r = inverse * g, g.copy()
    d, eta = -z, np.zeros(3)
    rz = r @ z
    alpha = rz / (d @ H @ d)
    eta = alpha * d
    r = r + alpha * (H @ d)
    z = inverse * r
    d = -z + (r @ z) / rz * d
    full_second = eta + ((r @ z) / (d @ H @ d)) * d
    radius = (np.linalg.norm(eta) + np.linalg.norm(full_second)) / 2.0
    roots = np.roots([d @ d, 2 * eta @ d, eta @ eta - radius**2])
    expected = eta + max(roots) * d
    step, Hstep, info = solve_inner(J, scale * g, scale * radius)
    assert int(info["num_inner"]) == 2
    assert info["stop_reason"] == "exceeded_tr"
    np.testing.assert_allclose(step / scale, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(step / scale), radius, rtol=1e-12)
    np.testing.assert_allclose(Hstep / scale, H @ expected, rtol=1e-12, atol=1e-12)


def test_preconditioning_keeps_true_negative_curvature():
    point, plan = euclidean_plan(2)
    H = jnp.diag(jnp.array([-1.0, 2.0]))
    precon = _gn_diagonal_preconditioner(jnp.diag(jnp.array([1.0, 3.0])), plan, point)
    step, Hstep, info = _tcg_core(
        lambda v: H @ v,
        jnp.array([1.0, 0.0]),
        plan,
        point,
        jnp.asarray(0.25),
        theta=1.0,
        kappa=0.1,
        min_inner=1,
        max_tcg_steps=2,
        precon=precon,
    )
    assert info["stop_reason"] == "negative_curvature"
    np.testing.assert_allclose(jnp.linalg.norm(step), 0.25, rtol=1e-12)
    assert float(step @ Hstep) < 0.0


def test_jit_and_vmap_keep_independent_inner_status_and_steps():
    J = jnp.diag(jnp.array([1.0, 10.0]))

    def run(g):
        step, _, info = solve_inner(J, g, 1.0)
        return step, info["num_inner"], info["stop_code"]

    gradients = jnp.array([[0.0, 0.0], [0.5, 1.0], [100.0, 0.0]])
    steps, counts, codes = jax.jit(jax.vmap(run))(gradients)
    assert list(np.asarray(counts)) == [0, 1, 1]
    np.testing.assert_allclose(steps[1], [-0.5, -0.01], atol=1e-12)
    np.testing.assert_allclose(steps[2], [-1.0, 0.0], atol=1e-12)
    assert list(np.asarray(codes)) == [3, 2, 1]


def test_unknown_preconditioner_is_rejected():
    with pytest.raises(ValueError, match="precon must"):
        riemannian_tr(precon="unknown")


def test_default_and_explicit_identity_have_identical_iterates_and_trace():
    initial = VectorParams(ManifoldLeaf(jnp.array([-1.2, 1.0]), Euclidean(2)))
    _, _, spec = flatten_params_with_spec(initial)

    def residual(x):
        return jnp.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

    a, info_a = riemannian_tr(max_steps=100)(residual, initial, spec)
    b, info_b = riemannian_tr(max_steps=100, precon="identity")(residual, initial, spec)
    np.testing.assert_array_equal(a.x.array, b.x.array)
    assert bool(info_a.done) and bool(info_b.done)
    for name in info_a.tr_trace._d:
        np.testing.assert_array_equal(info_a.tr_trace[name], info_b.tr_trace[name])


def test_diagonal_reduces_inner_work_without_changing_converged_gamma():
    rng = np.random.default_rng(166)
    n, k = 5, 2
    truth = jnp.asarray(rng.normal(size=(n, k)))
    target = truth @ truth.T
    start = truth + jnp.asarray(0.03 * rng.normal(size=(n, k)))
    initial = FactorParams(ManifoldLeaf(start, PSDFixedRank(n, k)))
    flat, _, spec = flatten_params_with_spec(initial)
    scales = jnp.logspace(-2, 2, n)
    weights = jnp.sqrt(scales[:, None] * scales[None, :])

    def residual(x, inputs):
        target, weights = inputs
        Y = x.reshape(n, k)
        return (weights * (Y @ Y.T - target)).ravel()

    inputs = (target, weights)
    J = np.asarray(jax.jacfwd(residual)(flat, inputs))
    spectrum = np.linalg.eigvalsh(J.T @ J)
    # One gauge zero is dropped by the existing fixed gauge count.
    assert spectrum[-1] / spectrum[spec.total_gauge_dim] > 1e4
    fits, work = [], []
    for precon in ("identity", "diagonal"):
        solver = riemannian_tr(
            precon=precon, max_steps=200, rtol=1e-8, atol=1e-10, ftol=0
        )
        fitted, info = solver(residual, initial, spec, args=inputs)
        assert bool(info.done)
        assert float(info.final_gradient_norm) < 1e-10
        gamma = fitted.Y.array @ fitted.Y.array.T
        np.testing.assert_allclose(gamma, target, atol=1e-8, rtol=1e-8)
        fits.append(gamma)
        work.append(int(jnp.sum(info.tr_trace.tcg_iters)))
    np.testing.assert_allclose(fits[0], fits[1], atol=1e-8, rtol=1e-8)
    assert work[1] < work[0] / 2
    # Entrywise Jacobi scaling may change the trajectory after a gauge
    # rotation. The converged invariant, rather than that trajectory, agrees.
    angle = 0.7
    rotation = jnp.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = FactorParams(ManifoldLeaf(start @ rotation, PSDFixedRank(n, k)))
    fitted, info = solver(residual, rotated, spec, args=inputs)
    assert bool(info.done)
    np.testing.assert_allclose(
        fitted.Y.array @ fitted.Y.array.T, target, atol=1e-8, rtol=1e-8
    )


def test_public_args_path_reuses_solver_under_jit_and_vmap():
    initial = VectorParams(ManifoldLeaf(jnp.zeros(2), Euclidean(2)))
    _, _, spec = flatten_params_with_spec(initial)

    def residual(x, target):
        return jnp.array([1.0, 10.0]) * (x - target)

    solver = riemannian_tr(precon="diagonal", max_steps=30)

    def run(target):
        fitted, info = solver(residual, initial, spec, args=target)
        return fitted.x.array, info.done, info.steps, info.tr_trace.tcg_iters

    targets = jnp.array([[0.3, -0.2], [2.0, 1.0]])
    points, done, steps, counts = jax.jit(jax.vmap(run))(targets)
    np.testing.assert_allclose(points, targets, rtol=1e-10, atol=1e-10)
    assert bool(jnp.all(done))
    assert bool(jnp.all(steps < 30))
    assert bool(jnp.all(jnp.sum(counts, axis=1) > 0))
    again, again_done, _, _ = jax.jit(run)(targets[0])
    np.testing.assert_allclose(again, targets[0], rtol=1e-10, atol=1e-10)
    assert bool(again_done)
