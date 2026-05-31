"""Tests for the certificate-based ``linear_solver()`` optimizer.

``linear_solver`` is the affine-in-``theta`` fast path (#82 item 2). For a
residual that is affine in the parameters it reaches the least-squares
minimiser in one Gauss--Newton step and *certifies* the result via the
first-order optimality condition; on a failed certificate (a genuinely
nonlinear residual) it delegates to a fallback optimiser.

The unit tests exercise the adapter directly on plain residual functions;
the integration tests drive it through :func:`emu_gmm.estimate` on a small
linear/OLS moment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    EmpiricalMeasure,
    Identity,
    IIDCovariance,
    estimate,
)
from emu_gmm.optimizer import linear_solver, optimistix_lm

# ---------------------------------------------------------------------------
# Unit tests on plain residual functions
# ---------------------------------------------------------------------------


def test_affine_just_identified():
    """Affine residual ``A @ theta - b`` with invertible square ``A``."""
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((3, 3))
    # Ensure invertibility (well-conditioned with overwhelming probability).
    b_np = rng.standard_normal(3)
    A = jnp.asarray(A_np)
    b = jnp.asarray(b_np)

    def residual_fn(theta):
        return A @ theta - b

    theta_init = jnp.zeros(3)
    theta_hat, info = linear_solver()(residual_fn, theta_init)

    theta_star = np.linalg.solve(A_np, b_np)
    assert np.allclose(np.asarray(theta_hat), theta_star, atol=1e-10)
    assert info.steps == 1
    assert info.backend == "linear"
    # Residual is solved to ~0.
    assert float(jnp.linalg.norm(residual_fn(theta_hat))) < 1e-10


def test_affine_over_identified():
    """Affine residual ``A @ theta - b`` with tall ``A`` (M=5 > K=2)."""
    rng = np.random.default_rng(1)
    A_np = rng.standard_normal((5, 2))
    b_np = rng.standard_normal(5)
    A = jnp.asarray(A_np)
    b = jnp.asarray(b_np)

    def residual_fn(theta):
        return A @ theta - b

    theta_init = jnp.zeros(2)
    theta_hat, info = linear_solver()(residual_fn, theta_init)

    # numpy least-squares reference solution.
    theta_star, *_ = np.linalg.lstsq(A_np, b_np, rcond=None)
    assert np.allclose(np.asarray(theta_hat), theta_star, atol=1e-10)
    assert info.steps == 1
    assert info.backend == "linear"


def test_nonlinear_delegates_to_fallback():
    """Nonlinear residual ``theta**2 - c`` falls back (certificate fails)."""
    c = jnp.asarray([2.0, 5.0])

    def residual_fn(theta):
        return theta**2 - c

    theta_init = jnp.asarray([1.0, 1.0])

    theta_hat, info = linear_solver()(residual_fn, theta_init)
    # It delegated: the backend is NOT the linear fast path.
    assert info.backend != "linear"

    # The returned estimate matches what the fallback alone produces, and
    # solves theta**2 = c to the fallback's tolerance.
    theta_fallback, _ = optimistix_lm()(residual_fn, theta_init)
    assert np.allclose(np.asarray(theta_hat), np.asarray(theta_fallback), atol=1e-8)
    assert np.allclose(np.asarray(theta_hat) ** 2, np.asarray(c), atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests via estimate()
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class OLSParams:
    """Two-parameter OLS slope vector."""

    b0: float
    b1: float


def _ols_panel(seed: int = 0, n: int = 400):
    """Within-demeaned synthetic OLS data with K=M=2 regressors.

    Returns ``(x, beta_true, beta_hat_closed_form)`` where ``x`` packs the
    two regressors and the outcome as columns ``[x0, x1, y]``.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    # Within-demean the regressors so the moment is a clean centered OLS.
    X = X - X.mean(axis=0, keepdims=True)
    beta_true = np.array([1.5, -0.7])
    eps = 0.1 * rng.standard_normal(n)
    y = X @ beta_true + eps
    y = y - y.mean()  # demean the outcome to match
    # Closed-form within-OLS estimate.
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    data = np.column_stack([X, y])  # (n, 3): x0, x1, y
    return jnp.asarray(data), beta_true, beta_hat


def _ols_residual(x_i, theta):
    """Per-observation OLS moment ``x_i * (y_i - x_i . beta)`` (M=K=2)."""
    x = x_i[:2]
    y = x_i[2]
    beta = jnp.asarray([theta.b0, theta.b1])
    return x * (y - jnp.dot(x, beta))


def test_estimate_linear_path_exact():
    """estimate() with weighting=Identity() fires the linear path: exact OLS."""
    data, _beta_true, beta_hat = _ols_panel(seed=0)
    n = data.shape[0]
    measure = EmpiricalMeasure(
        x=data,
        mask=jnp.ones((n, 2)),
        weights=jnp.ones(n),
    )

    result = estimate(
        model=_ols_residual,
        measure=measure,
        covariance=IIDCovariance(),
        weighting=Identity(),
        optimizer=linear_solver(),
        theta_init=OLSParams(b0=0.0, b1=0.0),
    )

    est = np.array([float(result.theta_hat.b0), float(result.theta_hat.b1)])
    assert np.allclose(est, beta_hat, atol=1e-9)
    # With Identity() whitening the residual is affine, so the linear path
    # should converge in a single step.
    assert int(result.iterations) == 1


def test_estimate_cu_falls_back():
    """estimate() with default CU weighting falls back but recovers same point.

    Under ContinuouslyUpdated weighting the whitened residual
    ``L(theta)^{-1} m(theta)`` is non-affine in ``theta`` even for a linear
    model (the whitening Cholesky depends on ``theta``). The certificate
    therefore fails and ``linear_solver`` delegates to its fallback --- by
    design. The point estimate must match the fallback (optimistix_lm)
    used directly.
    """
    data, _beta_true, _beta_hat = _ols_panel(seed=0)
    n = data.shape[0]

    def _make_measure():
        return EmpiricalMeasure(
            x=data,
            mask=jnp.ones((n, 2)),
            weights=jnp.ones(n),
        )

    common = dict(
        model=_ols_residual,
        covariance=IIDCovariance(),
        theta_init=OLSParams(b0=0.0, b1=0.0),
    )  # default weighting == ContinuouslyUpdated

    result_linear = estimate(
        measure=_make_measure(), optimizer=linear_solver(), **common
    )
    result_fallback = estimate(
        measure=_make_measure(), optimizer=optimistix_lm(), **common
    )

    est_linear = np.array(
        [float(result_linear.theta_hat.b0), float(result_linear.theta_hat.b1)]
    )
    est_fallback = np.array(
        [float(result_fallback.theta_hat.b0), float(result_fallback.theta_hat.b1)]
    )
    assert np.allclose(est_linear, est_fallback, atol=1e-8)


# ---------------------------------------------------------------------------
# Under jit, the fast path fires via lax.cond (rather than always delegating)
# ---------------------------------------------------------------------------


def test_jit_affine_fires_linear_path():
    """Under jit, an affine residual takes the one-step linear path
    (steps == 1) and lands on the exact least-squares solution.
    """
    rng = np.random.default_rng(7)
    A_np = rng.standard_normal((3, 3))
    b_np = rng.standard_normal(3)
    A, b = jnp.asarray(A_np), jnp.asarray(b_np)

    def residual_fn(theta):
        return A @ theta - b

    solver = linear_solver()
    theta_hat, info = jax.jit(lambda t0: solver(residual_fn, t0))(jnp.zeros(3))

    assert np.allclose(np.asarray(theta_hat), np.linalg.solve(A_np, b_np), atol=1e-10)
    assert int(info.steps) == 1  # the linear branch ran, not the fallback
    assert info.backend == "linear"
    # Matches the eager linear path exactly.
    theta_eager, _ = solver(residual_fn, jnp.zeros(3))
    assert np.allclose(np.asarray(theta_hat), np.asarray(theta_eager), atol=1e-12)


def test_jit_nonlinear_falls_back_via_cond():
    """Under jit, a nonlinear residual's certificate fails at runtime, so
    lax.cond runs the fallback branch (steps > 1) and still converges.
    """
    c = jnp.asarray([2.0, 5.0])

    def residual_fn(theta):
        return theta**2 - c

    solver = linear_solver()
    theta_hat, info = jax.jit(lambda t0: solver(residual_fn, t0))(
        jnp.asarray([1.0, 1.0])
    )
    # The fallback ran inside the cond: more than one step, and it converged.
    assert int(info.steps) > 1
    assert np.allclose(np.asarray(theta_hat) ** 2, np.asarray(c), atol=1e-6)


# ---------------------------------------------------------------------------
# verify=False: caller-asserted linearity (unconditional one-step; vmap-clean)
# ---------------------------------------------------------------------------


def test_verify_false_affine_exact_one_step():
    rng = np.random.default_rng(9)
    A_np = rng.standard_normal((3, 3))
    b_np = rng.standard_normal(3)
    A, b = jnp.asarray(A_np), jnp.asarray(b_np)

    def residual_fn(theta):
        return A @ theta - b

    theta_hat, info = linear_solver(verify=False)(residual_fn, jnp.zeros(3))
    assert np.allclose(np.asarray(theta_hat), np.linalg.solve(A_np, b_np), atol=1e-10)
    assert info.steps == 1
    assert info.backend == "linear"


def test_verify_false_vmaps_over_right_hand_sides():
    """The motivation: a vmapped (and jitted) MC over a known-linear moment
    solves every replicate in one step, with NO fallback / NO select cost.
    """
    rng = np.random.default_rng(11)
    A_np = rng.standard_normal((3, 3))
    A = jnp.asarray(A_np)
    b_batch = jnp.asarray(rng.standard_normal((16, 3)))
    solver = linear_solver(verify=False)

    def solve_for(b):
        def residual_fn(theta):
            return A @ theta - b

        theta_hat, _ = solver(residual_fn, jnp.zeros(3))
        return theta_hat

    thetas = jax.jit(jax.vmap(solve_for))(b_batch)  # (16, 3)
    expected = np.linalg.solve(A_np, np.asarray(b_batch).T).T  # (16, 3)
    assert np.allclose(np.asarray(thetas), expected, atol=1e-9)


def test_verify_false_does_not_fall_back_on_nonlinear():
    """verify=False trusts the caller: it takes ONE step and does not fall
    back, so on a nonlinear residual it returns the (non-converged) one-step
    point and final_objective is not ~0 (the misuse is surfaced, not fixed).
    """
    c = jnp.asarray([2.0, 5.0])

    def residual_fn(theta):
        return theta**2 - c

    theta_hat, info = linear_solver(verify=False)(residual_fn, jnp.asarray([1.0, 1.0]))
    assert info.steps == 1  # one unconditional step, no fallback
    assert info.backend == "linear"
    # One Gauss-Newton step from [1, 1] lands at [1.5, 3], not the root.
    assert np.allclose(np.asarray(theta_hat), [1.5, 3.0], atol=1e-6)
    assert float(info.final_objective) > 1e-3  # NOT converged -> misuse visible


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
