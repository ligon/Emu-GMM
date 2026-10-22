"""Numerical-stagnation exit for the Riemannian LM (#210).

Strict ``cost_new < cost`` acceptance at a float64 cost floor rejects every
step, the rejected-step rule ``lam -> 2 lam`` overflows the damping, and the
pre-#210 loop evaluated nonfinite proposals until ``max_steps``. The loop now
stops on a rejected step from which no later step can be accepted -- the
retracted proposal is bitwise the current point, or the step / damping is
nonfinite -- reporting ``status="stagnated"`` / ``OptimizerInfo.stagnated``
with ``done=False``. These are the regression gates the issue asks for: the
data-free 2-D and 1-D floor cases, the nonfinite-damping case, the traced
``args`` path, a budget below the stagnation point (still
``max_iterations``), an unchanged healthy solve, and the ``Positive``
rejected-start transient (must still recover, never stagnate).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import Euclidean, ParameterSpace, on
from emu_gmm._internal.params import flatten_params_with_spec
from emu_gmm.covariance import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.manifolds import Positive
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated

_TOL = dict(rtol=1e-10, atol=1e-10)


class _P2(ParameterSpace):
    x: jax.Array = on(Euclidean(2), default=jnp.full(2, 1e-9))


class _P1(ParameterSpace):
    x: jax.Array = on(Euclidean(1), default=jnp.full(1, 1e-9))


def _floor_residual(x):
    # Q(x) = 1 + x'x. For x ~ 1e-9 both the current and the proposed squared
    # cost round to exactly 1.0 in float64, so every step is rejected; yet
    # ||g|| = ||x|| ~ 1.4e-9 sits above the 2e-10 stationarity threshold, so
    # nothing can certify. The minimiser is x = 0.
    return jnp.concatenate([x, jnp.ones(1)])


def _solve(space, residual=_floor_residual, tol=_TOL, **kw):
    p = space.point()
    _, _, spec = flatten_params_with_spec(p)
    ans, info = riemannian_lm(**tol, **kw)(residual, p, spec)
    return p, ans, info


class TestCostFloorStagnation:
    """The issue's data-free fixture: finite, early, uncertified stop."""

    @pytest.mark.parametrize("space", [_P2, _P1], ids=["2d", "1d"])
    def test_float64_cost_floor_stops_as_stagnated(self, space):
        p, ans, info = _solve(space, max_steps=1600)
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        # Early: lam reaches the null-step point after ~60 doublings, far
        # inside the default 200-step budget -- never the 1600 of the report.
        assert 1 < int(info.steps) < 200
        # The returned point is bitwise the start (every step was rejected)
        # and finite; the objective is the representational floor 0.5 * 1.0.
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))
        assert np.all(np.isfinite(np.asarray(ans.x.array)))
        assert float(info.final_objective) == pytest.approx(0.5)

    def test_budget_below_stagnation_point_is_still_max_iterations(self):
        # With too few steps to reach the null-step point the loop exhausts
        # its budget exactly as before; ``stagnated`` is not a budget flag.
        _, _, info = _solve(_P2, max_steps=10)
        assert info.status == "max_iterations"
        assert bool(info.stagnated) is False
        assert bool(info.done) is False
        assert int(info.steps) == 10

    def test_nonfinite_damping_stops_at_once(self):
        # sqrt of a negative start: NaN residual and Jacobian, so the lambda
        # anchor and the first LM step are NaN. Pre-#210 this rejected (NaN <
        # cost is False) and doubled NaN for the whole budget.
        class _Neg(ParameterSpace):
            x: jax.Array = on(Euclidean(2), default=jnp.full(2, -1.0))

        def residual(x):
            return jnp.concatenate([jnp.sqrt(x), jnp.ones(1)])

        p, ans, info = _solve(_Neg, residual=residual, max_steps=500)
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        assert int(info.steps) == 1
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))


class TestStagnatedRidesTracedArgsPath:
    """The flag is a traced carry value: it must survive the memoised jitted
    solve (#124 ``args`` channel) and an outer ``jax.jit``, where ``status``
    collapses to ``"traced"`` and only the traced fields carry information.
    """

    def test_args_path_eager_outputs(self):
        def kernel(x, a):
            return jnp.concatenate([x, a])

        p = _P2.point()
        _, _, spec = flatten_params_with_spec(p)
        solver = riemannian_lm(**_TOL, max_steps=400)
        ans, info = solver(kernel, p, spec, args=jnp.ones(1))
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        assert int(info.steps) < 200
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))

    def test_under_outer_jit_flag_is_traced_and_true(self):
        def kernel(x, a):
            return jnp.concatenate([x, a])

        p = _P2.point()
        _, _, spec = flatten_params_with_spec(p)
        solver = riemannian_lm(**_TOL, max_steps=400)
        statuses: list[str] = []

        @jax.jit
        def run(a):
            _, info = solver(kernel, p, spec, args=a)
            statuses.append(info.status)
            return info.stagnated, info.done, info.steps

        stagnated, done, steps = run(jnp.ones(1))
        assert statuses == ["traced"]
        assert bool(stagnated) is True
        assert bool(done) is False
        assert int(steps) < 200


class _Far(ParameterSpace):
    x: jax.Array = on(Euclidean(2), default=jnp.array([3.0, -2.0]))


_TARGET = jnp.array([0.25, 1.5])


def _far_residual(x):
    # ||x - t||^2 + 1: the constant row puts the float64 cost floor at
    # ||x - t|| ~ sqrt(eps) ~ 1.5e-8, so the LM's last ACCEPTED step lands at
    # an error of ~1e-9 -- inside the default 2e-8 gradient threshold, but
    # above a 2e-10 one.
    return jnp.concatenate([x - _TARGET, jnp.ones(1)])


class TestHealthySolvesUntouched:
    """A converging solve never rejects a null step, so it certifies exactly
    as before and the flag stays False."""

    def test_euclidean_least_squares_converges_without_flag(self):
        _, ans, info = _solve(_Far, residual=_far_residual, tol={}, max_steps=100)
        assert info.status == "converged"
        assert bool(info.done) is True
        assert bool(info.stagnated) is False
        assert int(info.steps) < 10
        np.testing.assert_allclose(
            np.asarray(ans.x.array), np.asarray(_TARGET), atol=1e-8
        )

    def test_tolerance_below_the_cost_floor_is_reported_as_stagnation(self):
        # The SAME problem asked for 1e-10: the floor is reached (error ~1e-9)
        # before the gradient threshold, so no criterion can certify. Pre-#210
        # this ran to max_steps as "max_iterations"; it is now an early,
        # explicit stagnation report with the same (correct to ~1e-9) point.
        _, ans, info = _solve(_Far, residual=_far_residual, max_steps=400)
        assert info.status == "stagnated"
        assert bool(info.done) is False
        assert bool(info.stagnated) is True
        assert int(info.steps) < 200
        np.testing.assert_allclose(
            np.asarray(ans.x.array), np.asarray(_TARGET), atol=1e-8
        )


@jdc.pytree_dataclass
class _ScaleParams:
    sigma: jnp.ndarray
    __emu_manifolds__ = {"sigma": Positive()}


def _scale_residual(x, theta):
    s = theta.sigma
    return jnp.array([x[0] ** 2 - s**2, x[0] ** 4 - 3.0 * s**4])


class TestRejectedStartTransientIsNotStagnation:
    """The ``Positive`` stuck start (mirrors ``test_riemannian_lm_ftol``): the
    Gauss--Newton step overshoots and is rejected for a dozen-odd iterations
    before the LM breaks free. Those rejections are large, finite,
    point-moving steps -- not the absorbing null step -- so the solve must
    still climb to the truth and certify, with ``stagnated`` False.
    """

    def test_positive_sub_true_start_recovers(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.5, size=4000)  # sigma_true = 1.5
        meas = EmpiricalMeasure(
            x=jnp.asarray(draws[:, None]),
            mask=jnp.ones((4000, 2)),
            weights=jnp.ones(4000),
        )
        r = estimate(
            model=_scale_residual,
            measure=meas,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=riemannian_lm(),
            theta_init=_ScaleParams(sigma=jnp.asarray(0.5)),
        )
        info = r.diagnostics.optimizer_info
        assert bool(r.converged) is True
        assert info.status == "converged"
        assert bool(info.stagnated) is False
        assert float(r.theta_hat.sigma) == pytest.approx(1.5, abs=0.1)
