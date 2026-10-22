"""Numerical-stagnation exit for the Riemannian LM (#210).

Strict squared-norm acceptance at a float64 cost floor rejected every step,
the rejected-step rule ``lam -> 2 lam`` overflowed the damping, and the
pre-#210 loop evaluated nonfinite proposals until ``max_steps``. The loop now
stops on a rejected step from which no later step can be accepted -- the
retracted proposal is bitwise the current point, or the step / damping is
nonfinite -- reporting ``status="stagnated"`` / ``OptimizerInfo.stagnated``
with ``done=False``.

Since #211 the acceptance test reads the cancellation-free reduction, so the
exact-tie floor fixture the issue was filed on *converges* (see
``test_riemannian_lm_accurate_reduction.py``). The stagnation gates here are
therefore the two remaining routes into the exit: a **null step** (the
optimum ``1 - 1e-17`` is not representable next to ``1.0``, so no proposal
can move the point while the scale-blind gradient test cannot pass) and a
**nonfinite** step / damping (NaN residual at the start). Also covered: the
traced ``args`` path, a budget exhausted before any stagnation (still
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

# Null-step fixture: r = [a (x - 1) + b, 1] with b / a = 1e-17. The exact
# minimiser x = 1 - 1e-17 rounds to 1.0, so from x = 1.0 the first LM step
# d = -b / (a (1 + 1e-3)) ~ -1e-17 retracts to exactly 1.0 again (a null
# step), while the gradient a b = 1e3 sits far above the stationarity
# threshold: no criterion can certify and no proposal can move the point.
_A, _B = 1e10, 1e-7


class _One2(ParameterSpace):
    x: jax.Array = on(Euclidean(2), default=jnp.ones(2))


class _One1(ParameterSpace):
    x: jax.Array = on(Euclidean(1), default=jnp.ones(1))


def _null_residual(x):
    return jnp.concatenate([_A * (x - 1.0) + _B, jnp.ones(1)])


class _Far(ParameterSpace):
    x: jax.Array = on(Euclidean(2), default=jnp.array([3.0, -2.0]))


_TARGET = jnp.array([0.25, 1.5])


def _far_residual(x):
    return jnp.concatenate([x - _TARGET, jnp.ones(1)])


def _solve(space, residual, tol=_TOL, **kw):
    p = space.point()
    _, _, spec = flatten_params_with_spec(p)
    ans, info = riemannian_lm(**tol, **kw)(residual, p, spec)
    return p, ans, info


class TestNullStepStagnation:
    """A proposal that cannot move the point: early, finite, uncertified."""

    @pytest.mark.parametrize("space", [_One2, _One1], ids=["2d", "1d"])
    def test_unrepresentable_optimum_stops_as_stagnated(self, space):
        p, ans, info = _solve(space, _null_residual, max_steps=400)
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        assert int(info.steps) == 1
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))
        assert np.all(np.isfinite(np.asarray(ans.x.array)))

    def test_nonfinite_damping_stops_at_once(self):
        # sqrt of a negative start: NaN residual and Jacobian, so the lambda
        # anchor and the first LM step are NaN. Pre-#210 this rejected (NaN
        # compares False) and doubled NaN for the whole budget.
        class _Neg(ParameterSpace):
            x: jax.Array = on(Euclidean(2), default=jnp.full(2, -1.0))

        def residual(x):
            return jnp.concatenate([jnp.sqrt(x), jnp.ones(1)])

        p, ans, info = _solve(_Neg, residual, max_steps=500)
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        assert int(info.steps) == 1
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))

    def test_budget_exhausted_before_stagnation_is_max_iterations(self):
        # ``stagnated`` is not a budget flag: a converging solve cut off after
        # one step reports max_iterations with the flag False.
        _, _, info = _solve(_Far, _far_residual, tol={}, max_steps=1)
        assert info.status == "max_iterations"
        assert bool(info.stagnated) is False
        assert bool(info.done) is False
        assert int(info.steps) == 1


class TestStagnatedRidesTracedArgsPath:
    """The flag is a traced carry value: it must survive the memoised jitted
    solve (#124 ``args`` channel) and an outer ``jax.jit``, where ``status``
    collapses to ``"traced"`` and only the traced fields carry information.
    """

    @staticmethod
    def _kernel(x, b):
        return jnp.concatenate([_A * (x - 1.0) + b, jnp.ones(1)])

    def test_args_path_eager_outputs(self):
        p = _One2.point()
        _, _, spec = flatten_params_with_spec(p)
        solver = riemannian_lm(**_TOL, max_steps=400)
        ans, info = solver(self._kernel, p, spec, args=jnp.asarray(_B))
        assert info.status == "stagnated"
        assert bool(info.stagnated) is True
        assert bool(info.done) is False
        assert int(info.steps) == 1
        np.testing.assert_array_equal(np.asarray(ans.x.array), np.asarray(p.x.array))

    def test_under_outer_jit_flag_is_traced_and_true(self):
        p = _One2.point()
        _, _, spec = flatten_params_with_spec(p)
        solver = riemannian_lm(**_TOL, max_steps=400)
        statuses: list[str] = []

        @jax.jit
        def run(b):
            _, info = solver(self._kernel, p, spec, args=b)
            statuses.append(info.status)
            return info.stagnated, info.done, info.steps

        stagnated, done, steps = run(jnp.asarray(_B))
        assert statuses == ["traced"]
        assert bool(stagnated) is True
        assert bool(done) is False
        assert int(steps) == 1


class TestHealthySolvesUntouched:
    """A converging solve never rejects a null step, so it certifies exactly
    as before and the flag stays False."""

    def test_euclidean_least_squares_converges_without_flag(self):
        _, ans, info = _solve(_Far, _far_residual, tol={}, max_steps=100)
        assert info.status == "converged"
        assert bool(info.done) is True
        assert bool(info.stagnated) is False
        assert int(info.steps) < 10
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
