"""Accurate step acceptance for the Riemannian LM (#211).

The acceptance test reads the cost reduction as ``(r - r_new)'(r + r_new)``
-- algebraically ``||r||^2 - ||r_new||^2`` -- instead of the difference of
the two rounded squared norms, which cannot resolve a reduction below
``eps * cost``. The #210 fixture ``Q(x) = 1 + x'x`` at ``x ~ 1e-9`` is the
motivating case: both rounded costs are exactly ``1.0`` so every step was
rejected as a tie and the solve stagnated, although the exact reduction of
the first step is ``~2e-18 > 0``. With the product form the step is taken
and the gradient test certifies two steps later.

Guards: the floor fixtures now converge; a healthy least-squares asked for a
tolerance below the old floor now converges instead of stagnating; a
nonfinite proposal is still rejected (NaN / -inf reduction compares False);
and the ``Positive`` rejected-start transient -- whose rejected overshoots
are far outside the rounding band where the two tests can disagree --
certifies at the same point as before.
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
    # Q(x) = 1 + x'x: both rounded costs are exactly 1.0 at x ~ 1e-9.
    return jnp.concatenate([x, jnp.ones(1)])


class _Far(ParameterSpace):
    x: jax.Array = on(Euclidean(2), default=jnp.array([3.0, -2.0]))


_TARGET = jnp.array([0.25, 1.5])


def _far_residual(x):
    # ||x - t||^2 + 1: under the old test the LM's last accepted step landed
    # at an error ~1e-9 (the eps * cost floor), above a 2e-10 gradient
    # threshold, so a 1e-10 tolerance could never certify.
    return jnp.concatenate([x - _TARGET, jnp.ones(1)])


def _solve(space, residual, tol=_TOL, **kw):
    p = space.point()
    _, _, spec = flatten_params_with_spec(p)
    ans, info = riemannian_lm(**tol, **kw)(residual, p, spec)
    return p, ans, info


class TestCostFloorNowConverges:
    """The #210 fixtures: genuine descent through the rounded-cost tie."""

    @pytest.mark.parametrize("space", [_P2, _P1], ids=["2d", "1d"])
    def test_float64_cost_floor_certifies(self, space):
        _, ans, info = _solve(space, _floor_residual, max_steps=400)
        assert info.status == "converged"
        assert bool(info.done) is True
        assert bool(info.stagnated) is False
        # First step accepted (x -> x lam / (1 + lam) ~ 1e-12), gradient test
        # passes on the next: two steps, not sixty-four, not max_steps.
        assert int(info.steps) <= 3
        x = np.asarray(ans.x.array)
        assert np.all(np.isfinite(x))
        assert np.all(np.abs(x) < 1e-12)

    def test_tolerance_below_the_old_floor_now_converges(self):
        _, ans, info = _solve(_Far, _far_residual, max_steps=400)
        assert info.status == "converged"
        assert bool(info.done) is True
        assert bool(info.stagnated) is False
        assert int(info.steps) < 10
        np.testing.assert_allclose(
            np.asarray(ans.x.array), np.asarray(_TARGET), atol=1e-10
        )

    def test_default_tolerance_solve_unchanged(self):
        # Far outside the rounding band the two acceptance tests agree
        # bitwise, so a healthy solve certifies exactly as before.
        _, ans, info = _solve(_Far, _far_residual, tol={}, max_steps=100)
        assert info.status == "converged"
        assert int(info.steps) < 10
        np.testing.assert_allclose(
            np.asarray(ans.x.array), np.asarray(_TARGET), atol=1e-8
        )


class TestNonfiniteProposalStillRejected:
    """A NaN / inf ``r_new`` gives a NaN / -inf reduction, which compares
    False under ``> 0`` exactly as ``NaN < cost`` did: the proposal is
    rejected and the iterate stays finite."""

    def test_overflowing_proposal_is_rejected_and_solve_recovers(self):
        # exp(10 x) - 1: the Gauss--Newton step from x = 10 overshoots to a
        # point where the residual overflows to inf; the LM must reject that
        # proposal, grow lam, and still converge to x = 0.
        class _Hot(ParameterSpace):
            x: jax.Array = on(Euclidean(1), default=jnp.full(1, 10.0))

        def residual(x):
            return jnp.concatenate([jnp.exp(10.0 * x) - 1.0, jnp.ones(1)])

        _, ans, info = _solve(_Hot, residual, tol={}, max_steps=400)
        x = np.asarray(ans.x.array)
        assert np.all(np.isfinite(x))
        assert info.status == "converged"
        assert abs(float(x[0])) < 1e-6


@jdc.pytree_dataclass
class _ScaleParams:
    sigma: jnp.ndarray
    __emu_manifolds__ = {"sigma": Positive()}


def _scale_residual(x, theta):
    s = theta.sigma
    return jnp.array([x[0] ** 2 - s**2, x[0] ** 4 - 3.0 * s**4])


class TestRejectedStartTransientUnchanged:
    """The ``Positive`` stuck start: the rejected overshoots raise the cost by
    orders of magnitude, far outside the rounding band, so the accurate
    reduction makes the same accept / reject decisions and the solve lands
    on the same certified point."""

    def test_positive_sub_true_start_certifies_as_before(self):
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
        assert int(info.steps) < 40
        assert float(r.theta_hat.sigma) == pytest.approx(1.5, abs=0.1)
