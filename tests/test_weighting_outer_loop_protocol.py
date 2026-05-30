"""Tests for the :attr:`WeightingStrategy.requires_outer_loop` protocol lift (#53).

The estimator used to dispatch to the iterated-weighting outer loop via
``isinstance(weighting, IteratedWeighting)``. That coupling makes
:class:`IteratedWeighting` a privileged citizen and forces third-party
weighting strategies that also need an outer loop to monkey-patch or
fork the estimator.

The fix exposes two optional protocol points on
:class:`~emu_gmm.types.WeightingStrategy`:

- ``requires_outer_loop``: ``False`` (default) for Identity / Fixed /
  CU; ``True`` for IteratedWeighting.
- ``outer_loop_driver(...)``: the custom Python driver; called by the
  estimator when ``requires_outer_loop`` is ``True``.

These tests pin:

1. ``IteratedWeighting`` advertises ``requires_outer_loop = True`` and
   the other built-ins advertise ``False``.
2. The estimator no longer references ``IteratedWeighting`` by class
   in a dispatch branch.
3. A user-defined strategy that sets ``requires_outer_loop = True``
   and supplies its own ``outer_loop_driver`` runs end-to-end through
   :func:`emu_gmm.estimate` and gets the same result as
   ``IteratedWeighting`` on the same problem.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    Fixed,
    Identity,
    IteratedWeighting,
    SyntheticCovariance,
    SyntheticMeasure,
    estimate,
    optimistix_lm,
)
from emu_gmm._internal import params as params_mod
from emu_gmm._internal.cholesky import cholesky, forward_solve
from emu_gmm.types import (
    CovarianceStrategy,
    Measure,
    Optimizer,
    OptimizerInfo,
    ParamsLike,
    StructuralModel,
)
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Fixtures: a smooth, well-specified mean-only example. We reuse it for
# both the IteratedWeighting baseline and the third-party-driver case.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _MuParams:
    mu: float


def _sampler(key, theta):
    import jax

    return jax.random.normal(key, shape=(256, 2)) + jnp.array([2.0, 0.0])


def _psi(x, theta):
    return jnp.array([theta.mu - x[0], (x[0] - theta.mu) ** 2 - 1.0])


def _measure(seed: int = 0) -> SyntheticMeasure:
    import jax

    return SyntheticMeasure(
        key=jax.random.PRNGKey(seed),
        n_sim=256,
        sampler=_sampler,
    )


# ---------------------------------------------------------------------------
# Section 1: protocol attributes on built-in weightings.
# ---------------------------------------------------------------------------


class TestBuiltinsAdvertiseFlag:
    def test_identity_does_not_require_outer_loop(self):
        assert getattr(Identity(), "requires_outer_loop", None) is False

    def test_fixed_does_not_require_outer_loop(self):
        V0 = jnp.eye(2)
        assert getattr(Fixed.from_V0(V0), "requires_outer_loop", None) is False

    def test_continuously_updated_does_not_require_outer_loop(self):
        assert getattr(ContinuouslyUpdated(), "requires_outer_loop", None) is False

    def test_iterated_weighting_requires_outer_loop(self):
        w = IteratedWeighting(weighting_iterations=3, weighting_tol=1e-6)
        assert getattr(w, "requires_outer_loop", None) is True

    def test_iterated_weighting_exposes_outer_loop_driver(self):
        w = IteratedWeighting(weighting_iterations=3, weighting_tol=1e-6)
        assert callable(getattr(w, "outer_loop_driver", None))


# ---------------------------------------------------------------------------
# Section 2: estimator dispatch is by attribute, not by isinstance.
# ---------------------------------------------------------------------------


class TestNoIsinstanceDispatch:
    """The estimator's dispatch must not branch on IteratedWeighting."""

    def test_estimator_source_has_no_isinstance_iteratedweighting(self):
        import inspect

        from emu_gmm import estimator as est_mod

        src = inspect.getsource(est_mod)
        assert "isinstance(weighting, IteratedWeighting" not in src
        # And the lifted dispatch by attribute must be present.
        assert "requires_outer_loop" in src


# ---------------------------------------------------------------------------
# Section 3: a third-party strategy with requires_outer_loop=True works.
# ---------------------------------------------------------------------------


class _ThirdPartyIterated:
    """A third-party iterated weighting strategy.

    Demonstrates that the estimator's outer-loop dispatch is protocol-
    based, not :class:`IteratedWeighting`-specific. Implements
    ``whitening_residual`` (the CU fallback) and the optional
    ``outer_loop_driver`` with the same V-refresh logic the built-in
    :class:`IteratedWeighting` uses, but in a different class so the
    ``isinstance(..., IteratedWeighting)`` test would fail to match.
    """

    requires_outer_loop = True

    def __init__(self, weighting_iterations: int = 5, weighting_tol: float = 1e-6):
        self.weighting_iterations = weighting_iterations
        self.weighting_tol = weighting_tol

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        # CU fallback (identical to ContinuouslyUpdated.whitening_residual).
        L = cholesky(V)
        return forward_solve(L, m)

    def outer_loop_driver(
        self,
        model: StructuralModel,
        measure: Measure,
        covariance: CovarianceStrategy,
        theta_init_flat: Float[Array, " K"],
        treedef: Any,
        *,
        make_residual_fn: Any,
        cu_residual_fn: Any,
        apply_ridge: Any,
        optimizer: Optimizer,
    ) -> tuple[Float[Array, " K"], OptimizerInfo, str]:
        theta_k_flat = jnp.asarray(theta_init_flat)
        last_info: OptimizerInfo | None = None
        total_inner_steps = 0
        outer_status = "max_iterations"
        for _k in range(int(self.weighting_iterations)):
            theta_k = params_mod.unflatten_params(theta_k_flat, treedef)
            V_k = covariance.covariance(model, theta_k, measure)
            V_star_k = apply_ridge(V_k)
            fixed_k = Fixed.from_V0(V_star_k)
            inner_residual = make_residual_fn(fixed_k)
            theta_next_flat, info_k = optimizer(inner_residual, theta_k_flat)
            last_info = info_k
            try:
                total_inner_steps += int(info_k.steps)
            except (TypeError, ValueError):
                pass
            delta = jnp.linalg.norm(theta_next_flat - theta_k_flat)
            theta_k_flat = theta_next_flat
            if float(delta) < float(self.weighting_tol) * max(
                float(jnp.linalg.norm(theta_next_flat)), 1e-12
            ):
                outer_status = "converged"
                break
        assert last_info is not None
        y_final = cu_residual_fn(theta_k_flat)
        final_objective_cu = 0.5 * jnp.sum(y_final * y_final)
        final_info = OptimizerInfo(
            steps=total_inner_steps,
            status=outer_status,
            final_objective=final_objective_cu,
            backend=last_info.backend,
        )
        return theta_k_flat, final_info, outer_status


class TestThirdPartyOuterLoopDriver:
    """An ad-hoc strategy with ``requires_outer_loop = True`` runs through.

    The point of the protocol lift is that the estimator does not care
    whether the strategy is :class:`IteratedWeighting` or a different
    class entirely --- if it advertises ``requires_outer_loop = True``
    and supplies ``outer_loop_driver``, the estimator delegates.
    """

    def test_runs_end_to_end_and_matches_builtin(self):
        measure = _measure()
        cov = SyntheticCovariance()
        # Start near the truth (mu = 2). With Identity in the
        # ContinuouslyUpdated trial we'd never need an outer loop;
        # starting near the truth keeps the iterated path on its happy
        # path so we can compare apples to apples.
        theta0 = _MuParams(mu=2.0)
        opt = optimistix_lm(rtol=1e-10, atol=1e-10)

        r_third = estimate(
            model=_psi,
            measure=measure,
            covariance=cov,
            weighting=_ThirdPartyIterated(weighting_iterations=10, weighting_tol=1e-8),
            optimizer=opt,
            theta_init=theta0,
        )
        r_builtin = estimate(
            model=_psi,
            measure=measure,
            covariance=cov,
            weighting=IteratedWeighting(weighting_iterations=10, weighting_tol=1e-8),
            optimizer=opt,
            theta_init=theta0,
        )
        # Same convergence + same theta_hat to many digits: the two
        # drivers run the exact same V-refresh on the exact same DGP.
        assert r_third.converged
        assert r_builtin.converged
        assert float(r_third.theta_hat.mu) == pytest.approx(
            float(r_builtin.theta_hat.mu), abs=1e-9
        )
        # Same J-stat too.
        assert float(r_third.J_stat) == pytest.approx(float(r_builtin.J_stat), abs=1e-9)
