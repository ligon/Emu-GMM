"""Analytical round-trip acceptance test for emu_gmm.estimator.estimate.

This is the Phase 6 milestone: pairing the multi-asset Euler structural
model with an analytical closed-form expectation and verifying that
estimate(...) recovers the truth exactly (up to floating-point tolerance)
when there is no Monte Carlo noise.

The closed-form expectation comes from
:func:`emu_gmm.examples.euler.euler_analytical_expectation`, which solves
E[(c'/c)^{-gamma} (1 + r_j)] in closed form under the DGP's log-normal
consumption growth and linear-in-shock returns. At (BETA_TRUE,
GAMMA_TRUE) it evaluates to zero exactly.
"""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import pytest
from emu_gmm.covariance import AnalyticalCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    N_ASSETS,
    EulerParams,
    euler_analytical_expectation,
    euler_residual,
)
from emu_gmm.measures import AnalyticalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated


def identity_covariance(model, theta):
    """Identity ``V``: simplest positive-definite choice.

    The round-trip recovery property does not depend on the specific
    ``V`` as long as it is positive definite. The J-statistic limit
    law depends on V, but this acceptance test is about identification,
    not J-distribution.
    """
    del model, theta
    return jnp.eye(N_ASSETS)


# Tolerance for "floating-point precision" recovery. The analytical
# expectations are exact (no Monte Carlo noise), so the only error
# floor is the optimiser's tolerance + round-off in evaluating the
# closed-form moment expressions.
FP_TOL = 1e-5


class TestEulerAnalyticalRoundTripAcceptance:
    """Phase 6 milestone: closed-form Euler identification recovers truth."""

    def _run(self) -> EstimationResult:
        measure = AnalyticalMeasure(expectation_fn=euler_analytical_expectation)
        covariance = AnalyticalCovariance(covariance_fn=identity_covariance)
        return estimate(
            model=euler_residual,
            measure=measure,
            covariance=covariance,
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10, max_steps=200),
            theta_init=EulerParams(beta=0.8, gamma=1.0),
        )

    def test_recovers_beta_to_fp_precision(self):
        r = self._run()
        assert float(r.theta_hat.beta) == pytest.approx(BETA_TRUE, abs=FP_TOL)

    def test_recovers_gamma_to_fp_precision(self):
        r = self._run()
        assert float(r.theta_hat.gamma) == pytest.approx(GAMMA_TRUE, abs=FP_TOL)

    def test_final_objective_near_zero(self):
        """At the truth the whitened residual is exactly zero; any
        deviation is purely numerical."""
        r = self._run()
        assert r.diagnostics.final_objective < 1e-10

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        # M = N_ASSETS = 3, K = 2 -> J_dof = 1.
        assert r.J_dof == 1


class TestLabelledOutputs:
    """Sanity checks on the labelled output construction."""

    def test_sigma_theta_labelled(self):
        r = TestEulerAnalyticalRoundTripAcceptance()._run()
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert {a.name for a in r.Sigma_theta.axes} == {
            "parameters",
            "parameters_dual",
        }

    def test_v_x_labelled(self):
        r = TestEulerAnalyticalRoundTripAcceptance()._run()
        assert isinstance(r.V_X, ha.NamedArray)
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_to_pandas_param_names(self):
        r = TestEulerAnalyticalRoundTripAcceptance()._run()
        d = r.to_pandas()
        assert list(d["Sigma_theta"].index) == ["beta", "gamma"]
        assert list(d["Sigma_theta"].columns) == ["beta", "gamma"]


class TestProvenance:
    def test_strategies_echoed(self):
        r = TestEulerAnalyticalRoundTripAcceptance()._run()
        assert isinstance(r.theta_init, EulerParams)
        assert isinstance(r.measure, AnalyticalMeasure)
        assert isinstance(r.covariance, AnalyticalCovariance)
