"""Acceptance and unit tests for the analytical-path identification
round-trip.

The Phase 6 acceptance test (``test_recovers_truth_to_fp_precision``)
mirrors the Phase 5 normal-moments test but pairs
:class:`AnalyticalMeasure` + :class:`AnalyticalCovariance` instead of
the Monte Carlo equivalents. Because the expectations are evaluated
in closed form, there is no sampling noise: the optimiser should
recover the ground-truth parameters to within floating-point
precision, not merely to within a Monte Carlo tolerance.

Estimation problem: ``x ~ N(MU_TRUE, SIGMA_SQ_TRUE)`` with three
moment conditions parameterised by ``theta = (mu, sigma_sq)``::

    m1(x, theta) = x - theta.mu
    m2(x, theta) = (x - theta.mu)**2 - theta.sigma_sq
    m3(x, theta) = (x - theta.mu)**3

Let ``delta = MU_TRUE - theta.mu``. Under ``x ~ N(MU_TRUE, SIGMA_SQ_TRUE)``
the closed-form population expectations are::

    E[m1] = delta
    E[m2] = delta**2 + SIGMA_SQ_TRUE - theta.sigma_sq
    E[m3] = delta**3 + 3 * delta * SIGMA_SQ_TRUE

(The last uses the third central moment of a normal being zero.) All
three vanish iff ``theta = (MU_TRUE, SIGMA_SQ_TRUE)``.

The covariance is taken as the identity. This is an identification
check, not a J-statistic distribution check: an identity ``V`` is the
simplest legal choice and makes the optimisation objective a clean
``||m(theta)||**2`` quadratic in the moment vector. The estimator
recovery property is independent of ``V`` as long as ``V`` is positive
definite.
"""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm.covariance.analytical import AnalyticalCovariance
from emu_gmm.estimator import estimate
from emu_gmm.measures.analytical import AnalyticalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

# Ground truth.
MU_TRUE = 2.0
SIGMA_SQ_TRUE = 2.25


@jdc.pytree_dataclass
class NormalParams:
    mu: float
    sigma_sq: float


def normal_moments(x, theta):
    """Three moment conditions identifying (mu, sigma_sq) of a normal.

    Same per-observation residual as the Phase 5 acceptance test. The
    analytical measure does not actually call this with sample
    observations during expectation evaluation; the framework still
    invokes it once on a probe observation to detect label-bearing
    NamedArray returns. We supply a benign value (``x = [0.0]``) for
    that probe via the analytical measure's lack of a ``_draws``
    interface --- the estimator falls through to ``None``.
    """
    diff = x[0] - theta.mu
    return jnp.array(
        [
            diff,
            diff**2 - theta.sigma_sq,
            diff**3,
        ]
    )


def analytical_expectation(model, theta):
    """Closed-form E_mu[psi] under x ~ N(MU_TRUE, SIGMA_SQ_TRUE).

    With delta = MU_TRUE - theta.mu:
        E[diff]    = delta
        E[diff^2] = delta**2 + SIGMA_SQ_TRUE
        E[diff^3] = delta**3 + 3 * delta * SIGMA_SQ_TRUE  (third central
                                                           moment of N
                                                           is zero)
    """
    del model
    delta = MU_TRUE - theta.mu
    return jnp.array(
        [
            delta,
            delta**2 + SIGMA_SQ_TRUE - theta.sigma_sq,
            delta**3 + 3.0 * delta * SIGMA_SQ_TRUE,
        ]
    )


def identity_covariance(model, theta):
    """Identity ``V``: simplest positive-definite choice.

    This is an identification round-trip check, not a J-statistic
    distribution check. An identity ``V`` is the simplest valid choice;
    the recovery property of the estimator does not depend on the
    specific ``V`` as long as it is positive definite.
    """
    del model, theta
    return jnp.eye(3)


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------


# Tolerance for "floating-point precision" recovery. The analytical
# expectations are exact (no Monte Carlo noise), so the only error
# floor is the optimiser's tolerance + the round-off in computing
# delta**3, etc. Empirically the LM solver delivers recovery well
# below 1e-8 from the (0.0, 1.0) starting point.
FP_TOL = 1e-6


class TestAnalyticalRoundTripAcceptance:
    """Phase 6 milestone: closed-form identification recovers truth exactly."""

    def _run(self) -> EstimationResult:
        measure = AnalyticalMeasure(expectation_fn=analytical_expectation)
        covariance = AnalyticalCovariance(covariance_fn=identity_covariance)
        return estimate(
            model=normal_moments,
            measure=measure,
            covariance=covariance,
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10, max_steps=200),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )

    def test_recovers_mu_to_fp_precision(self):
        """theta_hat.mu equals MU_TRUE within floating-point tolerance."""
        r = self._run()
        assert float(r.theta_hat.mu) == pytest.approx(MU_TRUE, abs=FP_TOL)

    def test_recovers_sigma_sq_to_fp_precision(self):
        """theta_hat.sigma_sq equals SIGMA_SQ_TRUE within FP tolerance."""
        r = self._run()
        assert float(r.theta_hat.sigma_sq) == pytest.approx(SIGMA_SQ_TRUE, abs=FP_TOL)

    def test_final_objective_near_zero(self):
        """At the truth the whitened residual is exactly zero;
        any deviation is purely numerical."""
        r = self._run()
        assert r.diagnostics.final_objective < 1e-10

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        # M = 3, K = 2 -> J_dof = 1.
        assert r.J_dof == 1


# ---------------------------------------------------------------------------
# Smaller unit-style tests
# ---------------------------------------------------------------------------


class TestLabelledOutputs:
    """The estimator wraps matrix outputs in labelled NamedArrays
    regardless of whether the measure is analytical or synthetic."""

    def _run(self) -> EstimationResult:
        measure = AnalyticalMeasure(expectation_fn=analytical_expectation)
        covariance = AnalyticalCovariance(covariance_fn=identity_covariance)
        return estimate(
            model=normal_moments,
            measure=measure,
            covariance=covariance,
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )

    def test_sigma_theta_is_namedarray(self):
        r = self._run()
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert {a.name for a in r.Sigma_theta.axes} == {
            "parameters",
            "parameters_dual",
        }

    def test_v_x_is_namedarray(self):
        r = self._run()
        assert isinstance(r.V_X, ha.NamedArray)
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_label_context_populated(self):
        r = self._run()
        assert r.labels.param_names == ("mu", "sigma_sq")
        # Positional moment names: AnalyticalMeasure does not expose a
        # sample observation, so the estimator cannot probe the model
        # for a NamedArray return.
        assert r.labels.moment_names == ("m_0", "m_1", "m_2")

    def test_to_pandas_works(self):
        r = self._run()
        d = r.to_pandas()
        assert list(d["Sigma_theta"].index) == ["mu", "sigma_sq"]
        assert list(d["V_X"].index) == ["m_0", "m_1", "m_2"]


class TestProvenance:
    """Strategy objects supplied to estimate() are echoed in the result."""

    def test_strategies_echoed(self):
        measure = AnalyticalMeasure(expectation_fn=analytical_expectation)
        covariance = AnalyticalCovariance(covariance_fn=identity_covariance)
        r = estimate(
            model=normal_moments,
            measure=measure,
            covariance=covariance,
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )
        assert isinstance(r.measure, AnalyticalMeasure)
        assert isinstance(r.covariance, AnalyticalCovariance)
