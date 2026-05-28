"""Acceptance and unit tests for emu_gmm.estimator.estimate.

The first acceptance test (``test_normal_moments_recovers_truth``) is
the Phase 5 milestone: a structural-GMM estimation runs end-to-end and
recovers the true parameters from synthetic data.

Estimation problem: ``x ~ N(mu_true, sigma_sq_true)`` with three moment
conditions parameterised by ``theta = (mu, sigma_sq)``:

  E[x - mu]            = 0
  E[(x - mu)^2 - s^2]  = 0
  E[(x - mu)^3]        = 0

The third moment is identically zero for any symmetric distribution
and gives the over-identification (M=3, K=2, J_dof=1) needed for a
meaningful J-statistic.
"""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.estimator import estimate
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

# Ground truth.
MU_TRUE = 2.0
SIGMA_TRUE = 1.5
SIGMA_SQ_TRUE = SIGMA_TRUE**2  # 2.25

N_SIM = 5000


@jdc.pytree_dataclass
class NormalParams:
    mu: float
    sigma_sq: float


def normal_moments(x, theta):
    """Three moment conditions identifying (mu, sigma_sq) of a normal.

    Symmetric => third moment is identically zero, providing the
    over-identifying restriction.
    """
    diff = x[0] - theta.mu
    return jnp.array(
        [
            diff,
            diff**2 - theta.sigma_sq,
            diff**3,
        ]
    )


def normal_sampler(key, theta):
    """Sample from N(MU_TRUE, SIGMA_SQ_TRUE).

    Sampler does NOT depend on theta: the DGP is the "data" we're
    trying to match, and theta is the structural parameter being
    estimated.
    """
    z = jax.random.normal(key, (N_SIM, 1))
    return z * SIGMA_TRUE + MU_TRUE


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------


class TestNormalMomentsAcceptance:
    """The Phase 5 milestone: end-to-end estimation recovers the truth."""

    def _run(self) -> EstimationResult:
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=normal_sampler,
        )
        return estimate(
            model=normal_moments,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )

    def test_recovers_mu_and_sigma_sq(self):
        """theta_hat is within Monte Carlo error of the truth."""
        r = self._run()
        # With N_SIM=5000, Monte Carlo SE on mu is sigma/sqrt(N) ~ 0.02.
        # Allow 0.1 tolerance.
        assert float(r.theta_hat.mu) == pytest.approx(MU_TRUE, abs=0.1)
        assert float(r.theta_hat.sigma_sq) == pytest.approx(SIGMA_SQ_TRUE, abs=0.2)

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_stat_finite_and_small(self):
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        # At the truth, all three moments are zero in expectation. With
        # finite N_SIM, Monte Carlo error gives a small but nonzero J.
        assert r.J_stat < 50.0

    def test_J_dof_is_one(self):
        r = self._run()
        assert r.J_dof == 1  # M=3 - K=2 = 1

    def test_J_pvalue_finite(self):
        r = self._run()
        assert jnp.isfinite(r.J_pvalue)
        assert 0.0 <= r.J_pvalue <= 1.0

    def test_labelled_outputs(self):
        r = self._run()
        # Sigma_theta carries Params / ParamsDual labels.
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert {a.name for a in r.Sigma_theta.axes} == {
            "parameters",
            "parameters_dual",
        }
        # V_X carries Moments / MomentsDual labels.
        assert isinstance(r.V_X, ha.NamedArray)
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_label_context_populated(self):
        r = self._run()
        assert r.labels.param_names == ("mu", "sigma_sq")
        # Positional moment names since the model returns a plain array.
        assert r.labels.moment_names == ("m_0", "m_1", "m_2")
        # Variable names from the observation dimension.
        assert r.labels.variable_names == ("v_0",)

    def test_to_pandas_works(self):
        r = self._run()
        d = r.to_pandas()
        assert "Sigma_theta" in d
        assert list(d["Sigma_theta"].index) == ["mu", "sigma_sq"]
        assert list(d["Sigma_theta"].columns) == ["mu", "sigma_sq"]
        assert list(d["V_X"].index) == ["m_0", "m_1", "m_2"]

    def test_sigma_theta_finite(self):
        r = self._run()
        assert jnp.all(jnp.isfinite(r.Sigma_theta.array))

    def test_diagnostics_populated(self):
        r = self._run()
        assert jnp.isfinite(r.diagnostics.final_objective)
        assert jnp.isfinite(r.diagnostics.final_gradient_norm)
        # Gradient should be small at the optimum.
        assert r.diagnostics.final_gradient_norm < 1e-2
        # N_j should be n_sim for each moment under SyntheticMeasure.
        assert jnp.allclose(r.diagnostics.N_j.array, float(N_SIM))

    def test_provenance_echoed(self):
        r = self._run()
        # theta_init preserved.
        assert isinstance(r.theta_init, NormalParams)
        assert float(r.theta_init.mu) == 0.0
        # Strategy objects echoed.
        assert isinstance(r.measure, SyntheticMeasure)
        assert isinstance(r.covariance, SyntheticCovariance)


# ---------------------------------------------------------------------------
# Smaller unit-style tests
# ---------------------------------------------------------------------------


class TestEstimateDefaults:
    """Verify the defaults: weighting=ContinuouslyUpdated, regularization
    =DiagonalTikhonov, optimizer=optimistix_lm work without explicit
    arguments."""

    def test_minimal_call(self):
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=normal_sampler,
        )
        r = estimate(
            model=normal_moments,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )
        assert isinstance(r, EstimationResult)
        assert r.converged


class TestMomentNamesOverride:
    """The moment_names kwarg should override the positional fallback."""

    def test_kwarg_propagates(self):
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=normal_sampler,
        )
        r = estimate(
            model=normal_moments,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
            moment_names=("mean", "var", "skew"),
        )
        assert r.labels.moment_names == ("mean", "var", "skew")
        # Labels propagate into the V_X / N_j DataFrames.
        d = r.to_pandas()
        assert list(d["V_X"].index) == ["mean", "var", "skew"]
