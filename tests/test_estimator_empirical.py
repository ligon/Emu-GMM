"""End-to-end acceptance test on the empirical path.

Mirrors ``tests/test_estimator.py`` but routes through
:class:`emu_gmm.measures.empirical.EmpiricalMeasure` plus
:class:`emu_gmm.covariance.iid.IIDCovariance` rather than the synthetic
demo path. The data is pre-generated as a fixed JAX array (not produced
by a sampler at trace time), so the estimator runs against materialised
observations exactly as it would against real survey data.

Estimation problem: ``x ~ N(mu_true, sigma_sq_true)`` with three moment
conditions parameterised by ``theta = (mu, sigma_sq)``:

  E[x - mu]            = 0
  E[(x - mu)^2 - s^2]  = 0
  E[(x - mu)^3]        = 0

Same M=3, K=2, J_dof=1 over-identifying setup as the Phase 5
acceptance test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

# Ground truth.
MU_TRUE = 2.0
SIGMA_TRUE = 1.5
SIGMA_SQ_TRUE = SIGMA_TRUE**2  # 2.25

N_OBS = 5000


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


def _make_data(seed: int = 0) -> jnp.ndarray:
    """Pre-generate ``N_OBS`` draws from ``N(MU_TRUE, SIGMA_SQ_TRUE)``.

    The data is a fixed ``(N_OBS, 1)`` array; nothing about the
    optimisation traces through the sampler.
    """
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (N_OBS, 1))
    return z * SIGMA_TRUE + MU_TRUE


# ---------------------------------------------------------------------------


class TestEmpiricalAcceptance:
    """The Phase 7 milestone: the empirical path recovers the truth."""

    def _run(self) -> EstimationResult:
        x = _make_data(seed=0)
        # Full mask (no missingness), unit weights.
        measure = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N_OBS, 3)),  # three moments
            weights=jnp.ones(N_OBS),
        )
        return estimate(
            model=normal_moments,
            measure=measure,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )

    def test_recovers_mu_and_sigma_sq(self):
        r = self._run()
        # Monte Carlo SE on mu is sigma/sqrt(N) ~ 0.02 with N=5000.
        assert float(r.theta_hat.mu) == pytest.approx(MU_TRUE, abs=0.1)
        assert float(r.theta_hat.sigma_sq) == pytest.approx(SIGMA_SQ_TRUE, abs=0.2)

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        assert r.J_dof == 1  # M=3 - K=2 = 1

    def test_J_stat_finite_and_small(self):
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        assert r.J_stat < 50.0

    def test_J_pvalue_in_unit_interval(self):
        r = self._run()
        assert jnp.isfinite(r.J_pvalue)
        assert 0.0 <= r.J_pvalue <= 1.0

    def test_sigma_theta_finite(self):
        r = self._run()
        assert jnp.all(jnp.isfinite(r.Sigma_theta.array))

    def test_diagnostics_populated(self):
        r = self._run()
        assert jnp.isfinite(r.diagnostics.final_objective)
        assert jnp.isfinite(r.diagnostics.final_gradient_norm)
        # Gradient should be small at the optimum (looser than synthetic
        # case because the empirical sample is finite and not adapted at
        # trace time).
        assert r.diagnostics.final_gradient_norm < 1e-2
        # N_j for the empirical measure with unit weights is just N_OBS
        # (since the mask is all-ones).
        assert jnp.allclose(r.diagnostics.N_j.array, float(N_OBS))

    def test_provenance_echoed(self):
        r = self._run()
        assert isinstance(r.theta_init, NormalParams)
        assert float(r.theta_init.mu) == 0.0
        assert isinstance(r.measure, EmpiricalMeasure)
        assert isinstance(r.covariance, IIDCovariance)


# ---------------------------------------------------------------------------


class TestEmpiricalDefaults:
    """Verify the defaults still apply on the empirical path."""

    def test_minimal_call(self):
        x = _make_data(seed=1)
        measure = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N_OBS, 3)),
            weights=jnp.ones(N_OBS),
        )
        r = estimate(
            model=normal_moments,
            measure=measure,
            covariance=IIDCovariance(),
            theta_init=NormalParams(mu=0.0, sigma_sq=1.0),
        )
        assert isinstance(r, EstimationResult)
        assert r.converged
