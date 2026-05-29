"""Empirical-path acceptance test for emu_gmm.estimator.estimate.

This is the Phase 7 milestone: the multi-asset Euler model estimated
against pre-generated synthetic "data" fed through EmpiricalMeasure +
IIDCovariance. Not a real-data acceptance test (deferred until data
arrives) but a unit-flavored end-to-end check that the empirical path
works.

Truth: (BETA_TRUE, GAMMA_TRUE) = (0.96, 2.0). With N = 5000 draws,
recovery is expected within Monte Carlo error ~ sigma / sqrt(N).
"""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import pytest
from emu_gmm.covariance import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_data,
    euler_residual,
)
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

N_DATA = 5000


def _make_measure(seed: int = 0) -> EmpiricalMeasure:
    """Build EmpiricalMeasure from pre-generated Euler data."""
    x = euler_data(seed=seed, n=N_DATA)
    # mask of all-ones (no missingness) sized (N, M=3 moments)
    mask = jnp.ones((N_DATA, 3))
    weights = jnp.ones(N_DATA)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


class TestEulerEmpiricalAcceptance:
    """Phase 7 milestone: empirical-path recovery on pre-generated data."""

    def _run(self) -> EstimationResult:
        return estimate(
            model=euler_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )

    def test_recovers_beta(self):
        r = self._run()
        assert float(r.theta_hat.beta) == pytest.approx(BETA_TRUE, abs=0.05)

    def test_recovers_gamma(self):
        r = self._run()
        # gamma is harder than beta (depends on cross-asset spread); allow
        # generous tolerance.
        assert float(r.theta_hat.gamma) == pytest.approx(GAMMA_TRUE, abs=0.5)

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        assert r.J_dof == 1  # M=3, K=2

    def test_J_stat_finite_and_modest(self):
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        # Correct specification; J ~ chi^2_1 ~ 1 on average; allow up to
        # ~30 for sampling noise.
        assert r.J_stat < 30.0

    def test_labelled_outputs(self):
        r = self._run()
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert isinstance(r.V_X, ha.NamedArray)

    def test_label_context(self):
        r = self._run()
        assert r.labels.param_names == ("beta", "gamma")
        # Three moments from euler_residual.
        assert r.labels.moment_names == ("m_0", "m_1", "m_2")

    def test_diagnostics_N_j_is_sample_size(self):
        r = self._run()
        # All-ones mask + unit weights means N_j = N for every moment.
        assert jnp.allclose(r.diagnostics.N_j.array, float(N_DATA))

    def test_provenance_echoed(self):
        r = self._run()
        assert isinstance(r.measure, EmpiricalMeasure)
        assert isinstance(r.covariance, IIDCovariance)


class TestEmpiricalDefaults:
    """Minimal call (using framework defaults) works against empirical
    measure."""

    def test_minimal_call(self):
        r = estimate(
            model=euler_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )
        assert isinstance(r, EstimationResult)
        assert r.converged
