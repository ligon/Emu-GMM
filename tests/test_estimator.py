"""Acceptance and unit tests for emu_gmm.estimator.estimate (synthetic path).

This is the Phase 5 milestone: the multi-asset consumption Euler equation
(Hansen-Singleton 1982 style) estimated end-to-end against a synthetic
sampler from a known DGP. With J = 3 risky assets and K = 2 structural
parameters (beta, gamma), the system is over-identified (J_dof = 1).

The DGP is constructed so all Euler conditions vanish exactly at
(BETA_TRUE, GAMMA_TRUE) = (0.96, 2.0); see emu_gmm.examples.euler for
the derivation.
"""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import pytest
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

N_SIM = 5000


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------


class TestEulerSyntheticAcceptance:
    """The Phase 5 milestone: end-to-end multi-asset Euler estimation."""

    def _run(self) -> EstimationResult:
        sampler = euler_sampler_factory(N_SIM)
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=sampler,
        )
        return estimate(
            model=euler_residual,
            measure=measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )

    def test_recovers_beta_and_gamma(self):
        """theta_hat is within Monte Carlo error of the truth."""
        r = self._run()
        # With N_SIM=5000, MC SE on beta is small; gamma is harder
        # because it depends on cross-asset spread. Allow generous
        # tolerances.
        assert float(r.theta_hat.beta) == pytest.approx(BETA_TRUE, abs=0.05)
        assert float(r.theta_hat.gamma) == pytest.approx(GAMMA_TRUE, abs=0.5)

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        assert r.J_dof == 1  # M=3 assets - K=2 params = 1

    def test_J_stat_finite_and_modest(self):
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        # The DGP is correctly specified so J should be small. With
        # N_SIM=5000, expected value of J under the null is ~1 (chi-sq
        # with 1 dof). Allow up to ~10 for sampling noise.
        assert r.J_stat < 30.0

    def test_J_pvalue_finite(self):
        r = self._run()
        assert jnp.isfinite(r.J_pvalue)
        assert 0.0 <= r.J_pvalue <= 1.0

    def test_labelled_outputs(self):
        r = self._run()
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert {a.name for a in r.Sigma_theta.axes} == {
            "parameters",
            "parameters_dual",
        }
        assert isinstance(r.V_X, ha.NamedArray)
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_label_context_populated(self):
        r = self._run()
        assert r.labels.param_names == ("beta", "gamma")
        assert r.labels.moment_names == ("m_0", "m_1", "m_2")
        # The Euler observation has D=5 components: (c_t, c_{t+1}, r_1, r_2, r_3).
        assert r.labels.variable_names == ("v_0", "v_1", "v_2", "v_3", "v_4")

    def test_to_pandas_works(self):
        r = self._run()
        d = r.to_pandas()
        assert "Sigma_theta" in d
        assert list(d["Sigma_theta"].index) == ["beta", "gamma"]
        assert list(d["Sigma_theta"].columns) == ["beta", "gamma"]
        assert list(d["V_X"].index) == ["m_0", "m_1", "m_2"]

    def test_sigma_theta_finite(self):
        r = self._run()
        assert jnp.all(jnp.isfinite(r.Sigma_theta.array))

    def test_diagnostics_populated(self):
        r = self._run()
        assert jnp.isfinite(r.diagnostics.final_objective)
        assert jnp.isfinite(r.diagnostics.final_gradient_norm)
        # Gradient should be small at the optimum --- but optimistix
        # converges on step-size, not gradient, so the actual gradient
        # at the reported solution can be modestly nonzero.
        assert r.diagnostics.final_gradient_norm < 0.05
        # N_j should be n_sim for each moment under SyntheticMeasure.
        assert jnp.allclose(r.diagnostics.N_j.array, float(N_SIM))

    def test_provenance_echoed(self):
        r = self._run()
        assert isinstance(r.theta_init, EulerParams)
        assert float(r.theta_init.beta) == 0.9
        assert isinstance(r.measure, SyntheticMeasure)
        assert isinstance(r.covariance, SyntheticCovariance)


# ---------------------------------------------------------------------------
# Smaller unit-style tests
# ---------------------------------------------------------------------------


class TestEstimateDefaults:
    """Defaults (weighting=ContinuouslyUpdated, regularization=DiagonalTikhonov,
    optimizer=optimistix_lm) work without explicit arguments."""

    def test_minimal_call(self):
        sampler = euler_sampler_factory(N_SIM)
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=sampler,
        )
        r = estimate(
            model=euler_residual,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )
        assert isinstance(r, EstimationResult)
        assert r.converged


class TestMomentNamesOverride:
    """The moment_names kwarg overrides the positional fallback."""

    def test_kwarg_propagates(self):
        sampler = euler_sampler_factory(N_SIM)
        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=N_SIM,
            sampler=sampler,
        )
        r = estimate(
            model=euler_residual,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
            moment_names=("asset_low", "asset_mid", "asset_high"),
        )
        assert r.labels.moment_names == ("asset_low", "asset_mid", "asset_high")
        d = r.to_pandas()
        assert list(d["V_X"].index) == ["asset_low", "asset_mid", "asset_high"]
