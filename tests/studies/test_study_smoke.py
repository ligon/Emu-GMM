"""Layer-3 composition + end-to-end smokes.

The 30-rep calibration check here is a SMOKE TEST with deliberately
generous bands --- it proves the driver wiring (records -> summaries ->
StudyResult) on the bundled Euler DGP, and is NOT the #130 validation
study (which needs hundreds of reps and committed reports under
docs/validation/).

The masked-DGP smoke is the commitment-9 guard at the driver level: a
balanced ``mask=ones`` fixture cannot exercise the per-coordinate N_j
bookkeeping, so the driver must demonstrably support an unequal-N_j
measure end-to-end.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from emu_gmm import IIDCovariance, build_estimator
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.studies import StudyResult, monte_carlo_study

THETA_TRUE = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)


def _theta0() -> EulerParams:
    return EulerParams(beta=0.9, gamma=1.0)


def _balanced_dgp(n: int):
    sampler = euler_sampler_factory(n)

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        x = sampler(key, THETA_TRUE)
        return EmpiricalMeasure(x=x, mask=jnp.ones((n, 3)), weights=jnp.ones(n))

    return dgp


def _masked_dgp(n: int):
    """Unequal N_j by design: moment 0 fully observed, moment 1 on the
    first 60% of rows, moment 2 on the first 80% (deterministic MCAR
    pattern; same shape every rep so the kernel path is retained)."""
    sampler = euler_sampler_factory(n)
    rows = jnp.arange(n)
    mask = jnp.stack(
        [
            jnp.ones(n),
            (rows < int(0.6 * n)).astype(jnp.float64),
            (rows < int(0.8 * n)).astype(jnp.float64),
        ],
        axis=1,
    )

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        x = sampler(key, THETA_TRUE)
        return EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(n))

    return dgp


def _run(n: int):
    return build_estimator(
        euler_residual,
        measure=_balanced_dgp(n)(jax.random.PRNGKey(123)),
        covariance=IIDCovariance(),
        parameters=_theta0(),
    )


class TestEulerCalibrationSmoke:
    def test_thirty_rep_smoke(self, capsys):
        """30 reps, N=500: generous-band calibration smoke."""
        n_reps, n = 30, 500
        t0 = time.perf_counter()
        study = monte_carlo_study(
            _run(n),
            _balanced_dgp(n),
            n_reps=n_reps,
            key=jax.random.PRNGKey(2026),
            theta_init=_theta0(),
            theta0=THETA_TRUE,
        )
        wall = time.perf_counter() - t0
        print(f"\n30-rep Euler smoke wall time: {wall:.2f} s")

        assert isinstance(study, StudyResult)
        assert study.n_reps == n_reps
        assert study.n_used + study.n_excluded == n_reps
        # Essentially every rep should converge on this easy DGP.
        assert study.n_used >= n_reps - 2

        # Recovery: bias small in absolute terms (generous bands).
        b = study.bias_sd
        assert abs(b.bias[0]) < 0.02  # beta, truth 0.96
        assert abs(b.bias[1]) < 0.5  # gamma, truth 2.0
        # Analytic SE tracks MC SD within a factor of ~2 at 30 reps.
        assert np.all(b.se_ratio > 0.5) and np.all(b.se_ratio < 2.0)

        # Coverage at 95%: generous band per the smoke contract.
        assert np.all(study.coverage.coverage >= 0.85)
        assert np.all(study.coverage.coverage <= 1.0)

        # J-test size at alpha=0.05 within a generous band.
        i05 = study.size_power.alphas.index(0.05)
        assert study.size_power.reject_nominal[i05] <= 0.25
        # Adjusted == nominal when the ridge never binds; both sane.
        assert study.size_power.reject_adjusted[i05] <= 0.25

        # tau-binding column populated (frequency is a number in [0,1]).
        assert 0.0 <= study.tau_binding.binding_frequency <= 1.0
        assert study.tau_binding.tau_quantiles.shape == (5,)

        # J calibration: at 30 reps the KS-style deviation is noisy;
        # just demand it is not catastrophic.
        assert study.j_calibration.max_abs_deviation < 0.5
        assert study.j_calibration.J_dof == 1


class TestMaskedDGP:
    def test_unequal_nj_study_end_to_end(self):
        """The driver supports per-coordinate missingness (commitment 9):
        an unequal-N_j masked DGP runs end-to-end, converges, and yields
        finite summaries."""
        n_reps, n = 5, 600
        dgp = _masked_dgp(n)
        run = build_estimator(
            euler_residual,
            measure=dgp(jax.random.PRNGKey(123)),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        study = monte_carlo_study(
            run,
            dgp,
            n_reps=n_reps,
            key=jax.random.PRNGKey(3),
            theta_init=_theta0(),
            theta0=THETA_TRUE,
        )
        assert study.n_reps == n_reps
        assert study.n_used >= n_reps - 1  # masked DGP still easy
        used = study.records.converged_mask
        theta = np.asarray(study.records.records.theta_flat)[used]
        se = np.asarray(study.records.records.se)[used]
        assert np.isfinite(theta).all() and np.isfinite(se).all()
        assert np.isfinite(study.bias_sd.bias).all()
        assert np.isfinite(study.coverage.coverage).all()
        # Recovery still works under missingness (loose; 5 reps).
        assert abs(study.bias_sd.bias[0]) < 0.05

    def test_masked_dgp_really_is_unbalanced(self):
        """Fixture guard: the mask must actually produce unequal N_j."""
        m = _masked_dgp(600)(jax.random.PRNGKey(0))
        n_j = np.asarray(m.mask).sum(axis=0)
        assert n_j[0] == 600 and n_j[1] == 360 and n_j[2] == 480


class TestStudyResultDelegation:
    def test_study_composes_replicate_and_summarizers(self):
        """Layer 3 is delegation only: recomputing each summary from
        study.records reproduces the packaged values exactly."""
        from emu_gmm.studies import (
            bias_sd,
            coverage,
            j_calibration,
            size_power,
            tau_binding,
        )

        n_reps, n = 4, 400
        study = monte_carlo_study(
            _run(n),
            _balanced_dgp(n),
            n_reps=n_reps,
            key=jax.random.PRNGKey(5),
            theta_init=_theta0(),
            theta0=THETA_TRUE,
        )
        np.testing.assert_array_equal(
            study.bias_sd.bias, bias_sd(study.records, THETA_TRUE).bias
        )
        np.testing.assert_array_equal(
            study.coverage.coverage,
            coverage(study.records, THETA_TRUE, level=0.95).coverage,
        )
        np.testing.assert_array_equal(
            study.size_power.reject_nominal,
            size_power(study.records).reject_nominal,
        )
        assert (
            study.tau_binding.binding_frequency
            == tau_binding(study.records).binding_frequency
        )
        np.testing.assert_array_equal(
            study.j_calibration.ecdf, j_calibration(study.records).ecdf
        )
