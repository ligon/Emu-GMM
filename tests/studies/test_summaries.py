"""Layer-2 summarizer unit tests: exact arithmetic on hand-built
records, the exclude-but-count convention, and the n_used=0 edge."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from emu_gmm.examples.euler import EulerParams
from emu_gmm.studies import (
    bias_sd,
    coverage,
    j_calibration,
    size_power,
    tau_binding,
)
from emu_gmm.types import FitRecord


def _records(
    theta,
    se,
    *,
    converged=None,
    J_pvalue=None,
    J_pvalue_adjusted=None,
    tau=None,
    binding=None,
    param_names=("beta", "gamma"),
    J_dof=1,
) -> FitRecord:
    """Hand-build a stacked FitRecord with a leading rep axis."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    n = theta.shape[0]
    ones = jnp.ones(n, dtype=jnp.float64)

    def vec(v, default):
        return jnp.asarray(v if v is not None else default, dtype=jnp.float64)

    p = vec(J_pvalue, 0.5 * ones)
    return FitRecord(
        theta_flat=theta,
        se=jnp.asarray(se, dtype=jnp.float64),
        J_stat=ones,
        J_pvalue=p,
        J_pvalue_adjusted=vec(J_pvalue_adjusted, p),
        converged=vec(converged, ones),
        tau_realised=vec(tau, 0.0 * ones),
        binding_ridge=vec(binding, 0.0 * ones),
        J_dof=J_dof,
        param_names=param_names,
    )


class TestBiasSD:
    def test_exact_arithmetic(self):
        theta = [[1.0, 4.0], [3.0, 8.0]]
        se = [[0.5, 1.0], [1.5, 3.0]]
        out = bias_sd(_records(theta, se), theta0=[1.5, 5.0])
        # mean = (2, 6); bias = (0.5, 1.0)
        np.testing.assert_allclose(out.bias, [0.5, 1.0])
        # ddof=1 SD of {1,3} and {4,8}: sqrt(2), sqrt(8)
        np.testing.assert_allclose(out.mc_sd, [np.sqrt(2.0), np.sqrt(8.0)])
        np.testing.assert_allclose(out.mean_se, [1.0, 2.0])
        np.testing.assert_allclose(
            out.se_ratio, [1.0 / np.sqrt(2.0), 2.0 / np.sqrt(8.0)]
        )
        assert out.n_used == 2 and out.n_excluded == 0
        assert out.param_names == ("beta", "gamma")

    def test_excludes_but_counts_non_converged(self):
        theta = [[1.0, 4.0], [3.0, 8.0], [100.0, -100.0]]
        se = [[0.5, 1.0], [1.5, 3.0], [9.0, 9.0]]
        out = bias_sd(_records(theta, se, converged=[1.0, 1.0, 0.0]), theta0=[1.5, 5.0])
        # Identical to the 2-rep case: rep 3 excluded, but counted.
        np.testing.assert_allclose(out.bias, [0.5, 1.0])
        assert out.n_used == 2 and out.n_excluded == 1

    def test_theta0_accepts_param_pytree(self):
        theta = [[1.0, 4.0], [3.0, 8.0]]
        se = [[1.0, 1.0], [1.0, 1.0]]
        out = bias_sd(_records(theta, se), theta0=EulerParams(beta=1.5, gamma=5.0))
        np.testing.assert_allclose(out.bias, [0.5, 1.0])

    def test_theta0_wrong_length_raises(self):
        with pytest.raises(ValueError, match="ambient parameter axis"):
            bias_sd(_records([[1.0, 2.0]], [[1.0, 1.0]]), theta0=[1.0, 2.0, 3.0])

    def test_all_non_converged_yields_nans(self):
        out = bias_sd(
            _records([[1.0, 2.0]], [[1.0, 1.0]], converged=[0.0]),
            theta0=[0.0, 0.0],
        )
        assert out.n_used == 0 and out.n_excluded == 1
        assert np.isnan(out.bias).all() and np.isnan(out.mc_sd).all()

    def test_single_used_rep_has_nan_mc_sd(self):
        out = bias_sd(_records([[1.0, 2.0]], [[0.1, 0.2]]), theta0=[1.0, 2.0])
        assert out.n_used == 1
        np.testing.assert_allclose(out.bias, [0.0, 0.0])
        assert np.isnan(out.mc_sd).all()  # ddof=1 undefined; no warning


class TestCoverage:
    def test_exact_coverage(self):
        z = scipy.stats.norm.ppf(0.975)
        # theta0 = (0, 0). Rep 1 covers both; rep 2 covers only coord 2.
        theta = [[0.5, -0.5], [3.0, 0.5]]
        se = [[1.0, 1.0], [1.0, 1.0]]
        out = coverage(_records(theta, se), theta0=[0.0, 0.0], level=0.95)
        assert abs(0.5) <= z and abs(3.0) > z  # fixture sanity
        np.testing.assert_allclose(out.coverage, [0.5, 1.0])
        assert out.level == 0.95
        assert out.n_used == 2 and out.n_excluded == 0

    def test_excludes_non_converged(self):
        theta = [[0.0, 0.0], [50.0, 50.0]]
        se = [[1.0, 1.0], [1.0, 1.0]]
        out = coverage(_records(theta, se, converged=[1.0, 0.0]), theta0=[0.0, 0.0])
        np.testing.assert_allclose(out.coverage, [1.0, 1.0])
        assert out.n_used == 1 and out.n_excluded == 1

    def test_bad_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            coverage(_records([[0.0, 0.0]], [[1.0, 1.0]]), [0.0, 0.0], level=1.5)


class TestSizePower:
    def test_exact_rejection_rates(self):
        rec = _records(
            [[0.0, 0.0]] * 4,
            [[1.0, 1.0]] * 4,
            J_pvalue=[0.005, 0.02, 0.04, 0.5],
            J_pvalue_adjusted=[0.02, 0.06, 0.2, 0.6],
        )
        out = size_power(rec, alpha=(0.01, 0.05, 0.10))
        np.testing.assert_allclose(out.reject_nominal, [0.25, 0.75, 0.75])
        np.testing.assert_allclose(out.reject_adjusted, [0.0, 0.25, 0.5])
        assert out.alphas == (0.01, 0.05, 0.10)
        assert out.n_used == 4 and out.n_excluded == 0

    def test_excludes_non_converged(self):
        rec = _records(
            [[0.0, 0.0]] * 2,
            [[1.0, 1.0]] * 2,
            J_pvalue=[0.001, 0.9],
            converged=[0.0, 1.0],
        )
        out = size_power(rec, alpha=(0.05,))
        np.testing.assert_allclose(out.reject_nominal, [0.0])
        assert out.n_used == 1 and out.n_excluded == 1


class TestTauBinding:
    def test_frequency_and_quantiles(self):
        rec = _records(
            [[0.0, 0.0]] * 4,
            [[1.0, 1.0]] * 4,
            tau=[0.0, 1.0, 2.0, 3.0],
            binding=[0.0, 1.0, 1.0, 0.0],
        )
        out = tau_binding(rec, q=(0.0, 0.5, 1.0))
        assert out.binding_frequency == 0.5
        np.testing.assert_allclose(out.tau_quantiles, [0.0, 1.5, 3.0])
        assert out.quantile_levels == (0.0, 0.5, 1.0)
        assert out.n_used == 4 and out.n_excluded == 0

    def test_all_non_converged(self):
        rec = _records([[0.0, 0.0]], [[1.0, 1.0]], converged=[0.0])
        out = tau_binding(rec)
        assert np.isnan(out.binding_frequency)
        assert np.isnan(out.tau_quantiles).all()
        assert out.n_used == 0 and out.n_excluded == 1


class TestJCalibration:
    def test_exact_ecdf_deviation(self):
        # 10 p-values at the decile-cell midpoints 0.05, 0.15, ..., 0.95:
        # the ecdf at decile k/10 counts exactly k of them, so the
        # deviation is identically zero (perfect calibration).
        p = [(2 * k + 1) / 20.0 for k in range(10)]  # 0.05, 0.15, ..., 0.95
        rec = _records([[0.0, 0.0]] * 10, [[1.0, 1.0]] * 10, J_pvalue=p)
        out = j_calibration(rec)
        np.testing.assert_allclose(out.deciles, np.arange(1, 10) / 10.0)
        np.testing.assert_allclose(out.ecdf, out.deciles)
        np.testing.assert_allclose(out.deviation, np.zeros(9), atol=1e-15)
        assert out.max_abs_deviation == pytest.approx(0.0, abs=1e-15)
        assert out.J_dof == 1
        assert out.n_used == 10 and out.n_excluded == 0

    def test_skewed_pvalues_show_deviation(self):
        rec = _records(
            [[0.0, 0.0]] * 4, [[1.0, 1.0]] * 4, J_pvalue=[0.01, 0.02, 0.03, 0.04]
        )
        out = j_calibration(rec)
        np.testing.assert_allclose(out.ecdf, np.ones(9))
        assert out.max_abs_deviation == pytest.approx(0.9)


class TestAcceptsWrapperOrBareRecord:
    def test_summarizers_accept_bare_fitrecord(self):
        rec = _records([[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]])
        # No MCRecords wrapper anywhere above: every call in this module
        # already exercises the bare-FitRecord path. Spot-check coverage.
        out = coverage(rec, theta0=[1.0, 2.0])
        np.testing.assert_allclose(out.coverage, [1.0, 1.0])
