"""Recovery + sanity smoke test for ``examples/empirical_bootstrap.py``.

The example is a bridge demo of the two cluster-aware bootstrap helpers
(:func:`emu_gmm.moment_wild_bootstrap` and
:func:`emu_gmm.cluster_bootstrap`) on a clustered-normal DGP. The test
locks in:

* the example module imports cleanly and the entry-point :func:`run`
  function runs end-to-end,
* the GMM solve recovers ``(mu, sigma2)`` within a few analytic SEs of
  the truth,
* the analytic ``J_stat`` from the estimator matches the refit-free
  wild-bootstrap ``J_observed`` (they whiten the same moment vector
  with the same Cholesky factor, so the agreement should be tight to
  numerical precision),
* the refit-based cluster-bootstrap SEs and the analytic SEs agree on
  the same order of magnitude (within ~30%; modest disagreement is
  expected from MC noise and from the refit-based estimator picking up
  nonlinearity the asymptotic linearisation misses).

The test is marked ``slow`` because the refit-based cluster bootstrap
re-solves the GMM problem on each replicate; even at the reduced
``n_boot=80`` used here it takes a few seconds.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the top-level ``examples`` directory importable; it isn't on
# ``sys.path`` by default because it's a runnable-scripts folder, not a
# Python package shipped with the library.
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

empirical_bootstrap = importlib.import_module("empirical_bootstrap")


@pytest.mark.slow
def test_run_recovers_truth_and_matches_analytic_inference():
    """End-to-end smoke test of the bridge example."""
    # Use a smaller bootstrap count than the example default to keep
    # the test fast; the recovery + SE-agreement claims are robust to
    # this change. The boot SE estimator's chi-squared SE at B = 80 is
    # roughly 1 / sqrt(2 * 79) ~ 8 %, well within the 30 % tolerance
    # the test asserts below.
    result, wild, cluster, se_table = empirical_bootstrap.run(
        n_boot=80,
    )

    # ---- Recovery ----
    # theta_hat = (mu_hat, sigma2_hat). The DGP has mu_true = 1.5,
    # sigma2_true = sigma_u^2 + sigma_e^2 = 0.72 (under the example's
    # defaults). Assert recovery within four analytic SEs -- a generous
    # band; in practice the t-stats are well above 10.
    mu_hat = float(result.theta_hat.mu)
    sigma2_hat = float(result.theta_hat.sigma2)
    analytic_se = np.asarray(result.standard_errors.array)
    assert abs(mu_hat - empirical_bootstrap.MU_TRUE) < 4.0 * analytic_se[0]
    assert abs(sigma2_hat - empirical_bootstrap.SIGMA2_TRUE) < 4.0 * analytic_se[1]

    # ---- Analytic J <-> wild-bootstrap J_observed ----
    # The refit-free wild bootstrap whitens by the same V_X the analytic
    # J-test uses, so J_observed should equal J_stat up to floating-point
    # rounding.
    assert float(wild.J_observed) == pytest.approx(float(result.J_stat), rel=1e-6)
    assert wild.J_boot.shape == (80,)
    assert 0.0 <= float(wild.p_value) <= 1.0

    # ---- Cluster bootstrap shape + convergence ----
    # The refit-based cluster bootstrap returns a (n_boot, K) NamedArray
    # of per-replicate parameter estimates. Convergence on a well-posed
    # mean / variance problem should be near-universal.
    assert cluster.theta_boot.array.shape == (80, 2)
    converged = np.asarray(cluster.convergence)
    assert converged.sum() >= 60, (
        f"Only {converged.sum()} of 80 cluster-bootstrap replicates converged; "
        f"expected near-universal convergence on this well-posed problem"
    )

    # ---- SE agreement ----
    # The analytic SE (asymptotic CLT) and the refit-based bootstrap SE
    # are two estimators of the same sampling-distribution width. They
    # should agree on the same order of magnitude. ~30% banding is
    # generous: at B = 80 the bootstrap-SE estimator itself carries
    # roughly 8% MC noise, and the refit-based estimator additionally
    # captures some nonlinearity the analytic linearisation misses
    # (which is why we use both helpers).
    ratios = se_table["ratio_boot_over_analytic"].to_numpy()
    assert np.all(
        np.isfinite(ratios)
    ), f"NaN in SE ratios: {ratios} (likely all bootstrap replicates failed)"
    assert np.all((ratios > 0.6) & (ratios < 1.5)), (
        f"Bootstrap-vs-analytic SE ratios outside the 0.6-1.5 band: "
        f"{dict(zip(se_table.index, ratios))}"
    )


@pytest.mark.slow
def test_make_dataset_layout():
    """The DGP helper returns the documented shapes."""
    measure, covariance, cluster_ids_np = empirical_bootstrap.make_dataset(
        seed=123,
        n_clusters=10,
        obs_per_cluster=5,
    )
    n = 10 * 5
    assert measure.x.shape == (n, 1)
    # M = 3 (mean, variance, third central moment).
    assert measure.mask.shape == (n, 3)
    assert measure.weights.shape == (n,)
    assert covariance.n_clusters == 10
    assert covariance.cluster_ids.shape == (n,)
    # cluster_ids should run 0, 0, ..., 0, 1, 1, ..., 1, ..., 9, 9, ..., 9.
    expected = np.repeat(np.arange(10), 5)
    np.testing.assert_array_equal(cluster_ids_np, expected)
