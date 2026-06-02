r"""Tests for the adaptive (precision-targeted) bootstrap driver (#91).

Two layers:

1. **MCSE correctness** -- the Monte Carlo standard-error estimators are the
   statistical crux. Each is validated against a ground truth: the Maritz-
   Jarrett quantile SE and the analytic mean/SE/proportion SEs are checked
   (on average) against a direct resampling Monte Carlo / known asymptotics.
2. **Driver behaviour** -- stopping when the half-width meets the target, a
   loud ``converged=False`` at ``b_max``, NaN exclusion with denominator
   accounting, determinism, input validation, and one integration test that
   wraps the real ``cluster_bootstrap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AdaptiveBootstrapResult,
    BootstrapMean,
    BootstrapPValue,
    BootstrapQuantile,
    BootstrapSE,
    adaptive_bootstrap,
)
from emu_gmm.covariance import ClusteredCovariance
from emu_gmm.inference import cluster_bootstrap
from emu_gmm.inference.adaptive import maritz_jarrett_quantile_se
from emu_gmm.measures import EmpiricalMeasure

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. MCSE-estimator correctness.
# ---------------------------------------------------------------------------
class TestMCSEEstimators:
    def test_maritz_jarrett_median_matches_monte_carlo(self):
        """Mean MJ median-SE over samples == MC SD of the sample median."""
        rng = np.random.default_rng(0)
        n = 2000
        meds = [np.median(rng.standard_normal(n)) for _ in range(4000)]
        mc_truth = np.std(meds, ddof=1)
        mj = [
            maritz_jarrett_quantile_se(rng.standard_normal(n), 0.5) for _ in range(1500)
        ]
        assert np.mean(mj) == pytest.approx(mc_truth, rel=0.05)
        # And the normal-median asymptotic 1.2533/sqrt(n).
        assert mc_truth == pytest.approx(1.2533 / np.sqrt(n), rel=0.05)

    def test_maritz_jarrett_tail_quantile_matches_monte_carlo(self):
        rng = np.random.default_rng(1)
        n = 3000
        q = 0.9
        qs = [np.quantile(rng.standard_normal(n), q) for _ in range(4000)]
        mc_truth = np.std(qs, ddof=1)
        mj = [
            maritz_jarrett_quantile_se(rng.standard_normal(n), q) for _ in range(1500)
        ]
        assert np.mean(mj) == pytest.approx(mc_truth, rel=0.08)

    def test_maritz_jarrett_degenerate_small_n(self):
        assert maritz_jarrett_quantile_se(np.array([1.0]), 0.5) == 0.0
        assert maritz_jarrett_quantile_se(np.array([]), 0.5) == 0.0

    def test_se_target_mcse_formula(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(1000)
        value, mcse = BootstrapSE().evaluate(x)
        assert value == pytest.approx(float(np.std(x, ddof=1)))
        assert mcse == pytest.approx(value / np.sqrt(2 * (1000 - 1)), rel=1e-12)

    def test_mean_target_mcse_formula(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(800)
        value, mcse = BootstrapMean().evaluate(x)
        assert value == pytest.approx(float(np.mean(x)))
        assert mcse == pytest.approx(float(np.std(x, ddof=1)) / np.sqrt(800), rel=1e-12)

    def test_pvalue_target_value_and_binomial_mcse(self):
        # Replicates ~ N(0,1); observed at 1.645 => upper-tail p ~ 0.05.
        rng = np.random.default_rng(4)
        x = rng.standard_normal(20000)
        value, mcse = BootstrapPValue(1.645, "greater").evaluate(x)
        assert value == pytest.approx(0.05, abs=0.01)
        p_tilde = min(max(value, 1 / 20001), 1 - 1 / 20001)
        assert mcse == pytest.approx(np.sqrt(p_tilde * (1 - p_tilde) / 20000), rel=1e-9)

    def test_pvalue_empty_tail_has_positive_mcse(self):
        # No replicate exceeds a far-right observed value: raw count 0, but the
        # +1 correction keeps p and MCSE strictly positive (no false converge).
        x = np.zeros(500)  # all 0; observed 10 -> nothing >= 10
        value, mcse = BootstrapPValue(10.0, "greater").evaluate(x)
        assert value == pytest.approx(1 / 501)
        assert mcse > 0.0


# ---------------------------------------------------------------------------
# 2. Driver behaviour.
# ---------------------------------------------------------------------------
def _normal_draw(key, size):
    return jax.random.normal(key, (size,))


class TestDriverStopping:
    def test_converges_and_meets_halfwidth(self):
        r = adaptive_bootstrap(
            _normal_draw,
            BootstrapMean(),
            key=jax.random.PRNGKey(0),
            batch_size=500,
            b_min=500,
            b_max=200_000,
            atol=0.02,
        )
        assert r.converged
        # z*mcse == half_width, and half_width within the absolute target.
        z = 1.959963984540054
        assert r.half_width == pytest.approx(z * r.mcse, rel=1e-9)
        assert r.half_width <= 0.02
        # Mean of N(0,1) replicates ~ 0.
        assert abs(r.value) < 0.05
        assert r.n_valid == r.n_boot  # no NaNs here
        assert r.target == "mean"
        assert isinstance(r, AdaptiveBootstrapResult)

    def test_relative_tolerance_on_se(self):
        r = adaptive_bootstrap(
            _normal_draw,
            BootstrapSE(),
            key=jax.random.PRNGKey(1),
            batch_size=500,
            b_min=500,
            b_max=500_000,
            rtol=0.02,
        )
        assert r.converged
        assert r.value == pytest.approx(1.0, abs=0.05)  # SD of N(0,1)
        assert r.half_width <= 0.02 * abs(r.value)

    def test_loud_nonconvergence_at_bmax(self):
        # Impossible tolerance: must hit b_max and report converged=False,
        # but still return a finite value/mcse (the publication-relevant signal).
        r = adaptive_bootstrap(
            _normal_draw,
            BootstrapMean(),
            key=jax.random.PRNGKey(2),
            batch_size=400,
            b_min=400,
            b_max=2000,
            atol=1e-9,
        )
        assert not r.converged
        assert r.n_boot == 2000  # capped exactly
        assert np.isfinite(r.value) and np.isfinite(r.mcse)

    def test_bmax_trim_never_overshoots(self):
        # batch_size does not divide b_max: the final batch is trimmed.
        r = adaptive_bootstrap(
            _normal_draw,
            BootstrapMean(),
            key=jax.random.PRNGKey(3),
            batch_size=300,
            b_min=300,
            b_max=1000,
            atol=1e-9,  # force the cap
        )
        assert r.n_boot == 1000
        assert not r.converged

    def test_determinism_same_key(self):
        kw = dict(
            key=jax.random.PRNGKey(7),
            batch_size=500,
            b_min=500,
            b_max=100_000,
            atol=0.02,
        )
        r1 = adaptive_bootstrap(_normal_draw, BootstrapMean(), **kw)
        r2 = adaptive_bootstrap(_normal_draw, BootstrapMean(), **kw)
        assert r1.n_boot == r2.n_boot
        assert r1.value == r2.value
        np.testing.assert_array_equal(r1.replicates, r2.replicates)


class TestDriverNaNHandling:
    def test_nan_replicates_excluded_and_counted(self):
        # ~20% NaN; functional uses finite only, NaNs counted in n_invalid.
        def draw(key, size):
            x = jax.random.normal(key, (size,))
            u = jax.random.uniform(jax.random.fold_in(key, 1), (size,))
            return jnp.where(u < 0.2, jnp.nan, x)

        r = adaptive_bootstrap(
            draw,
            BootstrapMean(),
            key=jax.random.PRNGKey(5),
            batch_size=1000,
            b_min=1000,
            b_max=100_000,
            atol=0.03,
        )
        assert r.n_valid + r.n_invalid == r.n_boot
        assert r.n_invalid > 0
        assert 0.15 < r.n_invalid / r.n_boot < 0.25
        assert np.isfinite(r.value)
        # b_min is on VALID replicates: enough finite draws were accumulated.
        assert r.n_valid >= 1000


class TestDriverValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_size": 1, "atol": 0.1},
            {"b_min": 1, "atol": 0.1},
            {"b_max": 100, "b_min": 200, "atol": 0.1},
            {"confidence": 1.0, "atol": 0.1},
            {"confidence": 0.0, "atol": 0.1},
            {},  # atol == rtol == 0
            {"atol": -1.0},
        ],
    )
    def test_bad_inputs_raise(self, kwargs):
        base = dict(key=jax.random.PRNGKey(0), b_max=1000)
        base.update(kwargs)
        with pytest.raises(ValueError):
            adaptive_bootstrap(_normal_draw, BootstrapMean(), **base)


# ---------------------------------------------------------------------------
# 3. Integration: wrap the real cluster_bootstrap as the draw_batch.
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class _LocParams:
    mu: jnp.ndarray


def _loc_model(x, theta):
    # Single moment: x - mu. Exactly identified (M=K=1).
    return jnp.atleast_1d(x[0] - theta.mu)


class TestClusterBootstrapIntegration:
    def test_wraps_cluster_bootstrap_J_boot(self):
        rng = np.random.default_rng(0)
        n, g = 200, 20
        x = jnp.asarray(rng.normal(size=(n, 1)))
        measure = EmpiricalMeasure(x=x, mask=jnp.ones((n, 1)), weights=jnp.ones(n))
        cluster_ids = jnp.asarray(np.repeat(np.arange(g), n // g).astype(float))
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=g)
        theta0 = _LocParams(mu=jnp.asarray(0.0))

        def draw(key, size):
            res = cluster_bootstrap(
                _loc_model, theta0, measure, cov, n_boot=int(size), key=key
            )
            return np.asarray(res.theta_boot.array[:, 0])  # mu replicates

        # Tiny b_max + generous tol so it converges on the first batch (8 refits).
        r = adaptive_bootstrap(
            draw,
            BootstrapSE(),
            key=jax.random.PRNGKey(0),
            batch_size=8,
            b_min=8,
            b_max=24,
            atol=1e3,  # huge -> converge immediately once b_min valid drawn
        )
        assert r.converged
        assert r.n_boot == 8
        assert r.n_valid >= 1
        assert np.isfinite(r.value) and r.value > 0.0  # a positive SE of mu_boot


# ---------------------------------------------------------------------------
# 4. Quantile target end-to-end (CI-endpoint use case).
# ---------------------------------------------------------------------------
class TestQuantileTarget:
    def test_quantile_target_evaluate_matches_numpy(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4000)
        value, mcse = BootstrapQuantile(0.975).evaluate(x)
        assert value == pytest.approx(float(np.quantile(x, 0.975)))
        assert mcse == pytest.approx(maritz_jarrett_quantile_se(x, 0.975), rel=1e-12)

    def test_quantile_target_rejects_bad_q(self):
        with pytest.raises(ValueError):
            BootstrapQuantile(0.0)
        with pytest.raises(ValueError):
            BootstrapQuantile(1.0)

    def test_driver_converges_on_ci_endpoint(self):
        # 97.5% percentile-CI endpoint of N(0,1) replicates -> ~1.96.
        r = adaptive_bootstrap(
            _normal_draw,
            BootstrapQuantile(0.975),
            key=jax.random.PRNGKey(9),
            batch_size=1000,
            b_min=1000,
            b_max=500_000,
            atol=0.03,
        )
        assert r.converged
        assert r.value == pytest.approx(1.95996, abs=0.05)
        assert r.half_width <= 0.03
        assert r.target == "quantile[0.975]"
