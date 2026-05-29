"""Tests for emu_gmm.inference.wild_bootstrap.

Three contracts:

(a) Sign distribution check. Rademacher draws are equiprobable on
    ``{-1, +1}``; Mammen draws hit the two analytic atoms with the
    correct probabilities and satisfy ``E[eta] = 0``, ``E[eta^2] = 1``,
    ``E[eta^3] = 1``.

(b) Under H0 (well-specified moment model + theta = truth), the
    bootstrap p-value distribution is approximately uniform on [0, 1].
    We assess this with a single-replication chi-square / Kolmogorov
    sanity check at low decimal precision, not a tight calibration
    test --- the goal here is "this is in the right neighbourhood",
    not a finite-sample edgeworth-style guarantee.

(c) Cluster structure is honoured. Two observations sharing a cluster
    ID receive the same sign in every replicate; the bootstrap that
    ignores cluster IDs (treats every observation as its own cluster)
    produces a /different/ distribution of bootstrap moments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.inference.wild_bootstrap import (
    WildBootstrapResult,
    _bootstrap_moment,
    _draw_mammen,
    _draw_rademacher,
    _per_obs_signs,
    moment_wild_bootstrap,
)
from emu_gmm.measures.empirical import EmpiricalMeasure


@jdc.pytree_dataclass
class _P:
    """Trivial parameter holder for tests that need a theta object."""

    a: float


def _identity_psi(x, theta):
    """psi(x, theta) = x. Independent of theta."""
    return x


def _residual_psi(x, theta):
    """Residual ``x - theta.a`` for one-moment scalar mean tests."""
    return x - theta.a


# ---------------------------------------------------------------------------
# (a) Sign distribution
# ---------------------------------------------------------------------------


class TestSignDistribution:
    """Sanity checks on the sign-draw helpers."""

    def test_rademacher_values_in_pm_one(self):
        key = jax.random.PRNGKey(0)
        eta = _draw_rademacher(key, 1000)
        unique = jnp.unique(eta)
        assert set(np.asarray(unique).tolist()) == {-1.0, 1.0}

    def test_rademacher_mean_near_zero(self):
        """E[eta] = 0 under Rademacher."""
        key = jax.random.PRNGKey(1)
        eta = _draw_rademacher(key, 20000)
        mean = float(jnp.mean(eta))
        # 3-sigma band on Bernoulli(0.5) of 20000 draws: 3 / sqrt(20000) approx 0.021.
        assert abs(mean) < 0.05

    def test_rademacher_second_moment_one(self):
        """E[eta^2] = 1 exactly under Rademacher."""
        key = jax.random.PRNGKey(2)
        eta = _draw_rademacher(key, 1000)
        assert jnp.allclose(jnp.mean(eta * eta), 1.0)

    def test_mammen_two_atoms_only(self):
        key = jax.random.PRNGKey(3)
        eta = _draw_mammen(key, 5000)
        unique = jnp.unique(eta)
        assert unique.shape[0] == 2
        sqrt5 = float(jnp.sqrt(jnp.asarray(5.0)))
        atoms = np.sort(np.asarray(unique).tolist())
        expected = np.sort([-(sqrt5 - 1.0) / 2.0, (sqrt5 + 1.0) / 2.0])
        assert np.allclose(atoms, expected, atol=1e-7)

    def test_mammen_moments(self):
        """Mammen: E[eta]=0, E[eta^2]=1, E[eta^3]=1 (asymptotic)."""
        key = jax.random.PRNGKey(4)
        eta = _draw_mammen(key, 50000)
        m1 = float(jnp.mean(eta))
        m2 = float(jnp.mean(eta * eta))
        m3 = float(jnp.mean(eta * eta * eta))
        # Loose 3-sigma bounds; the third moment converges more slowly so
        # we allow a wider band.
        assert abs(m1) < 0.05
        assert abs(m2 - 1.0) < 0.05
        assert abs(m3 - 1.0) < 0.10

    def test_different_keys_produce_different_draws(self):
        """Two PRNG keys yield different sign vectors."""
        eta_a = _draw_rademacher(jax.random.PRNGKey(5), 100)
        eta_b = _draw_rademacher(jax.random.PRNGKey(6), 100)
        assert not jnp.allclose(eta_a, eta_b)


# ---------------------------------------------------------------------------
# (b) Bootstrap p-value distribution under H0
# ---------------------------------------------------------------------------


def _build_h0_setup(
    *, seed: int, N: int, n_clusters: int
) -> tuple[EmpiricalMeasure, ClusteredCovariance, _P]:
    """Construct an H0 setup: data drawn so the moment restriction holds
    exactly at ``theta = 0`` in population.

    The residual ``x - theta.a`` has zero population mean when ``a = 0``
    and ``x`` is iid mean-zero. ``n_clusters`` clusters are formed by
    contiguous chunks of observations of size ``N // n_clusters``.
    """
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (N, 1))
    measure = EmpiricalMeasure(
        x=x,
        mask=jnp.ones((N, 1)),
        weights=jnp.ones(N),
    )
    cluster_size = N // n_clusters
    cluster_ids = jnp.repeat(jnp.arange(n_clusters, dtype=jnp.float64), cluster_size)
    # Pad the tail if N isn't a multiple of n_clusters: last cluster
    # absorbs the remainder. Reshape via concatenation to keep cluster
    # IDs simple.
    extra = N - cluster_size * n_clusters
    if extra > 0:
        tail = jnp.full((extra,), float(n_clusters - 1))
        cluster_ids = jnp.concatenate([cluster_ids, tail])
    covariance = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=n_clusters)
    return measure, covariance, _P(a=0.0)


class TestUnderH0PValueUniformity:
    """Under H0 the bootstrap p-value should hover near uniform."""

    def test_p_value_near_chi_square_calibration(self):
        """One H0 draw yields a p-value in [0, 1] and the J-distribution
        looks chi-square-like in shape.
        """
        measure, covariance, theta_0 = _build_h0_setup(seed=42, N=200, n_clusters=20)
        key = jax.random.PRNGKey(7)
        result = moment_wild_bootstrap(
            _residual_psi,
            theta_0,
            measure,
            covariance,
            n_boot=500,
            key=key,
            sign="rademacher",
        )
        # p_value must be a valid probability.
        assert 0.0 <= result.p_value <= 1.0
        # Bootstrap J is nonnegative (it's a sum of squares).
        assert bool(jnp.all(result.J_boot >= 0.0))
        # Mean of J_boot should be in the right neighbourhood for a
        # one-moment problem: under H0 the J statistic is asymptotically
        # chi^2_1 with mean 1.
        boot_mean = float(jnp.mean(result.J_boot))
        # Wide acceptance band: cluster-robust variance with 20 clusters
        # only matches the chi^2 mean to within 25 percent or so.
        assert 0.5 < boot_mean < 2.0

    def test_p_value_distribution_across_seeds(self):
        """Across many H0 draws, the empirical p-value distribution is
        approximately uniform.

        Marked as a sanity rather than a tight calibration test: with
        50 seeds we want the empirical CDF at 0.5 to land somewhere
        between 0.25 and 0.75. Tighter calibration is a property of the
        bootstrap that's expensive to test in unit-test budget and is
        documented in the asymptotic theory.
        """
        p_values = []
        for seed in range(50):
            measure, covariance, theta_0 = _build_h0_setup(
                seed=100 + seed, N=200, n_clusters=20
            )
            key = jax.random.PRNGKey(7000 + seed)
            result = moment_wild_bootstrap(
                _residual_psi,
                theta_0,
                measure,
                covariance,
                n_boot=200,
                key=key,
                sign="rademacher",
            )
            p_values.append(result.p_value)
        p_arr = jnp.asarray(p_values)
        # Empirical CDF at 0.5 should be near 0.5 under uniformity.
        cdf_at_half = float(jnp.mean(p_arr <= 0.5))
        assert 0.25 <= cdf_at_half <= 0.75
        # Mean of a uniform is 0.5; with 50 samples the 3-sigma window
        # is roughly 0.5 +/- 0.42 --- very loose.
        assert 0.2 < float(jnp.mean(p_arr)) < 0.8


# ---------------------------------------------------------------------------
# (c) Cluster structure is honoured
# ---------------------------------------------------------------------------


class TestClusterStructure:
    """Observations sharing a cluster ID receive the same sign."""

    def test_per_obs_signs_constant_within_cluster(self):
        eta_c = jnp.array([1.0, -1.0, 1.0])
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
        eta_i = _per_obs_signs(eta_c, cluster_ids)
        # The first two observations are in cluster 0 -> same sign.
        assert float(eta_i[0]) == float(eta_i[1])
        # The last three observations are in cluster 2 -> same sign.
        assert float(eta_i[3]) == float(eta_i[4]) == float(eta_i[5])
        # Cluster 0 and cluster 1 carry different signs (as drawn).
        assert float(eta_i[0]) != float(eta_i[2])

    def test_cluster_signs_propagate_to_moment(self):
        """The bootstrap moment uses cluster-level signs.

        Two observations in the same cluster, equal contributions: when
        the cluster sign is +1 the bootstrap moment equals the original;
        when the sign is -1 it equals minus the original.
        """
        # Two obs, one moment, identical contributions, same cluster.
        contributions = jnp.array([[0.5], [0.5]])
        weight_mask = jnp.array([[1.0], [1.0]])

        # eta_c = [+1] -> eta_i = [+1, +1] -> m^* = +1.0 / 2.0
        eta_pos = jnp.array([1.0, 1.0])
        m_pos = _bootstrap_moment(contributions, weight_mask, eta_pos)
        assert jnp.allclose(m_pos, jnp.array([0.5]))

        # eta_c = [-1] -> eta_i = [-1, -1] -> m^* = -1.0 / 2.0
        eta_neg = jnp.array([-1.0, -1.0])
        m_neg = _bootstrap_moment(contributions, weight_mask, eta_neg)
        assert jnp.allclose(m_neg, jnp.array([-0.5]))

    def test_singleton_clusters_differ_from_grouped(self):
        """Same data, two cluster structures: the bootstrap J
        distribution differs.

        With all observations in one cluster, the sign is shared across
        the sample and the bootstrap J equals the analytic J exactly
        (modulo sign squaring). With per-observation clusters the J is
        averaged over independent flips and concentrates near zero.
        """
        N = 60
        key = jax.random.PRNGKey(11)
        x = jax.random.normal(key, (N, 1))
        measure = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 1)),
            weights=jnp.ones(N),
        )

        # All observations in one cluster.
        cov_one = ClusteredCovariance(cluster_ids=jnp.zeros(N), n_clusters=1)
        # Each observation its own cluster.
        cov_singleton = ClusteredCovariance(
            cluster_ids=jnp.arange(N, dtype=jnp.float64),
            n_clusters=N,
        )

        result_one = moment_wild_bootstrap(
            _identity_psi,
            _P(a=0.0),
            measure,
            cov_one,
            n_boot=500,
            key=jax.random.PRNGKey(12),
            sign="rademacher",
            V=jnp.eye(1),  # supply explicitly so the singleton case is well-conditioned
        )
        result_singleton = moment_wild_bootstrap(
            _identity_psi,
            _P(a=0.0),
            measure,
            cov_singleton,
            n_boot=500,
            key=jax.random.PRNGKey(13),
            sign="rademacher",
            V=jnp.eye(1),
        )

        # With one cluster, every replicate yields ||L^{-1} (+/- m_hat)||^2,
        # which equals ||L^{-1} m_hat||^2 (the sign cancels in the
        # squared norm). So J_boot is constant.
        std_one = float(jnp.std(result_one.J_boot))
        assert std_one < 1e-10

        # With N independent clusters the J_boot is highly variable.
        std_singleton = float(jnp.std(result_singleton.J_boot))
        assert std_singleton > 1e-3

    def test_singleton_vs_grouped_p_values_differ(self):
        """The p-value depends on the cluster structure."""
        N = 40
        # Construct data with strong within-cluster correlation: two
        # clusters of 20 observations each, intra-cluster mean shifted
        # away from zero. Estimate theta = 0 (mis-specified). The
        # analytic J differs from the bootstrap calibration, but the
        # /bootstrap/ p-value under the cluster-respecting design
        # should be higher (correct cluster-robust calibration) than
        # under the wrong, singleton design.
        key = jax.random.PRNGKey(14)
        eps = 0.3 * jax.random.normal(key, (N, 1))
        cluster_means = jnp.repeat(jnp.array([0.5, -0.5]), N // 2)
        x = cluster_means[:, None] + eps
        measure = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 1)),
            weights=jnp.ones(N),
        )
        cluster_ids = jnp.repeat(jnp.arange(2, dtype=jnp.float64), N // 2)
        cov_two = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2)
        cov_singleton = ClusteredCovariance(
            cluster_ids=jnp.arange(N, dtype=jnp.float64),
            n_clusters=N,
        )

        # Use the analytic V from each variance specification so the
        # bootstrap J distribution and the observed J are both calibrated
        # against the matching covariance.
        result_two = moment_wild_bootstrap(
            _identity_psi,
            _P(a=0.0),
            measure,
            cov_two,
            n_boot=1000,
            key=jax.random.PRNGKey(15),
            sign="rademacher",
        )
        result_singleton = moment_wild_bootstrap(
            _identity_psi,
            _P(a=0.0),
            measure,
            cov_singleton,
            n_boot=1000,
            key=jax.random.PRNGKey(16),
            sign="rademacher",
        )

        # The two designs produce different J distributions and so
        # different observed-J relative to their bootstrap quantiles;
        # the p-values should not be identical.
        assert result_two.p_value != result_singleton.p_value


# ---------------------------------------------------------------------------
# Result type and argument validation
# ---------------------------------------------------------------------------


class TestResultType:
    def test_returns_wild_bootstrap_result(self):
        measure, covariance, theta_0 = _build_h0_setup(seed=20, N=40, n_clusters=4)
        result = moment_wild_bootstrap(
            _residual_psi,
            theta_0,
            measure,
            covariance,
            n_boot=20,
            key=jax.random.PRNGKey(17),
            sign="rademacher",
        )
        assert isinstance(result, WildBootstrapResult)
        assert result.J_boot.shape == (20,)
        assert result.theta_boot is None  # v1 scope: refit-free
        assert result.sign == "rademacher"
        assert result.n_boot == 20

    def test_mammen_sign_path(self):
        measure, covariance, theta_0 = _build_h0_setup(seed=21, N=40, n_clusters=4)
        result = moment_wild_bootstrap(
            _residual_psi,
            theta_0,
            measure,
            covariance,
            n_boot=20,
            key=jax.random.PRNGKey(18),
            sign="mammen",
        )
        assert result.sign == "mammen"
        assert result.J_boot.shape == (20,)


class TestArgumentValidation:
    def test_invalid_sign_raises(self):
        measure, covariance, theta_0 = _build_h0_setup(seed=22, N=20, n_clusters=2)
        with pytest.raises(ValueError, match="sign must be"):
            moment_wild_bootstrap(
                _residual_psi,
                theta_0,
                measure,
                covariance,
                n_boot=5,
                key=jax.random.PRNGKey(19),
                sign="gaussian",
            )

    def test_non_positive_n_boot_raises(self):
        measure, covariance, theta_0 = _build_h0_setup(seed=23, N=20, n_clusters=2)
        with pytest.raises(ValueError, match="n_boot must be positive"):
            moment_wild_bootstrap(
                _residual_psi,
                theta_0,
                measure,
                covariance,
                n_boot=0,
                key=jax.random.PRNGKey(20),
            )

    def test_user_supplied_V_overrides_recompute(self):
        """If a V is passed in, it's used verbatim (rather than the V
        the covariance object would assemble)."""
        measure, covariance, theta_0 = _build_h0_setup(seed=24, N=40, n_clusters=4)
        # A huge V suppresses the whitened residual, pushing J -> 0.
        big_V = jnp.eye(1) * 1e6
        result = moment_wild_bootstrap(
            _residual_psi,
            theta_0,
            measure,
            covariance,
            n_boot=20,
            key=jax.random.PRNGKey(21),
            V=big_V,
        )
        # J = m' V^{-1} m -> 0 as V -> infty.
        assert float(jnp.max(result.J_boot)) < 1e-5
        assert result.J_observed < 1e-5


class TestReExport:
    """The public surface is re-exported at the package root."""

    def test_package_level_exports(self):
        import emu_gmm

        assert hasattr(emu_gmm, "moment_wild_bootstrap")
        assert hasattr(emu_gmm, "WildBootstrapResult")
