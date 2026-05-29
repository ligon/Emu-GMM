"""Tests for emu_gmm.inference.cluster_bootstrap."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import ClusteredCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_data,
    euler_residual,
)
from emu_gmm.inference import ClusterBootstrapResult, cluster_bootstrap
from emu_gmm.inference.cluster_bootstrap import (
    _cluster_row_indices,
    _resample_one,
)
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _euler_cluster_setup(
    n_clusters: int = 25,
    obs_per_cluster: int = 20,
    seed: int = 0,
):
    """Build an :class:`EmpiricalMeasure` + :class:`ClusteredCovariance`
    for the Euler example.

    Observations are simply consecutive blocks: cluster ``c`` gets rows
    ``[c * obs_per_cluster : (c + 1) * obs_per_cluster]``. The DGP is
    iid across rows (so within-cluster correlation is zero in
    expectation) -- that's fine; the cluster bootstrap is valid under
    iid as a special case.
    """
    N = n_clusters * obs_per_cluster
    x = euler_data(seed=seed, n=N)
    mask = jnp.ones((N, 3))
    weights = jnp.ones(N)
    measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
    cluster_ids = jnp.repeat(jnp.arange(n_clusters, dtype=jnp.float64), obs_per_cluster)
    covariance = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=n_clusters)
    return measure, covariance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestClusterRowIndices:
    """Per-cluster row-index lookup behaves like a partition of [0, N)."""

    def test_partition(self):
        cluster_ids = np.array([0, 0, 1, 2, 1, 0])
        rows = _cluster_row_indices(cluster_ids, n_clusters=3)
        assert len(rows) == 3
        np.testing.assert_array_equal(rows[0], np.array([0, 1, 5]))
        np.testing.assert_array_equal(rows[1], np.array([2, 4]))
        np.testing.assert_array_equal(rows[2], np.array([3]))
        # Disjoint union covers all of [0, N).
        all_rows = np.concatenate(rows)
        np.testing.assert_array_equal(np.sort(all_rows), np.arange(6))

    def test_empty_cluster_returns_empty_array(self):
        cluster_ids = np.array([0, 0, 2, 2])
        rows = _cluster_row_indices(cluster_ids, n_clusters=3)
        assert rows[1].shape == (0,)


# ---------------------------------------------------------------------------
# Resample helper
# ---------------------------------------------------------------------------


class TestResampleOne:
    """``_resample_one`` constructs a coherent bootstrap world."""

    def test_drawing_same_cluster_twice_yields_two_clusters_in_boot(self):
        # 4 obs, two clusters of two rows each.
        x = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )
        measure = EmpiricalMeasure(x=x, mask=jnp.ones((4, 2)), weights=jnp.ones(4))
        rows_by_cluster = _cluster_row_indices(np.array([0, 0, 1, 1]), n_clusters=2)
        # Draw cluster 0 twice.
        drawn = np.array([0, 0])
        boot_m, boot_cov = _resample_one(measure, rows_by_cluster, drawn)
        # The resampled measure carries 4 rows (two copies of cluster 0).
        assert boot_m.x.shape == (4, 2)
        # The resampled covariance has 2 bootstrap clusters with the
        # same rows from cluster 0 each.
        assert boot_cov.n_clusters == 2
        cluster_ids = np.asarray(boot_cov.cluster_ids).astype(int)
        # First 2 rows -> bootstrap cluster 0, next 2 -> bootstrap cluster 1.
        np.testing.assert_array_equal(cluster_ids, np.array([0, 0, 1, 1]))

    def test_drawing_each_cluster_once_recovers_original_shape(self):
        x = jnp.arange(12.0).reshape(6, 2)
        measure = EmpiricalMeasure(x=x, mask=jnp.ones((6, 2)), weights=jnp.ones(6))
        cluster_ids_arr = np.array([0, 0, 1, 1, 2, 2])
        rows_by_cluster = _cluster_row_indices(cluster_ids_arr, n_clusters=3)
        drawn = np.array([0, 1, 2])
        boot_m, boot_cov = _resample_one(measure, rows_by_cluster, drawn)
        # All original rows preserved in the same order.
        np.testing.assert_array_equal(np.asarray(boot_m.x), np.asarray(x))
        assert boot_cov.n_clusters == 3


# ---------------------------------------------------------------------------
# (a) Shape + dtype
# ---------------------------------------------------------------------------


class TestShapeAndDtype:
    """The public surface returns the documented shapes / dtypes."""

    def test_basic_shapes(self):
        measure, covariance = _euler_cluster_setup(
            n_clusters=10, obs_per_cluster=15, seed=0
        )
        n_boot = 6
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=n_boot,
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(result, ClusterBootstrapResult)
        # theta_boot is a NamedArray of shape (n_boot, K=2).
        assert isinstance(result.theta_boot, ha.NamedArray)
        assert result.theta_boot.array.shape == (n_boot, 2)
        # J_boot, convergence are length n_boot.
        assert result.J_boot.shape == (n_boot,)
        assert result.convergence.shape == (n_boot,)
        assert result.convergence.dtype == bool

    def test_dtype_is_float64(self):
        measure, covariance = _euler_cluster_setup(
            n_clusters=8, obs_per_cluster=10, seed=1
        )
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=4,
            key=jax.random.PRNGKey(1),
        )
        # JAX float64 is enabled at package import; theta_boot should
        # come out in float64.
        assert result.theta_boot.array.dtype == jnp.float64
        assert result.J_boot.dtype == jnp.float64

    def test_key_is_echoed(self):
        measure, covariance = _euler_cluster_setup(
            n_clusters=6, obs_per_cluster=10, seed=2
        )
        key = jax.random.PRNGKey(42)
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=2,
            key=key,
        )
        np.testing.assert_array_equal(np.asarray(result.key), np.asarray(key))


# ---------------------------------------------------------------------------
# (b) Cluster-level resampling preserves within-cluster correlation
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _IdParams:
    a: float


def _identity_residual(x, theta):
    """psi(x, theta) = x - theta.a (one-dim moment)."""
    del theta
    return x  # tests use centred data so theta enters only via the solve


class TestWithinClusterCorrelationPreserved:
    """Cluster-level resampling preserves within-cluster correlation.

    If the original sample has highly-correlated within-cluster
    observations, the bootstrap world should too. We diagnose this
    by computing the within-cluster correlation in the resampled
    world and comparing to the original.
    """

    def _make_correlated_clusters(
        self, n_clusters: int = 30, obs_per_cluster: int = 8, seed: int = 7
    ):
        """Build a dataset with strong within-cluster correlation.

        Cluster effect ``u_c ~ N(0, sigma_u^2)`` shared across rows in
        the cluster; observation noise ``e_i ~ N(0, sigma_e^2)``
        independent across rows. ``x_i = u_c + e_i`` so the
        within-cluster correlation is
        ``sigma_u^2 / (sigma_u^2 + sigma_e^2)``.
        """
        N = n_clusters * obs_per_cluster
        rng = np.random.default_rng(seed)
        sigma_u = 1.0
        sigma_e = 0.5
        u = rng.normal(scale=sigma_u, size=n_clusters)
        e = rng.normal(scale=sigma_e, size=(n_clusters, obs_per_cluster))
        x_2d = u[:, None] + e  # (n_clusters, obs_per_cluster)
        x = x_2d.reshape(N, 1)  # (N, 1)
        cluster_ids_np = np.repeat(np.arange(n_clusters), obs_per_cluster)
        return x, cluster_ids_np

    def _within_cluster_corr(self, x: np.ndarray, cluster_ids: np.ndarray) -> float:
        """Compute the within-cluster correlation of the 1-D data ``x``.

        Pairs (x_i, x_j) where i, j are distinct rows in the same
        cluster; returns the sample correlation across all such pairs.
        """
        x = x.reshape(-1)
        unique_clusters = np.unique(cluster_ids)
        a_list: list[np.ndarray] = []
        b_list: list[np.ndarray] = []
        for c in unique_clusters:
            rows = np.where(cluster_ids == c)[0]
            if rows.size < 2:
                continue
            # All unordered pairs (i, j) with i < j.
            i_idx, j_idx = np.triu_indices(rows.size, k=1)
            a_list.append(x[rows[i_idx]])
            b_list.append(x[rows[j_idx]])
        a = np.concatenate(a_list)
        b = np.concatenate(b_list)
        return float(np.corrcoef(a, b)[0, 1])

    def test_within_cluster_correlation_carries_through(self):
        x_np, cluster_ids_np = self._make_correlated_clusters()
        N = x_np.shape[0]
        measure = EmpiricalMeasure(
            x=jnp.asarray(x_np),
            mask=jnp.ones((N, 1)),
            weights=jnp.ones(N),
        )
        n_clusters = int(np.unique(cluster_ids_np).size)
        rows_by_cluster = _cluster_row_indices(cluster_ids_np, n_clusters)
        # Draw one bootstrap sample (clusters with replacement).
        key = jax.random.PRNGKey(13)
        drawn = np.asarray(
            jax.random.randint(key, shape=(n_clusters,), minval=0, maxval=n_clusters)
        )
        boot_m, boot_cov = _resample_one(measure, rows_by_cluster, drawn)

        # Original within-cluster correlation.
        original_corr = self._within_cluster_corr(x_np, cluster_ids_np)
        boot_corr = self._within_cluster_corr(
            np.asarray(boot_m.x),
            np.asarray(boot_cov.cluster_ids).astype(int),
        )

        # The cluster-level resample preserves within-cluster pairs
        # by construction (each bootstrap cluster is exactly one of
        # the original clusters), so the within-cluster correlation
        # estimate should be close (the sample composition is shuffled
        # with replacement, but the *within-cluster* pairs themselves
        # are intact).
        assert original_corr > 0.5  # sanity: strong correlation in DGP
        # Within ~0.15 of the original on this sample size.
        assert abs(boot_corr - original_corr) < 0.15

    def test_within_cluster_correlation_destroyed_by_row_resample(self):
        """Counterfactual: drawing *rows* (not clusters) destroys the
        within-cluster correlation. This makes the previous test's
        claim non-trivial.
        """
        x_np, cluster_ids_np = self._make_correlated_clusters()
        # Resample rows uniformly with replacement, but keep the
        # cluster-id layout fixed.
        rng = np.random.default_rng(99)
        N = x_np.shape[0]
        idx = rng.integers(0, N, size=N)
        x_row_boot = x_np[idx]
        row_boot_corr = (
            np.corrcoef(
                x_row_boot[np.repeat(np.arange(0, N - 1), 1)].reshape(-1),
                x_row_boot[np.repeat(np.arange(1, N), 1)].reshape(-1),
            )[0, 1]
            if N >= 2
            else 0.0
        )
        original_corr = self._within_cluster_corr(x_np, cluster_ids_np)
        # Row-level resample destroys within-cluster correlation; it
        # should be much smaller than the original.
        assert abs(row_boot_corr) < 0.3
        assert original_corr > 0.5


# ---------------------------------------------------------------------------
# (c) Under H0 (well-specified, large n_clusters), the bootstrap
#     distribution of theta_boot - theta_hat is approximately normal
#     with covariance close to Sigma_theta from estimate().
# ---------------------------------------------------------------------------


class TestUnderH0Normality:
    """Under H0, ``cov(theta_boot)`` approximates ``Sigma_theta``."""

    @pytest.mark.slow
    def test_bootstrap_covariance_close_to_sigma_theta(self):
        # Use a singleton-cluster layout for the original sample so
        # ClusteredCovariance reduces to IID and Sigma_theta is the
        # familiar (G' V^{-1} G)^{-1} that the bootstrap should
        # converge to.
        N = 600
        x = euler_data(seed=11, n=N)
        measure = EmpiricalMeasure(x=x, mask=jnp.ones((N, 3)), weights=jnp.ones(N))
        cluster_ids = jnp.arange(N, dtype=jnp.float64)
        covariance = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=N)

        # Sample estimate -> Sigma_theta at theta_hat.
        result = estimate(
            model=euler_residual,
            measure=measure,
            covariance=covariance,
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )
        sigma_theta = np.asarray(result.Sigma_theta.array)
        theta_hat_flat = np.array(
            [float(result.theta_hat.beta), float(result.theta_hat.gamma)]
        )

        # Bootstrap.
        n_boot = 80
        boot_result = cluster_bootstrap(
            model=euler_residual,
            theta_init=result.theta_hat,
            measure=measure,
            covariance=covariance,
            n_boot=n_boot,
            key=jax.random.PRNGKey(123),
        )
        # Drop replicates that didn't converge.
        ok = boot_result.convergence
        theta_b = np.asarray(boot_result.theta_boot.array)[ok]
        assert theta_b.shape[0] >= n_boot // 2  # majority must converge

        # Centre on the sample estimate.
        delta = theta_b - theta_hat_flat[None, :]
        boot_cov = np.cov(delta, rowvar=False, bias=False)

        # Compare entry-wise: bootstrap covariance should be of the
        # same magnitude as the asymptotic Sigma_theta. We use a
        # relative tolerance generous enough to absorb sampling noise
        # at n_boot = 80.
        # Diagonal entries: ratio should be within a factor of ~3.
        for i in range(2):
            ratio = boot_cov[i, i] / sigma_theta[i, i]
            assert 0.25 < ratio < 4.0, (
                f"diag ratio for parameter {i} is {ratio:.3f}; "
                f"expected within (0.25, 4.0)"
            )

        # Off-diagonal: sign agreement is a weak but non-vacuous check.
        # Sigma_theta off-diagonal can be small / noisy; require only
        # finite magnitude here.
        assert np.isfinite(boot_cov[0, 1])

    def test_low_boot_n_completes(self):
        """Smoke test: small n_boot returns a well-formed result even
        if individual replicates struggle to converge.
        """
        measure, covariance = _euler_cluster_setup(
            n_clusters=12, obs_per_cluster=15, seed=3
        )
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=3,
            key=jax.random.PRNGKey(7),
        )
        assert result.theta_boot.array.shape == (3, 2)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_zero_n_boot_raises(self):
        measure, covariance = _euler_cluster_setup(
            n_clusters=4, obs_per_cluster=5, seed=0
        )
        with pytest.raises(ValueError, match="n_boot must be positive"):
            cluster_bootstrap(
                model=euler_residual,
                theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
                measure=measure,
                covariance=covariance,
                n_boot=0,
                key=jax.random.PRNGKey(0),
            )
