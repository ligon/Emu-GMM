"""Tests for emu_gmm.covariance.clustered."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.types import CovarianceStrategy


@jdc.pytree_dataclass
class _P:
    a: float
    b: float


def _identity_psi(x, theta):
    """psi(x, theta) = x."""
    return x


# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_covariance_protocol(self):
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 1.0])
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2)
        assert isinstance(cov, CovarianceStrategy)


# ---------------------------------------------------------------------------


class TestSingletonClusters:
    """With each cluster of size one, the cluster-totals form reduces to
    :class:`IIDCovariance`.
    """

    def test_matches_iid_on_uniform_weights_full_mask(self):
        N = 8
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (N, 3))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 3)),
            weights=jnp.ones(N),
        )
        iid = IIDCovariance()
        # Singleton clusters: id[i] = i; n_clusters = N.
        cluster_ids = jnp.arange(N, dtype=jnp.float32)
        clustered = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=N)
        V_iid = iid.covariance(_identity_psi, _P(0.0, 0.0), meas)
        V_clu = clustered.covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert jnp.allclose(V_iid, V_clu, atol=1e-7)

    def test_matches_iid_with_mask_and_weights(self):
        """Same equivalence with non-trivial mask and non-uniform weights."""
        x = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        weights = jnp.array([1.0, 0.5, 2.0, 1.5, 1.0])
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        cluster_ids = jnp.arange(5, dtype=jnp.float32)
        V_iid = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        V_clu = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=5).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        assert jnp.allclose(V_iid, V_clu, atol=1e-6)


# ---------------------------------------------------------------------------


class TestTwoClusterFormula:
    """Two clusters, hand-computed against the analytic formula."""

    def test_two_clusters_known_values(self):
        # 4 observations, 2 clusters of 2 each, 2 moments.
        # psi_i = x_i = the observations themselves.
        psi_vals = jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [-1.0, 1.0],
                [2.0, -2.0],
            ]
        )
        N = 4
        meas = EmpiricalMeasure(
            x=psi_vals,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 1.0])
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2)
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)

        # Cluster totals: c0 = (1 + 3, 2 + 4) = (4, 6);
        # c1 = (-1 + 2, 1 - 2) = (1, -1).
        # numer_jk = sum_c c_j * c_k:
        #  [0,0] = 4*4 + 1*1 = 17
        #  [0,1] = 4*6 + 1*(-1) = 23
        #  [1,1] = 6*6 + (-1)^2 = 37
        # N_j = 4 for both j.
        # V_jk = numer_jk / (N_j * N_k) = / 16.
        assert float(V[0, 0]) == pytest.approx(17.0 / 16.0, rel=1e-6)
        assert float(V[0, 1]) == pytest.approx(23.0 / 16.0, rel=1e-6)
        assert float(V[1, 0]) == pytest.approx(23.0 / 16.0, rel=1e-6)
        assert float(V[1, 1]) == pytest.approx(37.0 / 16.0, rel=1e-6)

    def test_two_clusters_picks_up_within_cluster_correlation(self):
        """If observations within the same cluster are perfectly aligned,
        the cluster-robust V differs from the IID V.
        """
        # Make the within-cluster contributions correlated by construction:
        # cluster 0 has both rows positive, cluster 1 has both rows negative.
        psi_vals = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [-1.0, -1.0],
            ]
        )
        N = 4
        meas = EmpiricalMeasure(
            x=psi_vals,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        # IID covariance.
        V_iid = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        # Clustered with two clusters of 2.
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 1.0])
        V_clu = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        # IID: sum_i psi_i psi_i' = 4 * [[1,1],[1,1]] = [[4,4],[4,4]];
        # divided by N^2 = 16 -> [[0.25, 0.25],[0.25, 0.25]].
        np.testing.assert_allclose(np.asarray(V_iid), np.full((2, 2), 0.25), atol=1e-7)
        # Clustered: totals (2,2) and (-2,-2); outer-product sum = 2 * [[4,4],[4,4]]
        # = [[8,8],[8,8]]; divided by 16 = [[0.5,0.5],[0.5,0.5]].
        np.testing.assert_allclose(np.asarray(V_clu), np.full((2, 2), 0.5), atol=1e-7)
        # Clustered exceeds IID when within-cluster correlation is positive.
        assert float(V_clu[0, 0]) > float(V_iid[0, 0])


# ---------------------------------------------------------------------------


class TestNaNSafety:
    """Cluster-totals form guards against NaN at masked-out cells."""

    def test_nan_in_psi_at_masked_cells_does_not_poison(self):
        """psi returning NaN where mask == 0 still yields a finite V."""
        # 4 obs, 2 moments, 2 clusters. Moment 1 missing on rows 0, 2.
        x = jnp.array(
            [
                [1.0, jnp.nan],
                [2.0, 20.0],
                [3.0, jnp.nan],
                [4.0, 40.0],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(4))
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 1.0])
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2)
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert bool(jnp.all(jnp.isfinite(V)))


class TestJit:
    def test_covariance_jits(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((10, 2)),
            weights=jnp.ones(10),
        )
        cluster_ids = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=5)
        theta = _P(0.0, 0.0)

        @jax.jit
        def compute(c, t, m):
            return c.covariance(_identity_psi, t, m)

        V_eager = cov.covariance(_identity_psi, theta, meas)
        V_jit = compute(cov, theta, meas)
        assert jnp.allclose(V_eager, V_jit, atol=1e-7)
