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


# ---------------------------------------------------------------------------


def _numpy_cluster_correction(mask, cluster_ids, n_clusters):
    """Independent per-pair G_jk/(G_jk-1) correction matrix."""
    mask = np.asarray(mask)
    cl = np.rint(np.asarray(cluster_ids)).astype(int)
    M = mask.shape[1]
    s = np.zeros((n_clusters, M))
    for i in range(len(cl)):
        s[cl[i]] += (mask[i] > 0).astype(float)
    s = (s > 0).astype(float)
    G = s.T @ s  # (M, M) per-pair cluster counts
    return np.where(G >= 2, G / np.where(G >= 2, G - 1, 1.0), 1.0)


class TestDofCorrection:
    """Finite-cluster G_jk/(G_jk-1) degrees-of-freedom correction (#82 item 1)."""

    def test_complete_data_is_scalar_G_over_Gminus1(self):
        # 2 clusters, complete data -> G_jk == 2 for every pair -> factor 2.0.
        psi_vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -2.0]])
        meas = EmpiricalMeasure(x=psi_vals, mask=jnp.ones((4, 2)), weights=jnp.ones(4))
        cl = jnp.array([0.0, 0.0, 1.0, 1.0])
        V0 = ClusteredCovariance(cluster_ids=cl, n_clusters=2).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        V1 = ClusteredCovariance(
            cluster_ids=cl, n_clusters=2, dof_correction=True
        ).covariance(_identity_psi, _P(0.0, 0.0), meas)
        np.testing.assert_allclose(np.asarray(V1), 2.0 * np.asarray(V0), atol=1e-7)

    def test_per_pair_under_missingness_matches_numpy(self):
        # coord 1 observed only in clusters 0,1; coord 0 in all 3 clusters.
        x = jnp.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
        )
        mask = jnp.array(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
        )
        cl = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(6))
        V0 = ClusteredCovariance(cluster_ids=cl, n_clusters=3).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        V1 = ClusteredCovariance(
            cluster_ids=cl, n_clusters=3, dof_correction=True
        ).covariance(_identity_psi, _P(0.0, 0.0), meas)
        corr = _numpy_cluster_correction(mask, cl, 3)
        # G_00=3 -> 1.5 ; G_11=2 -> 2 ; G_01=2 -> 2 (per-pair, NOT a single scalar)
        assert corr[0, 0] == pytest.approx(1.5)
        assert corr[1, 1] == pytest.approx(2.0)
        assert corr[0, 1] == pytest.approx(2.0)
        np.testing.assert_allclose(np.asarray(V1), np.asarray(V0) * corr, atol=1e-9)

    def test_default_is_uncorrected_and_correction_inflates(self):
        psi_vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -2.0]])
        meas = EmpiricalMeasure(x=psi_vals, mask=jnp.ones((4, 2)), weights=jnp.ones(4))
        cl = jnp.array([0.0, 0.0, 1.0, 1.0])
        V_default = ClusteredCovariance(cluster_ids=cl, n_clusters=2).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        V_off = ClusteredCovariance(
            cluster_ids=cl, n_clusters=2, dof_correction=False
        ).covariance(_identity_psi, _P(0.0, 0.0), meas)
        V_on = ClusteredCovariance(
            cluster_ids=cl, n_clusters=2, dof_correction=True
        ).covariance(_identity_psi, _P(0.0, 0.0), meas)
        np.testing.assert_array_equal(np.asarray(V_default), np.asarray(V_off))
        # correction strictly inflates the (nonzero) diagonal
        assert float(V_on[0, 0]) > float(V_off[0, 0])

    def test_cached_self_parity_with_correction(self):
        x = jnp.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
        )
        mask = jnp.array(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
        )
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(6))
        cov = ClusteredCovariance(
            cluster_ids=jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            n_clusters=3,
            dof_correction=True,
        )
        cached = meas.expectation_and_contributions(_identity_psi, _P(0.0, 0.0))
        V_self = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        V_cached = cov.covariance(
            _identity_psi, _P(0.0, 0.0), meas, cached_intermediates=cached
        )
        np.testing.assert_array_equal(np.asarray(V_self), np.asarray(V_cached))

    def test_is_pytree_static_field(self):
        cov = ClusteredCovariance(
            cluster_ids=jnp.array([0.0, 1.0]), n_clusters=2, dof_correction=True
        )
        # dof_correction is static -> NOT a pytree leaf; jit-safe.
        leaves = jax.tree_util.tree_leaves(cov)
        assert not any(np.asarray(leaf).dtype == bool for leaf in leaves)
        assert isinstance(cov, CovarianceStrategy)


class TestClusterIdRobustness:
    """Cluster ids are rounded (not truncated) and integer dtype is accepted
    (#82 item 3).
    """

    def test_near_integer_float_ids_round_not_truncate(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -2.0]])
        meas = EmpiricalMeasure(x=x, mask=jnp.ones((4, 2)), weights=jnp.ones(4))
        theta = _P(0.0, 0.0)
        ids_exact = jnp.array([0.0, 1.0, 2.0, 2.0])
        # Obs 1's id sits a hair below 1.0 -- rounds to 1, truncates to 0.
        ids_near = jnp.array([0.0, 1.0 - 1e-7, 2.0, 2.0])
        V_exact = ClusteredCovariance(cluster_ids=ids_exact, n_clusters=3).covariance(
            _identity_psi, theta, meas
        )
        V_near = ClusteredCovariance(cluster_ids=ids_near, n_clusters=3).covariance(
            _identity_psi, theta, meas
        )
        np.testing.assert_allclose(np.asarray(V_near), np.asarray(V_exact), atol=1e-9)
        # Truncation would mis-bin obs 1 into cluster 0 -> a genuinely
        # different V (confirms the test discriminates the bug).
        V_trunc = ClusteredCovariance(
            cluster_ids=jnp.array([0.0, 0.0, 2.0, 2.0]), n_clusters=3
        ).covariance(_identity_psi, theta, meas)
        assert np.max(np.abs(np.asarray(V_exact) - np.asarray(V_trunc))) > 1e-6

    def test_integer_dtype_ids_accepted(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -2.0]])
        meas = EmpiricalMeasure(x=x, mask=jnp.ones((4, 2)), weights=jnp.ones(4))
        theta = _P(0.0, 0.0)
        V_int = ClusteredCovariance(
            cluster_ids=jnp.array([0, 0, 1, 1], dtype=jnp.int32), n_clusters=2
        ).covariance(_identity_psi, theta, meas)
        V_float = ClusteredCovariance(
            cluster_ids=jnp.array([0.0, 0.0, 1.0, 1.0]), n_clusters=2
        ).covariance(_identity_psi, theta, meas)
        np.testing.assert_allclose(np.asarray(V_int), np.asarray(V_float), atol=1e-9)


# ---------------------------------------------------------------------------
# The centered= knob, shared with IIDCovariance via _OuterProductCovariance:
# subtract the per-coordinate mean moment before the per-cluster totals.
# ---------------------------------------------------------------------------
def _clustered_centered_reference(X, mask, cluster_ids, n_clusters, dof=False):
    """Independent numpy reference for the masked, centered clustered V_X."""
    X, mask = np.asarray(X), np.asarray(mask)
    cl = np.rint(np.asarray(cluster_ids)).astype(int)
    N, M = X.shape
    Nj = (mask).sum(0)  # unit weights
    m = (mask * X).sum(0) / Nj  # per-coordinate observed mean
    contrib = mask * (X - m[None, :])  # d_ij * (psi_j - m_j); masked rows -> 0
    totals = np.zeros((n_clusters, M))
    for i in range(N):
        totals[cl[i]] += contrib[i]
    numer = totals.T @ totals
    if dof:
        numer = numer * _numpy_cluster_correction(mask, cluster_ids, n_clusters)
    return numer / np.outer(Nj, Nj)


class TestCenteredKnob:
    def test_static_field_default_false(self):
        cl = jnp.array([0.0, 0.0, 1.0, 1.0])
        assert ClusteredCovariance(cluster_ids=cl, n_clusters=2).centered is False
        assert (
            ClusteredCovariance(cluster_ids=cl, n_clusters=2, centered=True).centered
            is True
        )

    def test_matches_numpy_reference_masked(self):
        X = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 6.0],
            ]
        )
        mask = np.array([[1, 1], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1]], dtype=float)
        cl = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.asarray(mask), weights=jnp.ones(6)
        )
        for dof in (False, True):
            vc = np.asarray(
                ClusteredCovariance(
                    cluster_ids=cl, n_clusters=3, centered=True, dof_correction=dof
                ).covariance(_identity_psi, _P(0.0, 0.0), meas)
            )
            ref = _clustered_centered_reference(X, mask, cl, 3, dof=dof)
            np.testing.assert_allclose(vc, ref, rtol=1e-6, atol=1e-9)

    def test_equal_when_moments_already_have_zero_mean(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0.0, 1.0, size=(8, 2))
        X = X - X.mean(axis=0)  # exact zero column means -> centering is a no-op
        cl = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.ones((8, 2)), weights=jnp.ones(8)
        )
        base = ClusteredCovariance(cluster_ids=cl, n_clusters=4)
        cent = ClusteredCovariance(cluster_ids=cl, n_clusters=4, centered=True)
        vu = np.asarray(base.covariance(_identity_psi, _P(0.0, 0.0), meas))
        vc = np.asarray(cent.covariance(_identity_psi, _P(0.0, 0.0), meas))
        np.testing.assert_allclose(vu, vc, atol=1e-12)

    def test_cached_self_parity_centered(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        mask = np.array([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=float)
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.asarray(mask), weights=jnp.ones(4)
        )
        cov = ClusteredCovariance(
            cluster_ids=jnp.array([0.0, 0.0, 1.0, 1.0]), n_clusters=2, centered=True
        )
        cached = meas.expectation_and_contributions(_identity_psi, _P(0.0, 0.0))
        V_self = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        V_cached = cov.covariance(
            _identity_psi, _P(0.0, 0.0), meas, cached_intermediates=cached
        )
        np.testing.assert_allclose(np.asarray(V_self), np.asarray(V_cached), atol=1e-12)


class TestSingletonReducesToIID:
    """IID is the singleton-cluster reduction of Clustered in BOTH centering
    directions --- the invariant the shared _OuterProductCovariance template
    guarantees (same mean subtracted on both sides).
    """

    @pytest.mark.parametrize("centered", [False, True])
    def test_singleton_matches_iid_masked_weighted(self, centered):
        X = jnp.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]])
        mask = jnp.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
        weights = jnp.array([1.0, 0.5, 2.0, 1.5, 1.0])
        meas = EmpiricalMeasure(x=X, mask=mask, weights=weights)
        cl = jnp.arange(5, dtype=jnp.float32)  # singleton clusters
        V_iid = IIDCovariance(centered=centered).covariance(
            _identity_psi, _P(0.0, 0.0), meas
        )
        V_clu = ClusteredCovariance(
            cluster_ids=cl, n_clusters=5, centered=centered
        ).covariance(_identity_psi, _P(0.0, 0.0), meas)
        np.testing.assert_allclose(np.asarray(V_iid), np.asarray(V_clu), atol=1e-12)
