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
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.penalty import TikhonovPenalty
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated
from jaxtyping import Array, Float

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


# ---------------------------------------------------------------------------
# Parameter-name labelling (HIGH #1 in PR #37 review)
# ---------------------------------------------------------------------------


class TestParamNamesPreserved:
    """The parameter names from ``theta_init``'s dataclass survive
    through to :attr:`ClusterBootstrapResult.coords` and the
    :attr:`ClusterBootstrapResult.param_names` field.
    """

    def test_coords_parameters_matches_input_param_names(self):
        """``result.coords['parameters']`` echoes the dataclass field
        names in PyTree-flatten order (the same order that
        ``EstimationResult.coef_table`` indexes by).
        """
        from emu_gmm._internal import params as params_mod

        measure, covariance = _euler_cluster_setup(
            n_clusters=5, obs_per_cluster=8, seed=0
        )
        theta_init = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
        expected_names = tuple(params_mod.param_names(theta_init))
        # Sanity: EulerParams has fields (beta, gamma) in that order.
        assert expected_names == ("beta", "gamma")

        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=theta_init,
            measure=measure,
            covariance=covariance,
            n_boot=3,
            key=jax.random.PRNGKey(0),
        )

        # The result carries the parameter names verbatim ...
        assert result.param_names == expected_names
        # ... and exposes them via the ``coords`` mapping under the
        # canonical ``parameters`` axis key, so downstream tabular
        # gestures (``pd.Series(boot_se, index=result.coords['parameters'])``)
        # match ``EstimationResult.coef_table.index``.
        assert result.coords["parameters"] == expected_names
        # The bootstrap axis carries positional replicate indices.
        assert result.coords["bootstrap"] == (0, 1, 2)


# ---------------------------------------------------------------------------
# Inner-failure surfacing (HIGH #2 in PR #37 review)
# ---------------------------------------------------------------------------


class TestInnerFailureSurfacing:
    """A deliberately-failing inner ``estimate()`` call must surface
    as either (a) a non-convergence flag (for documented divergence
    pathways) or (b) a raised exception (for everything else,
    including programming bugs in the user's ``psi`` function).

    The previous code wrapped the inner call in ``except Exception``
    which silently turned a bug-driven ``TypeError`` into a NaN row.
    The fix is to catch only the documented divergence exceptions
    and let everything else propagate.
    """

    def test_user_bug_propagates_rather_than_being_swallowed(self, monkeypatch):
        """A ``TypeError`` raised from inside ``estimate()`` -- standing
        in for any bug in the user's ``psi`` function -- must surface
        as a raised exception, not as a swallowed NaN row.
        """
        # Pull the *submodule* out of ``sys.modules``; the package
        # ``emu_gmm.inference`` re-exports the function of the same
        # name as the submodule, so a plain ``import`` of
        # ``emu_gmm.inference.cluster_bootstrap`` resolves to the
        # function (Python attribute lookup wins over the submodule).
        # ``sys.modules`` is the canonical way to grab the module
        # object regardless of any name shadowing in the parent
        # package, which is exactly what ``monkeypatch.setattr``
        # needs to rebind the ``estimate`` name that
        # ``cluster_bootstrap`` looks up at call time.
        import sys

        cb_mod = sys.modules["emu_gmm.inference.cluster_bootstrap"]

        def _buggy_estimate(*args, **kwargs):
            # Stand-in for a bug in the user's psi function: a
            # TypeError that the old broad ``except Exception``
            # would have silently masked as ``convergence=False``.
            raise TypeError(
                "intentional bug: user's psi function returned the wrong type"
            )

        monkeypatch.setattr(cb_mod, "estimate", _buggy_estimate)

        measure, covariance = _euler_cluster_setup(
            n_clusters=4, obs_per_cluster=5, seed=0
        )
        with pytest.raises(TypeError, match="intentional bug"):
            cluster_bootstrap(
                model=euler_residual,
                theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
                measure=measure,
                covariance=covariance,
                n_boot=2,
                key=jax.random.PRNGKey(0),
            )

    def test_documented_divergence_surfaces_as_non_convergence_flag(self, monkeypatch):
        """A ``LinAlgError`` raised from inside ``estimate()`` --
        standing in for a singular-matrix divergence -- is one of
        the documented bootstrap-divergence pathways and must
        surface as ``convergence=False`` plus a NaN row in
        ``theta_boot`` / ``J_boot``, *not* as a raised exception.
        """
        # See ``test_user_bug_propagates_rather_than_being_swallowed``
        # for why we go through ``sys.modules``.
        import sys

        cb_mod = sys.modules["emu_gmm.inference.cluster_bootstrap"]

        def _diverging_estimate(*args, **kwargs):
            raise np.linalg.LinAlgError(
                "intentional divergence: singular V on this bootstrap world"
            )

        monkeypatch.setattr(cb_mod, "estimate", _diverging_estimate)

        measure, covariance = _euler_cluster_setup(
            n_clusters=4, obs_per_cluster=5, seed=0
        )
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=3,
            key=jax.random.PRNGKey(0),
        )
        # All three replicates diverged in the documented way: the
        # framework records this on the result rather than raising.
        assert not np.any(result.convergence)
        assert np.all(np.isnan(np.asarray(result.theta_boot.array)))
        assert np.all(np.isnan(np.asarray(result.J_boot)))


# ---------------------------------------------------------------------------
# Pytree-dataclass invariant (#55)
# ---------------------------------------------------------------------------


class TestPytreeRoundTrip:
    """``ClusterBootstrapResult`` is a ``@jdc.pytree_dataclass``.

    The other three inference-result types
    (:class:`JTestResult`, :class:`KStatisticResult`,
    :class:`WildBootstrapResult`) are already pytree-dataclasses;
    ``ClusterBootstrapResult`` joins them so the whole inference
    surface honours the same ``vmap`` / ``jit`` contract. This test
    pins the invariant.
    """

    def test_tree_flatten_unflatten_round_trip(self):
        """The result is a JAX pytree: flatten then unflatten reproduces it."""
        measure, covariance = _euler_cluster_setup(
            n_clusters=5, obs_per_cluster=8, seed=0
        )
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=3,
            key=jax.random.PRNGKey(0),
        )
        leaves, treedef = jax.tree_util.tree_flatten(result)
        # Non-empty leaf list confirms registration as a pytree.
        assert len(leaves) > 0
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, ClusterBootstrapResult)
        # Static field (param_names) rides on the treedef; traced leaves
        # are identical objects after the no-op round trip.
        assert rebuilt.param_names == result.param_names
        np.testing.assert_array_equal(
            np.asarray(rebuilt.theta_boot.array),
            np.asarray(result.theta_boot.array),
        )
        np.testing.assert_array_equal(
            np.asarray(rebuilt.J_boot), np.asarray(result.J_boot)
        )
        np.testing.assert_array_equal(
            np.asarray(rebuilt.convergence), np.asarray(result.convergence)
        )

    def test_tree_map_doubles_traced_leaves(self):
        """``jax.tree_util.tree_map`` reaches the traced fields.

        A scalar multiply applied via ``tree_map`` doubles the numeric
        leaves (``theta_boot``, ``J_boot``, ``convergence`` cast,
        ``key``); static fields ride along on the treedef untouched.
        """
        measure, covariance = _euler_cluster_setup(
            n_clusters=4, obs_per_cluster=6, seed=1
        )
        result = cluster_bootstrap(
            model=euler_residual,
            theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            measure=measure,
            covariance=covariance,
            n_boot=2,
            key=jax.random.PRNGKey(1),
        )
        # nan_to_num because diverged replicates produce NaNs in
        # theta_boot / J_boot and 2 * NaN is still NaN; we want a
        # numeric comparison.
        doubled = jax.tree_util.tree_map(lambda a: jnp.nan_to_num(a) * 2, result)
        assert isinstance(doubled, ClusterBootstrapResult)
        # Static fields are preserved.
        assert doubled.param_names == result.param_names
        np.testing.assert_allclose(
            np.asarray(doubled.theta_boot.array),
            2.0 * np.nan_to_num(np.asarray(result.theta_boot.array)),
        )
        np.testing.assert_allclose(
            np.asarray(doubled.J_boot),
            2.0 * np.nan_to_num(np.asarray(result.J_boot)),
        )

    def test_vmap_over_seeds_stacks_results(self):
        """Multiple bootstrap calls stack into a single batched pytree.

        ``cluster_bootstrap`` itself uses a host-side Python loop over
        replicates and cannot be traced under ``vmap`` directly. But
        the *output* is a pytree, so several independent calls can be
        stacked along a leading "seed" axis via
        ``jax.tree_util.tree_map(jnp.stack, *results)`` --- the
        regression check for #55, where the previous plain
        ``@dataclass(frozen=True)`` form raised at the stacking step
        because the parent treedef wasn't registered.
        """
        measure, covariance = _euler_cluster_setup(
            n_clusters=5, obs_per_cluster=6, seed=2
        )
        seeds = [10, 11, 12]
        results = [
            cluster_bootstrap(
                model=euler_residual,
                theta_init=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
                measure=measure,
                covariance=covariance,
                n_boot=2,
                key=jax.random.PRNGKey(s),
            )
            for s in seeds
        ]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *results)
        assert isinstance(stacked, ClusterBootstrapResult)
        # Leading axis = number of seeds; per-replicate axes preserved.
        assert stacked.theta_boot.array.shape == (len(seeds), 2, 2)
        assert stacked.J_boot.shape == (len(seeds), 2)
        assert stacked.convergence.shape == (len(seeds), 2)
        # Static field rebuilt from the first call's treedef and shared
        # across all stacked results (same parameter names; checked
        # via tree-equality on the static fields).
        assert stacked.param_names == results[0].param_names


# ---------------------------------------------------------------------------
# #150: penalty passthrough + manifold / non-scalar parameter support
#
# Two independent blockers fixed together, both inside cluster_bootstrap:
#   (1) the v1 scalar-only ``flatten_params`` rejected manifold leaves, so
#       the whole refit bootstrap died on a ``PSDFixedRank`` factor;
#   (2) ``penalty=`` was not forwarded to the per-replicate ``estimate()``,
#       so a penalized point estimate was refit through an *unpenalized*
#       objective (a different, non-reportable optimum, not just a wider CI).
# ---------------------------------------------------------------------------


def _reconstruct_replicate0(
    measure,
    covariance,
    theta_init,
    model,
    *,
    key,
    penalty,
    optimizer=None,
):
    """Deterministically rebuild bootstrap replicate ``b = 0`` and refit it.

    Mirrors :func:`cluster_bootstrap`'s internal draw *exactly* --
    ``split(key, n_boot=1)`` then a uniform ``randint`` over clusters --
    so the package's ``theta_boot[0]`` can be compared against an
    independently-computed refit. That is the rigorous check that the
    refit actually used the forwarded ``penalty`` (and the deferred
    optimiser auto-dispatch), not a behavioural proxy.
    """
    cluster_ids_np = np.asarray(covariance.cluster_ids).astype(np.int64)
    n_clusters = int(covariance.n_clusters)
    rows_by_cluster = _cluster_row_indices(cluster_ids_np, n_clusters)
    keys = jax.random.split(key, 1)
    drawn = np.asarray(
        jax.random.randint(keys[0], shape=(n_clusters,), minval=0, maxval=n_clusters)
    )
    boot_measure, boot_cov = _resample_one(measure, rows_by_cluster, drawn)
    return estimate(
        model=model,
        measure=boot_measure,
        covariance=boot_cov,
        weighting=ContinuouslyUpdated(),
        regularization=DiagonalTikhonov(),
        optimizer=optimizer,
        penalty=penalty,
        theta_init=theta_init,
    )


class TestPenaltyForwarding:
    """``penalty=`` reaches each refit (#150, second blocker)."""

    def test_penalty_changes_refit_and_is_forwarded(self):
        """The refit re-solves the *penalized* objective, not the
        unpenalized one. Proven by exact reconstruction of replicate 0.
        """
        measure, covariance = _euler_cluster_setup(
            n_clusters=10, obs_per_cluster=20, seed=0
        )
        theta_init = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
        penalty = TikhonovPenalty(c=jnp.asarray(2.0))
        key = jax.random.PRNGKey(20240617)

        boot = cluster_bootstrap(
            model=euler_residual,
            theta_init=theta_init,
            measure=measure,
            covariance=covariance,
            n_boot=1,
            key=key,
            penalty=penalty,
        )
        boot0 = np.asarray(boot.theta_boot.array)[0]

        res_pen = _reconstruct_replicate0(
            measure, covariance, theta_init, euler_residual, key=key, penalty=penalty
        )
        recon_pen = np.array(
            [float(res_pen.theta_hat.beta), float(res_pen.theta_hat.gamma)]
        )
        res_nopen = _reconstruct_replicate0(
            measure, covariance, theta_init, euler_residual, key=key, penalty=None
        )
        recon_nopen = np.array(
            [float(res_nopen.theta_hat.beta), float(res_nopen.theta_hat.gamma)]
        )

        # The penalty is non-trivial: it moves the optimum.
        assert np.max(np.abs(recon_pen - recon_nopen)) > 1e-3
        # The bootstrap refit matched the *penalized* reconstruction ...
        np.testing.assert_allclose(boot0, recon_pen, rtol=1e-6, atol=1e-8)
        # ... and NOT the unpenalized one (the pre-#150 behaviour).
        assert np.max(np.abs(boot0 - recon_nopen)) > 1e-3

    def test_default_penalty_none_is_bitwise_unpenalized(self):
        """Omitting ``penalty`` reproduces the prior (unpenalized) refit."""
        measure, covariance = _euler_cluster_setup(
            n_clusters=10, obs_per_cluster=20, seed=1
        )
        theta_init = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)
        key = jax.random.PRNGKey(7)
        boot = cluster_bootstrap(
            model=euler_residual,
            theta_init=theta_init,
            measure=measure,
            covariance=covariance,
            n_boot=1,
            key=key,
        )
        boot0 = np.asarray(boot.theta_boot.array)[0]
        res_nopen = _reconstruct_replicate0(
            measure, covariance, theta_init, euler_residual, key=key, penalty=None
        )
        recon_nopen = np.array(
            [float(res_nopen.theta_hat.beta), float(res_nopen.theta_hat.gamma)]
        )
        np.testing.assert_allclose(boot0, recon_nopen, rtol=1e-6, atol=1e-8)


# --- manifold fixture: PSDFixedRank(4, 2) factor A + Euclidean(1) phi ------
# Gauge-invariant moments triu(A A') ++ phi (adapted from
# tests/inference/test_k_statistic_gauge.py).

_N_SIDE = 4
_K_RANK = 2
_GAUGE_DIM = _K_RANK * (_K_RANK - 1) // 2  # 1
_AMBIENT_P = _N_SIDE * _K_RANK + 1  # 9 (vec A ++ phi)
_M_MANIFOLD = _N_SIDE * (_N_SIDE + 1) // 2 + 1  # 11
_TRIU_IDX = jnp.array(np.triu_indices(_N_SIDE)).T


@jdc.pytree_dataclass
class _ManifoldParams:
    """``PSDFixedRank(4, 2)`` factor ``A`` + ``Euclidean(1)`` ``phi``."""

    A: ManifoldLeaf
    phi: ManifoldLeaf


def _make_manifold_params(A, phi) -> _ManifoldParams:
    return _ManifoldParams(
        A=ManifoldLeaf(jnp.asarray(A), PSDFixedRank(_N_SIDE, _K_RANK)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _manifold_model(x, theta):
    """psi = (triu(A A') ++ phi) - x: gauge-invariant in A by construction."""
    A = theta.A.array
    phi = theta.phi.array[0]
    g = (A @ A.T)[_TRIU_IDX[:, 0], _TRIU_IDX[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _manifold_cluster_setup(n_clusters=6, obs_per_cluster=40, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(_N_SIDE, _K_RANK)))
    phi_true = 0.7
    g_true = (A_true @ A_true.T)[_TRIU_IDX[:, 0], _TRIU_IDX[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    N = n_clusters * obs_per_cluster
    x = np.asarray(target)[None, :] + noise * rng.standard_normal((N, _M_MANIFOLD))
    measure = EmpiricalMeasure(
        x=jnp.asarray(x),
        mask=jnp.ones((N, _M_MANIFOLD)),
        weights=jnp.ones(N),
    )
    cluster_ids = jnp.repeat(jnp.arange(n_clusters, dtype=jnp.float64), obs_per_cluster)
    covariance = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=n_clusters)
    theta_true = _make_manifold_params(A_true, phi_true)
    return theta_true, measure, covariance


@pytest.mark.slow
class TestManifoldSupport:
    """``cluster_bootstrap`` runs on a manifold parameter tree (#150,
    first blocker). Pre-#150 it died in the scalar-only ``flatten_params``
    (``"all parameter leaves must be 0-d scalars in v1"``) before any
    resampling.
    """

    def test_runs_on_psd_fixed_rank_tree(self):
        theta_true, measure, covariance = _manifold_cluster_setup()
        # optimizer=None -> estimate() auto-dispatches riemannian_lm() for
        # the manifold tree (the deferred-default fix; the v1 optimistix_lm
        # cannot retract on a PSDFixedRank factor).
        result = cluster_bootstrap(
            model=_manifold_model,
            theta_init=theta_true,
            measure=measure,
            covariance=covariance,
            n_boot=4,
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(result, ClusterBootstrapResult)
        # theta_boot is over the AMBIENT flatten axis (9 = 4*2 + 1),
        # matching Sigma_theta's parameters axis.
        assert result.theta_boot.array.shape == (4, _AMBIENT_P)
        assert result.J_boot.shape == (4,)
        assert result.convergence.shape == (4,)

    def test_param_names_are_ambient_coordinate_labels(self):
        theta_true, measure, covariance = _manifold_cluster_setup()
        result = cluster_bootstrap(
            model=_manifold_model,
            theta_init=theta_true,
            measure=measure,
            covariance=covariance,
            n_boot=2,
            key=jax.random.PRNGKey(1),
        )
        names = result.param_names
        assert len(names) == _AMBIENT_P
        # PSDFixedRank factor entries get positional ambient labels ...
        assert names[0] == "A[0,0]"
        assert "A[3,1]" in names
        # ... and the Euclidean leaf contributes one labelled entry.
        assert names[-1] == "phi[0]"
        # Identical to the manifold-aware flatten Sigma_theta / eigenvalue_se
        # use -- the labelling contract the issue asked for.
        from emu_gmm._internal import labels as labels_mod
        from emu_gmm._internal import params as params_mod

        _, _, spec = params_mod.flatten_params_for_ad(theta_true)
        assert names == tuple(labels_mod.tangent_basis_names(spec))

    def test_replicates_produce_finite_ambient_estimates(self):
        theta_true, measure, covariance = _manifold_cluster_setup(seed=3)
        result = cluster_bootstrap(
            model=_manifold_model,
            theta_init=theta_true,
            measure=measure,
            covariance=covariance,
            n_boot=4,
            key=jax.random.PRNGKey(2),
        )
        boot = np.asarray(result.theta_boot.array)
        # Every replicate completes (no documented-divergence NaN rows) and
        # yields a finite ambient estimate -- the actual deliverable.
        #
        # We deliberately do NOT gate on ``result.convergence``: the
        # Riemannian LM does not always *certify* convergence on the
        # gauge-invariant quotient at the default tolerance (a direct
        # full-sample solve on this same fixture also reports
        # ``converged=False`` while returning a finite, sensible estimate,
        # J ~ 6 with well-separated Gamma eigenvalues). That certification
        # behaviour is a manifold-LM property orthogonal to #150; here we
        # only assert the bootstrap returns usable finite draws.
        assert boot.shape == (4, _AMBIENT_P)
        assert np.all(np.isfinite(boot))


@jdc.pytree_dataclass
class _PhiRidgePenalty:
    """Manifold-aware in-objective ridge on the Euclidean ``phi`` leaf.

    Stand-in for the consumer's ``CSlopePenalty`` (#150): a penalty that
    reads a *specific* leaf of a manifold parameter tree. The bundled
    :class:`~emu_gmm.penalty.TikhonovPenalty` cannot, because it routes
    through the v1 scalar-only ``flatten_params`` -- so a manifold + penalty
    use case must supply its own ``PenaltyStrategy``, which the package's
    ``estimate()`` penalty path already accepts.
    """

    c: Float[Array, ""]

    def penalty(self, theta) -> Float[Array, ""]:
        return jnp.asarray(self.c) * jnp.sum(theta.phi.array**2)

    def gradient(self, theta):
        return jax.grad(self.penalty)(theta)


@pytest.mark.slow
class TestManifoldWithPenalty:
    """The motivating #150 case: a manifold factor AND an in-objective
    penalty at once. Both fixes must compose.
    """

    def test_manifold_and_penalty_compose(self):
        from emu_gmm._internal import params as params_mod

        theta_true, measure, covariance = _manifold_cluster_setup(seed=5)
        penalty = _PhiRidgePenalty(c=jnp.asarray(5.0))
        key = jax.random.PRNGKey(99)

        boot = cluster_bootstrap(
            model=_manifold_model,
            theta_init=theta_true,
            measure=measure,
            covariance=covariance,
            n_boot=1,
            key=key,
            penalty=penalty,
        )
        boot0 = np.asarray(boot.theta_boot.array)[0]

        res_pen = _reconstruct_replicate0(
            measure, covariance, theta_true, _manifold_model, key=key, penalty=penalty
        )
        recon_pen = np.asarray(params_mod.flatten_params_for_ad(res_pen.theta_hat)[0])
        res_nopen = _reconstruct_replicate0(
            measure, covariance, theta_true, _manifold_model, key=key, penalty=None
        )
        recon_nopen = np.asarray(
            params_mod.flatten_params_for_ad(res_nopen.theta_hat)[0]
        )

        # Both fixes compose: the ambient theta_boot is the penalized refit.
        np.testing.assert_allclose(boot0, recon_pen, rtol=1e-6, atol=1e-8)
        # The phi-ridge actually moved the phi coordinate (last ambient entry).
        assert abs(float(recon_pen[-1]) - float(recon_nopen[-1])) > 1e-3
