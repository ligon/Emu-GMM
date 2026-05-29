"""Integration test for the ``MomentRestriction``-equivalent primitives.

Issue #2 / port-blocker. ManifoldGMM's ``MomentRestriction`` exposes
three primitives that bootstrap, K-statistic, and other resampling-based
inference routines reach into:

  - ``moment_contributions(theta)`` --- per-row ``g_i(theta)`` matrix
  - ``omega_hat(theta)`` --- empirical covariance of the moment estimator
  - ``jacobian(theta)`` --- average Jacobian ``D bar g_N(theta)``

This test checks that the ``EmpiricalMeasure`` + ``CovarianceStrategy``
combination exposes the equivalent surface in emu-gmm, so that
Seasonality (and similar) scripts can build their bootstrap / K-stat
machinery without reaching into framework internals.

The architectural commitment that measure and covariance stay
orthogonal (``CLAUDE.md`` point 1) means the primitives are surfaced as
three separate public calls rather than bundled on a restriction
object. This test demonstrates the composition pattern callers can
use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure


@jdc.pytree_dataclass
class _Params:
    a: float
    b: float


def _psi(x, theta):
    """A two-moment model: psi_0 = a + x[0], psi_1 = b * x[1]."""
    return jnp.array([theta.a + x[0], theta.b * x[1]])


# ---------------------------------------------------------------------------


class TestPrimitivesExposed:
    """The three primitives are public methods on the measure / covariance.

    No reach into private attributes; everything via the documented API.
    """

    def test_moment_contributions_is_public(self):
        meas = EmpiricalMeasure(
            x=jnp.zeros((4, 2)),
            mask=jnp.ones((4, 2)),
            weights=jnp.ones(4),
        )
        theta = _Params(a=0.0, b=1.0)
        g = meas.moment_contributions(_psi, theta)
        assert g.shape == (4, 2)

    def test_jacobian_is_public(self):
        meas = EmpiricalMeasure(
            x=jnp.ones((4, 2)),
            mask=jnp.ones((4, 2)),
            weights=jnp.ones(4),
        )
        theta = _Params(a=0.5, b=2.0)
        G = meas.jacobian(_psi, theta)
        # 2 moments, 2 params.
        assert G.shape == (2, 2)

    def test_omega_hat_is_public_via_covariance(self):
        N = 10
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        theta = _Params(a=0.0, b=1.0)
        omega = IIDCovariance().covariance(_psi, theta, meas)
        assert omega.shape == (2, 2)
        # PSD by construction.
        evs = jnp.linalg.eigvalsh(omega)
        assert float(jnp.min(evs)) >= -1e-10


# ---------------------------------------------------------------------------


class TestCompositionMatchesHandRolled:
    """The three public primitives compose into the same outputs an
    inline implementation would produce, with no hidden state.

    This is the parity contract Seasonality's bootstrap and K-stat code
    can rely on.
    """

    def test_moment_contributions_compose_to_expectation(self):
        """sum(g_i) / N_j == expectation()_j for each moment.

        This is the consistency relation between ``moment_contributions``
        and the aggregated ``expectation``: callers can write their own
        weighted / resampled aggregation and get the same answer for the
        full-sample average.
        """
        N = 50
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        theta = _Params(a=0.3, b=-1.0)
        g = meas.moment_contributions(_psi, theta)
        N_j = jnp.sum(meas.mask * meas.weights[:, None], axis=0)
        m_recomposed = jnp.sum(g, axis=0) / N_j
        m_direct = meas.expectation(_psi, theta)
        assert jnp.allclose(m_recomposed, m_direct, atol=1e-12)

    def test_omega_hat_iid_matches_hand_rolled_from_contributions(self):
        """IIDCovariance == (1 / (N_j N_k)) * sum_i g_ij g_ik.

        Reconstruct the iid covariance from ``moment_contributions``
        directly --- this is the formula Seasonality's bootstrap code
        builds for its resampled estimates.
        """
        N = 30
        key = jax.random.PRNGKey(11)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        theta = _Params(a=0.0, b=1.0)
        g = meas.moment_contributions(_psi, theta)  # (N, 2)
        N_j = jnp.sum(meas.mask * meas.weights[:, None], axis=0)
        # Hand-rolled iid: (1 / (N_j N_k)) sum_i g_ij g_ik.
        # contributions g already include weights w_i once; the iid
        # formula in IIDCovariance multiplies psi by w_i twice (w_i^2)
        # inside the einsum. So for unit weights the two coincide.
        omega_handrolled = (g.T @ g) / jnp.outer(N_j, N_j)
        omega_framework = IIDCovariance().covariance(_psi, theta, meas)
        assert jnp.allclose(omega_handrolled, omega_framework, atol=1e-12)

    def test_omega_hat_clustered_matches_hand_rolled_from_contributions(self):
        """ClusteredCovariance == (1 / (N_j N_k)) sum_c (sum_{i in c} g_ij)(sum_{i in c} g_ik).

        For unit weights, the cluster-totals form built directly from
        ``moment_contributions`` matches the framework's
        ``ClusteredCovariance.covariance``. Demonstrates that a
        bootstrap or K-stat routine can resample cluster IDs and
        recompute omega_hat correctly using only the public primitives.
        """
        N = 12
        # Three clusters of size 4 each.
        cluster_ids_np = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=float)
        n_clusters = 3
        key = jax.random.PRNGKey(13)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        theta = _Params(a=0.0, b=1.0)
        g = meas.moment_contributions(_psi, theta)  # (N, 2)
        N_j = jnp.sum(meas.mask * meas.weights[:, None], axis=0)

        # Hand-rolled cluster totals from public primitives.
        cluster_totals = np.zeros((n_clusters, 2))
        g_np = np.asarray(g)
        for c in range(n_clusters):
            cluster_totals[c] = g_np[cluster_ids_np == c].sum(axis=0)
        omega_handrolled = (cluster_totals.T @ cluster_totals) / np.outer(N_j, N_j)

        cov = ClusteredCovariance(
            cluster_ids=jnp.asarray(cluster_ids_np), n_clusters=n_clusters
        )
        omega_framework = cov.covariance(_psi, theta, meas)
        np.testing.assert_allclose(
            omega_handrolled, np.asarray(omega_framework), atol=1e-10
        )

    def test_jacobian_returns_average_gradient(self):
        """``jacobian(psi, theta)`` returns D bar g_N(theta), the average
        gradient of psi over the sample under the mask.
        """
        N = 20
        key = jax.random.PRNGKey(23)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        theta = _Params(a=0.5, b=2.0)
        G = meas.jacobian(_psi, theta)
        # For psi = [a + x[0], b * x[1]]:
        #   dG[0]/da = 1, dG[0]/db = 0
        #   dG[1]/da = 0, dG[1]/db = mean(x[:, 1])
        assert float(G[0, 0]) == 1.0
        assert float(G[0, 1]) == 0.0
        assert float(G[1, 0]) == 0.0
        assert float(G[1, 1]) == float(jnp.mean(x[:, 1]))


# ---------------------------------------------------------------------------


class TestJitFriendly:
    """All three primitives must be JIT-friendly for bootstrap loops.

    A resampling routine that JITs the inner computation needs every
    primitive to trace through cleanly.
    """

    def test_pipeline_jits(self):
        N = 16
        key = jax.random.PRNGKey(31)
        x = jax.random.normal(key, (N, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )

        @jax.jit
        def restriction_at(theta):
            g = meas.moment_contributions(_psi, theta)
            G = meas.jacobian(_psi, theta)
            V = IIDCovariance().covariance(_psi, theta, meas)
            return g, G, V

        theta = _Params(a=0.0, b=1.0)
        g, G, V = restriction_at(theta)
        # Sanity: shapes are right and outputs are finite.
        assert g.shape == (N, 2)
        assert G.shape == (2, 2)
        assert V.shape == (2, 2)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.all(jnp.isfinite(G)))
        assert bool(jnp.all(jnp.isfinite(V)))
