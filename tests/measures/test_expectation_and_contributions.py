"""Tests for ``EmpiricalMeasure.expectation_and_contributions``.

The shared primitive co-computes the per-coordinate mean and the
intermediates (``psi_safe``, ``weight_mask``, ``N_j``) that
:class:`~emu_gmm.covariance.iid.IIDCovariance` and
:class:`~emu_gmm.covariance.clustered.ClusteredCovariance` independently
rebuild on every ``residual_fn`` call. These tests pin the contract that

* :meth:`expectation` is exactly the first element of the new return
  tuple (no algebraic drift across the two callers).
* Feeding the cached tuple to a covariance strategy as
  ``cached_intermediates`` produces the same ``V`` as the
  self-computing fall-through path (back-compat for third-party
  callers).
* The shapes and dtypes of the cached payload match what the
  downstream covariance strategies and the diagnostics layer expect.

See ``docs/reviews/v1x-performance-review.org`` finding #4 for the
motivating measurement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure


@jdc.pytree_dataclass
class _P:
    a: float
    b: float


def _two_moment_psi(x, theta):
    return jnp.array([theta.a + x[0], theta.b * x[1]])


def _identity_psi(x, theta):
    return x


def _make_measure(N: int, D: int, *, masked: bool = False, weighted: bool = False):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (N, D))
    if masked:
        # Roughly half the cells in moment 1 dropped, moment 0 fully observed.
        mask_col0 = jnp.ones(N)
        mask_col1 = jnp.where(jnp.arange(N) % 2 == 0, 1.0, 0.0)
        mask = jnp.stack([mask_col0, mask_col1], axis=1)
    else:
        mask = jnp.ones((N, D))
    if weighted:
        weights = 0.5 + jnp.arange(N, dtype=jnp.float64) / N
    else:
        weights = jnp.ones(N)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


class TestExpectationEquivalence:
    """expectation_and_contributions()[0] == expectation()."""

    def test_full_mask_uniform_weights(self):
        meas = _make_measure(50, 2)
        theta = _P(a=0.3, b=1.7)
        m_old = meas.expectation(_two_moment_psi, theta)
        m_new, _, _, _ = meas.expectation_and_contributions(_two_moment_psi, theta)
        assert jnp.allclose(m_old, m_new, atol=1e-12)

    def test_masked_weighted(self):
        meas = _make_measure(40, 2, masked=True, weighted=True)
        theta = _P(a=-0.2, b=2.5)
        m_old = meas.expectation(_two_moment_psi, theta)
        m_new, _, _, _ = meas.expectation_and_contributions(_two_moment_psi, theta)
        assert jnp.allclose(m_old, m_new, atol=1e-12)

    def test_shapes_and_dtypes(self):
        meas = _make_measure(20, 2, masked=True)
        theta = _P(a=0.0, b=1.0)
        m, psi_safe, weight_mask, N_j = meas.expectation_and_contributions(
            _two_moment_psi, theta
        )
        assert m.shape == (2,)
        assert psi_safe.shape == (20, 2)
        assert weight_mask.shape == (20, 2)
        assert N_j.shape == (2,)
        # All floating-point; respect the measure dtype.
        assert m.dtype == psi_safe.dtype == weight_mask.dtype == N_j.dtype
        assert jnp.issubdtype(m.dtype, jnp.floating)


class TestCachedIntermediatesIID:
    """IIDCovariance.covariance(...) is invariant under the cache path."""

    @pytest.mark.parametrize(
        "masked,weighted",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_iid_cached_matches_uncached(self, masked, weighted):
        meas = _make_measure(60, 2, masked=masked, weighted=weighted)
        theta = _P(a=0.4, b=-1.1)
        cached = meas.expectation_and_contributions(_two_moment_psi, theta)
        cov = IIDCovariance()
        V_uncached = cov.covariance(_two_moment_psi, theta, meas)
        V_cached = cov.covariance(
            _two_moment_psi, theta, meas, cached_intermediates=cached
        )
        assert jnp.allclose(V_uncached, V_cached, atol=1e-12)


class TestCachedIntermediatesClustered:
    """ClusteredCovariance.covariance(...) is invariant under the cache path."""

    @pytest.mark.parametrize(
        "masked,weighted",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_clustered_cached_matches_uncached(self, masked, weighted):
        N = 24
        meas = _make_measure(N, 2, masked=masked, weighted=weighted)
        theta = _P(a=0.1, b=0.9)
        # Three clusters of size 8 each.
        cluster_ids = (jnp.arange(N, dtype=jnp.float32) // 8).astype(jnp.float32)
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=3)
        cached = meas.expectation_and_contributions(_two_moment_psi, theta)
        V_uncached = cov.covariance(_two_moment_psi, theta, meas)
        V_cached = cov.covariance(
            _two_moment_psi, theta, meas, cached_intermediates=cached
        )
        assert jnp.allclose(V_uncached, V_cached, atol=1e-12)

    def test_clustered_singletons_match_iid_under_cache(self):
        """IID and singleton-cluster paths agree under the cache too."""
        N = 12
        meas = _make_measure(N, 3, masked=False, weighted=True)
        theta = _P(a=0.0, b=0.0)
        cached = meas.expectation_and_contributions(_identity_psi, theta)
        iid_cov = IIDCovariance()
        clu_cov = ClusteredCovariance(
            cluster_ids=jnp.arange(N, dtype=jnp.float32), n_clusters=N
        )
        V_iid = iid_cov.covariance(
            _identity_psi, theta, meas, cached_intermediates=cached
        )
        V_clu = clu_cov.covariance(
            _identity_psi, theta, meas, cached_intermediates=cached
        )
        assert jnp.allclose(V_iid, V_clu, atol=1e-10)
