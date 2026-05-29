"""Tests for ``SyntheticMeasure.moments_and_contributions``.

The shared SMM primitive co-computes the per-coordinate sample mean
and the per-draw ``(n_sim, M)`` residual matrix that
:class:`~emu_gmm.covariance.synthetic.SyntheticCovariance` independently
rebuilds when called separately. CRN guarantees identity across the
two return values; the cache path on the covariance reuses ``psi_batch``
directly.

See ``docs/reviews/v1x-performance-review.org`` finding #5 for the
motivating measurement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from emu_gmm.covariance.synthetic import SyntheticCovariance
from emu_gmm.measures.synthetic import SyntheticMeasure


@jdc.pytree_dataclass
class _Params:
    a: float
    b: float


def _sampler(key, theta):
    """Sample (n_sim, 2) draws from a standard normal."""
    return jax.random.normal(key, shape=(200, 2))


def _two_moment_psi(x, theta):
    """psi(x, theta) = [a + x[0], b * x[1]]."""
    return jnp.array([theta.a + x[0], theta.b * x[1]])


def _make_measure(n_sim: int = 200) -> SyntheticMeasure:
    def _sampler_n(key, theta, _n=n_sim):
        return jax.random.normal(key, shape=(_n, 2))

    return SyntheticMeasure(
        key=jax.random.PRNGKey(7),
        n_sim=n_sim,
        sampler=_sampler_n,
    )


class TestExpectationEquivalence:
    """moments_and_contributions()[0] == expectation()."""

    def test_matches_expectation(self):
        meas = _make_measure(200)
        theta = _Params(a=0.3, b=1.7)
        m_old = meas.expectation(_two_moment_psi, theta)
        m_new, _psi = meas.moments_and_contributions(_two_moment_psi, theta)
        assert jnp.allclose(m_old, m_new, atol=1e-12)

    def test_psi_batch_mean_matches_m(self):
        """m == mean(psi_batch) exactly under CRN."""
        meas = _make_measure(150)
        theta = _Params(a=-0.4, b=2.0)
        m, psi_batch = meas.moments_and_contributions(_two_moment_psi, theta)
        assert psi_batch.shape == (150, 2)
        assert jnp.allclose(m, jnp.mean(psi_batch, axis=0), atol=1e-12)

    def test_dtypes(self):
        meas = _make_measure(64)
        theta = _Params(a=0.0, b=1.0)
        m, psi_batch = meas.moments_and_contributions(_two_moment_psi, theta)
        assert jnp.issubdtype(m.dtype, jnp.floating)
        assert m.dtype == psi_batch.dtype


class TestCachedIntermediatesSynthetic:
    """SyntheticCovariance.covariance(...) is invariant under the cache path."""

    def test_cached_matches_uncached(self):
        meas = _make_measure(200)
        theta = _Params(a=0.4, b=-1.1)
        cov = SyntheticCovariance()
        cached = meas.moments_and_contributions(_two_moment_psi, theta)
        V_uncached = cov.covariance(_two_moment_psi, theta, meas)
        V_cached = cov.covariance(
            _two_moment_psi, theta, meas, cached_intermediates=cached
        )
        assert jnp.allclose(V_uncached, V_cached, atol=1e-12)

    def test_cached_matches_uncached_varied_params(self):
        meas = _make_measure(128)
        cov = SyntheticCovariance()
        for theta in (
            _Params(a=0.0, b=1.0),
            _Params(a=1.5, b=-0.5),
            _Params(a=-2.0, b=3.0),
        ):
            cached = meas.moments_and_contributions(_two_moment_psi, theta)
            V_uncached = cov.covariance(_two_moment_psi, theta, meas)
            V_cached = cov.covariance(
                _two_moment_psi, theta, meas, cached_intermediates=cached
            )
            assert jnp.allclose(V_uncached, V_cached, atol=1e-12)
