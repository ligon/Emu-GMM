"""Tests for emu_gmm.covariance.synthetic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest

from emu_gmm.covariance.synthetic import SyntheticCovariance
from emu_gmm.measures.synthetic import SyntheticMeasure


@jdc.pytree_dataclass
class _P:
    a: float
    b: float


def _identity_sampler(key, theta):
    """Return n_sim standard normal observations of dimension 2."""
    return jax.random.normal(key, shape=(5000, 2))


def _identity_psi(x, theta):
    """psi(x, theta) = x: 2-moment model returning the observation."""
    return x


# ---------------------------------------------------------------------------


class TestCovariance:
    def test_shape(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=5000, sampler=_identity_sampler
        )
        cov = SyntheticCovariance()
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert V.shape == (2, 2)

    def test_symmetric(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=5000, sampler=_identity_sampler
        )
        cov = SyntheticCovariance()
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert jnp.allclose(V, V.T, atol=1e-7)

    def test_psd(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=5000, sampler=_identity_sampler
        )
        cov = SyntheticCovariance()
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        eigs = jnp.linalg.eigvalsh(V)
        assert float(jnp.min(eigs)) >= -1e-10  # numerical tolerance

    def test_scaling_matches_variance_of_mean(self):
        """For psi(x, theta) = x with x ~ N(0, I), Var(mean) = (1/n) I.
        SyntheticCovariance should match within Monte Carlo error.
        """
        n_sim = 5000
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=_identity_sampler
        )
        cov = SyntheticCovariance()
        V = cov.covariance(_identity_psi, _P(0.0, 0.0), meas)
        # Expected diagonal: 1/n_sim = 2e-4.
        expected_diag = 1.0 / n_sim
        for i in range(2):
            assert float(V[i, i]) == pytest.approx(expected_diag, rel=0.1)
        # Off-diagonal: 0.
        assert float(V[0, 1]) == pytest.approx(0.0, abs=expected_diag * 0.5)

    def test_larger_n_gives_smaller_V(self):
        """Var(mean) scales as 1/n; doubling n should roughly halve V."""

        def small_sampler(key, theta):
            return jax.random.normal(key, shape=(1000, 2))

        def large_sampler(key, theta):
            return jax.random.normal(key, shape=(4000, 2))

        small = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=1000, sampler=small_sampler
        )
        large = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=4000, sampler=large_sampler
        )
        cov = SyntheticCovariance()
        V_small = cov.covariance(_identity_psi, _P(0.0, 0.0), small)
        V_large = cov.covariance(_identity_psi, _P(0.0, 0.0), large)
        # Ratio of diagonals should be ~4 (within MC error).
        for i in range(2):
            ratio = float(V_small[i, i]) / float(V_large[i, i])
            assert ratio == pytest.approx(4.0, rel=0.4)


# ---------------------------------------------------------------------------


class TestJitCompatibility:
    def test_covariance_jits(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=1000, sampler=_identity_sampler
        )
        cov = SyntheticCovariance()
        theta = _P(0.0, 0.0)

        @jax.jit
        def compute(c, m, t):
            return c.covariance(_identity_psi, t, m)

        V_eager = cov.covariance(_identity_psi, theta, meas)
        V_jit = compute(cov, meas, theta)
        assert jnp.allclose(V_eager, V_jit, atol=1e-7)
