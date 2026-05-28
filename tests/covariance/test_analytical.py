"""Tests for emu_gmm.covariance.analytical."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal import axes as axes_mod
from emu_gmm.covariance.analytical import AnalyticalCovariance
from emu_gmm.types import CovarianceStrategy


@jdc.pytree_dataclass
class _P:
    a: float
    b: float


def _constant_covariance(model, theta):
    """V = I_2 regardless of theta."""
    del model, theta
    return jnp.eye(2)


def _theta_dependent_covariance(model, theta):
    """V(theta) = diag(theta.a**2, theta.b**2)."""
    del model
    return jnp.diag(jnp.array([theta.a**2, theta.b**2]))


def _dummy_psi(x, theta):
    del x, theta
    return jnp.array([0.0, 0.0])


# ---------------------------------------------------------------------------


class TestCovariance:
    def test_satisfies_covariance_strategy_protocol(self):
        cov = AnalyticalCovariance(covariance_fn=_constant_covariance)
        assert isinstance(cov, CovarianceStrategy)

    def test_constant_covariance(self):
        cov = AnalyticalCovariance(covariance_fn=_constant_covariance)
        V = cov.covariance(_dummy_psi, _P(1.0, 2.0), measure=None)
        assert V.shape == (2, 2)
        assert jnp.allclose(V, jnp.eye(2))

    def test_theta_dependent_covariance(self):
        cov = AnalyticalCovariance(covariance_fn=_theta_dependent_covariance)
        V = cov.covariance(_dummy_psi, _P(1.5, 2.5), measure=None)
        assert V.shape == (2, 2)
        # Diagonal: (1.5**2, 2.5**2) = (2.25, 6.25)
        assert float(V[0, 0]) == pytest.approx(2.25)
        assert float(V[1, 1]) == pytest.approx(6.25)
        # Off-diagonal: 0
        assert float(V[0, 1]) == pytest.approx(0.0)
        assert float(V[1, 0]) == pytest.approx(0.0)

    def test_measure_argument_ignored(self):
        """The measure argument is accepted but not used."""
        cov = AnalyticalCovariance(covariance_fn=_constant_covariance)
        # An arbitrary value for measure should not affect the result.
        V_none = cov.covariance(_dummy_psi, _P(0.0, 0.0), measure=None)
        V_str = cov.covariance(_dummy_psi, _P(0.0, 0.0), measure="anything")
        assert jnp.allclose(V_none, V_str)

    def test_handles_namedarray_return(self):
        """covariance_fn may return a haliax NamedArray; covariance strips it."""
        Moments = axes_mod.moments_axis(2)
        MomentsDual = axes_mod.moments_dual_axis(2)

        def labelled_covariance(model, theta):
            del model, theta
            return ha.named(jnp.eye(2), (Moments, MomentsDual))

        cov = AnalyticalCovariance(covariance_fn=labelled_covariance)
        V = cov.covariance(_dummy_psi, _P(0.0, 0.0), measure=None)
        assert V.shape == (2, 2)
        assert not isinstance(V, ha.NamedArray)
        assert jnp.allclose(V, jnp.eye(2))


# ---------------------------------------------------------------------------


class TestJitCompatibility:
    def test_covariance_jits(self):
        cov = AnalyticalCovariance(covariance_fn=_theta_dependent_covariance)
        theta = _P(1.5, 2.5)

        @jax.jit
        def compute(c, t):
            return c.covariance(_dummy_psi, t, measure=None)

        V_eager = cov.covariance(_dummy_psi, theta, measure=None)
        V_jit = compute(cov, theta)
        assert jnp.allclose(V_eager, V_jit, atol=1e-7)


# ---------------------------------------------------------------------------


class TestPyTreeBehaviour:
    def test_is_pytree(self):
        cov = AnalyticalCovariance(covariance_fn=_constant_covariance)
        leaves, _ = jax.tree_util.tree_flatten(cov)
        # covariance_fn is static; no traced leaves.
        assert len(leaves) == 0
