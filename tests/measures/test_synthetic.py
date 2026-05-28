"""Tests for emu_gmm.measures.synthetic."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest

from emu_gmm._internal import axes as axes_mod
from emu_gmm.measures.synthetic import SyntheticMeasure


@jdc.pytree_dataclass
class _LinearParams:
    a: float
    b: float


# A trivial sampler that ignores theta and returns scalars drawn from
# standard normal. The "observation" is a single (1,) array.
def _standard_normal_sampler(key, theta):
    return jax.random.normal(key, shape=(1000, 1))


def _linear_residual(x, theta):
    """psi(x, theta) = [theta.a + theta.b * x[0]]: a 1-moment model."""
    return jnp.array([theta.a + theta.b * x[0]])


# ---------------------------------------------------------------------------


class TestExpectation:
    def test_mean_of_standard_normal_residual(self):
        # E[a + b*x] = a (since E[x] = 0 under standard normal). With
        # N_SIM = 1000 the Monte Carlo error is ~ 0.03 per draw.
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=1000,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        m = meas.expectation(_linear_residual, theta)
        assert m.shape == (1,)
        # Should be close to a=0.5 within Monte Carlo error.
        assert float(m[0]) == pytest.approx(0.5, abs=0.15)

    def test_crn_reproducibility(self):
        """Same key + same theta -> identical results across calls."""
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(42),
            n_sim=500,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        m1 = meas.expectation(_linear_residual, theta)
        m2 = meas.expectation(_linear_residual, theta)
        assert jnp.allclose(m1, m2)

    def test_different_keys_differ(self):
        """Two measures with different keys should give different draws."""
        theta = _LinearParams(a=0.0, b=1.0)
        m_a = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        ).expectation(_linear_residual, theta)
        m_b = SyntheticMeasure(
            key=jax.random.PRNGKey(1),
            n_sim=200,
            sampler=_standard_normal_sampler,
        ).expectation(_linear_residual, theta)
        assert not jnp.allclose(m_a, m_b)

    def test_handles_namedarray_return(self):
        """psi may return a haliax NamedArray; expectation strips it."""
        Moments = axes_mod.moments_axis(1)

        def labelled_residual(x, theta):
            return ha.named(
                jnp.array([theta.a + theta.b * x[0]]),
                (Moments,),
            )

        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        m = meas.expectation(labelled_residual, theta)
        assert m.shape == (1,)
        # Plain array out (not a NamedArray).
        assert not isinstance(m, ha.NamedArray)


# ---------------------------------------------------------------------------


class TestJacobian:
    def test_shape(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_linear_residual, theta)
        assert G.shape == (1, 2)  # M=1, K=2

    def test_against_analytical(self):
        # For psi = [a + b*x] under standard-normal x:
        # E[psi] = a; d E[psi]/d a = 1, d E[psi]/d b = E[x] ≈ 0.
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=5000,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_linear_residual, theta)
        assert float(G[0, 0]) == pytest.approx(1.0, abs=1e-5)
        # d/db is E[x], which is ~0 under SN with N=5000 (MC error ~0.015).
        assert float(G[0, 1]) == pytest.approx(0.0, abs=0.05)

    def test_theta_dependent_sampler_jacobian(self):
        """When the sampler does depend on theta, the Jacobian includes
        the path through the sampler."""

        # Sampler: x = theta.a + standard_normal. Then E[psi(x, theta)]
        # for psi = [theta.b * x] = theta.b * theta.a.
        # Jacobian: d/da = theta.b; d/db = theta.a.

        def theta_dependent_sampler(key, theta):
            z = jax.random.normal(key, shape=(5000, 1))
            return z + theta.a  # broadcast

        def psi(x, theta):
            return jnp.array([theta.b * x[0]])

        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(7),
            n_sim=5000,
            sampler=theta_dependent_sampler,
        )
        theta = _LinearParams(a=0.3, b=1.5)
        G = meas.jacobian(psi, theta)
        # d/da = b = 1.5
        assert float(G[0, 0]) == pytest.approx(1.5, abs=1e-4)
        # d/db = a = 0.3 (plus MC noise from E[z] term, which has noise
        # because the sampler does depend on theta).
        assert float(G[0, 1]) == pytest.approx(0.3, abs=0.05)


# ---------------------------------------------------------------------------


class TestJitCompatibility:
    def test_expectation_jits(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.expectation(_linear_residual, t)

        eager = meas.expectation(_linear_residual, theta)
        jit_result = compute(meas, theta)
        assert jnp.allclose(eager, jit_result)

    def test_jacobian_jits(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.jacobian(_linear_residual, t)

        G_eager = meas.jacobian(_linear_residual, theta)
        G_jit = compute(meas, theta)
        assert jnp.allclose(G_eager, G_jit)


# ---------------------------------------------------------------------------


class TestPyTreeBehaviour:
    def test_is_pytree(self):
        meas = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )
        leaves, _ = jax.tree_util.tree_flatten(meas)
        # Only `key` is a traced leaf; n_sim and sampler are static.
        assert len(leaves) == 1
        assert leaves[0].shape == (2,)  # PRNGKey is (2,) uint32

    def test_static_fields_not_traced(self):
        """Confirm that two measures with the same key but different
        static n_sim or sampler trigger separate jit traces (correct)."""
        sm_a = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=100,
            sampler=_standard_normal_sampler,
        )
        sm_b = SyntheticMeasure(
            key=jax.random.PRNGKey(0),
            n_sim=200,
            sampler=_standard_normal_sampler,
        )

        # tree_flatten preserves static fields as auxiliary data;
        # different aux data implies different treedefs.
        _, def_a = jax.tree_util.tree_flatten(sm_a)
        _, def_b = jax.tree_util.tree_flatten(sm_b)
        assert def_a != def_b
