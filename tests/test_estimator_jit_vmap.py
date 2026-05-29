"""``estimate()`` is jit / vmap compatible at the eager boundary.

The framework documents ``estimate()`` as jit and vmap friendly; this
was previously broken by six ``float(...)`` casts in the eager path
plus a ``scipy.stats.chi2.sf`` boundary that demanded concrete Python
values. The fix routes scalar diagnostics as 0-d JAX arrays and uses
``jax.scipy.stats.chi2.sf`` for the J-test p-value.

Reference: docs/reviews/v1x-api-design.org §1 [HIGH] finding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    DiagonalTikhonov,
    SyntheticCovariance,
    SyntheticMeasure,
    estimate,
    optimistix_lm,
)
from emu_gmm.examples.euler import EulerParams, euler_residual, euler_sampler_factory

# Smaller N_SIM than the acceptance test: jit-trace cost is the same
# regardless, and vmap multiplies the compute. 200 keeps the suite fast.
N_SIM = 200


def _build_inputs():
    """Build the (measure, covariance, model, optimiser, ...) tuple used
    in every test below."""
    sampler = euler_sampler_factory(N_SIM)
    measure = SyntheticMeasure(
        key=jax.random.PRNGKey(0),
        n_sim=N_SIM,
        sampler=sampler,
    )
    return {
        "model": euler_residual,
        "measure": measure,
        "covariance": SyntheticCovariance(),
        "weighting": ContinuouslyUpdated(),
        "regularization": DiagonalTikhonov(),
        "optimizer": optimistix_lm(),
    }


class TestEstimateJitCompatible:
    """``jax.jit(estimate)`` traces end-to-end without hitting any
    Python boundary on a traced value."""

    def test_jit_over_theta_init_returns_finite_J_stat(self):
        inputs = _build_inputs()

        def run(theta):
            r = estimate(**inputs, theta_init=theta)
            return r.J_stat

        theta = EulerParams(beta=0.9, gamma=1.0)
        eager = float(run(theta))
        jitted = float(jax.jit(run)(theta))
        assert jnp.isfinite(jitted)
        # Numerical agreement to working precision: jit shouldn't change
        # the answer, only the route.
        assert jitted == pytest.approx(eager, rel=1e-6, abs=1e-10)

    def test_jit_returns_traced_pvalue(self):
        """``J_pvalue`` is computed via ``jax.scipy.stats.chi2.sf``, so
        it traces; the result is a 0-d JAX array in [0, 1]."""
        inputs = _build_inputs()

        def run(theta):
            return estimate(**inputs, theta_init=theta).J_pvalue

        p = jax.jit(run)(EulerParams(beta=0.9, gamma=1.0))
        p_val = float(p)
        assert 0.0 <= p_val <= 1.0

    def test_jit_compatible_diagnostics(self):
        """All scalar diagnostic fields survive a jit boundary."""
        inputs = _build_inputs()

        def run(theta):
            r = estimate(**inputs, theta_init=theta)
            d = r.diagnostics
            return (
                d.tau_realised,
                d.kappa_V,
                d.cholesky_pivot_min,
                d.final_gradient_norm,
            )

        outputs = jax.jit(run)(EulerParams(beta=0.9, gamma=1.0))
        for x in outputs:
            assert jnp.isfinite(x)


class TestEstimateVmapCompatible:
    """``jax.vmap(estimate)`` over a batch of starting points works
    end-to-end and produces a leading batch dimension on every scalar
    output."""

    def test_vmap_over_theta_init_returns_batched_J_stat(self):
        inputs = _build_inputs()

        def run(theta):
            return estimate(**inputs, theta_init=theta).J_stat

        batch = EulerParams(
            beta=jnp.array([0.9, 0.95, 1.0]),
            gamma=jnp.array([1.0, 2.0, 2.5]),
        )
        J = jax.vmap(run)(batch)
        # Leading batch dimension equal to 3.
        assert J.shape == (3,)
        assert jnp.all(jnp.isfinite(J))

    def test_vmap_over_theta_init_batched_diagnostics(self):
        inputs = _build_inputs()

        def run(theta):
            r = estimate(**inputs, theta_init=theta)
            return r.diagnostics.tau_realised, r.diagnostics.kappa_V

        batch = EulerParams(
            beta=jnp.array([0.9, 0.95, 1.0]),
            gamma=jnp.array([1.0, 2.0, 2.5]),
        )
        tau, kappa = jax.vmap(run)(batch)
        assert tau.shape == (3,)
        assert kappa.shape == (3,)
        assert jnp.all(jnp.isfinite(kappa))

    def test_vmap_then_jit_composes(self):
        """``jit(vmap(estimate))`` traces and matches eager."""
        inputs = _build_inputs()

        def run(theta):
            return estimate(**inputs, theta_init=theta).J_stat

        batch = EulerParams(
            beta=jnp.array([0.9, 0.95, 1.0]),
            gamma=jnp.array([1.0, 2.0, 2.5]),
        )
        eager = jax.vmap(run)(batch)
        jitted = jax.jit(jax.vmap(run))(batch)
        assert jnp.allclose(eager, jitted, rtol=1e-6, atol=1e-10)
