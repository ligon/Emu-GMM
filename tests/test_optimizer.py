"""Tests for emu_gmm.optimizer.

Both adapters are exercised on the classical Rosenbrock NLLS problem:

    residual(x) = [10 * (x[1] - x[0]**2), 1 - x[0]]

which has a unique minimum at ``x = (1, 1)`` where ``||residual||^2 = 0``.
The far-from-optimum start ``(-1.2, 1.0)`` is the textbook hard case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from emu_gmm import types as t
from emu_gmm.optimizer import optimistix_lm, scipy_lm


# ---------------------------------------------------------------------------


def _rosenbrock_residual(x: jnp.ndarray) -> jnp.ndarray:
    """NLLS Rosenbrock residual; ``||r||^2`` is the classical objective."""
    return jnp.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


_X_INIT = jnp.array([-1.2, 1.0])
_X_OPT = jnp.array([1.0, 1.0])


# ---------------------------------------------------------------------------


class TestOptimistixLM:
    def test_satisfies_protocol(self):
        opt = optimistix_lm()
        assert isinstance(opt, t.Optimizer)

    def test_solves_rosenbrock(self):
        opt = optimistix_lm()
        x_opt, info = opt(_rosenbrock_residual, _X_INIT)
        assert jnp.allclose(x_opt, _X_OPT, atol=1e-4)

    def test_info_reports_progress(self):
        opt = optimistix_lm()
        _, info = opt(_rosenbrock_residual, _X_INIT)
        assert int(info.steps) > 0
        assert float(info.final_objective) < 1e-8
        assert info.backend == "optimistix"
        assert info.status == "converged"

    def test_max_iterations_status(self):
        # max_steps=1 is far too few for Rosenbrock from (-1.2, 1.0);
        # optimistix should hit the cap and report it.
        opt = optimistix_lm(max_steps=1)
        _, info = opt(_rosenbrock_residual, _X_INIT)
        assert info.status == "max_iterations"

    def test_custom_tolerances(self):
        # Looser tolerances should still converge; sanity check.
        opt = optimistix_lm(rtol=1e-4, atol=1e-4)
        x_opt, info = opt(_rosenbrock_residual, _X_INIT)
        assert jnp.allclose(x_opt, _X_OPT, atol=1e-2)
        assert info.backend == "optimistix"

    def test_jit_compatibility(self):
        """Wrapping the call site in ``jax.jit`` succeeds."""
        opt = optimistix_lm()

        @jax.jit
        def run(x0):
            x_opt, info = opt(_rosenbrock_residual, x0)
            return x_opt, info.final_objective

        x_opt, final = run(_X_INIT)
        assert jnp.allclose(x_opt, _X_OPT, atol=1e-4)
        assert float(final) < 1e-8


# ---------------------------------------------------------------------------


class TestScipyLM:
    # NOTE: scipy_lm is *not* jit/vmap compatible because the optimiser
    # loop runs in interpreted Python inside SciPy. Use optimistix_lm()
    # when JIT-purity matters.

    def test_satisfies_protocol(self):
        opt = scipy_lm()
        assert isinstance(opt, t.Optimizer)

    def test_solves_rosenbrock(self):
        opt = scipy_lm()
        x_opt, info = opt(_rosenbrock_residual, _X_INIT)
        assert jnp.allclose(x_opt, _X_OPT, atol=1e-4)

    def test_info_reports_progress(self):
        opt = scipy_lm()
        _, info = opt(_rosenbrock_residual, _X_INIT)
        assert int(info.steps) > 0
        assert float(info.final_objective) < 1e-8
        assert info.backend == "scipy"
        assert info.status == "converged"

    def test_max_iterations_status(self):
        # max_nfev=2 forces SciPy LM to return status=0 (max iterations
        # exceeded) from a hostile start.
        opt = scipy_lm(max_nfev=2)
        _, info = opt(_rosenbrock_residual, jnp.array([-100.0, 100.0]))
        assert info.status == "max_iterations"

    def test_method_kwarg_rejected(self):
        """The adapter always uses method='lm'; passing it is an error."""
        with pytest.raises(ValueError):
            scipy_lm(method="trf")

    def test_returns_jax_arrays(self):
        """``theta_opt`` is a JAX array even though SciPy returns NumPy."""
        opt = scipy_lm()
        x_opt, _ = opt(_rosenbrock_residual, _X_INIT)
        assert isinstance(x_opt, jnp.ndarray)
