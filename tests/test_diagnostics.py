"""Tests for emu_gmm.diagnostics."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import haliax as ha
import jax.numpy as jnp
import pytest
from emu_gmm._internal import axes as axes_mod
from emu_gmm.diagnostics import build_diagnostics, log_to_stdout
from emu_gmm.types import Diagnostics, OptimizerInfo


def _stub_optimizer_info() -> OptimizerInfo:
    return OptimizerInfo(
        steps=7, status="converged", final_objective=0.5, backend="stub"
    )


class TestBuildDiagnostics:
    def test_basic(self):
        Moments = axes_mod.moments_axis(3)
        d = build_diagnostics(
            tau_realised=0.001,
            kappa_V=1e3,
            binding_ridge=False,
            cholesky_pivot_min=0.05,
            final_objective=0.5,
            final_gradient_norm=1e-9,
            N_j_array=jnp.array([100.0, 100.0, 100.0]),
            moment_residual_array=jnp.array([1e-4, -2e-4, 5e-5]),
            moments_axis=Moments,
            optimizer_info=_stub_optimizer_info(),
        )
        assert isinstance(d, Diagnostics)
        # Scalar diagnostics are kept as 0-d JAX arrays so the estimator
        # traces under jit / vmap; ``float(...)`` recovers the scalar.
        assert float(d.tau_realised) == pytest.approx(0.001)
        assert float(d.kappa_V) == pytest.approx(1e3)
        assert bool(d.binding_ridge) is False
        assert float(d.cholesky_pivot_min) == pytest.approx(0.05)
        assert float(d.final_objective) == pytest.approx(0.5)

    def test_labelled_per_moment_fields(self):
        Moments = axes_mod.moments_axis(2)
        d = build_diagnostics(
            tau_realised=0.0,
            kappa_V=1.0,
            binding_ridge=False,
            cholesky_pivot_min=1.0,
            final_objective=0.0,
            final_gradient_norm=0.0,
            N_j_array=jnp.array([500.0, 500.0]),
            moment_residual_array=jnp.array([0.01, -0.02]),
            moments_axis=Moments,
            optimizer_info=_stub_optimizer_info(),
        )
        assert isinstance(d.N_j, ha.NamedArray)
        assert d.N_j.axes == (Moments,)
        assert jnp.allclose(d.N_j.array, jnp.array([500.0, 500.0]))
        assert isinstance(d.moment_residual, ha.NamedArray)
        assert jnp.allclose(d.moment_residual.array, jnp.array([0.01, -0.02]))

    def test_optimizer_info_passes_through(self):
        Moments = axes_mod.moments_axis(1)
        info = OptimizerInfo(
            steps=42, status="max_iterations", final_objective=1.5, backend="optimistix"
        )
        d = build_diagnostics(
            tau_realised=0.5,
            kappa_V=2e5,
            binding_ridge=True,
            cholesky_pivot_min=0.001,
            final_objective=1.5,
            final_gradient_norm=2e-3,
            N_j_array=jnp.array([10.0]),
            moment_residual_array=jnp.array([0.0]),
            moments_axis=Moments,
            optimizer_info=info,
        )
        assert d.optimizer_info is info
        assert d.optimizer_info.steps == 42
        assert d.optimizer_info.status == "max_iterations"

    def test_no_nans_in_finite_inputs(self):
        Moments = axes_mod.moments_axis(2)
        d = build_diagnostics(
            tau_realised=0.0,
            kappa_V=1.0,
            binding_ridge=False,
            cholesky_pivot_min=1.0,
            final_objective=0.0,
            final_gradient_norm=0.0,
            N_j_array=jnp.array([1.0, 1.0]),
            moment_residual_array=jnp.array([0.0, 0.0]),
            moments_axis=Moments,
            optimizer_info=_stub_optimizer_info(),
        )
        assert not jnp.isnan(d.N_j.array).any()
        assert not jnp.isnan(d.moment_residual.array).any()


class TestLogToStdout:
    def test_emits_expected_format(self):
        logger = log_to_stdout()
        buf = io.StringIO()
        with redirect_stdout(buf):
            logger(step=3, tau=1.5e-3, kappa=1.2e4, objective=0.0123)
        out = buf.getvalue()
        assert "[emu-gmm]" in out
        assert "step=   3" in out
        assert "tau=" in out
        assert "kappa=" in out
        assert "Q=" in out

    def test_custom_prefix(self):
        logger = log_to_stdout(prefix="<run-A>")
        buf = io.StringIO()
        with redirect_stdout(buf):
            logger(step=0, tau=0.0, kappa=1.0, objective=0.0)
        assert "<run-A>" in buf.getvalue()
