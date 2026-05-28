"""Tests for emu_gmm.types."""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import jax_dataclasses as jdc
import pandas as pd
import pytest

from emu_gmm import types as t
from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import labels as labels_mod


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


# ---------------------------------------------------------------------------
# Protocol structural-typing checks
# ---------------------------------------------------------------------------


class _StubMeasure:
    def expectation(self, psi, theta):
        return jnp.zeros(2)

    def jacobian(self, psi, theta):
        return jnp.zeros((2, 2))


class _StubCovariance:
    def covariance(self, psi, theta, measure):
        return jnp.eye(2)


class _StubWeighting:
    def whitening_residual(self, m, V, theta):
        return m


class _StubRegularization:
    def apply(self, V):
        return V, 0.0


class _StubOptimizer:
    def __call__(self, residual_fn, theta_init):
        return theta_init, t.OptimizerInfo(
            steps=0, status="converged", final_objective=0.0, backend="stub"
        )


class TestProtocols:
    def test_measure_protocol(self):
        assert isinstance(_StubMeasure(), t.Measure)

    def test_covariance_protocol(self):
        assert isinstance(_StubCovariance(), t.CovarianceStrategy)

    def test_weighting_protocol(self):
        assert isinstance(_StubWeighting(), t.WeightingStrategy)

    def test_regularization_protocol(self):
        assert isinstance(_StubRegularization(), t.RegularizationStrategy)

    def test_optimizer_protocol(self):
        assert isinstance(_StubOptimizer(), t.Optimizer)

    def test_non_implementation_fails(self):
        class _Empty:
            pass

        assert not isinstance(_Empty(), t.Measure)
        assert not isinstance(_Empty(), t.CovarianceStrategy)


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def _make_result() -> t.EstimationResult:
    """Build a synthetic EstimationResult for use across tests."""
    Params = axes_mod.params_axis(2)
    ParamsDual = axes_mod.params_dual_axis(2)
    Moments = axes_mod.moments_axis(3)
    MomentsDual = axes_mod.moments_dual_axis(3)

    sigma = labels_mod.label_matrix(
        jnp.array([[0.01, 0.001], [0.001, 0.02]]), Params, ParamsDual
    )
    v_x = labels_mod.label_matrix(
        jnp.eye(3) * 0.1, Moments, MomentsDual
    )
    n_j = labels_mod.label_vector(jnp.array([100.0, 100.0, 100.0]), Moments)
    m_res = labels_mod.label_vector(jnp.array([1e-4, -2e-4, 5e-5]), Moments)

    opt_info = t.OptimizerInfo(
        steps=12, status="converged", final_objective=1.3, backend="stub"
    )
    diagnostics = t.Diagnostics(
        tau_realised=0.001,
        kappa_V=1e3,
        binding_ridge=False,
        cholesky_pivot_min=0.05,
        final_objective=1.3,
        final_gradient_norm=1e-9,
        N_j=n_j,
        moment_residual=m_res,
        optimizer_info=opt_info,
    )

    lc = labels_mod.LabelContext(
        param_names=("beta", "gamma"),
        moment_names=("euler_a", "euler_b", "euler_c"),
        variable_names=("c_t", "c_tp1", "r"),
        obs_name="hh",
    )

    return t.EstimationResult(
        theta_hat=_EulerParams(beta=0.95, gamma=2.0),
        Sigma_theta=sigma,
        V_X=v_x,
        J_stat=1.3,
        J_dof=1,
        J_pvalue=0.25,
        converged=True,
        iterations=12,
        theta_init=_EulerParams(beta=0.9, gamma=1.5),
        measure=_StubMeasure(),
        covariance=_StubCovariance(),
        weighting=_StubWeighting(),
        regularization=_StubRegularization(),
        diagnostics=diagnostics,
        labels=lc,
    )


class TestDataclassConstruction:
    def test_optimizer_info(self):
        info = t.OptimizerInfo(
            steps=5, status="converged", final_objective=0.1, backend="optimistix"
        )
        assert info.steps == 5
        assert info.status == "converged"

    def test_diagnostics(self):
        Moments = axes_mod.moments_axis(2)
        n_j = labels_mod.label_vector(jnp.array([10.0, 10.0]), Moments)
        m_res = labels_mod.label_vector(jnp.array([0.0, 0.0]), Moments)
        info = t.OptimizerInfo(
            steps=1, status="converged", final_objective=0.0, backend="stub"
        )
        d = t.Diagnostics(
            tau_realised=0.0,
            kappa_V=1.0,
            binding_ridge=False,
            cholesky_pivot_min=1.0,
            final_objective=0.0,
            final_gradient_norm=0.0,
            N_j=n_j,
            moment_residual=m_res,
            optimizer_info=info,
        )
        assert d.binding_ridge is False
        assert d.optimizer_info is info

    def test_estimation_result(self):
        r = _make_result()
        assert isinstance(r, t.EstimationResult)
        assert r.theta_hat.beta == pytest.approx(0.95)
        assert r.converged is True
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        assert isinstance(r.V_X, ha.NamedArray)


# ---------------------------------------------------------------------------
# to_pandas materialisation
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_returns_expected_keys(self):
        r = _make_result()
        d = r.to_pandas()
        assert set(d.keys()) == {
            "Sigma_theta",
            "V_X",
            "N_j",
            "moment_residual",
            "summary",
        }

    def test_sigma_theta_dataframe(self):
        r = _make_result()
        sigma = r.to_pandas()["Sigma_theta"]
        assert isinstance(sigma, pd.DataFrame)
        assert list(sigma.index) == ["beta", "gamma"]
        assert list(sigma.columns) == ["beta", "gamma"]
        assert sigma.loc["beta", "beta"] == pytest.approx(0.01)
        assert sigma.loc["beta", "gamma"] == pytest.approx(0.001)

    def test_v_x_dataframe(self):
        r = _make_result()
        v = r.to_pandas()["V_X"]
        assert isinstance(v, pd.DataFrame)
        assert list(v.index) == ["euler_a", "euler_b", "euler_c"]
        assert v.loc["euler_a", "euler_a"] == pytest.approx(0.1)

    def test_n_j_series(self):
        r = _make_result()
        s = r.to_pandas()["N_j"]
        assert isinstance(s, pd.Series)
        assert list(s.index) == ["euler_a", "euler_b", "euler_c"]
        assert s["euler_a"] == pytest.approx(100.0)
        assert s.name == "N_j"

    def test_moment_residual_series(self):
        r = _make_result()
        s = r.to_pandas()["moment_residual"]
        assert isinstance(s, pd.Series)
        assert s["euler_a"] == pytest.approx(1e-4)

    def test_summary_contains_expected_fields(self):
        r = _make_result()
        s = r.to_pandas()["summary"]
        assert isinstance(s, pd.Series)
        assert "J_stat" in s.index
        assert "J_dof" in s.index
        assert "tau_realised" in s.index
        assert "kappa_V" in s.index
        assert s["J_stat"] == pytest.approx(1.3)
        assert s["converged"]
