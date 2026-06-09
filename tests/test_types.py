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
    v_x = labels_mod.label_matrix(jnp.eye(3) * 0.1, Moments, MomentsDual)
    n_j = labels_mod.label_vector(jnp.array([100.0, 100.0, 100.0]), Moments)
    m_res = labels_mod.label_vector(jnp.array([1e-4, -2e-4, 5e-5]), Moments)

    opt_info = t.OptimizerInfo(
        steps=12, status="converged", final_objective=1.3, backend="stub"
    )
    diagnostics = t.Diagnostics(
        tau_realised=jnp.asarray(0.001),
        kappa_V=jnp.asarray(1e3),
        binding_ridge=jnp.asarray(False),
        cholesky_pivot_min=jnp.asarray(0.05),
        final_objective=jnp.asarray(1.3),
        final_gradient_norm=jnp.asarray(1e-9),
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
        J_stat=jnp.asarray(1.3),
        J_dof=1,
        J_pvalue=jnp.asarray(0.25),
        J_pvalue_adjusted=jnp.asarray(0.25),
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
            "coefficients",
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


# ---------------------------------------------------------------------------
# Standard errors + coefficient table (Fix 3, v1.x API safety)
# ---------------------------------------------------------------------------


class TestStandardErrorsAndCoefTable:
    def test_standard_errors_is_named_array(self):
        r = _make_result()
        se = r.standard_errors
        assert isinstance(se, ha.NamedArray)
        assert se.axes[0].name == "parameters"
        assert se.axes[0].size == 2

    def test_standard_errors_match_sqrt_diag(self):
        """``standard_errors`` is exactly ``sqrt(diag(Sigma_theta))``.

        This is the property users currently hand-roll
        (``jnp.sqrt(jnp.diag(Sigma_theta.values))``); the API exposes
        it directly.
        """
        r = _make_result()
        sigma_arr = jnp.asarray(r.Sigma_theta.array)
        expected = jnp.sqrt(jnp.diag(sigma_arr))
        actual = jnp.asarray(r.standard_errors.array)
        assert jnp.allclose(actual, expected)

    def test_standard_errors_cached(self):
        """Repeated access returns the same object (functools cache)."""
        r = _make_result()
        a = r.standard_errors
        b = r.standard_errors
        assert a is b

    def test_coef_table_is_dataframe_with_four_columns(self):
        r = _make_result()
        tbl = r.coef_table
        assert isinstance(tbl, pd.DataFrame)
        assert list(tbl.columns) == ["estimate", "std_error", "t_stat", "p_value"]
        # Index is the param names.
        assert list(tbl.index) == ["beta", "gamma"]

    def test_coef_table_populates_all_columns(self):
        """For the EulerParams stub the diagonal of Sigma_theta is
        positive and theta_hat is finite, so every cell in the table
        should be finite (no NaN).
        """
        r = _make_result()
        tbl = r.coef_table
        assert tbl.notna().all().all()

    def test_coef_table_estimate_column_matches_theta_hat(self):
        """``estimate`` column equals the flattened ``theta_hat``."""
        r = _make_result()
        tbl = r.coef_table
        # _EulerParams has beta=0.95, gamma=2.0 in PyTree-traversal order
        # (alphabetical for jdc.pytree_dataclass-on-jax-tree-utils?). Use
        # the param-name index to read deterministically.
        assert tbl.loc["beta", "estimate"] == pytest.approx(0.95)
        assert tbl.loc["gamma", "estimate"] == pytest.approx(2.0)

    def test_coef_table_std_error_matches_sqrt_diag(self):
        r = _make_result()
        tbl = r.coef_table
        sigma_arr = jnp.asarray(r.Sigma_theta.array)
        expected = jnp.sqrt(jnp.diag(sigma_arr))
        assert tbl["std_error"].to_numpy()[0] == pytest.approx(float(expected[0]))
        assert tbl["std_error"].to_numpy()[1] == pytest.approx(float(expected[1]))

    def test_coef_table_t_stat_is_estimate_over_std_error(self):
        r = _make_result()
        tbl = r.coef_table
        ratio = tbl["estimate"] / tbl["std_error"]
        assert tbl["t_stat"].to_numpy()[0] == pytest.approx(ratio.to_numpy()[0])
        assert tbl["t_stat"].to_numpy()[1] == pytest.approx(ratio.to_numpy()[1])

    def test_coef_table_p_value_two_sided_normal(self):
        """``p_value`` is the two-sided normal-reference p-value."""
        import scipy.stats

        r = _make_result()
        tbl = r.coef_table
        for name in ("beta", "gamma"):
            t_val = float(tbl.loc[name, "t_stat"])
            expected = 2.0 * scipy.stats.norm.sf(abs(t_val))
            assert float(tbl.loc[name, "p_value"]) == pytest.approx(expected)

    def test_to_pandas_exposes_coef_table_as_coefficients(self):
        """``to_pandas()['coefficients']`` returns the coefficient
        table; this is the round-trip the acceptance criterion calls
        out.
        """
        r = _make_result()
        d = r.to_pandas()
        assert "coefficients" in d
        coeffs = d["coefficients"]
        assert isinstance(coeffs, pd.DataFrame)
        assert list(coeffs.columns) == ["estimate", "std_error", "t_stat", "p_value"]
        # All four columns populated (no NaN under the stub fixture).
        assert coeffs.notna().all().all()
        # Same object as the cached_property (round-trip identity).
        assert coeffs is r.coef_table


# ---------------------------------------------------------------------------
# LabelContext public re-export (#56)
# ---------------------------------------------------------------------------


class TestLabelContextPublicReexport:
    """``LabelContext`` annotates ``EstimationResult.labels`` (line 359 of
    ``types.py``) but its definition lived only in
    ``emu_gmm._internal.labels``, which the project conventions treat as
    off-limits. Re-export it through ``emu_gmm.types`` and the top-level
    package so users can write the import the result-object's signature
    suggests.
    """

    def test_import_from_top_level_package(self):
        """``from emu_gmm import LabelContext`` works."""
        from emu_gmm import LabelContext as _Top
        from emu_gmm._internal.labels import LabelContext as _Internal

        # Re-export is the *same* class, not a copy: ``isinstance``
        # against either name resolves to the same type, and the
        # internal definition is the source of truth.
        assert _Top is _Internal

    def test_import_from_types_module(self):
        """``from emu_gmm.types import LabelContext`` works."""
        from emu_gmm._internal.labels import LabelContext as _Internal
        from emu_gmm.types import LabelContext as _Types

        assert _Types is _Internal

    def test_label_context_in_types_all(self):
        """``LabelContext`` appears in ``emu_gmm.types.__all__``."""
        assert "LabelContext" in t.__all__

    def test_label_context_in_package_all(self):
        """``LabelContext`` appears in the top-level ``__all__``."""
        import emu_gmm

        assert "LabelContext" in emu_gmm.__all__

    def test_result_labels_type_resolves_to_public_path(self):
        """``type(result.labels)`` is the re-exported class.

        The re-exported name is an alias for the internal class, so
        ``type(result.labels)`` still reports the original
        ``__qualname__`` (Python attaches qualnames at class-definition
        time, not at re-export). The contract for users is the
        identity check above --- ``emu_gmm.LabelContext is
        type(result.labels)`` --- which lets ``help(result.labels)``
        and ``isinstance(result.labels, emu_gmm.LabelContext)`` work
        from the public path without spelunking through
        ``emu_gmm._internal``.
        """
        import emu_gmm

        r = _make_result()
        assert type(r.labels) is emu_gmm.LabelContext


# ---------------------------------------------------------------------------
# Pickle convenience (#23)
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _RelabelableParams:
    """Module-level params class whose __module__ a test temporarily
    rewrites to '__main__' to exercise the portability warning."""

    a: float


class TestPickleConvenience:
    def test_round_trip(self, tmp_path):
        r = _make_result()
        path = tmp_path / "result.pkl"
        r.to_pickle(path)
        r2 = t.EstimationResult.from_pickle(path)
        assert isinstance(r2, t.EstimationResult)
        assert float(r2.theta_hat.beta) == float(r.theta_hat.beta)
        assert float(r2.theta_hat.gamma) == float(r.theta_hat.gamma)
        assert float(r2.J_stat) == float(r.J_stat)
        assert r2.J_dof == r.J_dof
        import numpy as np

        np.testing.assert_array_equal(
            np.asarray(r2.Sigma_theta.array), np.asarray(r.Sigma_theta.array)
        )
        assert r2.labels.param_names == r.labels.param_names

    def test_from_pickle_rejects_wrong_payload(self, tmp_path):
        import pickle

        import pytest

        path = tmp_path / "notaresult.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"theta": [1.0, 2.0]}, fh)
        with pytest.raises(TypeError, match="not an EstimationResult"):
            t.EstimationResult.from_pickle(path)

    def test_main_namespace_warns_at_save_time(self, tmp_path):
        """A __main__-namespaced parameter class triggers the portability
        warning at save time (when it is still fixable), not a load-time
        AttributeError in another process (#23)."""
        import dataclasses
        import sys

        import pytest

        main_mod = sys.modules["__main__"]
        orig_module = _RelabelableParams.__module__
        try:
            _RelabelableParams.__module__ = "__main__"
            # Register on __main__ so the pickle itself still resolves
            # (the K-Aggregators shim pattern); the warning is the point.
            main_mod._RelabelableParams = _RelabelableParams
            r = dataclasses.replace(
                _make_result(),
                theta_hat=_RelabelableParams(a=1.0),
                theta_init=_RelabelableParams(a=0.5),
            )
            path = tmp_path / "main_namespaced.pkl"
            with pytest.warns(UserWarning, match="__main__"):
                r.to_pickle(path)
            # And it still loads in THIS process (shim present).
            r2 = t.EstimationResult.from_pickle(path)
            assert float(r2.theta_hat.a) == 1.0
        finally:
            _RelabelableParams.__module__ = orig_module
            if hasattr(main_mod, "_RelabelableParams"):
                delattr(main_mod, "_RelabelableParams")

    def test_importable_classes_do_not_warn(self, tmp_path):
        import warnings

        r = _make_result()
        path = tmp_path / "clean.pkl"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r.to_pickle(path)
