"""Tests for emu_gmm.diagnostics."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import haliax as ha
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal import axes as axes_mod
from emu_gmm.covariance import AnalyticalCovariance
from emu_gmm.diagnostics import (
    build_diagnostics,
    build_optimizer_health,
    compute_cond_info,
    log_to_stdout,
)
from emu_gmm.estimator import estimate
from emu_gmm.measures import AnalyticalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import Diagnostics, OptimizerInfo
from emu_gmm.weighting import ContinuouslyUpdated


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


# ---------------------------------------------------------------------------
# Hessian-condition trio (issue #10)
# ---------------------------------------------------------------------------


class TestComputeCondInfo:
    """Unit tests for :func:`compute_cond_info`.

    The function returns a dict with three keys (``raw`` / ``data_only``
    / ``exclude_gauge``). In v1 all three equal ``cond(G' V^{-1} G)``;
    once issue #7 (penalty hook) and the v2 manifold epic land, the
    latter two will diverge from ``raw``.
    """

    def test_returns_three_keys(self):
        G = jnp.eye(3, 2)
        V = jnp.eye(3)
        info = compute_cond_info(G, V)
        assert set(info.keys()) == {"raw", "data_only", "exclude_gauge"}

    def test_well_conditioned_identity(self):
        """With G = I and V = I, info_matrix = I, cond = 1."""
        G = jnp.eye(3)
        V = jnp.eye(3)
        info = compute_cond_info(G, V)
        assert info["raw"] == pytest.approx(1.0, abs=1e-10)
        assert info["data_only"] == pytest.approx(1.0, abs=1e-10)
        assert info["exclude_gauge"] == pytest.approx(1.0, abs=1e-10)

    def test_v1_aliases_coincide(self):
        """In v1, data_only and exclude_gauge alias to raw."""
        G = jnp.array([[1.0, 0.0], [0.5, 1.0], [0.2, 0.3]])
        V = jnp.diag(jnp.array([1.0, 2.0, 0.5]))
        info = compute_cond_info(G, V)
        assert info["data_only"] == pytest.approx(info["raw"], rel=1e-12)
        assert info["exclude_gauge"] == pytest.approx(info["raw"], rel=1e-12)

    def test_ill_conditioned_when_G_near_rank_deficient(self):
        """Near-collinear columns of G blow up cond(G'G)."""
        # Two nearly identical columns differ by 1e-6 in one row.
        G = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0 + 1e-6],
                [1.0, 1.0],
            ]
        )
        V = jnp.eye(3)
        info = compute_cond_info(G, V)
        # cond(G'G) ~ (2/eps)^2 = 4 / 1e-12 = 4e12 for this construction;
        # comfortably above the 1e6 threshold required for the
        # ill-conditioned acceptance case.
        assert info["raw"] > 1e6

    def test_matches_estimator_info_matrix_construction(self):
        """compute_cond_info reproduces cond(G' V^{-1} G) computed
        directly. This is the same number estimator.py uses for
        Sigma_theta, so the diagnostic is internally consistent.
        """
        G = jnp.array([[1.0, 0.5], [0.3, 1.0], [0.7, 0.2]])
        V = jnp.array(
            [
                [2.0, 0.3, 0.1],
                [0.3, 1.5, 0.2],
                [0.1, 0.2, 1.2],
            ]
        )
        info = compute_cond_info(G, V)
        # Independent reconstruction.
        info_matrix = G.T @ jnp.linalg.solve(V, G)
        expected = float(jnp.linalg.cond(info_matrix))
        assert info["raw"] == pytest.approx(expected, rel=1e-6)


class TestBuildOptimizerHealth:
    """Unit tests for :func:`build_optimizer_health`."""

    def test_basic_fields_present(self):
        info = OptimizerInfo(
            steps=12,
            status="converged",
            final_objective=0.001,
            backend="optimistix",
        )
        h = build_optimizer_health(info, final_gradient_norm=1.5e-9)
        assert set(h.keys()) == {
            "iters",
            "grad_norm",
            "step_norm",
            "accepted_step_count",
        }
        assert h["iters"] == 12
        assert h["grad_norm"] == pytest.approx(1.5e-9, rel=1e-12)
        # step_norm and accepted_step_count not exposed by either LM
        # backend in v1; default to None.
        assert h["step_norm"] is None
        assert h["accepted_step_count"] is None

    def test_optional_fields_pass_through(self):
        info = OptimizerInfo(
            steps=3, status="converged", final_objective=0.0, backend="stub"
        )
        h = build_optimizer_health(
            info,
            final_gradient_norm=0.0,
            step_norm=2.5e-7,
            accepted_step_count=2,
        )
        assert h["step_norm"] == pytest.approx(2.5e-7)
        assert h["accepted_step_count"] == 2


# ---------------------------------------------------------------------------
# build_diagnostics accepts cond_info and optimizer_health
# ---------------------------------------------------------------------------


class TestBuildDiagnosticsExtraFields:
    def test_cond_info_passthrough(self):
        Moments = axes_mod.moments_axis(1)
        d = build_diagnostics(
            tau_realised=0.0,
            kappa_V=1.0,
            binding_ridge=False,
            cholesky_pivot_min=1.0,
            final_objective=0.0,
            final_gradient_norm=0.0,
            N_j_array=jnp.array([1.0]),
            moment_residual_array=jnp.array([0.0]),
            moments_axis=Moments,
            optimizer_info=_stub_optimizer_info(),
            cond_info={"raw": 2.0, "data_only": 2.0, "exclude_gauge": 2.0},
            optimizer_health={
                "iters": 5,
                "grad_norm": 1e-9,
                "step_norm": None,
                "accepted_step_count": None,
            },
        )
        assert d.cond_info["raw"] == pytest.approx(2.0)
        assert d.cond_info["data_only"] == pytest.approx(2.0)
        assert d.cond_info["exclude_gauge"] == pytest.approx(2.0)
        assert d.optimizer_health["iters"] == 5

    def test_defaults_to_empty_dicts(self):
        """Backwards-compat: caller may omit cond_info/optimizer_health
        and receive an empty dict (rather than e.g. ``None``)."""
        Moments = axes_mod.moments_axis(1)
        d = build_diagnostics(
            tau_realised=0.0,
            kappa_V=1.0,
            binding_ridge=False,
            cholesky_pivot_min=1.0,
            final_objective=0.0,
            final_gradient_norm=0.0,
            N_j_array=jnp.array([1.0]),
            moment_residual_array=jnp.array([0.0]),
            moments_axis=Moments,
            optimizer_info=_stub_optimizer_info(),
        )
        assert d.cond_info == {}
        assert d.optimizer_health == {}


# ---------------------------------------------------------------------------
# Integration: estimate() populates cond_info and optimizer_health
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _TwoParam:
    """Minimal parameter container for the toy-AnalyticalMeasure tests."""

    a: float
    b: float


def _model_unused(x, theta):  # pragma: no cover - never called
    return jnp.zeros(())


def _well_conditioned_setup():
    """Closed-form moment vector with a well-conditioned Jacobian.

    M = 3, K = 2, theta_true = (1.0, 2.0). The Jacobian is constant in
    theta and is orthogonal up to scale, so cond(G' V^{-1} G) is O(1).
    """

    def expectation_fn(model, theta):
        a, b = theta.a, theta.b
        # m_0 = a - 1, m_1 = b - 2, m_2 = (a + b) - 3
        return jnp.stack([a - 1.0, b - 2.0, (a + b) - 3.0])

    def covariance_fn(model, theta):
        return jnp.eye(3)

    return expectation_fn, covariance_fn


def _ill_conditioned_setup():
    """Closed-form moment vector with a near-rank-deficient Jacobian.

    M = 3, K = 2. Both moments depend on (a + b) with only a tiny 1e-6
    perturbation differentiating them, so G has near-collinear columns
    and cond(G' V^{-1} G) blows up beyond 1e6.

    Theta_true = (0.0, 0.0): all moments vanish there.
    """
    eps = 1e-6

    def expectation_fn(model, theta):
        a, b = theta.a, theta.b
        s = a + b
        return jnp.stack(
            [
                s,
                s + eps * (a - b),
                s,
            ]
        )

    def covariance_fn(model, theta):
        return jnp.eye(3)

    return expectation_fn, covariance_fn


class TestEstimateCondInfoAndOptimizerHealth:
    """Acceptance tests for the new diagnostic fields under
    :func:`emu_gmm.estimate`."""

    def test_well_conditioned_cond_info_small(self):
        exp_fn, cov_fn = _well_conditioned_setup()
        measure = AnalyticalMeasure(expectation_fn=exp_fn)
        result = estimate(
            model=_model_unused,
            measure=measure,
            covariance=AnalyticalCovariance(covariance_fn=cov_fn),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.0, b=0.0),
        )
        info = result.diagnostics.cond_info
        assert "raw" in info
        assert "data_only" in info
        assert "exclude_gauge" in info
        # Well-conditioned: O(1) cond. Use 1e3 as the issue spec
        # threshold; the actual value here is well below 100.
        assert info["raw"] < 1e3
        # v1 alias behaviour.
        assert info["data_only"] == pytest.approx(info["raw"], rel=1e-12)
        assert info["exclude_gauge"] == pytest.approx(info["raw"], rel=1e-12)

    def test_ill_conditioned_cond_info_large(self):
        exp_fn, cov_fn = _ill_conditioned_setup()
        measure = AnalyticalMeasure(expectation_fn=exp_fn)
        result = estimate(
            model=_model_unused,
            measure=measure,
            covariance=AnalyticalCovariance(covariance_fn=cov_fn),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-6, atol=1e-6, max_steps=400),
            theta_init=_TwoParam(a=0.1, b=-0.1),
        )
        info = result.diagnostics.cond_info
        # Near-rank-deficient G drives cond(G' V^{-1} G) above 1e6.
        assert info["raw"] > 1e6

    def test_optimizer_health_iters_positive(self):
        exp_fn, cov_fn = _well_conditioned_setup()
        measure = AnalyticalMeasure(expectation_fn=exp_fn)
        result = estimate(
            model=_model_unused,
            measure=measure,
            covariance=AnalyticalCovariance(covariance_fn=cov_fn),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.0, b=0.0),
        )
        health = result.diagnostics.optimizer_health
        assert "iters" in health
        assert "grad_norm" in health
        assert "step_norm" in health
        assert "accepted_step_count" in health
        # iters strictly positive: a converged LM run always took at
        # least one step (even if zero correction was accepted at the
        # first probe).
        assert int(health["iters"]) > 0
        # grad_norm finite and non-negative.
        assert health["grad_norm"] >= 0.0

    def test_optimizer_health_grad_norm_finite(self):
        exp_fn, cov_fn = _well_conditioned_setup()
        measure = AnalyticalMeasure(expectation_fn=exp_fn)
        result = estimate(
            model=_model_unused,
            measure=measure,
            covariance=AnalyticalCovariance(covariance_fn=cov_fn),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
            theta_init=_TwoParam(a=0.0, b=0.0),
        )
        health = result.diagnostics.optimizer_health
        assert jnp.isfinite(jnp.asarray(health["grad_norm"]))
        # And it should be small at the optimum.
        assert float(health["grad_norm"]) < 1e-4


from emu_gmm.diagnostics import regularization_adjusted_pvalue  # noqa: E402


class TestAdjustedPvalueGaugeAware:
    """#137: the projector's Gram inverse must be gauge-aware.

    The old plain inv(G~'G~) was singular for gauge-bearing manifolds
    and returned silently wrong, plausible-looking p-values. Pins:
    (1) v1 / full-rank G: bitwise-unchanged (drop 0 short-circuits to
        inv); (2) rank-deficient G (one exact gauge direction): matches
        an independent numpy eigendecomposition reference, and differs
        materially from what the old inv-based formula produced.
    """

    @staticmethod
    def _reference(J, V, V_star, G, gauge_dim):
        import numpy as np
        import scipy.stats as st

        L = np.linalg.cholesky(np.asarray(V_star))
        Gt = np.linalg.solve(L, np.asarray(G))
        Vt = np.linalg.solve(L, np.linalg.solve(L, np.asarray(V)).T).T
        Vt = 0.5 * (Vt + Vt.T)
        # Moore-Penrose projector at the true rank.
        P = np.eye(G.shape[0]) - Gt @ np.linalg.pinv(Gt.T @ Gt) @ Gt.T
        w = np.linalg.eigvalsh(0.5 * (P @ Vt @ P + (P @ Vt @ P).T))
        w = np.where(w > 0, w, 0.0)
        s1, s2 = w.sum(), (w**2).sum()
        c, v = s2 / s1, s1**2 / s2
        return float(st.chi2.sf(float(J) / c, v))

    def _fixture(self, gauge_dim):
        import numpy as np

        rng = np.random.default_rng(42)
        M, D = 6, 3
        A = rng.standard_normal((M, M))
        V = A @ A.T / M  # PD
        V_star = V + 0.05 * np.diag(np.diag(V))  # binding-ridge V*
        G = rng.standard_normal((M, D))
        if gauge_dim:
            # an EXACT nullspace direction: last column = first - second
            G[:, 2] = G[:, 0] - G[:, 1]
            # rotate so the nullspace is not axis-aligned (generic case)
            Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
            G = G @ Q
        return jnp.asarray(V), jnp.asarray(V_star), jnp.asarray(G)

    def test_full_rank_unchanged(self):
        import numpy as np

        V, V_star, G = self._fixture(gauge_dim=0)
        J = jnp.asarray(3.7)
        p_new = regularization_adjusted_pvalue(J, V, V_star, G)
        ref = self._reference(J, V, V_star, G, 0)
        np.testing.assert_allclose(float(p_new), ref, rtol=1e-10)

    def test_gauge_deficient_matches_reference_and_old_was_wrong(self):
        import numpy as np

        V, V_star, G = self._fixture(gauge_dim=1)
        J = jnp.asarray(3.7)
        p_new = float(
            regularization_adjusted_pvalue(J, V, V_star, G, gauge_nullspace_dim=1)
        )
        ref = self._reference(J, V, V_star, G, 1)
        np.testing.assert_allclose(p_new, ref, rtol=1e-8)
        # The OLD formula (plain inv on the singular Gram matrix):
        # reproduce it and confirm it disagreed materially -- the audit's
        # silently-wrong-but-plausible failure mode.
        L = np.linalg.cholesky(np.asarray(V_star))
        Gt = np.linalg.solve(L, np.asarray(G))
        Vt = np.linalg.solve(L, np.linalg.solve(L, np.asarray(V)).T).T
        with np.errstate(all="ignore"):
            P_old = np.eye(6) - Gt @ np.linalg.inv(Gt.T @ Gt) @ Gt.T
            w = np.linalg.eigvalsh(0.5 * (P_old @ Vt @ P_old + (P_old @ Vt @ P_old).T))
        # Either the old path NaN'd outright, or produced a finite but
        # wrong p; both count as the defect.
        if np.isfinite(w).all():
            w = np.where(w > 0, w, 0.0)
            import scipy.stats as st

            s1, s2 = w.sum(), (w**2).sum()
            p_old = float(st.chi2.sf(3.7 / (s2 / s1), s1**2 / s2))
            assert abs(p_old - p_new) > 1e-3, (p_old, p_new)
