"""Regularisation-adjusted J p-value tests.

When the diagonal-Tikhonov ridge is *binding* (tau exceeds
``tau_threshold``), the J-statistic's asymptotic distribution is no
longer the nominal chi^2_{M-K} but a weighted sum of M-K independent
chi^2_1 variates whose weights are the eigenvalues of (V_star)^{-1} V
projected onto the orthogonal complement of the column space of
G(theta_0). See ``docs/mcar-asymptotics.org`` Theorem 6.

These tests verify:

1. When ``binding_ridge=False`` (the typical case), the adjusted
   p-value matches the nominal chi^2 value to floating-point precision.
2. When ``binding_ridge=True``, the two p-values differ (the adjusted
   value reflects the weighted-chi^2 limit, the nominal value does not).
3. The ``regularization_adjusted_pvalue`` helper traces under jit.

Reference: docs/reviews/v1x-test-gaps.org HIGH binding-ridge finding;
docs/mcar-asymptotics.org line 142.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    ContinuouslyUpdated,
    DiagonalTikhonov,
    estimate,
    optimistix_lm,
)
from emu_gmm.diagnostics import regularization_adjusted_pvalue


@jdc.pytree_dataclass
class _Params2D:
    a: float
    b: float


def _dummy_model(x, theta):
    del x, theta
    return jnp.zeros((3,))


def _two_param_moments(model, theta: _Params2D):
    """M=3 moment vector linear in (a, b); over-identified."""
    del model
    return jnp.array(
        [
            theta.a + 0.5 * theta.b - 0.1,
            -0.3 * theta.a + theta.b - 0.05,
            0.7 * theta.a + 0.4 * theta.b + 0.02,
        ]
    )


def _ill_conditioned_cov(model, theta):
    """Fixed V with kappa ~ 1e6 (independent of theta); designed to
    trigger the DiagonalTikhonov ridge at kappa_target=1e3."""
    del model, theta
    rng = np.random.default_rng(seed=11)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    eigvals = jnp.array([1.0, 1e-3, 1e-6])
    D = jnp.diag(eigvals)
    V = jnp.asarray(Q) @ D @ jnp.asarray(Q).T
    # symmetrise
    return 0.5 * (V + V.T)


def _well_conditioned_cov(model, theta):
    """Identity covariance: kappa = 1, ridge never binds."""
    del model, theta
    return jnp.eye(3)


class TestRegularizationAdjustedPvalue:
    """End-to-end: ``estimate`` returns a sensible adjusted p-value."""

    def _common_kwargs(self, cov_fn, kappa_target=1.0e3, tau_threshold=1.0e-2):
        return {
            "model": _dummy_model,
            "measure": AnalyticalMeasure(expectation_fn=_two_param_moments),
            "covariance": AnalyticalCovariance(covariance_fn=cov_fn),
            "weighting": ContinuouslyUpdated(),
            "regularization": DiagonalTikhonov(
                kappa_target=kappa_target, tau_threshold=tau_threshold
            ),
            "optimizer": optimistix_lm(rtol=1e-8, atol=1e-8),
            "theta_init": _Params2D(a=0.0, b=0.0),
        }

    def test_pvalue_adjusted_matches_nominal_when_not_binding(self):
        """With a well-conditioned V the ridge does not bind, so the
        adjusted p-value matches the nominal one to roundoff."""
        result = estimate(**self._common_kwargs(_well_conditioned_cov))
        assert bool(result.diagnostics.binding_ridge) is False
        assert float(result.J_pvalue) == pytest.approx(
            float(result.J_pvalue_adjusted), rel=1e-10, abs=1e-12
        )

    def test_pvalue_adjusted_differs_from_nominal_when_binding(self):
        """With an ill-conditioned V and a tight threshold, the ridge
        binds and the adjusted p-value differs from the nominal.

        The size of the difference depends on how aggressive the
        regularisation is; we use ``kappa_target=10`` to force a
        non-trivial tau and hence a measurable departure from the
        nominal chi^2_{M-K} limit. ``tau_threshold=1e-6`` ensures the
        binding flag is on.
        """
        result = estimate(
            **self._common_kwargs(
                _ill_conditioned_cov,
                kappa_target=10.0,
                tau_threshold=1.0e-6,
            )
        )
        assert bool(result.diagnostics.binding_ridge) is True
        # Both are valid probabilities.
        p_nom = float(result.J_pvalue)
        p_adj = float(result.J_pvalue_adjusted)
        assert 0.0 <= p_nom <= 1.0
        assert 0.0 <= p_adj <= 1.0
        # The two must differ by a measurable amount: the adjustment
        # codepath is active and the eigenvalue spectrum is non-degenerate.
        assert abs(p_nom - p_adj) > 1.0e-4


class TestRegularizationAdjustedPvalueHelper:
    """Direct unit tests for the ``regularization_adjusted_pvalue`` helper."""

    def test_no_ridge_recovers_nominal_chi2(self):
        """V == V_star (tau = 0) reduces the weighted-chi^2 limit to
        the nominal chi^2_{M-K}. Verified via the Welch-Satterthwaite
        approximation: with all weights = 1, c = 1, v = M - K."""
        V = jnp.eye(4)
        V_star = V
        G = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.2, -0.3],
            ]
        )
        J_stat = jnp.asarray(3.0)
        p_adj = regularization_adjusted_pvalue(J_stat, V, V_star, G)
        p_nom = jax.scipy.stats.chi2.sf(J_stat, V.shape[0] - G.shape[1])
        assert float(p_adj) == pytest.approx(float(p_nom), rel=1e-6, abs=1e-8)

    def test_traces_under_jit(self):
        """The helper composes with ``jax.jit`` without hitting any
        concrete-Python boundary."""
        V = jnp.diag(jnp.array([1.0, 0.5, 2.0]))
        V_star = V + 0.05 * jnp.diag(jnp.diag(V))
        G = jnp.array([[1.0], [-1.0], [0.5]])
        J_stat = jnp.asarray(2.0)
        out = jax.jit(regularization_adjusted_pvalue)(J_stat, V, V_star, G)
        assert 0.0 <= float(out) <= 1.0
