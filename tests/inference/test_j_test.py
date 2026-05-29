"""Tests for the zero-parameter J-test helper.

The helper evaluates the standard over-identifying-restrictions
statistic at a user-supplied ``theta_null`` without going through
parameter estimation: ``J = m' V^{-1} m ~ chi^2_M`` (dof = M, not
M - K, because no parameters are estimated).

Three regimes are covered:

1. Truth-at-null --- synthetic data drawn from the DGP, ``theta_null``
   set to the truth: ``J`` should be small (no MC bias at this scale).
2. Misspecified ``theta_null`` --- same data, ``theta_null`` far from
   the truth: ``J`` should be large (the moment conditions are
   violated).
3. Shape and label structure --- :class:`JTestResult` carries the right
   types, axes, and dof.
"""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import pytest
import scipy.stats
from emu_gmm import DiagonalTikhonov, JTestResult, SyntheticCovariance, j_test
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import SyntheticMeasure

N_SIM = 5000


def _measure(seed: int = 0) -> SyntheticMeasure:
    sampler = euler_sampler_factory(N_SIM)
    return SyntheticMeasure(
        key=jax.random.PRNGKey(seed),
        n_sim=N_SIM,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# (a) Truth at null: J should be small.
# ---------------------------------------------------------------------------


class TestJTestAtTruth:
    """At ``theta_null = truth``, the moment conditions hold by construction."""

    def _run(self) -> JTestResult:
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )

    def test_returns_j_test_result(self):
        r = self._run()
        assert isinstance(r, JTestResult)

    def test_J_dof_equals_M(self):
        """Zero estimated parameters → dof equals moment count (M=3)."""
        r = self._run()
        assert r.J_dof == 3

    def test_J_stat_finite_and_small(self):
        """At truth the only source of nonzero J is Monte Carlo noise.

        Under H0, J is asymptotically chi^2_3 with mean 3. With
        N_SIM=5000 we expect J ~= 3 +/- a few; allow 30 to absorb
        sampling variation across CRN-frozen keys.
        """
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        assert r.J_stat < 30.0

    def test_J_pvalue_in_unit_interval(self):
        r = self._run()
        assert jnp.isfinite(r.J_pvalue)
        assert 0.0 <= r.J_pvalue <= 1.0

    def test_pvalue_matches_chi2_sf(self):
        """The reported p-value is exactly the chi^2_dof survival
        function evaluated at J_stat."""
        r = self._run()
        expected = float(scipy.stats.chi2.sf(r.J_stat, r.J_dof))
        assert r.J_pvalue == pytest.approx(expected)


# ---------------------------------------------------------------------------
# (b) Misspecified null: J should be large.
# ---------------------------------------------------------------------------


class TestJTestMisspecified:
    """At a ``theta_null`` far from the truth, J should reject."""

    def _run(self) -> JTestResult:
        # gamma = 5.0 is well away from GAMMA_TRUE=2.0; with NU spread
        # (0.5, 1.0, 1.5) the cross-asset slope makes the moment vector
        # depart non-trivially from zero.
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=0.80, gamma=5.0),
        )

    def test_J_stat_large(self):
        """J is well into the chi^2_3 tail."""
        r = self._run()
        # 99.9% quantile of chi^2_3 is ~16.27. We use a comfortable
        # margin: the true population J at this misspecification is
        # huge (the moment is O(1) per coordinate, V scales as
        # 1/n_sim, so J scales as n_sim).
        assert r.J_stat > 100.0

    def test_J_pvalue_tiny(self):
        r = self._run()
        assert r.J_pvalue < 1e-6


# ---------------------------------------------------------------------------
# (c) Shape and label structure.
# ---------------------------------------------------------------------------


class TestJTestStructure:
    """``JTestResult`` fields have the expected types and axes."""

    def _run(self) -> JTestResult:
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )

    def test_V_X_is_named_array(self):
        r = self._run()
        assert isinstance(r.V_X, ha.NamedArray)

    def test_V_X_axes(self):
        r = self._run()
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_V_X_shape(self):
        r = self._run()
        assert r.V_X.array.shape == (3, 3)

    def test_V_X_symmetric(self):
        r = self._run()
        V = r.V_X.array
        assert jnp.allclose(V, V.T, atol=1e-10)

    def test_V_X_positive_definite(self):
        """Regularised V should be PD --- Cholesky should not produce NaN."""
        r = self._run()
        L = jnp.linalg.cholesky(r.V_X.array)
        assert jnp.all(jnp.isfinite(L))

    def test_scalar_fields_are_python_scalars(self):
        """J_stat, J_pvalue should be plain Python floats; J_dof a plain int.

        Downstream consumers (notably the K-Aggregators
        ``cross_moment_test_via_emu_gmm`` port that this helper unblocks)
        need vanilla Python scalars so they can build their own result
        records without unwrapping JAX arrays.
        """
        r = self._run()
        assert isinstance(r.J_stat, float)
        assert isinstance(r.J_pvalue, float)
        assert isinstance(r.J_dof, int)


# ---------------------------------------------------------------------------
# Default regularization behaviour.
# ---------------------------------------------------------------------------


class TestRegularizationKwarg:
    """The ``regularization`` kwarg defaults to ``DiagonalTikhonov`` and
    accepts an explicit override."""

    def test_default_runs(self):
        r = j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )
        assert jnp.isfinite(r.J_stat)

    def test_explicit_regularization_accepted(self):
        r = j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            regularization=DiagonalTikhonov(kappa_target=1e8),
        )
        assert jnp.isfinite(r.J_stat)
