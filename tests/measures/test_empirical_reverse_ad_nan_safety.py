"""Reverse-mode-AD safety of NaN-masked aggregations on EmpiricalMeasure.

Regression for the JAX-AD high-severity finding in
``docs/reviews/v1x-jax-ad-review.org``: the canonical "double where"
guard only protects against NaN cells in the input ``x``, not against
:math:`\\psi` evaluating to NaN/Inf at the chosen sentinel. For
residuals that are partial at zero (``log``, ``1/x``, ``sqrt``) the
previous implementation substituted ``0.0`` at masked-out cells and
produced ``NaN`` cotangents under ``jax.grad`` / ``jax.jacrev`` even
though the primal value was finite. The fix substitutes the
per-column observed mean instead (see
:func:`emu_gmm._internal.nan_safety.safe_x_for_psi`); these tests
exercise the partial-residual case end-to-end and assert that all of
the reverse-mode and forward-mode derivatives are finite.

The tests live here (rather than in ``test_empirical.py``) so the
regression is self-contained and easy to delete if the API ever
changes; they intentionally do not depend on the rest of the file.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure


@jdc.pytree_dataclass
class _LogParams:
    """One-parameter intercept for the log-residual model."""

    a: float


def _log_residual(x, theta):
    """psi(x, theta) = [log(x[0]) + theta.a].

    The log makes the residual partial at zero --- evaluating
    ``log(0)`` returns ``-inf`` and ``grad`` of it returns ``inf`` /
    ``NaN``. This is the canonical case where the old ``0.0`` sentinel
    poisons reverse-mode AD even when the mask zeroes the primal
    contribution.
    """
    return jnp.array([jnp.log(x[0]) + theta.a])


def _inverse_residual(x, theta):
    """psi(x, theta) = [theta.a / x[0]] --- partial at zero via division."""
    return jnp.array([theta.a / x[0]])


def _sqrt_residual(x, theta):
    """psi(x, theta) = [sqrt(x[0]) - theta.a] --- partial at negatives."""
    return jnp.array([jnp.sqrt(x[0]) - theta.a])


def _nan_laden_measure(
    n_obs: int = 8, frac_masked: float = 0.25, seed: int = 0
) -> EmpiricalMeasure:
    """Build a (N, 1)-column measure with a fraction of the rows NaN."""
    rng = np.random.default_rng(seed)
    # Strictly positive observations so log / sqrt / 1/x are all defined
    # on the *observed* rows.
    base = rng.uniform(0.5, 2.0, size=(n_obs, 1))
    n_mask = max(1, int(round(frac_masked * n_obs)))
    nan_idx = rng.choice(n_obs, size=n_mask, replace=False)
    x = base.copy()
    x[nan_idx, 0] = np.nan
    return EmpiricalMeasure.from_nan_aware(x)


# ---------------------------------------------------------------------------
# EmpiricalMeasure.expectation


class TestExpectationReverseAdNanSafety:
    @pytest.mark.parametrize("psi", [_log_residual, _inverse_residual, _sqrt_residual])
    def test_grad_is_finite_on_partial_residual(self, psi):
        """jax.grad of a scalar objective through expectation is finite.

        The primal is finite under the old code too; the bug shows up
        only on the cotangent flow through the masked-out cells.
        """
        meas = _nan_laden_measure()

        def half_obj(a):
            theta = _LogParams(a=a)
            m = meas.expectation(psi, theta)
            return 0.5 * jnp.sum(m * m)

        # Primal must be finite (sanity).
        assert jnp.isfinite(half_obj(0.3))
        # Reverse-mode gradient must be finite (the regression).
        g_rev = jax.grad(half_obj)(0.3)
        assert jnp.isfinite(
            g_rev
        ), f"jax.grad of expectation poisoned by NaN cotangent: {g_rev}"
        # Forward-mode and reverse-mode should agree.
        g_fwd = jax.jacfwd(half_obj)(0.3)
        assert jnp.isfinite(g_fwd)
        np.testing.assert_allclose(float(g_rev), float(g_fwd), rtol=1e-6)

    def test_jacrev_through_expectation_is_finite(self):
        """jax.jacrev directly on expectation: vector-valued output path."""
        meas = _nan_laden_measure()

        def expect_at(a):
            return meas.expectation(_log_residual, _LogParams(a=a))

        J = jax.jacrev(expect_at)(0.3)
        assert jnp.all(jnp.isfinite(J)), f"jacrev poisoned: {J}"


# ---------------------------------------------------------------------------
# EmpiricalMeasure.moment_contributions


class TestMomentContributionsReverseAdNanSafety:
    def test_grad_through_moment_contributions_is_finite(self):
        meas = _nan_laden_measure()

        def half_obj(a):
            theta = _LogParams(a=a)
            g = meas.moment_contributions(_log_residual, theta)
            # Sum-of-squares aggregate so grad has the same shape as a.
            return 0.5 * jnp.sum(g * g)

        assert jnp.isfinite(half_obj(0.3))
        g_rev = jax.grad(half_obj)(0.3)
        assert jnp.isfinite(
            g_rev
        ), f"moment_contributions reverse grad poisoned: {g_rev}"


# ---------------------------------------------------------------------------
# EmpiricalMeasure.jacobian


class TestJacobianReverseAdNanSafety:
    def test_grad_through_jacobian_is_finite(self):
        """The jacobian uses jacfwd internally, but reverse-mode AD
        through its output must still be finite at the masked-out cells.
        """
        meas = _nan_laden_measure()

        def half_obj(a):
            theta = _LogParams(a=a)
            J = meas.jacobian(_log_residual, theta)
            return 0.5 * jnp.sum(J * J)

        assert jnp.isfinite(half_obj(0.3))
        g_rev = jax.grad(half_obj)(0.3)
        assert jnp.isfinite(g_rev), f"jacobian reverse grad poisoned: {g_rev}"


# ---------------------------------------------------------------------------
# IIDCovariance


class TestIIDCovarianceReverseAdNanSafety:
    def test_grad_through_iid_covariance_is_finite(self):
        meas = _nan_laden_measure()
        cov = IIDCovariance()

        def half_obj(a):
            theta = _LogParams(a=a)
            V = cov.covariance(_log_residual, theta, meas)
            return 0.5 * jnp.sum(V * V)

        assert jnp.isfinite(half_obj(0.3))
        g_rev = jax.grad(half_obj)(0.3)
        assert jnp.isfinite(g_rev), f"IIDCovariance reverse grad poisoned: {g_rev}"


# ---------------------------------------------------------------------------
# ClusteredCovariance


class TestClusteredCovarianceReverseAdNanSafety:
    def test_grad_through_clustered_covariance_is_finite(self):
        meas = _nan_laden_measure(n_obs=8)
        # Two clusters of size four.
        cluster_ids = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=2)

        def half_obj(a):
            theta = _LogParams(a=a)
            V = cov.covariance(_log_residual, theta, meas)
            return 0.5 * jnp.sum(V * V)

        assert jnp.isfinite(half_obj(0.3))
        g_rev = jax.grad(half_obj)(0.3)
        assert jnp.isfinite(
            g_rev
        ), f"ClusteredCovariance reverse grad poisoned: {g_rev}"


# ---------------------------------------------------------------------------
# Primal-value invariance: the change of sentinel must not change the
# observed primal aggregate (masked-out rows still contribute zero).


class TestPrimalValueUnchanged:
    """The sentinel substitution is observable only on cotangents; the
    forward value must equal the unmasked aggregate over the observed
    rows alone.
    """

    def test_expectation_matches_observed_only_aggregate(self):
        rng = np.random.default_rng(7)
        base = rng.uniform(0.5, 2.0, size=(10, 1))
        x = base.copy()
        x[[2, 5, 9], 0] = np.nan
        meas = EmpiricalMeasure.from_nan_aware(x)
        theta = _LogParams(a=0.0)

        m = meas.expectation(_log_residual, theta)
        # Reference: take log on only the observed rows and average.
        observed = base[[0, 1, 3, 4, 6, 7, 8], 0]
        ref = float(np.mean(np.log(observed)))
        assert float(m[0]) == pytest.approx(ref, rel=1e-6, abs=1e-8)
