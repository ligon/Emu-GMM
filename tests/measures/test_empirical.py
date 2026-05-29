"""Tests for emu_gmm.measures.empirical."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas as pd
import pytest
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.types import Measure


@jdc.pytree_dataclass
class _LinearParams:
    a: float
    b: float


def _linear_residual(x, theta):
    """psi(x, theta) = [theta.a + theta.b * x[0]]: a 1-moment model."""
    return jnp.array([theta.a + theta.b * x[0]])


def _two_moment_residual(x, theta):
    """psi(x, theta) = [a + x[0], b * x[1]]: a 2-moment, 2-variable model."""
    return jnp.array([theta.a + x[0], theta.b * x[1]])


# ---------------------------------------------------------------------------


class TestExpectation:
    def test_satisfies_measure_protocol(self):
        meas = EmpiricalMeasure(
            x=jnp.zeros((4, 1)),
            mask=jnp.ones((4, 1)),
            weights=jnp.ones(4),
        )
        assert isinstance(meas, Measure)

    def test_uniform_data_all_ones_mask_weights_matches_np_mean(self):
        """E[psi] = (1/N) sum psi when mask and weights are all-ones."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (200, 1))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((200, 1)),
            weights=jnp.ones(200),
        )
        theta = _LinearParams(a=0.5, b=2.0)
        # Reference: vmap psi and take a plain mean.
        psi_batch = jax.vmap(lambda xi: _linear_residual(xi, theta))(x)
        m = meas.expectation(_linear_residual, theta)
        assert m.shape == (1,)
        assert float(m[0]) == pytest.approx(float(jnp.mean(psi_batch)), rel=1e-6)
        # Equivalent to a direct numpy mean of (a + b*x).
        ref = 0.5 + 2.0 * float(np.mean(x))
        assert float(m[0]) == pytest.approx(ref, rel=1e-5)

    def test_per_coordinate_missingness(self):
        """Per-moment normalisation uses only the rows where that moment
        is observable.
        """
        # 6 rows, 2 moments. Moment 0 fully observed; moment 1 observed
        # only on the first 3 rows.
        x = jnp.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        weights = jnp.ones(6)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        theta = _LinearParams(a=0.0, b=1.0)
        # psi(x, theta) = [a + b*x[0], b*x[1]] (using _two_moment_residual style)
        m = meas.expectation(_two_moment_residual, theta)
        # Moment 0 = mean of (0 + 1*x[0]) over all 6 rows = (1+2+...+6)/6 = 3.5
        assert float(m[0]) == pytest.approx(3.5, rel=1e-6)
        # Moment 1 = mean of (1*x[1]) over first 3 rows = (10+20+30)/3 = 20.0
        assert float(m[1]) == pytest.approx(20.0, rel=1e-6)

    def test_non_uniform_weights_produce_weighted_mean(self):
        """Weights enter as d_ij * w_i in numerator and denominator."""
        # Three observations with explicit weights.
        x = jnp.array([[1.0], [2.0], [3.0]])
        mask = jnp.ones((3, 1))
        weights = jnp.array([1.0, 2.0, 3.0])
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        theta = _LinearParams(a=0.0, b=1.0)
        m = meas.expectation(_linear_residual, theta)
        # Weighted mean of x with weights [1, 2, 3]:
        # (1*1 + 2*2 + 3*3) / (1 + 2 + 3) = (1 + 4 + 9) / 6 = 14/6
        assert float(m[0]) == pytest.approx(14.0 / 6.0, rel=1e-6)

    def test_all_ones_weights_equals_unweighted(self):
        """A vector of all-ones weights gives the unweighted mean."""
        key = jax.random.PRNGKey(11)
        x = jax.random.uniform(key, (50, 1))
        meas_w = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((50, 1)),
            weights=jnp.ones(50),
        )
        theta = _LinearParams(a=0.3, b=-1.2)
        m_w = meas_w.expectation(_linear_residual, theta)
        psi_batch = jax.vmap(lambda xi: _linear_residual(xi, theta))(x)
        assert jnp.allclose(m_w, jnp.mean(psi_batch, axis=0))


# ---------------------------------------------------------------------------


class TestJacobian:
    def test_shape(self):
        meas = EmpiricalMeasure(
            x=jnp.ones((10, 1)),
            mask=jnp.ones((10, 1)),
            weights=jnp.ones(10),
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_linear_residual, theta)
        assert G.shape == (1, 2)

    def test_against_analytical(self):
        """For psi = [a + b * x[0]]:
        E[psi] = a + b * E[x], so d/da = 1 and d/db = E[x].
        """
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (300, 1))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((300, 1)),
            weights=jnp.ones(300),
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_linear_residual, theta)
        e_x = float(jnp.mean(x[:, 0]))
        assert float(G[0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(G[0, 1]) == pytest.approx(e_x, abs=1e-6)

    def test_jacobian_respects_mask(self):
        """The per-moment Jacobian uses only the observable rows."""
        # Make moment 1 (b * x[1]) observable only on the first three rows.
        x = jnp.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        weights = jnp.ones(6)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        theta = _LinearParams(a=0.0, b=1.0)
        G = meas.jacobian(_two_moment_residual, theta)
        # G[0, 0] = d/da of moment 0 (= a + x[0]) = 1.
        assert float(G[0, 0]) == pytest.approx(1.0, abs=1e-6)
        # G[0, 1] = d/db of moment 0 = 0 (no b dependence).
        assert float(G[0, 1]) == pytest.approx(0.0, abs=1e-6)
        # G[1, 0] = d/da of moment 1 (= b * x[1]) = 0.
        assert float(G[1, 0]) == pytest.approx(0.0, abs=1e-6)
        # G[1, 1] = d/db of moment 1 = mean of x[1] over first 3 rows = 20.
        assert float(G[1, 1]) == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_dataframe_round_trip(self):
        """DataFrame -> measure -> x array matches df.to_numpy()."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        meas = EmpiricalMeasure.from_pandas(df)
        assert meas.x.shape == (3, 2)
        np.testing.assert_allclose(np.asarray(meas.x), df.to_numpy())
        # Default weights are all-ones.
        np.testing.assert_allclose(np.asarray(meas.weights), np.ones(3))
        # Default mask is all-ones (N, D) since no explicit mask was given.
        np.testing.assert_allclose(np.asarray(meas.mask), np.ones((3, 2)))

    def test_series_weights(self):
        """A pd.Series of weights propagates through."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        weights = pd.Series([0.25, 0.5, 0.75, 1.0])
        meas = EmpiricalMeasure.from_pandas(df, weights=weights)
        np.testing.assert_allclose(np.asarray(meas.weights), weights.to_numpy())

    def test_mask_dataframe(self):
        """A DataFrame mask propagates through with shape (N, M)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        mask = pd.DataFrame(
            {
                "m0": [1.0, 1.0, 1.0],
                "m1": [1.0, 0.0, 1.0],
                "m2": [0.0, 1.0, 1.0],
            }
        )
        meas = EmpiricalMeasure.from_pandas(df, mask=mask)
        assert meas.mask.shape == (3, 3)
        np.testing.assert_allclose(np.asarray(meas.mask), mask.to_numpy())


# ---------------------------------------------------------------------------


class TestNanMaskSemantics:
    """NaN-laden residual rows at mask=0 positions are dropped, not propagated.

    Surfaces the "port from manifoldgmm.GMM(restriction=...)" use case
    (issue #1): consumers that hand the framework pandas DataFrames with
    NaN-encoded missingness expect the natural pandas convention --- NaN
    means "missing, drop from the moment estimator" --- to be respected
    through the autodiff tape.
    """

    def test_expectation_drops_nan_at_masked_positions(self):
        """NaN in psi(x_i, theta) is zeroed where mask[i, j] == 0."""
        # Row 1 has NaN; mask says row 1 is unobservable for moment 0.
        x = jnp.array([[1.0], [float("nan")], [3.0]])
        mask = jnp.array([[1.0], [0.0], [1.0]])
        weights = jnp.ones(3)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)

        def psi(xi, theta):
            return jnp.array([xi[0] + theta])

        m = meas.expectation(psi, jnp.asarray(0.0))
        # Expected: weighted mean of rows 0, 2 (1.0 and 3.0) = 2.0
        assert jnp.isfinite(m).all()
        assert float(m[0]) == pytest.approx(2.0, abs=1e-6)

    def test_jacobian_drops_nan_at_masked_positions(self):
        """NaN in grad psi(x_i, theta) is zeroed where mask[i, j] == 0."""
        x = jnp.array([[1.0], [float("nan")], [3.0]])
        mask = jnp.array([[1.0], [0.0], [1.0]])
        weights = jnp.ones(3)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)

        # psi has theta.mu * x[0] so d/dtheta = x[0]. NaN at row 1 must
        # not propagate.
        @jdc.pytree_dataclass
        class P:
            mu: float

        def psi(xi, theta):
            return jnp.array([theta.mu * xi[0]])

        G = meas.jacobian(psi, P(mu=0.5))
        # d psi / d theta = x[0]; masked mean over rows 0, 2 = (1+3)/2 = 2.
        assert jnp.isfinite(G).all()
        assert float(G[0, 0]) == pytest.approx(2.0, abs=1e-6)

    def test_observed_nan_still_propagates(self):
        """NaN at an observed (mask=1) position is *not* silently dropped.

        That case is a genuine residual-evaluation bug, not a missingness
        signal; surfacing it as a NaN moment is the right behaviour.
        """
        x = jnp.array([[1.0], [float("nan")], [3.0]])
        # Mask claims row 1 IS observed: the user is asserting psi(x_1, .)
        # is well-defined despite the NaN input.
        mask = jnp.array([[1.0], [1.0], [1.0]])
        weights = jnp.ones(3)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)

        def psi(xi, theta):
            return jnp.array([xi[0]])

        m = meas.expectation(psi, jnp.asarray(0.0))
        assert jnp.isnan(m[0])

    def test_from_pandas_derives_mask_from_nan(self):
        """``from_pandas`` with no explicit mask derives one from NaN positions."""
        df = pd.DataFrame(
            {
                "x0": [1.0, float("nan"), 3.0, 4.0],
                "x1": [10.0, 20.0, float("nan"), 40.0],
            }
        )
        meas = EmpiricalMeasure.from_pandas(df)
        # Row 1 and row 2 each have at least one NaN -> unobserved.
        expected_mask = jnp.array(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
        np.testing.assert_allclose(np.asarray(meas.mask), np.asarray(expected_mask))
        # NaN in x is replaced with zero so it doesn't poison the AD tape.
        assert not bool(jnp.any(jnp.isnan(meas.x)))

    def test_from_pandas_no_nan_yields_all_ones_mask(self):
        """When df has no NaN, the derived mask is all-ones."""
        df = pd.DataFrame({"x0": [1.0, 2.0, 3.0], "x1": [10.0, 20.0, 30.0]})
        meas = EmpiricalMeasure.from_pandas(df)
        np.testing.assert_allclose(np.asarray(meas.mask), np.ones((3, 2)))

    def test_from_pandas_explicit_mask_strips_nan_x(self):
        """Explicit-mask path still zeros NaN in x to keep AD finite."""
        df = pd.DataFrame({"x0": [1.0, float("nan"), 3.0], "x1": [10.0, 20.0, 30.0]})
        # User asserts row 1 is unobservable on moment 0 but observable on
        # moment 1.
        mask = pd.DataFrame({"m0": [1.0, 0.0, 1.0], "m1": [1.0, 1.0, 1.0]})
        meas = EmpiricalMeasure.from_pandas(df, mask=mask)
        # NaN stripped from x.
        assert not bool(jnp.any(jnp.isnan(meas.x)))
        # Mask preserved verbatim.
        np.testing.assert_allclose(np.asarray(meas.mask), mask.to_numpy())

    def test_end_to_end_nan_in_psi_via_estimator(self):
        """End-to-end: estimator runs on NaN-laden empirical data via mask.

        Over-identified two-moment model: psi(x, theta) = [x - mu,
        (x - mu)^2 - sigma^2]. NaN rows are dropped via the mask. The
        estimator should converge to the unmasked-sample mean and the
        unmasked-sample (uncentered) second moment.
        """
        import jax
        from emu_gmm import (
            ContinuouslyUpdated,
            DiagonalTikhonov,
            IIDCovariance,
            estimate,
            optimistix_lm,
        )

        @jdc.pytree_dataclass
        class _TwoParam:
            mu: float
            sigma2: float

        key = jax.random.PRNGKey(0)
        n = 500
        x_clean = jax.random.normal(key, (n, 1))
        # Inject NaN into 80 rows at random.
        nan_idx = jax.random.choice(key, n, (80,), replace=False)
        x = x_clean.at[nan_idx, 0].set(jnp.nan)
        # Single column -> single moment dimension D = 1; widen the mask
        # to cover M = 2 moments so the row drops for both.
        mask = jnp.ones((n, 2)).at[nan_idx, 0].set(0.0).at[nan_idx, 1].set(0.0)

        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(n))

        def psi(xi, theta):
            d = xi[0] - theta.mu
            return jnp.array([d, d * d - theta.sigma2])

        r = estimate(
            model=psi,
            measure=meas,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(),
            theta_init=_TwoParam(mu=0.5, sigma2=2.0),
        )
        unmasked = jnp.array(mask[:, 0], dtype=bool)
        expected_mu = float(jnp.mean(x_clean[unmasked, 0]))
        expected_sigma2 = float(jnp.mean((x_clean[unmasked, 0] - expected_mu) ** 2))
        assert float(r.theta_hat.mu) == pytest.approx(expected_mu, abs=1e-3)
        assert float(r.theta_hat.sigma2) == pytest.approx(expected_sigma2, abs=1e-3)
        assert r.converged


# ---------------------------------------------------------------------------


class TestJit:
    def test_expectation_jits(self):
        x = jnp.linspace(0.0, 1.0, 20).reshape(20, 1)
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((20, 1)),
            weights=jnp.ones(20),
        )
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.expectation(_linear_residual, t)

        eager = meas.expectation(_linear_residual, theta)
        jit_result = compute(meas, theta)
        assert jnp.allclose(eager, jit_result)

    def test_jacobian_jits(self):
        x = jnp.linspace(0.0, 1.0, 20).reshape(20, 1)
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((20, 1)),
            weights=jnp.ones(20),
        )
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.jacobian(_linear_residual, t)

        G_eager = meas.jacobian(_linear_residual, theta)
        G_jit = compute(meas, theta)
        assert jnp.allclose(G_eager, G_jit)
