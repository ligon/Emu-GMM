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


class TestNaNAware:
    """NaN-as-missing semantics at the I/O boundary.

    The hot path is mask-based per ``docs/design.org``; the constructor
    layer converts NaN cells into 0/1 masks and zeroes the NaN cells in
    the stored ``x`` so that downstream JAX arithmetic and AD are NaN-free.
    """

    def test_from_pandas_infers_mask_from_nan(self):
        """When no mask is supplied, ``~df.isna()`` becomes the mask."""
        df = pd.DataFrame(
            {
                "r0": [1.0, 2.0, 3.0, 4.0],
                "r1": [10.0, float("nan"), 30.0, float("nan")],
            }
        )
        meas = EmpiricalMeasure.from_pandas(df)
        # Mask inferred per-cell from NaN.
        expected_mask = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(np.asarray(meas.mask), expected_mask)
        # NaN cells in x are replaced with zero.
        assert not np.any(np.isnan(np.asarray(meas.x)))
        np.testing.assert_allclose(
            np.asarray(meas.x),
            np.array([[1.0, 10.0], [2.0, 0.0], [3.0, 30.0], [4.0, 0.0]]),
        )

    def test_from_pandas_explicit_mask_overrides_nan_inference(self):
        """An explicit mask wins over NaN-inference."""
        df = pd.DataFrame(
            {
                "r0": [1.0, 2.0, 3.0],
                "r1": [10.0, float("nan"), 30.0],
            }
        )
        # Force moment 1 fully off even though only row 1 is NaN.
        explicit_mask = pd.DataFrame({"m0": [1.0, 1.0, 1.0], "m1": [0.0, 0.0, 0.0]})
        meas = EmpiricalMeasure.from_pandas(df, mask=explicit_mask)
        np.testing.assert_allclose(np.asarray(meas.mask), explicit_mask.to_numpy())

    def test_from_pandas_nan_aware_false_preserves_legacy(self):
        """``nan_aware=False`` reproduces the legacy all-ones-mask behaviour."""
        df = pd.DataFrame(
            {
                "r0": [1.0, 2.0],
                "r1": [10.0, float("nan")],
            }
        )
        meas = EmpiricalMeasure.from_pandas(df, nan_aware=False)
        np.testing.assert_allclose(np.asarray(meas.mask), np.ones((2, 2)))
        # And NaN is preserved in x.
        assert bool(jnp.isnan(meas.x[1, 1]))

    def test_from_nan_aware_constructor(self):
        """``from_nan_aware`` derives mask from NaN in a raw array."""
        x_np = np.array([[1.0, np.nan], [2.0, 20.0], [np.nan, 30.0]])
        meas = EmpiricalMeasure.from_nan_aware(x_np)
        expected_mask = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(np.asarray(meas.mask), expected_mask)
        # NaN cells in x replaced with zero.
        assert not np.any(np.isnan(np.asarray(meas.x)))
        np.testing.assert_allclose(np.asarray(meas.weights), np.ones(3))

    def test_from_nan_aware_with_weights(self):
        x_np = np.array([[1.0], [np.nan], [3.0]])
        w = np.array([0.5, 1.0, 1.5])
        meas = EmpiricalMeasure.from_nan_aware(x_np, weights=w)
        np.testing.assert_allclose(np.asarray(meas.weights), w)
        np.testing.assert_allclose(np.asarray(meas.mask), [[1.0], [0.0], [1.0]])

    def test_expectation_nan_safe_with_nan_psi_at_masked_cells(self):
        """A psi that returns NaN where mask == 0 still yields a finite mean.

        Exemplifies the Seasonality / IMRS non-holder pattern: the
        residual is only defined for holders; the framework must zero
        the masked-out contributions before the sum so that the
        per-coordinate :math:`N_j` reflects the holder count and the
        moment sum is finite.
        """
        # Three observations; moment 0 missing on row 1 (a "non-holder").
        x = jnp.array([[1.0, np.nan], [np.nan, 5.0], [3.0, np.nan]])
        # NaN-aware mask: 1 wherever finite.
        meas = EmpiricalMeasure.from_nan_aware(x)

        def psi(xi, theta):
            # Returns the row verbatim; NaN cells reach the aggregator.
            return xi

        # psi as written reads x[1, 0] = 0.0 (cleaned by from_nan_aware),
        # so no NaN should hit the sum. Exercise the deeper guarantee
        # by hand-crafting a measure where x has NaN at masked cells.
        x_with_nan = jnp.array(
            [
                [1.0, jnp.nan],
                [jnp.nan, 5.0],
                [3.0, jnp.nan],
            ]
        )
        mask = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        meas2 = EmpiricalMeasure(x=x_with_nan, mask=mask, weights=jnp.ones(3))
        m = meas2.expectation(psi, _LinearParams(0.0, 0.0))
        assert jnp.all(jnp.isfinite(m))
        # Moment 0 = mean of {1.0, 3.0} = 2.0.
        assert float(m[0]) == pytest.approx(2.0, rel=1e-6)
        # Moment 1 = mean of {5.0} = 5.0.
        assert float(m[1]) == pytest.approx(5.0, rel=1e-6)
        # And the cleaned-NaN path via from_nan_aware agrees.
        m_clean = meas.expectation(psi, _LinearParams(0.0, 0.0))
        np.testing.assert_allclose(np.asarray(m_clean), np.asarray(m), atol=1e-7)

    def test_expectation_gradient_nan_safe(self):
        """The AD tape is NaN-free even when psi has NaN gradients at
        masked cells. Without the where-guard, ``0 * NaN`` propagates
        into the reverse-mode tangent and the gradient is NaN.
        """
        x = jnp.array([[1.0, jnp.nan], [3.0, jnp.nan], [2.0, 7.0]])
        mask = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(3))

        # psi(x, theta) = [theta.a * x[0], theta.b * x[1]]. Gradient
        # with respect to theta.b at the masked-out rows is NaN
        # (NaN * 1 = NaN) without the protective where.
        def psi(xi, theta):
            return jnp.array([theta.a * xi[0], theta.b * xi[1]])

        def total(t_flat):
            theta = _LinearParams(a=t_flat[0], b=t_flat[1])
            return jnp.sum(meas.expectation(psi, theta))

        g = jax.grad(total)(jnp.array([0.5, 2.0]))
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_jacobian_nan_safe(self):
        """``jacobian`` also zeroes NaN-grad cells at masked positions."""
        x = jnp.array([[1.0, jnp.nan], [3.0, jnp.nan], [2.0, 7.0]])
        mask = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(3))

        def psi(xi, theta):
            return jnp.array([theta.a * xi[0], theta.b * xi[1]])

        G = meas.jacobian(psi, _LinearParams(a=0.5, b=2.0))
        assert bool(jnp.all(jnp.isfinite(G)))
        # G[0, 0] = mean of x[:, 0] over all three rows = (1 + 3 + 2) / 3.
        assert float(G[0, 0]) == pytest.approx((1.0 + 3.0 + 2.0) / 3.0, rel=1e-6)
        # G[0, 1] = d/db of moment 0 = 0.
        assert float(G[0, 1]) == pytest.approx(0.0, abs=1e-6)
        # G[1, 1] = mean of x[:, 1] over only row 2 = 7.
        assert float(G[1, 1]) == pytest.approx(7.0, rel=1e-6)

    def test_per_column_n_reflects_holder_count(self):
        """Seasonality non-holder pattern: per-moment :math:`N_j`
        reflects the per-asset holder count.
        """
        # 5 observations; "asset 0" held by all, "asset 1" only by the
        # last two rows.
        df = pd.DataFrame(
            {
                "r0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "r1": [float("nan")] * 3 + [40.0, 50.0],
            }
        )
        meas = EmpiricalMeasure.from_pandas(df)
        # N_j = sum_i d_ij * w_i. With w_i = 1, this is the per-column
        # holder count.
        N_j = jnp.sum(meas.mask * meas.weights[:, None], axis=0)
        assert float(N_j[0]) == pytest.approx(5.0)
        assert float(N_j[1]) == pytest.approx(2.0)


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
