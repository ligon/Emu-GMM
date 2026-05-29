"""Tests for emu_gmm.covariance.iid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance.iid import IIDCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.types import CovarianceStrategy


@jdc.pytree_dataclass
class _P:
    a: float
    b: float


def _identity_psi(x, theta):
    """psi(x, theta) = x: return the observation unchanged."""
    return x


def _two_moment_psi(x, theta):
    """psi(x, theta) = [a + x[0], b * x[1]]."""
    return jnp.array([theta.a + x[0], theta.b * x[1]])


# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_covariance_protocol(self):
        cov = IIDCovariance()
        assert isinstance(cov, CovarianceStrategy)


# ---------------------------------------------------------------------------


class TestShapeAndSymmetry:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((50, 2)),
            weights=jnp.ones(50),
        )
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert V.shape == (2, 2)

    def test_symmetric(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 3))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((50, 3)),
            weights=jnp.ones(50),
        )
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert jnp.allclose(V, V.T, atol=1e-7)


# ---------------------------------------------------------------------------


class TestFormula:
    def test_full_mask_uniform_weights_matches_outer_product_mean(self):
        """With all-ones mask and unit weights, V_X = (1 / N^2) sum_i psi_i psi_i'.

        This is the framework's convention (variance of the moment
        estimator, not of an individual draw).
        """
        N = 5
        psi_vals = jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [-1.0, 0.5],
                [0.5, -1.0],
                [2.0, 1.0],
            ]
        )
        # Construct a measure whose psi(x_i, theta) returns psi_vals[i].
        x = psi_vals  # _identity_psi(x) returns x
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 2)),
            weights=jnp.ones(N),
        )
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        # Direct formula: (1/(N*N)) sum_i psi_i psi_i'.
        expected = jnp.einsum("ij,ik->jk", x, x) / (N * N)
        assert jnp.allclose(V, expected, atol=1e-7)

    def test_pairwise_overlap_strict_pattern(self):
        """On a 3-moment, 6-observation toy with moment 2 missing on the
        bottom half, V[0,1] uses all 6 rows but V[0,2] and V[1,2] use
        only the top 3.
        """
        N = 6
        # Choose psi values explicitly so we can predict the result.
        psi_vals = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
            ]
        )
        # Moment 2 only observable on rows 0-2.
        mask = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        weights = jnp.ones(N)
        meas = EmpiricalMeasure(x=psi_vals, mask=mask, weights=weights)
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)

        # N_j: N_0 = N_1 = 6, N_2 = 3.
        # V[0,0] = sum_{i in 0..5} (psi_0)^2 / 36 = (1+4+9+16+25+36)/36
        v_00_expected = (1 + 4 + 9 + 16 + 25 + 36) / 36.0
        assert float(V[0, 0]) == pytest.approx(v_00_expected, rel=1e-6)
        # V[0,1] same since psi_0 = psi_1.
        assert float(V[0, 1]) == pytest.approx(v_00_expected, rel=1e-6)
        # V[0,2] = sum_{i in 0..2} psi_0 * psi_2 / (N_0 * N_2) = (1+4+9)/(6*3)
        v_02_expected = (1 + 4 + 9) / (6.0 * 3.0)
        assert float(V[0, 2]) == pytest.approx(v_02_expected, rel=1e-6)
        # V[1,2] same.
        assert float(V[1, 2]) == pytest.approx(v_02_expected, rel=1e-6)
        # V[2,2] = sum_{i in 0..2} psi_2^2 / (N_2^2) = (1+4+9)/9.
        v_22_expected = (1 + 4 + 9) / 9.0
        assert float(V[2, 2]) == pytest.approx(v_22_expected, rel=1e-6)
        # Symmetry check across the missingness boundary.
        assert jnp.allclose(V, V.T, atol=1e-7)

    def test_weights_squared(self):
        """The weights enter the numerator as w_i^2 per the formula."""
        N = 3
        psi_vals = jnp.array([[1.0], [1.0], [1.0]])  # constant
        x = psi_vals
        # Weights 1, 2, 3.
        weights = jnp.array([1.0, 2.0, 3.0])
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((N, 1)),
            weights=weights,
        )
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        # N_0 = 1 + 2 + 3 = 6. Numerator = sum w_i^2 * psi_i^2 = 1+4+9 = 14.
        # V[0,0] = 14 / 36.
        assert float(V[0, 0]) == pytest.approx(14.0 / 36.0, rel=1e-6)


# ---------------------------------------------------------------------------


class TestJit:
    def test_covariance_jits(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((50, 2)),
            weights=jnp.ones(50),
        )
        cov = IIDCovariance()
        theta = _P(0.0, 0.0)

        @jax.jit
        def compute(c, t, m):
            return c.covariance(_identity_psi, t, m)

        V_eager = cov.covariance(_identity_psi, theta, meas)
        V_jit = compute(cov, theta, meas)
        assert jnp.allclose(V_eager, V_jit, atol=1e-7)


# ---------------------------------------------------------------------------


class TestUseWithStructuralModel:
    """Smoke test combining IIDCovariance with a non-trivial psi."""

    def test_two_moment_psi_runs(self):
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (40, 2))
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((40, 2)),
            weights=jnp.ones(40),
        )
        cov = IIDCovariance()
        theta = _P(a=0.0, b=1.0)
        V = cov.covariance(_two_moment_psi, theta, meas)
        assert V.shape == (2, 2)
        assert jnp.all(jnp.isfinite(V))
        # PSD up to numerical tolerance.
        eigs = jnp.linalg.eigvalsh(V)
        assert float(jnp.min(eigs)) >= -1e-9

    def test_numpy_reference_full_mask(self):
        """End-to-end cross-check against a direct numpy reference."""
        rng = np.random.default_rng(123)
        x_np = rng.standard_normal((30, 2))
        x = jnp.asarray(x_np)
        meas = EmpiricalMeasure(
            x=x,
            mask=jnp.ones((30, 2)),
            weights=jnp.ones(30),
        )
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        N = 30
        expected = x_np.T @ x_np / (N * N)
        np.testing.assert_allclose(np.asarray(V), expected, atol=1e-7)


class TestNanMaskSemantics:
    """NaN at mask=0 positions does not poison the covariance.

    Mirrors the empirical-measure NaN handling (see issue #1 port pattern).
    """

    def test_nan_in_x_at_masked_row_does_not_poison(self):
        # 5 rows; row 2 has NaN in column 0 and is masked out.
        x = jnp.array(
            [
                [1.0, 0.5],
                [2.0, 0.6],
                [float("nan"), 0.7],
                [4.0, 0.8],
                [5.0, 0.9],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        weights = jnp.ones(5)
        meas = EmpiricalMeasure(x=x, mask=mask, weights=weights)

        def psi(xi, theta):
            return jnp.array([xi[0], xi[1]])

        V = IIDCovariance().covariance(psi, _P(0.0, 0.0), meas)
        assert jnp.all(jnp.isfinite(V))
