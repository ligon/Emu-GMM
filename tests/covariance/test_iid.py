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


# ---------------------------------------------------------------------------


class TestNaNSafety:
    """The cluster-totals form NaN-cleans masked-out cells via where(...)."""

    def test_nan_in_psi_at_masked_cells_does_not_poison(self):
        """A psi returning NaN at masked-out cells still yields a finite V."""
        # Two-moment psi; moment 1 missing on rows 0 and 1.
        x = jnp.array(
            [
                [1.0, jnp.nan],
                [2.0, jnp.nan],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        meas = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(4))
        V = IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas)
        assert bool(jnp.all(jnp.isfinite(V)))
        # V[0, 0] = sum_i (x_i)^2 / N_0^2 = (1 + 4 + 9 + 16) / 16 = 30 / 16.
        assert float(V[0, 0]) == pytest.approx(30.0 / 16.0, rel=1e-6)
        # V[1, 1] = sum over rows 2-3 of x_i^2 / N_1^2 = (900 + 1600) / 4.
        assert float(V[1, 1]) == pytest.approx((900.0 + 1600.0) / 4.0, rel=1e-6)
        # V[0, 1] uses pairwise overlap (rows 2-3): (3*30 + 4*40) / (4 * 2).
        assert float(V[0, 1]) == pytest.approx(
            (3.0 * 30.0 + 4.0 * 40.0) / (4.0 * 2.0), rel=1e-6
        )


# ---------------------------------------------------------------------------
# The centered= knob (docs/validation/r-reference-crosschecks.org): subtract the
# per-coordinate mean moment before the pairwise-overlap outer products.
# ---------------------------------------------------------------------------
def _masked_centered_reference(X, mask):
    """Independent numpy reference for the masked, centered V_X."""
    N, M = X.shape
    Nj = mask.sum(0)
    m = (mask * X).sum(0) / Nj  # per-coordinate mean over OBSERVED rows
    V = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            num = np.sum(mask[:, j] * mask[:, k] * (X[:, j] - m[j]) * (X[:, k] - m[k]))
            V[j, k] = num / (Nj[j] * Nj[k])
    return V


class TestCenteredKnob:
    def _meas(self, seed=0, n=40, m=3, loc=1.0):
        rng = np.random.default_rng(seed)
        X = rng.normal(loc, 2.0, size=(n, m))  # nonzero mean so centering bites
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.ones((n, m)), weights=jnp.ones(n)
        )
        return X, meas

    def test_static_field_default_false(self):
        assert IIDCovariance().centered is False
        assert IIDCovariance(centered=True).centered is True

    def test_unmasked_matches_hand_computation(self):
        X, meas = self._meas()
        n = X.shape[0]
        vu = np.asarray(IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas))
        vc = np.asarray(
            IIDCovariance(centered=True).covariance(_identity_psi, _P(0.0, 0.0), meas)
        )
        # uncentered: sum_i x x' / N^2 == (X'X)/N^2; centered: biased cov / N.
        np.testing.assert_allclose(vu, (X.T @ X) / n**2, rtol=1e-6)
        np.testing.assert_allclose(
            vc, np.cov(X, rowvar=False, bias=True) / n, rtol=1e-6
        )

    def test_both_forms_are_psd(self):
        _, meas = self._meas()
        for cov in (IIDCovariance(), IIDCovariance(centered=True)):
            V = np.asarray(cov.covariance(_identity_psi, _P(0.0, 0.0), meas))
            assert np.all(np.linalg.eigvalsh(V) >= -1e-12)

    def test_equal_when_moments_already_have_zero_mean(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0.0, 1.0, size=(60, 2))
        X = X - X.mean(axis=0)  # exact zero column means -> centering is a no-op
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.ones((60, 2)), weights=jnp.ones(60)
        )
        vu = np.asarray(IIDCovariance().covariance(_identity_psi, _P(0.0, 0.0), meas))
        vc = np.asarray(
            IIDCovariance(centered=True).covariance(_identity_psi, _P(0.0, 0.0), meas)
        )
        np.testing.assert_allclose(vu, vc, atol=1e-12)

    def test_masked_centering_uses_per_coordinate_observed_mean(self):
        # Partial observability: the centre is the mean over each coordinate's
        # OWN observed rows, and each (j,k) sum is over rows observing both.
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        mask = np.array([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=float)
        meas = EmpiricalMeasure(
            x=jnp.asarray(X), mask=jnp.asarray(mask), weights=jnp.ones(4)
        )
        vc = np.asarray(
            IIDCovariance(centered=True).covariance(_identity_psi, _P(0.0, 0.0), meas)
        )
        np.testing.assert_allclose(vc, _masked_centered_reference(X, mask), rtol=1e-6)
        assert np.all(np.linalg.eigvalsh(vc) >= -1e-12)
