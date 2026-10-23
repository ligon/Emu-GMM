"""The array entry point shares Kleibergen algebra and flags numeric failures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import k_statistic_from_arrays
from emu_gmm.covariance import AnalyticalCovariance
from emu_gmm.inference import KStatisticResult, k_statistic
from emu_gmm.measures import AnalyticalMeasure


def _arrays():
    m = jnp.array([0.4, -0.2, 0.3])
    G = jnp.array([[0.8], [0.1], [0.6]])
    V = jnp.array([[0.5, 0.1, 0.03], [0.1, 0.8, 0.02], [0.03, 0.02, 0.3]])
    sigma = jnp.array([[[0.1, 0.03, -0.02], [0.07, 0.1, 0.04], [-0.03, 0.02, 0.08]]])
    return m, G, V, sigma


def _call(m=None, G=None, V=None, sigma=None, **kwargs):
    defaults = _arrays()
    return k_statistic_from_arrays(
        defaults[0] if m is None else m,
        defaults[1] if G is None else G,
        defaults[2] if V is None else V,
        score_covariance=defaults[3] if sigma is None else sigma,
        **kwargs,
    )


def _assert_equal(left, right):
    for field in ("K", "S", "J", "p_K", "p_S", "p_J"):
        np.testing.assert_allclose(
            getattr(left, field), getattr(right, field), rtol=1e-12, atol=1e-14
        )
    assert (left.df_K, left.df_S, left.df_J) == (
        right.df_K,
        right.df_S,
        right.df_J,
    )


@jdc.pytree_dataclass
class _Theta:
    beta: float


class TestSharedKleibergenAlgebra:
    def test_exports_and_nonzero_correction_match_high_level(self):
        import emu_gmm
        import emu_gmm.inference as inference

        assert inference.k_statistic_from_arrays is k_statistic_from_arrays
        assert "k_statistic_from_arrays" in emu_gmm.__all__
        assert "k_statistic_from_arrays" in inference.__all__
        m, G, V, sigma = _arrays()
        measure = AnalyticalMeasure(
            expectation_fn=lambda model, theta: m + G[:, 0] * theta.beta
        )

        def unexpected_covariance(*args):
            raise AssertionError("The supplied V must bypass covariance construction.")

        high_level = k_statistic(
            _Theta(beta=0.0),
            measure,
            AnalyticalCovariance(covariance_fn=unexpected_covariance),
            lambda x, theta: jnp.zeros(3),
            V=V,
            score_cov_fn=lambda model, theta: sigma,
        )
        actual = _call()
        _assert_equal(actual, high_level)
        assert high_level.invalid_input is None
        assert not bool(actual.invalid_input)
        raw_g = _call(sigma=jnp.zeros_like(sigma))
        assert abs(float(actual.K - raw_g.K)) > 1e-3

        # Independent projection fixes the tensor orientation and mean units.
        d_tilde = np.asarray(G) - np.einsum(
            "pmk,k->mp", np.asarray(sigma), np.linalg.solve(V, m)
        )
        L = np.linalg.cholesky(V)
        m_w = np.linalg.solve(L, m)
        U, _, _ = np.linalg.svd(np.linalg.solve(L, d_tilde), full_matrices=False)
        expected_K = np.sum((U.T @ m_w) ** 2)
        np.testing.assert_allclose(actual.K, expected_K, rtol=1e-13)
        np.testing.assert_allclose(actual.K + actual.S, actual.J, rtol=1e-14)

    def test_unequal_coordinate_normalization_transforms_every_tensor(self):
        m, G, V, sigma = _arrays()
        assert not np.array_equal(sigma[0], sigma[0].T)
        # Different coordinate denominators require pairwise covariance scaling.
        H = jnp.diag(1.0 / jnp.array([3.0, 70.0, 1000.0]))
        transformed = k_statistic_from_arrays(
            H @ m,
            H @ G,
            H @ V @ H.T,
            score_covariance=jax.vmap(lambda C: H @ C @ H.T)(sigma),
        )
        _assert_equal(transformed, _call())
        assert not bool(transformed.invalid_input)

    def test_quotient_dimension_allows_fewer_moments_than_ambient_parameters(self):
        m = jnp.array([0.2, -0.3])
        G = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        V = jnp.array([[0.4, 0.1], [0.1, 0.7]])
        sigma = jnp.stack([0.1 * V, 0.2 * V, 0.1 * V])
        result = k_statistic_from_arrays(
            m, G, V, score_covariance=sigma, gauge_nullspace_dim=1
        )
        assert (result.df_K, result.df_S, result.df_J) == (2, 0, 2)
        np.testing.assert_allclose(result.K, m @ jnp.linalg.solve(V, m), rtol=1e-13)
        assert abs(float(result.S)) < 1e-28
        assert np.isnan(result.p_S)
        assert not bool(result.invalid_input)

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
    def test_real_input_precisions_use_supported_linear_algebra(self, dtype):
        m, G, V, sigma = (a.astype(dtype) for a in _arrays())
        result = k_statistic_from_arrays(m, G, V, score_covariance=sigma)
        assert not bool(result.invalid_input)
        assert np.isfinite(result.K)
        assert result.K.dtype == jnp.promote_types(dtype, jnp.float32)

    def test_jit_and_vmap_preserve_all_result_fields(self):
        m, G, V, sigma = _arrays()

        def kernel(moment, covariance, cross):
            return k_statistic_from_arrays(
                moment, G, covariance, score_covariance=cross
            )

        eager = kernel(m, V, sigma)
        compiled = jax.jit(kernel)(m, V, sigma)
        _assert_equal(eager, compiled)
        assert not bool(compiled.invalid_input)
        covariances = jnp.stack([V, -V, V])
        moments = jnp.stack([m, m, m.at[1].set(jnp.nan)])
        crosses = jnp.stack([sigma] * 3)
        batch = jax.jit(jax.vmap(kernel))(moments, covariances, crosses)
        np.testing.assert_array_equal(batch.invalid_input, [False, True, True])
        np.testing.assert_allclose(batch.K[0], eager.K, rtol=1e-13)
        assert np.isnan(batch.K[1:]).all()
        assert np.isnan(batch.p_J[1:]).all()
        assert batch.df_K == 1


class TestStructuralRefusals:
    def test_cross_covariance_is_mandatory(self):
        m, G, V, _ = _arrays()
        with pytest.raises(TypeError, match="score_covariance"):
            k_statistic_from_arrays(m, G, V)
        with pytest.raises(ValueError, match="score_covariance is required"):
            k_statistic_from_arrays(m, G, V, score_covariance=None)

    @pytest.mark.parametrize(
        "updates",
        [
            {"m": jnp.ones((3, 1))},
            {"G": jnp.ones(3)},
            {"m": jnp.ones(2)},
            {"V": jnp.eye(2)},
            {"sigma": jnp.zeros((3, 3))},
            {"sigma": jnp.zeros((2, 3, 3))},
            {"G": jnp.zeros((3, 0))},
        ],
    )
    def test_bad_shapes_raise(self, updates):
        with pytest.raises(ValueError):
            _call(**updates)

    @pytest.mark.parametrize("count", [True, False, 0.0, jnp.array(0)])
    def test_count_must_be_static_integer(self, count):
        with pytest.raises(TypeError, match="static integer"):
            _call(gauge_nullspace_dim=count)

    @pytest.mark.parametrize("count", [-1, 2])
    def test_count_outside_parameter_dimension_raises(self, count):
        with pytest.raises(ValueError, match="must lie"):
            _call(gauge_nullspace_dim=count)

    def test_underidentified_raises(self):
        with pytest.raises(ValueError, match="under-identified"):
            _call(G=jnp.ones((3, 4)), sigma=jnp.zeros((4, 3, 3)))


class TestNumericalFailureEvents:
    @pytest.mark.parametrize("scale", [1e20, 1e300])
    def test_asymmetry_refusal_is_invariant_to_unequal_moment_units(self, scale):
        root_scale = np.sqrt(scale)
        m = jnp.array([root_scale, 1 / root_scale])
        G = m[:, None]
        V = jnp.array([[scale, 1.0], [0.0, 1 / scale]])
        sigma = jnp.zeros((1, 2, 2))
        H = jnp.diag(jnp.array([1 / root_scale, root_scale]))
        for call in (k_statistic_from_arrays, jax.jit(k_statistic_from_arrays)):
            original = call(m, G, V, score_covariance=sigma)
            standardized = call(H @ m, H @ G, H @ V @ H.T, score_covariance=sigma)
            for result in (original, standardized):
                assert bool(result.invalid_input)
                assert np.isnan(result.J)

    def test_huge_marginal_scales_do_not_overflow_symmetry_tolerance(self):
        m = jnp.full(2, 1e150)
        G = m[:, None]
        V = 1e300 * jnp.eye(2)
        sigma = jnp.zeros((1, 2, 2))
        assert not bool(
            k_statistic_from_arrays(m, G, V, score_covariance=sigma).invalid_input
        )
        # sqrt(V_jj * V_kk) would overflow, incorrectly accepting this mismatch.
        asymmetric = V.at[0, 1].set(1e295)
        result = k_statistic_from_arrays(m, G, asymmetric, score_covariance=sigma)
        assert bool(result.invalid_input)
        assert np.isnan(result.J)

    @pytest.mark.parametrize(
        "updates",
        [
            {"m": jnp.array([jnp.nan, 0.0, 0.0])},
            {"G": jnp.full((3, 1), jnp.inf)},
            {"sigma": jnp.full((1, 3, 3), jnp.nan)},
            {"V": jnp.full((3, 3), jnp.inf)},
            {"V": -jnp.eye(3)},
            {"V": jnp.ones((3, 3))},
            # A lower-triangle-only factorization must not accept this.
            {"V": jnp.eye(3).at[0, 1].set(10.0)},
            # No absolute symmetry tolerance that hides small-scale asymmetry.
            {"V": (1e-18 * jnp.eye(3)).at[0, 1].set(1e-22)},
            {"m": jnp.array([1.0j, 0.0, 0.0])},
            {"G": jnp.ones((3, 1), dtype=jnp.complex128)},
            {"V": jnp.eye(3, dtype=jnp.complex128)},
            {"sigma": jnp.ones((1, 3, 3), dtype=jnp.complex128)},
            {"m": jnp.array([True, False, True])},
            # Finite inputs can overflow the statistic; flag that event too.
            {"m": jnp.full(3, 1e300), "sigma": jnp.zeros((1, 3, 3))},
        ],
    )
    def test_invalid_numeric_data_are_flagged_and_nan(self, updates):
        result = _call(**updates)
        assert bool(result.invalid_input)
        assert result.invalid_input.shape == ()
        for field in ("K", "S", "J", "p_K", "p_S", "p_J"):
            assert np.isnan(getattr(result, field))

    def test_roundoff_symmetry_is_accepted(self):
        _, _, V, _ = _arrays()
        V = V.at[0, 1].add(jnp.finfo(V.dtype).eps)
        result = _call(V=V)
        assert not bool(result.invalid_input)
        assert np.isfinite(result.K)

    def test_legacy_result_constructor_remains_valid(self):
        result = _call()
        legacy = KStatisticResult(
            result.K,
            result.S,
            result.J,
            result.p_K,
            result.p_S,
            result.p_J,
            result.df_K,
            result.df_S,
            result.df_J,
        )
        assert legacy.invalid_input is None
