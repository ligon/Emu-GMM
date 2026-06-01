r"""Phase-5 acceptance: components() readout + manifold-aware result path.

Drives a real ``estimate()`` on ``Product(PSDFixedRank(5, K), Euclidean(1))``
and checks the Phase-5 contract (manifold epic #12):

* ``result.theta.components()`` / ``result.components()`` return the per-leaf
  tuple ``(A, phi)`` in field order with ``A.shape == (5, K)``; the caller
  computes ``Gamma_hat = A @ A.T`` and ``eigvalsh(Gamma_hat)``;
* ``coef_table`` and ``standard_errors`` return WITHOUT raising on the
  non-scalar result; the coef_table / ``to_pandas`` Sigma_theta axes carry
  POSITIONAL tangent labels (``Y[0,0]`` ... ``phi[0]``), NOT scalar field
  names (INT-12/R5);
* a warm start reading ``prev.components()`` round-trips into a new
  ``estimate()`` (same ``Gamma_hat``).

The v1 bitwise non-regression for scalar / Positive trees is checked in
``test_estimator_result_phase5_v1.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side


@jdc.pytree_dataclass
class ProductParams:
    """A PSDFixedRank ``Y`` leaf plus a Euclidean(1) ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k):
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _moment_count(k):
    return N * (N + 1) // 2 + 1


_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)


def _model(x, theta):
    Y = theta.Y.array
    phi = theta.phi.array[0]
    Gamma = Y @ Y.T
    g = Gamma[_TRIU[:, 0], _TRIU[:, 1]]
    model_m = jnp.concatenate([g, jnp.reshape(phi, (1,))])
    return model_m - x


def _run_estimate(k, seed=0, theta_init=None):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    M = _moment_count(k)
    assert int(target.shape[0]) == M

    n_sim = 200

    def sampler(key, theta):
        noise = 0.1 * jax.random.normal(key, (n_sim, M))
        return target[None, :] + noise

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)

    if theta_init is None:
        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
        theta_init = _make_params(Y0, 0.65, k)

    result = estimate(
        _model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=400),
        theta_init=theta_init,
    )
    return result, M, k, A_true, phi_true


@pytest.mark.parametrize("k", [2, 3])
class TestComponentsReadout:
    def test_components_shapes_and_field_order(self, k):
        result, M, _, _, _ = _run_estimate(k, seed=10 + k)
        # Both accessor forms must agree and return (A, phi) in field order.
        comps_method = result.components()
        comps_theta = result.theta.components()
        assert len(comps_method) == 2
        assert len(comps_theta) == 2
        A, phi = comps_method
        assert A.shape == (N, k)
        assert int(np.prod(phi.shape)) == 1  # scalar (stored as (1,))
        # field order: Y first, phi second
        A2, phi2 = comps_theta
        assert jnp.allclose(A, A2)
        assert jnp.allclose(phi, phi2)

    def test_caller_computes_gamma_and_eigvals(self, k):
        result, M, _, A_true, _ = _run_estimate(k, seed=20 + k)
        A, phi = result.components()
        Gamma_hat = A @ A.T
        # Gauge-invariant recovery of Gamma (the caller's structural readout).
        # Phase 5 only asserts components() yields a *usable* Gamma_hat near
        # the truth; the tight recovery gate is Phase 6. The synthetic DGP
        # carries 0.1 i.i.d. moment noise so a loose atol is correct here.
        Gamma_true = A_true @ A_true.T
        assert jnp.allclose(Gamma_hat, Gamma_true, atol=5e-2)
        evals = jnp.linalg.eigvalsh(Gamma_hat)
        # rank-k PSD: smallest n-k eigenvalues near zero, k positive.
        assert int(jnp.sum(evals > 1e-6)) == k

    def test_components_identity_stable(self, k):
        result, _, _, _, _ = _run_estimate(k, seed=25 + k)
        a = result.components()
        b = result.components()
        # Pure: same arrays returned (value identity).
        for x, y in zip(a, b, strict=True):
            assert jnp.allclose(x, y)


@pytest.mark.parametrize("k", [2, 3])
class TestResultPathNoRaise:
    def test_coef_table_no_raise_positional_labels(self, k):
        result, M, _, _, _ = _run_estimate(k, seed=30 + k)
        D = N * k + 1
        tab = result.coef_table  # must NOT raise
        assert len(tab) == D
        assert list(tab.columns) == ["estimate", "std_error", "t_stat", "p_value"]
        # Positional tangent labels, NOT scalar field-names.
        idx = list(tab.index)
        assert idx != ["Y", "phi"]
        assert "Y[0,0]" in idx
        assert "phi[0]" in idx
        # The estimate column == the ambient flatten the SE axis uses.
        from emu_gmm._internal.params import flatten_params_with_spec

        flat, _, _ = flatten_params_with_spec(result.theta_hat)
        assert jnp.allclose(jnp.asarray(tab["estimate"].to_numpy()), flat)

    def test_standard_errors_no_raise_and_sized_total_dimension(self, k):
        result, M, _, _, _ = _run_estimate(k, seed=40 + k)
        D = N * k + 1
        se = result.standard_errors  # must NOT raise
        assert int(se.array.shape[0]) == D
        assert int(se.array.shape[0]) == result.Sigma_theta.array.shape[0]

    def test_to_pandas_sigma_positional_labels(self, k):
        result, M, _, _, _ = _run_estimate(k, seed=50 + k)
        D = N * k + 1
        out = result.to_pandas()  # must NOT raise
        sigma = out["Sigma_theta"]
        assert sigma.shape == (D, D)
        assert list(sigma.index) == list(sigma.columns)
        assert "Y[0,0]" in list(sigma.index)
        coeffs = out["coefficients"]
        assert len(coeffs) == D
        assert not coeffs.index.isnull().any()


class TestWarmStart:
    def test_components_round_trip_seeds_new_estimate(self):
        k = 2
        result, M, _, A_true, phi_true = _run_estimate(k, seed=101)
        A, phi = result.components()
        Gamma_first = A @ A.T

        # Seed a NEW estimate from prev.components() (the warm-start path).
        theta_init2 = _make_params(A, float(jnp.reshape(phi, ())), k)
        result2, _, _, _, _ = _run_estimate(k, seed=101, theta_init=theta_init2)
        A2, phi2 = result2.components()
        Gamma_second = A2 @ A2.T

        # Gauge-invariant round-trip: same Gamma_hat (raw Y differs by O(k)).
        assert jnp.allclose(Gamma_first, Gamma_second, atol=1e-3)
        assert jnp.allclose(
            float(jnp.reshape(phi, ())), float(jnp.reshape(phi2, ())), atol=1e-3
        )

    def test_theta_property_components_match_method(self):
        result, _, _, _, _ = _run_estimate(2, seed=102)
        a = result.theta.components()
        b = result.components()
        for x, y in zip(a, b, strict=True):
            assert jnp.allclose(x, y)

    def test_theta_hat_pytree_unchanged(self):
        # R19: result.theta_hat is still the raw user dataclass, not wrapped.
        result, _, _, _, _ = _run_estimate(2, seed=103)
        assert isinstance(result.theta_hat, ProductParams)
        assert isinstance(result.theta_hat.Y, ManifoldLeaf)
        assert result.theta_hat.Y.array.shape == (N, 2)
