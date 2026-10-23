"""Independent checks of the design-matched K cross-covariance."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import StratifiedCovariance
from emu_gmm.inference import k_statistic
from emu_gmm.inference.k_statistic import (
    _sigma_jm_iid_from_contributions,
    _sigma_jm_stratified_from_contributions,
)
from emu_gmm.measures import AnalyticalMeasure, EmpiricalMeasure


@jdc.pytree_dataclass
class _Theta:
    value: float


def _psi(x, theta):
    y, regressor = x
    residual = y - theta.value * regressor
    return jnp.array([residual, residual * regressor, residual * regressor**2])


def _weak_psi(x, theta):
    outcome, regressor, instrument = x
    residual = outcome - theta.value * regressor
    return residual * jnp.array([1.0, instrument, instrument * instrument - 1.0])


def _oracle(g, D, mask, weights, psu_ids, cell_ids, stratum_ids, fpc):
    """NumPy PSU/cell loops, independent of the augmented JAX engine."""
    n_moments, n_params = D.shape[1:]
    N_j = np.sum(mask * weights[:, None], axis=0)
    out = np.zeros((n_params, n_moments, n_moments))
    psus = np.unique(psu_ids)
    cells = np.unique(cell_ids)
    for cell in cells:
        cell_psus = [
            psu for psu in psus if cell_ids[np.flatnonzero(psu_ids == psu)[0]] == cell
        ]
        if fpc:
            stratum = stratum_ids[np.flatnonzero(cell_ids == cell)[0]]
            stratum_psus = {
                psu
                for psu in psus
                if stratum_ids[np.flatnonzero(psu_ids == psu)[0]] == stratum
            }
            factor = 1.0 - len(cell_psus) / len(stratum_psus)
        else:
            factor = 1.0
        for j in range(n_moments):
            for k in range(n_moments):
                overlap = [
                    psu
                    for psu in cell_psus
                    if np.any(mask[psu_ids == psu, j] > 0)
                    and np.any(mask[psu_ids == psu, k] > 0)
                ]
                H = len(overlap)
                if H < 2 or N_j[j] == 0 or N_j[k] == 0:
                    continue
                G = np.array([np.sum(g[psu_ids == psu, k]) for psu in overlap])
                for a in range(n_params):
                    F = np.array([np.sum(D[psu_ids == psu, j, a]) for psu in overlap])
                    out[a, j, k] += (
                        factor
                        * H
                        / (H - 1)
                        * np.sum((F - F.mean()) * (G - G.mean()))
                        / (N_j[j] * N_j[k])
                    )
    return out


@pytest.mark.parametrize("fpc", [False, True])
def test_unequal_support_weights_and_fpc_match_independent_oracle(fpc):
    mask = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 0.0, 1.5, 0.5, 3.0])
    psu_ids = np.arange(6)
    cell_ids = np.array([0, 0, 0, 1, 1, 1])
    stratum_ids = np.zeros(6, dtype=int)
    raw_g = np.arange(1.0, 19.0).reshape(6, 3) / 7
    raw_D = np.arange(1.0, 37.0).reshape(6, 3, 2) / 11
    g = raw_g * mask * weights[:, None]
    D = raw_D * mask[:, :, None] * weights[:, None, None]
    N_j = np.sum(mask * weights[:, None], axis=0)
    covariance = StratifiedCovariance(
        psu_ids=jnp.asarray(psu_ids),
        cell_ids=jnp.asarray(cell_ids),
        stratum_ids=jnp.asarray(stratum_ids),
        n_psu=6,
        n_cells=2,
        n_strata=1,
        fpc=fpc,
    )
    got = _sigma_jm_stratified_from_contributions(
        jnp.asarray(g),
        jnp.asarray(D),
        jnp.asarray(N_j),
        jnp.asarray(mask),
        covariance,
        _psi,
        _Theta(0.0),
    )
    expected = _oracle(g, D, mask, weights, psu_ids, cell_ids, stratum_ids, fpc)
    np.testing.assert_allclose(got, expected, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(got)[:, 1, 2], 0.0)


def test_one_cell_singleton_psus_reduce_to_centered_sample_cross_covariance():
    rng = np.random.default_rng(141)
    n = 8
    g = rng.normal(size=(n, 3))
    D = rng.normal(size=(n, 3, 2))
    mask = np.ones_like(g)
    covariance = StratifiedCovariance(
        psu_ids=jnp.arange(n, dtype=float),
        cell_ids=jnp.zeros(n),
        stratum_ids=jnp.zeros(n),
        n_psu=n,
        n_cells=1,
        n_strata=1,
    )
    got = _sigma_jm_stratified_from_contributions(
        jnp.asarray(g),
        jnp.asarray(D),
        jnp.full((3,), float(n)),
        jnp.asarray(mask),
        covariance,
        _psi,
        _Theta(0.0),
    )
    expected = (
        np.einsum(
            "nja,nk->ajk",
            D - D.mean(axis=0),
            g - g.mean(axis=0),
        )
        * n
        / (n - 1)
        / n**2
    )
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_single_supported_psu_and_zero_weighted_denominator_give_zero():
    g = jnp.array([[2.0, 0.0], [4.0, 0.0]])
    D = jnp.array([[[3.0], [0.0]], [[5.0], [0.0]]])
    mask = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    covariance = StratifiedCovariance(
        psu_ids=jnp.array([0.0, 1.0]),
        cell_ids=jnp.array([0.0, 1.0]),
        stratum_ids=jnp.array([0.0, 0.0]),
        n_psu=2,
        n_cells=2,
        n_strata=1,
    )
    got = _sigma_jm_stratified_from_contributions(
        g, D, jnp.array([2.0, 0.0]), mask, covariance, _psi, _Theta(0.0)
    )
    np.testing.assert_array_equal(np.asarray(got), 0.0)


def test_stratified_missing_contributions_refuses_even_strong_id_fallback():
    measure = AnalyticalMeasure(
        expectation_fn=lambda _model, theta: jnp.array(
            [theta.value + 0.1, theta.value - 0.1]
        )
    )
    covariance = StratifiedCovariance(
        psu_ids=jnp.zeros(1),
        cell_ids=jnp.zeros(1),
        stratum_ids=jnp.zeros(1),
        n_psu=1,
        n_cells=1,
        n_strata=1,
    )
    with pytest.raises(ValueError, match="requires empirical moment"):
        k_statistic(
            _Theta(0.0),
            measure,
            covariance,
            _psi,
            V=jnp.eye(2),
            strong_id_fallback=True,
        )


def test_public_k_matches_independent_score_covariance_override():
    rng = np.random.default_rng(742)
    n = 24
    x = rng.normal(size=n)
    y = 0.2 * x + rng.normal(size=n)
    mask = rng.choice([0.0, 1.0], size=(n, 3), p=[0.2, 0.8])
    mask[:4, :] = 1.0
    weights = rng.uniform(0.5, 2.0, size=n)
    measure = EmpiricalMeasure(
        x=jnp.asarray(np.column_stack([y, x])),
        mask=jnp.asarray(mask),
        weights=jnp.asarray(weights),
    )
    covariance = StratifiedCovariance(
        psu_ids=jnp.repeat(jnp.arange(12, dtype=float), 2),
        cell_ids=jnp.repeat(jnp.arange(2, dtype=float), 12),
        stratum_ids=jnp.zeros(n),
        n_psu=12,
        n_cells=2,
        n_strata=1,
    )
    theta = _Theta(0.0)
    g = np.asarray(measure.moment_contributions(_psi, theta))
    D = np.asarray(measure.jacobian_contributions(_psi, theta))
    sigma = _oracle(
        g,
        D,
        mask,
        weights,
        np.repeat(np.arange(12), 2),
        np.repeat(np.arange(2), 12),
        np.zeros(n, dtype=int),
        False,
    )
    auto = k_statistic(theta, measure, covariance, _psi)
    explicit = k_statistic(
        theta,
        measure,
        covariance,
        _psi,
        score_cov_fn=lambda _model, _theta: jnp.asarray(sigma),
    )
    np.testing.assert_allclose(auto.K, explicit.K, atol=1e-12)
    np.testing.assert_allclose(auto.S, explicit.S, atol=1e-12)


def test_fixed_cell_weak_null_calibration_discriminates_iid_cross_covariance():
    """Seeded fixed-cell DGP: matching the cross sandwich calibrates K.

    Cell means of both the outcome and regressor differ. The cell counts are
    fixed, so those means cancel from the null moment but not from a wrong
    uncentered IID derivative/moment cross covariance. Identification is
    weak (population mean derivative is proportional to 0.02).
    """
    n_rep, n_obs = 600, 120
    rng = np.random.default_rng(141)
    innovation = rng.standard_normal((n_rep, n_obs))
    instrument = rng.standard_normal((n_rep, n_obs))
    cell_sign = np.r_[np.ones(n_obs // 2), -np.ones(n_obs // 2)]
    outcome = innovation + 2.0 * cell_sign
    regressor = 0.02 + innovation * instrument + 1.5 * cell_sign
    data = jnp.asarray(np.stack([outcome, regressor, instrument], axis=-1))
    covariance = StratifiedCovariance(
        psu_ids=jnp.arange(n_obs, dtype=float),
        cell_ids=jnp.repeat(jnp.arange(2, dtype=float), n_obs // 2),
        stratum_ids=jnp.zeros(n_obs),
        n_psu=n_obs,
        n_cells=2,
        n_strata=1,
    )
    mask = jnp.ones((n_obs, 3))
    weights = jnp.ones(n_obs)
    theta = _Theta(0.0)

    def one_rep(x):
        measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        matched = k_statistic(theta, measure, covariance, _weak_psi)
        g = measure.moment_contributions(_weak_psi, theta)
        D = measure.jacobian_contributions(_weak_psi, theta)
        wrong = _sigma_jm_iid_from_contributions(g, D, jnp.full((3,), float(n_obs)))
        mismatched = k_statistic(
            theta,
            measure,
            covariance,
            _weak_psi,
            score_cov_fn=lambda _model, _theta: wrong,
        )
        return jnp.array([matched.p_K, mismatched.p_K])

    p_values = np.asarray(jax.jit(jax.vmap(one_rep))(data))
    matched_mean = float(np.mean(p_values[:, 0]))
    mismatched_mean = float(np.mean(p_values[:, 1]))
    assert matched_mean == pytest.approx(0.5, abs=0.035)
    assert matched_mean > mismatched_mean + 0.04
    assert 20 <= np.count_nonzero(p_values[:, 0] < 0.05) <= 42
