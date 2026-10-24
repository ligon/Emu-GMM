"""Independent mixed derivative/moment covariance checks (ledger Section 19)."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AnalyticalMeasure,
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
    k_statistic,
)
from emu_gmm.inference.k_statistic import (
    _sigma_jm_design_aware_from_contributions,
    _sigma_jm_stratified_from_contributions,
)


@jdc.pytree_dataclass
class _Theta:
    a: float
    b: float


def _model(x, theta):
    return x[:3] - theta.a * x[3:6] - theta.b * x[6:9]


def _fixture(*, fixed=True, centered=False, dof=False, fpc=False, couple=True):
    rng = np.random.default_rng(14119)
    n = 40
    strata = np.repeat(np.arange(2), 20)
    bundles = np.repeat(np.arange(10), 4)
    cell = 2 * strata + np.tile(np.repeat(np.arange(2), 2), 10)
    psu = np.repeat(np.arange(20), 2)
    x = rng.normal(size=(n, 9)) + (strata[:, None] - 0.5) * 3.0
    mask = rng.choice([0.0, 0.5, 1.0], size=(n, 3), p=[0.2, 0.3, 0.5])
    mask[::4] = 1.0
    weights = rng.uniform(0.3, 2.0, n)
    weights[1] = 0.0  # Observed zero-weight support must remain observed.
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.asarray(weights)
    )
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(psu),
        cell_ids=jnp.asarray(cell),
        stratum_ids=jnp.asarray(strata),
        n_psu=20,
        n_cells=4,
        n_strata=2,
        fpc=fpc,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(bundles),
        n_clusters=10,
        centered=centered,
        dof_correction=dof,
    )
    covariance = DesignAwareCovariance.from_design_mask(
        design,
        sampling,
        jnp.asarray([1.0, 0.0, 1.0]),
        fixed_strata=fixed,
        couple=couple,
    )
    return measure, covariance


def _contributions(measure):
    theta = _Theta(0.1, -0.2)
    g = np.asarray(measure.moment_contributions(_model, theta))
    D = np.asarray(measure.jacobian_contributions(_model, theta))
    N = np.sum(np.asarray(measure.mask) * np.asarray(measure.weights)[:, None], axis=0)
    return g, D, N


def _oracle(g, D, measure, covariance):
    """Original-coordinate NumPy loops; no augmented covariance invocation."""
    mask, weights = np.asarray(measure.mask), np.asarray(measure.weights)
    e = mask * weights[:, None]
    N = e.sum(axis=0)
    psu = np.asarray(covariance.design.psu_ids)
    cell = np.asarray(covariance.design.cell_ids)
    strata = np.asarray(covariance.design.stratum_ids)
    bundle = np.asarray(covariance.sampling.cluster_ids)
    t = np.asarray(covariance.design_moment_mask) > 0
    out = np.zeros((D.shape[2], g.shape[1], g.shape[1]))
    for a in range(D.shape[2]):
        for j in range(g.shape[1]):
            for k in range(g.shape[1]):
                if N[j] == 0 or N[k] == 0:
                    continue
                total = 0.0
                if t[j] and t[k]:
                    for c in np.unique(cell):
                        ids = [
                            p
                            for p in np.unique(psu[cell == c])
                            if np.any(mask[psu == p, j] > 0)
                            and np.any(mask[psu == p, k] > 0)
                        ]
                        H = len(ids)
                        if H < 2:
                            continue
                        F = np.array([D[psu == p, j, a].sum() for p in ids])
                        G = np.array([g[psu == p, k].sum() for p in ids])
                        factor = 1.0
                        if covariance.design.fpc:
                            s = strata[np.flatnonzero(cell == c)[0]]
                            factor -= len(np.unique(psu[cell == c])) / len(
                                np.unique(psu[strata == s])
                            )
                        total += (
                            factor * H / (H - 1) * ((F - F.mean()) @ (G - G.mean()))
                        )
                elif t[j] != t[k] and not covariance.couple:
                    continue
                elif covariance.fixed_strata:
                    for s in np.unique(strata):
                        cells = np.unique(cell[strata == s])
                        for c in cells:
                            for d in cells:
                                ids = [
                                    h
                                    for h in np.unique(bundle[strata == s])
                                    if np.any(mask[(bundle == h) & (cell == c), j] > 0)
                                    and np.any(mask[(bundle == h) & (cell == d), k] > 0)
                                ]
                                H = len(ids)
                                if H < 2:
                                    continue
                                ej = np.array(
                                    [
                                        e[(bundle == h) & (cell == c), j].sum()
                                        for h in ids
                                    ]
                                )
                                ek = np.array(
                                    [
                                        e[(bundle == h) & (cell == d), k].sum()
                                        for h in ids
                                    ]
                                )
                                if ej @ ej == 0 or ek @ ek == 0:
                                    continue
                                Pj = np.eye(H) - np.outer(ej, ej) / (ej @ ej)
                                Pk = np.eye(H) - np.outer(ek, ek) / (ek @ ek)
                                trace = np.trace(Pj @ Pk)
                                if trace <= np.sqrt(np.finfo(float).eps) * H:
                                    continue
                                F = np.array(
                                    [
                                        D[(bundle == h) & (cell == c), j, a].sum()
                                        for h in ids
                                    ]
                                )
                                G = np.array(
                                    [
                                        g[(bundle == h) & (cell == d), k].sum()
                                        for h in ids
                                    ]
                                )
                                total += H / trace * ((Pj @ F) @ (Pk @ G))
                else:
                    F, G = [], []
                    for h in np.unique(bundle):
                        selected = bundle == h
                        f, v = D[selected, j, a].sum(), g[selected, k].sum()
                        if not t[j] and not t[k] and covariance.sampling.centered:
                            f -= e[selected, j].sum() * D[:, j, a].sum() / N[j]
                            v -= e[selected, k].sum() * g[:, k].sum() / N[k]
                        F.append(f)
                        G.append(v)
                    total = np.array(F) @ np.array(G)
                    if covariance.sampling.dof_correction:
                        H = sum(
                            np.any(mask[bundle == h, j] > 0)
                            and np.any(mask[bundle == h, k] > 0)
                            for h in np.unique(bundle)
                        )
                        if H >= 2:
                            total *= H / (H - 1)
                out[a, j, k] = total / (N[j] * N[k])
    return out


def _automatic(g, D, N, measure, covariance):
    return _sigma_jm_design_aware_from_contributions(
        jnp.asarray(g),
        jnp.asarray(D),
        jnp.asarray(N),
        measure.mask,
        measure.weights,
        covariance,
        _model,
        _Theta(0.1, -0.2),
    )


@pytest.mark.parametrize("fixed,dof", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("fpc,couple", [(False, True), (True, True), (True, False)])
def test_original_coordinate_oracle_matches_all_mixed_modes(
    fixed, dof, centered, fpc, couple
):
    measure, covariance = _fixture(
        fixed=fixed, dof=dof, centered=centered, fpc=fpc, couple=couple
    )
    g, D, N = _contributions(measure)
    got = np.asarray(_automatic(g, D, N, measure, covariance))
    np.testing.assert_allclose(
        got, _oracle(g, D, measure, covariance), atol=2e-13, rtol=2e-13
    )
    assert not np.allclose(got, got.transpose(0, 2, 1))


@pytest.mark.parametrize("fixed", [False, True])
def test_all_design_is_bitwise_pure_and_all_sampling_matches_oracle(fixed):
    measure, covariance = _fixture(fixed=fixed, centered=True)
    g, D, N = _contributions(measure)
    pure = _sigma_jm_stratified_from_contributions(
        jnp.asarray(g),
        jnp.asarray(D),
        jnp.asarray(N),
        measure.mask,
        covariance.design,
        _model,
        _Theta(0.1, -0.2),
    )
    all_design = replace(covariance, design_moment_mask=jnp.ones(3), all_design=True)
    np.testing.assert_array_equal(_automatic(g, D, N, measure, all_design), pure)
    all_sampling = replace(
        covariance, design_moment_mask=jnp.zeros(3), all_design=False
    )
    np.testing.assert_allclose(
        _automatic(g, D, N, measure, all_sampling),
        _oracle(g, D, measure, all_sampling),
        atol=2e-13,
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_public_k_oracle_and_jit_vmap_match(fixed):
    measure, covariance = _fixture(fixed=fixed, dof=not fixed, centered=True)
    theta = _Theta(0.1, -0.2)
    g, D, N = _contributions(measure)
    sigma = _oracle(g, D, measure, covariance)
    automatic = k_statistic(theta, measure, covariance, _model)
    explicit = k_statistic(
        theta, measure, covariance, _model, score_cov_fn=lambda *_: jnp.asarray(sigma)
    )
    np.testing.assert_allclose(automatic.K, explicit.K, atol=1e-12)
    np.testing.assert_allclose(automatic.S, explicit.S, atol=1e-12)
    run = jax.jit(
        jax.vmap(
            lambda x, cov: k_statistic(theta, replace(measure, x=x), cov, _model).K,
            in_axes=(0, None),
        )
    )
    got = run(jnp.stack([measure.x, measure.x]), covariance)
    np.testing.assert_allclose(got, jnp.repeat(automatic.K, 2), atol=1e-12)


@pytest.mark.parametrize("fixed", [False, True])
def test_sparse_support_zero_denominator_and_zero_exposure(fixed):
    measure, covariance = _fixture(fixed=fixed)
    mask = np.asarray(measure.mask).copy()
    mask[:, 1:] = 0.0
    mask[0, 1] = 1.0  # One supporting PSU/bundle; fixed kernel contributes zero.
    weights = np.asarray(measure.weights).copy()
    weights[::4] = 0.0
    measure = replace(measure, mask=jnp.asarray(mask), weights=jnp.asarray(weights))
    g, D, N = _contributions(measure)
    np.testing.assert_allclose(
        _automatic(g, D, N, measure, covariance),
        _oracle(g, D, measure, covariance),
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        np.asarray(_automatic(g, D, N, measure, covariance))[:, :, 2], 0.0
    )


def test_fixed_mode_retains_weighted_exposure_and_ignores_centered_metadata():
    measure, covariance = _fixture()
    g, D, N = _contributions(measure)
    got = _automatic(g, D, N, measure, covariance)
    centered = replace(covariance, sampling=replace(covariance.sampling, centered=True))
    np.testing.assert_array_equal(got, _automatic(g, D, N, measure, centered))
    # Discriminate the tempting binary-mask/unit-weight contribution adapter.
    count_measure = replace(
        measure,
        mask=(measure.mask > 0).astype(float),
        weights=jnp.ones_like(measure.weights),
    )
    count_oracle = _oracle(g, D, count_measure, covariance)
    count_N = np.asarray(count_measure.mask).sum(axis=0)
    count_oracle *= np.outer(count_N, count_N)[None] / np.outer(N, N)[None]
    assert np.max(np.abs(np.asarray(got) - count_oracle)) > 1e-3


def test_fixed_projection_trace_degeneracy_with_zero_weight_support():
    # Both bundles support both coordinates, but their positive exposures
    # are orthogonal. The product of exposure projections has trace zero.
    measure = EmpiricalMeasure(
        x=jnp.arange(36, dtype=float).reshape(4, 9) / 7,
        mask=jnp.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        ),
        weights=jnp.asarray([1.0, 0.0, 0.0, 1.0]),
    )
    design = StratifiedCovariance(
        psu_ids=jnp.arange(4, dtype=float),
        cell_ids=jnp.zeros(4),
        stratum_ids=jnp.zeros(4),
        n_psu=4,
        n_cells=1,
        n_strata=1,
    )
    covariance = DesignAwareCovariance.from_design_mask(
        design,
        ClusteredCovariance(
            cluster_ids=jnp.asarray([0.0, 0.0, 1.0, 1.0]), n_clusters=2
        ),
        jnp.asarray([1.0, 0.0, 1.0]),
        fixed_strata=True,
    )
    g, D, N = _contributions(measure)
    got = np.asarray(_automatic(g, D, N, measure, covariance))
    np.testing.assert_allclose(got, _oracle(g, D, measure, covariance), atol=1e-13)
    np.testing.assert_array_equal(got[:, 0, 1], 0.0)


def test_missing_contributions_still_fail_closed_and_override_takes_precedence():
    _, covariance = _fixture()
    measure = AnalyticalMeasure(expectation_fn=lambda *_: jnp.zeros(3))
    theta = _Theta(0.0, 0.0)
    with pytest.raises(
        ValueError, match="DesignAwareCovariance requires empirical moment"
    ):
        k_statistic(
            theta, measure, covariance, _model, V=jnp.eye(3), strong_id_fallback=True
        )
    result = k_statistic(
        theta,
        measure,
        covariance,
        _model,
        V=jnp.eye(3),
        score_cov_fn=lambda *_: jnp.zeros((2, 3, 3)),
    )
    assert np.isfinite(result.K)
