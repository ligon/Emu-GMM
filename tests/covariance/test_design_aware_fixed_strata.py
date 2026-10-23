"""Deterministic checks for opt-in fixed-stratum sampling centering (#145)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
)


def _identity(x, theta):
    del theta
    return x


def _fixture(rho=1.0, profile=True, single_support=False):
    # Each sampling cluster pairs distinct T and S PSUs within one stratum.
    # The common z changes across clusters within a stratum: a genuine
    # paired cross-arm shock, not a stratum-wide nuisance intercept.
    shocks = np.array([[-2.0, 0.5, 1.5], [1.0, -2.0, 1.0]])
    x, mask, weights, psu, cell, strata, cluster = [], [], [], [], [], [], []
    g = 0
    for s in range(2):
        for h in range(3):
            z = shocks[s, h]
            for arm in range(2):
                row = [0.0, 0.0, 0.0]
                observed = [0.0, 0.0, 0.0]
                if arm == 0:
                    row[0] = z + ([9.0, -7.0][s] if profile else 0.0)
                    observed[0] = 1.0
                else:
                    row[1] = rho * z + ([-5.0, 11.0][s] if profile else 0.0)
                    row[2] = (h - 1.0) + ([3.0, -2.0][s] if profile else 0.0)
                    observed[1] = 1.0
                    observed[2] = float(h == 0 or (s == 0 and h == 1))
                    if single_support:
                        observed[2] = float(h == 0)
                x.append(row)
                mask.append(observed)
                weights.append(1.0 + 0.2 * h + 0.1 * arm + 0.05 * s)
                psu.append(g)
                cell.append(2 * s + arm)
                strata.append(s)
                cluster.append(3 * s + h)
                g += 1
    return {
        "x": np.asarray(x),
        "mask": np.asarray(mask),
        "weights": np.asarray(weights),
        "psu": np.asarray(psu),
        "cell": np.asarray(cell),
        "strata": np.asarray(strata),
        "cluster": np.asarray(cluster),
    }


def _objects(fx, *, dof=False, fixed=True, couple=True, all_design=False):
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(fx["psu"]),
        cell_ids=jnp.asarray(fx["cell"]),
        stratum_ids=jnp.asarray(fx["strata"]),
        n_psu=12,
        n_cells=4,
        n_strata=2,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(fx["cluster"]),
        n_clusters=6,
        dof_correction=dof,
    )
    mask = jnp.ones(3) if all_design else jnp.array([1.0, 0.0, 0.0])
    dac = DesignAwareCovariance.from_design_mask(
        design, sampling, mask, couple=couple, fixed_strata=fixed
    )
    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    return dac, design, measure


def _reference(fx):
    """Independent NumPy ordered-cell-pair projection and trace correction."""
    x, mask, weights = fx["x"], fx["mask"], fx["weights"]
    G, C, M = 6, 4, x.shape[1]
    totals = np.zeros((G, C, M))
    exposure = np.zeros_like(totals)
    support = np.zeros((G, C, M), dtype=bool)
    for i, h in enumerate(fx["cluster"]):
        h = int(h)
        c = int(fx["cell"][i])
        totals[h, c] += mask[i] * weights[i] * x[i]
        exposure[h, c] += mask[i] * weights[i]
        support[h, c] |= mask[i] > 0
    Nj = (mask * weights[:, None]).sum(axis=0)
    out = np.zeros((M, M))
    counts = np.zeros((2, M, M))
    cell_counts = np.zeros((C, C, M, M))
    cell_trace = np.zeros_like(cell_counts)
    cell_usable = np.zeros_like(cell_counts, dtype=bool)
    for j in range(M):
        for k in range(M):
            numerator = 0.0
            for s in range(2):
                hs = np.arange(3 * s, 3 * s + 3)
                counts[s, j, k] = np.sum(
                    support[hs, :, j].any(axis=1) & support[hs, :, k].any(axis=1)
                )
                for c in (2 * s, 2 * s + 1):
                    for d in (2 * s, 2 * s + 1):
                        ids = hs[support[hs, c, j] & support[hs, d, k]]
                        H = len(ids)
                        cell_counts[c, d, j, k] = H
                        if H < 2:
                            continue
                        ej, ek = exposure[ids, c, j], exposure[ids, d, k]
                        Pj = np.eye(H) - np.outer(ej, ej) / (ej @ ej)
                        Pk = np.eye(H) - np.outer(ek, ek) / (ek @ ek)
                        trace = np.trace(Pj @ Pk)
                        cell_trace[c, d, j, k] = trace
                        if trace <= np.sqrt(np.finfo(float).eps) * H:
                            continue
                        cell_usable[c, d, j, k] = True
                        numerator += (
                            H
                            / trace
                            * ((Pj @ totals[ids, c, j]) @ (Pk @ totals[ids, d, k]))
                        )
            if Nj[j] * Nj[k] > 0:
                out[j, k] = numerator / (Nj[j] * Nj[k])
    return out, counts, cell_counts, cell_trace, cell_usable


def test_fixed_stratum_matches_pair_exposure_reference_and_profile_invariance():
    fx = _fixture(profile=True)
    dac, _, measure = _objects(fx)
    V = np.asarray(dac.covariance(_identity, None, measure))
    reference, counts, cell_counts, cell_trace, cell_usable = _reference(fx)
    np.testing.assert_allclose(V[1:, 1:], reference[1:, 1:], atol=1e-12)
    np.testing.assert_allclose(V[0, 1:], reference[0, 1:], atol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(dac.fixed_stratum_pair_support(measure)), counts
    )
    observed_cell = dac.fixed_cell_pair_diagnostics(measure)
    np.testing.assert_array_equal(np.asarray(observed_cell[0]), cell_counts)
    np.testing.assert_allclose(np.asarray(observed_cell[1]), cell_trace, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(observed_cell[2]), cell_usable)
    summary_count, summary_trace, summary_usable = (
        np.asarray(value) for value in dac.fixed_stratum_pair_diagnostics(measure)
    )
    np.testing.assert_array_equal(summary_count, counts)
    for s in range(2):
        within = slice(2 * s, 2 * s + 2)
        np.testing.assert_allclose(
            summary_trace[s],
            np.where(cell_usable[within, within], cell_trace[within, within], 0.0).sum(
                axis=(0, 1)
            ),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            summary_usable[s], cell_usable[within, within].any(axis=(0, 1))
        )
    # Both ordered cross-cell terms enter the variance of a sum. For this
    # fixture, the T/S covariance has exactly one supported cell direction.
    assert cell_counts[0, 1, 0, 1] == 3
    assert cell_counts[1, 0, 0, 1] == 0
    # Unequal exposure makes total-mean centering invalid; projection
    # removes a fixed per-observation profile in both SS and TS.
    fx0 = _fixture(profile=False)
    dac0, _, measure0 = _objects(fx0)
    V0 = np.asarray(dac0.covariance(_identity, None, measure0))
    np.testing.assert_allclose(V[1:, 1:], V0[1:, 1:], atol=1e-12)
    np.testing.assert_allclose(V[0, 1:], V0[0, 1:], atol=1e-12)
    old, _, old_measure = _objects(fx, fixed=False)
    old0, _, old_measure0 = _objects(fx0, fixed=False)
    assert (
        abs(
            float(old.covariance(_identity, None, old_measure)[0, 1])
            - float(old0.covariance(_identity, None, old_measure0)[0, 1])
        )
        > 0.01
    )


def test_cross_arm_shock_sign_and_zero_shock():
    values = []
    for rho in (1.0, -1.0, 0.0):
        dac, _, measure = _objects(_fixture(rho=rho))
        values.append(float(dac.cross_block(_identity, None, measure)[0, 1]))
    assert values[0] > 0
    np.testing.assert_allclose(values[1], -values[0], atol=1e-14)
    np.testing.assert_allclose(values[2], 0.0, atol=1e-14)


def test_coincident_psu_clusters_recover_cell_neyman_assembly():
    """One cell per cluster makes the cell-pair expansion exactly Neyman."""
    psu = jnp.arange(12)
    cell = jnp.repeat(jnp.arange(4), 3)
    strata = cell // 2
    design = StratifiedCovariance(
        psu_ids=psu,
        cell_ids=cell,
        stratum_ids=strata,
        n_psu=12,
        n_cells=4,
        n_strata=2,
    )
    sampling = ClusteredCovariance(cluster_ids=psu, n_clusters=12)
    mixed = DesignAwareCovariance.from_design_mask(
        design, sampling, jnp.array([1.0, 0.0, 0.0]), fixed_strata=True
    )
    index = jnp.arange(12)
    treatment = (cell % 2 == 0).astype(jnp.float64)
    x = jnp.stack(
        [
            treatment * (10.0 * strata + index % 3 - 1.0),
            20.0 * cell + index % 3 - 1.0,
            -5.0 * cell + (index % 3 - 1.0) ** 2,
        ],
        axis=1,
    )
    mask = jnp.ones_like(x).at[jnp.arange(2, 12, 3), 2].set(0.0)
    measure = EmpiricalMeasure(x=x, mask=mask, weights=jnp.ones(12))
    np.testing.assert_allclose(
        np.asarray(mixed.covariance(_identity, None, measure)),
        np.asarray(design.covariance(_identity, None, measure)),
        atol=1e-12,
    )
    cell_count = np.asarray(mixed.fixed_cell_pair_diagnostics(measure)[0])
    assert not np.any(cell_count[0, 1])
    assert not np.any(cell_count[1, 0])


def test_local_dof_decomposition_and_unsupported_pair():
    fx = _fixture(single_support=True)
    raw, _, measure = _objects(fx)
    V_raw = np.asarray(raw.covariance(_identity, None, measure))
    ref, counts, cell_counts, _trace, _usable = _reference(fx)
    assert np.all(counts[:, 0, 2] == 1)
    assert cell_counts[0, 1, 0, 2] == 1
    assert V_raw[0, 2] == 0.0
    np.testing.assert_allclose(V_raw[0, 1:], ref[0, 1:], atol=1e-12)
    # The local trace factor is the complete finite-cluster correction
    # under the stated homogeneous-cluster working model.
    assert V_raw[0, 1] != 0
    assert V_raw[1, 1] != 0
    block, _, _ = _objects(fx, couple=False)
    np.testing.assert_allclose(
        V_raw,
        np.asarray(block.covariance(_identity, None, measure))
        + np.asarray(raw.cross_block(_identity, None, measure)),
        atol=1e-12,
    )


def test_fixed_strata_rejects_extra_global_dof_correction():
    with pytest.raises(ValueError, match="local projection trace already corrects"):
        _objects(_fixture(), dof=True)
    # The pre-existing random-stratum mode still accepts #119's flag.
    default, _, _ = _objects(_fixture(), dof=True, fixed=False)
    assert default.sampling.dof_correction


def test_all_design_bitwise_and_jit_cache_parity():
    fx = _fixture()
    dac, design, measure = _objects(fx, all_design=True)
    np.testing.assert_array_equal(
        np.asarray(dac.covariance(_identity, None, measure)),
        np.asarray(design.covariance(_identity, None, measure)),
    )
    # The sampling correction is immaterial to the exact design-only path.
    dac_dof, _, _ = _objects(fx, all_design=True, dof=True)
    np.testing.assert_array_equal(
        np.asarray(dac_dof.covariance(_identity, None, measure)),
        np.asarray(design.covariance(_identity, None, measure)),
    )
    mixed, _, measure = _objects(fx)
    cached = measure.expectation_and_contributions(_identity, None)
    eager = np.asarray(mixed.covariance(_identity, None, measure))
    np.testing.assert_array_equal(
        eager,
        np.asarray(
            mixed.covariance(_identity, None, measure, cached_intermediates=cached)
        ),
    )
    calls = []

    def counted(x, theta):
        calls.append(None)
        return _identity(x, theta)

    jit_cov = jax.jit(lambda m: mixed.covariance(counted, None, m))
    np.testing.assert_allclose(np.asarray(jit_cov(measure)), eager, atol=1e-12)
    first = len(calls)
    assert first > 0
    np.testing.assert_allclose(np.asarray(jit_cov(measure)), eager, atol=1e-12)
    assert len(calls) == first


def test_fixed_strata_requires_nested_sampling_clusters():
    fx = _fixture()
    fx["cluster"][6] = 0  # cluster 0 now crosses strata 0 and 1
    with pytest.raises(ValueError, match="nested in strata"):
        _objects(fx)


def test_fixed_strata_requires_cells_nested_in_strata():
    fx = _fixture()
    fx["cell"][6] = 0  # cell 0 now spans strata 0 and 1
    with pytest.raises(ValueError, match="cells nested in strata"):
        _objects(fx)


@pytest.mark.parametrize("field", ["cluster", "strata", "cell"])
@pytest.mark.parametrize("value", [0.49, np.nan, np.inf])
def test_fixed_strata_rejects_nonintegral_or_nonfinite_group_ids(field, value):
    fx = _fixture()
    fx[field] = fx[field].astype(float)
    fx[field][0] = value
    with pytest.raises(ValueError, match="finite integer-valued"):
        _objects(fx)
    if value == 0.49:
        # The pre-existing random-stratum path keeps its rounding convention.
        default, _, _ = _objects(fx, fixed=False)
        assert not default.fixed_strata


def test_one_sampling_cluster_per_stratum_is_rejected():
    fx = _fixture()
    _, design, _measure = _objects(fx)
    sampling = ClusteredCovariance(cluster_ids=jnp.asarray(fx["strata"]), n_clusters=2)
    with pytest.raises(ValueError, match="at least two sampling clusters"):
        DesignAwareCovariance.from_design_mask(
            design, sampling, jnp.array([1.0, 0.0, 0.0]), fixed_strata=True
        )


def test_nearly_orthogonal_pair_exposures_are_not_inflated():
    fx = _fixture()
    # Only clusters 0 and 1 support pair (0,2). Their positive exposures
    # approach orthogonal vectors: e_0=(1,eps), e_2=(eps,1).
    eps = 1e-9
    fx["weights"][[0, 1, 2, 3]] = [1.0, eps, eps, 1.0]
    fx["mask"][7, 2] = 0.0
    dac, _, measure = _objects(fx)
    support, trace, usable = (
        np.asarray(value) for value in dac.fixed_cell_pair_diagnostics(measure)
    )
    assert support[0, 1, 0, 2] == 2
    assert support[2, 3, 0, 2] == 0
    assert trace[0, 1, 0, 2] <= np.sqrt(np.finfo(float).eps) * support[0, 1, 0, 2]
    assert not usable[0, 1, 0, 2]
    assert usable[0, 0, 0, 0]
    summary_support, summary_trace, summary_usable = (
        np.asarray(value) for value in dac.fixed_stratum_pair_diagnostics(measure)
    )
    assert summary_support[0, 0, 2] == 2
    assert summary_trace[0, 0, 2] == 0.0
    assert not summary_usable[0, 0, 2]
    np.testing.assert_array_equal(
        np.asarray(dac.fixed_stratum_pair_support(measure)), summary_support
    )
    V = np.asarray(dac.covariance(_identity, None, measure))
    assert V[0, 2] == 0.0
    assert V[2, 0] == 0.0
