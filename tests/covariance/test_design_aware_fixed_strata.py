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
    """Independent NumPy pair/exposure projection with local trace factor."""
    x, mask, weights = fx["x"], fx["mask"], fx["weights"]
    G, M = 6, x.shape[1]
    totals = np.zeros((G, M))
    exposure = np.zeros((G, M))
    support = np.zeros((G, M), dtype=bool)
    for i, h in enumerate(fx["cluster"]):
        h = int(h)
        totals[h] += mask[i] * weights[i] * x[i]
        exposure[h] += mask[i] * weights[i]
        support[h] |= mask[i] > 0
    Nj = (mask * weights[:, None]).sum(axis=0)
    out = np.zeros((M, M))
    counts = np.zeros((2, M, M))
    for j in range(M):
        for k in range(M):
            numerator = 0.0
            for s in range(2):
                hs = np.arange(3 * s, 3 * s + 3)
                hs = hs[support[hs, j] & support[hs, k]]
                H = len(hs)
                counts[s, j, k] = H
                if H < 2:
                    continue
                ej, ek = exposure[hs, j], exposure[hs, k]
                Pj = np.eye(H) - np.outer(ej, ej) / (ej @ ej)
                Pk = np.eye(H) - np.outer(ek, ek) / (ek @ ek)
                numerator += (
                    H
                    / np.trace(Pj @ Pk)
                    * ((Pj @ totals[hs, j]) @ (Pk @ totals[hs, k]))
                )
            if Nj[j] * Nj[k] > 0:
                out[j, k] = numerator / (Nj[j] * Nj[k])
    return out, counts


def test_fixed_stratum_matches_pair_exposure_reference_and_profile_invariance():
    fx = _fixture(profile=True)
    dac, _, measure = _objects(fx)
    V = np.asarray(dac.covariance(_identity, None, measure))
    reference, counts = _reference(fx)
    np.testing.assert_allclose(V[1:, 1:], reference[1:, 1:], atol=1e-12)
    np.testing.assert_allclose(V[0, 1:], reference[0, 1:], atol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(dac.fixed_stratum_pair_support(measure)), counts
    )
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


def test_local_dof_decomposition_and_unsupported_pair():
    fx = _fixture(single_support=True)
    raw, _, measure = _objects(fx)
    V_raw = np.asarray(raw.covariance(_identity, None, measure))
    ref, counts = _reference(fx)
    assert np.all(counts[:, 0, 2] == 1)
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


def test_one_sampling_cluster_per_stratum_has_visible_zero_rule():
    fx = _fixture()
    _, design, measure = _objects(fx)
    sampling = ClusteredCovariance(cluster_ids=jnp.asarray(fx["strata"]), n_clusters=2)
    dac = DesignAwareCovariance.from_design_mask(
        design, sampling, jnp.array([1.0, 0.0, 0.0]), fixed_strata=True
    )
    support = np.asarray(dac.fixed_stratum_pair_support(measure))
    assert np.all(support <= 1.0)
    V = np.asarray(dac.covariance(_identity, None, measure))
    assert np.array_equal(V[1:, 1:], np.zeros((2, 2)))
    assert np.array_equal(V[0, 1:], np.zeros(2))
    np.testing.assert_array_equal(
        V[:1, :1], np.asarray(design.covariance(_identity, None, measure))[:1, :1]
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
        np.asarray(value) for value in dac.fixed_stratum_pair_diagnostics(measure)
    )
    assert support[0, 0, 2] == 2
    assert support[1, 0, 2] == 0
    assert trace[0, 0, 2] <= np.sqrt(np.finfo(float).eps) * support[0, 0, 2]
    assert not usable[0, 0, 2]
    assert usable[0, 0, 0]
    np.testing.assert_array_equal(
        np.asarray(dac.fixed_stratum_pair_support(measure)), support
    )
    V = np.asarray(dac.covariance(_identity, None, measure))
    assert V[0, 2] == 0.0
    assert V[2, 0] == 0.0
