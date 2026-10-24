"""Paired fixed-stratum mixed refit bootstrap (#251)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
    estimate,
    mixed_bootstrap,
)
from emu_gmm.inference.mixed_bootstrap import (
    _draw_bundles,
    _paired_layout,
    _resample_mixed_one,
)


@jdc.pytree_dataclass
class _Mean:
    theta: float


@jdc.pytree_dataclass
class _Pair:
    theta_t: float
    theta_s: float


def _identity(x, params):
    del params
    return x[:2]


def _pair_mean(x, params):
    return jnp.asarray([x[0] - params.theta_t, x[1] - params.theta_s])


def _paired_fixture(sign=1, *, ragged=False, couple=True, centered=False):
    """Two strata, four bundles each, one PSU per arm in each bundle."""
    x, mask, weights, psu, cell, stratum, bundle = [], [], [], [], [], [], []
    shocks = (-3.0, -1.0, 1.0, 3.0)
    for s in range(2):
        for b, shock in enumerate(shocks):
            g = s * 4 + b
            for arm in range(2):
                p = 2 * g + arm
                for obs in range(2):
                    t = shock + (2.0 if s == 0 else -1.0) + 0.1 * obs
                    v = sign * shock + (4.0 if s == 0 else -3.0) - 0.1 * obs
                    x.append((t, 0.0) if arm == 0 else (0.0, v))
                    mask.append((1.0, 0.0) if arm == 0 else (0.0, 1.0))
                    weights.append(1.0 + 0.05 * (g + obs) if ragged else 1.0)
                    psu.append(p)
                    cell.append(2 * s + arm)
                    stratum.append(s)
                    bundle.append(g)
    x_arr = np.asarray(x)
    mask_arr = np.asarray(mask)
    if ragged:
        mask_arr[::7, 0] = 0.0
        mask_arr[5::9, 1] = 0.0
    measure = EmpiricalMeasure(
        x=jnp.asarray(x_arr),
        mask=jnp.asarray(mask_arr),
        weights=jnp.asarray(weights),
    )
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(psu, dtype=jnp.float64),
        cell_ids=jnp.asarray(cell, dtype=jnp.float64),
        stratum_ids=jnp.asarray(stratum, dtype=jnp.float64),
        n_psu=16,
        n_cells=4,
        n_strata=2,
        fpc=False,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(bundle, dtype=jnp.float64),
        n_clusters=8,
        centered=centered,
    )
    covariance = DesignAwareCovariance.from_design_mask(
        design,
        sampling,
        jnp.asarray([1.0, 0.0]),
        couple=couple,
        fixed_strata=True,
    )
    return measure, covariance


def _with_layout(covariance, *, psu=None, cell=None, stratum=None, bundle=None):
    design = covariance.design
    altered_design = StratifiedCovariance(
        psu_ids=jnp.asarray(design.psu_ids if psu is None else psu),
        cell_ids=jnp.asarray(design.cell_ids if cell is None else cell),
        stratum_ids=jnp.asarray(design.stratum_ids if stratum is None else stratum),
        n_psu=design.n_psu,
        n_cells=design.n_cells,
        n_strata=design.n_strata,
    )
    sampling = covariance.sampling
    altered_sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(sampling.cluster_ids if bundle is None else bundle),
        n_clusters=sampling.n_clusters,
    )
    # Layout validation belongs to the bootstrap even when the covariance
    # factory would independently reject a malformed fixed-stratum layout.
    return DesignAwareCovariance(
        design=altered_design,
        sampling=altered_sampling,
        design_moment_mask=covariance.design_moment_mask,
        all_design=False,
        couple=True,
        fixed_strata=True,
    )


def test_bundle_draw_preserves_cells_rows_masks_weights_and_fresh_ids():
    measure, cov = _paired_fixture(ragged=True, centered=True, couple=False)
    rows, strata, psu = _paired_layout(measure, cov)
    assert tuple(len(groups) for groups in strata) == (4, 4)
    drawn = np.asarray([0, 0, 1, 3, 4, 4, 6, 7])
    boot_measure, boot_cov = _resample_mixed_one(measure, cov, rows, psu, drawn)
    source = np.concatenate([rows[int(g)] for g in drawn])
    np.testing.assert_array_equal(boot_measure.x, np.asarray(measure.x)[source])
    np.testing.assert_array_equal(boot_measure.mask, np.asarray(measure.mask)[source])
    np.testing.assert_array_equal(
        boot_measure.weights, np.asarray(measure.weights)[source]
    )
    np.testing.assert_array_equal(
        boot_cov.design.cell_ids, np.asarray(cov.design.cell_ids)[source]
    )
    np.testing.assert_array_equal(
        boot_cov.design.stratum_ids, np.asarray(cov.design.stratum_ids)[source]
    )
    assert boot_cov.fixed_strata and not boot_cov.couple
    assert boot_cov.sampling.centered
    assert boot_cov.design.n_psu == cov.design.n_psu == 16
    assert boot_cov.sampling.n_clusters == cov.sampling.n_clusters == 8
    new_psu = np.asarray(boot_cov.design.psu_ids, dtype=int)
    new_bundle = np.asarray(boot_cov.sampling.cluster_ids, dtype=int)
    np.testing.assert_array_equal(np.unique(new_psu), np.arange(16))
    np.testing.assert_array_equal(np.unique(new_bundle), np.arange(8))
    for c in range(4):
        assert np.unique(new_psu[np.asarray(boot_cov.design.cell_ids) == c]).size == 4
    for g in range(8):
        psus = np.unique(new_psu[new_bundle == g])
        assert psus.size == 2
        assert (
            np.unique(np.asarray(boot_cov.design.stratum_ids)[new_bundle == g]).size
            == 1
        )
    # The repeated original bundle is copied into two independent new bundles.
    first, second = np.flatnonzero(new_bundle == 0), np.flatnonzero(new_bundle == 1)
    np.testing.assert_array_equal(
        np.asarray(boot_measure.x)[first], np.asarray(boot_measure.x)[second]
    )
    assert np.intersect1d(new_psu[first], new_psu[second]).size == 0


def test_complete_multi_psu_bundle_count_vector_and_duplicate_ids():
    """A positive (2,1) cell-count vector is supported, not just (1,1)."""
    x, mask, psu, cell, stratum, bundle = [], [], [], [], [], []
    for s in range(2):
        for b in range(3):
            g = 3 * s + b
            for arm, n_psu in enumerate((2, 1)):
                for within in range(n_psu):
                    x.append((float(g + within), float(arm + g)))
                    mask.append((1.0, 0.0) if arm == 0 else (0.0, 1.0))
                    psu.append(len(psu))
                    cell.append(2 * s + arm)
                    stratum.append(s)
                    bundle.append(g)
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.ones(len(x))
    )
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(psu, dtype=jnp.float64),
        cell_ids=jnp.asarray(cell, dtype=jnp.float64),
        stratum_ids=jnp.asarray(stratum, dtype=jnp.float64),
        n_psu=18,
        n_cells=4,
        n_strata=2,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(bundle, dtype=jnp.float64), n_clusters=6
    )
    cov = DesignAwareCovariance.from_design_mask(
        design, sampling, jnp.asarray([1.0, 0.0]), fixed_strata=True
    )
    rows, _, ids = _paired_layout(measure, cov)
    drawn = np.asarray([0, 0, 2, 3, 4, 5])
    boot_measure, boot_cov = _resample_mixed_one(measure, cov, rows, ids, drawn)
    new_psu = np.asarray(boot_cov.design.psu_ids, dtype=int)
    new_bundle = np.asarray(boot_cov.sampling.cluster_ids, dtype=int)
    new_cell = np.asarray(boot_cov.design.cell_ids, dtype=int)
    assert boot_cov.design.n_psu == 18
    np.testing.assert_array_equal(np.unique(new_psu), np.arange(18))
    np.testing.assert_array_equal(np.unique(new_bundle), np.arange(6))
    assert tuple(np.unique(new_psu[new_cell == c]).size for c in range(4)) == (
        6,
        3,
        6,
        3,
    )
    for g in range(6):
        assert tuple(
            np.unique(new_psu[(new_bundle == g) & (new_cell == c)]).size
            for c in (2 * (g // 3), 2 * (g // 3) + 1)
        ) == (2, 1)
    np.testing.assert_array_equal(
        np.asarray(boot_measure.x)[new_bundle == 0],
        np.asarray(boot_measure.x)[new_bundle == 1],
    )
    assert not np.intersect1d(new_psu[new_bundle == 0], new_psu[new_bundle == 1]).size


@pytest.mark.parametrize("sign", [-1, 1])
def test_paired_shock_keeps_signed_cross_block(sign):
    measure, cov = _paired_fixture(sign=sign)
    rows, _, psu = _paired_layout(measure, cov)
    drawn = np.asarray([0, 0, 1, 3, 4, 4, 6, 7])
    boot_measure, boot_cov = _resample_mixed_one(measure, cov, rows, psu, drawn)
    cross = np.asarray(boot_cov.cross_block(_identity, None, boot_measure))
    assert sign * cross[0, 1] > 0
    coupled = np.asarray(boot_cov.covariance(_identity, None, boot_measure))
    block_cov = DesignAwareCovariance.from_design_mask(
        boot_cov.design,
        boot_cov.sampling,
        boot_cov.design_moment_mask,
        couple=False,
        fixed_strata=True,
    )
    block = np.asarray(block_cov.covariance(_identity, None, boot_measure))
    np.testing.assert_allclose(coupled, block + cross, atol=1e-12, rtol=1e-12)


def test_draw_counts_within_strata_for_many_keys():
    measure, cov = _paired_fixture()
    _, strata, _ = _paired_layout(measure, cov)
    for seed in range(8):
        drawn = _draw_bundles(strata, jax.random.PRNGKey(seed))
        assert np.sum(drawn < 4) == 4
        assert np.sum(drawn >= 4) == 4


@pytest.mark.parametrize(
    ("field", "change", "message"),
    [
        ("psu", (2, 0), "PSU spans cells"),
        ("bundle", (0, 1), "PSU spans sampling bundles"),
        ("stratum", (0, 1), "PSU spans cells or strata"),
        ("cell", (0, 0.5), "integer IDs"),
    ],
)
def test_invalid_layout_rejected(field, change, message):
    measure, cov = _paired_fixture()
    arrays = {
        "psu": np.asarray(cov.design.psu_ids).copy(),
        "cell": np.asarray(cov.design.cell_ids).copy(),
        "stratum": np.asarray(cov.design.stratum_ids).copy(),
        "bundle": np.asarray(cov.sampling.cluster_ids).copy(),
    }
    arrays[field][change[0]] = change[1]
    bad_cov = _with_layout(cov, **{field: arrays[field]})
    with pytest.raises(ValueError, match=message):
        _paired_layout(measure, bad_cov)


def test_cross_stratum_bundle_rejected():
    measure, cov = _paired_fixture()
    bundle = np.asarray(cov.sampling.cluster_ids).copy()
    bundle[16:18] = 0  # Move a whole PSU from stratum 1 into bundle 0.
    with pytest.raises(ValueError, match="sampling bundle spans strata"):
        _paired_layout(measure, _with_layout(cov, bundle=bundle))


def test_incomplete_bundle_rejected_after_whole_psu_move():
    measure, cov = _paired_fixture()
    bundle = np.asarray(cov.sampling.cluster_ids).copy()
    bundle[4:6] = 0  # Move PSU 2, leaving bundle 1 without its arm-0 PSU.
    with pytest.raises(ValueError, match="bundle must be complete"):
        _paired_layout(measure, _with_layout(cov, bundle=bundle))


def test_unbalanced_complete_bundle_rejected():
    measure, cov = _paired_fixture()
    # Add one whole arm-0 PSU to bundle 0. Every bundle remains complete,
    # but bundle 0's count vector is (2,1) versus (1,1) for its peers.
    extra = np.asarray([0, 1])
    extended = EmpiricalMeasure(
        x=jnp.asarray(
            np.concatenate([np.asarray(measure.x), np.asarray(measure.x)[extra]])
        ),
        mask=jnp.asarray(
            np.concatenate([np.asarray(measure.mask), np.asarray(measure.mask)[extra]])
        ),
        weights=jnp.asarray(
            np.concatenate(
                [np.asarray(measure.weights), np.asarray(measure.weights)[extra]]
            )
        ),
    )
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(np.concatenate([np.asarray(cov.design.psu_ids), [16, 16]])),
        cell_ids=jnp.asarray(np.concatenate([np.asarray(cov.design.cell_ids), [0, 0]])),
        stratum_ids=jnp.asarray(
            np.concatenate([np.asarray(cov.design.stratum_ids), [0, 0]])
        ),
        n_psu=17,
        n_cells=4,
        n_strata=2,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(
            np.concatenate([np.asarray(cov.sampling.cluster_ids), [0, 0]])
        ),
        n_clusters=8,
    )
    bad = DesignAwareCovariance.from_design_mask(
        design,
        sampling,
        cov.design_moment_mask,
        fixed_strata=True,
    )
    with pytest.raises(ValueError, match="balanced"):
        _paired_layout(extended, bad)


def test_unsupported_covariance_modes_rejected():
    measure, cov = _paired_fixture()

    def run(bad):
        mixed_bootstrap(
            _identity,
            _Mean(theta=0.0),
            measure,
            bad,
            n_boot=1,
            key=jax.random.PRNGKey(0),
        )

    random_mode = DesignAwareCovariance.from_design_mask(
        cov.design,
        cov.sampling,
        cov.design_moment_mask,
    )
    with pytest.raises(ValueError, match="same partition"):
        run(random_mode)
    fpc_design = StratifiedCovariance(
        psu_ids=cov.design.psu_ids,
        cell_ids=cov.design.cell_ids,
        stratum_ids=cov.design.stratum_ids,
        n_psu=16,
        n_cells=4,
        n_strata=2,
        fpc=True,
    )
    with pytest.raises(ValueError, match="fpc=False"):
        run(
            DesignAwareCovariance.from_design_mask(
                fpc_design,
                cov.sampling,
                cov.design_moment_mask,
                fixed_strata=True,
            )
        )
    bad_sampling = ClusteredCovariance(
        cluster_ids=cov.sampling.cluster_ids,
        n_clusters=8,
        dof_correction=True,
    )
    with pytest.raises(ValueError, match="dof_correction=False"):
        run(
            DesignAwareCovariance(
                design=cov.design,
                sampling=bad_sampling,
                design_moment_mask=cov.design_moment_mask,
                all_design=False,
                fixed_strata=True,
            )
        )


def test_shared_refit_keeps_nan_events_and_labels(monkeypatch):
    measure, cov = _paired_fixture()
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    calls = {"n": 0}

    def fake_estimate(*, theta_init, covariance, **kwargs):
        assert isinstance(covariance, DesignAwareCovariance)
        assert covariance.fixed_strata
        calls["n"] += 1
        invalid = calls["n"] == 1
        diag = SimpleNamespace(
            N_j=SimpleNamespace(
                array=jnp.asarray([0.0, 1.0] if invalid else [1.0, 1.0])
            ),
            tau_saturated=invalid,
        )
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(0.0),
            converged=not invalid,
            diagnostics=diag,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    with pytest.warns(UserWarning, match="mixed_bootstrap: 1 of 2"):
        boot = mixed_bootstrap(
            _identity,
            _Mean(theta=0.0),
            measure,
            cov,
            n_boot=2,
            key=jax.random.PRNGKey(251),
        )
    assert boot.n_valid == 1 and boot.n_nonconverged == 1
    assert np.isnan(np.asarray(boot.theta_boot.array)[0]).all()
    assert np.isnan(np.asarray(boot.J_boot)[0])
    assert np.asarray(boot.zero_support).tolist() == [True, False]
    assert np.asarray(boot.tau_saturated).tolist() == [True, False]
    assert boot.param_names == ("theta",)
    assert boot.coords["parameters"] == ("theta",)


def test_real_estimate_refit_smoke():
    measure, cov = _paired_fixture()
    point = estimate(
        _pair_mean,
        measure,
        covariance=cov,
        theta_init=_Pair(theta_t=0.0, theta_s=0.0),
    )
    assert point.converged
    boot = mixed_bootstrap(
        _pair_mean,
        point.theta_hat,
        measure,
        cov,
        n_boot=2,
        key=jax.random.PRNGKey(251),
    )
    assert boot.n_valid == 2
    assert boot.param_names == ("theta_t", "theta_s")
    assert np.isfinite(np.asarray(boot.theta_boot.array)).all()
