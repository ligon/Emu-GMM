"""Whole-stratum mixed refits and independent pooled-ratio oracles (#251)."""

import importlib
import itertools
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
    mixed_bootstrap,
)
from emu_gmm.inference.mixed_bootstrap import _resample_strata_one, _stratum_layout
from emu_gmm.inference.stratified_bootstrap import _cell_layout


@jdc.pytree_dataclass
class _Pair:
    theta_t: float
    theta_s: float


def _mean(row, theta):
    return jnp.array([row[0] - theta.theta_t, row[1] - theta.theta_s])


def _fixture(counts=((2, 2), (3, 2), (2, 3)), *, weighted=True, absent=False):
    """Independent strata with unequal PSU sizes, counts and arm exposures."""
    x, mask, weights, psu, cell, strata = [], [], [], [], [], []
    p, c = 0, 0
    for s, arms in enumerate(counts):
        for arm, count in enumerate(arms):
            if absent and s == 1 and arm == 1:
                continue
            for k in range(count):
                for obs in range(1 + (k % 2)):
                    value = 2.0 * s + (-1.0 if arm else 1.0) * (k + 0.1 * obs)
                    x.append((value, 0.0) if arm == 0 else (0.0, value))
                    mask.append((1.0, 0.0) if arm == 0 else (0.0, 1.0))
                    weights.append(1.0 + 0.1 * len(x) if weighted else 1.0)
                    psu.append(p)
                    cell.append(c)
                    strata.append(s)
                p += 1
            c += 1
    n = len(counts)
    measure = EmpiricalMeasure(
        x=jnp.array(x), mask=jnp.array(mask), weights=jnp.array(weights)
    )
    design = StratifiedCovariance(
        psu_ids=jnp.array(psu, dtype=float),
        cell_ids=jnp.array(cell, dtype=float),
        stratum_ids=jnp.array(strata, dtype=float),
        n_psu=p,
        n_cells=c,
        n_strata=n,
    )
    # A reversed numeric labelling still denotes precisely the same partition.
    sampling = ClusteredCovariance(
        cluster_ids=jnp.array(n - 1 - np.array(strata), dtype=float), n_clusters=n
    )
    return measure, DesignAwareCovariance.from_design_mask(
        design, sampling, jnp.array([1.0, 0.0])
    )


def _alter(cov, *, psu=None, cell=None, strata=None, cluster=None, **flags):
    d, s = cov.design, cov.sampling
    design = StratifiedCovariance(
        psu_ids=d.psu_ids if psu is None else jnp.asarray(psu),
        cell_ids=d.cell_ids if cell is None else jnp.asarray(cell),
        stratum_ids=d.stratum_ids if strata is None else jnp.asarray(strata),
        n_psu=d.n_psu,
        n_cells=d.n_cells,
        n_strata=d.n_strata,
        fpc=flags.get("fpc", d.fpc),
    )
    sampling = ClusteredCovariance(
        cluster_ids=s.cluster_ids if cluster is None else jnp.asarray(cluster),
        n_clusters=flags.get("n_clusters", s.n_clusters),
        dof_correction=flags.get("dof_correction", s.dof_correction),
        centered=flags.get("centered", s.centered),
    )
    return DesignAwareCovariance.from_design_mask(
        design, sampling, cov.design_moment_mask, couple=flags.get("couple", cov.couple)
    )


def _pooled(measure, rows=None):
    if rows is None:
        rows = np.arange(len(measure.x))
    exposure = np.asarray(measure.mask)[rows] * np.asarray(measure.weights)[rows, None]
    return (exposure * np.asarray(measure.x)[rows]).sum(0) / exposure.sum(0)


@pytest.mark.parametrize(
    "centered,dof,couple", [(False, False, True), (True, True, False)]
)
def test_variable_strata_copy_all_rows_and_fresh_group_levels(centered, dof, couple):
    measure, cov = _fixture(absent=True)
    cov = _alter(cov, centered=centered, dof_correction=dof, couple=couple)
    rows = _stratum_layout(measure, cov)
    drawn = np.array([1, 1, 0])
    boot, spec = _resample_strata_one(measure, cov, rows, drawn)
    source = np.concatenate([rows[s] for s in drawn])
    for field in ("x", "mask", "weights"):
        np.testing.assert_array_equal(
            getattr(boot, field), np.asarray(getattr(measure, field))[source]
        )
    assert not spec.fixed_strata and not spec.design.fpc
    assert spec.couple == couple
    assert spec.sampling.centered == centered
    assert spec.sampling.dof_correction == dof
    np.testing.assert_array_equal(spec.design_moment_mask, cov.design_moment_mask)
    np.testing.assert_array_equal(spec.sampling.cluster_ids, spec.design.stratum_ids)
    _cell_layout(boot, spec.design)  # No copied PSU/cell is split or pooled.
    assert spec.design.n_strata == spec.sampling.n_clusters == len(drawn)
    assert spec.design.n_cells == 4  # Two copies of an absent-arm stratum.
    offset = 0
    old_psus, old_cells = set(), set()
    for i, s in enumerate(drawn):
        section = slice(offset, offset + len(rows[s]))
        psus = set(np.asarray(spec.design.psu_ids)[section])
        cells = set(np.asarray(spec.design.cell_ids)[section])
        assert not old_psus.intersection(psus) and not old_cells.intersection(cells)
        assert np.all(np.asarray(spec.design.stratum_ids)[section] == i)
        old_psus.update(psus)
        old_cells.update(cells)
        offset += len(rows[s])
    assert len(old_psus) == spec.design.n_psu
    assert len(old_cells) == spec.design.n_cells


def test_all_stratum_draws_match_independent_weighted_ratio_oracle():
    measure, cov = _fixture()
    rows = _stratum_layout(measure, cov)
    actual, expected, equal_stratum_means = [], [], []
    for draw in itertools.product(range(3), repeat=3):
        boot, _ = _resample_strata_one(measure, cov, rows, np.array(draw))
        actual.append(np.asarray(boot.expectation(_mean, _Pair(0.0, 0.0))))
        source = np.concatenate([rows[s] for s in draw])
        expected.append(_pooled(measure, source))
        equal_stratum_means.append(
            np.mean([_pooled(measure, rows[s]) for s in draw], axis=0)
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert np.max(np.abs(np.array(expected) - equal_stratum_means)) > 0.05
    np.testing.assert_allclose(
        np.cov(actual, rowvar=False, ddof=0), np.cov(expected, rowvar=False, ddof=0)
    )


def test_random_common_profile_exposes_analytic_TT_target_difference():
    measure, cov = _fixture(counts=((2, 2), (2, 2)), weighted=False)
    strata = np.asarray(cov.design.stratum_ids, dtype=int)
    alpha = np.array([-1.0, 1.0])[strata]
    measure = EmpiricalMeasure(
        x=jnp.array(alpha[:, None] * np.asarray(measure.mask)),
        mask=measure.mask,
        weights=measure.weights,
    )
    # Equalize PSU sizes to retain a balanced empirical profile exactly.
    keep = np.r_[True, np.diff(np.asarray(cov.design.psu_ids)) != 0]
    measure = EmpiricalMeasure(
        x=measure.x[keep], mask=measure.mask[keep], weights=measure.weights[keep]
    )
    cov = _alter(
        cov,
        psu=cov.design.psu_ids[keep],
        cell=cov.design.cell_ids[keep],
        strata=cov.design.stratum_ids[keep],
        cluster=cov.sampling.cluster_ids[keep],
    )
    raw = np.asarray(cov.covariance(_mean, _Pair(0.0, 0.0), measure))
    np.testing.assert_allclose(raw, [[0.0, 0.5], [0.5, 0.5]])
    assert np.linalg.eigvalsh(raw)[0] < 0
    rows = _stratum_layout(measure, cov)
    means = []
    for draw in itertools.product(range(2), repeat=2):
        boot, _ = _resample_strata_one(measure, cov, rows, np.array(draw))
        means.append(_pooled(boot))
    np.testing.assert_allclose(
        np.cov(means, rowvar=False, ddof=0), np.full((2, 2), 0.5)
    )


def test_copied_spec_retains_coupled_decomposition():
    measure, cov = _fixture()
    rows = _stratum_layout(measure, cov)
    boot, spec = _resample_strata_one(measure, cov, rows, np.array([2, 0, 2]))
    block = _alter(spec, couple=False)
    args = (_mean, _Pair(0.0, 0.0), boot)
    cross = np.asarray(spec.cross_block(*args))
    assert abs(cross[0, 1]) > 0.01
    np.testing.assert_allclose(
        spec.covariance(*args), block.covariance(*args) + cross, rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("layout", ["finer", "crossing", "coarser"])
def test_other_sampling_partitions_rejected(layout):
    measure, cov = _fixture()
    strata = np.asarray(cov.design.stratum_ids, dtype=int)
    if layout == "finer":
        cluster = np.asarray(cov.design.psu_ids)
        n = cov.design.n_psu
    elif layout == "crossing":
        cluster = np.arange(len(strata)) % 3
        n = 3
    else:
        cluster = np.zeros(len(strata))
        n = 1
    with pytest.raises(ValueError, match="same partition"):
        mixed_bootstrap(
            _mean,
            _Pair(0.0, 0.0),
            measure,
            _alter(cov, cluster=cluster, n_clusters=n),
            n_boot=1,
            key=jax.random.PRNGKey(0),
        )


@pytest.mark.parametrize(
    "problem",
    [
        "fractional",
        "nonfinite",
        "empty_cluster",
        "empty_stratum",
        "split_psu",
        "split_cell",
    ],
)
def test_malformed_whole_stratum_layout_rejected(problem):
    measure, cov = _fixture()
    if problem in ("fractional", "nonfinite"):
        cluster = np.asarray(cov.sampling.cluster_ids).copy()
        cluster[0] = 0.5 if problem == "fractional" else np.nan
        cov = _alter(cov, cluster=cluster)
    elif problem == "empty_cluster":
        cov = _alter(cov, n_clusters=4)
    elif problem == "empty_stratum":
        cov = _alter(cov, strata=np.minimum(np.asarray(cov.design.stratum_ids), 1))
    elif problem == "split_psu":
        cell = np.asarray(cov.design.cell_ids).copy()
        cell[2] = cell[-1]
        cov = _alter(cov, cell=cell)
    else:
        # Keep every cell populated while making one span two strata.
        cell = np.asarray(cov.design.cell_ids).copy()
        cell[0], cell[-1] = cell[-1], cell[0]
        cov = _alter(cov, cell=cell)
    with pytest.raises(ValueError):
        _stratum_layout(measure, cov)


def test_random_fpc_rejected():
    measure, cov = _fixture()
    with pytest.raises(ValueError, match="fpc=False"):
        mixed_bootstrap(
            _mean,
            _Pair(0.0, 0.0),
            measure,
            _alter(cov, fpc=True),
            n_boot=1,
            key=jax.random.PRNGKey(0),
        )


def test_public_random_draws_preserve_ratios_events_and_labels(monkeypatch):
    measure, cov = _fixture(absent=True)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    seen = []

    def analytic_fit(*, measure, covariance, theta_init, **kwargs):
        assert not covariance.fixed_strata
        seen.append((measure, covariance))
        exposure = (
            np.asarray(measure.mask) * np.asarray(measure.weights)[:, None]
        ).sum(0)
        invalid = bool(np.any(exposure == 0))
        theta = theta_init if invalid else _Pair(*_pooled(measure))
        return SimpleNamespace(
            theta_hat=theta,
            objective_value=jnp.array(0.0),
            converged=not invalid,
            diagnostics=SimpleNamespace(
                N_j=SimpleNamespace(array=exposure), tau_saturated=invalid
            ),
        )

    monkeypatch.setattr(module, "estimate", analytic_fit)
    with pytest.warns(UserWarning, match="zero-support"):
        boot = mixed_bootstrap(
            _mean, _Pair(0.0, 0.0), measure, cov, n_boot=24, key=jax.random.PRNGKey(251)
        )
    assert 0 < boot.n_valid < 24
    valid = np.asarray(boot.convergence)
    np.testing.assert_array_equal(~valid, boot.zero_support)
    np.testing.assert_array_equal(~valid, boot.tau_saturated)
    assert np.isnan(np.asarray(boot.theta_boot.array)[~valid]).all()
    assert np.isnan(np.asarray(boot.J_boot)[~valid]).all()
    assert boot.param_names == ("theta_t", "theta_s")
    assert boot.coords["parameters"] == ("theta_t", "theta_s")
    for i in np.flatnonzero(valid):
        np.testing.assert_allclose(
            np.asarray(boot.theta_boot.array)[i], _pooled(seen[i][0])
        )
    assert any(len(world.x) != len(measure.x) for world, _ in seen)


def test_public_random_real_refits_match_ratio_oracle():
    measure, cov = _fixture()
    # Remove realized T stratum profiles for this regular CU fit control;
    # the preceding counterexample deliberately retains their #145 discrepancy.
    x = np.asarray(measure.x).copy()
    rows = _stratum_layout(measure, cov)
    for group in rows:
        t_rows = group[np.asarray(measure.mask)[group, 0] > 0]
        x[t_rows, 0] -= _pooled(measure, group)[0]
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=measure.mask, weights=measure.weights
    )
    boot = mixed_bootstrap(
        _mean,
        _Pair(0.0, 0.0),
        measure,
        cov,
        n_boot=2,
        key=jax.random.PRNGKey(251),
    )
    expected = []
    for key in jax.random.split(jax.random.PRNGKey(251), 2):
        drawn = np.asarray(jax.random.randint(key, (3,), 0, 3))
        expected.append(_pooled(measure, np.concatenate([rows[s] for s in drawn])))
    assert boot.n_valid == 2
    np.testing.assert_allclose(boot.theta_boot.array, expected, atol=1e-8, rtol=1e-8)
