"""Pure stratified PSU refit bootstrap (#155)."""

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
    EmpiricalMeasure,
    StratifiedCovariance,
    estimate,
    stratified_bootstrap,
)
from emu_gmm.inference.cluster_bootstrap import _resample_one
from emu_gmm.inference.stratified_bootstrap import (
    _cell_layout,
    _draw_psu_indices,
    _resample_stratified_one,
)


@jdc.pytree_dataclass
class _Mean:
    theta: float


def _mean_psi(x, params):
    return jnp.array([x[0] - params.theta])


def _fixture(counts=(2, 3, 4, 5), *, rows_per_psu=2, seed=155):
    """Unbalanced 2-stratum x 2-arm cells with fixed cell mean shifts."""
    rng = np.random.default_rng(seed)
    y, psu, cell, stratum = [], [], [], []
    g = 0
    for c, count in enumerate(counts):
        for _ in range(count):
            value = (2.0, -1.0, 4.0, 0.5)[c] + rng.normal()
            for _ in range(rows_per_psu):
                y.append(value)
                psu.append(g)
                cell.append(c)
                stratum.append(c // 2)
            g += 1
    n = len(y)
    measure = EmpiricalMeasure(
        x=jnp.asarray(np.asarray(y)[:, None]),
        mask=jnp.ones((n, 1)),
        weights=jnp.ones(n),
    )
    cov = StratifiedCovariance(
        psu_ids=jnp.asarray(psu, dtype=jnp.float64),
        cell_ids=jnp.asarray(cell, dtype=jnp.float64),
        stratum_ids=jnp.asarray(stratum, dtype=jnp.float64),
        n_psu=g,
        n_cells=len(counts),
        n_strata=(len(counts) + 1) // 2,
        fpc=False,
    )
    return measure, cov


def test_unbalanced_cells_preserve_each_psu_count_and_metadata():
    measure, cov = _fixture()
    rows, cells, cell_of_psu, stratum_of_psu = _cell_layout(measure, cov)
    assert tuple(len(psus) for psus in cells) == (2, 3, 4, 5)
    for seed in range(8):
        drawn = _draw_psu_indices(cells, jax.random.PRNGKey(seed))
        assert tuple(int(np.sum(cell_of_psu[drawn] == c)) for c in range(4)) == (
            2,
            3,
            4,
            5,
        )
        boot_measure, boot_cov = _resample_stratified_one(
            measure, cov, rows, cell_of_psu, stratum_of_psu, drawn
        )
        ids = np.asarray(boot_cov.psu_ids, dtype=int)
        np.testing.assert_array_equal(np.unique(ids), np.arange(cov.n_psu))
        for new_g, old_g in enumerate(drawn):
            g_rows = np.flatnonzero(ids == new_g)
            assert len(g_rows) == len(rows[int(old_g)])
            assert np.all(np.asarray(boot_cov.cell_ids)[g_rows] == cell_of_psu[old_g])
            assert np.all(
                np.asarray(boot_cov.stratum_ids)[g_rows] == stratum_of_psu[old_g]
            )
        assert boot_measure.x.shape[0] == measure.x.shape[0]


def test_one_cell_draw_stream_and_refit_plumbing_match_cluster(monkeypatch):
    measure, cov = _fixture(counts=(5,), rows_per_psu=2)
    rows, cells, cell_of_psu, stratum_of_psu = _cell_layout(measure, cov)
    key = jax.random.PRNGKey(155)
    for draw_key in jax.random.split(key, 4):
        drawn = _draw_psu_indices(cells, draw_key)
        expected = np.asarray(jax.random.randint(draw_key, (5,), 0, 5))
        np.testing.assert_array_equal(drawn, expected)
        strat_measure, strat_cov = _resample_stratified_one(
            measure, cov, rows, cell_of_psu, stratum_of_psu, drawn
        )
        cluster_measure, cluster_cov = _resample_one(measure, rows, drawn)
        np.testing.assert_array_equal(strat_measure.x, cluster_measure.x)
        np.testing.assert_array_equal(strat_measure.mask, cluster_measure.mask)
        np.testing.assert_array_equal(strat_measure.weights, cluster_measure.weights)
        np.testing.assert_array_equal(strat_cov.psu_ids, cluster_cov.cluster_ids)

    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    seen = []

    def fake_estimate(*, measure, theta_init, **kwargs):
        seen.append(np.asarray(measure.x))
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(0.0),
            converged=True,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    strat = stratified_bootstrap(
        _mean_psi, _Mean(theta=1.0), measure, cov, n_boot=3, key=key
    )
    cluster = module.cluster_bootstrap(
        _mean_psi,
        _Mean(theta=1.0),
        measure,
        ClusteredCovariance(cluster_ids=cov.psu_ids, n_clusters=cov.n_psu),
        n_boot=3,
        key=key,
    )
    for a, b in zip(seen[:3], seen[3:], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(strat.theta_boot.array, cluster.theta_boot.array)
    assert strat.param_names == cluster.param_names == ("theta",)


def test_nonconverged_refit_is_nan_and_fpc_refused(monkeypatch):
    measure, cov = _fixture(counts=(3,), rows_per_psu=1)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    calls = {"n": 0}

    def fake_estimate(*, theta_init, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(0.0),
            converged=calls["n"] != 1,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    with pytest.warns(UserWarning, match="stratified_bootstrap: 1 of 2"):
        boot = stratified_bootstrap(
            _mean_psi,
            _Mean(theta=1.0),
            measure,
            cov,
            n_boot=2,
            key=jax.random.PRNGKey(1),
        )
    assert boot.n_nonconverged == 1
    assert np.isnan(np.asarray(boot.theta_boot.array)[0]).all()
    assert np.isnan(np.asarray(boot.J_boot)[0])
    assert np.asarray(boot.convergence).tolist() == [False, True]
    with pytest.raises(ValueError, match="fpc=False"):
        stratified_bootstrap(
            _mean_psi,
            _Mean(theta=1.0),
            measure,
            StratifiedCovariance(
                psu_ids=cov.psu_ids,
                cell_ids=cov.cell_ids,
                stratum_ids=cov.stratum_ids,
                n_psu=cov.n_psu,
                n_cells=cov.n_cells,
                n_strata=cov.n_strata,
                fpc=True,
            ),
            n_boot=1,
            key=jax.random.PRNGKey(1),
        )


def test_layout_rejects_psu_spanning_cells():
    measure, cov = _fixture(counts=(2, 3, 4, 5), rows_per_psu=2)
    bad_cells = np.asarray(cov.cell_ids).copy()
    bad_cells[0] = 1
    bad_cov = StratifiedCovariance(
        psu_ids=cov.psu_ids,
        cell_ids=jnp.asarray(bad_cells),
        stratum_ids=cov.stratum_ids,
        n_psu=cov.n_psu,
        n_cells=cov.n_cells,
        n_strata=cov.n_strata,
    )
    with pytest.raises(ValueError, match="PSU spans cells"):
        _cell_layout(measure, bad_cov)

    empty_cell_cov = StratifiedCovariance(
        psu_ids=cov.psu_ids,
        cell_ids=cov.cell_ids,
        stratum_ids=cov.stratum_ids,
        n_psu=cov.n_psu,
        n_cells=cov.n_cells + 1,
        n_strata=cov.n_strata,
    )
    with pytest.raises(ValueError, match="each declared cell"):
        _cell_layout(measure, empty_cell_cov)


@pytest.mark.slow
def test_known_dgp_bootstrap_se_tracks_analytic_stratified_se():
    counts = (12, 15, 18, 21)
    measure, cov = _fixture(counts=counts, rows_per_psu=1, seed=155)
    y = np.asarray(measure.x)[:, 0]
    theta_init = _Mean(theta=float(y.mean()))
    result = estimate(_mean_psi, measure, covariance=cov, theta_init=theta_init)
    analytic_se = float(np.asarray(result.asymptotic().se())[0])
    numerator = 0.0
    offset = 0
    for count in counts:
        cell_y = y[offset : offset + count]
        numerator += count / (count - 1) * np.sum((cell_y - cell_y.mean()) ** 2)
        offset += count
    assert analytic_se == pytest.approx(np.sqrt(numerator) / len(y), rel=1e-6)

    boot = stratified_bootstrap(
        _mean_psi,
        result.theta_hat,
        measure,
        cov,
        n_boot=60,
        key=jax.random.PRNGKey(155),
    )
    valid = np.asarray(boot.convergence, dtype=bool)
    assert valid.sum() >= 55
    boot_se = float(np.std(np.asarray(boot.theta_boot.array)[valid, 0], ddof=1))
    assert 0.65 < boot_se / analytic_se < 1.35
