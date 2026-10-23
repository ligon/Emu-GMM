"""Exact null recentring for the whole-cluster refit J law (issue 185)."""

from __future__ import annotations

import importlib
from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import ClusteredCovariance, EmpiricalMeasure, estimate
from emu_gmm.inference import cluster_bootstrap
from emu_gmm.inference.cluster_bootstrap import _exact_null_offset


@jdc.pytree_dataclass
class _Mean:
    theta: float


def _psi(x, params):
    return jnp.asarray([x[0] - params.theta, x[1] - params.theta])


def _law_fixture(
    y, *, mask=None, weights=None, cluster_ids=None, centered=True, dof=False
):
    y = np.asarray(y, dtype=np.float64)
    if cluster_ids is None:
        cluster_ids = np.arange(len(y))
    G = int(np.max(cluster_ids)) + 1
    if mask is None:
        mask = np.ones((G, 2))
    if weights is None:
        weights = np.ones(G)
    measure = EmpiricalMeasure(
        x=jnp.asarray(y), mask=jnp.asarray(mask), weights=jnp.asarray(weights)
    )
    covariance = ClusteredCovariance(
        cluster_ids=jnp.asarray(cluster_ids, dtype=jnp.float64),
        n_clusters=G,
        centered=centered,
        dof_correction=dof,
    )
    return measure, covariance


def _world_mean(exposure, total):
    G = exposure.shape[0]
    return np.mean(
        [
            total[list(world)].sum(axis=0) / exposure[list(world)].sum(axis=0)
            for world in product(range(G), repeat=G)
        ],
        axis=0,
    )


def _original_result(measure, *, objective=2.0):
    theta_hat = _Mean(theta=0.0)
    mean, _, _, N_j = measure.expectation_and_contributions(_psi, theta_hat)
    return SimpleNamespace(
        theta_hat=theta_hat,
        objective_value=jnp.asarray(objective),
        converged=True,
        n_overid=1,
        diagnostics=SimpleNamespace(
            moment_residual=SimpleNamespace(array=mean),
            N_j=SimpleNamespace(array=N_j),
        ),
    )


def test_exact_offset_matches_all_small_cluster_worlds():
    two_n = np.asarray([[1.0], [2.0]])
    two_u = np.asarray([[0.0], [2.0]])
    np.testing.assert_allclose(_exact_null_offset(two_n, two_u), [7 / 12], atol=1e-12)
    assert float(two_u.sum() / two_n.sum()) == pytest.approx(2 / 3)

    exposure = np.asarray([[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]])
    total = np.asarray([[0.0, 6.0], [2.0, -1.0], [-4.0, 2.0]])
    np.testing.assert_allclose(
        _exact_null_offset(exposure, total),
        _world_mean(exposure, total),
        atol=1e-11,
    )
    assert not np.allclose(
        _exact_null_offset(exposure, total), total.sum(axis=0) / exposure.sum(axis=0)
    )


def test_original_bootstrap_forwards_centering_and_dof(monkeypatch):
    measure, covariance = _law_fixture([[1.0, 2.0], [0.0, -1.0], [2.0, 0.0]], dof=True)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    seen = []

    def fake_estimate(*, covariance, theta_init, **kwargs):
        seen.append((covariance.centered, covariance.dof_correction))
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(0.0),
            converged=True,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    cluster_bootstrap(
        _psi, _Mean(theta=0.0), measure, covariance, n_boot=3, key=jax.random.PRNGKey(1)
    )
    assert seen == [(True, True)] * 3


def test_frozen_offset_reaches_every_refit_under_unequal_support(monkeypatch):
    y = [[0.0, 1.0], [1.0, 2.0], [2.0, 7.0], [3.0, 4.0]]
    mask = [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    weights = [1.0, 1.0, 2.0, 3.0]
    cluster_ids = [0, 1, 1, 2]
    measure, covariance = _law_fixture(
        y, mask=mask, weights=weights, cluster_ids=cluster_ids, dof=True
    )
    original = _original_result(measure)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    observed_models = []

    def fake_estimate(*, model, covariance, theta_init, **kwargs):
        observed_models.append(
            (
                np.asarray(model(jnp.asarray([10.0, 20.0]), theta_init)),
                covariance.centered,
                covariance.dof_correction,
            )
        )
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(1.0),
            converged=True,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    boot = cluster_bootstrap(
        _psi,
        _Mean(theta=0.0),
        measure,
        covariance,
        n_boot=3,
        key=jax.random.PRNGKey(2),
        null_result=original,
    )
    row_exposure = np.asarray(mask) * np.asarray(weights)[:, None]
    exposure = np.zeros((3, 2))
    total = np.zeros((3, 2))
    np.add.at(exposure, cluster_ids, row_exposure)
    np.add.at(total, cluster_ids, row_exposure * np.asarray(y))
    expected = _world_mean(exposure, total)
    np.testing.assert_allclose(boot.null_offset, expected, atol=1e-11)
    for shifted, centered, dof in observed_models:
        np.testing.assert_allclose(shifted, np.asarray([10.0, 20.0]) - expected)
        assert centered and dof
    assert boot.j_pvalue() == pytest.approx(1.0 / 4.0)


def test_null_J_pvalue_excludes_nonconverged_rows_and_counts_valid_only(monkeypatch):
    measure, covariance = _law_fixture([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    original = _original_result(measure)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")
    outcomes = [(True, 3.0), (False, 100.0), (True, 1.0)]

    def fake_estimate(*, theta_init, **kwargs):
        converged, J = outcomes.pop(0)
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(J),
            converged=converged,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    with pytest.warns(UserWarning, match="1 of 3 replicate refits did not converge"):
        boot = cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=3,
            key=jax.random.PRNGKey(19),
            null_result=original,
        )
    assert boot.n_valid == 2
    assert boot.n_nonconverged == 1
    np.testing.assert_array_equal(boot.convergence, [True, False, True])
    assert np.isnan(np.asarray(boot.theta_boot.array)[1]).all()
    assert np.isnan(np.asarray(boot.J_boot)[1])
    assert boot.j_pvalue() == pytest.approx((1 + 1) / (1 + 2))


def test_null_result_remains_a_stackable_pytree(monkeypatch):
    measure, covariance = _law_fixture([[0.0, 1.0], [1.0, 2.0]])
    original = _original_result(measure)
    module = importlib.import_module("emu_gmm.inference.cluster_bootstrap")

    def fake_estimate(*, theta_init, **kwargs):
        return SimpleNamespace(
            theta_hat=theta_init,
            objective_value=jnp.asarray(1.0),
            converged=True,
            diagnostics=None,
        )

    monkeypatch.setattr(module, "estimate", fake_estimate)
    results = [
        cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=2,
            key=jax.random.PRNGKey(seed),
            null_result=original,
        )
        for seed in (1, 2)
    ]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *results)
    assert stacked.J_boot.shape == (2, 2)
    assert stacked.null_offset.shape == (2, 2)
    assert stacked.observed_J.shape == (2,)


def test_recentring_refuses_unusable_domain():
    y = [[0.0, 1.0], [1.0, 2.0]]
    measure, covariance = _law_fixture(y, weights=[1.0, -1.0])
    original = _original_result(measure, objective=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=1,
            key=jax.random.PRNGKey(1),
            null_result=original,
        )
    measure, covariance = _law_fixture(y, mask=[[1, 1], [1, 0]])
    original = _original_result(measure, objective=1.0)
    with pytest.raises(ValueError, match="positive cluster exposure"):
        cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=1,
            key=jax.random.PRNGKey(1),
            null_result=original,
        )

    measure, covariance = _law_fixture(y, cluster_ids=[0.0, 1.5])
    original = _original_result(measure, objective=1.0)
    with pytest.raises(ValueError, match="invalid cluster IDs"):
        cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=1,
            key=jax.random.PRNGKey(1),
            null_result=original,
        )


@pytest.mark.parametrize("field", ["moment_residual", "N_j"])
def test_recentring_refuses_mismatched_original_fit(field):
    measure, covariance = _law_fixture([[0.0, 1.0], [1.0, 2.0]])
    original = _original_result(measure)
    value = getattr(original.diagnostics, field)
    setattr(original.diagnostics, field, SimpleNamespace(array=value.array + 1.0))
    with pytest.raises(ValueError, match="do not match"):
        cluster_bootstrap(
            _psi,
            _Mean(theta=0.0),
            measure,
            covariance,
            n_boot=1,
            key=jax.random.PRNGKey(1),
            null_result=original,
        )


def test_package_J_matches_centered_contrast_formula():
    rng = np.random.default_rng(185)
    y = rng.normal(size=(12, 2))
    measure, covariance = _law_fixture(y)
    fit = estimate(_psi, measure, covariance=covariance, theta_init=_Mean(theta=0.0))
    assert fit.converged
    key = jax.random.PRNGKey(185)
    recentered = cluster_bootstrap(
        _psi, fit.theta_hat, measure, covariance, n_boot=4, key=key, null_result=fit
    )
    ordinary = cluster_bootstrap(
        _psi, fit.theta_hat, measure, covariance, n_boot=4, key=key
    )
    d = y[:, 0] - y[:, 1]
    G = len(d)
    observed = (G * d.mean()) ** 2 / np.sum((d - d.mean()) ** 2)
    assert float(fit.objective_value) == pytest.approx(observed, rel=1e-6)
    keys = jax.random.split(key, 4)
    for b, draw_key in enumerate(keys):
        drawn = np.asarray(jax.random.randint(draw_key, (G,), 0, G))
        d_boot = d[drawn]
        denominator = np.sum((d_boot - d_boot.mean()) ** 2)
        expected_recentered = (G * (d_boot.mean() - d.mean())) ** 2 / denominator
        expected_ordinary = (G * d_boot.mean()) ** 2 / denominator
        assert float(recentered.J_boot[b]) == pytest.approx(
            expected_recentered, rel=1e-5
        )
        assert float(ordinary.J_boot[b]) == pytest.approx(expected_ordinary, rel=1e-5)
    assert recentered.n_valid == ordinary.n_valid == 4
    assert not np.allclose(recentered.J_boot, ordinary.J_boot)
    with pytest.raises(ValueError, match="null-recentred"):
        ordinary.j_pvalue()


def test_gaussian_null_calibration_distinguishes_ordinary_J_law():
    """Analytic reduction cross-checked against package J above."""
    rng = np.random.default_rng(185)
    R, B, G = 500, 500, 20
    differences = rng.normal(scale=np.sqrt(2), size=(R, G))
    original_mean = differences.mean(axis=1)
    observed = (G * original_mean) ** 2 / np.sum(
        (differences - original_mean[:, None]) ** 2, axis=1
    )
    index = rng.integers(0, G, size=(R, B, G))
    worlds = np.take_along_axis(differences[:, None, :], index, axis=2)
    world_mean = worlds.mean(axis=2)
    denominator = np.sum((worlds - world_mean[:, :, None]) ** 2, axis=2)
    assert np.all(denominator > 0)
    recentered_J = (G * (world_mean - original_mean[:, None])) ** 2 / denominator
    ordinary_J = (G * world_mean) ** 2 / denominator
    p_recentered = (1 + np.sum(recentered_J >= observed[:, None], axis=1)) / (B + 1)
    p_ordinary = (1 + np.sum(ordinary_J >= observed[:, None], axis=1)) / (B + 1)
    rejection_recentered = float(np.mean(p_recentered <= 0.05))
    rejection_ordinary = float(np.mean(p_ordinary <= 0.05))
    assert 0.03 < rejection_recentered < 0.09
    assert rejection_ordinary < 0.01
