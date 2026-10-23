"""Finite-grid random-set carrier and known-DGP inversion checks (#180)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    EmpiricalMeasure,
    IIDCovariance,
    RandomSetLaw,
    k_confidence_set,
    profiled_k_confidence_set,
)
from emu_gmm.inference.confidence_set import _assemble_set


def _set(p: list[float], *, nonconverged: tuple[int, ...] = ()):
    grid = np.array([0.0, 1.0, 2.0])
    results = [SimpleNamespace(p_K=x) for x in p]
    mask = np.isin(np.arange(grid.size), nonconverged)
    return _assemble_set(
        grid, results, "K", 0.05, False, nonconverged=mask  # type: ignore[arg-type]
    )


def test_three_valued_coverage_capacity_and_conditioning():
    law = RandomSetLaw.from_k_confidence_sets(
        [
            _set([0.01, 0.9, 0.01]),
            _set([np.nan, 0.01, 0.01]),
            _set([0.01, 0.01, 0.01]),
            _set([0.01, 0.9, 0.01], nonconverged=(1,)),
        ]
    )
    assert law.point_verdicts(1.0) == (True, False, False, None)
    point = law.coverage(1.0)
    assert (point.n_total, point.n_used, point.n_decidable, point.n_hit) == (4, 4, 3, 1)
    assert (point.n_invalid, point.n_nonconverged, point.n_unknown) == (0, 1, 1)
    assert point.estimate == pytest.approx(1 / 3)
    assert law.prob(1.0) == point

    cap = law.capacity([0.0, 1.0])
    assert (cap.n_decidable, cap.n_hit, cap.n_invalid, cap.n_nonconverged) == (
        2,
        1,
        1,
        1,
    )
    assert cap.estimate == 0.5
    assert law.event_share("grid_empty").n_selected == 3
    assert law.event_share("grid_empty", negate=True).n_selected == 1
    selected = law.given("grid_empty")
    assert selected.origin_indices == (1, 2, 3)
    assert (selected.coverage(1.0).n_total, selected.coverage(1.0).n_used) == (4, 3)
    assert selected.coverage(1.0).n_decidable == 2
    empty = selected.given("grid_empty", negate=True)
    assert empty.n_used == 0
    assert empty.event_share("grid_empty").n_total == 0
    assert np.isnan(empty.coverage(1.0).estimate)


def test_grid_limits_point_summaries_and_immutable_input():
    draw = _set([0.01, 0.9, 0.01])
    law = RandomSetLaw.from_k_confidence_sets([draw])
    draw.in_set[1] = False
    assert law.point_verdicts(1.0) == (True,)
    with pytest.raises(ValueError, match="scanned grid"):
        law.coverage(1.1)
    with pytest.raises(ValueError, match="scanned grid"):
        law.capacity([3.0])
    edge_law = RandomSetLaw.from_k_confidence_sets([_set([0.9, 0.9, 0.01])])
    assert edge_law.draws[0].open_left
    with pytest.raises(ValueError, match="scanned grid"):
        edge_law.coverage(-0.1)
    with pytest.raises(ValueError, match="at least one"):
        law.capacity([])
    for query in (law.mean, law.cov, law.se, law.quantile):
        with pytest.raises(TypeError, match="no point-valued"):
            query()
    with pytest.raises(ValueError, match="intervals disagree"):
        RandomSetLaw.from_k_confidence_sets(
            [replace(law.draws[0], intervals=[(0.0, 2.0)])]
        )
    with pytest.raises(ValueError, match="grid/alpha/statistic"):
        RandomSetLaw.from_k_confidence_sets(
            [
                law.draws[0],
                _assemble_set(
                    np.array([1.0, 2.0]),
                    [SimpleNamespace(p_K=0.1)] * 2,
                    "K",
                    0.05,
                    False,
                ),
            ]
        )


def test_inert_roundtrip_and_grid_empty_one_draw(tmp_path):
    # A one-realization all-rejected scan is the K-Aggregators event shape;
    # it does not imply that the global parameter-space set is empty.
    law = RandomSetLaw.from_k_confidence_sets([_set([0.01, 0.02, 0.01])])
    path = tmp_path / "sets.npz"
    law.save(path)
    loaded = RandomSetLaw.load(path)
    assert loaded.event_share("grid_empty").n_selected == 1
    assert loaded.coverage(1.0).estimate == 0.0
    np.testing.assert_array_equal(loaded.draws[0].p_grid, law.draws[0].p_grid)
    np.testing.assert_array_equal(loaded.draws[0].in_set, law.draws[0].in_set)
    assert loaded.draws[0].topology == "empty"
    assert loaded.draws[0].intervals == []

    empty = loaded.given("grid_empty", negate=True)
    empty.save(path)
    restored = RandomSetLaw.load(path)
    assert restored.n_used == 0
    assert restored.source_n_total == 1
    assert np.isnan(restored.coverage(1.0).estimate)


def test_inert_roundtrip_preserves_unknowns_and_refuses_tampering(tmp_path):
    law = RandomSetLaw.from_k_confidence_sets(
        [
            _set([np.nan, 0.9, 0.01]),
            _set([0.01, 0.9, 0.01], nonconverged=(1,)),
        ]
    )
    path = tmp_path / "events.npz"
    law.save(path)
    restored = RandomSetLaw.load(path)
    assert restored.point_verdicts(0.0) == (None, False)
    assert restored.point_verdicts(1.0) == (True, None)
    assert restored.coverage(1.0).n_nonconverged == 1
    assert restored.coverage(0.0).n_invalid == 1

    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    arrays["in_set"][0, 0] = True  # invalid p-value cannot be accepted
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="in_set disagrees"):
        RandomSetLaw.load(path)


@jdc.pytree_dataclass
class _Theta:
    theta: float


def _linear_psi(x, params):
    y, regressor = x[0], x[1]
    resid = y - regressor * params.theta
    return jnp.array([resid * regressor, resid * regressor * regressor, resid])


def _linear_measure(seed: int, n: int = 220):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    return EmpiricalMeasure.from_arrays(jnp.asarray(np.column_stack([y, x])), M=3)


def test_known_dgp_fixed_curve_coverage_at_true_scanned_point():
    grid = np.array([0.6, 1.0, 1.4])
    sets = [
        k_confidence_set(
            lambda g: _Theta(theta=g),
            grid,
            _linear_measure(seed),
            IIDCovariance(),
            _linear_psi,
        )
        for seed in range(24)
    ]
    law = RandomSetLaw.from_k_confidence_sets(sets)
    report = law.coverage(1.0)
    direct = np.array([s.p_grid[1] > s.alpha for s in sets])
    assert report.n_total == report.n_used == report.n_decidable == 24
    assert report.n_invalid == report.n_nonconverged == 0
    assert report.n_hit == int(direct.sum())
    assert report.estimate == pytest.approx(float(direct.mean()))
    # A deliberately broad finite-rep falsification band around nominal 95%.
    assert 0.70 <= report.estimate <= 1.0


@jdc.pytree_dataclass
class _IVTheta:
    interest: float
    nuisance: float


def _iv_psi(x, params):
    y, a, b = x[0], x[1], x[2]
    return x[3:6] * (y - params.interest * a - params.nuisance * b)


def test_profiled_set_carrier_matches_scanned_verdict():
    rng = np.random.default_rng(180)
    n = 180
    z = rng.normal(size=(n, 3))
    a = z @ np.array([1.4, 1.2, 1.1]) + 0.4 * rng.normal(size=n)
    b = z @ np.array([1.0, 0.9, 1.1]) + 0.4 * rng.normal(size=n)
    y = 1.5 * a - 0.7 * b + 0.3 * rng.normal(size=n)
    measure = EmpiricalMeasure.from_arrays(
        jnp.asarray(np.column_stack([y, a, b, z])), M=3
    )
    grid = np.array([1.45, 1.5, 1.55])
    profiled = profiled_k_confidence_set(
        lambda g: _IVTheta(interest=g, nuisance=-0.7),
        grid,
        measure,
        IIDCovariance(),
        _iv_psi,
        profile=["interest"],
    )
    law = RandomSetLaw.from_k_confidence_sets([profiled])
    assert law.point_verdicts(1.5) == (bool(profiled.in_set[1]),)
    assert law.coverage(1.5).n_decidable == 1
    assert law.coverage(1.5).n_hit == int(profiled.in_set[1])
