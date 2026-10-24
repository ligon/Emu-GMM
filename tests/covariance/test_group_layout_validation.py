"""Covariance grouping boundaries, rounded compatibility and tracing (#191)."""

from __future__ import annotations

from dataclasses import replace

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
    build_estimator,
    k_statistic,
)
from emu_gmm.covariance._group_layout import validate_group_ids


@jdc.pytree_dataclass
class _Theta:
    value: float


def _model(x, theta):
    return x[:2] - theta.value


def _measure(n=4):
    return EmpiricalMeasure(
        x=jnp.arange(2 * n, dtype=float).reshape(n, 2) + 1,
        mask=jnp.ones((n, 2)),
        weights=jnp.ones(n),
    )


def _design(**kwargs):
    return StratifiedCovariance(
        **{
            "psu_ids": jnp.array([0.0, 1.0, 2.0, 3.0]),
            "cell_ids": jnp.array([0.0, 0.0, 1.0, 1.0]),
            "stratum_ids": jnp.zeros(4),
            "n_psu": 4,
            "n_cells": 2,
            "n_strata": 1,
            **kwargs,
        }
    )


def _run(covariance, measure=None):
    return covariance.covariance(
        _model, _Theta(0.0), _measure() if measure is None else measure
    )


@pytest.mark.parametrize(
    "ids",
    [
        [-1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, np.nan, 1.0, 1.0],
        [0.0, np.inf, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.0],
    ],
)
def test_clustered_invalid_values_fail_before_segment_drop(ids):
    covariance = ClusteredCovariance(cluster_ids=jnp.asarray(ids), n_clusters=2)
    with pytest.raises(ValueError, match="cluster_ids"):
        _run(covariance)
    with pytest.raises(ValueError, match="cluster_ids"):
        covariance._finite_cluster_correction(jnp.ones((4, 2)))
    # Even V supplied by a caller does not bypass the score-side guard.
    with pytest.raises(ValueError, match="cluster_ids"):
        k_statistic(_Theta(0.0), _measure(), covariance, _model, V=jnp.eye(2))


@pytest.mark.parametrize("count", [0, -1, True, np.bool_(True), 2.0, 2.5, 2**31])
def test_declared_count_requires_positive_integral_nonboolean(count):
    with pytest.raises(ValueError, match="declared count"):
        _run(ClusteredCovariance(cluster_ids=jnp.zeros(4), n_clusters=count))
    for field in ("n_psu", "n_cells", "n_strata"):
        with pytest.raises(ValueError, match="declared count"):
            _run(_design(**{field: count}))


@pytest.mark.parametrize(
    "ids",
    [
        jnp.zeros((4, 1)),
        jnp.zeros(3),
        jnp.asarray(0.0),
        np.array(["a"] * 4),
        jnp.zeros(4, dtype=complex),
    ],
)
def test_malformed_id_rank_dtype_and_row_length(ids):
    with pytest.raises(ValueError, match="cluster_ids"):
        _run(ClusteredCovariance(cluster_ids=ids, n_clusters=2))
    for field in ("psu_ids", "cell_ids", "stratum_ids"):
        with pytest.raises(ValueError, match=field):
            _run(_design(**{field: ids}))


@pytest.mark.parametrize("field", ["psu_ids", "cell_ids", "stratum_ids"])
@pytest.mark.parametrize("bad", [-1.0, 99.0, np.nan, np.inf, 0.2])
def test_stratified_invalid_values_checked_even_without_fpc(field, bad):
    covariance = _design()
    ids = np.asarray(getattr(covariance, field)).copy()
    ids[0] = bad
    with pytest.raises(ValueError, match=field):
        _run(replace(covariance, **{field: jnp.asarray(ids)}))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"psu_ids": jnp.array([0.0, 1.0, 0.0, 3.0])}, "PSU spans cells"),
        (
            {
                "psu_ids": jnp.array([0.0, 0.0, 2.0, 3.0]),
                "stratum_ids": jnp.array([0.0, 1.0, 1.0, 1.0]),
                "n_strata": 2,
            },
            "PSU spans strata",
        ),
        (
            {"stratum_ids": jnp.array([0.0, 1.0, 1.0, 1.0]), "n_strata": 2},
            "cell spans strata",
        ),
    ],
)
@pytest.mark.parametrize("fpc", [False, True])
def test_split_psu_cell_and_stratum_rejected(kwargs, message, fpc):
    with pytest.raises(ValueError, match=message):
        _run(_design(fpc=fpc, **kwargs))


def test_cell_split_that_previously_gave_negative_fpc_is_rejected():
    # Legacy max-stratum resolution assigned this shared cell to stratum 1:
    # H_cell=3, H_stratum1=1, so its FPC was 1 - 3/1 = -2.
    covariance = StratifiedCovariance(
        psu_ids=jnp.arange(3, dtype=float),
        cell_ids=jnp.zeros(3),
        stratum_ids=jnp.array([0.0, 0.0, 1.0]),
        n_psu=3,
        n_cells=1,
        n_strata=2,
        fpc=True,
    )
    with pytest.raises(ValueError, match="cell spans strata"):
        _run(covariance, _measure(3))


def test_near_integer_compatibility_empty_slots_and_zero_weight_support():
    measure = replace(_measure(), weights=jnp.array([1.0, 0.0, 2.0, 1.0]))
    ids = jnp.array([0.0, 1.0 - 1e-7, 3.0, 3.0])
    near = ClusteredCovariance(
        cluster_ids=ids, n_clusters=np.int64(5), dof_correction=True
    )
    exact = replace(near, cluster_ids=jnp.round(ids))
    np.testing.assert_array_equal(_run(near, measure), _run(exact, measure))
    np.testing.assert_array_equal(
        near._finite_cluster_correction(measure.mask), jnp.full((2, 2), 1.5)
    )
    pure = _design(fpc=True)
    gapped = replace(
        pure,
        psu_ids=jnp.array([0.0, 2.0, 3.0, 4.0]),
        n_psu=6,
        cell_ids=jnp.array([0.0, 0.0, 2.0, 2.0]),
        n_cells=4,
        n_strata=3,
    )
    np.testing.assert_array_equal(_run(pure, measure), _run(gapped, measure))
    jittered = replace(pure, psu_ids=pure.psu_ids + 1e-7, cell_ids=pure.cell_ids + 1e-7)
    np.testing.assert_array_equal(_run(pure, measure), _run(jittered, measure))


@pytest.mark.parametrize("kind", ["cluster", "stratified", "mixed"])
def test_concrete_closed_over_and_dynamic_valid_tracing_parity(kind):
    measure = _measure()
    sampling = ClusteredCovariance(
        cluster_ids=jnp.array([0.0, 0.0, 1.0, 1.0]), n_clusters=3, dof_correction=True
    )
    covariance = sampling if kind == "cluster" else _design(fpc=True)
    if kind == "mixed":
        covariance = DesignAwareCovariance.from_design_mask(
            covariance, sampling, jnp.array([1.0, 0.0])
        )
    expected = _run(covariance)
    np.testing.assert_allclose(
        jax.jit(lambda: _run(covariance))(), expected, atol=1e-13
    )
    np.testing.assert_allclose(jax.jit(_run)(covariance), expected, atol=1e-13)
    run = jax.jit(
        jax.vmap(lambda x, cov: _run(cov, replace(measure, x=x)), in_axes=(0, None))
    )
    np.testing.assert_allclose(
        run(jnp.stack([measure.x, measure.x]), covariance),
        jnp.stack([expected, expected]),
        atol=1e-13,
    )


def test_traced_shape_checks_and_abstract_layouts_do_not_convert_values():
    assert (
        validate_group_ids(
            jax.ShapeDtypeStruct((4,), jnp.float64), 2, name="ids", n_rows=4
        )
        is None
    )
    with pytest.raises(ValueError, match="one ID per row"):
        jax.jit(lambda ids: _run(ClusteredCovariance(cluster_ids=ids, n_clusters=2)))(
            jnp.zeros(3)
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        jax.jit(lambda ids: _run(ClusteredCovariance(cluster_ids=ids, n_clusters=2)))(
            jnp.zeros((4, 1))
        )
    # Tree reconstruction itself may temporarily contain arbitrary sentinel
    # leaves; validation belongs at evaluation rather than __post_init__.
    tree = jax.tree_util.tree_map(lambda _: object(), _design())
    assert tree.n_psu == 4


def test_dtype_roundoff_tolerance_remains_capped_for_large_float_ids():
    np.testing.assert_array_equal(
        validate_group_ids(np.array([1.0 - 1e-7], dtype=np.float32), 2, name="ids"),
        np.array([1]),
    )
    with pytest.raises(ValueError, match="integer-valued"):
        validate_group_ids(
            np.array([1000000.25], dtype=np.float32), 1000002, name="ids"
        )


def test_factory_anchor_rejects_malformed_layout_before_solver():
    covariance = _design(psu_ids=jnp.array([0.0, 1.0, 2.0, 4.0]))
    with pytest.raises(ValueError, match="psu_ids.*rounded IDs"):
        build_estimator(
            _model, measure=_measure(), covariance=covariance, theta_init=_Theta(0.0)
        )


@pytest.mark.parametrize("fixed", [False, True])
def test_mixed_cross_and_diagnostics_validate_used_sampling_ids(fixed):
    sampling = ClusteredCovariance(
        cluster_ids=jnp.array([0.0, 0.0, 1.0, 1.0]), n_clusters=2
    )
    covariance = DesignAwareCovariance.from_design_mask(
        _design(), sampling, jnp.array([1.0, 0.0]), fixed_strata=fixed
    )
    malformed = replace(
        covariance,
        sampling=replace(sampling, cluster_ids=jnp.array([0.0, 0.0, 1.0, 2.0])),
    )
    with pytest.raises(ValueError, match="cluster_ids"):
        _run(malformed)
    with pytest.raises(ValueError, match="cluster_ids"):
        malformed.cross_block(_model, _Theta(0.0), _measure())
    if fixed:
        for diagnostic in (
            malformed.fixed_cell_pair_diagnostics,
            malformed.fixed_stratum_pair_diagnostics,
        ):
            with pytest.raises(ValueError, match="cluster_ids"):
                diagnostic(_measure())
    all_design = replace(malformed, design_moment_mask=jnp.ones(2), all_design=True)
    np.testing.assert_array_equal(_run(all_design), _run(_design()))


def test_fixed_factory_retains_stricter_exact_integer_contract():
    sampling = ClusteredCovariance(
        cluster_ids=jnp.array([0.0, 0.0, 1.0 - 1e-7, 1.0]), n_clusters=2
    )
    with pytest.raises(ValueError, match="integer-valued"):
        DesignAwareCovariance.from_design_mask(
            _design(), sampling, jnp.array([1.0, 0.0]), fixed_strata=True
        )
