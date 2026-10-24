"""Refit bootstrap for two explicitly scoped mixed covariance designs (#251).

Fixed strata use complete balanced paired bundles; random strata use whole
strata when sampling clusters and strata are the same partition. See ledger
Sections 18 and 27 for the different conditioning targets and exclusions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.covariance.stratified import DesignAwareCovariance, StratifiedCovariance
from emu_gmm.inference.cluster_bootstrap import (
    ClusterBootstrapResult,
    _cluster_row_indices,
    _refit_bootstrap_worlds,
    _resample_one,
)
from emu_gmm.inference.stratified_bootstrap import _cell_layout
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.penalty import PenaltyStrategy
from emu_gmm.types import (
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)

MixedBootstrapResult = ClusterBootstrapResult


def _ids(values: Any, n: int, n_rows: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.shape != (n_rows,)
        or raw.dtype.kind not in "biuf"
        or not np.isfinite(raw).all()
        or not np.array_equal(raw, np.rint(raw))
        or np.any(raw < 0)
        or np.any(raw >= n)
    ):
        raise ValueError(f"mixed_bootstrap: {name} must be integer IDs in [0,{n})")
    return raw.astype(np.int64)


def _paired_layout(
    measure: EmpiricalMeasure, covariance: DesignAwareCovariance
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Validate complete balanced bundles and return grouping metadata.

    This adds the PSU-to-bundle check missing from the fixed covariance
    constructor; that constructor checks bundle-to-stratum nesting only.
    """
    n_rows = int(measure.x.shape[0])
    if n_rows == 0:
        raise ValueError("mixed_bootstrap: the sample must have rows")
    design, sampling = covariance.design, covariance.sampling
    # Reuse the pure-design PSU/cell/stratum nesting guard (ledger Section 13).
    rows_by_psu, _, cell_of_psu, _ = _cell_layout(measure, design)
    psu = np.asarray(design.psu_ids, dtype=np.int64)
    cell = np.asarray(design.cell_ids, dtype=np.int64)
    stratum = np.asarray(design.stratum_ids, dtype=np.int64)
    bundle = _ids(sampling.cluster_ids, sampling.n_clusters, n_rows, "cluster_ids")

    rows_by_bundle = _cluster_row_indices(bundle, sampling.n_clusters)
    if any(rows.size == 0 for rows in rows_by_bundle):
        raise ValueError("mixed_bootstrap: each declared sampling bundle needs rows")
    bundle_of_psu = np.empty(design.n_psu, dtype=np.int64)
    for g, rows in enumerate(rows_by_psu):
        if np.unique(bundle[rows]).size != 1:
            raise ValueError("mixed_bootstrap: a PSU spans sampling bundles")
        bundle_of_psu[g] = bundle[rows[0]]

    stratum_of_bundle = np.empty(sampling.n_clusters, dtype=np.int64)
    for g, rows in enumerate(rows_by_bundle):
        if np.unique(stratum[rows]).size != 1:
            raise ValueError("mixed_bootstrap: a sampling bundle spans strata")
        stratum_of_bundle[g] = stratum[rows[0]]
    bundles_by_stratum = [
        np.flatnonzero(stratum_of_bundle == s) for s in range(design.n_strata)
    ]
    if any(bundles.size < 2 for bundles in bundles_by_stratum):
        raise ValueError(
            "mixed_bootstrap: every declared stratum needs at least two "
            "sampling bundles"
        )
    for s, bundles in enumerate(bundles_by_stratum):
        cells = np.unique(cell[stratum == s])
        if cells.size == 0:
            raise ValueError("mixed_bootstrap: each declared stratum needs cells")
        reference: np.ndarray | None = None
        for bundle_id in bundles:
            counts = np.bincount(
                cell_of_psu[bundle_of_psu == bundle_id], minlength=design.n_cells
            )
            if np.any(counts[cells] == 0):
                raise ValueError("mixed_bootstrap: every bundle must be complete")
            if reference is None:
                reference = counts
            elif not np.array_equal(counts, reference):
                raise ValueError(
                    "mixed_bootstrap: bundles must be balanced within strata"
                )
    return rows_by_bundle, bundles_by_stratum, psu


def _draw_bundles(bundles_by_stratum: list[np.ndarray], key: jax.Array) -> np.ndarray:
    """Draw the original bundle count within each stratum (#251 ledger)."""
    keys = (
        (key,)
        if len(bundles_by_stratum) == 1
        else jax.random.split(key, len(bundles_by_stratum))
    )
    drawn = []
    for bundles, stratum_key in zip(bundles_by_stratum, keys, strict=True):
        positions = np.asarray(
            jax.random.randint(stratum_key, (len(bundles),), 0, len(bundles))
        )
        drawn.append(bundles[positions])
    return np.concatenate(drawn)


def _resample_mixed_one(
    measure: EmpiricalMeasure,
    covariance: DesignAwareCovariance,
    rows_by_bundle: list[np.ndarray],
    psu: np.ndarray,
    drawn: np.ndarray,
) -> tuple[EmpiricalMeasure, DesignAwareCovariance]:
    """Clone paired bundles, renumber both ID levels, and rebuild the spec."""
    boot_measure, cluster_layout = _resample_one(measure, rows_by_bundle, drawn)
    source_rows = np.concatenate([rows_by_bundle[int(g)] for g in drawn])
    new_psu = np.empty(source_rows.size, dtype=np.int64)
    next_id = 0
    offset = 0
    for g in drawn:
        rows = rows_by_bundle[int(g)]
        _, within = np.unique(psu[rows], return_inverse=True)
        new_psu[offset : offset + rows.size] = next_id + within
        next_id += int(np.max(within)) + 1
        offset += rows.size
    design = covariance.design
    boot_design = StratifiedCovariance(
        psu_ids=jnp.asarray(new_psu, dtype=jnp.float64),
        cell_ids=jnp.asarray(design.cell_ids)[source_rows],
        stratum_ids=jnp.asarray(design.stratum_ids)[source_rows],
        n_psu=next_id,
        n_cells=int(design.n_cells),
        n_strata=int(design.n_strata),
        fpc=False,
    )
    boot_sampling = ClusteredCovariance(
        cluster_ids=cluster_layout.cluster_ids,
        n_clusters=int(len(drawn)),
        dof_correction=False,
        centered=covariance.sampling.centered,
    )
    boot_cov = DesignAwareCovariance.from_design_mask(
        boot_design,
        boot_sampling,
        covariance.design_moment_mask,
        couple=covariance.couple,
        fixed_strata=True,
    )
    return boot_measure, boot_cov


def _stratum_layout(
    measure: EmpiricalMeasure, covariance: DesignAwareCovariance
) -> list[np.ndarray]:
    """Validate the same sampling/stratum partition, irrespective of labels."""
    n_rows = int(measure.x.shape[0])
    if n_rows == 0:
        raise ValueError("mixed_bootstrap: the sample must have rows")
    design, sampling = covariance.design, covariance.sampling
    _cell_layout(measure, design)
    stratum = np.asarray(design.stratum_ids, dtype=np.int64)
    cluster = _ids(sampling.cluster_ids, sampling.n_clusters, n_rows, "cluster_ids")
    rows = _cluster_row_indices(stratum, design.n_strata)
    clusters = _cluster_row_indices(cluster, sampling.n_clusters)
    if any(group.size == 0 for group in rows + clusters):
        raise ValueError(
            "mixed_bootstrap: every declared stratum and cluster needs rows"
        )
    if any(np.unique(cluster[group]).size != 1 for group in rows) or any(
        np.unique(stratum[group]).size != 1 for group in clusters
    ):
        raise ValueError(
            "mixed_bootstrap: random mode requires sampling clusters and strata "
            "to be the same partition"
        )
    return rows


def _resample_strata_one(
    measure: EmpiricalMeasure,
    covariance: DesignAwareCovariance,
    rows_by_stratum: list[np.ndarray],
    drawn: np.ndarray,
) -> tuple[EmpiricalMeasure, DesignAwareCovariance]:
    """Clone entire strata with fresh IDs at all four grouping levels."""
    boot_measure, sampling = _resample_one(
        measure, rows_by_stratum, drawn, source_covariance=covariance.sampling
    )
    design = covariance.design
    old_psu, old_cell = np.asarray(design.psu_ids), np.asarray(design.cell_ids)
    psus, cells = [], []
    n_psu, n_cells = 0, 0
    for s in drawn:
        rows = rows_by_stratum[int(s)]
        unique_psu, psu_inverse = np.unique(old_psu[rows], return_inverse=True)
        unique_cell, cell_inverse = np.unique(old_cell[rows], return_inverse=True)
        psus.append(n_psu + psu_inverse)
        cells.append(n_cells + cell_inverse)
        n_psu += len(unique_psu)
        n_cells += len(unique_cell)
    boot_design = StratifiedCovariance(
        psu_ids=jnp.asarray(np.concatenate(psus), dtype=jnp.float64),
        cell_ids=jnp.asarray(np.concatenate(cells), dtype=jnp.float64),
        stratum_ids=sampling.cluster_ids,
        n_psu=n_psu,
        n_cells=n_cells,
        n_strata=int(len(drawn)),
        fpc=False,
    )
    boot_cov = DesignAwareCovariance.from_design_mask(
        boot_design,
        sampling,
        covariance.design_moment_mask,
        couple=covariance.couple,
        fixed_strata=False,
    )
    return boot_measure, boot_cov


def mixed_bootstrap(
    model: StructuralModel,
    theta_init: ParamsLike,
    measure: EmpiricalMeasure,
    covariance: DesignAwareCovariance,
    *,
    n_boot: int,
    key: jax.Array,
    weighting: WeightingStrategy | None = None,
    regularization: RegularizationStrategy | None = None,
    optimizer: Callable | None = None,
    penalty: PenaltyStrategy | None = None,
) -> MixedBootstrapResult:
    """Refit mixed GMM under its explicitly selected resampling experiment.

    With ``fixed_strata=True``, each sampling bundle has the same positive
    PSU-count vector across cells in its stratum. Draw the original bundle
    count with replacement within each stratum, preserving fixed cell counts.
    ``sampling.centered`` is retained as metadata; the fixed-mode #145
    exposure projection supplies centering for SS and TS. This mode requires
    ``sampling.dof_correction=False``.

    Default random mode requires sampling clusters and strata to be the same
    partition. Draw the original number of whole strata with replacement,
    retaining their realized sizes, cell counts, rows, masks and weights.
    All copied stratum/cluster/cell/PSU IDs are fresh. Sampling centering and
    dof flags are preserved; coordinate means remain pooled weighted ratios.
    This targets unconditional iid-stratum variation, assuming finite moments,
    no dominant stratum, positive limiting support and a regular consistent
    estimator. The default design TT can omit random common stratum profiles
    (#145); refit variability does not repair analytic TT, J chi-square
    calibration or general studentization.

    Both modes reject FPC: without-replacement support needs an explicit
    population/assignment law. Sparse support uses the shared NaN/event path.
    IDs inside arbitrary model-data columns are not rewritten; model grouping
    reductions must respect the intended relabeling/population contract.
    """
    if not isinstance(covariance, DesignAwareCovariance):
        raise TypeError("mixed_bootstrap requires DesignAwareCovariance")
    if covariance.design.fpc:
        raise ValueError("mixed_bootstrap requires design.fpc=False")
    if covariance.fixed_strata and covariance.sampling.dof_correction:
        raise ValueError("mixed_bootstrap requires sampling.dof_correction=False")
    if covariance.all_design:
        raise ValueError("mixed_bootstrap requires both design and sampling moments")
    if n_boot <= 0:
        raise ValueError(f"mixed_bootstrap: n_boot must be positive, got {n_boot}")
    keys = jax.random.split(key, n_boot)
    if covariance.fixed_strata:
        rows, strata, psu = _paired_layout(measure, covariance)
        drawn_all = [_draw_bundles(strata, keys[b]) for b in range(n_boot)]

        def resample_one(b):
            return _resample_mixed_one(measure, covariance, rows, psu, drawn_all[b])

    else:
        rows = _stratum_layout(measure, covariance)
        n_strata = len(rows)
        drawn_all = [
            np.asarray(jax.random.randint(k, (n_strata,), 0, n_strata)) for k in keys
        ]

        def resample_one(b):
            return _resample_strata_one(measure, covariance, rows, drawn_all[b])

    return _refit_bootstrap_worlds(
        model,
        theta_init,
        n_boot=n_boot,
        key=key,
        resample_one=resample_one,
        weighting=weighting,
        regularization=regularization,
        optimizer=optimizer,
        penalty=penalty,
        name="mixed_bootstrap",
    )


__all__ = ["mixed_bootstrap", "MixedBootstrapResult"]
