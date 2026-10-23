"""Refit bootstrap for pure stratified, cell-centered PSU covariance (#155).

The design cell is ``(stratum, arm)``. Resampling PSUs inside each cell
preserves its PSU count, matching the centering unit of
:class:`StratifiedCovariance`. This first contract supports ``fpc=False``.
See ledger Section 13.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from emu_gmm.covariance.stratified import StratifiedCovariance
from emu_gmm.inference.cluster_bootstrap import (
    ClusterBootstrapResult,
    _cluster_row_indices,
    _refit_bootstrap_worlds,
    _resample_one,
)
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.penalty import PenaltyStrategy
from emu_gmm.types import (
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)

StratifiedBootstrapResult = ClusterBootstrapResult


def _integer_ids(values: Any, n: int, n_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if (
        arr.shape != (n_rows,)
        or not np.isfinite(arr).all()
        or not np.array_equal(arr, np.rint(arr))
        or np.any(arr < 0)
        or np.any(arr >= n)
    ):
        raise ValueError(f"stratified_bootstrap: {name} must be integer IDs in [0,{n})")
    return arr.astype(np.int64)


def _cell_layout(
    measure: EmpiricalMeasure, covariance: StratifiedCovariance
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Validate one PSU/cell/stratum per row and group populated PSUs."""
    n_rows = int(measure.x.shape[0])
    psu = _integer_ids(covariance.psu_ids, covariance.n_psu, n_rows, "psu_ids")
    cell = _integer_ids(covariance.cell_ids, covariance.n_cells, n_rows, "cell_ids")
    stratum = _integer_ids(
        covariance.stratum_ids, covariance.n_strata, n_rows, "stratum_ids"
    )
    rows_by_psu = _cluster_row_indices(psu, int(covariance.n_psu))
    if any(rows.size == 0 for rows in rows_by_psu):
        raise ValueError("stratified_bootstrap: each declared PSU must have rows")
    cell_of_psu = np.empty(covariance.n_psu, dtype=np.int64)
    stratum_of_psu = np.empty(covariance.n_psu, dtype=np.int64)
    for g, rows in enumerate(rows_by_psu):
        if np.unique(cell[rows]).size != 1 or np.unique(stratum[rows]).size != 1:
            raise ValueError("stratified_bootstrap: a PSU spans cells or strata")
        cell_of_psu[g] = cell[rows[0]]
        stratum_of_psu[g] = stratum[rows[0]]
    psus_by_cell = [np.flatnonzero(cell_of_psu == c) for c in range(covariance.n_cells)]
    if any(psus.size == 0 for psus in psus_by_cell):
        raise ValueError("stratified_bootstrap: each declared cell needs a PSU")
    for psus in psus_by_cell:
        if np.unique(stratum_of_psu[psus]).size != 1:
            raise ValueError("stratified_bootstrap: a cell spans strata")
    return rows_by_psu, psus_by_cell, cell_of_psu, stratum_of_psu


def _draw_psu_indices(psus_by_cell: list[np.ndarray], key: jax.Array) -> np.ndarray:
    """Draw H_c PSU indices inside each cell; one cell uses the cluster stream."""
    cell_keys = (
        (key,) if len(psus_by_cell) == 1 else jax.random.split(key, len(psus_by_cell))
    )
    drawn = []
    for psus, cell_key in zip(psus_by_cell, cell_keys, strict=True):
        positions = np.asarray(jax.random.randint(cell_key, (len(psus),), 0, len(psus)))
        drawn.append(psus[positions])
    return np.concatenate(drawn)


def _resample_stratified_one(
    measure: EmpiricalMeasure,
    covariance: StratifiedCovariance,
    rows_by_psu: list[np.ndarray],
    cell_of_psu: np.ndarray,
    stratum_of_psu: np.ndarray,
    drawn: np.ndarray,
) -> tuple[EmpiricalMeasure, StratifiedCovariance]:
    """Reuse the tested row assembly and rebuild cell/stratum metadata."""
    boot_measure, cluster_layout = _resample_one(measure, rows_by_psu, drawn)
    cell_ids = np.concatenate(
        [np.full(rows_by_psu[int(g)].size, cell_of_psu[int(g)]) for g in drawn]
    )
    stratum_ids = np.concatenate(
        [np.full(rows_by_psu[int(g)].size, stratum_of_psu[int(g)]) for g in drawn]
    )
    boot_cov = StratifiedCovariance(
        psu_ids=cluster_layout.cluster_ids,
        cell_ids=jnp.asarray(cell_ids, dtype=jnp.float64),
        stratum_ids=jnp.asarray(stratum_ids, dtype=jnp.float64),
        n_psu=int(len(drawn)),
        n_cells=int(covariance.n_cells),
        n_strata=int(covariance.n_strata),
        fpc=False,
    )
    return boot_measure, boot_cov


def stratified_bootstrap(
    model: StructuralModel,
    theta_init: ParamsLike,
    measure: EmpiricalMeasure,
    covariance: StratifiedCovariance,
    *,
    n_boot: int,
    key: jax.Array,
    weighting: WeightingStrategy | None = None,
    regularization: RegularizationStrategy | None = None,
    optimizer: Callable | None = None,
    penalty: PenaltyStrategy | None = None,
) -> StratifiedBootstrapResult:
    """Refit GMM after sampling PSUs within each fixed stratum-by-arm cell.

    Each replicate draws exactly as many PSUs in cell ``c`` as the original
    cell contains, with replacement. Repeated PSUs receive distinct new IDs;
    the resampled rows, masks, weights, cells, and strata travel together.
    ``fpc=True`` is refused because this pairs bootstrap targets the
    superpopulation variance without a finite-population adjustment.
    Every declared PSU and cell must be populated; empty declared slots
    cannot be resampled while preserving fixed cell counts.

    The result has the same labelled ``theta_boot``/``J_boot`` and NaN-on-
    nonconvergence contract as :func:`cluster_bootstrap`. With one cell,
    the *drawn PSU indices* match its cluster stream for the same key;
    centered and uncentered covariance estimates may still differ.
    """
    if not isinstance(covariance, StratifiedCovariance):
        raise TypeError("stratified_bootstrap requires StratifiedCovariance")
    if covariance.fpc:
        raise ValueError("stratified_bootstrap requires fpc=False")
    if n_boot <= 0:
        raise ValueError(f"stratified_bootstrap: n_boot must be positive, got {n_boot}")
    rows, cells, cell_of_psu, stratum_of_psu = _cell_layout(measure, covariance)
    keys = jax.random.split(key, n_boot)
    drawn_all = [_draw_psu_indices(cells, keys[b]) for b in range(n_boot)]
    return _refit_bootstrap_worlds(
        model,
        theta_init,
        n_boot=n_boot,
        key=key,
        resample_one=lambda b: _resample_stratified_one(
            measure, covariance, rows, cell_of_psu, stratum_of_psu, drawn_all[b]
        ),
        weighting=weighting,
        regularization=regularization,
        optimizer=optimizer,
        penalty=penalty,
        name="stratified_bootstrap",
    )


__all__ = ["stratified_bootstrap", "StratifiedBootstrapResult"]
