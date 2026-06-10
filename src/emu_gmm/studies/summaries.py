"""Layer 2 of the Monte Carlo driver (#114): pure summarizers.

Cheap, separately-testable reductions over the stacked records that
:func:`emu_gmm.studies.replicate` returns. Each summarizer:

* accepts either an :class:`~emu_gmm.studies.MCRecords` wrapper or a
  bare stacked :class:`~emu_gmm.types.FitRecord`;
* **excludes-but-counts** non-converged replicates (the adaptive
  bootstrap's convention, #91): silently dropping them biases size,
  silently including them poisons moments. Every summary surfaces
  ``n_used`` / ``n_excluded``;
* is plain numpy on the host --- no pandas (the ``to_pandas``
  materializer lives on :class:`~emu_gmm.studies.MCRecords`), nothing
  traced.

New diagnostics (K-statistic size, ``gamma_se`` coverage for manifold
parameters, ...) belong here as new functions over the same records ---
never inside the layer-1 loop.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import scipy.stats

from emu_gmm._internal import params as params_mod
from emu_gmm.studies.driver import MCRecords
from emu_gmm.types import FitRecord


def _stacked(records: MCRecords | FitRecord) -> FitRecord:
    """Unwrap an :class:`MCRecords` to its stacked :class:`FitRecord`."""
    if isinstance(records, MCRecords):
        return records.records
    return records


def _used(rec: FitRecord) -> tuple[np.ndarray, int, int]:
    """The exclude-but-count mask: (converged mask, n_used, n_excluded)."""
    mask = np.asarray(rec.converged) > 0.5
    n_used = int(mask.sum())
    return mask, n_used, int(mask.size) - n_used


def _as_flat_theta0(theta0: Any, d: int) -> np.ndarray:
    """Coerce ``theta0`` to the records' flat ambient axis (length D).

    Accepts a length-D array-like, or the user's parameter pytree
    (flattened with the same dispatch ``FitRecord`` itself uses).
    """
    if isinstance(theta0, list | tuple) or hasattr(theta0, "__array__"):
        arr = np.asarray(theta0, dtype=float)
    else:
        try:
            flat, _ = params_mod.flatten_params(theta0)
        except Exception:
            flat, _, _ = params_mod.flatten_params_with_spec(theta0)
        arr = np.asarray(flat, dtype=float)
    if arr.shape != (d,):
        raise ValueError(
            f"theta0 has flat shape {arr.shape}; the records carry a "
            f"length-{d} ambient parameter axis."
        )
    return arr


def _nan_vec(d: int) -> np.ndarray:
    return np.full((d,), np.nan)


@dataclasses.dataclass(frozen=True)
class BiasSD:
    """Per-coordinate recovery summary (axis = ``param_names``)."""

    bias: np.ndarray  # mean(theta_hat) - theta0
    mc_sd: np.ndarray  # SD of theta_hat across used reps (ddof=1)
    mean_se: np.ndarray  # mean of the analytic SEs
    se_ratio: np.ndarray  # mean_se / mc_sd
    param_names: tuple[str, ...]
    n_used: int
    n_excluded: int


def bias_sd(records: MCRecords | FitRecord, theta0: Any) -> BiasSD:
    """Bias, Monte Carlo SD, mean analytic SE, and SE/MC-SD per coordinate.

    ``se_ratio`` near 1 is the "analytic SE tracks the sampling
    distribution" check from the validation harness
    (``docs/validation/seasonality-mc-2026-05-29.org``).
    """
    rec = _stacked(records)
    mask, n_used, n_excluded = _used(rec)
    d = int(np.asarray(rec.theta_flat).shape[1])
    t0 = _as_flat_theta0(theta0, d)
    names = tuple(rec.param_names)
    if n_used == 0:
        nan = _nan_vec(d)
        return BiasSD(nan, nan, nan, nan, names, 0, n_excluded)
    theta = np.asarray(rec.theta_flat)[mask]
    se = np.asarray(rec.se)[mask]
    bias = theta.mean(axis=0) - t0
    mc_sd = theta.std(axis=0, ddof=1) if n_used >= 2 else _nan_vec(d)
    mean_se = se.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_ratio = mean_se / mc_sd
    return BiasSD(bias, mc_sd, mean_se, se_ratio, names, n_used, n_excluded)


@dataclasses.dataclass(frozen=True)
class Coverage:
    """Per-coordinate Wald confidence-interval coverage."""

    coverage: np.ndarray  # fraction of used reps with theta0 in the CI
    level: float
    param_names: tuple[str, ...]
    n_used: int
    n_excluded: int


def coverage(
    records: MCRecords | FitRecord, theta0: Any, level: float = 0.95
) -> Coverage:
    """Empirical coverage of the per-coordinate Wald CI at ``level``.

    The CI is ``theta_hat +/- z_{level} * se`` with the records' own
    analytic SEs; coverage is the fraction of **used** (converged) reps
    whose interval contains ``theta0``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"coverage(): level must be in (0, 1), got {level}")
    rec = _stacked(records)
    mask, n_used, n_excluded = _used(rec)
    d = int(np.asarray(rec.theta_flat).shape[1])
    t0 = _as_flat_theta0(theta0, d)
    names = tuple(rec.param_names)
    if n_used == 0:
        return Coverage(_nan_vec(d), level, names, 0, n_excluded)
    theta = np.asarray(rec.theta_flat)[mask]
    se = np.asarray(rec.se)[mask]
    z = float(scipy.stats.norm.ppf(0.5 + level / 2.0))
    covered = np.abs(theta - t0[None, :]) <= z * se
    return Coverage(covered.mean(axis=0), level, names, n_used, n_excluded)


@dataclasses.dataclass(frozen=True)
class SizePower:
    """J-test rejection rates at each ``alpha`` (axis = ``alphas``)."""

    alphas: tuple[float, ...]
    reject_nominal: np.ndarray  # from J_pvalue (chi^2_{M-K})
    reject_adjusted: np.ndarray  # from J_pvalue_adjusted (ridge-aware)
    n_used: int
    n_excluded: int


def size_power(
    records: MCRecords | FitRecord,
    alpha: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> SizePower:
    """J-test rejection frequencies at each level in ``alpha``.

    Under a correctly-specified DGP these are empirical *size*; under a
    misspecified alternative, *power*. Computed from both the nominal
    ``J_pvalue`` and the regularisation-adjusted ``J_pvalue_adjusted``
    --- their divergence is the #130 "does the ridge distort
    calibration" evidence.
    """
    rec = _stacked(records)
    mask, n_used, n_excluded = _used(rec)
    alphas = tuple(float(a) for a in alpha)
    if n_used == 0:
        nan = np.full((len(alphas),), np.nan)
        return SizePower(alphas, nan, nan.copy(), 0, n_excluded)
    p_nom = np.asarray(rec.J_pvalue)[mask]
    p_adj = np.asarray(rec.J_pvalue_adjusted)[mask]
    reject_nom = np.array([(p_nom < a).mean() for a in alphas])
    reject_adj = np.array([(p_adj < a).mean() for a in alphas])
    return SizePower(alphas, reject_nom, reject_adj, n_used, n_excluded)


@dataclasses.dataclass(frozen=True)
class TauBinding:
    """Regularisation-binding summary (the #130 tau column)."""

    binding_frequency: float  # fraction of used reps with binding_ridge
    quantile_levels: tuple[float, ...]
    tau_quantiles: np.ndarray  # quantiles of tau_realised over used reps
    n_used: int
    n_excluded: int


def tau_binding(
    records: MCRecords | FitRecord,
    q: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> TauBinding:
    """How often the ``DiagonalTikhonov`` ridge binds, and how large.

    The empirical answer to "is the regularized regime the rule or the
    exception under realistic missingness" (#130, review point 2):
    ``binding_ridge`` frequency plus quantiles of the realised tau.
    """
    rec = _stacked(records)
    mask, n_used, n_excluded = _used(rec)
    levels = tuple(float(x) for x in q)
    if n_used == 0:
        return TauBinding(
            float("nan"), levels, np.full((len(levels),), np.nan), 0, n_excluded
        )
    binding = np.asarray(rec.binding_ridge)[mask]
    tau = np.asarray(rec.tau_realised)[mask]
    return TauBinding(
        float(binding.mean()),
        levels,
        np.quantile(tau, np.array(levels)),
        n_used,
        n_excluded,
    )


@dataclasses.dataclass(frozen=True)
class JCalibration:
    """Uniformity summary for the J p-value under the null."""

    deciles: np.ndarray  # the probed CDF points (0.1 .. 0.9)
    ecdf: np.ndarray  # empirical CDF of J_pvalue at each decile
    deviation: np.ndarray  # ecdf - deciles (0 under perfect calibration)
    max_abs_deviation: float  # KS-style sup over the probed deciles
    J_dof: int
    n_used: int
    n_excluded: int


def j_calibration(records: MCRecords | FitRecord) -> JCalibration:
    """Empirical-CDF deviations of ``J_pvalue`` from U(0,1) at deciles.

    Under a correct null, ``J_pvalue`` is asymptotically uniform; the
    deviations ``ecdf(d) - d`` at the deciles summarise calibration of
    the chi^2_{M-K} reference without committing to one alpha.
    """
    rec = _stacked(records)
    mask, n_used, n_excluded = _used(rec)
    deciles = np.arange(1, 10) / 10.0
    dof = int(rec.J_dof)
    if n_used == 0:
        nan = np.full_like(deciles, np.nan)
        return JCalibration(deciles, nan, nan.copy(), float("nan"), dof, 0, n_excluded)
    p = np.asarray(rec.J_pvalue)[mask]
    ecdf = np.array([(p <= d).mean() for d in deciles])
    deviation = ecdf - deciles
    return JCalibration(
        deciles,
        ecdf,
        deviation,
        float(np.abs(deviation).max()),
        dof,
        n_used,
        n_excluded,
    )


__all__ = [
    "BiasSD",
    "Coverage",
    "SizePower",
    "TauBinding",
    "JCalibration",
    "bias_sd",
    "coverage",
    "size_power",
    "tau_binding",
    "j_calibration",
]
