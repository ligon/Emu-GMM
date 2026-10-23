"""Opt-in numerical rank policy and diagnostics for an asymptotic law.

The cutoff is a coordinate-dependent numerical choice at the fitted point.
It does not establish population nonidentification or boundary-valid inference.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclasses.dataclass(frozen=True)
class NumericalRankPolicy:
    """Explicit relative bread cutoff in native flat ambient coordinates."""

    relative_tol: float
    coordinates: str = "native_ambient_flat"
    metric: str = "euclidean_frobenius"

    def __post_init__(self) -> None:
        if isinstance(self.relative_tol, bool):
            raise ValueError("relative_tol must be finite and strictly between 0 and 1")
        tol = float(self.relative_tol)
        if not math.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("relative_tol must be finite and strictly between 0 and 1")
        if self.coordinates != "native_ambient_flat":
            raise ValueError("only native_ambient_flat coordinates are supported")
        if self.metric != "euclidean_frobenius":
            raise ValueError("only the euclidean_frobenius metric is supported")
        object.__setattr__(self, "relative_tol", float(tol))


@dataclasses.dataclass(frozen=True)
class PSDFactorGeometry:
    """Machine-rank check for one fitted ``PSDFixedRank`` factor."""

    leaf_index: int
    singular_values: tuple[float, ...]
    status: str  # "regular" | "singular"


@dataclasses.dataclass(frozen=True)
class NumericalRankReport:
    """Signed spectrum and full excluded projector for one selected policy."""

    policy: NumericalRankPolicy
    signed_eigenvalues: np.ndarray
    cutoff: float
    retained_mask: np.ndarray
    excluded_projector: np.ndarray
    effective_rank: int
    nominal_gauge_dim: int
    extra_excluded: int
    largest_excluded_eigenvalue: float | None
    smallest_retained_eigenvalue: float | None
    excluded_cutoff_margin: float | None
    retained_cutoff_margin: float | None
    negative_allowance: float
    status: str
    projector_boundary_ambiguous: bool
    param_names: tuple[str, ...]
    leaf_shapes: tuple[tuple[int, ...], ...]
    factor_geometries: tuple[PSDFactorGeometry, ...] = ()

    def __post_init__(self) -> None:
        eig = _readonly_array(self.signed_eigenvalues, dtype=float)
        mask = _readonly_array(self.retained_mask, dtype=bool)
        projector = _readonly_array(self.excluded_projector, dtype=float)
        d = eig.size
        if eig.shape != (d,) or mask.shape != (d,) or projector.shape != (d, d):
            raise ValueError("rank report spectrum, mask, or projector has wrong shape")
        if d == 0 or len(self.param_names) != d:
            raise ValueError("rank report labels must match a nonempty ambient axis")
        if not 0 <= self.nominal_gauge_dim < d:
            raise ValueError("rank report nominal gauge dimension is out of bounds")
        if self.effective_rank != int(mask.sum()):
            raise ValueError("rank report effective rank disagrees with its mask")
        if self.extra_excluded != d - self.effective_rank - self.nominal_gauge_dim:
            raise ValueError(
                "rank report extra exclusion count disagrees with its mask"
            )
        if self.extra_excluded < 0:
            raise ValueError("rank report excludes fewer than nominal gauge directions")
        if self.status not in {
            "ok",
            "all_zero_bread",
            "no_positive_bread",
            "materially_negative_bread",
            "nonfinite_bread",
        }:
            raise ValueError(f"unknown rank report status {self.status!r}")
        if not np.all(np.isfinite(projector)):
            raise ValueError("rank report projector must be finite")
        error_tol = 64 * d * np.finfo(float).eps
        if not np.allclose(projector, projector.T, atol=error_tol, rtol=0):
            raise ValueError("rank report projector must be symmetric")
        if not np.allclose(projector @ projector, projector, atol=error_tol, rtol=0):
            raise ValueError("rank report projector must be idempotent")
        if abs(float(np.trace(projector)) - (d - self.effective_rank)) > error_tol:
            raise ValueError("rank report projector rank disagrees with its mask")
        if self.status != "nonfinite_bread" and (
            not np.all(np.isfinite(eig)) or np.any(np.diff(eig) < 0)
        ):
            raise ValueError("rank report signed spectrum must be finite and sorted")
        if self.status != "nonfinite_bread":
            if not math.isclose(
                self.cutoff,
                self.policy.relative_tol * float(eig[-1]),
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "rank report cutoff disagrees with policy and spectrum"
                )
            expected = (
                np.zeros(d, dtype=bool)
                if eig[-1] <= 0
                else (np.arange(d) >= self.nominal_gauge_dim) & (eig > self.cutoff)
            )
            if not np.array_equal(mask, expected):
                raise ValueError(
                    "rank report mask disagrees with cutoff and gauge count"
                )
        if sum(math.prod(shape) for shape in self.leaf_shapes) != d:
            raise ValueError("rank report leaf shapes do not match the ambient axis")
        for factor in self.factor_geometries:
            if (
                factor.status not in {"regular", "singular"}
                or not 0 <= factor.leaf_index < len(self.leaf_shapes)
                or len(self.leaf_shapes[factor.leaf_index]) != 2
            ):
                raise ValueError("rank report has invalid PSD-factor geometry")
        object.__setattr__(self, "signed_eigenvalues", eig)
        object.__setattr__(self, "retained_mask", mask)
        object.__setattr__(self, "excluded_projector", projector)

    @property
    def is_valid(self) -> bool:
        """Whether this policy admits a retained-subspace Gaussian readout."""
        return self.status == "ok"

    @property
    def geometry_status(self) -> str:
        """Regular quotient availability for the fitted PSD factors."""
        if any(f.status == "singular" for f in self.factor_geometries):
            return "singular"
        return "regular" if self.factor_geometries else "not_applicable"


@dataclasses.dataclass(frozen=True)
class FunctionalSensitivity:
    """Local derivative loading on the whole excluded ambient subspace."""

    first_order_norm: np.ndarray
    second_order_norm: np.ndarray
    projected_jacobian: np.ndarray
    projected_hessian: np.ndarray
    derivative_orders_checked: tuple[int, ...] = (1, 2)
    higher_orders_unassessed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "first_order_norm",
            "second_order_norm",
            "projected_jacobian",
            "projected_hessian",
        ):
            object.__setattr__(
                self, name, _readonly_array(getattr(self, name), dtype=float)
            )


__all__ = [
    "NumericalRankPolicy",
    "NumericalRankReport",
    "PSDFactorGeometry",
    "FunctionalSensitivity",
]
