"""Eager group-value guards and trace-safe structural checks (ledger 22).

Validate layouts on concrete inputs before passing genuinely dynamic IDs
through JIT/vmap. Runtime values of tracer/abstract arrays are not checked;
shape, dtype and declared counts still are. No callbacks run in kernels.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import jax
import numpy as np


def validate_group_ids(
    values: Any, count: int, *, name: str, n_rows: int | None = None
) -> np.ndarray | None:
    """Check count/shape/dtype and, eagerly, finite rounded IDs in range.

    Float tolerance is ``min(1e-6 + 8*eps*max(1, abs(round(id))), 1e-3)``.
    The absolute floor preserves the established ``1-1e-7`` rounding
    contract; the cap excludes labels a substantial fraction of a group
    away. Integer/bool ID arrays must already be exact. Declared unused
    slots are allowed. Return rounded concrete IDs, or None under tracing.
    """
    if (
        isinstance(count, bool | np.bool_)
        or not isinstance(count, Integral)
        or count <= 0
        or count > np.iinfo(np.int32).max
    ):
        raise ValueError(
            f"{name}: declared count must be a positive int32-sized integer"
        )
    abstract = isinstance(
        values, jax.core.Tracer | jax.ShapeDtypeStruct | jax.core.AbstractValue
    )
    if values is None or type(values) is object:
        raise ValueError(f"{name} must be a one-dimensional numeric ID array")
    raw = values if abstract else np.asarray(values)
    shape = tuple(raw.shape)
    dtype = np.dtype(raw.dtype)
    if len(shape) != 1 or dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric ID array")
    if n_rows is not None and shape != (n_rows,):
        raise ValueError(
            f"{name} must have one ID per row; got {shape}, expected ({n_rows},)"
        )
    if abstract:
        return None
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite IDs")
    rounded = np.rint(raw)
    tolerance = 0.0
    if dtype.kind == "f":
        tolerance = np.minimum(
            1e-6 + 8 * np.finfo(dtype).eps * np.maximum(1.0, np.abs(rounded)), 1e-3
        )
    if np.any(np.abs(raw - rounded) > tolerance):
        raise ValueError(
            f"{name} must contain integer-valued IDs (within rounding tolerance)"
        )
    if np.any((rounded < 0) | (rounded >= count)):
        raise ValueError(f"{name} rounded IDs must lie in [0, {count})")
    return rounded.astype(np.int64)


def validate_nesting(
    group: np.ndarray | None, parent: np.ndarray | None, *, message: str
) -> None:
    """Reject a concrete group assigned to several parents; skip tracers."""
    if group is None or parent is None:
        return
    order = np.argsort(group, kind="stable")
    if np.any(
        (group[order][1:] == group[order][:-1])
        & (parent[order][1:] != parent[order][:-1])
    ):
        raise ValueError(message)
