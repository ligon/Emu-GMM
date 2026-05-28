"""Polymorphic-input adapter and axis-label tracking.

The framework accepts data inputs as :class:`pandas.DataFrame`,
:class:`haliax.NamedArray`, or plain :class:`jax.numpy.ndarray`, and
emits labelled outputs (typically :class:`haliax.NamedArray`) carrying
coordinate strings derived from those inputs. This module is the bridge.

Two responsibilities:

1. /Input normalisation/: ``normalise_x``, ``normalise_weights``,
   ``normalise_mask`` strip the input wrappers, returning plain JAX
   arrays plus the labels extracted from the input.
2. /Output labelling/: ``label_matrix`` wraps a plain matrix into a
   :class:`haliax.NamedArray` with the given row/column axes.

The :class:`LabelContext` dataclass holds the labels collected during
input normalisation and threads through the estimation pipeline as a
/static/ closure variable --- it is intentionally not a JAX PyTree, so
that JIT compilation never traces label strings.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import haliax as ha
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import axes as axes_mod

# Pandas is optional at runtime (it's a v1 dep, but we don't want
# this module to fail at import time if pandas changes its public API).
try:
    import pandas as pd

    _HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    _HAVE_PANDAS = False


@dataclasses.dataclass(frozen=True)
class LabelContext:
    """Axis labels for one estimation run.

    Not a JAX PyTree; lives outside ``jit`` / ``vmap`` boundaries as a
    static closure variable. All fields are tuples (hashable) so the
    whole struct is hashable for use as a static argument to ``jit``.
    """

    param_names: tuple[str, ...] = ()
    moment_names: tuple[str, ...] = ()
    variable_names: tuple[str, ...] = ()
    obs_name: str | None = None

    def with_moment_names(self, names: tuple[str, ...]) -> "LabelContext":
        """Return a copy with ``moment_names`` replaced."""
        return dataclasses.replace(self, moment_names=tuple(names))


def _is_pandas_frame(x: Any) -> bool:
    return _HAVE_PANDAS and isinstance(x, pd.DataFrame)


def _is_pandas_series(x: Any) -> bool:
    return _HAVE_PANDAS and isinstance(x, pd.Series)


def _is_haliax_named(x: Any) -> bool:
    return isinstance(x, ha.NamedArray)


def normalise_x(
    x_in: Any,
) -> tuple[Float[Array, "N D"], tuple[str, ...], str | None]:
    """Strip a data-input wrapper, returning the JAX array plus labels.

    Parameters
    ----------
    x_in
        One of: :class:`pandas.DataFrame`, :class:`haliax.NamedArray`,
        or any object convertible via :func:`jax.numpy.asarray` to a
        2-D array.

    Returns
    -------
    array : (N, D) jax array
    variable_names : tuple[str, ...]
        Column / D-axis labels. Pandas column names if input was a
        DataFrame; Haliax axis-1 coordinates if input was a NamedArray;
        positional fallback (``("v_0", "v_1", ...)``) otherwise.
    obs_name : str | None
        Observation-axis label. Pandas index name if non-None; otherwise
        ``None``.

    Raises
    ------
    ValueError
        If the input cannot be coerced to a 2-D array.
    """
    if _is_pandas_frame(x_in):
        cols = tuple(str(c) for c in x_in.columns)
        idx_name = x_in.index.name
        arr = jnp.asarray(x_in.to_numpy())
    elif _is_haliax_named(x_in):
        if len(x_in.axes) != 2:
            raise ValueError(
                f"normalise_x: haliax input must have 2 axes "
                f"(observations, variables); got axes "
                f"{[a.name for a in x_in.axes]}"
            )
        # First axis presumed to be observations; second to be variables.
        cols = (
            (x_in.axes[1].name,) if False else tuple([x_in.axes[1].name])
        )  # placeholder
        # Actually a NamedArray's axes are AXIS-LEVEL names (one per dim),
        # not per-coordinate names. For per-coordinate names we'd need
        # a different mechanism. For v1, treat the axis name as the
        # collective label and use positional names for the columns.
        cols = tuple(f"v_{i}" for i in range(x_in.axes[1].size))
        idx_name = x_in.axes[0].name
        arr = x_in.array
    else:
        arr = jnp.asarray(x_in)
        if arr.ndim != 2:
            raise ValueError(
                f"normalise_x: expected a 2-D array, got shape {arr.shape}"
            )
        cols = tuple(f"v_{i}" for i in range(arr.shape[1]))
        idx_name = None

    return arr, cols, idx_name


def normalise_weights(
    w_in: Any | None,
    n: int,
) -> Float[Array, " N"]:
    """Strip a weight-input wrapper into a 1-D JAX array of length ``N``.

    ``None`` produces an all-ones default.

    Parameters
    ----------
    w_in
        :class:`pandas.Series`, :class:`haliax.NamedArray`, plain array,
        or ``None``.
    n
        Expected length (number of observations).

    Returns
    -------
    weights : (N,) jax array

    Raises
    ------
    ValueError
        If the weights array's length does not match ``n``.
    """
    if w_in is None:
        return jnp.ones(n)
    if _is_pandas_series(w_in):
        arr = jnp.asarray(w_in.to_numpy())
    elif _is_haliax_named(w_in):
        arr = w_in.array
    else:
        arr = jnp.asarray(w_in)
    arr = arr.reshape(-1)
    if int(arr.shape[0]) != n:
        raise ValueError(
            f"normalise_weights: expected length {n}, got {int(arr.shape[0])}"
        )
    return arr


def normalise_mask(
    mask_in: Any | None,
    n: int,
    m: int,
) -> Float[Array, "N M"]:
    """Strip a mask-input wrapper into a 0/1 array of shape ``(N, M)``.

    ``None`` produces an all-ones default (no missingness).

    Parameters
    ----------
    mask_in
        :class:`pandas.DataFrame`, :class:`haliax.NamedArray`, plain
        array of shape ``(N, M)``, or ``None``.
    n, m
        Expected number of observations and moments.

    Returns
    -------
    mask : (N, M) jax array of 0/1 floats
    """
    if mask_in is None:
        return jnp.ones((n, m))
    if _is_pandas_frame(mask_in):
        arr = jnp.asarray(mask_in.to_numpy())
    elif _is_haliax_named(mask_in):
        arr = mask_in.array
    else:
        arr = jnp.asarray(mask_in)
    if arr.shape != (n, m):
        raise ValueError(f"normalise_mask: expected shape ({n}, {m}), got {arr.shape}")
    return arr.astype(jnp.float32)


def resolve_moment_names(
    model_return: Any | None,
    kwarg_names: tuple[str, ...] | None,
    m: int,
) -> tuple[str, ...]:
    """Apply the moment-name precedence policy.

    Precedence (highest first):

    1. If ``model_return`` is a :class:`haliax.NamedArray` with a
       ``moments`` axis, use the axis name's coordinates (positional
       fallback inside the axis, since haliax axes don't carry
       per-coordinate names by default).
    2. If ``kwarg_names`` is provided, use it.
    3. Positional fallback: ``("m_0", "m_1", ...)``.

    Parameters
    ----------
    model_return
        Optional probe of the structural model's return type, used to
        detect a labelled output. Pass ``None`` to skip this branch.
    kwarg_names
        Optional ``moment_names`` keyword from ``estimate(...)``.
    m
        Expected number of moments. The returned tuple has length ``m``.

    Returns
    -------
    names : tuple[str, ...] of length ``m``

    Raises
    ------
    ValueError
        If a supplied source has the wrong length.
    """
    if isinstance(model_return, ha.NamedArray):
        moments_axes = [a for a in model_return.axes if a.name == axes_mod.MOMENTS_NAME]
        if moments_axes:
            ax = moments_axes[0]
            if ax.size != m:
                raise ValueError(
                    f"resolve_moment_names: model returned NamedArray with "
                    f"moments axis size {ax.size}, expected {m}"
                )
            # Per-coordinate names not carried by Axis; positional inside the axis.
            return tuple(f"m_{i}" for i in range(m))

    if kwarg_names is not None:
        if len(kwarg_names) != m:
            raise ValueError(
                f"resolve_moment_names: moment_names has length "
                f"{len(kwarg_names)}, expected {m}"
            )
        return tuple(str(n) for n in kwarg_names)

    return tuple(f"m_{i}" for i in range(m))


def label_matrix(
    arr: Float[Array, "R C"],
    row_axis: ha.Axis,
    col_axis: ha.Axis,
) -> ha.NamedArray:
    """Wrap a plain (R, C) array in a :class:`haliax.NamedArray`.

    Parameters
    ----------
    arr
        2-D JAX array.
    row_axis, col_axis
        Haliax axes for the two dimensions. Sizes must match ``arr.shape``.

    Returns
    -------
    named : :class:`haliax.NamedArray` with axes ``(row_axis, col_axis)``.

    Raises
    ------
    ValueError
        If axis sizes don't match the array.
    """
    if arr.ndim != 2:
        raise ValueError(f"label_matrix: expected 2-D array, got shape {arr.shape}")
    if arr.shape != (row_axis.size, col_axis.size):
        raise ValueError(
            f"label_matrix: array shape {arr.shape} does not match axes "
            f"({row_axis.name}={row_axis.size}, {col_axis.name}={col_axis.size})"
        )
    return ha.named(arr, (row_axis, col_axis))


def label_vector(
    arr: Float[Array, " N"],
    axis: ha.Axis,
) -> ha.NamedArray:
    """Wrap a plain (N,) array in a :class:`haliax.NamedArray`.

    Parameters
    ----------
    arr
        1-D JAX array.
    axis
        Haliax axis. Size must match ``arr.shape[0]``.
    """
    if arr.ndim != 1:
        raise ValueError(f"label_vector: expected 1-D array, got shape {arr.shape}")
    if int(arr.shape[0]) != axis.size:
        raise ValueError(
            f"label_vector: array length {int(arr.shape[0])} does not match "
            f"axis {axis.name}={axis.size}"
        )
    return ha.named(arr, (axis,))


__all__ = [
    "LabelContext",
    "normalise_x",
    "normalise_weights",
    "normalise_mask",
    "resolve_moment_names",
    "label_matrix",
    "label_vector",
]
