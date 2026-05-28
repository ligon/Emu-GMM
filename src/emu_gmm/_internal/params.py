"""Parameter dataclass <-> flat JAX array conversion.

The framework's user-facing API takes parameters as a
``@jdc.pytree_dataclass`` (or any flat-scalar PyTree), but the
optimiser, AD, and linear-algebra layers want a 1-D ``jax.numpy.ndarray``
of length ``K``. These helpers bridge the two representations.

For v1, nested dataclasses are not supported: every leaf of the
parameter tree must be a 0-d (scalar) value. A clear error is raised if
this is violated.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def flatten_params(
    params: Any,
) -> tuple[Float[Array, " K"], jax.tree_util.PyTreeDef]:
    """Flatten a parameter PyTree into a 1-D JAX array.

    Parameters
    ----------
    params
        Typically a ``@jdc.pytree_dataclass`` instance with scalar fields.
        Any PyTree whose leaves are 0-d will work.

    Returns
    -------
    flat
        1-D ``jax.numpy.ndarray`` of length ``K``, with leaves stacked
        in PyTree-traversal order.
    treedef
        ``PyTreeDef`` for reconstruction via :func:`unflatten_params`.

    Raises
    ------
    ValueError
        If any leaf is not a 0-d value after conversion via
        :func:`jax.numpy.asarray`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat_leaves = []
    for i, leaf in enumerate(leaves):
        arr = jnp.asarray(leaf)
        if arr.ndim != 0:
            raise ValueError(
                f"flatten_params: leaf {i} has shape {arr.shape}; "
                "all parameter leaves must be 0-d scalars in v1"
            )
        flat_leaves.append(arr)
    return jnp.stack(flat_leaves), treedef


def unflatten_params(
    flat: Float[Array, " K"],
    treedef: jax.tree_util.PyTreeDef,
) -> Any:
    """Reconstruct a parameter PyTree from a flat 1-D array.

    Inverse of :func:`flatten_params`.

    Parameters
    ----------
    flat
        1-D array of length ``K``.
    treedef
        ``PyTreeDef`` produced by :func:`flatten_params`.

    Returns
    -------
    params
        The reconstructed parameter PyTree (same type as the original).

    Raises
    ------
    ValueError
        If ``flat`` is not 1-D, or its length does not match the number
        of leaves expected by ``treedef``.
    """
    flat_arr = jnp.asarray(flat)
    if flat_arr.ndim != 1:
        raise ValueError(
            f"unflatten_params: flat array must be 1-D, got shape "
            f"{flat_arr.shape}"
        )
    n = treedef.num_leaves
    if int(flat_arr.shape[0]) != n:
        raise ValueError(
            f"unflatten_params: flat array has {int(flat_arr.shape[0])} "
            f"elements but treedef expects {n} leaves"
        )
    leaves = [flat_arr[i] for i in range(n)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def param_names(params: Any) -> list[str]:
    """Return the dataclass field names in canonical (declaration) order.

    For v1, parameter dataclasses must be flat: every field must be a
    scalar (not a nested dataclass). Nested structures raise a clear
    error rather than silently producing wrong names.

    Parameters
    ----------
    params
        A dataclass instance, typically ``@jdc.pytree_dataclass``.

    Returns
    -------
    names
        Field names in declaration order; matches the leaf order of
        :func:`flatten_params` for flat dataclasses.

    Raises
    ------
    TypeError
        If ``params`` is not a dataclass instance.
    NotImplementedError
        If any field is itself a dataclass (nested parameter structure).
    """
    if not dataclasses.is_dataclass(params):
        raise TypeError(
            f"param_names expects a dataclass instance, got "
            f"{type(params).__name__}"
        )
    names: list[str] = []
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if dataclasses.is_dataclass(value):
            raise NotImplementedError(
                f"Nested dataclass parameter field {field.name!r} is not "
                "supported in v1; use a flat dataclass with scalar fields"
            )
        names.append(field.name)
    return names


__all__ = ["flatten_params", "unflatten_params", "param_names"]
