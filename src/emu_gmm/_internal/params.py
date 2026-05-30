"""Parameter dataclass <-> flat JAX array conversion.

The framework's user-facing API takes parameters as a
``@jdc.pytree_dataclass`` (or any flat-scalar PyTree), but the
optimiser, AD, and linear-algebra layers want a 1-D ``jax.numpy.ndarray``
of length ``K``. These helpers bridge the two representations.

For v1, every leaf of the parameter tree was a 0-d (scalar) value.
v2 adds :func:`manifold_spec_from_params`, a helper that walks a
parameter PyTree and reports per-leaf manifold metadata. v1-style
trees produce a :class:`ManifoldSpec` consisting entirely of
:class:`Euclidean` (scalar) leaves --- the same flat-array layout, just
annotated. The v1 ``(flat, treedef)`` return signature of
:func:`flatten_params` is preserved bitwise; downstream code that wants
the manifold spec calls :func:`manifold_spec_from_params` alongside.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from emu_gmm.manifolds.euclidean import Euclidean
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec


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
            f"unflatten_params: flat array must be 1-D, got shape " f"{flat_arr.shape}"
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
            f"param_names expects a dataclass instance, got " f"{type(params).__name__}"
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


def manifold_spec_from_params(params: Any) -> ManifoldSpec:
    """Build a :class:`ManifoldSpec` describing the leaves of ``params``.

    For v1-style parameter trees (every leaf is a 0-d scalar), this
    returns a :class:`ManifoldSpec` whose ``leaf_specs`` are all
    :class:`Euclidean` (scalar) entries (per plan §2.8: scalar leaves
    map to ``Euclidean()`` --- the 0-d-shape Euclidean --- not to
    ``Euclidean(1)``). The resulting ``total_ambient_dim`` and
    ``total_dimension`` equal the v1 flat length ``K``, and
    ``total_gauge_dim == 0``.

    v2-style parameter trees may carry larger ambient shapes per leaf;
    that path activates once v2's :class:`ManifoldLeaf` wrapper lands
    in a later phase. For now this helper handles the v1 contract and
    leaves a clear seam for v2 leaves.

    Parameters
    ----------
    params
        A parameter PyTree. Dataclass field names are used for
        ``LeafSpec.field_name`` when the PyTree root is a dataclass.

    Returns
    -------
    spec
        A frozen, jit-hashable :class:`ManifoldSpec`.

    Notes
    -----
    The returned spec is consumed downstream by Phase 5 dispatch
    (see plan §2.6) and Phase 6 label generation (plan §2.10).
    """
    if dataclasses.is_dataclass(params):
        field_names: list[str | None] = [f.name for f in dataclasses.fields(params)]
    else:
        leaves_only, _ = jax.tree_util.tree_flatten(params)
        field_names = [None] * len(leaves_only)

    leaves, _ = jax.tree_util.tree_flatten(params)
    if len(leaves) != len(field_names):
        # Dataclass with non-scalar field structure --- fall back to
        # positional None names. (v2 may handle nested dataclasses
        # explicitly; for v1 contracts, leaves and field names match.)
        field_names = [None] * len(leaves)

    # Per-field manifold annotations (v2 lite slice, plan §2.8): a
    # parameter dataclass may carry an ``__emu_manifolds__`` class
    # attribute mapping field-name -> :class:`ManifoldParam` (e.g.
    # ``{"sigma": Positive()}``). Scalar Positive leaves keep the same
    # 0-d flatten layout, so this annotation is the *only* change needed
    # to route a field onto a non-Euclidean manifold; the
    # flatten/unflatten 2-tuple contract is preserved bitwise.
    field_manifolds: dict[str, Any] = {}
    if dataclasses.is_dataclass(params):
        field_manifolds = dict(getattr(type(params), "__emu_manifolds__", {}) or {})

    leaf_specs: list[LeafSpec] = []
    offset = 0
    total_dim = 0
    total_gauge = 0
    for leaf, name in zip(leaves, field_names, strict=True):
        arr = jnp.asarray(leaf)
        if arr.ndim != 0:
            raise NotImplementedError(
                f"manifold_spec_from_params: leaf {name!r} has shape "
                f"{arr.shape}; v2 ManifoldLeaf wrapping is required for "
                "non-scalar leaves (lands in a later phase)."
            )
        annotated = field_manifolds.get(name) if name is not None else None
        manifold = annotated if annotated is not None else Euclidean()
        leaf_specs.append(
            LeafSpec(
                offset=offset,
                ambient_shape=(),
                manifold=manifold,
                field_name=name,
            )
        )
        offset += manifold.dimension
        total_dim += manifold.dimension
        total_gauge += manifold.gauge_dim

    return ManifoldSpec(
        leaf_specs=tuple(leaf_specs),
        total_ambient_dim=offset,
        total_dimension=total_dim,
        total_gauge_dim=total_gauge,
    )


__all__ = [
    "flatten_params",
    "unflatten_params",
    "param_names",
    "manifold_spec_from_params",
]
