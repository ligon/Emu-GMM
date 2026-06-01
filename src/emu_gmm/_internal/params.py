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
import numpy as np
from jaxtyping import Array, Float

from emu_gmm.manifolds.euclidean import Euclidean

# Phase 1 note: importing ManifoldLeaf here guarantees its
# ``@register_pytree_node_class`` decorator runs before any
# ``flatten_params_with_spec`` / ``unflatten_params`` call, so JAX
# recognises the node (red-team R32). v1 paths never construct or see a
# ManifoldLeaf; it is exercised only by the manifold-aware 3-tuple path.
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
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
    manifold_spec: ManifoldSpec | None = None,
) -> Any:
    """Reconstruct a parameter PyTree from a flat 1-D array.

    Inverse of :func:`flatten_params` (when ``manifold_spec is None``)
    and of :func:`flatten_params_with_spec` (when ``manifold_spec`` is
    supplied).

    Parameters
    ----------
    flat
        1-D array. For the v1 path its length equals
        ``treedef.num_leaves``; for the manifold-aware path its length
        equals ``manifold_spec.total_ambient_dim``.
    treedef
        ``PyTreeDef`` produced by :func:`flatten_params` (v1) or by
        :func:`flatten_params_with_spec` (manifold-aware).
    manifold_spec
        Optional :class:`ManifoldSpec`. When ``None`` (the default), the
        v1 scalar-reindex behaviour runs **bitwise unchanged** --- every
        existing caller (estimator, types, riemannian_lm) hits exactly
        this path. When supplied, each leaf's block is sliced as
        ``flat[offset:offset + prod(ambient_shape)]`` and reshaped to its
        ``ambient_shape`` (C / row-major order; red-team R20).

    Returns
    -------
    params
        The reconstructed parameter PyTree (same type as the original).

    Raises
    ------
    ValueError
        If ``flat`` is not 1-D, or its length does not match the number
        of leaves (v1) / the total ambient dim (manifold-aware) expected.
    """
    flat_arr = jnp.asarray(flat)
    if flat_arr.ndim != 1:
        raise ValueError(
            f"unflatten_params: flat array must be 1-D, got shape " f"{flat_arr.shape}"
        )

    if manifold_spec is None:
        # ---- v1 path: bitwise-identical to the original (red-team R4/R10).
        n = treedef.num_leaves
        if int(flat_arr.shape[0]) != n:
            raise ValueError(
                f"unflatten_params: flat array has {int(flat_arr.shape[0])} "
                f"elements but treedef expects {n} leaves"
            )
        leaves = [flat_arr[i] for i in range(n)]
        return jax.tree_util.tree_unflatten(treedef, leaves)

    # ---- manifold-aware path.
    leaf_specs = manifold_spec.leaf_specs
    n = treedef.num_leaves
    if len(leaf_specs) != n:
        raise ValueError(
            f"unflatten_params: manifold_spec has {len(leaf_specs)} leaf "
            f"specs but treedef expects {n} leaves"
        )
    total = sum(int(np.prod(ls.ambient_shape)) for ls in leaf_specs)
    if int(flat_arr.shape[0]) != total:
        raise ValueError(
            f"unflatten_params: flat array has {int(flat_arr.shape[0])} "
            f"elements but manifold_spec accounts for {total} "
            "(sum of prod(ambient_shape) over leaves)"
        )

    leaves = []
    for ls in leaf_specs:
        size = int(np.prod(ls.ambient_shape))
        if ls.ambient_shape == ():
            # Scalar leaf: 0-d, matching v1 reconstruction exactly
            # (red-team R10/R33/R37). flat[offset] is a 0-d array.
            leaf = flat_arr[ls.offset]
        else:
            block = flat_arr[ls.offset : ls.offset + size]
            leaf = jnp.reshape(block, ls.ambient_shape)
        # Restore the leaf's original dtype: jnp.concatenate promotes the
        # flat buffer to a common dtype, so a bare float32/int32 scalar
        # would otherwise come back float64. For wrapped leaves this
        # agrees with the ManifoldLeaf aux_data dtype (harmless no-op).
        if ls.dtype is not None:
            leaf = leaf.astype(ls.dtype)
        leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def flatten_params_with_spec(
    params: Any,
) -> tuple[Float[Array, " D"], jax.tree_util.PyTreeDef, ManifoldSpec]:
    """Manifold-aware flatten: ``(flat, treedef, manifold_spec)`` 3-tuple.

    This is the v2 entry point (manifold epic #12, Phase 1). It is a
    **completely independent** function from :func:`flatten_params`; the
    v1 2-tuple ``flatten_params`` is left untouched and stays scalar-only
    (red-team R1/R15/R16). New manifold-aware call sites use this 3-tuple.

    Each leaf's ambient block is ravelled in C (row-major) order
    (red-team R20) and concatenated into a single length-
    ``total_ambient_dim`` buffer. A :class:`ManifoldLeaf`-wrapped leaf
    contributes a block of size ``prod(ambient_shape)`` governed by its
    own :class:`ManifoldParam`; a bare scalar leaf contributes a length-1
    block governed by ``Euclidean()`` (the 0-d-shape Euclidean --- *not*
    ``Euclidean(1)``), reproducing the v1 layout for all-scalar trees.

    For an **all-scalar** tree this returns ``flat`` / ``treedef``
    identical to ``flatten_params(params)`` (the first two elements),
    same element order and dtype (red-team R2/R3) --- verified in the
    test-suite.

    Parameters
    ----------
    params
        A parameter PyTree. Non-scalar leaves must be wrapped in a
        :class:`ManifoldLeaf` carrying their :class:`ManifoldParam`.

    Returns
    -------
    flat
        1-D ambient buffer of length ``total_ambient_dim``.
    treedef
        ``PyTreeDef`` for reconstruction via :func:`unflatten_params`
        (pass the returned ``manifold_spec`` as its third argument). The
        treedef descends into each :class:`ManifoldLeaf`, so
        ``tree_unflatten`` re-wraps the reshaped block automatically.
    manifold_spec
        A frozen, jit-hashable :class:`ManifoldSpec` with per-leaf
        ``offset`` / ``ambient_shape`` / ``manifold`` / ``field_name``,
        in PyTree-leaf-walk order.

    Notes
    -----
    The function is pure: it never mutates ``params`` and is idempotent
    (red-team R24). ``K == 0`` (empty tree) is caught upstream at the
    estimator boundary (red-team R31).
    """
    # Field names from the dataclass root, when available (mirrors
    # manifold_spec_from_params); else positional None.
    if dataclasses.is_dataclass(params):
        field_names: list[str | None] = [f.name for f in dataclasses.fields(params)]
    else:
        field_names = []

    # Walk treating ManifoldLeaf as opaque so we can read each leaf's
    # manifold + ambient shape. The *returned* treedef (below) is the
    # FULL treedef --- it descends into ManifoldLeaf so tree_unflatten
    # re-wraps blocks automatically and the recorded dtype is restored.
    wrapped_leaves = jax.tree_util.tree_leaves(
        params, is_leaf=lambda x: isinstance(x, ManifoldLeaf)
    )
    if len(field_names) != len(wrapped_leaves):
        # Non-flat dataclass / non-dataclass root: positional names.
        field_names = [None] * len(wrapped_leaves)

    _, treedef = jax.tree_util.tree_flatten(params)

    leaf_specs: list[LeafSpec] = []
    blocks: list[Any] = []
    offset = 0
    total_dim = 0
    total_gauge = 0
    for leaf, name in zip(wrapped_leaves, field_names, strict=True):
        if isinstance(leaf, ManifoldLeaf):
            arr = jnp.asarray(leaf.array)
            manifold = leaf.manifold
            ambient_shape = tuple(int(s) for s in arr.shape)
            # Guard against an annotation / array-shape mismatch
            # (red-team R17/R22): the manifold's declared ambient_shape,
            # when it exposes one, must match the actual array shape.
            decl = getattr(manifold, "ambient_shape", None)
            if decl is not None and tuple(int(s) for s in decl) != ambient_shape:
                raise ValueError(
                    f"flatten_params_with_spec: leaf {name!r} array shape "
                    f"{ambient_shape} does not match its manifold's "
                    f"ambient_shape {tuple(decl)}"
                )
        else:
            arr = jnp.asarray(leaf)
            ambient_shape = tuple(int(s) for s in arr.shape)
            # Bare scalar leaf -> Euclidean() (0-d). Reject the non-scalar
            # bare case loudly: a matrix must arrive wrapped in a
            # ManifoldLeaf so its manifold is known (red-team R29).
            if ambient_shape != ():
                raise ValueError(
                    f"flatten_params_with_spec: bare (unwrapped) leaf "
                    f"{name!r} has shape {ambient_shape}; non-scalar leaves "
                    "must be wrapped in a ManifoldLeaf carrying their "
                    "ManifoldParam"
                )
            manifold = Euclidean()

        size = int(np.prod(ambient_shape))
        leaf_specs.append(
            LeafSpec(
                offset=offset,
                ambient_shape=ambient_shape,
                manifold=manifold,
                field_name=name,
                dtype=arr.dtype,
            )
        )
        # C/row-major ravel; pinned so flatten and unflatten agree
        # (red-team R20). reshape(-1) is C-order in JAX.
        blocks.append(jnp.reshape(arr, (size,)))
        offset += size
        total_dim += int(manifold.dimension)
        total_gauge += int(manifold.gauge_dim)

    if blocks:
        flat = jnp.concatenate(blocks)
    else:  # K == 0: caught upstream, but keep a well-typed empty buffer.
        flat = jnp.zeros((0,))

    # Load-bearing invariant: block widths exactly tile the buffer
    # (red-team R5/R9/R11/R23). offset == sum(prod(ambient_shape)).
    total_ambient = sum(int(np.prod(ls.ambient_shape)) for ls in leaf_specs)
    assert total_ambient == int(flat.shape[0]) == offset, (
        f"flatten_params_with_spec: block widths sum to {total_ambient} "
        f"but flat buffer has length {int(flat.shape[0])}"
    )

    spec = ManifoldSpec(
        leaf_specs=tuple(leaf_specs),
        total_ambient_dim=offset,
        total_dimension=total_dim,
        total_gauge_dim=total_gauge,
    )
    return flat, treedef, spec


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
    "flatten_params_with_spec",
    "unflatten_params",
    "param_names",
    "manifold_spec_from_params",
]
