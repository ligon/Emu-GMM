"""Frozen, jit-hashable :class:`ManifoldSpec` / :class:`LeafSpec` (plan §2.7).

A :class:`ManifoldSpec` is the per-PyTree manifold metadata that the
v2 ``flatten_params`` returns alongside the flat ambient buffer and
``treedef``. Both containers are ``@dataclass(frozen=True)`` so they
hash by structural identity --- the frozen-tuple membership of
``leaf_specs`` keeps it jit-friendly when carried as a static argument
(see :func:`jax.jit`'s ``static_argnums`` / ``static_argnames``).

The contained :class:`ManifoldParam` instances themselves carry only
shape-and-sizes metadata (no traced arrays), so the composed
``__hash__`` is well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emu_gmm.manifolds.base import ManifoldParam


@dataclass(frozen=True)
class LeafSpec:
    """Per-leaf metadata for one entry of a parameter PyTree.

    Attributes
    ----------
    offset
        Offset into the flat ambient buffer where this leaf begins.
    ambient_shape
        Shape this leaf's block reshapes back into when unflattening.
        ``()`` for v1-style 0-d scalar leaves; ``(d,)`` for
        :class:`Euclidean(d)`; ``(n, k)`` for :class:`PSDFixedRank(n, k)`.
    manifold
        The :class:`ManifoldParam` instance attached to this leaf.
    field_name
        The user's dataclass field name, when known; ``None`` for
        unnamed PyTree leaves. Used by
        :func:`emu_gmm._internal.labels.tangent_basis_names` to emit
        readable labels.
    """

    offset: int
    ambient_shape: tuple[int, ...]
    manifold: ManifoldParam
    field_name: str | None = None


@dataclass(frozen=True)
class ManifoldSpec:
    """Full manifold metadata for one parameter PyTree.

    Attributes
    ----------
    leaf_specs
        Frozen tuple of :class:`LeafSpec` per leaf, in PyTree-leaf-walk
        order (matching :func:`jax.tree_util.tree_leaves`).
    total_ambient_dim
        Sum of ambient sizes per leaf == length of the flat buffer.
    total_dimension
        Sum of ``manifold.dimension`` across leaves. For all-Euclidean
        v1 trees this equals ``total_ambient_dim``; for trees containing
        :class:`PSDFixedRank` this *also* equals ``total_ambient_dim``
        (ambient-storage convention; plan §2.1).
    total_gauge_dim
        Sum of ``manifold.gauge_dim`` across leaves. ``0`` for v1 trees.
    """

    leaf_specs: tuple[LeafSpec, ...] = field(default_factory=tuple)
    total_ambient_dim: int = 0
    total_dimension: int = 0
    total_gauge_dim: int = 0


__all__ = ["ManifoldSpec", "LeafSpec"]
