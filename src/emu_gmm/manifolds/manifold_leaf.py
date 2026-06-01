r"""Opaque :class:`ManifoldLeaf` PyTree node (manifold epic #12, Phase 1).

A :class:`ManifoldLeaf` wraps one *non-scalar* parameter block (an
``(n, k)`` ``PSDFixedRank`` factor ``A``, a length-``d`` ``Euclidean``
vector, ...) together with the :class:`ManifoldParam` that governs it.
It is the v2 surface that lets the manifold-aware flatten core
(:func:`emu_gmm._internal.params.flatten_params_with_spec`) recover the
per-leaf manifold *and* its ambient shape from a parameter PyTree while
keeping the array itself a first-class traced child.

PyTree contract (the load-bearing part)
----------------------------------------
``tree_flatten(self)`` returns::

    children = (self.array,)          # exactly ONE traced child
    aux_data = (self.manifold, dtype) # static, hashable metadata only

so that:

* the wrapped array is the *only* leaf JAX sees --- ``vmap`` / ``grad`` /
  ``tree_map`` operate on it as an array, never on the manifold object
  (red-team R6);
* the manifold rides in ``aux_data``, which JAX keys ``jit`` / ``vmap``
  caches on by *hash + equality*. Every concrete manifold under
  :mod:`emu_gmm.manifolds` is hashable and immutable
  (``PSDFixedRank``/``Euclidean``/``Positive``/``Product``), so two
  structurally identical leaves produce identical ``treedef``\s and the
  cache is hit, not missed (red-team R7/R18/R19);
* ``aux_data`` carries *only* static, instance-independent metadata (the
  :class:`ManifoldParam` and the recorded dtype) --- never an array, an
  ``id()``, a cache, or a mutable container (red-team R19/R27). The dtype
  is recorded so a flatten -> unflatten round-trip through a (possibly
  promoted) flat buffer restores the leaf's *exact* dtype (red-team
  R8/R12).

The instance is immutable (frozen): mutation raises. Use tree operations
(``tree_map`` / ``tree_unflatten``) to produce a new leaf rather than
mutating in place (red-team R30).

This node is an *implementation detail of Phase 2+*. Phase-1 (all-scalar)
parameter trees do **not** wrap their leaves in :class:`ManifoldLeaf`;
the v1 ``flatten_params`` 2-tuple stays bitwise unchanged for them.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from emu_gmm.manifolds.base import ManifoldParam


@jax.tree_util.register_pytree_node_class
class ManifoldLeaf:
    r"""Immutable wrapper carrying one parameter array + its manifold.

    Parameters
    ----------
    array
        The ambient-storage parameter block (e.g. an ``(n, k)`` matrix
        for ``PSDFixedRank(n, k)``, a length-``d`` vector for
        ``Euclidean(d)``). Converted via :func:`jax.numpy.asarray`.
    manifold
        The :class:`ManifoldParam` governing this leaf. Must be hashable
        and immutable (all concrete manifolds in this package are).

    Notes
    -----
    The leaf is registered as a custom PyTree node whose single child is
    ``array`` and whose ``aux_data`` is ``(manifold, dtype)``. See the
    module docstring for the full PyTree contract and the red-team
    rationale.
    """

    __slots__ = ("array", "manifold")

    # Declared so static type-checkers see the slot attributes (they are
    # populated via object.__setattr__ to honour immutability).
    array: Any
    manifold: ManifoldParam

    def __init__(self, array: Any, manifold: ManifoldParam) -> None:
        if not isinstance(manifold, ManifoldParam):
            raise TypeError(
                "ManifoldLeaf.manifold must satisfy the ManifoldParam "
                f"protocol; got {type(manifold).__name__}"
            )
        # aux_data must be hashable so jit/vmap caches are stable
        # (red-team R7/R18). Fail fast at construction rather than at a
        # later, opaque cache-miss.
        try:
            hash(manifold)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                "ManifoldLeaf.manifold must be hashable (it rides in "
                f"PyTree aux_data); {type(manifold).__name__} is not"
            ) from exc
        object.__setattr__(self, "array", jnp.asarray(array))
        object.__setattr__(self, "manifold", manifold)

    # ------------------------------------------------------------------
    # Convenience accessor.
    # ------------------------------------------------------------------
    @property
    def ambient_shape(self) -> tuple[int, ...]:
        """Ambient shape of the wrapped block (the array's own shape)."""
        return tuple(int(s) for s in self.array.shape)

    # ------------------------------------------------------------------
    # Immutability (red-team R30): mutation must raise.
    # ------------------------------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ManifoldLeaf is immutable; build a new instance (or use "
            "tree_map / tree_unflatten) instead of assigning to "
            f"{name!r}."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ManifoldLeaf is immutable; cannot delete attributes.")

    # ------------------------------------------------------------------
    # PyTree registration.
    # ------------------------------------------------------------------
    def tree_flatten(self) -> tuple[tuple[Any], tuple[ManifoldParam, Any]]:
        """Return ``((array,), (manifold, dtype))``.

        Exactly one traced child (the array); ``aux_data`` is the
        ``(manifold, dtype)`` pair. Both entries are static, hashable,
        and instance-independent, so structurally identical leaves hash
        identically and the jit/vmap cache is hit.
        """
        return (self.array,), (self.manifold, self.array.dtype)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[ManifoldParam, Any], children: tuple[Any, ...]
    ) -> ManifoldLeaf:
        """Reconstruct from ``aux_data`` and ``children``.

        The child array is cast back to the recorded dtype so the leaf's
        dtype survives a round-trip through a promoted flat buffer
        (red-team R8/R12).
        """
        (array,) = children
        manifold, dtype = aux_data
        # Bypass __init__ validation: during tracing ``array`` may be a
        # tracer and ``manifold`` is already a validated manifold.
        obj = object.__new__(cls)
        object.__setattr__(obj, "array", jnp.asarray(array).astype(dtype))
        object.__setattr__(obj, "manifold", manifold)
        return obj

    # ------------------------------------------------------------------
    # Repr.
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        shape = getattr(self.array, "shape", None)
        return f"ManifoldLeaf(array.shape={shape}, manifold={self.manifold!r})"


__all__ = ["ManifoldLeaf"]
