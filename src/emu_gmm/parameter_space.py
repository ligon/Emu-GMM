r"""Ergonomic :class:`ParameterSpace` declaration layer (#107).

This is **additive sugar** over the validated manifold core (manifold
epic #12). It does *not* change the math or the internal representation:
a bound :class:`ParameterSpace` instance is an ordinary
``@jdc.pytree_dataclass`` whose non-scalar fields are
:class:`~emu_gmm.manifolds.manifold_leaf.ManifoldLeaf`-wrapped, i.e. a
valid ``theta_init`` that flows through :func:`emu_gmm.estimate`
unchanged.

Motivation
----------
Today the parameter geometry is restated at every instance construction::

    theta_init = Normal(ManifoldLeaf(jnp.eye(K),  PSDFixedRank(K, K)),
                        ManifoldLeaf(jnp.zeros(K), Euclidean(K)))

The manifold is *instance-invariant* but repeated per-instance, and the
class field ``A: ManifoldLeaf`` is type-erased --- the class does not
document *which* manifold. :class:`ParameterSpace` lets the user declare
field -> manifold **once** in the class body via the :func:`on` field
descriptor::

    class Normal(ParameterSpace):
        A:  Array = on(PSDFixedRank(K, K), default=jnp.eye(K))
        mu: Array = on(Euclidean(K),       default=jnp.zeros(K))

    Normal.point()        # bound instance from per-field defaults (deterministic)
    Normal.point(seed)    # random on-manifold point (per-leaf random_point)

Both ``.point(...)`` calls return a :class:`ManifoldLeaf` PyTree (a valid
``theta_init``).

Integration mechanism
---------------------
We use ``__init_subclass__`` on :class:`ParameterSpace`: when a user
subclasses it, the hook reads the class' ``on(...)`` field descriptors,
records the per-field manifold + default in ``__emu_fields__``, and then
**lowers the subclass to a real** ``@jdc.pytree_dataclass`` with one
:class:`ManifoldLeaf`-typed field per declaration (declaration order). The
leaves are therefore the *same* ``ManifoldLeaf`` nodes the core already
understands, and the bound instance is a genuine ``dataclasses`` dataclass:
``manifold_spec_from_params`` / ``param_names`` / ``flatten_params*`` read
its field names and topology exactly as they do for a hand-built
``@jdc.pytree_dataclass`` of ``ManifoldLeaf`` fields. We chose
``__init_subclass__`` over a decorator (no extra call site) and over a
custom metaclass (no metaclass conflict; the hook is the minimal surface
that does the job); ``jdc`` does the actual PyTree registration so we do
not hand-roll flatten/unflatten.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from emu_gmm.manifolds.base import ManifoldParam
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf

# Sentinel: no default supplied to ``on(...)``.
_NO_DEFAULT = object()


@dataclasses.dataclass(frozen=True)
class _FieldSpec:
    """One declared ``field -> manifold`` mapping (the ``on(...)`` payload)."""

    manifold: ManifoldParam
    default: Any  # raw array-like default, or ``_NO_DEFAULT``


def on(manifold: ManifoldParam, default: Any = _NO_DEFAULT) -> Any:
    """Field descriptor declaring the manifold (and optional default) of a field.

    Used in a :class:`ParameterSpace` class body::

        class Normal(ParameterSpace):
            A:  Array = on(PSDFixedRank(K, K), default=jnp.eye(K))
            mu: Array = on(Euclidean(K),       default=jnp.zeros(K))

    Parameters
    ----------
    manifold
        The :class:`~emu_gmm.manifolds.base.ManifoldParam` governing this
        field. Must satisfy the protocol (all native manifolds do).
    default
        Optional deterministic default point for this field, consumed by
        :meth:`ParameterSpace.point` (the no-seed call). An array-like of
        the manifold's ``ambient_shape``. When omitted, ``.point()`` raises
        a clear error for this field --- there is deliberately no implicit
        canonical point (``random_point`` is random, not canonical).

    Returns
    -------
    _FieldSpec
        An opaque descriptor stashed as the class attribute;
        :meth:`ParameterSpace.__init_subclass__` consumes it.
    """
    if not isinstance(manifold, ManifoldParam):
        raise TypeError(
            "on(...): manifold must satisfy the ManifoldParam protocol; "
            f"got {type(manifold).__name__}"
        )
    return _FieldSpec(manifold=manifold, default=default)


class ParameterSpace:
    """Declarative parameter-geometry base class (#107).

    Subclass and annotate each field with :func:`on` to declare its
    manifold once::

        class Normal(ParameterSpace):
            A:  Array = on(PSDFixedRank(K, K), default=jnp.eye(K))
            mu: Array = on(Euclidean(K),       default=jnp.zeros(K))

    The *class* is the parameter space; a *bound instance* is the space
    located at a point. Build a bound instance with :meth:`point`:

    * ``Normal.point()`` --- the per-field ``default`` point (deterministic).
    * ``Normal.point(seed)`` --- a random on-manifold point, composing each
      leaf manifold's ``random_point(key_i)`` with keys split from
      ``jax.random.PRNGKey(seed)``.

    Both return a :class:`ManifoldLeaf` PyTree (a valid ``theta_init``). The
    subclass is lowered to a ``@jdc.pytree_dataclass`` with one
    :class:`ManifoldLeaf`-typed field per declaration, so it flows through
    :func:`emu_gmm.estimate` exactly like a hand-built dataclass of
    ``ManifoldLeaf`` fields. Subclassing a :class:`ParameterSpace` merges the
    parent's declared fields (parents first, then the child's; a re-declared
    field overrides the parent spec but keeps its inherited position).

    A space whose fields are *all* Euclidean is estimated with the default
    ``optimistix_lm`` optimiser (Euclidean leaves carry no gauge/curvature), and
    non-scalar Euclidean leaves are fully supported: a space of only non-scalar
    Euclidean leaves (e.g. a bare ``Euclidean(K)`` mean vector with no
    constrained/gauge leaf) estimates correctly via the ambient flatten path.
    (This was the #110 dispatch trap --- the all-Euclidean path used to route to
    the scalar-only v1 flatten, which rejected non-0-d leaves; fixed by keying
    the flatten/Jacobian representation on whether every leaf is scalar rather
    than on the optimiser dispatch mode.)
    """

    # Populated per-subclass by __init_subclass__: ordered field -> _FieldSpec.
    __emu_fields__: ClassVar[dict[str, _FieldSpec]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # --- FIX 2 (declaration order) ---
        # Collect *this class's own* declared fields in true definition order.
        # ``vars(cls)`` (the class body dict) preserves definition order
        # regardless of whether the field was annotated (``A: Array = on(...)``)
        # or bare (``mu = on(...)``); keying off ``__annotations__`` first
        # would float the annotated fields ahead of the bare ones and corrupt
        # the order that ``param_names`` / ``manifold_spec`` read off. We do
        # NOT touch ``__annotations__`` for order.
        own_fields: dict[str, _FieldSpec] = {}
        for name, val in vars(cls).items():
            if isinstance(val, _FieldSpec):
                own_fields[name] = val

        # --- FIX 1 (two-level inheritance: MERGE across the MRO) ---
        # ``__init_subclass__`` runs once per subclass and (below) lowers the
        # class to a jdc dataclass keyed on ITS OWN field set. A subclass of a
        # ParameterSpace therefore must re-establish the *full* field set:
        # parent-declared fields first (in MRO order, parents before child),
        # then this class's own fields, preserving declaration order. A child
        # re-declaring a parent field OVERRIDES the parent's spec but KEEPS the
        # parent's position (the field is not moved to the end) -- documented
        # dedupe choice: inherited position wins, child spec wins.
        merged: dict[str, _FieldSpec] = {}
        for base in reversed(cls.__mro__[1:]):  # parents first, child last
            base_fields = base.__dict__.get("__emu_fields__")
            if isinstance(base_fields, dict):
                for name, fs in base_fields.items():
                    merged[name] = fs  # parent position established here
        for name, fs in own_fields.items():
            merged[name] = fs  # child overrides spec; keeps inherited position

        fields = merged
        if not fields:
            raise TypeError(
                f"ParameterSpace subclass {cls.__name__!r} declares no "
                "fields; annotate at least one field with on(manifold, ...)."
            )
        cls.__emu_fields__ = fields

        # Lower to a real @jdc.pytree_dataclass: one ManifoldLeaf-typed field
        # per MERGED declaration (parent + own), in declaration order. We
        # rewrite the class' annotations to ManifoldLeaf for the FULL merged
        # set (FIX 1: a subclass otherwise inherits only the parent's lowered
        # field and ``point()`` raises missing-positional-arg) and strip the
        # _FieldSpec class attrs so the dataclass machinery sees plain
        # (no-default) fields. The resulting class is a genuine dataclass whose
        # pytree children are the per-field ManifoldLeaf nodes --- the exact
        # topology the manifold core already validates.
        for name in fields:
            if name in cls.__dict__:
                delattr(cls, name)
        cls.__annotations__ = {name: ManifoldLeaf for name in fields}
        jdc.pytree_dataclass(cls)

    # ------------------------------------------------------------------
    # Point construction.
    # ------------------------------------------------------------------
    @classmethod
    def point(cls, seed: int | None = None) -> ParameterSpace:
        """Materialise a bound instance (a :class:`ManifoldLeaf` PyTree).

        Parameters
        ----------
        seed
            ``None`` (default): use each field's declared ``default``
            (deterministic). A field with no ``default`` raises a clear
            error --- there is no implicit canonical point.

            An ``int``: draw a random on-manifold point, splitting
            ``jax.random.PRNGKey(seed)`` into one subkey per field and
            calling each leaf manifold's ``random_point(key_i)``.

        Returns
        -------
        ParameterSpace
            A bound instance whose fields are :class:`ManifoldLeaf` values.
            A valid ``theta_init`` / ``parameters=`` argument.
        """
        declared = cls.__emu_fields__
        leaves: dict[str, ManifoldLeaf] = {}
        if seed is None:
            for name, fs in declared.items():
                if fs.default is _NO_DEFAULT or fs.default is None:
                    raise ValueError(
                        f"{cls.__name__}.point(): field {name!r} has no default "
                        "available (a None default is not usable); pass a "
                        "(non-None) default to on(manifold, default=...) or call "
                        ".point(seed) for a random start."
                    )
                # Cast EVERY default to canonical float64 (a parameter is a
                # continuous real). An int or float32 default would otherwise
                # pin an int64/float32 leaf whose treedef differs from the seed
                # path's float64 ``random_point`` (a spurious jit recompile on
                # switch; broken gradients for an int leaf) --- the float64
                # commitment. This matches ``point(seed)``'s dtype exactly.
                arr = jnp.asarray(fs.default).astype(jnp.float64)
                leaves[name] = ManifoldLeaf(arr, fs.manifold)
        else:
            key = jax.random.PRNGKey(int(seed))
            subkeys = jax.random.split(key, len(declared))
            for (name, fs), subkey in zip(declared.items(), subkeys, strict=True):
                arr = jnp.asarray(fs.manifold.random_point(subkey))
                leaves[name] = ManifoldLeaf(arr, fs.manifold)
        return cls(**leaves)


__all__ = ["ParameterSpace", "on"]
