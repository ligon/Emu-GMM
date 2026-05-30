"""Product manifold :class:`Product` (Phase 3, plan §5).

Composes a finite tuple of :class:`ManifoldParam` factors into a
single manifold. Points and tangent vectors are represented as
*tuples* of per-factor arrays (a JAX PyTree); every operator delegates
factor-wise.

Pymanopt's :class:`pymanopt.manifolds.product.Product` uses a
``_ProductTangentVector`` list-subclass that overloads arithmetic; the
JAX port replaces that with a plain ``tuple`` of arrays since
:func:`jax.tree_util.tree_map` does the arithmetic for free.

The :attr:`Product.ambient_shape` attribute is left unset (raises
:class:`NotImplementedError` on access) --- Products don't have a single
ambient shape, and the :class:`ManifoldSpec` flatten/unflatten path
walks the PyTree leaf-by-leaf with per-leaf shapes anyway. Nested
``Product`` instances are rejected at construction time (matches
pymanopt's :file:`product.py:24-25`); users compose by flattening
factors at construction.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from emu_gmm.manifolds.base import ManifoldParam


class Product:
    """Cartesian-product manifold of a finite tuple of :class:`ManifoldParam` factors.

    Parameters
    ----------
    *factors
        Manifold factors. Must be non-empty; no factor may itself be a
        :class:`Product` (flatten products at construction).

    Notes
    -----
    Points and tangent vectors are tuples ``(p_0, ..., p_{n-1})`` whose
    entries match the corresponding factor's ambient shape. All
    factor-wise operators delegate via :func:`jax.tree_util.tree_map`
    where the structure permits.
    """

    def __init__(self, *factors: ManifoldParam) -> None:
        if not factors:
            raise ValueError("Product requires at least one factor manifold")
        for f in factors:
            if isinstance(f, Product):
                raise ValueError(
                    "Nested Product manifolds are not allowed; flatten "
                    "factors at construction"
                )
        self.factors: tuple[ManifoldParam, ...] = tuple(factors)
        self.dimension: int = int(sum(int(f.dimension) for f in self.factors))
        self.gauge_dim: int = int(sum(int(f.gauge_dim) for f in self.factors))

    # ``ambient_shape`` would conflate per-factor shapes; not defined here.
    @property
    def ambient_shape(self) -> tuple[int, ...]:
        raise NotImplementedError(
            "Product manifolds have per-factor ambient shapes; the v2 "
            "ManifoldSpec carries them. Inspect Product.factors instead."
        )

    # ------------------------------------------------------------------
    # Hash / equality / repr (needed so ManifoldSpec stays hashable).
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(("Product", self.factors))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.factors == other.factors

    def __repr__(self) -> str:
        return f"Product({', '.join(repr(f) for f in self.factors)})"

    # ------------------------------------------------------------------
    # Factor-wise operators.
    # ------------------------------------------------------------------
    def projection(self, point: Any, ambient_vector: Any) -> tuple[Any, ...]:
        return tuple(
            f.projection(p, v)
            for f, p, v in zip(self.factors, point, ambient_vector, strict=True)
        )

    def retraction(self, point: Any, tangent_vector: Any) -> tuple[Any, ...]:
        return tuple(
            f.retraction(p, t)
            for f, p, t in zip(self.factors, point, tangent_vector, strict=True)
        )

    def retraction_differential(self, point: Any) -> tuple[Any, ...]:
        """Factor-wise retraction differential ``dR_x(v)/dv|_0``."""
        return tuple(
            f.retraction_differential(p)
            for f, p in zip(self.factors, point, strict=True)
        )

    def riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> tuple[Any, ...]:
        return tuple(
            f.riemannian_gradient(p, g)
            for f, p, g in zip(self.factors, point, euclidean_gradient, strict=True)
        )

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> tuple[Any, ...]:
        """Factor-wise Phase-4 gradient conversion."""
        return tuple(
            f.euclidean_to_riemannian_gradient(p, g)
            for f, p, g in zip(self.factors, point, euclidean_gradient, strict=True)
        )

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:
        """Sum of factor-wise Riemannian inner products."""
        per_factor = [
            f.inner_product(p, uu, vv)
            for f, p, uu, vv in zip(self.factors, point, u, v, strict=True)
        ]
        return jnp.sum(jnp.stack(per_factor))

    def norm(self, point: Any, tangent_vector: Any) -> Any:
        """Riemannian norm ``sqrt(inner_product(v, v))`` across factors."""
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def distance(self, point_a: Any, point_b: Any) -> Any:
        per_factor = jnp.stack(
            [
                f.distance(a, b)
                for f, a, b in zip(self.factors, point_a, point_b, strict=True)
            ]
        )
        return jnp.sqrt(jnp.sum(per_factor**2))

    def random_point(self, key: Any) -> tuple[Any, ...]:
        keys = jax.random.split(key, len(self.factors))
        return tuple(f.random_point(k) for f, k in zip(self.factors, keys, strict=True))

    def tangent_basis_names(self, field_name: str) -> list[str]:
        """Stitch factor-wise labels with a ``<field>_f<i>_`` prefix.

        For the rare case a :class:`Product` sits as a leaf inside a
        user's parameter PyTree, this distinguishes factor labels by
        index. The common case (Product as the *outer* manifold of a
        flat dataclass) bypasses this --- the dataclass's field names
        propagate via the v2
        :func:`emu_gmm._internal.labels.tangent_basis_names` helper.
        """
        labels: list[str] = []
        for i, f in enumerate(self.factors):
            sub = f.tangent_basis_names(f"{field_name}_f{i}")
            labels.extend(sub)
        return labels


__all__ = ["Product"]
