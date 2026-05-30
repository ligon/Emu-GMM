"""Trivial :class:`Euclidean` manifold reference implementation.

A flat reference for :class:`ManifoldParam`. The protocol's operators
collapse to identity / array arithmetic: the projection is the identity,
the retraction is vector addition, the Riemannian gradient *is* the
ambient gradient. See plan §2.8 for the v1 back-compat path: a v1-style
0-d scalar leaf maps to ``Euclidean()`` (empty ``ambient_shape``,
``dimension=1``), **not** ``Euclidean(1)`` (shape ``(1,)``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class Euclidean:
    """Flat real Euclidean space :math:`\\mathbb{R}^{\\text{shape}}`.

    Parameters
    ----------
    *shape
        Ambient shape of one parameter array. ``Euclidean()`` (no args)
        models a 0-d scalar (the v1 back-compat case); ``Euclidean(d)``
        models a length-``d`` vector; ``Euclidean(m, n)`` models a 2-D
        matrix block.

    Notes
    -----
    The :class:`Euclidean` manifold has :math:`\\text{gauge\\_dim} = 0`
    (no quotient structure). All operators are linear and pure.
    """

    gauge_dim: int = 0

    def __init__(self, *shape: int) -> None:
        self.ambient_shape: tuple[int, ...] = tuple(int(s) for s in shape)
        # np.prod(()) == 1.0; cast back to int for the protocol.
        self.dimension: int = int(np.prod(self.ambient_shape))

    # ------------------------------------------------------------------
    # Hash / equality so :class:`ManifoldSpec` can be a frozen dataclass.
    # ------------------------------------------------------------------
    def __hash__(self) -> int:  # noqa: D401
        return hash(("Euclidean", self.ambient_shape))

    def __eq__(self, other: object) -> bool:  # noqa: D401
        if not isinstance(other, Euclidean):
            return NotImplemented
        return self.ambient_shape == other.ambient_shape

    def __repr__(self) -> str:
        if not self.ambient_shape:
            return "Euclidean()"
        return f"Euclidean{self.ambient_shape!r}"

    # ------------------------------------------------------------------
    # ManifoldParam operators (all linear / identity for Euclidean).
    # ------------------------------------------------------------------
    def projection(self, point: Any, ambient_vector: Any) -> Any:  # noqa: ARG002
        """Identity: ambient vectors are already tangent."""
        return ambient_vector

    def retraction(self, point: Any, tangent_vector: Any) -> Any:
        """Vector addition: :math:`R_x(v) = x + v`."""
        return point + tangent_vector

    def riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:  # noqa: ARG002
        """Identity: the embedded gradient equals the ambient gradient."""
        return euclidean_gradient

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:  # noqa: ARG002
        """Phase-4 canonical name; identity for :class:`Euclidean`."""
        return euclidean_gradient

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:  # noqa: ARG002
        """Standard Euclidean inner product :math:`\\sum_i u_i v_i`."""
        return jnp.sum(jnp.asarray(u) * jnp.asarray(v))

    def norm(self, point: Any, tangent_vector: Any) -> Any:
        """Riemannian norm; ``sqrt(inner_product(v, v))``."""
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def distance(self, point_a: Any, point_b: Any) -> Any:
        """Frobenius distance :math:`\\lVert a - b\\rVert_F`."""
        return jnp.linalg.norm(jnp.asarray(point_a) - jnp.asarray(point_b))

    def random_point(self, key: Any) -> Any:
        """Draw an i.i.d.\\ standard-normal sample of shape ``ambient_shape``."""
        return jax.random.normal(key, self.ambient_shape, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Label generation (plan §2.10).
    # ------------------------------------------------------------------
    def tangent_basis_names(self, field_name: str) -> list[str]:
        """Return one label per ambient coordinate.

        For ``Euclidean()`` (scalar / v1 leaf): ``[field_name]`` (the
        leaf name itself; matches v1 :func:`param_names` output exactly).
        For ``Euclidean(d)``: ``["<field>_t_0", ..., "<field>_t_{d-1}"]``.
        For ``Euclidean(m, n)``: ``["<field>_t_<i>_<j>"]`` flattened in
        row-major order.
        """
        if not self.ambient_shape:
            return [field_name]
        labels: list[str] = []
        for idx in np.ndindex(self.ambient_shape):
            suffix = "_".join(str(i) for i in idx)
            labels.append(f"{field_name}_t_{suffix}")
        return labels


__all__ = ["Euclidean"]
