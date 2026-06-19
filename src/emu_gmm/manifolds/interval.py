r"""The scalar :class:`Interval` manifold :math:`(\mathrm{lo}, \mathrm{hi})`.

Models a bounded open interval as a 1-D Riemannian manifold via the logit
transform :math:`\varphi(x) = \log\frac{x-\mathrm{lo}}{\mathrm{hi}-x}`, the
interval analogue of :class:`~emu_gmm.manifolds.positive.Positive`'s ``log``.
The pullback (Fisher--Rao-style) metric is

.. math::

    g_x(u, v) = \varphi'(x)^2\, u\, v, \qquad
    \varphi'(x) = \frac{\mathrm{hi}-\mathrm{lo}}{(x-\mathrm{lo})(\mathrm{hi}-x)},

which blows up at *both* endpoints, so :math:`\mathrm{lo}` and
:math:`\mathrm{hi}` recede to infinite geodesic distance: the exponential
retraction

.. math::

    R_x(v) = \varphi^{-1}\!\big(\varphi(x) + v\,\varphi'(x)\big)
           = \mathrm{lo} + (\mathrm{hi}-\mathrm{lo})\,
             \mathrm{sigmoid}\!\big(\varphi(x) + v\,\varphi'(x)\big)

never steps to or past either bound, for *any* ``v`` in R. It satisfies
``R_x(0) = x`` and ``DR_x(0) = Id`` (the tangent coordinate ``v`` is the
natural / ambient coordinate, exactly as for :class:`Positive`).

Motivation (#152, the CU regularity boundary): GMM/CUE asymptotics require a
COMPACT parameter space with the moment-covariance ``V_X`` bounded away from
singular. For a scale parameter ``sigma`` whose data variance can vanish,
``V_X(sigma) -> 0`` as ``sigma -> 0`` (the moment conditions become
non-stochastic), so ``W = V_X^{-1}`` is undefined at the boundary and the CUE
regularity condition fails. Restricting ``sigma`` to a compact
``[lo, hi]`` with ``lo > 0`` restores it: ``V_X`` is bounded away from singular
and second moments are uniformly finite on the box. :class:`Interval` is the
positivity-/boundedness-preserving geometry for such a parameter --- the same
role :class:`Positive` plays for the open half-line.

Storage convention mirrors :class:`Positive`: points and tangent vectors are
0-d JAX scalars (``ambient_shape == ()``), ``gauge_dim == 0``, and the reported
``Sigma_theta`` is on the natural (ambient) ``sigma`` scale
(``retraction_differential == 1``). Frozen dataclass with ``lo``/``hi`` float
fields so it is hashable and rides as a static field through ``jit``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Interval:
    """The bounded open interval ``(lo, hi)`` with the logit-pullback metric.

    A single bounded scalar parameter ``lo < x < hi``. Dimension 1, no gauge
    structure (``gauge_dim == 0``), ambient shape ``()``. Instances are equal /
    hash-equal iff their ``(lo, hi)`` match (the
    :class:`~emu_gmm.manifolds.spec.ManifoldSpec` frozen-dataclass requirement).
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (float(self.lo) < float(self.hi)):
            raise ValueError(
                f"Interval requires lo < hi; got lo={self.lo}, hi={self.hi}"
            )

    # ------------------------------------------------------------------
    @property
    def ambient_shape(self) -> tuple[int, ...]:
        """A single scalar; ``()``."""
        return ()

    @property
    def dimension(self) -> int:
        """Intrinsic dimension ``1``."""
        return 1

    @property
    def gauge_dim(self) -> int:
        """No quotient structure; ``0``."""
        return 0

    # ------------------------------------------------------------------
    def _phi_prime(self, x: Any) -> Any:
        """``phi'(x) = (hi - lo) / ((x - lo)(hi - x))`` (the metric scale)."""
        span = self.hi - self.lo
        return span / ((x - self.lo) * (self.hi - x))

    # ------------------------------------------------------------------
    # ManifoldParam operators.
    # ------------------------------------------------------------------
    def projection(self, point: Any, ambient_vector: Any) -> Any:  # noqa: ARG002
        """Identity: the tangent space of a 1-D manifold is all of R."""
        del point
        return ambient_vector

    def retraction(self, point: Any, tangent_vector: Any) -> Any:
        r"""Exponential retraction ``R_x(v) = lo + span*sigmoid(phi(x)+v*phi'(x))``.

        Stays strictly inside ``(lo, hi)`` for any ``v`` in R, with
        ``R_x(0) = x`` and ``DR_x(0) = 1`` (tangent to ``x + v`` to first order).
        """
        span = self.hi - self.lo
        phi_x = jnp.log((point - self.lo) / (self.hi - point))
        y = phi_x + tangent_vector * self._phi_prime(point)
        return self.lo + span * jax.nn.sigmoid(y)

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:
        """Pullback metric ``g_x(u, v) = phi'(x)^2 u v``."""
        phip = self._phi_prime(point)
        return jnp.sum(jnp.asarray(u) * jnp.asarray(v)) * (phip**2)

    def norm(self, point: Any, tangent_vector: Any) -> Any:
        """Riemannian norm ``sqrt(g_x(v, v)) = |v| * phi'(x)``."""
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def riemannian_gradient(self, point: Any, euclidean_gradient: Any) -> Any:
        """Alias for :meth:`euclidean_to_riemannian_gradient`."""
        return self.euclidean_to_riemannian_gradient(point, euclidean_gradient)

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:
        """Riemannian gradient ``egrad / phi'(x)^2`` (inverse metric).

        From ``g_x(rgrad, v) = egrad*v`` for all ``v``:
        ``rgrad*phi'(x)^2*v = egrad*v`` gives ``rgrad = egrad / phi'(x)^2``.
        """
        phip = self._phi_prime(point)
        return euclidean_gradient / (phip**2)

    def retraction_differential(self, point: Any) -> Any:
        """``dR_x(v)/dv|_{v=0} = 1`` (natural-scale convention, like Positive).

        ``Sigma_theta`` is reported on the ambient ``x`` scale (the column
        scaling is ``1``); the affine :meth:`inner_product` /
        :meth:`euclidean_to_riemannian_gradient` exist for protocol
        completeness and to derive the retraction, not to rescale the reported
        covariance (first-order asymptotics are parameterisation-invariant at an
        interior optimum).
        """
        return jnp.ones_like(jnp.asarray(point))

    def distance(self, point_a: Any, point_b: Any) -> Any:
        """Geodesic distance ``|phi(b) - phi(a)|``."""
        phi_a = jnp.log((point_a - self.lo) / (self.hi - point_a))
        phi_b = jnp.log((point_b - self.lo) / (self.hi - point_b))
        return jnp.abs(phi_b - phi_a)

    def random_point(self, key: Any) -> Any:
        """Draw a uniform point in ``(lo, hi)`` of shape ``()``."""
        u = jax.random.uniform(key, (), dtype=jnp.float64)
        return self.lo + (self.hi - self.lo) * u

    def random_tangent_vector(self, key: Any, point: Any) -> Any:  # noqa: ARG002
        """Standard-normal tangent scalar (tangent space is all of R)."""
        del point
        return jax.random.normal(key, (), dtype=jnp.float64)

    def zero_vector(self, point: Any) -> Any:  # noqa: ARG002
        """Zero tangent scalar."""
        del point
        return jnp.zeros(())

    # ------------------------------------------------------------------
    def tangent_basis_names(self, field_name: str) -> list[str]:
        """Scalar leaf: ``[field_name]`` (matches ``Positive()`` / ``Euclidean()``)."""
        return [field_name]


__all__ = ["Interval"]
