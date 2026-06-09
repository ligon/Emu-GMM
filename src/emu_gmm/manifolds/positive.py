"""The scalar :class:`Positive` manifold :math:`\\mathbb{R}_{>0}`.

Models the open positive half-line as a 1-D Riemannian manifold with the
affine-invariant (log-Euclidean / Fisher--Rao) metric

.. math::

    g_x(u, v) = \\frac{u\\,v}{x^2}.

This metric makes :math:`\\mathbb{R}_{>0}` geodesically complete: the
boundary at ``0`` recedes to infinite distance, so the exponential
retraction never steps to :math:`x \\le 0`. This is the load-bearing
property that unblocks downstream ``sigma > 0`` parameters (Seasonality):
a plain ``x + v`` Euclidean retraction can cross zero; the exponential
retraction :math:`R_x(v) = x \\exp(v / x)` cannot.

Storage convention (plan §2.1): both points and tangent vectors are 0-d
JAX scalars (``ambient_shape == ()``), so a :class:`Positive` leaf is
flatten/unflatten bitwise-identical to a v1 scalar leaf --- no
:class:`ManifoldLeaf` PyTree node is needed for the scalar slice. Modeled
as a frozen, field-free dataclass so it is hashable and rides as a static
field through ``jit``. Float64 by package import.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Positive:
    """The open positive half-line with the affine-invariant metric.

    A single positive scalar parameter :math:`x > 0`. Dimension 1, no
    gauge structure (``gauge_dim == 0``), ambient shape ``()``.

    Notes
    -----
    All ``Positive()`` instances are equal and hash-equal (no fields),
    satisfying the :class:`~emu_gmm.manifolds.spec.ManifoldSpec`
    frozen-dataclass requirement.
    """

    @property
    def ambient_shape(self) -> tuple[int, ...]:
        """A single scalar; ``()``."""
        return ()

    @property
    def dimension(self) -> int:
        """Intrinsic dimension ``1`` (``int(prod(())) == 1``; plan §2.8)."""
        return 1

    @property
    def gauge_dim(self) -> int:
        """No quotient structure; ``0``."""
        return 0

    # ------------------------------------------------------------------
    # ManifoldParam operators.
    # ------------------------------------------------------------------
    def projection(self, point: Any, ambient_vector: Any) -> Any:  # noqa: ARG002
        """Identity: the tangent space of a 1-D manifold is all of R."""
        del point
        return ambient_vector

    def retraction(self, point: Any, tangent_vector: Any) -> Any:
        """Exponential retraction :math:`R_x(v) = x \\exp(v / x)`.

        The exact affine-invariant exponential map on
        :math:`\\mathbb{R}_{>0}`. Geodesically complete: keeps ``x > 0``
        for any ``v`` in R, and is tangent to ``x + v`` to first order
        (matches the Euclidean retraction at ``v = 0``).
        """
        return point * jnp.exp(tangent_vector / point)

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:
        """Affine-invariant metric :math:`g_x(u, v) = u v / x^2`."""
        return jnp.sum(jnp.asarray(u) * jnp.asarray(v)) / (point**2)

    def norm(self, point: Any, tangent_vector: Any) -> Any:
        """Riemannian norm ``sqrt(inner_product(v, v)) = |v| / x``."""
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def riemannian_gradient(self, point: Any, euclidean_gradient: Any) -> Any:
        """Alias for :meth:`euclidean_to_riemannian_gradient`."""
        return self.euclidean_to_riemannian_gradient(point, euclidean_gradient)

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:
        """Riemannian gradient :math:`x^2\\,\\mathrm{egrad}`.

        From :math:`g_x(\\mathrm{rgrad}, v) = \\langle \\mathrm{egrad},
        v\\rangle = \\mathrm{egrad}\\,v` for all ``v``: ``rgrad v / x^2 =
        egrad v`` gives ``rgrad = x^2 egrad``. This is the only
        non-identity operator and is what makes the estimator's
        information matrix differ from the raw ambient Jacobian form.
        """
        return point**2 * euclidean_gradient

    def retraction_differential(self, point: Any) -> Any:
        """Retraction differential :math:`dR_x(v)/dv|_{v=0} = 1`.

        For :math:`R_x(v) = x\\,e^{v/x}`, :math:`dR_x/dv = e^{v/x}`, so the
        differential at :math:`v = 0` is :math:`1` --- as it must be for
        any first-order retraction (:math:`DR_x(0) = \\mathrm{Id}`). The
        tangent coordinate ``v`` is therefore the *natural* (ambient)
        coordinate: a unit ``v`` perturbation maps to a unit ``sigma``
        perturbation at first order.

        Covariance convention (matches ``../ManifoldGMM``; see
        ``docs/implementation-plan-v2-manifold.org`` §2.4): the reported
        ``Sigma_theta`` is the ambient / natural-scale inverse-information
        ``(G' Lambda G)^{-1}`` (the efficient-GMM form; not a sandwich --
        see issue #133 for the robust-sandwich correction) --- i.e. ``Var(sigma_hat)``, NOT the
        log-scale ``Var(log sigma_hat)``. Scaling column ``j`` of ``G`` by
        this differential (``1``) leaves the information matrix at its
        ambient value. The affine-invariant :meth:`inner_product` /
        :meth:`norm` / :meth:`euclidean_to_riemannian_gradient` (``x^2``)
        remain on the manifold for protocol completeness and to derive the
        retraction, but are deliberately NOT used to rescale the reported
        covariance. The manifold's value-add is the positivity-preserving
        retraction, not a different asymptotic variance: first-order
        asymptotics are parameterisation-invariant at an interior optimum.
        """
        return jnp.ones_like(jnp.asarray(point))

    def distance(self, point_a: Any, point_b: Any) -> Any:
        """Geodesic distance :math:`|\\log b - \\log a|`."""
        return jnp.abs(jnp.log(point_b) - jnp.log(point_a))

    def random_point(self, key: Any) -> Any:
        """Draw a log-normal positive scalar of shape ``()``."""
        return jnp.exp(jax.random.normal(key, (), dtype=jnp.float64))

    def random_tangent_vector(self, key: Any, point: Any) -> Any:  # noqa: ARG002
        """Standard-normal tangent scalar (tangent space is all of R)."""
        del point
        return jax.random.normal(key, (), dtype=jnp.float64)

    def zero_vector(self, point: Any) -> Any:  # noqa: ARG002
        """Zero tangent scalar."""
        del point
        return jnp.zeros(())

    # ------------------------------------------------------------------
    # Label generation (plan §2.10).
    # ------------------------------------------------------------------
    def tangent_basis_names(self, field_name: str) -> list[str]:
        """Scalar leaf: ``[field_name]`` (matches ``Euclidean()`` / v1)."""
        return [field_name]


__all__ = ["Positive"]
