"""Runtime-checkable :class:`ManifoldParam` protocol.

Every concrete manifold under :mod:`emu_gmm.manifolds` satisfies this
protocol. See plan §2.7 for the rationale and the full surface; §2.9
for why ``gauge_dim`` is a required attribute (not a ``getattr``-with-
default lookup).

Storage convention (plan §2.1): tangent vectors and points share the
``ambient_shape`` array shape. The protocol does *not* require any
horizontal-basis transformation at the boundary; gauge bookkeeping is
the manifold's responsibility, surfaced via ``gauge_dim``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ManifoldParam(Protocol):
    """Protocol for a Riemannian manifold of parameters.

    Attributes
    ----------
    dimension
        The ambient (storage) dimension. For native manifolds this is
        ``int(prod(ambient_shape))``; for :class:`PSDFixedRank(n, k)`
        this is ``n * k`` (the ambient :math:`nk`, *not* the quotient
        :math:`nk - k(k-1)/2`; see plan §2.1).
    gauge_dim
        Dimension of the gauge nullspace. ``0`` for non-quotient
        manifolds (:class:`Euclidean`, full-rank SPD); ``k*(k-1)/2`` for
        :class:`PSDFixedRank(n, k)`; sum-of-factors for :class:`Product`.
    ambient_shape
        Shape of one ambient-storage array; used by the flatten/unflatten
        path to compute per-leaf offsets and reshape blocks back.

    Methods
    -------
    projection(point, ambient_vector) -> tangent_vector
        Project an ambient-shape vector onto the tangent space at
        ``point``. Idempotent.
    retraction(point, tangent_vector) -> point
        First-order retraction of a tangent vector at ``point`` back to
        the manifold.
    retraction_differential(point) -> scalar
        The per-coordinate retraction differential ``dR_x(v)/dv|_{v=0}``
        --- the factor mapping a tangent perturbation to its ambient
        image. ``1`` for :class:`Euclidean` (additive retraction);
        ``point`` for :class:`emu_gmm.manifolds.positive.Positive`
        (exponential retraction). This is the scaling the estimator
        applies to the Jacobian columns when building ``Sigma_theta`` in
        tangent coordinates (delta-method push-through), and the same
        ``step_scale`` the Riemannian LM solver uses for its
        metric-correct Gauss--Newton step. Distinct from
        ``euclidean_to_riemannian_gradient`` (the inverse-metric gradient
        conversion); the two must not be conflated for inference.
    riemannian_gradient(point, euclidean_gradient) -> tangent_vector
        Convert an ambient-space Euclidean gradient to a Riemannian
        gradient (tangent vector). For embedded-metric manifolds this is
        the projection of the Euclidean gradient.
    euclidean_to_riemannian_gradient(point, euclidean_gradient) -> tangent
        Phase-4 canonical name for the same conversion (matches plan §498
        and the ``v2-chunk-a-types-ops`` operator surface). For the v1
        :class:`Euclidean` reference it is the identity; for
        :class:`emu_gmm.manifolds.positive.Positive` it scales by
        ``x**2``. The estimator's information-matrix block calls this
        name explicitly; ``riemannian_gradient`` is retained as a
        backward-compatible alias on the concrete manifolds.
    inner_product(point, u, v) -> scalar
        Riemannian inner product of two tangent vectors at ``point``.
        Phase-4 addition (additive, non-breaking): required by
        :class:`emu_gmm.manifolds.riemannian_lm.RiemannianLM` for its
        metric-correct convergence test.
    norm(point, tangent_vector) -> scalar
        Riemannian norm of a tangent vector; defaults to
        ``sqrt(inner_product(point, v, v))``. Phase-4 addition.
    distance(point_a, point_b) -> scalar
        Geodesic (or chord) distance between two manifold points.
    random_point(key) -> point
        Sample a random manifold point given a ``jax.random.PRNGKey``.
    tangent_basis_names(field_name) -> list[str]
        Return ``dimension`` labels naming each ambient coordinate of a
        tangent vector. For :class:`Euclidean` with an empty
        ``ambient_shape`` (scalar v1 leaf) this is ``[field_name]``;
        otherwise the names embed the leaf field name plus a
        ``_t_<i><j>...`` suffix per ambient index.
    invariants() -> dict[str, Callable[[ambient_array], array]]
        The manifold's canonical **gauge-invariant** functionals of a leaf
        living on it, as ``{name: functional}``. Each functional maps the
        leaf's ambient-shape array to a 1-D array and depends on the point
        only through gauge invariants, so it is meaningful both per draw
        (empirical grade) and under the delta method (asymptotic grade).
        The flat manifolds (:class:`Euclidean` / ``Positive`` / ``Interval``)
        expose ``{"value": ravel}`` (the coordinate itself); a quotient
        manifold exposes its invariants (``PSDFixedRank`` ->
        ``{"eigenvalues", "gamma"}`` of :math:`\Gamma = A A^\top`, never the
        gauge-arbitrary raw factor). This is what lets an
        :class:`~emu_gmm.law.EstimatorLaw` offer a sensible per-leaf query
        set from geometry alone (``law.leaf(name).se("eigenvalues")``).
    """

    dimension: int
    gauge_dim: int
    ambient_shape: tuple[int, ...]

    def projection(self, point: Any, ambient_vector: Any) -> Any: ...

    def retraction(self, point: Any, tangent_vector: Any) -> Any: ...

    def retraction_differential(self, point: Any) -> Any: ...

    def riemannian_gradient(self, point: Any, euclidean_gradient: Any) -> Any: ...

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any: ...

    def inner_product(self, point: Any, u: Any, v: Any) -> Any: ...

    def norm(self, point: Any, tangent_vector: Any) -> Any: ...

    def distance(self, point_a: Any, point_b: Any) -> Any: ...

    def random_point(self, key: Any) -> Any: ...

    def tangent_basis_names(self, field_name: str) -> list[str]: ...

    def invariants(self) -> dict[str, Callable[[Any], Any]]: ...


__all__ = ["ManifoldParam"]
