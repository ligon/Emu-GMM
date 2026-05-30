"""The :class:`RiemannianOptimizer` protocol (plan §2.6 / §7).

A v2 optimiser that consumes a :class:`~emu_gmm.manifolds.spec.ManifoldSpec`
alongside the residual closure and the *original* parameter PyTree. The
extra ``manifold_spec`` third positional argument is what the estimator's
``_resolve_optimizer`` dispatch uses (via :func:`inspect.signature`) to
distinguish a v2 :class:`RiemannianOptimizer` from a v1
:class:`~emu_gmm.types.Optimizer` (which has only ``residual_fn`` and
``theta_init``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jaxtyping import Array, Float

from emu_gmm.manifolds.spec import ManifoldSpec

if TYPE_CHECKING:
    from emu_gmm.types import OptimizerInfo


@runtime_checkable
class RiemannianOptimizer(Protocol):
    """Manifold-aware non-linear least-squares solver callable.

    Solves ``min_theta || residual_fn(theta_flat) ||^2`` where the step
    is taken in the tangent space of the manifold described by
    ``manifold_spec`` and retracted back. The optimiser owns the
    flat <-> pytree round-trip; ``theta_init`` is the *original* PyTree.
    """

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Any,
        manifold_spec: ManifoldSpec,
    ) -> tuple[Any, OptimizerInfo]: ...


__all__ = ["RiemannianOptimizer"]
