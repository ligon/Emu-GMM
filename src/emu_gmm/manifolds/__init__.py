"""Riemannian manifold support for v2 (see docs/implementation-plan-v2-manifold.org).

This package exposes :class:`ManifoldParam` (the runtime-checkable
protocol every concrete manifold satisfies), three native manifold
implementations (:class:`Euclidean`, :class:`PSDFixedRank`,
:class:`Product`), and the :class:`ManifoldSpec` / :class:`LeafSpec`
metadata containers used by the v2 flatten/unflatten path.

The manifold operators are JAX-native (jit/vmap/grad-clean in float64).
The package is import-time cheap: importing :mod:`emu_gmm.manifolds`
does not pull pymanopt, scipy.linalg.solve_continuous_lyapunov, or any
other optional dependency.

See plan §2.7 for the protocol surface, §2.1 for the ambient-storage
decision on :class:`PSDFixedRank`, §2.8 for v1 back-compatibility, and
§2.10 for the :func:`tangent_basis_names` contract.
"""

from __future__ import annotations

from emu_gmm.manifolds.base import ManifoldParam
from emu_gmm.manifolds.euclidean import Euclidean
from emu_gmm.manifolds.product import Product
from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec

__all__ = [
    "ManifoldParam",
    "Euclidean",
    "PSDFixedRank",
    "Product",
    "ManifoldSpec",
    "LeafSpec",
]
