"""Riemannian manifold support (see docs/manifold-epic-progress.org).

A first-class problem-tuple menu, peer to Measures and Covariance strategies:
the user expresses a non-scalar / constrained parameter by wrapping each leaf in
``ManifoldLeaf(array, manifold)`` and estimates with the ordinary
``emu_gmm.estimate`` entry point (a non-Euclidean parameter auto-routes to
``riemannian_lm``). These types are re-exported at the top level, e.g.
``from emu_gmm import PSDFixedRank, Euclidean, Product, ManifoldLeaf``.

Estimation is gauge-aware end-to-end: recovery, a calibrated J-statistic, and
gauge-invariant standard errors on functionals of ``Gamma = A @ A.T`` via
``result.eigenvalue_se()`` / ``result.gamma_se()`` / ``result.functional_se(f)``.
(Epic #12, PRs #97--#103. ``ManifoldSpec`` / ``LeafSpec`` are exported
from ``emu_gmm.manifolds`` for advanced callers but are *not* re-exported
at the package top level --- they are flatten/inference plumbing, not part
of the everyday estimation surface.)

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
from emu_gmm.manifolds.interval import Interval
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.optimizer import RiemannianOptimizer
from emu_gmm.manifolds.positive import Positive
from emu_gmm.manifolds.product import Product
from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec

__all__ = [
    "ManifoldParam",
    "Euclidean",
    "ManifoldLeaf",
    "Interval",
    "Positive",
    "PSDFixedRank",
    "Product",
    "ManifoldSpec",
    "LeafSpec",
    "RiemannianOptimizer",
    "riemannian_lm",
]
