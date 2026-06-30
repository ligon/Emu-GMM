"""Post-estimation inference helpers for emu-gmm.

This subpackage hosts weak-identification-robust and specification
diagnostics that operate on a hypothesised :math:`\\theta_0` plus the
same ``(model, measure, covariance)`` triple used by
:func:`emu_gmm.estimate`, plus resampling-based standard-error helpers.

Currently exposed:

- :func:`k_statistic` and :class:`KStatisticResult` --- Kleibergen
  (Econometrica 2005) :math:`K`-, :math:`S`-, :math:`J`-statistic
  decomposition. See :mod:`emu_gmm.inference.k_statistic` for the
  full derivation.
- :func:`k_confidence_set` and :class:`KConfidenceSet` --- the
  identification-robust confidence set by grid inversion of the
  K-statistic (or J/S for Anderson--Rubin-style sets), with explicit
  empty / interval / disconnected / open-edge topology. See
  :mod:`emu_gmm.inference.confidence_set`.
- :func:`profiled_k_confidence_set` --- the profiled (nuisance-concentrated)
  sibling of :func:`k_confidence_set`: re-optimises the nuisance parameter
  leaves (a manifold ``PSDFixedRank`` factor included) at each grid value
  before inverting the K/S/J statistic. Also reachable via
  ``k_confidence_set(..., profile=[...])`` (#176).
- :func:`j_test` and :class:`JTestResult` --- zero-parameter test of
  over-identifying restrictions. Returns ``J = m' V^{-1} m ~ chi^2_M``
  evaluated at a user-supplied ``theta_null``, without invoking the
  :func:`emu_gmm.estimate` minimisation loop.
- :func:`identification_strength` and :class:`IdentificationStrength` ---
  per-parameter-block identification-strength diagnostic (the
  concentration / partial-first-stage curvature of each block, gauge-aware).
  See :mod:`emu_gmm.inference.identification`.
- :func:`moment_wild_bootstrap` and :class:`WildBootstrapResult` ---
  cluster-wild Rademacher / Mammen J-statistic bootstrap for moment
  models with NaN-masked moments and few clusters. See
  :mod:`emu_gmm.inference.wild_bootstrap` for the algorithm description
  and v1 scope.
- :func:`cluster_bootstrap` and :class:`ClusterBootstrapResult` ---
  refit-based cluster bootstrap helper for
  :class:`~emu_gmm.covariance.clustered.ClusteredCovariance`.
- :func:`adaptive_bootstrap` and :class:`AdaptiveBootstrapResult` ---
  a precision-targeted (Andrews-Buchinsky) *stopping rule* wrapping any
  batched bootstrap: draw batches until the reported functional
  (:class:`BootstrapMean` / :class:`BootstrapSE` / :class:`BootstrapQuantile`
  / :class:`BootstrapPValue`) meets a Monte Carlo precision target, with a
  loud ``converged`` flag at ``b_max``. See
  :mod:`emu_gmm.inference.adaptive`.
"""

from __future__ import annotations

from emu_gmm.inference.adaptive import (
    AdaptiveBootstrapResult,
    BootstrapMean,
    BootstrapPValue,
    BootstrapQuantile,
    BootstrapSE,
    adaptive_bootstrap,
    maritz_jarrett_quantile_se,
)
from emu_gmm.inference.cluster_bootstrap import (
    ClusterBootstrapResult,
    cluster_bootstrap,
)
from emu_gmm.inference.confidence_set import (
    KConfidenceSet,
    k_confidence_set,
    profiled_k_confidence_set,
)
from emu_gmm.inference.functional_se import (
    eigenvalue_se,
    functional_se,
    gamma_se,
    gamma_vech,
    vech_indices,
)
from emu_gmm.inference.identification import (
    BlockStrength,
    IdentificationStrength,
    identification_strength,
)
from emu_gmm.inference.j_test import JTestResult, j_test
from emu_gmm.inference.k_statistic import KStatisticResult, k_statistic
from emu_gmm.inference.wild_bootstrap import (
    WildBootstrapResult,
    moment_wild_bootstrap,
)

__all__ = [
    "AdaptiveBootstrapResult",
    "BlockStrength",
    "BootstrapMean",
    "BootstrapPValue",
    "BootstrapQuantile",
    "BootstrapSE",
    "ClusterBootstrapResult",
    "IdentificationStrength",
    "JTestResult",
    "KConfidenceSet",
    "KStatisticResult",
    "WildBootstrapResult",
    "adaptive_bootstrap",
    "cluster_bootstrap",
    "maritz_jarrett_quantile_se",
    "eigenvalue_se",
    "functional_se",
    "gamma_se",
    "gamma_vech",
    "identification_strength",
    "j_test",
    "k_confidence_set",
    "k_statistic",
    "moment_wild_bootstrap",
    "profiled_k_confidence_set",
    "vech_indices",
]
