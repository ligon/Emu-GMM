"""Post-estimation inference helpers for emu-gmm.

This subpackage hosts weak-identification-robust and specification
diagnostics that operate on a hypothesised :math:`\\theta_0` plus the
same ``(model, measure, covariance)`` triple used by
:func:`emu_gmm.estimate`.

Currently exposed:

- :func:`k_statistic` and :class:`KStatisticResult` --- Kleibergen
  (Econometrica 2005) :math:`K`-, :math:`S`-, :math:`J`-statistic
  decomposition. See :mod:`emu_gmm.inference.k_statistic` for the
  full derivation.
- :func:`j_test` and :class:`JTestResult` --- zero-parameter test of
  over-identifying restrictions. Returns ``J = m' V^{-1} m ~ chi^2_M``
  evaluated at a user-supplied ``theta_null``, without invoking the
  :func:`emu_gmm.estimate` minimisation loop.
"""

from __future__ import annotations

from emu_gmm.inference.j_test import JTestResult, j_test
from emu_gmm.inference.k_statistic import KStatisticResult, k_statistic

__all__ = ["JTestResult", "KStatisticResult", "j_test", "k_statistic"]
