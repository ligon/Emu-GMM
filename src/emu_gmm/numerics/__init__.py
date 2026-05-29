"""Numerical helpers used across the framework and by external callers.

This subpackage collects small, self-contained linear-algebra utilities
that don't belong to any particular estimator stage but are useful in
their own right --- both inside :mod:`emu_gmm` and to downstream
libraries (e.g. K-Aggregators uses :func:`ridge_inverse` to ridge an
``Omega^{-1}`` outside the main estimation path).

Public API:

- :func:`ridge_inverse` --- one-shot adaptive diagonal-Tikhonov ridge
  plus Cholesky inverse.
"""

from emu_gmm.numerics.ridge_inverse import ridge_inverse

__all__ = ["ridge_inverse"]
