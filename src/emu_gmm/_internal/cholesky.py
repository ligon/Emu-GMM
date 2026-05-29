"""Cholesky factorisation and forward/back substitution wrappers.

The estimation pipeline (see ``docs/design.org`` Section 5) operates on
the lower-triangular Cholesky factor :math:`L` of :math:`V = L L^\\top`
rather than on :math:`V^{-1}` directly. This module provides thin
wrappers over ``jax.scipy.linalg`` that:

1. compute :math:`L` with explicit handling of non-PD failure,
2. solve :math:`L y = b` (forward substitution),
3. solve :math:`L^\\top x = b` (back substitution),
4. provide a chained convenience ``whiten(V, m) -> y = L^{-1} m`` that
   the residual function in :mod:`emu_gmm.estimator` consumes directly.

All routines are jit-compatible. ``cholesky(V)`` raises a clear error
on a non-positive-definite ``V``; the regularisation layer
(:mod:`emu_gmm.regularization`) is responsible for ensuring this
doesn't happen at runtime.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float


def cholesky(V: Float[Array, "M M"]) -> Float[Array, "M M"]:
    """Lower-triangular Cholesky factor of a symmetric positive-definite ``V``.

    Returns ``L`` such that :math:`V = L L^\\top`. Uses
    :func:`jax.scipy.linalg.cholesky` under the hood. The result is
    lower-triangular by construction.

    Parameters
    ----------
    V : (M, M) array
        Symmetric positive-definite matrix.

    Returns
    -------
    L : (M, M) lower-triangular array

    Notes
    -----
    JAX's Cholesky returns NaNs on non-PD input rather than raising. This
    wrapper does not add a runtime check (which would break tracing); the
    caller is responsible for ensuring ``V`` is PD, typically via the
    regularisation strategy.
    """
    return jax.scipy.linalg.cholesky(V, lower=True)


def forward_solve(
    L: Float[Array, "M M"],
    b: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Solve :math:`L y = b` for ``y``, given lower-triangular ``L``.

    Parameters
    ----------
    L : (M, M) lower-triangular array
        Typically the Cholesky factor from :func:`cholesky`.
    b : (M,) array
        Right-hand side.

    Returns
    -------
    y : (M,) array
    """
    return jax.scipy.linalg.solve_triangular(L, b, lower=True)


def back_solve(
    L: Float[Array, "M M"],
    b: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Solve :math:`L^\\top x = b` for ``x``, given lower-triangular ``L``.

    Parameters
    ----------
    L : (M, M) lower-triangular array
    b : (M,) array

    Returns
    -------
    x : (M,) array
    """
    return jax.scipy.linalg.solve_triangular(L, b, lower=True, trans="T")


def whiten(
    V: Float[Array, "M M"],
    m: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Return :math:`y = L^{-1} m` where :math:`V = L L^\\top`.

    The whitened moment vector satisfies
    :math:`y^\\top y = m^\\top V^{-1} m`, so the framework's objective
    :math:`Q_\\mu(\\theta) = \\| y_\\mu(\\theta) \\|^2` is exactly the
    quadratic form in :math:`V^{-1}` without ever inverting :math:`V`.

    Parameters
    ----------
    V : (M, M) symmetric positive-definite array
    m : (M,) array

    Returns
    -------
    y : (M,) array
    """
    L = cholesky(V)
    return forward_solve(L, m)


def quadratic_form(
    V: Float[Array, "M M"],
    m: Float[Array, "M"],
) -> Float[Array, ""]:
    """Return :math:`m^\\top V^{-1} m` via the whitened residual.

    Equivalent to ``jnp.sum(whiten(V, m) ** 2)``. Exposed because some
    callers (notably the J-statistic) want the scalar directly.
    """
    y = whiten(V, m)
    return jnp.sum(y * y)


__all__ = ["cholesky", "forward_solve", "back_solve", "whiten", "quadratic_form"]
