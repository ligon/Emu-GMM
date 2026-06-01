r"""Gauge-aware pseudo-inverse via a fixed-count eigenvalue rule (Phase 4).

The information matrix of a GMM problem whose parameter lives on a quotient
manifold (e.g. :class:`~emu_gmm.manifolds.psd_fixed_rank.PSDFixedRank`, where
:math:`Y \sim Y Q` for :math:`Q \in O(k)`) is **structurally rank-deficient**:
the horizontal-projected sandwich :math:`G' \Lambda G` has an exact
``gauge_dim``-dimensional nullspace (the gauge / vertical directions, with
``gauge_dim == k(k-1)/2`` for ``PSDFixedRank(n, k)``). A plain
:func:`jax.numpy.linalg.inv` on such a matrix returns ``inf`` / ``nan`` (or
huge finite garbage that mimics weak identification).

:func:`pinv_eigvalrule` builds the Moore--Penrose pseudo-inverse on the
**identified** subspace by dropping exactly ``drop_smallest`` eigenvalues of
the symmetrised info matrix **by count** (a static Python int -- vmap/jit-safe,
per the ManifoldGMM PR #33 convention), *not* by a magnitude threshold. The
count is the manifold spec's ``total_gauge_dim``, known concretely at trace
time, so the eigenvalue slice ``[drop_smallest:]`` has a static shape.

Why count, not threshold (red-team R2/R10)
------------------------------------------
A magnitude threshold conflates a *small but identified* eigenvalue (genuine
weak identification, which the user must see) with an *exact-zero gauge*
eigenvalue (a property of the quotient, harmless and expected). Dropping by a
fixed count keeps the two distinct: the dropped directions are *always* the
``gauge_dim`` smallest, the gauge nullspace is reported separately as
``gauge_nullspace_dim`` in the diagnostics, and the rule never silently
absorbs a weakly-identified direction.

Why ``eigh`` (red-team R1/R12/R23)
----------------------------------
``info`` is symmetric (``Z' Z``). :func:`jax.numpy.linalg.eigh` returns real
eigenvalues in **ascending** order with orthonormal eigenvectors, so the
``drop_smallest`` smallest sit at indices ``[0:drop_smallest]`` and the
reconstruction ``Q_keep diag(1/w_keep) Q_keep'`` is numerically stable. We
symmetrise (``0.5 * (info + info.T)``) first to neutralise the tiny asymmetry
floating-point rounding leaves in ``Z' Z`` (red-team R23).

v1 bitwise non-regression (red-team R13)
----------------------------------------
For ``drop_smallest == 0`` (every v1 / all-Euclidean / scalar-Positive tree,
``total_gauge_dim == 0``) the function returns :func:`jax.numpy.linalg.inv`
**bitwise** -- it short-circuits to ``inv`` rather than reconstructing through
``eigh`` -- so the landed Positive(1,1) slice and the 226 v1 tests are
preserved exactly.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def pinv_eigvalrule(
    info: Float[Array, "D D"],
    *,
    drop_smallest: int,
) -> Float[Array, "D D"]:
    r"""Pseudo-inverse dropping the ``drop_smallest`` smallest eigenvalues.

    Parameters
    ----------
    info
        A symmetric ``(D, D)`` information matrix (typically ``Z' Z`` from
        the horizontal-projected GMM sandwich). Symmetrised internally
        before the eigendecomposition, so a tiny floating-point asymmetry
        in ``info`` is harmless.
    drop_smallest
        A **static Python int** (NOT a JAX array / tracer): the number of
        smallest eigenvalues to drop, equal to the manifold spec's
        ``total_gauge_dim``. Must satisfy ``0 <= drop_smallest <= D``. When
        ``0`` the function returns :func:`jax.numpy.linalg.inv(info)`
        bitwise (v1 non-regression).

    Returns
    -------
    The ``(D, D)`` pseudo-inverse reconstructed from the
    ``D - drop_smallest`` largest eigenpairs:
    ``Q_keep @ diag(1 / w_keep) @ Q_keep.T``. The result is the true
    Moore--Penrose pseudo-inverse on the identified subspace (the
    ``drop_smallest`` gauge directions are pinned to exact zero).

    Examples
    --------
    For ``Product(PSDFixedRank(5, 2), Euclidean(1))`` the gauge dimension
    is ``2 * 1 // 2 == 1``::

        Sigma = pinv_eigvalrule(G_T_Lambda_G, drop_smallest=1)
    """
    if not isinstance(drop_smallest, int):
        raise TypeError(
            "pinv_eigvalrule: drop_smallest must be a static Python int "
            f"(vmap/jit-safe), got {type(drop_smallest).__name__}. Pass "
            "manifold_spec.total_gauge_dim directly."
        )
    if drop_smallest < 0:
        raise ValueError(
            f"pinv_eigvalrule: drop_smallest must be >= 0, got {drop_smallest}"
        )

    # ---- v1 bitwise path: no gauge nullspace -> exact inv() (red-team R13).
    if drop_smallest == 0:
        return jnp.linalg.inv(info)

    # Symmetrise to neutralise rounding asymmetry, then eigh (ascending
    # eigenvalues, orthonormal eigenvectors; red-team R1/R12/R23).
    info_sym = 0.5 * (info + info.T)
    w, q = jnp.linalg.eigh(info_sym)

    # Drop the ``drop_smallest`` smallest eigenvalues BY COUNT. eigh returns
    # ascending order, so the gauge zeros are the FIRST ``drop_smallest``
    # entries; keep the trailing identified block (red-team R1/R11).
    w_keep = w[drop_smallest:]
    q_keep = q[:, drop_smallest:]
    return (q_keep * (1.0 / w_keep)) @ q_keep.T


__all__ = ["pinv_eigvalrule"]
