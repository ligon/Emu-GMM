r"""Delta-method standard errors for functionals of the estimate (#42, Phase 7).

This is the gauge-invariant standard-error pull-back the consumer
(K-Aggregators) needs: interpretable SEs of *functionals* of the
gauge-invariant content of ``theta_hat`` -- the cross-price substitution
matrix ``Gamma = A @ A.T`` (its entries / eigenvalues) and ``phi``.

Background
----------
The Phase-4 minimal :math:`\Sigma_\theta` is the ambient-horizontal
``pinv_eigvalrule`` sandwich :math:`(G_{\mathrm{riem}}'\Lambda
G_{\mathrm{riem}})^+` over the *ambient* tangent
:math:`\theta=(\mathrm{vec}(Y),\phi)`. Two facts about it are load-bearing
here (commitments 5, 7; design brief; red-team R1/R7):

* It is reported in the **natural / ambient (convention-B) coordinate
  system** -- the retraction differential (unit at :math:`v=0` for every
  native retraction) was already folded into ``G_riem`` at Phase 4, so the
  delta method applies to :math:`J_f` taken in those *same* ambient
  coordinates with **no further retraction scaling** (R1). The flat ambient
  vector is ``concat([comp.ravel() for comp in components()])`` in
  leaf-walk order, C/row-major within each leaf -- exactly the order
  :func:`emu_gmm._internal.params.flatten_params_with_spec` lays down and
  the order ``Sigma_theta``'s axes are sized by.

* The **exact** ``total_gauge_dim`` :math:`= \sum_\ell k_\ell(k_\ell-1)/2`
  gauge directions are pinned to zero (``pinv_eigvalrule`` dropped exactly
  that many smallest eigenvalues). So :math:`\Sigma_\theta` annihilates the
  O(K) gauge fibre.

Method (delta method, gauge-invariant by construction)
------------------------------------------------------
For a functional :math:`f(\theta)=g(\Gamma(\theta),\phi)` that depends on
:math:`\theta` ONLY through gauge invariants:

* :math:`J_f = \partial f/\partial\theta` evaluated at the **fixed**
  :math:`\hat\theta` (AD of ``f`` composed with the ambient reshape on the
  flat ambient vector; commitment 5 -- at :math:`\hat\theta`, **never**
  through the solver's ``lax.while_loop``);
* :math:`\mathrm{Cov}(f)=J_f\,\Sigma_\theta\,J_f'`;
  :math:`\mathrm{SE}(f)=\sqrt{\mathrm{diag}}`.

Gauge invariance is then automatic: a gauge-invariant :math:`f` has
:math:`J_f` orthogonal to the O(K) gauge fibre, so it annihilates the same
nullspace already pinned out of :math:`\Sigma_\theta`. The result is
identical for two gauge-equivalent solves ``Y0`` and ``Y0 @ Q`` (R6/R9/R17).
A gauge-VIOLATING :math:`f` (one that leaks raw ``Y`` not only through
``Gamma``) picks up nonzero variance from those directions -- the negative
control that proves the invariance of the good functionals is real, not an
artefact of always-zero (R6/R9/R32).

Coordinate-system contract (important)
--------------------------------------
``f`` MUST be a function of the **ambient components** ``(A, phi, ...)`` --
the exact arrays :meth:`emu_gmm.types.EstimationResult.components` returns --
in their natural (convention-B) scale. ``Sigma_theta`` is in those ambient
coordinates; the formula applies with no extra scaling. Do **not** pass
``f`` in a log / manifold-intrinsic / otherwise transformed basis (R7).
``f`` must be JAX-AD-able and gauge-invariant; gauge-invariance cannot be
checked at runtime and is the caller's responsibility (R32).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

# Threshold (relative to the largest |eigenvalue|) below which an
# eigenvalue of Gamma is treated as a structural zero of a rank-deficient
# PSD and excluded from ``eigenvalue_se`` (R3/R5/R20/R27).
_EV_REL_FLOOR = 1e-10


def _component_shapes(components: Sequence[Any]) -> list[tuple[int, ...]]:
    return [tuple(int(s) for s in jnp.asarray(c).shape) for c in components]


def _flatten_components(components: Sequence[Any]) -> Float[Array, " D"]:
    """Concatenate per-leaf ambient arrays into the flat ambient vector.

    C/row-major within each leaf, in leaf-walk order -- the exact layout
    ``flatten_params_with_spec`` produces and ``Sigma_theta`` is sized by.
    A 0-d scalar leaf contributes one entry (matching the v1 flatten).
    """
    blocks = [jnp.reshape(jnp.asarray(c), (-1,)) for c in components]
    if not blocks:
        return jnp.zeros((0,))
    return jnp.concatenate(blocks)


def _unflatten_to_components(
    theta_flat: Float[Array, " D"],
    shapes: Sequence[tuple[int, ...]],
) -> tuple[Any, ...]:
    """Inverse of :func:`_flatten_components`: slice + reshape per leaf.

    Pure JAX (no PyTree treedef, no ManifoldLeaf re-wrapping): the
    delta-method ``f`` works directly on the raw ambient arrays, so the AD
    closure never touches ``unflatten_params`` and there is no hidden
    coordinate transformation to mismatch ``Sigma_theta`` (R2/R4/R11/R31).
    """
    out: list[Any] = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape)) if shape != () else 1
        block = jax.lax.dynamic_slice_in_dim(theta_flat, offset, size)
        if shape == ():
            out.append(jnp.reshape(block, ()))
        else:
            out.append(jnp.reshape(block, shape))
        offset += size
    return tuple(out)


def functional_se(
    f: Callable[[tuple[Any, ...]], Any],
    components: Sequence[Any],
    sigma_theta: Float[Array, "D D"],
) -> tuple[Float[Array, " p"], Float[Array, "p p"]]:
    r"""Delta-method SE and covariance of ``f(components)``.

    Parameters
    ----------
    f
        A callable mapping the components tuple ``(A, phi, ...)`` (exactly
        what :meth:`EstimationResult.components` returns, same order) to a
        1-D JAX array of length ``p`` (a scalar / 0-d is treated as
        ``p == 1``). ``f`` must be JAX-AD-able and **gauge-invariant**: it
        must depend on each gauge-bearing leaf only through that leaf's
        gauge invariants (e.g. ``Gamma = A @ A.T`` for a ``PSDFixedRank``
        ``A``). A gauge-violating ``f`` returns a *valid but
        gauge-dependent* SE (used as a negative control); the routine
        cannot detect the violation (R32).
    components
        The per-leaf ambient arrays at ``theta_hat`` (``result.components()``).
    sigma_theta
        The Phase-4 ambient ``(D, D)`` covariance (``result.Sigma_theta``
        array), with the exact gauge nullspace already pinned to zero.

    Returns
    -------
    se
        ``(p,)`` standard errors ``sqrt(diag(Cov))`` (negative diagonal
        entries from finite-precision round-off clip to ``nan``, matching
        :attr:`EstimationResult.standard_errors`).
    cov
        ``(p, p)`` covariance ``J_f @ Sigma_theta @ J_f.T``, symmetrised.

    Notes
    -----
    Uses ``jacrev`` -> ``J_f`` is ``(p, D)`` and the sandwich is
    ``J_f @ Sigma_theta @ J_f.T`` -> ``(p, p)`` (R8). The Jacobian is taken
    at the FIXED ``theta_hat`` (delta method; never through the solver --
    R16). Eager-only: call outside any ``jax.jit`` boundary (R36).
    """
    sigma = jnp.asarray(getattr(sigma_theta, "array", sigma_theta))
    shapes = _component_shapes(components)
    theta_flat = _flatten_components(components)
    D = int(theta_flat.shape[0])
    if sigma.shape != (D, D):
        raise ValueError(
            f"functional_se: Sigma_theta has shape {tuple(sigma.shape)} but the "
            f"flattened components imply ambient dimension D={D}; the two must "
            "match (this indicates a manifold-spec / components routing bug)"
        )

    def g(tf: Float[Array, " D"]) -> Float[Array, " p"]:
        comps = _unflatten_to_components(tf, shapes)
        y = jnp.asarray(f(tuple(comps)))
        y = jnp.atleast_1d(y)
        if y.ndim != 1:
            raise ValueError(
                "functional_se: f must return a scalar or a 1-D array; got "
                f"output of ndim {y.ndim} (shape {tuple(y.shape)}). Flatten "
                "the output inside f (e.g. ravel a matrix to vech)."
            )
        return y

    J_f = jnp.asarray(jax.jacrev(g)(theta_flat))  # (p, D)
    p = int(J_f.shape[0])
    if J_f.shape != (p, D):
        raise ValueError(
            f"functional_se: Jacobian has shape {tuple(J_f.shape)}, expected "
            f"(p, D) with D={D}"
        )
    cov = J_f @ sigma @ J_f.T
    cov = 0.5 * (cov + cov.T)  # neutralise rounding asymmetry (R33)
    diag = jnp.diag(cov)
    se = jnp.sqrt(jnp.where(diag >= 0.0, diag, jnp.nan))  # clip tiny negatives
    return se, cov


def _gamma_from_components(components: Sequence[Any]) -> Float[Array, "n n"]:
    """``Gamma = A @ A.T`` from the first (PSDFixedRank) component.

    The K-Aggregators contract: the first leaf is the ``PSDFixedRank``
    factor ``A`` of the cross-price substitution matrix.
    """
    A = jnp.asarray(components[0])
    return A @ A.T


def vech_indices(n: int) -> tuple[Array, Array]:
    """Row-major lower-triangular ``(i, j)`` index arrays for ``vech``.

    The canonical ``vech`` order (R13/R29): iterate rows ``i = 0..n-1`` and
    within each row columns ``j = 0..i``, i.e.
    ``Gamma[0,0], Gamma[1,0], Gamma[1,1], Gamma[2,0], ...`` -- length
    ``n(n+1)/2``. Returned as a pair of index arrays so callers can pull the
    entries with ``Gamma[ii, jj]``.
    """
    ii, jj = jnp.tril_indices(n)
    return ii, jj


def gamma_vech(components: Sequence[Any]) -> Float[Array, " q"]:
    """Lower-triangular vectorisation ``vech(Gamma)`` (row-major).

    ``Gamma = A @ A.T`` from the first component; length ``n(n+1)/2``.
    """
    G = _gamma_from_components(components)
    n = int(G.shape[0])
    ii, jj = vech_indices(n)
    return G[ii, jj]


def gamma_se(
    components: Sequence[Any],
    sigma_theta: Float[Array, "D D"],
) -> tuple[Float[Array, " q"], Float[Array, "q q"]]:
    r"""Delta-method SE / covariance of ``vech(Gamma)``, ``Gamma = A @ A.T``.

    Returns ``(se, cov)`` for the ``q = n(n+1)/2`` unique lower-triangular
    entries of ``Gamma`` in row-major ``vech`` order (see
    :func:`vech_indices`). Gauge-invariant: ``Gamma`` is unchanged under
    ``A -> A @ Q`` for ``Q in O(K)``.
    """
    return functional_se(gamma_vech, components, sigma_theta)


def gamma_eigenvalues(components: Sequence[Any], k: int) -> Float[Array, " k"]:
    """The ``k`` largest eigenvalues of ``Gamma = A @ A.T`` (ascending).

    For a rank-``k`` ``Gamma in R^{n x n}`` these are the ``k`` *nonzero*
    eigenvalues; the ``n - k`` structural zeros are excluded. Slicing the
    top ``k`` of ``eigvalsh`` (ascending) keeps the AD away from the
    degenerate zero block, whose eigenvalue Jacobian is undefined
    (R3/R5/R18/R20/R27). ``eigvalsh`` returns eigenvalues in ascending
    order, so the ``k`` returned values are likewise ascending.
    """
    G = _gamma_from_components(components)
    ev = jnp.linalg.eigvalsh(G)  # ascending, length n
    n = int(ev.shape[0])
    if k > n:
        raise ValueError(
            f"gamma_eigenvalues: requested k={k} eigenvalues but Gamma is " f"{n}x{n}"
        )
    return ev[n - k :]  # top-k (largest), ascending


def eigenvalue_se(
    components: Sequence[Any],
    sigma_theta: Float[Array, "D D"],
    rank: int,
) -> tuple[Float[Array, " k"], Float[Array, "k k"]]:
    r"""Delta-method SE / covariance of the ``rank`` nonzero eigenvalues of
    ``Gamma = A @ A.T`` -- the K-Aggregators primary (R3/R5/R18/R27).

    Parameters
    ----------
    components, sigma_theta
        As in :func:`functional_se`.
    rank
        The number ``k`` of nonzero eigenvalues (the rank of the
        ``PSDFixedRank`` factor). The returned vectors have length ``k``,
        ordered ascending to match ``jnp.linalg.eigvalsh``. The ``n - k``
        structural zeros are NOT returned: their eigenvalue Jacobian is
        degenerate (the zero block is a repeated eigenvalue) so their SE is
        undefined, and they carry no semantic content for the consumer.

    Degenerate eigenvalues
    ----------------------
    The eigenvalue map is non-differentiable where two of the *returned*
    (nonzero) eigenvalues coincide: the individual eigenvalues are then not
    smooth functions of ``Gamma`` (only symmetric functions of the
    degenerate block -- their sum / mean -- are). ``jax.numpy.linalg.eigvalsh``
    does not crash there (it returns a finite, but **eigenbasis-dependent and
    hence non-unique**, derivative); the resulting per-eigenvalue SE is
    therefore not well-defined at exact degeneracy and should not be trusted.
    This is a measure-zero / non-generic event (the generic case is distinct
    nonzero eigenvalues, where the SEs are well-defined and exact). If the
    spectrum of ``Gamma_hat`` has (near-)repeated nonzero eigenvalues, report
    a symmetric functional of the degenerate block (e.g. its sum via
    :func:`functional_se`) instead of the individual eigenvalues. A
    near-degenerate (small gap) spectrum yields large but finite SEs -- the
    honest reflection of the ill-conditioning of the split.
    """

    def f(comps: tuple[Any, ...]) -> Float[Array, " k"]:
        return gamma_eigenvalues(comps, rank)

    return functional_se(f, components, sigma_theta)


def count_nonzero_eigenvalues(
    components: Sequence[Any], rel_floor: float = _EV_REL_FLOOR
) -> int:
    """Numerically-nonzero eigenvalue count of ``Gamma = A @ A.T``.

    Eigenvalues with ``|lambda| > rel_floor * max|lambda|`` count as
    nonzero. Used by :meth:`EstimationResult.eigenvalue_se` to default the
    ``rank`` argument when the caller omits it (R20).
    """
    G = _gamma_from_components(components)
    ev = jnp.linalg.eigvalsh(G)
    mx = float(jnp.max(jnp.abs(ev))) if ev.shape[0] else 0.0
    if mx == 0.0:
        return 0
    return int(jnp.sum(jnp.abs(ev) > rel_floor * mx))


__all__ = [
    "functional_se",
    "gamma_se",
    "gamma_vech",
    "vech_indices",
    "gamma_eigenvalues",
    "eigenvalue_se",
    "count_nonzero_eigenvalues",
]
