r"""The asymptotic (sandwich) covariance, as a pure function of optimization ingredients.

The estimation / inference split (``docs/optimization-result-law-split.org``):
the *optimization* produces the moment Jacobian :math:`G`, the realised
weighting matrix :math:`\Lambda`, and (at :math:`\hat\theta`) the raw moment
covariance :math:`V`; turning those into a sampling covariance of
:math:`\hat\theta` is the *inference* step, and it needs an assumption (a CLT).
This module is that step, factored out of both the estimator and
:class:`~emu_gmm.law.AsymptoticLaw` so there is ONE implementation and no
``types -> law`` import cycle (it lives under ``_internal``, which
:mod:`emu_gmm.types` may import freely).

The form is the #133 weighting-aware sandwich

.. math::
    \Sigma \;=\; B^{+}\, M\, B^{+},\qquad
    B = G'\Lambda G,\quad M = G'\Lambda V \Lambda G,

with :math:`B^{+}` the gauge-aware pseudo-inverse (drop ``gauge_dim`` smallest
eigenvalues BY COUNT; the gauge nullspace passes through as exact zeros). When
the weighting is efficient and the ridge does not bind (:math:`\Lambda =
V^{-1}`) the meat collapses to the bread and :math:`\Sigma = B^{-1}` -- the
inverse-information special case. Referencing this to a Gaussian law is the
caller's (the Law's) assumption; this function only does the algebra.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule
from emu_gmm.inference.numerical_rank import (
    NumericalRankPolicy,
    NumericalRankReport,
    PSDFactorGeometry,
)


def asymptotic_covariance(
    moment_jacobian: Float[Array, "M D"],
    weighting_matrix: Float[Array, "M M"],
    meat: Float[Array, "M M"],
    *,
    gauge_dim: int = 0,
) -> Float[Array, "D D"]:
    r"""The sandwich covariance :math:`B^{+} M B^{+}` from optimization ingredients.

    Parameters
    ----------
    moment_jacobian
        :math:`G = \partial\bar m/\partial\theta` at :math:`\hat\theta`, the
        (horizontal-projected) moment Jacobian, shape ``(M, D)``.
    weighting_matrix
        :math:`\Lambda`, the realised metric in :math:`\bar m'\Lambda\bar m`
        (for CU at convergence this is the frozen, ridged :math:`(V^\star)^{-1}`;
        for ``Identity`` / ``Fixed`` it is that fixed weight), shape ``(M, M)``.
    meat
        :math:`V`, the RAW (unregularised) moment covariance at
        :math:`\hat\theta`, shape ``(M, M)``. May be indefinite in a
        binding-ridge regime, which propagates to negative
        :math:`\mathrm{diag}(\Sigma)` and hence ``nan`` standard errors BY
        DESIGN (the #138 event; the caller surfaces it).
    gauge_dim
        Number of gauge nullspace directions to drop from the bread's
        pseudo-inverse BY COUNT (``manifold_spec.total_gauge_dim``; ``0`` for a
        Euclidean / all-scalar problem).

    Returns
    -------
    :math:`\Sigma`, the ``(D, D)`` asymptotic covariance, symmetrised.

    Notes
    -----
    Given :math:`\Lambda = W_i' W_i` with :math:`W_i = L_w^{-1}` (the estimator's
    materialisation), :math:`G'\Lambda G` and :math:`G'\Lambda V\Lambda G` are the
    SAME quantities as the estimator's whitened bread / meat
    (:math:`Z_w'Z_w` / :math:`Z_w'CZ_w`, :math:`Z_w = W_i G`) --- but formed by a
    different matrix-product ASSOCIATION (:math:`\Lambda` is materialised first,
    then contracted with :math:`G`), so the reconstruction agrees with the
    pre-split ``Sigma_theta`` to machine precision (~1e-16 absolute in practice),
    NOT literally bit-for-bit. The golden-fidelity gate asserts this at
    ``rtol=1e-9``.
    """
    G = jnp.asarray(moment_jacobian)
    Lam = jnp.asarray(weighting_matrix)
    V = jnp.asarray(meat)
    bread = G.T @ Lam @ G  # B = G' Lambda G
    meat_theta = G.T @ Lam @ V @ Lam @ G  # M = G' Lambda V Lambda G
    meat_theta = 0.5 * (meat_theta + meat_theta.T)
    bread_pinv = pinv_eigvalrule(bread, drop_smallest=int(gauge_dim))
    sigma = bread_pinv @ meat_theta @ bread_pinv
    return 0.5 * (sigma + sigma.T)


def ranked_asymptotic_covariance(
    moment_jacobian: Float[Array, "M D"],
    weighting_matrix: Float[Array, "M M"],
    meat: Float[Array, "M M"],
    *,
    policy: NumericalRankPolicy,
    gauge_dim: int,
    param_names: tuple[str, ...],
    leaf_shapes: tuple[tuple[int, ...], ...],
    factor_geometries: tuple[PSDFactorGeometry, ...] = (),
) -> tuple[np.ndarray, NumericalRankReport]:
    """Opt-in numerical retained-subspace form of the same raw-meat sandwich.

    This eager diagnostic branch leaves :func:`asymptotic_covariance` and its
    count-only JAX path unchanged. It uses the realised weight and raw meat,
    then applies the selected policy to the signed ambient bread spectrum.
    """
    if not isinstance(policy, NumericalRankPolicy):
        raise TypeError("policy must be a NumericalRankPolicy")
    G = jnp.asarray(moment_jacobian)
    Lam = jnp.asarray(weighting_matrix)
    V = jnp.asarray(meat)
    bread = np.asarray(G.T @ Lam @ G, dtype=float)
    meat_theta = np.asarray(G.T @ Lam @ V @ Lam @ G, dtype=float)
    bread = 0.5 * (bread + bread.T)
    meat_theta = 0.5 * (meat_theta + meat_theta.T)
    d = bread.shape[0]
    if bread.shape != (d, d) or not 0 <= gauge_dim < d:
        raise ValueError("bread shape or nominal gauge count is invalid")

    if not np.all(np.isfinite(bread)):
        eigenvalues = np.full(d, np.nan)
        retained = np.zeros(d, dtype=bool)
        projector = np.eye(d)
        sigma = np.full((d, d), np.nan)
        cutoff = float("nan")
        negative_allowance = float("nan")
        status = "nonfinite_bread"
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(bread)
        top = float(eigenvalues[-1])
        cutoff = policy.relative_tol * top
        negative_allowance = 64 * d * np.finfo(float).eps * top if top > 0 else 0.0
        if top <= 0:
            retained = np.zeros(d, dtype=bool)
            status = (
                "all_zero_bread" if np.all(eigenvalues == 0) else "no_positive_bread"
            )
        else:
            retained = (np.arange(d) >= gauge_dim) & (eigenvalues > cutoff)
            status = (
                "materially_negative_bread"
                if eigenvalues[0] < -negative_allowance
                else "ok"
            )
        excluded_vectors = eigenvectors[:, ~retained]
        projector = excluded_vectors @ excluded_vectors.T
        if status == "ok":
            kept_vectors = eigenvectors[:, retained]
            bread_pinv = (kept_vectors / eigenvalues[retained]) @ kept_vectors.T
            sigma = bread_pinv @ meat_theta @ bread_pinv
            sigma = 0.5 * (sigma + sigma.T)
        else:
            sigma = np.full((d, d), np.nan)

    largest_excluded = float(eigenvalues[~retained][-1]) if np.any(~retained) else None
    smallest_retained = float(eigenvalues[retained][0]) if np.any(retained) else None
    gap = (
        smallest_retained - largest_excluded
        if smallest_retained is not None and largest_excluded is not None
        else None
    )
    report = NumericalRankReport(
        policy=policy,
        signed_eigenvalues=eigenvalues,
        cutoff=cutoff,
        retained_mask=retained,
        excluded_projector=projector,
        effective_rank=int(retained.sum()),
        nominal_gauge_dim=gauge_dim,
        extra_excluded=int((~retained).sum()) - gauge_dim,
        largest_excluded_eigenvalue=largest_excluded,
        smallest_retained_eigenvalue=smallest_retained,
        excluded_cutoff_margin=(
            cutoff - largest_excluded if largest_excluded is not None else None
        ),
        retained_cutoff_margin=(
            smallest_retained - cutoff if smallest_retained is not None else None
        ),
        negative_allowance=negative_allowance,
        status=status,
        projector_boundary_ambiguous=(
            bool(gap <= 64 * d * np.finfo(float).eps * abs(eigenvalues[-1]))
            if gap is not None and np.isfinite(gap)
            else False
        ),
        param_names=param_names,
        leaf_shapes=leaf_shapes,
        factor_geometries=factor_geometries,
    )
    return sigma, report


__all__ = ["asymptotic_covariance", "ranked_asymptotic_covariance"]
