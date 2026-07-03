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
from jaxtyping import Array, Float

from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule


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
    materialisation), :math:`G'\Lambda G` and :math:`G'\Lambda V\Lambda G`
    reproduce the whitened bread / meat by the SAME arithmetic, so this
    reconstruction is exact (not merely ``allclose``) against the pre-split
    ``Sigma_theta`` -- the property the golden-fidelity gate asserts.
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


__all__ = ["asymptotic_covariance"]
