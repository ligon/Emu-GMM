"""Kleibergen K-statistic and its K/S/J decomposition.

Kleibergen (Econometrica 2005, "Testing Parameters in GMM without Assuming
that They are Identified", Vol. 73, No. 4, pp. 1103-1123) shows that the
J statistic at a hypothesised :math:`\\theta_0` admits an orthogonal
decomposition into a :math:`K` component (chi-squared with :math:`p`
degrees of freedom under :math:`H_0: \\theta = \\theta_0`, regardless of
identification strength) and a :math:`S` component (chi-squared with
:math:`M - p`). The :math:`K`-statistic alone is the headline
weak-IV-robust test; the :math:`S`-statistic is interpretable as an
overidentification residual orthogonal to the score direction.

Notation in this module:

==============  ==========================================  ======
Symbol          Definition                                  Shape
==============  ==========================================  ======
:math:`m`       :math:`\\mathbb{E}_\\mu[\\psi(\\cdot,\\theta_0)]`    (M,)
:math:`G`       :math:`\\nabla_\\theta \\mathbb{E}_\\mu[\\psi(\\cdot,\\theta_0)]` (M, p)
:math:`V`       Variance of the moment estimator            (M, M)
:math:`L`       Lower Cholesky factor: :math:`V^\\star = L L'` (M, M)
:math:`\\tilde m`  :math:`L^{-1} m`                            (M,)
:math:`\\tilde G`  :math:`L^{-1} G`                            (M, p)
:math:`P_G`     :math:`\\tilde G (\\tilde G' \\tilde G)^{-1} \\tilde G'` (M, M)
==============  ==========================================  ======

The decomposition is

.. math::

    K(\\theta_0) &= \\tilde m^\\top P_G \\tilde m \\;\\sim\\; \\chi^2_{p} \\\\
    J(\\theta_0) &= \\tilde m^\\top \\tilde m \\;\\sim\\; \\chi^2_{M} \\\\
    S(\\theta_0) &= J(\\theta_0) - K(\\theta_0) \\;\\sim\\; \\chi^2_{M - p}

with the two components asymptotically independent under
:math:`H_0: \\theta = \\theta_0`. See Kleibergen (2005) eqs. (15)-(17)
and Proposition 2.

The implementation computes :math:`P_G \\tilde m` via the thin QR
factorisation :math:`\\tilde G = Q R`: then :math:`P_G \\tilde m =
Q Q^\\top \\tilde m` and :math:`K = \\| Q^\\top \\tilde m \\|^2`. This is
numerically more stable than forming :math:`(\\tilde G' \\tilde G)^{-1}`
directly and inherits the same regularised :math:`V^\\star` that the
estimator pipeline uses (controlled by ``regularization``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import scipy.stats
from jaxtyping import Array, Float

from emu_gmm._internal import cholesky as cho
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    EstimationResult,
    Measure,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
)


@jdc.pytree_dataclass
class KStatisticResult:
    """Output of :func:`k_statistic`: K/S/J decomposition at :math:`\\theta_0`.

    Three chi-squared statistics plus their (static) degrees of freedom
    and (Python float) p-values. The pytree-dataclass packaging means the
    record can pass through ``jit`` / ``vmap`` boundaries when only the
    array-valued fields are needed; the p-values are evaluated on the
    host via :func:`scipy.stats.chi2.sf` and therefore are not traced.

    Attributes
    ----------
    K : float
        Kleibergen :math:`K`-statistic; :math:`\\chi^2_p` under
        :math:`H_0: \\theta = \\theta_0`, with :math:`p` = #parameters.
        Robust to weak identification.
    S : float
        Overidentification residual orthogonal to the score direction;
        :math:`\\chi^2_{M - p}` under :math:`H_0`.
    J : float
        Hansen :math:`J`-statistic at :math:`\\theta_0`; :math:`\\chi^2_M`
        under :math:`H_0`. Equals :math:`K + S` by construction.
    df_K, df_S, df_J : int (static)
        Degrees of freedom: ``df_K = p``, ``df_J = M``, ``df_S = M - p``.
    p_K, p_S, p_J : float
        Upper-tail chi-squared p-values for ``K``, ``S``, ``J``.
    """

    K: Float[Array, ""]
    S: Float[Array, ""]
    J: Float[Array, ""]
    df_K: int = jdc.static_field()  # type: ignore[attr-defined]
    df_S: int = jdc.static_field()  # type: ignore[attr-defined]
    df_J: int = jdc.static_field()  # type: ignore[attr-defined]
    p_K: float = jdc.static_field()  # type: ignore[attr-defined]
    p_S: float = jdc.static_field()  # type: ignore[attr-defined]
    p_J: float = jdc.static_field()  # type: ignore[attr-defined]


def _resolve_theta_null(
    result_or_theta_null: EstimationResult | ParamsLike,
) -> ParamsLike:
    """If the first arg is an :class:`EstimationResult`, return ``theta_hat``.

    Otherwise return the input unchanged. Lets callers write
    ``k_statistic(result, ...)`` to evaluate the decomposition at the
    point estimate (a diagnostic sanity check on the fitted model) or
    ``k_statistic(theta_0, ...)`` to test a non-trivial null.
    """
    if isinstance(result_or_theta_null, EstimationResult):
        return result_or_theta_null.theta_hat
    return result_or_theta_null


def _stats_from_whitened(
    m_tilde: Float[Array, " M"],
    G_tilde: Float[Array, "M K"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Return ``(K, S, J)`` from whitened ``m_tilde`` and ``G_tilde``.

    Uses a thin QR factorisation :math:`\\tilde G = Q R` (``mode="reduced"``);
    then :math:`K = \\|Q^\\top \\tilde m\\|^2`. This avoids forming
    :math:`(\\tilde G' \\tilde G)^{-1}` and is well-defined whenever
    :math:`\\tilde G` has full column rank (rank-deficient :math:`\\tilde G`
    is exactly the under-identified case; in that limit :math:`Q` still
    has orthonormal columns of dimension equal to the rank of
    :math:`\\tilde G`, but JAX's QR returns a fixed shape and the user
    will see a degenerate ``K`` --- callers should check rank upstream).
    """
    # Q : (M, K), R : (K, K). With M >= K, "reduced" gives a thin QR.
    Q, _R = jnp.linalg.qr(G_tilde, mode="reduced")
    proj = Q.T @ m_tilde  # (K,)
    K_stat = jnp.sum(proj * proj)
    J_stat = jnp.sum(m_tilde * m_tilde)
    S_stat = J_stat - K_stat
    return K_stat, S_stat, J_stat


def _to_plain(value: Any) -> Float[Array, "..."]:
    """Strip a haliax NamedArray wrapper if present."""
    if hasattr(value, "array"):
        return jnp.asarray(value.array)
    return jnp.asarray(value)


def k_statistic(
    result_or_theta_null: EstimationResult | ParamsLike,
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    *,
    regularization: RegularizationStrategy | None = None,
) -> KStatisticResult:
    """Compute the Kleibergen :math:`K`/:math:`S`/:math:`J` decomposition.

    Evaluates the three chi-squared statistics at a hypothesised
    :math:`\\theta_0` under the supplied ``(measure, covariance, model)``
    triple. By construction :math:`J = K + S`, and under
    :math:`H_0: \\theta = \\theta_0` they are asymptotically distributed as
    :math:`\\chi^2_M`, :math:`\\chi^2_p`, and :math:`\\chi^2_{M-p}`
    respectively, with :math:`K` and :math:`S` asymptotically independent.

    Parameters
    ----------
    result_or_theta_null : :class:`EstimationResult` or parameter dataclass
        Either a fitted :class:`EstimationResult` (in which case
        :math:`\\theta_0 = \\hat\\theta`, useful as a diagnostic at the
        point estimate) or a user-supplied parameter dataclass specifying
        the null.
    measure : :class:`emu_gmm.types.Measure`
        Integration operator. Same protocol as :func:`emu_gmm.estimate`.
    covariance : :class:`emu_gmm.types.CovarianceStrategy`
        Constructor for :math:`V_\\mu(\\theta_0)`. Same protocol as
        :func:`emu_gmm.estimate`.
    model : :data:`emu_gmm.types.StructuralModel`
        Per-observation residual ``psi(x, theta) -> (M,) array``.
    regularization : :class:`emu_gmm.types.RegularizationStrategy`, optional
        Adaptive PD-restoration applied to :math:`V` before factorisation.
        Defaults to :class:`emu_gmm.regularization.DiagonalTikhonov` with
        framework defaults (``kappa_target=1e6``), matching
        :func:`emu_gmm.estimate`.

    Returns
    -------
    :class:`KStatisticResult`
        ``K``, ``S``, ``J`` statistics with their degrees of freedom and
        p-values. ``K`` is the headline weak-identification-robust test;
        a non-rejection of ``K`` at level :math:`\\alpha` provides a
        confidence region for :math:`\\theta` that is valid uniformly
        over the identification strength (Kleibergen 2005, Proposition 2).

    Notes
    -----
    The computation:

    1. :math:`m = \\mathbb{E}_\\mu[\\psi(\\cdot, \\theta_0)]`,
       :math:`G = \\nabla_\\theta \\mathbb{E}_\\mu[\\psi(\\cdot, \\theta_0)]`,
       :math:`V = V_\\mu(\\theta_0)`.
    2. :math:`V^\\star, \\tau = \\text{regularization.apply}(V)` (ridge),
       :math:`L` lower Cholesky of :math:`V^\\star`.
    3. :math:`\\tilde m = L^{-1} m`,  :math:`\\tilde G = L^{-1} G`.
    4. Thin QR :math:`\\tilde G = Q R`,  :math:`P_G \\tilde m = Q Q^\\top \\tilde m`.
    5. :math:`K = \\|Q^\\top \\tilde m\\|^2`,  :math:`J = \\|\\tilde m\\|^2`,
       :math:`S = J - K`.

    Sign and projection convention: this matches Kleibergen (2005) eqs.
    (15)-(17): the :math:`K`-statistic is the *length* (squared) of the
    projection of the whitened moment vector onto the column space of
    the whitened Jacobian. The :math:`S`-statistic is the squared length
    of the *orthogonal complement*. Both are non-negative by construction.

    The :math:`G` used here is the population/measure Jacobian
    :math:`\\nabla_\\theta \\mathbb{E}_\\mu[\\psi]`. Kleibergen's original
    paper additionally subtracts the covariance between :math:`\\sqrt n
    \\bar\\psi` and the score (an "orthogonalised" :math:`\\tilde D`). For
    correctly-specified models and at the null hypothesis that quantity
    converges to :math:`G`, so the two forms are asymptotically
    equivalent; the simpler form is what every downstream consumer in
    this workspace uses.
    """
    theta_0 = _resolve_theta_null(result_or_theta_null)

    if regularization is None:
        regularization = DiagonalTikhonov()

    # 1. Moment vector, Jacobian, and variance at theta_0.
    m = _to_plain(measure.expectation(model, theta_0))
    G = _to_plain(measure.jacobian(model, theta_0))
    V = _to_plain(covariance.covariance(model, theta_0, measure))

    # 2. PD-restore V via the regularisation strategy.
    V_star, _tau = regularization.apply(V)

    # 3. Whiten via Cholesky.
    L = cho.cholesky(V_star)
    m_tilde = jax.scipy.linalg.solve_triangular(L, m, lower=True)
    G_tilde = jax.scipy.linalg.solve_triangular(L, G, lower=True)

    # 4-5. K / S / J via thin QR.
    K_stat, S_stat, J_stat = _stats_from_whitened(m_tilde, G_tilde)

    # Degrees of freedom (static).
    M = int(m.shape[0])
    p = int(G.shape[1])
    df_K = p
    df_J = M
    df_S = max(M - p, 0)

    # Host-side p-values; convert traced scalars to Python floats first.
    K_host = float(K_stat)
    S_host = float(S_stat)
    J_host = float(J_stat)
    p_K = float(scipy.stats.chi2.sf(K_host, df_K)) if df_K > 0 else float("nan")
    p_S = float(scipy.stats.chi2.sf(S_host, df_S)) if df_S > 0 else float("nan")
    p_J = float(scipy.stats.chi2.sf(J_host, df_J)) if df_J > 0 else float("nan")

    return KStatisticResult(
        K=K_stat,
        S=S_stat,
        J=J_stat,
        df_K=df_K,
        df_S=df_S,
        df_J=df_J,
        p_K=p_K,
        p_S=p_S,
        p_J=p_J,
    )


__all__ = ["k_statistic", "KStatisticResult"]
