"""Kleibergen K-statistic with orthogonalised Jacobian D-tilde.

Kleibergen (Econometrica 2005, "Testing Parameters in GMM without Assuming
that They are Identified", Vol. 73, No. 4, pp. 1103-1123) constructs a
weak-identification-robust test by replacing the raw moment Jacobian
:math:`G(\\theta_0)` with an *orthogonalised* Jacobian
:math:`\\widetilde D(\\theta_0)` that removes the part of :math:`G` that
is asymptotically correlated with the moment vector :math:`m(\\theta_0)`.
Concretely, column :math:`j` of :math:`\\widetilde D` is

.. math::
    \\widetilde D_j(\\theta_0)
    \\;=\\;
    G_j(\\theta_0)
    \\;-\\;
    \\Sigma_{G_j, m}(\\theta_0)\\, V(\\theta_0)^{-1}\\, m(\\theta_0)

where :math:`\\Sigma_{G_j, m}` is the :math:`M \\times M` cross-covariance
between the :math:`j`-th column of the per-observation moment Jacobian
and the per-observation moment vector. See Kleibergen (2005) eqs.
(8)-(9), and the modern restatements in Newey and Windmeijer (Econometrica
2009) and Hayashi's GMM chapter (3.6).

This module computes the K/S/J decomposition

.. math::

    K(\\theta_0) &= \\| \\mathrm{proj}_{\\,\\mathrm{col}(L^{-1} \\widetilde D)}\\, L^{-1} m \\|^2 \\;\\sim\\; \\chi^2_{p} \\\\
    J(\\theta_0) &= \\| L^{-1} m \\|^2 \\;\\sim\\; \\chi^2_{M} \\\\
    S(\\theta_0) &= J(\\theta_0) - K(\\theta_0) \\;\\sim\\; \\chi^2_{M - p}

where :math:`L L^\\top = V^\\star` is the Cholesky factor of the (adaptively
regularised) variance. The two components are asymptotically independent
under :math:`H_0: \\theta = \\theta_0`. See Kleibergen (2005) eqs. (16)-(17)
and Proposition 2; the headline property is that :math:`K(\\theta_0)`
remains :math:`\\chi^2_p` *regardless of identification strength* — the
property that the raw-:math:`G` form does not deliver.

Estimating :math:`\\Sigma_{G_j, m}` from a sample
----------------------------------------------------

The default implementation reads per-observation contributions from the
measure: :math:`g_i(\\theta_0) \\in \\mathbb{R}^M` (moment contributions)
and :math:`D_i(\\theta_0) \\in \\mathbb{R}^{M \\times p}` (Jacobian
contributions). The sample cross-covariance is then

.. math::
    \\widehat\\Sigma_{G_j, m}
    \\;=\\;
    \\frac{1}{N} \\sum_i (D_i)_j (g_i - \\bar g)^\\top
    \\;-\\; \\bar D_j \\bar g^\\top

(equivalently, the centred sample mean of :math:`(D_i)_j g_i^\\top`).

The :class:`~emu_gmm.measures.EmpiricalMeasure` and
:class:`~emu_gmm.measures.SyntheticMeasure` expose ``moment_contributions``
and ``jacobian_contributions`` methods that produce these per-observation
tensors directly. :class:`~emu_gmm.measures.AnalyticalMeasure` does not
have a finite-sample backing; users with closed-form populations may pass
a ``score_cov_fn`` keyword to supply :math:`\\Sigma_{G_j, m}` directly, or
accept the strong-identification fallback in which the correction is set
to zero (which recovers the older raw-:math:`G` form and is asymptotically
equivalent under strong identification with :math:`m(\\theta_0) = 0`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
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

    A :func:`jax_dataclasses.pytree_dataclass` so the record threads
    cleanly through ``jit`` / ``vmap`` boundaries: every field except
    the three integer degrees-of-freedom is a 0-d JAX array. The dofs
    are :func:`jdc.static_field` so they are baked into the compiled
    graph rather than traced.

    Attributes
    ----------
    K : 0-d float array
        Kleibergen :math:`K`-statistic; :math:`\\chi^2_p` under
        :math:`H_0: \\theta = \\theta_0`, with :math:`p` = #parameters.
        Robust to weak identification under the D-tilde construction.
    S : 0-d float array
        Overidentification residual orthogonal to the score direction;
        :math:`\\chi^2_{M - p}` under :math:`H_0`. Computed directly as
        :math:`\\|(I - QQ^\\top) \\tilde m\\|^2` for numerical
        non-negativity rather than via :math:`J - K` subtraction.
    J : 0-d float array
        Hansen :math:`J`-statistic at :math:`\\theta_0`; :math:`\\chi^2_M`
        under :math:`H_0`. Equals :math:`K + S` by construction.
    p_K, p_S, p_J : 0-d float array
        Upper-tail chi-squared p-values, computed via
        :func:`jax.scipy.stats.chi2.sf` so they trace under ``jit``.
        ``p_S`` is ``nan`` when ``df_S == 0`` (just-identified problem).
    df_K, df_S, df_J : int (static)
        Degrees of freedom: ``df_K = p``, ``df_J = M``, ``df_S = M - p``.
    """

    K: Float[Array, ""]
    S: Float[Array, ""]
    J: Float[Array, ""]
    p_K: Float[Array, ""]
    p_S: Float[Array, ""]
    p_J: Float[Array, ""]
    df_K: int = jdc.static_field()  # type: ignore[attr-defined]
    df_S: int = jdc.static_field()  # type: ignore[attr-defined]
    df_J: int = jdc.static_field()  # type: ignore[attr-defined]


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


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a :class:`haliax.NamedArray` wrapper if present.

    Mirrors the convention used by the framework's measure and covariance
    implementations: ``isinstance(value, ha.NamedArray)`` rather than
    duck-typing on ``hasattr(value, "array")`` so a non-haliax object
    with an unrelated ``.array`` attribute is not silently coerced.
    """
    if isinstance(value, ha.NamedArray):
        return jnp.asarray(value.array)
    return jnp.asarray(value)


def _sigma_jm_from_contributions(
    g: Float[Array, "N M"],
    D: Float[Array, "N M p"],
) -> Float[Array, "p M M"]:
    """Sample cross-covariance :math:`\\widehat\\Sigma_{G_j, m}` for each ``j``.

    Given per-observation moment contributions ``g`` and per-observation
    Jacobian contributions ``D``, returns a ``(p, M, M)`` tensor whose
    ``j``-th slice is

    .. math::
        \\widehat\\Sigma_{G_j, m}
        \\;=\\;
        \\frac{1}{N}\\sum_i (D_i)_j (g_i - \\bar g)^\\top
        \\;-\\; \\bar D_j (\\bar g - \\bar g)^\\top
        \\;=\\;
        \\frac{1}{N}\\sum_i (D_i)_j g_i^\\top
        \\;-\\; \\bar D_j \\bar g^\\top

    using the standard centred-cross-product identity to keep the
    expression jit-friendly and numerically stable.
    """
    n = g.shape[0]
    g_mean = jnp.mean(g, axis=0)  # (M,)
    D_mean = jnp.mean(D, axis=0)  # (M, p)
    # Cross-product per parameter j:
    #   sum_i D[i, :, j] * g[i, :]'
    # An einsum lays this out as (p, M, M).
    raw_cross = jnp.einsum("nmj,nk->jmk", D, g) / n  # (p, M, M)
    # Centring correction: subtract D_mean_j * g_mean'
    centred = raw_cross - jnp.einsum("mj,k->jmk", D_mean, g_mean)
    return centred


def _compute_d_tilde(
    measure: Measure,
    model: StructuralModel,
    theta_0: ParamsLike,
    m: Float[Array, " M"],
    G: Float[Array, "M p"],
    V_star: Float[Array, "M M"],
    score_cov_fn: Callable[..., Float[Array, "p M M"]] | None,
) -> Float[Array, "M p"]:
    """Return the orthogonalised Jacobian :math:`\\widetilde D` of Kleibergen 2005.

    Dispatch:

    1. If the caller supplied ``score_cov_fn``, evaluate it at
       ``(model, theta_0)`` to get ``Sigma_jm`` of shape ``(p, M, M)``.
    2. Else, if the measure exposes both ``moment_contributions`` and
       ``jacobian_contributions`` (the standard duck-type for empirical /
       synthetic measures), call them and compute the sample
       cross-covariance via :func:`_sigma_jm_from_contributions`.
    3. Else, fall back to ``Sigma_jm = 0`` so
       :math:`\\widetilde D \\equiv G`. This recovers the strong-identification
       limit and is asymptotically equivalent to D-tilde when
       :math:`m(\\theta_0) = 0`; it is *not* weak-identification-robust.

    Once ``Sigma_jm`` is in hand,
    :math:`\\widetilde D_j = G_j - \\Sigma_{G_j, m} V^{-1} m`, computed
    via the Cholesky of ``V_star``.
    """
    p = G.shape[1]
    M = G.shape[0]

    if score_cov_fn is not None:
        sigma_jm = _to_plain(score_cov_fn(model, theta_0))
        if sigma_jm.shape != (p, M, M):
            raise ValueError(
                f"k_statistic: score_cov_fn returned shape {sigma_jm.shape}; "
                f"expected (p={p}, M={M}, M={M})."
            )
    elif hasattr(measure, "moment_contributions") and hasattr(
        measure, "jacobian_contributions"
    ):
        g = _to_plain(measure.moment_contributions(model, theta_0))
        D = _to_plain(measure.jacobian_contributions(model, theta_0))
        sigma_jm = _sigma_jm_from_contributions(g, D)
    else:
        # Strong-identification fallback: zero correction.
        sigma_jm = jnp.zeros((p, M, M), dtype=G.dtype)

    # z = V_star^{-1} m via Cholesky: solve L y = m, then L' z = y.
    L = cho.cholesky(V_star)
    y = cho.forward_solve(L, m)  # L y = m
    z = cho.back_solve(L, y)  # L' z = y -> z = V^{-1} m

    # correction[:, j] = sigma_jm[j] @ z  ->  einsum "jmk,k->mj"
    correction = jnp.einsum("jmk,k->mj", sigma_jm, z)
    return G - correction


def _kappa_chi2_sf(stat: Float[Array, ""], df: int) -> Float[Array, ""]:
    """Upper-tail chi-squared survival function, traceable under jit.

    Returns :func:`jax.scipy.stats.chi2.sf` when ``df > 0``; returns
    ``nan`` (as a 0-d JAX array) when ``df == 0`` so the just-identified
    overidentification residual has a well-defined sentinel rather than
    accidentally evaluating ``chi2.sf(0, 0) = 1`` or failing.
    """
    if df <= 0:
        return jnp.asarray(jnp.nan)
    return jax.scipy.stats.chi2.sf(stat, df)


def _stats_from_whitened(
    m_tilde: Float[Array, " M"],
    D_tilde_w: Float[Array, "M p"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Return ``(K, S, J)`` from whitened ``m_tilde`` and ``D_tilde_w``.

    Uses thin QR :math:`\\widetilde D_w = Q R` (``mode="reduced"``);
    then :math:`K = \\|Q^\\top \\tilde m\\|^2`.

    The :math:`S` statistic is computed *directly* as
    :math:`\\|(I - QQ^\\top) \\tilde m\\|^2` rather than via
    :math:`J - K` subtraction. The two are algebraically identical but
    the subtraction can yield tiny-negative ``S`` on rank-deficient
    :math:`\\widetilde D_w` (where the QR returns a column at near-zero
    scale); the residual-norm form is non-negative by construction.
    """
    Q, _R = jnp.linalg.qr(D_tilde_w, mode="reduced")
    proj = Q.T @ m_tilde  # (p,)
    K_stat = jnp.sum(proj * proj)
    # Residual after projecting out col(Q). Equivalent to (I - Q Q') m_tilde.
    resid = m_tilde - Q @ proj
    S_stat = jnp.sum(resid * resid)
    J_stat = K_stat + S_stat
    return K_stat, S_stat, J_stat


def _k_statistic_arrays(
    theta_0: ParamsLike,
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    regularization: RegularizationStrategy,
    score_cov_fn: Callable[..., Float[Array, "p M M"]] | None,
    V_override: Float[Array, "M M"] | None,
    L_override: Float[Array, "M M"] | None,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    int,
    int,
    int,
]:
    """Compute (K, S, J, df_K, df_S, df_J) — the array-only kernel.

    Separated from the public :func:`k_statistic` so the kernel is
    jit/vmap-compatible (no Python-level branches on traced values,
    no eager ``float()`` conversions). The outer wrapper adds p-values
    on the JAX side and packages the result.
    """
    # 1. Moment vector, Jacobian, and (regularised) variance at theta_0.
    m = _to_plain(measure.expectation(model, theta_0))
    G = _to_plain(measure.jacobian(model, theta_0))

    if V_override is not None:
        V_star = jnp.asarray(V_override)
    else:
        V = _to_plain(covariance.covariance(model, theta_0, measure))
        V_star, _tau = regularization.apply(V)

    # 2. D-tilde: orthogonalise G against m under the V metric.
    D_tilde = _compute_d_tilde(
        measure=measure,
        model=model,
        theta_0=theta_0,
        m=m,
        G=G,
        V_star=V_star,
        score_cov_fn=score_cov_fn,
    )

    # 3. Whiten via Cholesky (re-using L if supplied).
    if L_override is not None:
        L = jnp.asarray(L_override)
    else:
        L = cho.cholesky(V_star)
    m_tilde = jax.scipy.linalg.solve_triangular(L, m, lower=True)
    D_tilde_w = jax.scipy.linalg.solve_triangular(L, D_tilde, lower=True)

    # 4-5. K / S / J via thin QR.
    K_stat, S_stat, J_stat = _stats_from_whitened(m_tilde, D_tilde_w)

    # Degrees of freedom (static; raised as ValueError below if degenerate).
    M = int(m.shape[0])
    p = int(G.shape[1])
    df_K = p
    df_J = M
    df_S = M - p
    return K_stat, S_stat, J_stat, df_K, df_S, df_J


def k_statistic(
    result_or_theta_null: EstimationResult | ParamsLike,
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    *,
    regularization: RegularizationStrategy | None = None,
    score_cov_fn: Callable[..., Float[Array, "p M M"]] | None = None,
    V: Float[Array, "M M"] | None = None,
    L: Float[Array, "M M"] | None = None,
) -> KStatisticResult:
    """Compute the Kleibergen :math:`K`/:math:`S`/:math:`J` decomposition.

    Evaluates the three chi-squared statistics at a hypothesised
    :math:`\\theta_0`. Under :math:`H_0: \\theta = \\theta_0` the limits
    are :math:`K \\sim \\chi^2_p`, :math:`S \\sim \\chi^2_{M-p}`, and
    :math:`J = K + S \\sim \\chi^2_M`, with :math:`K` and :math:`S`
    asymptotically independent. The :math:`K`-statistic uses the
    orthogonalised Jacobian :math:`\\widetilde D` of Kleibergen (2005)
    so the :math:`\\chi^2_p` limit holds *regardless of identification
    strength* (Kleibergen 2005, Proposition 2).

    Parameters
    ----------
    result_or_theta_null : :class:`EstimationResult` or parameter dataclass
        Either a fitted :class:`EstimationResult` (in which case
        :math:`\\theta_0 = \\hat\\theta`, useful as a diagnostic at the
        point estimate) or a user-supplied parameter dataclass specifying
        the null.
    measure : :class:`emu_gmm.types.Measure`
        Integration operator. For empirical / synthetic measures the
        D-tilde correction is computed from per-observation contributions
        via the measure's ``moment_contributions`` and
        ``jacobian_contributions`` methods. For analytical / population
        measures, supply ``score_cov_fn`` (see below) or accept the
        strong-identification fallback.
    covariance : :class:`emu_gmm.types.CovarianceStrategy`
        Constructor for :math:`V_\\mu(\\theta_0)`.
    model : :data:`emu_gmm.types.StructuralModel`
        Per-observation residual ``psi(x, theta) -> (M,) array``.
    regularization : :class:`emu_gmm.types.RegularizationStrategy`, optional
        Adaptive PD-restoration applied to :math:`V` before factorisation.
        Defaults to :class:`emu_gmm.regularization.DiagonalTikhonov` with
        framework defaults. Ignored when ``V`` is supplied.
    score_cov_fn : callable, keyword-only, optional
        ``score_cov_fn(model, theta_0) -> (p, M, M) array``. Returns the
        cross-covariance tensor :math:`\\Sigma_{G_j, m}` directly. Use
        this for :class:`AnalyticalMeasure` or any other population
        measure where the closed-form covariance is known on paper. When
        omitted, the function tries to read per-observation contributions
        from the measure, falling back to a zero correction (strong-ID
        limit) if neither route applies.
    V : (M, M) jax array, keyword-only, optional
        Pre-computed regularised variance :math:`V^\\star`. When passed,
        bypasses both the ``covariance.covariance`` call and the
        ``regularization.apply`` step; intended for the
        ``k_statistic(result, ...)`` overload where the caller wants the
        decomposition to use the ridge frozen during :func:`estimate`.
    L : (M, M) jax array, keyword-only, optional
        Pre-computed Cholesky factor of ``V``. When passed alongside
        ``V``, skips the Cholesky call inside this routine.

    Returns
    -------
    :class:`KStatisticResult`
        ``K``, ``S``, ``J`` statistics with their degrees of freedom and
        p-values. All scalar quantities are 0-d JAX arrays so the result
        passes through ``jit`` / ``vmap`` boundaries; the dofs are static.

    Raises
    ------
    ValueError
        If the problem is under-identified (``M < p``); under-identified
        problems silently elide the overidentification residual and the
        :math:`\\chi^2_{M-p}` limit is undefined, so the routine refuses
        to compute a degenerate decomposition.

    Notes
    -----
    The implementation is split into a jit-compatible kernel
    (:func:`_k_statistic_arrays`) plus an outer wrapper that adds the
    chi-squared p-values via :func:`jax.scipy.stats.chi2.sf` (also
    traceable). Callers wanting maximum speed can pre-jit the entire
    function or vmap it over a batch of nulls.
    """
    theta_0 = _resolve_theta_null(result_or_theta_null)
    if regularization is None:
        regularization = DiagonalTikhonov()

    # Quick under-identified guard — uses the SHAPE only (static), so
    # this branch is fine inside jit as long as the user's measure
    # returns a stable shape.
    m_probe = _to_plain(measure.expectation(model, theta_0))
    G_probe = _to_plain(measure.jacobian(model, theta_0))
    M = int(m_probe.shape[0])
    p = int(G_probe.shape[1])
    if M < p:
        raise ValueError(
            f"k_statistic: under-identified problem (M={M} moments < p={p} "
            f"parameters). The chi^2_{{M-p}} limit for S is undefined when "
            f"M < p; refuse rather than silently returning a degenerate "
            f"decomposition."
        )

    K_stat, S_stat, J_stat, df_K, df_S, df_J = _k_statistic_arrays(
        theta_0=theta_0,
        measure=measure,
        covariance=covariance,
        model=model,
        regularization=regularization,
        score_cov_fn=score_cov_fn,
        V_override=V,
        L_override=L,
    )

    # p-values via jax.scipy.stats.chi2.sf — traceable under jit / vmap.
    p_K = _kappa_chi2_sf(K_stat, df_K)
    p_S = _kappa_chi2_sf(S_stat, df_S)
    p_J = _kappa_chi2_sf(J_stat, df_J)

    return KStatisticResult(
        K=K_stat,
        S=S_stat,
        J=J_stat,
        p_K=p_K,
        p_S=p_S,
        p_J=p_J,
        df_K=df_K,
        df_S=df_S,
        df_J=df_J,
    )


__all__ = ["k_statistic", "KStatisticResult"]
