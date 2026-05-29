"""Diagnostics builders and logging hooks for :func:`emu_gmm.estimate`.

The :class:`emu_gmm.types.Diagnostics` dataclass is constructed once at
the end of an estimation. This module provides:

- :func:`build_diagnostics`: assemble a :class:`Diagnostics` from the
  raw arrays and scalars computed by the estimator pipeline, wrapping
  the per-moment fields in labelled :class:`haliax.NamedArray` instances.
- :func:`compute_cond_info`: build the Hessian condition trio
  (``raw`` / ``data_only`` / ``exclude_gauge``) at theta_hat from
  ``G`` and ``V*``. See CLAUDE.md commitment 5 and issue #10.
- :func:`build_optimizer_health`: assemble the optimiser-health summary
  dict from an :class:`OptimizerInfo` and a final gradient norm.
- :func:`log_to_stdout`: a simple console logger usable as a per-step
  hook during optimisation; prints :math:`\\tau`, :math:`\\kappa(V^\\star)`,
  and the current objective.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats
from jaxtyping import Array, Float

from emu_gmm._internal import labels as labels_mod
from emu_gmm.types import Diagnostics, OptimizerInfo


def build_diagnostics(
    *,
    tau_realised: Any,
    kappa_V: Any,
    binding_ridge: Any,
    cholesky_pivot_min: Any,
    final_objective: Any,
    final_gradient_norm: Any,
    N_j_array: Float[Array, " M"],
    moment_residual_array: Float[Array, " M"],
    moments_axis: ha.Axis,
    optimizer_info: OptimizerInfo,
    cond_info: dict[str, float] | None = None,
    optimizer_health: dict[str, Any] | None = None,
    final_objective_data: Any | None = None,
    final_objective_full: Any | None = None,
) -> Diagnostics:
    """Assemble a :class:`Diagnostics` from raw estimator-pipeline values.

    The labelled per-moment fields (``N_j``, ``moment_residual``) are
    wrapped in :class:`haliax.NamedArray` instances on the supplied
    ``moments_axis``. Scalar fields are converted to 0-d JAX arrays so
    the result is jit / vmap compatible; users cast to Python floats at
    the eager boundary (e.g. via :meth:`EstimationResult.to_pandas`).

    Parameters
    ----------
    tau_realised, kappa_V, binding_ridge, cholesky_pivot_min,
    final_objective, final_gradient_norm
        Scalar diagnostics produced during the estimation pipeline.
        May be Python scalars or 0-d JAX arrays; both are normalised to
        :class:`jax.Array`. ``final_objective`` is the legacy alias for
        ``final_objective_data``; see the split below.
    final_objective_data, final_objective_full : optional
        Split of the data-only criterion (``J_stat`` /
        :math:`\\|y\\|^2`) and the full criterion the optimiser saw
        (data plus any in-objective :class:`PenaltyStrategy`). When
        omitted both default to ``final_objective`` so the unpenalised
        path keeps a single value across the three fields.
    N_j_array : (M,) array
        Effective sample size per moment coordinate. For synthetic
        measures this is constant (``n_sim``); for empirical measures
        with missingness it is :math:`\\sum_i d_{ij} w_i`.
    moment_residual_array : (M,) array
        :math:`\\bar m_X(\\hat\\theta)`, the moment vector at the estimate.
    moments_axis : :class:`haliax.Axis`
        Axis for the labelled per-moment outputs.
    optimizer_info : :class:`OptimizerInfo`
        Backend-specific solver info.
    cond_info : dict, optional
        Hessian condition trio at theta_hat. See
        :func:`compute_cond_info`. Defaults to an empty dict if the
        caller has not computed it (e.g. unit tests).
    optimizer_health : dict, optional
        Lightweight optimiser-health summary. See
        :func:`build_optimizer_health`. Defaults to an empty dict.

    Returns
    -------
    :class:`Diagnostics`
    """
    if final_objective_data is None:
        final_objective_data = final_objective
    if final_objective_full is None:
        final_objective_full = final_objective
    return Diagnostics(
        tau_realised=jnp.asarray(tau_realised),
        kappa_V=jnp.asarray(kappa_V),
        binding_ridge=jnp.asarray(binding_ridge),
        cholesky_pivot_min=jnp.asarray(cholesky_pivot_min),
        final_objective_data=jnp.asarray(final_objective_data),
        final_objective_full=jnp.asarray(final_objective_full),
        final_objective=jnp.asarray(final_objective),
        final_gradient_norm=jnp.asarray(final_gradient_norm),
        N_j=labels_mod.label_vector(jnp.asarray(N_j_array), moments_axis),
        moment_residual=labels_mod.label_vector(
            jnp.asarray(moment_residual_array), moments_axis
        ),
        optimizer_info=optimizer_info,
        cond_info=dict(cond_info) if cond_info is not None else {},
        optimizer_health=(
            dict(optimizer_health) if optimizer_health is not None else {}
        ),
    )


def regularization_adjusted_pvalue(
    J_stat: Float[Array, ""],
    V: Float[Array, "M M"],
    V_star: Float[Array, "M M"],
    G: Float[Array, "M K"],
) -> Float[Array, ""]:
    """Regularisation-adjusted J-test p-value (weighted-chi^2 limit).

    Per :file:`docs/mcar-asymptotics.org` Theorem 6, when ``V`` is
    replaced by :math:`V^\\star = V + \\tau \\operatorname{diag}(V)`
    with :math:`\\tau` held fixed across the sample path, the J-statistic
    limit is a generalised chi-squared --- a weighted sum of independent
    :math:`\\chi^2_1` variates whose weights are the eigenvalues of
    :math:`(V^\\star)^{-1} V` projected onto the orthogonal complement
    of the column space of :math:`G(\\theta_0)`. When :math:`\\tau = 0`
    the weights coincide at 1, recovering :math:`\\chi^2_{M-K}`.

    For v1, the weighted-chi^2 survival function is approximated via the
    Welch--Satterthwaite scheme [cite:@satterthwaite1946]: match the
    first two moments of :math:`\\sum_i w_i Z_i^2` (mean :math:`\\sum w_i`,
    variance :math:`2 \\sum w_i^2`) to a scaled chi-squared
    :math:`c \\cdot \\chi^2_v` with :math:`c = \\sum w_i^2 / \\sum w_i`
    and :math:`v = (\\sum w_i)^2 / \\sum w_i^2`. This is the standard
    approximation in the linear-model literature; Davies' / Imhof's
    exact CDF would be more accurate but requires non-trivial code that
    falls outside v1's scope.

    Concretely, working in the whitened coordinates
    :math:`\\tilde y = L_\\star^{-1} m` with
    :math:`L_\\star L_\\star^\\top = V^\\star`,

    - :math:`\\tilde G = L_\\star^{-1} G` is the whitened Jacobian;
    - :math:`\\tilde V = L_\\star^{-1} V L_\\star^{-\\top}` is the
      whitened variance under the unregularised limit;
    - :math:`P = I - \\tilde G (\\tilde G^\\top \\tilde G)^{-1}
      \\tilde G^\\top` projects onto the orthogonal complement of the
      column space of :math:`\\tilde G`;
    - the weights are the (M-K) non-zero eigenvalues of
      :math:`P \\tilde V P`.

    Parameters
    ----------
    J_stat : 0-d scalar array
        Realised J-statistic :math:`\\|y\\|^2`.
    V : (M, M) array
        Unregularised covariance estimate at :math:`\\hat\\theta`.
    V_star : (M, M) array
        Regularised covariance :math:`V + \\tau \\operatorname{diag}(V)`
        used by the weighting strategy.
    G : (M, K) array
        Moment Jacobian :math:`E_\\mu[\\nabla_\\theta \\psi]` at
        :math:`\\hat\\theta`.

    Returns
    -------
    p : 0-d scalar array
        Approximate survival function value
        :math:`P(\\sum_i w_i Z_i^2 \\geq J_{\\mathrm{stat}})`.

    Notes
    -----
    Traceable under ``jit`` and ``vmap``: the eigendecomposition is
    via :func:`jax.numpy.linalg.eigh` and the chi-squared survival
    function via :func:`jax.scipy.stats.chi2.sf`. Both accept tracers.
    """
    M, K = G.shape
    # Cholesky-whiten V into the L_star coordinates.
    L_star = jnp.linalg.cholesky(V_star)
    # ~G = L_star^{-1} G
    G_tilde = jax.scipy.linalg.solve_triangular(L_star, G, lower=True)
    # ~V = L_star^{-1} V L_star^{-T}: this is what (V_star)^{-1} V's
    # eigenvalues live as in whitened space.
    Vinv_V_root_left = jax.scipy.linalg.solve_triangular(L_star, V, lower=True)
    V_tilde = jax.scipy.linalg.solve_triangular(
        L_star, Vinv_V_root_left.T, lower=True
    ).T
    # Symmetrise (cancel out roundoff).
    V_tilde = 0.5 * (V_tilde + V_tilde.T)

    # Orthogonal projector onto the orthogonal complement of col(~G).
    # P = I_M - ~G (~G' ~G)^{-1} ~G'.
    GtG = G_tilde.T @ G_tilde
    GtG_inv = jnp.linalg.inv(GtG)
    P = jnp.eye(M) - G_tilde @ GtG_inv @ G_tilde.T

    # Eigenvalues of P ~V P. Of the M eigenvalues, K are exactly zero
    # (projector kills the col(~G) directions); the remaining M-K are
    # the weights for the weighted-chi^2 limit. We keep all of them and
    # sum: zeros contribute nothing.
    PVP = P @ V_tilde @ P
    PVP = 0.5 * (PVP + PVP.T)
    weights = jnp.linalg.eigvalsh(PVP)
    # Clamp small negative eigenvalues (numerical noise) to zero so the
    # Satterthwaite ratio is well defined.
    weights = jnp.where(weights > 0.0, weights, 0.0)

    # Welch-Satterthwaite: ~ c * chi^2_v, with c = sum w^2 / sum w,
    # v = (sum w)^2 / sum w^2. Guard against the degenerate
    # all-weights-zero case (would only happen with M=K, which the
    # dimension guard already prevents from reaching here in practice).
    s1 = jnp.sum(weights)
    s2 = jnp.sum(weights**2)
    safe = s1 > 0.0
    c = jnp.where(safe, s2 / jnp.where(safe, s1, 1.0), 1.0)
    v = jnp.where(safe, (s1**2) / jnp.where(safe, s2, 1.0), float(M - K))

    return jax.scipy.stats.chi2.sf(J_stat / c, v)


def compute_cond_info(
    G: Float[Array, "M K"],
    V_star: Float[Array, "M M"],
    penalty_hessian: Float[Array, "K K"] | None = None,
) -> dict[str, float]:
    """Condition number of the information matrix :math:`G' \\Lambda G`.

    Per CLAUDE.md commitment 5 the v1 information matrix is constructed
    as :math:`G' \\Lambda G` with :math:`\\Lambda = (V^\\star)^{-1}`,
    realised via the Cholesky factor of :math:`V^\\star`. This routine
    reports :math:`\\kappa(G' \\Lambda G)` --- a direct identifier
    proxy that catches near-rank-deficient :math:`G` even when
    :math:`V^\\star` is well conditioned.

    The return value is a dict with three keys to mirror ManifoldGMM's
    ``compute_hessian_cond(data_only, exclude_gauge)`` (see issue #10):

    - ``'raw'``: :math:`\\kappa(G' \\Lambda G + \\tfrac{1}{2} H_p)`
      --- the condition number of the *full* information matrix that
      governs the curvature of the criterion the optimiser saw. When no
      penalty is supplied :math:`H_p \\equiv 0` so this equals the
      data-only number.
    - ``'data_only'``: :math:`\\kappa(G' \\Lambda G)`, the data-only
      information matrix with any in-objective :class:`PenaltyStrategy`
      contribution excluded. This is the asymptotic-inference-relevant
      quantity (delta-method variance is built from the data Hessian
      alone) and the correct identifier proxy when the penalty is a
      stabiliser rather than a prior.
    - ``'exclude_gauge'``: alias to ``'raw'`` for v1. The v2 manifold
      epic will reinterpret this as the condition of the information
      matrix projected onto the orthogonal complement of the gauge
      nullspace; for the flat-parameter v1 the gauge group is trivial
      and the two coincide.

    Parameters
    ----------
    G : (M, K) array
        Moment-Jacobian at :math:`\\hat\\theta`, ``E_mu[grad_theta psi]``.
    V_star : (M, M) array
        Regularised moment-variance matrix at :math:`\\hat\\theta`.
        Assumed symmetric PD.
    penalty_hessian : (K, K) array, optional
        :math:`H_p = \\nabla^2_\\theta p(\\hat\\theta)`, the Hessian of
        the in-objective penalty at :math:`\\hat\\theta`. When supplied,
        ``'raw'`` adds :math:`\\tfrac{1}{2} H_p` to the data information
        matrix; ``'data_only'`` ignores it. The factor of one-half comes
        from the NLLS embedding: the optimiser minimises
        :math:`\\tfrac{1}{2}\\|r\\|^2 = \\tfrac{1}{2}(\\|y\\|^2 + p)`,
        whose Hessian at the optimum (ignoring higher-order residual
        curvature, the standard Gauss-Newton view) is
        :math:`G' \\Lambda G + \\tfrac{1}{2} H_p`.

    Returns
    -------
    dict[str, float]
        ``{'raw': ..., 'data_only': ..., 'exclude_gauge': ...}``.
    """
    G_arr = jnp.asarray(G)
    V_arr = jnp.asarray(V_star)
    # Information matrix via Cholesky of V*. Match estimator.py's
    # construction so the reported cond is the cond of the same matrix
    # used for Sigma_theta.
    L = jnp.linalg.cholesky(V_arr)
    Z = jax.scipy.linalg.solve_triangular(L, G_arr, lower=True)
    info_matrix_data = Z.T @ Z  # G' Lambda G, data-only.

    # Data-only condition: never includes the penalty.
    data_only_arr = jnp.linalg.cond(info_matrix_data)
    try:
        data_only: Any = float(data_only_arr)
    except (TypeError, ValueError):
        data_only = data_only_arr

    # Raw / full condition: G' Lambda G + (1/2) H_p when a penalty is
    # supplied, else just G' Lambda G. cond is scale-invariant so the
    # (1/2) is only "correct" up to a global rescale, but we keep it
    # explicit because the same matrix is the right one to use when
    # mixing penalty and data contributions in any downstream sum.
    if penalty_hessian is None:
        info_matrix_full = info_matrix_data
    else:
        H_p = jnp.asarray(penalty_hessian)
        info_matrix_full = info_matrix_data + 0.5 * H_p
    raw_arr = jnp.linalg.cond(info_matrix_full)
    try:
        raw: Any = float(raw_arr)
    except (TypeError, ValueError):
        raw = raw_arr

    # ``exclude_gauge`` aliases to ``raw`` for v1: the flat-parameter
    # path has no manifold gauge nullspace. v2 manifold epic will
    # distinguish them.
    return {
        "raw": raw,
        "data_only": data_only,
        "exclude_gauge": raw,
    }


def build_optimizer_health(
    optimizer_info: OptimizerInfo,
    final_gradient_norm: float,
    step_norm: float | None = None,
    accepted_step_count: int | None = None,
) -> dict[str, Any]:
    """Build the optimiser-health summary dict (issue #10).

    Parameters
    ----------
    optimizer_info : :class:`OptimizerInfo`
        Backend-specific solver info. ``optimizer_info.steps`` becomes
        the ``'iters'`` field.
    final_gradient_norm : float
        Norm of :math:`\\nabla_\\theta \\tfrac{1}{2}\\|r\\|^2` at the
        optimum, where ``r`` is the *full* residual vector the
        optimiser saw. When ``estimate(..., penalty=...)`` supplies an
        in-objective penalty this includes the penalty contribution
        from the appended :math:`\\sqrt{p(\\theta)}` row; without a
        penalty it is the unpenalised data-only gradient norm
        :math:`\\|\\nabla_\\theta \\tfrac{1}{2}\\|y\\|^2\\|`.
    step_norm : float, optional
        Norm of the last accepted step. Defaults to ``None`` because
        neither :class:`optimistix.LevenbergMarquardt` nor
        :func:`scipy.optimize.least_squares` expose this directly in
        their result objects.
    accepted_step_count : int, optional
        Number of accepted (vs rejected) LM steps. Defaults to ``None``
        for the same reason as ``step_norm``.

    Returns
    -------
    dict[str, Any]
        ``{'iters': int, 'grad_norm': float, 'step_norm': float | None,
        'accepted_step_count': int | None}``.
    """
    iters_val = optimizer_info.steps
    try:
        iters: Any = int(iters_val)
    except (TypeError, ValueError):
        # Traced under jit: leave the JAX scalar in place.
        iters = iters_val
    try:
        grad_norm: Any = float(final_gradient_norm)
    except (TypeError, ValueError):
        # Traced under jit: leave the JAX scalar in place.
        grad_norm = final_gradient_norm
    return {
        "iters": iters,
        "grad_norm": grad_norm,
        "step_norm": step_norm,
        "accepted_step_count": accepted_step_count,
    }


def log_to_stdout(prefix: str = "[emu-gmm]") -> Any:
    """Return a callable that prints per-step diagnostics to stdout.

    The returned callable accepts keyword arguments ``step``, ``tau``,
    ``kappa``, ``objective`` and emits a single-line summary. Intended
    as a lightweight hook for interactive debugging; production logging
    should use a structured logger.

    Parameters
    ----------
    prefix : str
        String prepended to every log line.

    Returns
    -------
    callable
        ``logger(step, tau, kappa, objective) -> None``.
    """

    def _log(
        step: int,
        tau: float,
        kappa: float,
        objective: float,
    ) -> None:
        print(
            f"{prefix} step={step:>4d}  "
            f"tau={tau:.3e}  kappa={kappa:.3e}  Q={objective:.6e}"
        )

    return _log


__all__ = [
    "build_diagnostics",
    "build_optimizer_health",
    "compute_cond_info",
    "log_to_stdout",
    "regularization_adjusted_pvalue",
]
