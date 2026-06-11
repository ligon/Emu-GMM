"""Diagnostics builders and logging hooks for :func:`emu_gmm.estimate`.

The :class:`emu_gmm.types.Diagnostics` dataclass is constructed once at
the end of an estimation. This module provides:

- :func:`build_diagnostics`: assemble a :class:`Diagnostics` from the
  raw arrays and scalars computed by the estimator pipeline, wrapping
  the per-moment fields in labelled :class:`haliax.NamedArray` instances.
- :func:`compute_cond_info`: build the Hessian condition trio
  (``raw`` / ``data_only`` / ``exclude_gauge``) at theta_hat from
  ``G`` and ``V*``; ``exclude_gauge`` is the gauge-aware quotient
  condition number (drop ``gauge_nullspace_dim`` smallest eigenvalues
  by count; issue #20). See CLAUDE.md commitment 5 and issue #10.
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
from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule
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
    gauge_nullspace_dim: int = 0,
    sigma_meat_indefinite: Any = False,
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
        gauge_nullspace_dim=int(gauge_nullspace_dim),
        sigma_meat_indefinite=sigma_meat_indefinite,
    )


def regularization_adjusted_pvalue(
    J_stat: Float[Array, ""],
    V: Float[Array, "M M"],
    V_star: Float[Array, "M M"],
    G: Float[Array, "M K"],
    gauge_nullspace_dim: int = 0,
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

    The weighted-chi^2 survival function is approximated via the
    Welch--Satterthwaite scheme [cite:@satterthwaite1946]: match the
    first two moments of :math:`\\sum_i w_i Z_i^2` (mean :math:`\\sum w_i`,
    variance :math:`2 \\sum w_i^2`) to a scaled chi-squared
    :math:`c \\cdot \\chi^2_v` with :math:`c = \\sum w_i^2 / \\sum w_i`
    and :math:`v = (\\sum w_i)^2 / \\sum w_i^2`. Davies' / Imhof's exact
    CDF was considered and CLOSED AS NOT NEEDED (owner decision
    2026-06-11, on the #130 validation evidence): in the engineered
    fixed-tau binding regime --- the case this function exists for ---
    the W-S approximation brings the J p-value from a failing
    uniformity KS of 0.211 (nominal) to a passing 0.085 over the full
    law; the residual miscalibrated cases trace to a mis-centered
    covariance (#145), which no reference-distribution exactness can
    repair. See docs/validation/ladder-mc-2026-06-10.org (study 5).

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
        :math:`\\hat\\theta`. For manifold parameters this is the
        AMBIENT Jacobian; gauge invariance of the moment makes the
        gauge directions an exact nullspace of ``G``, accounted for via
        ``gauge_nullspace_dim``.
    gauge_nullspace_dim : static int, optional
        Exact dimension of ``G``'s gauge nullspace
        (``manifold_spec.total_gauge_dim``); ``0`` for v1 / scalar
        parameters. #137: the projector formerly used a plain
        ``inv(G~'G~)``, which is SINGULAR for gauge-bearing manifolds
        (PSDFixedRank) and returned silently wrong, plausible-looking
        p-values. The Gram inverse now drops exactly this many smallest
        eigenvalues BY COUNT (:func:`pinv_eigvalrule`, the same rule the
        Sigma_theta bread uses); ``P = I - G~ (G~'G~)^+ G~'`` is the
        correct orthogonal projector onto ``col(G~)``'s complement at
        any rank.

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

    *Whitening assumption (#137):* the derivation hardcodes the
    EFFICIENT whitening ``L_star = chol(V_star)`` --- i.e. it adjusts
    the J of a CU / Iterated solve. Under ``Identity`` / ``Fixed``
    weighting the realised ``J = ||y||^2`` lives in a different metric
    and has no chi-squared limit to adjust in the first place (the same
    caveat as the nominal ``J_pvalue``); treat both p-values as
    unavailable there rather than approximate.
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
    # #137: a plain inv() here is singular for gauge-bearing manifolds
    # (the gauge directions are an exact nullspace of G). Drop exactly
    # ``gauge_nullspace_dim`` smallest eigenvalues BY COUNT -- the same
    # rule as the Sigma_theta bread three call sites away; for
    # gauge_nullspace_dim == 0 this is bitwise inv() (the v1 path).
    # P = I - G~ (G~'G~)^+ G~' is the orthogonal projector onto
    # col(G~)'s complement at any rank (Moore-Penrose property).
    GtG_inv = pinv_eigvalrule(GtG, drop_smallest=int(gauge_nullspace_dim))
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
    gauge_nullspace_dim: int = 0,
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
    - ``'exclude_gauge'``: the *quotient* condition number (issue #20,
      surfaced by the K-Aggregators consumer). For a gauge-bearing
      manifold parameter (``PSDFixedRank(n, K)``:
      ``gauge_dim = K(K-1)/2``) the information matrix has exactly
      ``gauge_nullspace_dim`` spectrally-zero directions BY
      CONSTRUCTION, so the full-spectrum ``'raw'`` is meaningless for
      ``K >= 2``. This entry drops the ``gauge_nullspace_dim`` smallest
      eigenvalues of the (symmetrised) full information matrix **by
      count** -- the same drop-by-count rule as
      :func:`~emu_gmm._internal.pinv_eigvalrule.pinv_eigvalrule` and
      the #137/#41 projectors -- and reports ``max/min`` over the
      remaining spectrum. Any *additional* near-zero eigenvalues beyond
      the dropped count then blow this number up too: that is genuine
      structural rank-deficiency, exactly the signal the consumer's
      identification analysis tests for. When
      ``gauge_nullspace_dim == 0`` (every v1 / all-Euclidean tree) this
      is the bitwise alias of ``'raw'``, as before.

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
    gauge_nullspace_dim : int, optional
        A **static Python int** (NOT a JAX array / tracer): the number
        of exact gauge-nullspace directions of the information matrix,
        i.e. the manifold spec's ``total_gauge_dim``
        (``K(K-1)/2`` per ``PSDFixedRank(n, K)`` leaf). Must satisfy
        ``0 <= gauge_nullspace_dim < K``. The estimator passes
        ``manifold_spec.total_gauge_dim``; the default ``0`` keeps the
        v1 flat-parameter behaviour bitwise (``'exclude_gauge'``
        aliases ``'raw'``).

    Returns
    -------
    dict[str, float]
        ``{'raw': ..., 'data_only': ..., 'exclude_gauge': ...}``.
    """
    # Static-int guards, mirroring pinv_eigvalrule (the count is a trace-
    # time constant: the eigenvalue slice below must have static shape).
    if not isinstance(gauge_nullspace_dim, int) or isinstance(
        gauge_nullspace_dim, bool
    ):
        raise TypeError(
            "compute_cond_info: gauge_nullspace_dim must be a static "
            f"Python int (vmap/jit-safe), got "
            f"{type(gauge_nullspace_dim).__name__}. Pass "
            "manifold_spec.total_gauge_dim directly."
        )
    if gauge_nullspace_dim < 0:
        raise ValueError(
            "compute_cond_info: gauge_nullspace_dim must be >= 0, got "
            f"{gauge_nullspace_dim}"
        )
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

    # ``exclude_gauge`` (#20): the quotient condition number. With no
    # gauge nullspace (every v1 / all-Euclidean tree) it stays the
    # bitwise alias of ``raw``; with ``gauge_nullspace_dim > 0`` it is
    # the condition over the spectrum EXCLUDING the gauge_dim smallest
    # eigenvalues BY COUNT (the pinv_eigvalrule / #137 / #41 rule: the
    # gauge zeros are a property of the quotient, known exactly at trace
    # time, never a magnitude threshold).
    if gauge_nullspace_dim == 0:
        exclude_gauge: Any = raw
    else:
        K_dim = int(info_matrix_full.shape[-1])
        if gauge_nullspace_dim >= K_dim:
            raise ValueError(
                "compute_cond_info: gauge_nullspace_dim must be < K = "
                f"{K_dim} (the size of the information matrix), got "
                f"{gauge_nullspace_dim}. Dropping the whole spectrum "
                "would leave an empty identified subspace."
            )
        # eigvalsh on the symmetrised matrix (neutralise the rounding
        # asymmetry of Z'Z); ascending order, so the gauge zeros are the
        # FIRST ``gauge_nullspace_dim`` entries and the static slice
        # keeps the identified block.
        info_sym = 0.5 * (info_matrix_full + info_matrix_full.T)
        eigs = jnp.linalg.eigvalsh(info_sym)
        eigs_keep = eigs[gauge_nullspace_dim:]
        w_min = eigs_keep[0]
        w_max = eigs_keep[-1]
        # Guard on the SIGNED minimum (commitment 3's convention): a
        # non-positive smallest kept eigenvalue means the identified
        # block is numerically singular / indefinite -- the quotient
        # condition number is +inf, a visible event (the #140
        # NaN-is-an-event convention: never absorb it into a finite
        # number). This matches jnp.linalg.cond's inf-for-singular
        # convention used by ``raw`` / ``data_only``.
        quotient_arr = jnp.where(
            w_min > 0.0,
            w_max / jnp.where(w_min > 0.0, w_min, 1.0),
            jnp.inf,
        )
        try:
            exclude_gauge = float(quotient_arr)
        except (TypeError, ValueError):
            exclude_gauge = quotient_arr

    return {
        "raw": raw,
        "data_only": data_only,
        "exclude_gauge": exclude_gauge,
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
