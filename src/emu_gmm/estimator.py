"""The :func:`estimate` entry point.

Ties Phases 1-4 together into one stateless function:

1. Build a :class:`LabelContext` from the user's parameter dataclass and
   --- if the structural model returns a labelled vector --- from the
   model's output. The context lives as a static closure variable; it
   does not enter the traced inputs of any jit'd function.
2. Flatten ``theta_init`` to a 1-D JAX array via
   :func:`emu_gmm._internal.params.flatten_params`.
3. Construct the residual function ``y(theta_flat)`` by routing through
   ``measure.expectation`` -> ``covariance.covariance`` ->
   ``regularization.apply`` -> ``weighting.whitening_residual``. Hand it
   to the :class:`Optimizer`.
4. At the optimum, compute :math:`G(\\hat\\theta)`,
   :math:`\\Sigma_{\\hat\\theta}`, the J statistic, and a labelled
   :class:`Diagnostics` record. Wrap everything into an
   :class:`EstimationResult` and return.

See ``docs/api-sketch.org`` Section 4 and ``docs/implementation-plan.org``
Section 7 for the architectural spec.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats
from jaxtyping import Array, Float

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import cholesky as cho
from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal import params as params_mod
from emu_gmm.diagnostics import (
    build_diagnostics,
    build_optimizer_health,
    compute_cond_info,
    regularization_adjusted_pvalue,
)
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    Emu_GMM_DimensionError,
    EstimationResult,
    Measure,
    Optimizer,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import ContinuouslyUpdated


def _sample_observation(measure: Measure, theta_init: ParamsLike) -> Any | None:
    """Try to extract one sample observation from a measure for label probing.

    Returns ``None`` if the measure type doesn't expose a sample-extraction
    interface (in which case moment labels fall through to the kwarg or
    positional fallback).
    """
    if hasattr(measure, "_draws"):
        # SyntheticMeasure
        draws = measure._draws(theta_init)
        return draws[0]
    if hasattr(measure, "x"):
        # EmpiricalMeasure (future)
        return measure.x[0]
    return None


def _effective_n_per_moment(
    measure: Measure, theta_hat: ParamsLike, m: int
) -> Float[Array, " M"]:
    """Return effective sample size per moment coordinate.

    For SyntheticMeasure: a constant ``n_sim`` broadcast to length ``m``.
    For EmpiricalMeasure (future): ``sum_i d_ij * w_i`` per coordinate.
    Otherwise: ones (analytical measures have no sample size).
    """
    if hasattr(measure, "n_sim"):
        return jnp.full((m,), float(measure.n_sim))
    if hasattr(measure, "mask") and hasattr(measure, "weights"):
        return jnp.sum(measure.mask * measure.weights[:, None], axis=0)
    return jnp.ones(m)


def _binding_ridge(
    regularization: RegularizationStrategy, tau: Float[Array, ""]
) -> Float[Array, ""]:
    """Whether the regularisation ridge is "binding" relative to its threshold.

    Returns a traced 0-d boolean JAX array so the result can flow
    through ``jit`` / ``vmap`` without forcing a concrete-Python boundary.
    Only :class:`DiagonalTikhonov` exposes a ``tau_threshold``; other
    regularisers default to ``False``.
    """
    threshold = getattr(regularization, "tau_threshold", None)
    if threshold is None:
        return jnp.asarray(False)
    return jnp.asarray(tau) > jnp.asarray(threshold)


def estimate(
    model: StructuralModel,
    measure: Measure,
    *,
    covariance: CovarianceStrategy,
    weighting: WeightingStrategy | None = None,
    regularization: RegularizationStrategy | None = None,
    optimizer: Optimizer | None = None,
    theta_init: ParamsLike,
    moment_names: tuple[str, ...] | None = None,
) -> EstimationResult:
    """Estimate :math:`\\hat\\theta` by minimising
    :math:`Q_\\mu(\\theta) = \\| L_\\mu(\\theta)^{-1}\\, \\mathbb{E}_\\mu[\\psi(\\cdot,\\theta)] \\|^2`.

    Parameters
    ----------
    model : :data:`StructuralModel`
        Per-observation residual function ``psi(x, theta) -> (M,) array``.
        May return a :class:`haliax.NamedArray` to communicate moment labels.
    measure : :class:`Measure`
        Integration operator: ``SyntheticMeasure``, ``EmpiricalMeasure``,
        or ``AnalyticalMeasure``.
    covariance : :class:`CovarianceStrategy`
        Constructor for :math:`V_\\mu(\\theta)`.
    weighting : :class:`WeightingStrategy`, optional
        Defaults to :class:`emu_gmm.weighting.ContinuouslyUpdated`.
    regularization : :class:`RegularizationStrategy`, optional
        Defaults to :class:`emu_gmm.regularization.DiagonalTikhonov` with
        the framework defaults (``kappa_target = 1e6``).
    optimizer : :class:`Optimizer`, optional
        Defaults to :func:`emu_gmm.optimizer.optimistix_lm` with default
        tolerances.
    theta_init
        Starting parameters as a ``@jdc.pytree_dataclass`` with scalar
        fields. The user's dataclass type is preserved in the returned
        ``EstimationResult.theta_hat``.
    moment_names : tuple of str, optional
        Override for moment labels. Precedence: model-return NamedArray
        > this kwarg > positional ``("m_0", "m_1", ...)``.

    Returns
    -------
    :class:`EstimationResult`
    """
    # Defaults
    if weighting is None:
        weighting = ContinuouslyUpdated()
    if regularization is None:
        regularization = DiagonalTikhonov()
    if optimizer is None:
        optimizer = optimistix_lm()

    # Probe M by evaluating the expectation once at theta_init.
    m_probe = measure.expectation(model, theta_init)
    m_probe_arr = jnp.asarray(m_probe)
    if m_probe_arr.ndim == 0 or m_probe_arr.shape[0] == 0:
        raise Emu_GMM_DimensionError(
            "estimate() requires M >= 1 moments; the supplied measure "
            "returned an empty moment vector at theta_init. For a "
            "degenerate zero-moment problem there is nothing to estimate; "
            "for a zero-parameter, M-moment over-identifying-restrictions "
            "J-test use the helper requested in issue #29 "
            "(``emu_gmm.j_test``), once it lands."
        )
    M = int(m_probe_arr.shape[0])

    # Probe K from the parameter dataclass before touching ``flatten_params``
    # (which fails with an opaque ``jnp.stack of empty list`` error when
    # the dataclass has zero fields).
    K_probe = len(tuple(params_mod.param_names(theta_init)))
    if K_probe == 0:
        raise Emu_GMM_DimensionError(
            "estimate() requires K >= 1 parameters; the supplied "
            "theta_init has no fields. For a zero-parameter, M-moment "
            "J-test of over-identifying restrictions use the helper "
            "requested in issue #29 (``emu_gmm.j_test``), once it lands."
        )
    if K_probe > M:
        raise Emu_GMM_DimensionError(
            f"estimate() requires M >= K (over-/just-identified); got "
            f"M={M} moments and K={K_probe} parameters (under-identified). "
            "An under-identified problem has rank-deficient G' V^{-1} G "
            "and yields inf/nan Sigma_theta. Reduce the parameter count, "
            "add moments, or impose an identifying restriction."
        )

    # Probe for labelled output by calling model on one sample observation,
    # if the measure exposes one.
    x_sample = _sample_observation(measure, theta_init)
    model_return = model(x_sample, theta_init) if x_sample is not None else None

    # Resolve all labels.
    param_names = tuple(params_mod.param_names(theta_init))
    moment_names_resolved = labels_mod.resolve_moment_names(
        model_return=model_return,
        kwarg_names=moment_names,
        m=M,
    )
    variable_names: tuple[str, ...] = ()
    if x_sample is not None:
        x_arr = jnp.asarray(x_sample)
        if x_arr.ndim >= 1:
            variable_names = tuple(f"v_{i}" for i in range(int(x_arr.shape[0])))
    label_context = labels_mod.LabelContext(
        param_names=param_names,
        moment_names=moment_names_resolved,
        variable_names=variable_names,
    )

    # Flatten parameters.
    theta_init_flat, treedef = params_mod.flatten_params(theta_init)
    K = int(theta_init_flat.shape[0])

    # Anchor-once-then-freeze tau policy (design.org §5; CLAUDE.md
    # commitment 3). Compute V at theta_init, run the adaptive tau
    # search once, freeze the resulting tau; the residual closure then
    # applies the *same* tau deterministically at every theta. This
    # preserves smoothness of the residual surface (a hard requirement
    # for LM and other Jacobian-based optimisers) and the delta-method
    # argument that yields asymptotic normality.
    V0 = covariance.covariance(model, theta_init, measure)
    _V0_star, tau_anchor = regularization.apply(V0)
    # Cast tau_anchor to a 0-d JAX array we can close over and reuse.
    tau_anchor = jnp.asarray(tau_anchor)

    def _apply_anchored(V: Float[Array, "M M"]) -> Float[Array, "M M"]:
        """Apply the ridge at the anchored ``tau_anchor`` deterministically.

        Prefer ``regularization.apply_fixed_tau`` when the strategy
        exposes it; fall back to the algebraic form for arbitrary
        third-party implementations of :class:`RegularizationStrategy`.
        """
        if hasattr(regularization, "apply_fixed_tau"):
            return regularization.apply_fixed_tau(V, tau_anchor)
        return V + tau_anchor * jnp.diag(jnp.diag(V))

    # Detect whether the measure / covariance combination supports the
    # shared ``expectation_and_contributions`` primitive. When the
    # measure exposes ``moments_and_contributions``
    # (:class:`~emu_gmm.measures.synthetic.SyntheticMeasure`) or
    # ``expectation_and_contributions``
    # (:class:`~emu_gmm.measures.empirical.EmpiricalMeasure`) AND the
    # paired covariance strategy accepts ``cached_intermediates`` in
    # its ``covariance`` signature, the residual closure runs
    # ``vmap(psi)`` once and threads the cached payload into the
    # covariance strategy --- halving the per-step ``vmap`` cost (see
    # ``docs/reviews/v1x-performance-review.org`` findings #4 and #5).
    # Third-party covariance strategies that do not advertise
    # ``cached_intermediates`` are routed through the back-compat path
    # that calls ``measure.expectation`` and ``covariance.covariance``
    # separately, preserving the v1 contract.
    try:
        _cov_sig = inspect.signature(covariance.covariance)
        _cov_accepts_cache = "cached_intermediates" in _cov_sig.parameters
    except (TypeError, ValueError):  # builtins / C-extensions
        _cov_accepts_cache = False
    _emp_cache_method = getattr(measure, "expectation_and_contributions", None)
    _syn_cache_method = getattr(measure, "moments_and_contributions", None)
    if _cov_accepts_cache and _emp_cache_method is not None:
        _cache_method = _emp_cache_method
    elif _cov_accepts_cache and _syn_cache_method is not None:
        _cache_method = _syn_cache_method
    else:
        _cache_method = None

    # Residual closure: produces the whitened moment vector y.
    def residual_fn(theta_flat: Float[Array, " K"]) -> Float[Array, " M"]:
        theta = params_mod.unflatten_params(theta_flat, treedef)
        if _cache_method is not None:
            cached = _cache_method(model, theta)
            m = cached[0]
            # The minimal :class:`CovarianceStrategy` protocol does not
            # advertise ``cached_intermediates``; concrete IID / Clustered
            # / Synthetic strategies extend the signature with the kwarg
            # and the signature probe above gates this call. ``Any``
            # cast bypasses mypy's protocol-narrow check.
            V = cast(Any, covariance).covariance(
                model, theta, measure, cached_intermediates=cached
            )
        else:
            m = measure.expectation(model, theta)
            V = covariance.covariance(model, theta, measure)
        V_star = _apply_anchored(V)
        y = weighting.whitening_residual(m, V_star, theta)
        return y

    # Optimise.
    theta_hat_flat, optimizer_info = optimizer(residual_fn, theta_init_flat)
    theta_hat = params_mod.unflatten_params(theta_hat_flat, treedef)

    # ------------------------------------------------------------------
    # Post-optimum inference. The previous structure walked the
    # residual pipeline (expectation -> covariance -> regularisation ->
    # Cholesky -> whitened residual) three times in eager mode at
    # ``theta_hat`` --- once via ``measure.expectation`` +
    # ``covariance.covariance``, once via ``weighting.whitening_residual``,
    # and once more via ``jax.grad(half_obj)`` which re-traces
    # ``residual_fn`` from scratch. At the typical v1 sizes this
    # totalled ~0.4s of cumulative small-op dispatch on every call
    # (see ``docs/reviews/v1x-performance-review.org`` finding #6).
    #
    # We hoist the entire raw inference pipeline into a single jit'd
    # helper so XLA fuses the pipeline once. The estimate() entry point
    # calls it exactly one time at the end. ``J_dof`` is a Python int
    # (it's static, derived from the static ``M`` and ``K``) and the
    # only-defined-when-overidentified branch for ``J_pvalue`` is
    # resolved with a static guard outside the jit'd body --- ``J_dof``
    # cannot be traced as a chi^2 dof argument anyway, and the
    # weighted-chi^2 adjustment is only meaningful when overidentified.
    half_M_minus_K_overidentified = (M - K) > 0
    J_dof = max(M - K, 0)

    def _compute_inference(
        theta_flat: Float[Array, " K"],
    ) -> tuple[
        Float[Array, "K K"],  # Sigma_theta_arr
        Float[Array, "M M"],  # V_star_hat (== V_X data)
        Float[Array, ""],  # J_stat
        Float[Array, ""],  # kappa_V
        Float[Array, ""],  # tau_hat
        Float[Array, "K K"],  # info_matrix (kept for diagnostics parity)
        Float[Array, " M"],  # m_hat
        Float[Array, "M M"],  # L_hat (Cholesky factor of V_star_hat)
        Float[Array, " M"],  # y_hat
        Float[Array, "M K"],  # G_hat
        Float[Array, "M M"],  # V_hat (unregularised; needed for adjusted p)
        Float[Array, ""],  # cholesky_pivot_min
        Float[Array, ""],  # final_gradient_norm
        Float[Array, ""],  # J_pvalue
        Float[Array, ""],  # J_pvalue_adjusted
    ]:
        """One-shot raw inference pipeline at ``theta_flat``.

        Folds the moment evaluation, covariance, anchored ridge,
        Cholesky factorisation, whitened residual, Jacobian,
        ``Sigma_theta = (G' V^{-1} G)^{-1}``, J-statistic, p-values,
        and the half-objective gradient norm into one jit-compiled
        block. All the loose pieces the surrounding code used to
        materialise via separate eager calls are returned together so
        the caller can wrap them in labelled outputs / diagnostics.
        """
        theta_local = params_mod.unflatten_params(theta_flat, treedef)
        # Re-walk the cached residual path so the post-optimum pass
        # benefits from the same fusion the optimiser saw.
        if _cache_method is not None:
            cached = _cache_method(model, theta_local)
            m_local = cached[0]
            V_local = cast(Any, covariance).covariance(
                model, theta_local, measure, cached_intermediates=cached
            )
        else:
            m_local = measure.expectation(model, theta_local)
            V_local = covariance.covariance(model, theta_local, measure)
        V_star_local = _apply_anchored(V_local)
        # Cholesky of V_star (single factorisation reused below).
        L_local = cho.cholesky(V_star_local)
        # Whitened residual via the weighting policy (CU recomputes the
        # Cholesky from V_star_local --- under jit XLA folds the
        # duplicate Cholesky into the same kernel).
        y_local = weighting.whitening_residual(m_local, V_star_local, theta_local)
        J_local = jnp.sum(y_local * y_local)
        # Jacobian.
        G_local_raw = measure.jacobian(model, theta_local)
        if hasattr(G_local_raw, "array"):
            G_local_raw = G_local_raw.array
        G_local = jnp.asarray(G_local_raw)
        # Sigma_theta via the Cholesky factor of V_star_local.
        Z_local = jax.scipy.linalg.solve_triangular(L_local, G_local, lower=True)
        info_local = Z_local.T @ Z_local
        Sigma_local = jnp.linalg.inv(info_local)
        kappa_local = jnp.linalg.cond(V_star_local)
        pivot_min_local = jnp.min(jnp.diag(L_local))

        # Half-objective gradient at the optimum --- folds into the
        # same fused kernel under jit, eliminating the standalone
        # ``jax.grad(half_obj)`` retrace.
        def _half(tf):
            theta_inner = params_mod.unflatten_params(tf, treedef)
            if _cache_method is not None:
                inner_cached = _cache_method(model, theta_inner)
                m_inner = inner_cached[0]
                V_inner = cast(Any, covariance).covariance(
                    model, theta_inner, measure, cached_intermediates=inner_cached
                )
            else:
                m_inner = measure.expectation(model, theta_inner)
                V_inner = covariance.covariance(model, theta_inner, measure)
            V_star_inner = _apply_anchored(V_inner)
            y_inner = weighting.whitening_residual(m_inner, V_star_inner, theta_inner)
            return 0.5 * jnp.sum(y_inner * y_inner)

        grad_norm_local = jnp.linalg.norm(jax.grad(_half)(theta_flat))
        # J-test p-values (both kept as traced 0-d arrays even in the
        # under-/just-identified case so the return signature is
        # invariant under jit).
        if half_M_minus_K_overidentified:
            J_pv = jax.scipy.stats.chi2.sf(J_local, J_dof)
            J_pv_adj_binding = regularization_adjusted_pvalue(
                J_local, V_local, V_star_local, G_local
            )
            binding_flag = jnp.asarray(_binding_ridge(regularization, tau_anchor))
            J_pv_adj = jnp.where(binding_flag, J_pv_adj_binding, J_pv)
        else:
            J_pv = jnp.asarray(jnp.nan)
            J_pv_adj = jnp.asarray(jnp.nan)
        return (
            Sigma_local,
            V_star_local,
            J_local,
            kappa_local,
            tau_anchor,
            info_local,
            jnp.asarray(m_local),
            L_local,
            y_local,
            G_local,
            V_local,
            pivot_min_local,
            grad_norm_local,
            J_pv,
            J_pv_adj,
        )

    _compute_inference_jit = jax.jit(_compute_inference)
    (
        Sigma_theta_arr,
        V_star_hat,
        J_stat,
        kappa_V,
        tau_hat,
        info_matrix,
        m_hat,
        L,
        y_hat,
        G_hat,
        V_hat,
        cholesky_pivot_min,
        final_gradient_norm,
        J_pvalue,
        J_pvalue_adjusted,
    ) = _compute_inference_jit(theta_hat_flat)

    # Labelled outputs.
    Params = axes_mod.params_axis(K)
    ParamsDual = axes_mod.params_dual_axis(K)
    Moments = axes_mod.moments_axis(M)
    MomentsDual = axes_mod.moments_dual_axis(M)

    Sigma_theta = labels_mod.label_matrix(Sigma_theta_arr, Params, ParamsDual)
    V_X = labels_mod.label_matrix(V_star_hat, Moments, MomentsDual)

    # Scalar diagnostics produced by the jit'd pipeline are kept as
    # 0-d JAX arrays; the eager-only ``to_pandas`` / ``__repr__``
    # boundary casts to Python floats.
    binding_ridge = _binding_ridge(regularization, tau_hat)

    N_j_arr = _effective_n_per_moment(measure, theta_hat, M)

    # Hessian-condition trio (issue #10). G' Lambda G with
    # Lambda = (V_star_hat)^{-1} is the v1 information matrix
    # (CLAUDE.md commitment 5); cond_info reports its condition number
    # along three views (raw / data_only / exclude_gauge). For v1
    # data_only and exclude_gauge alias to raw; #7 (penalty hook) and
    # the v2 manifold epic will distinguish them. ``compute_cond_info``
    # operates on the *already-computed* ``G_hat`` and ``V_star_hat``
    # returned from the jit'd block, so the cost is a small extra
    # Cholesky + matmul --- not a full residual pipeline retrace.
    cond_info = compute_cond_info(G_hat, V_star_hat)

    # Optimiser-health summary. step_norm / accepted_step_count are
    # left as None because neither optimistix.LevenbergMarquardt nor
    # scipy.optimize.least_squares expose them in their result objects.
    optimizer_health = build_optimizer_health(
        optimizer_info=optimizer_info,
        final_gradient_norm=final_gradient_norm,
    )

    diagnostics = build_diagnostics(
        tau_realised=tau_hat,
        kappa_V=kappa_V,
        binding_ridge=binding_ridge,
        cholesky_pivot_min=cholesky_pivot_min,
        final_objective=J_stat,
        final_gradient_norm=final_gradient_norm,
        N_j_array=N_j_arr,
        moment_residual_array=m_hat,
        moments_axis=Moments,
        optimizer_info=optimizer_info,
        cond_info=cond_info,
        optimizer_health=optimizer_health,
    )

    # ``converged`` and ``iterations`` are derived from the optimiser's
    # info. Under ``jit`` / ``vmap`` the status is the literal string
    # ``"traced"`` and steps are a 0-d JAX array; both must avoid
    # ``int()`` / Python branches that touch traced values.
    converged = optimizer_info.status in ("converged", "traced")
    iterations = optimizer_info.steps

    return EstimationResult(
        theta_hat=theta_hat,
        Sigma_theta=Sigma_theta,
        V_X=V_X,
        J_stat=J_stat,
        J_dof=J_dof,
        J_pvalue=J_pvalue,
        J_pvalue_adjusted=J_pvalue_adjusted,
        converged=converged,
        iterations=iterations,
        theta_init=theta_init,
        measure=measure,
        covariance=covariance,
        weighting=weighting,
        regularization=regularization,
        diagnostics=diagnostics,
        labels=label_context,
    )


__all__ = ["estimate"]
