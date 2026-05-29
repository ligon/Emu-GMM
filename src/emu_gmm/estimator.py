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

    # Inference quantities at theta_hat. Use the *anchored* tau here too,
    # so reported diagnostics are consistent with the surface the
    # optimiser actually saw.
    m_hat = jnp.asarray(measure.expectation(model, theta_hat))
    V_hat = covariance.covariance(model, theta_hat, measure)
    V_star_hat = _apply_anchored(V_hat)
    tau_hat = tau_anchor

    # G = E_mu[grad_theta psi] : (M, K).
    G_hat = measure.jacobian(model, theta_hat)
    if hasattr(G_hat, "array"):
        G_hat = G_hat.array
    G_hat = jnp.asarray(G_hat)

    # Sigma_theta = (G' V^{-1} G)^{-1} via Cholesky of V_star_hat.
    # Solve L Z = G for Z; then info_matrix = Z' Z.
    L = cho.cholesky(V_star_hat)
    Z = jax.scipy.linalg.solve_triangular(L, G_hat, lower=True)  # (M, K)
    info_matrix = Z.T @ Z  # (K, K)
    # Use jnp.linalg.inv for v1; under-identified problems will yield NaN.
    Sigma_theta_arr = jnp.linalg.inv(info_matrix)

    # J-stat. Keep as a 0-d JAX array so the result flows through
    # ``jit`` / ``vmap``; users cast at the eager boundary (e.g. inside
    # ``to_pandas``).
    y_hat = weighting.whitening_residual(m_hat, V_star_hat, theta_hat)
    J_stat = jnp.sum(y_hat * y_hat)
    J_dof = max(M - K, 0)
    if J_dof > 0:
        # ``jax.scipy.stats.chi2.sf`` is traceable; ``scipy.stats.chi2.sf``
        # is not. The dof is a static Python int (it comes from M and K,
        # which are static closure variables).
        J_pvalue = jax.scipy.stats.chi2.sf(J_stat, J_dof)
        # Regularisation-adjusted p-value: weighted-chi^2 limit per
        # mcar-asymptotics.org Theorem 6. Computed unconditionally and
        # surfaced as ``J_pvalue_adjusted`` so users can compare against
        # the nominal value; we then pick between them based on whether
        # the ridge is binding.
        J_pvalue_adjusted_unbinding = J_pvalue  # tau ~= 0 case
        J_pvalue_adjusted_binding = regularization_adjusted_pvalue(
            J_stat, V_hat, V_star_hat, G_hat
        )
        binding_flag = jnp.asarray(_binding_ridge(regularization, tau_hat))
        J_pvalue_adjusted = jnp.where(
            binding_flag, J_pvalue_adjusted_binding, J_pvalue_adjusted_unbinding
        )
    else:
        J_pvalue = jnp.asarray(jnp.nan)  # under- or just-identified
        J_pvalue_adjusted = jnp.asarray(jnp.nan)

    # Labelled outputs.
    Params = axes_mod.params_axis(K)
    ParamsDual = axes_mod.params_dual_axis(K)
    Moments = axes_mod.moments_axis(M)
    MomentsDual = axes_mod.moments_dual_axis(M)

    Sigma_theta = labels_mod.label_matrix(Sigma_theta_arr, Params, ParamsDual)
    V_X = labels_mod.label_matrix(V_star_hat, Moments, MomentsDual)

    # Diagnostics. All scalars are kept as 0-d JAX arrays so the
    # ``estimate`` call traces cleanly under ``jit`` / ``vmap``. The
    # eager-only ``to_pandas`` / ``__repr__`` boundary casts to Python
    # floats; user code that does ``float(result.J_stat)`` continues to
    # work because 0-d JAX arrays are float()-castable outside of trace.
    kappa_V = jnp.linalg.cond(V_star_hat)
    binding_ridge = _binding_ridge(regularization, tau_hat)
    cholesky_pivot_min = jnp.min(jnp.diag(L))

    # Gradient of (1/2)||y||^2 at the optimum.
    def half_obj(tf: Float[Array, " K"]) -> Float[Array, ""]:
        y = residual_fn(tf)
        return 0.5 * jnp.sum(y * y)

    final_grad = jax.grad(half_obj)(theta_hat_flat)
    final_gradient_norm = jnp.linalg.norm(final_grad)

    N_j_arr = _effective_n_per_moment(measure, theta_hat, M)

    # Hessian-condition trio (issue #10). G' Lambda G with
    # Lambda = (V_star_hat)^{-1} is the v1 information matrix
    # (CLAUDE.md commitment 5); cond_info reports its condition number
    # along three views (raw / data_only / exclude_gauge). For v1
    # data_only and exclude_gauge alias to raw; #7 (penalty hook) and
    # the v2 manifold epic will distinguish them.
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
