"""The :func:`estimate` entry point and the :func:`build_estimator` factory.

Ties Phases 1-4 together into a stateless function:

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

The :func:`build_estimator` factory exposes the setup-once /
call-many pattern: it does the one-time work (label probing, anchored-
ridge construction, jit compilation of the post-optimum inference
block, residual-closure construction) and returns a callable
``f(theta_init, measure) -> EstimationResult`` that pays the compile
cost only on the *first* call. :func:`estimate` is now a thin wrapper
around :func:`build_estimator`; its result is bitwise-equivalent to
the v1 path for one-shot callers.

Caching contract
----------------
The residual closure built inside :func:`build_estimator` captures the
*template* ``measure`` instance. Subsequent calls of the returned
callable reuse the same closure --- and therefore the same JAX
compilation cache entry inside the optimiser --- whenever the
call-time ``measure`` is the same object (``measure is template``).
Calls with a different measure object rebuild the closure; if the
measure's pytree structure is unchanged this still hits JAX's pjit
cache for the post-optimum inference block, but the optimiser path
will trace one more time because its internal cache keys on the new
closure identity.

See ``docs/api-sketch.org`` Section 4 and
``docs/implementation-plan.org`` Section 7 for the architectural spec.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
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
from emu_gmm.penalty import PenaltyStrategy
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    Emu_GMM_DimensionError,
    EstimationResult,
    Measure,
    Optimizer,
    OptimizerInfo,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import ContinuouslyUpdated, Fixed, IteratedWeighting


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


def _run_iterated_weighting(
    *,
    weighting: IteratedWeighting,
    make_residual_fn: Any,
    cu_residual_fn: Any,
    model: StructuralModel,
    measure: Measure,
    covariance: CovarianceStrategy,
    apply_ridge: Any,
    optimizer: Optimizer,
    theta_init_flat: Float[Array, " K"],
    treedef: Any,
) -> tuple[Float[Array, " K"], OptimizerInfo, str]:
    """Drive the outer iterated-GMM loop in pure Python.

    Each outer step refreshes :math:`V(\\theta_k)`, applies the anchored
    ridge, freezes the resulting Cholesky factor into a
    :class:`Fixed`-weight residual, and delegates the inner solve to
    the user's :class:`Optimizer`. See the docstring of
    :class:`~emu_gmm.weighting.IteratedWeighting` for the full
    semantics, the rescaled-tolerance termination test, and the
    inner-solve divergence handling.
    """
    theta_k_flat = jnp.asarray(theta_init_flat)
    last_info: OptimizerInfo | None = None
    total_inner_steps = 0
    outer_status = "max_iterations"
    saw_inner_non_convergence = False
    inner_non_convergence_statuses: list[str] = []

    rescale_eps = 1e-12

    for _k in range(int(weighting.weighting_iterations)):
        theta_k = params_mod.unflatten_params(theta_k_flat, treedef)
        V_k = covariance.covariance(model, theta_k, measure)
        V_star_k = apply_ridge(V_k)
        fixed_k = Fixed.from_V0(V_star_k)
        inner_residual = make_residual_fn(fixed_k)

        theta_next_flat, info_k = optimizer(inner_residual, theta_k_flat)
        last_info = info_k
        inner_status = str(getattr(info_k, "status", ""))
        if inner_status not in ("converged", "traced"):
            saw_inner_non_convergence = True
            inner_non_convergence_statuses.append(inner_status)
        try:
            total_inner_steps += int(info_k.steps)
        except (TypeError, ValueError):
            pass

        delta = jnp.linalg.norm(theta_next_flat - theta_k_flat)
        theta_scale = float(jnp.maximum(jnp.linalg.norm(theta_next_flat), rescale_eps))
        theta_k_flat = theta_next_flat
        if float(delta) < float(weighting.weighting_tol) * theta_scale:
            outer_status = "converged"
            break

    if saw_inner_non_convergence:
        outer_status = "inner_non_convergence"
        warnings.warn(
            "IteratedWeighting saw at least one inner Fixed-weight solve "
            "that did not certify convergence (inner statuses: "
            f"{inner_non_convergence_statuses!r}). The outer V-refresh "
            "is built on top of the inner LM step, so a non-converged "
            "inner solve invalidates the resulting V_{k+1}. The returned "
            "theta is the last accepted iterate but should not be trusted "
            "as a GMM estimate; rerun with a larger inner iteration "
            "budget or switch to ContinuouslyUpdated weighting.",
            UserWarning,
            stacklevel=3,
        )
    elif outer_status == "max_iterations":
        warnings.warn(
            "IteratedWeighting exhausted "
            f"{weighting.weighting_iterations} outer iterations without "
            f"reaching weighting_tol={weighting.weighting_tol:g} "
            "(rescaled by max(||theta||, eps)). The V-refresh fixed "
            "point was not reached; iterated GMM is not guaranteed to be "
            "a contraction on misspecified models. Consider switching to "
            "ContinuouslyUpdated weighting.",
            UserWarning,
            stacklevel=3,
        )

    assert last_info is not None

    y_final = cu_residual_fn(theta_k_flat)
    final_objective_cu = 0.5 * jnp.sum(y_final * y_final)

    final_info = OptimizerInfo(
        steps=total_inner_steps,
        status=outer_status,
        final_objective=final_objective_cu,
        backend=last_info.backend,
    )
    return theta_k_flat, final_info, outer_status


def build_estimator(
    model: StructuralModel,
    *,
    measure: Measure,
    covariance: CovarianceStrategy,
    weighting: WeightingStrategy | None = None,
    regularization: RegularizationStrategy | None = None,
    optimizer: Optimizer | None = None,
    theta_init: ParamsLike,
    moment_names: tuple[str, ...] | None = None,
    penalty: PenaltyStrategy | None = None,
) -> Callable[[ParamsLike, Measure], EstimationResult]:
    """Build a re-usable estimator callable.

    Pays the one-time setup cost (label probing, anchored-ridge
    construction, residual-closure construction, jit compilation of
    the post-optimum inference block) at construction time. The
    returned callable ``f(theta_init, measure) -> EstimationResult``
    then runs the optimiser + inference, reusing the jit-compiled
    inner functions across calls.

    This matters at scale. With the bare :func:`estimate` entry point,
    each call rebuilds the residual closure and the inference block,
    so JAX's pjit cache (which keys on closure identity) misses every
    time --- optimistix LM ends up being re-traced per call. At
    N=1000 the perf review measured ~1.5 s per call; at N=58k
    (cereal-scale) it rises to 3.0-3.3 s. :func:`build_estimator`
    amortises that cost: on the second and subsequent calls with the
    same ``measure`` instance (or any pytree-equal structure once the
    factory's residual closure has been traced), the optimiser path
    hits the JAX compilation cache and runs the kernel directly.

    Parameters
    ----------
    model : :data:`StructuralModel`
        Per-observation residual function ``psi(x, theta) -> (M,) array``.
        Captured in the factory's closure; the returned callable does not
        accept a different model.
    measure : :class:`Measure`
        A template measure. Used here for label probing and ridge
        anchoring. The returned callable's *cache-hit path* requires
        being invoked with the same measure object (``measure is
        template_measure``); calls with a different measure are still
        correct but rebuild the residual closure so the optimiser's
        internal JIT cache misses once on that new closure.
    covariance : :class:`CovarianceStrategy`
        Captured in the factory's closure; fixed for the lifetime of the
        returned callable.
    weighting, regularization, optimizer, theta_init, moment_names, penalty
        See :func:`estimate`.

    Returns
    -------
    Callable taking ``(theta_init, measure) -> EstimationResult``.
    """
    # Defaults.
    if weighting is None:
        weighting = ContinuouslyUpdated()
    if regularization is None:
        regularization = DiagonalTikhonov()
    if optimizer is None:
        optimizer = optimistix_lm()

    # Capture the template measure so subsequent calls can detect
    # identity reuse and short-circuit residual-closure rebuilding.
    template_measure = measure

    # Probe M by evaluating the expectation once at theta_init.
    m_probe = template_measure.expectation(model, theta_init)
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

    # Probe K from the parameter dataclass before touching
    # ``flatten_params`` (which fails with an opaque ``jnp.stack of
    # empty list`` error when the dataclass has zero fields).
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

    # Probe for labelled output by calling model on one sample
    # observation, if the measure exposes one.
    x_sample = _sample_observation(template_measure, theta_init)
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

    # Flatten the template parameters --- the treedef pinned here is
    # used for unflatten on every call.
    _, treedef = params_mod.flatten_params(theta_init)
    K = K_probe

    # Anchor-once-then-freeze tau policy (design.org §5; CLAUDE.md
    # commitment 3).
    V0 = covariance.covariance(model, theta_init, template_measure)
    _V0_star, tau_anchor = regularization.apply(V0)
    tau_anchor = jnp.asarray(tau_anchor)

    def _apply_anchored(V: Float[Array, "M M"]) -> Float[Array, "M M"]:
        """Apply the ridge at the anchored ``tau_anchor`` deterministically."""
        if hasattr(regularization, "apply_fixed_tau"):
            return regularization.apply_fixed_tau(V, tau_anchor)
        return V + tau_anchor * jnp.diag(jnp.diag(V))

    # Detect whether the measure / covariance combination supports the
    # shared ``expectation_and_contributions`` primitive.
    try:
        _cov_sig = inspect.signature(covariance.covariance)
        _cov_accepts_cache = "cached_intermediates" in _cov_sig.parameters
    except (TypeError, ValueError):
        _cov_accepts_cache = False
    if _cov_accepts_cache and hasattr(
        template_measure, "expectation_and_contributions"
    ):
        _cache_attr_name: str | None = "expectation_and_contributions"
    elif _cov_accepts_cache and hasattr(template_measure, "moments_and_contributions"):
        _cache_attr_name = "moments_and_contributions"
    else:
        _cache_attr_name = None

    # Residual closure factory. ``measure_local`` is captured in the
    # closure so jit / pjit cache identity is preserved on subsequent
    # calls with the same measure instance.
    def _make_residual_fn(
        weighting_for_solve: WeightingStrategy,
        measure_local: Measure,
    ) -> Callable[[Float[Array, " K"]], Float[Array, " M"]]:
        cache_method = (
            getattr(measure_local, _cache_attr_name)
            if _cache_attr_name is not None
            else None
        )
        if penalty is None:

            def residual_fn(
                theta_flat: Float[Array, " K"],
            ) -> Float[Array, " M"]:
                theta = params_mod.unflatten_params(theta_flat, treedef)
                if cache_method is not None:
                    cached = cache_method(model, theta)
                    m = cached[0]
                    V = cast(Any, covariance).covariance(
                        model,
                        theta,
                        measure_local,
                        cached_intermediates=cached,
                    )
                else:
                    m = measure_local.expectation(model, theta)
                    V = covariance.covariance(model, theta, measure_local)
                V_star = _apply_anchored(V)
                y = weighting_for_solve.whitening_residual(m, V_star, theta)
                return y

        else:

            def residual_fn(
                theta_flat: Float[Array, " K"],
            ) -> Float[Array, " M"]:
                theta = params_mod.unflatten_params(theta_flat, treedef)
                if cache_method is not None:
                    cached = cache_method(model, theta)
                    m = cached[0]
                    V = cast(Any, covariance).covariance(
                        model,
                        theta,
                        measure_local,
                        cached_intermediates=cached,
                    )
                else:
                    m = measure_local.expectation(model, theta)
                    V = covariance.covariance(model, theta, measure_local)
                V_star = _apply_anchored(V)
                y = weighting_for_solve.whitening_residual(m, V_star, theta)
                # The objective is ||y||^2 + p(theta); appending sqrt(p)
                # as a residual row reproduces it exactly under ||.||^2.
                # p is C^infty and >= 0, but sqrt(p) has a singular
                # gradient where p = 0. We lift with a tiny floor so the
                # LM Jacobian is finite at exact zero. The floor (1e-30)
                # is far below float64 tolerance and dominated by any
                # nonzero p.
                p = penalty.penalty(theta)
                extra = jnp.sqrt(p + 1e-30)
                return jnp.concatenate([y, jnp.atleast_1d(extra)])

        return residual_fn

    # Pre-build the residual closure for the template measure --- this
    # is what gives subsequent calls (with the same measure) their
    # closure-identity-based cache hits inside the optimiser.
    _residual_fn_template = _make_residual_fn(weighting, template_measure)

    # ------------------------------------------------------------------
    # Post-optimum inference, jit'd once per factory. Built as a
    # measure-parameterised closure factory so that calls with a
    # different measure object correctly rebuild the inference closure
    # (preserving correctness), but the *same* measure path reuses the
    # cached jit entry.
    half_M_minus_K_overidentified = (M - K) > 0
    J_dof = max(M - K, 0)

    def _make_compute_inference_jit(
        measure_local: Measure,
        residual_fn_local: Callable[[Float[Array, " K"]], Float[Array, " M"]],
    ) -> Callable[[Float[Array, " K"]], tuple[Any, ...]]:
        cache_method = (
            getattr(measure_local, _cache_attr_name)
            if _cache_attr_name is not None
            else None
        )

        def _compute_inference(
            theta_flat: Float[Array, " K"],
        ) -> tuple[
            Float[Array, "K K"],  # Sigma_theta_arr
            Float[Array, "M M"],  # V_star_hat
            Float[Array, ""],  # J_stat
            Float[Array, ""],  # kappa_V
            Float[Array, ""],  # tau_hat
            Float[Array, "K K"],  # info_matrix
            Float[Array, " M"],  # m_hat
            Float[Array, "M M"],  # L_hat
            Float[Array, " M"],  # y_hat
            Float[Array, "M K"],  # G_hat
            Float[Array, "M M"],  # V_hat
            Float[Array, ""],  # cholesky_pivot_min
            Float[Array, ""],  # final_gradient_norm
            Float[Array, ""],  # J_pvalue
            Float[Array, ""],  # J_pvalue_adjusted
        ]:
            theta_local = params_mod.unflatten_params(theta_flat, treedef)
            if cache_method is not None:
                cached = cache_method(model, theta_local)
                m_local = cached[0]
                V_local = cast(Any, covariance).covariance(
                    model,
                    theta_local,
                    measure_local,
                    cached_intermediates=cached,
                )
            else:
                m_local = measure_local.expectation(model, theta_local)
                V_local = covariance.covariance(model, theta_local, measure_local)
            V_star_local = _apply_anchored(V_local)
            L_local = cho.cholesky(V_star_local)
            y_local = weighting.whitening_residual(m_local, V_star_local, theta_local)
            J_local = jnp.sum(y_local * y_local)
            G_local_raw = measure_local.jacobian(model, theta_local)
            if hasattr(G_local_raw, "array"):
                G_local_raw = G_local_raw.array
            G_local = jnp.asarray(G_local_raw)
            Z_local = jax.scipy.linalg.solve_triangular(L_local, G_local, lower=True)
            info_local = Z_local.T @ Z_local
            Sigma_local = jnp.linalg.inv(info_local)
            kappa_local = jnp.linalg.cond(V_star_local)
            pivot_min_local = jnp.min(jnp.diag(L_local))

            def _half(tf: Float[Array, " K"]) -> Float[Array, ""]:
                r = residual_fn_local(tf)
                return 0.5 * jnp.sum(r * r)

            grad_norm_local = jnp.linalg.norm(jax.grad(_half)(theta_flat))
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

        return jax.jit(_compute_inference)

    # Pre-build the inference closure for the template measure so
    # subsequent same-measure calls hit the jit cache.
    _compute_inference_jit_template = _make_compute_inference_jit(
        template_measure, _residual_fn_template
    )

    def _run(
        theta_init_call: ParamsLike,
        measure_call: Measure,
    ) -> EstimationResult:
        theta_init_flat, _ = params_mod.flatten_params(theta_init_call)
        # Reuse the pre-built residual / inference closures when the
        # measure identity matches the template. The pjit cache inside
        # optimistix keys on the residual_fn closure identity, so this
        # is what delivers the second-call no-retrace property.
        if measure_call is template_measure:
            residual_fn = _residual_fn_template
            compute_inference_jit = _compute_inference_jit_template
        else:
            residual_fn = _make_residual_fn(weighting, measure_call)
            compute_inference_jit = _make_compute_inference_jit(
                measure_call, residual_fn
            )

        iterated_status: str | None = None
        # Dispatch by the WeightingStrategy protocol's optional
        # ``requires_outer_loop`` flag rather than by isinstance against
        # IteratedWeighting. Third-party strategies that need their own
        # Python-level outer loop can opt in by setting
        # ``requires_outer_loop = True`` and implementing
        # ``outer_loop_driver`` with the documented signature.
        if getattr(weighting, "requires_outer_loop", False):
            # The CU-fallback residual is the user-facing whitened
            # residual of the strategy (its ``whitening_residual`` is
            # the CU form for IteratedWeighting and equivalent for any
            # other outer-loop strategy). The driver uses it to report
            # ``final_objective`` consistent with that fallback.
            cu_residual_fn = residual_fn
            (
                theta_hat_flat,
                optimizer_info,
                iterated_status,
            ) = cast(Any, weighting).outer_loop_driver(
                model,
                measure_call,
                covariance,
                theta_init_flat,
                treedef,
                make_residual_fn=lambda w: _make_residual_fn(w, measure_call),
                cu_residual_fn=cu_residual_fn,
                apply_ridge=_apply_anchored,
                optimizer=optimizer,
            )
        else:
            theta_hat_flat, optimizer_info = optimizer(residual_fn, theta_init_flat)

        theta_hat = params_mod.unflatten_params(theta_hat_flat, treedef)

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
        ) = compute_inference_jit(theta_hat_flat)

        Params = axes_mod.params_axis(K)
        ParamsDual = axes_mod.params_dual_axis(K)
        Moments = axes_mod.moments_axis(M)
        MomentsDual = axes_mod.moments_dual_axis(M)

        Sigma_theta = labels_mod.label_matrix(Sigma_theta_arr, Params, ParamsDual)
        V_X = labels_mod.label_matrix(V_star_hat, Moments, MomentsDual)

        binding_ridge = _binding_ridge(regularization, tau_hat)

        if penalty is None:
            final_objective_data = J_stat
            final_objective_full = J_stat
            penalty_hessian = None
        else:
            p_hat = penalty.penalty(theta_hat)
            final_objective_data = J_stat
            final_objective_full = J_stat + p_hat

            def _p_flat(tf: Float[Array, " K"]) -> Float[Array, ""]:
                return penalty.penalty(params_mod.unflatten_params(tf, treedef))

            penalty_hessian = jax.hessian(_p_flat)(theta_hat_flat)

        N_j_arr = _effective_n_per_moment(measure_call, theta_hat, M)
        cond_info = compute_cond_info(
            G_hat, V_star_hat, penalty_hessian=penalty_hessian
        )

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
            final_objective_data=final_objective_data,
            final_objective_full=final_objective_full,
            final_gradient_norm=final_gradient_norm,
            N_j_array=N_j_arr,
            moment_residual_array=m_hat,
            moments_axis=Moments,
            optimizer_info=optimizer_info,
            cond_info=cond_info,
            optimizer_health=optimizer_health,
        )

        converged = optimizer_info.status in ("converged", "traced")
        iterations = optimizer_info.steps
        if iterated_status in ("max_iterations", "inner_non_convergence"):
            converged = False

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
            theta_init=theta_init_call,
            measure=measure_call,
            covariance=covariance,
            weighting=weighting,
            regularization=regularization,
            diagnostics=diagnostics,
            labels=label_context,
        )

    return _run


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
    penalty: PenaltyStrategy | None = None,
) -> EstimationResult:
    """Estimate :math:`\\hat\\theta` by minimising
    :math:`Q_\\mu(\\theta) = \\| L_\\mu(\\theta)^{-1}\\, \\mathbb{E}_\\mu[\\psi(\\cdot,\\theta)] \\|^2`.

    Thin wrapper around :func:`build_estimator`: builds an estimator
    callable from the supplied arguments and immediately invokes it.
    Bitwise-equivalent to the v1 implementation for one-shot callers;
    callers that estimate the same model on many ``(theta_init,
    measure)`` pairs should prefer :func:`build_estimator` so the JIT
    cost is paid only once.

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
    penalty : :class:`PenaltyStrategy`, optional
        In-objective parameter penalty. When supplied, the criterion
        becomes :math:`Q_{\\mu,\\mathrm{pen}}(\\theta) = Q_\\mu(\\theta) +
        p(\\theta)` and the NLLS residual gets :math:`\\sqrt{p(\\theta)}`
        appended so the LM Jacobian picks up the parameter-space ridge
        via JAX AD. ``penalty=None`` preserves v1 behaviour bitwise.

    Returns
    -------
    :class:`EstimationResult`
    """
    run = build_estimator(
        model,
        measure=measure,
        covariance=covariance,
        weighting=weighting,
        regularization=regularization,
        optimizer=optimizer,
        theta_init=theta_init,
        moment_names=moment_names,
        penalty=penalty,
    )
    return run(theta_init, measure)


__all__ = ["estimate", "build_estimator"]
