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
import warnings
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

    Each outer step:

    1. evaluates :math:`V(\\theta_k)` via the covariance strategy and
       applies the *anchored* ridge to it,
    2. builds a :class:`Fixed`-weight strategy at the resulting Cholesky
       factor :math:`L_k`,
    3. delegates the inner solve to the user's optimiser, which runs
       under JIT.

    The loop terminates on either:

    - :math:`\\| \\theta_{k+1} - \\theta_k \\|_2 < \\texttt{weighting_tol}
      \\cdot \\max(\\| \\theta_k \\|_2, \\texttt{eps})` (status
      ``"converged"``) --- the tolerance is rescaled by the current
      parameter norm so that ``weighting_tol`` carries meaning across
      problems whose parameters differ by orders of magnitude, or
    - having performed ``weighting_iterations`` outer steps without
      meeting the rescaled tolerance (status ``"max_iterations"``).

    Inner-solve divergence handling
    -------------------------------
    Each inner :class:`Fixed`-weight solve returns its own
    :class:`~emu_gmm.types.OptimizerInfo`. If any inner ``info_k.status``
    is neither ``"converged"`` nor ``"traced"`` (i.e. the inner LM /
    least-squares run hit ``max_iterations`` or otherwise failed to
    certify convergence), the iterated driver emits a
    :class:`UserWarning` and returns outer status
    ``"inner_non_convergence"`` so the caller can flip
    ``EstimationResult.converged`` to ``False``.

    On ``"max_iterations"`` a :class:`UserWarning` is emitted so the
    caller is told the V-refresh fixed point was not reached; the
    partially-iterated ``theta_k`` is returned regardless. The returned
    :class:`~emu_gmm.types.OptimizerInfo` carries the outer-loop status
    in its ``status`` field, and its ``final_objective`` is the
    user-facing CU-fallback objective :math:`Q(\\hat\\theta) =
    \\| L(\\hat\\theta)^{-1} m(\\hat\\theta) \\|^2` evaluated at the
    returned :math:`\\hat\\theta` --- *not* the inner Fixed-weight
    objective at the penultimate :math:`V_k`. The two coincide when the
    outer loop converges but differ in finite samples when it does not,
    and the user-facing value is the one consistent with how
    :class:`IteratedWeighting` reports objective values to downstream
    diagnostics.
    """
    theta_k_flat = jnp.asarray(theta_init_flat)
    last_info: OptimizerInfo | None = None
    total_inner_steps = 0
    outer_status = "max_iterations"
    # ``inner_non_convergence`` overrides ``max_iterations`` once seen,
    # because it indicates a deeper failure (the inner LM gave up).
    saw_inner_non_convergence = False
    inner_non_convergence_statuses: list[str] = []

    # eps for the rescaled-tolerance test; chosen at the float64 noise
    # floor so the absolute test still triggers when |theta_k| -> 0.
    rescale_eps = 1e-12

    for _k in range(int(weighting.weighting_iterations)):
        theta_k = params_mod.unflatten_params(theta_k_flat, treedef)
        # Refresh V at the current theta_k and apply the *anchored* ridge
        # so the inner Fixed-weight surface uses the same tau the rest of
        # the framework does. Then freeze a Fixed-weight closure at the
        # resulting Cholesky anchor for the inner solve.
        V_k = covariance.covariance(model, theta_k, measure)
        V_star_k = apply_ridge(V_k)
        fixed_k = Fixed.from_V0(V_star_k)
        inner_residual = make_residual_fn(fixed_k)

        theta_next_flat, info_k = optimizer(inner_residual, theta_k_flat)
        last_info = info_k
        # Inspect inner-solve status. ``"traced"`` is the placeholder
        # returned under jit (concrete status is not available); we
        # treat it as success because the iterated path is documented as
        # eager and any traced-status appearance there means the user
        # built a custom optimiser that doesn't surface a status.
        inner_status = str(getattr(info_k, "status", ""))
        if inner_status not in ("converged", "traced"):
            saw_inner_non_convergence = True
            inner_non_convergence_statuses.append(inner_status)
        try:
            total_inner_steps += int(info_k.steps)
        except (TypeError, ValueError):
            # ``steps`` may still be a traced scalar under jit; in eager
            # use (the contract for the iterated path) it is concrete.
            pass

        delta = jnp.linalg.norm(theta_next_flat - theta_k_flat)
        # Rescale the tolerance by the current parameter norm so the
        # test is meaningful when parameters differ by orders of
        # magnitude (e.g. one component O(1), another O(1e6)). The
        # ``rescale_eps`` floor protects the limit |theta_k| -> 0, where
        # an absolute test on ``weighting_tol`` is still the right thing.
        theta_scale = float(jnp.maximum(jnp.linalg.norm(theta_next_flat), rescale_eps))
        theta_k_flat = theta_next_flat
        if float(delta) < float(weighting.weighting_tol) * theta_scale:
            outer_status = "converged"
            break

    if saw_inner_non_convergence:
        # Inner divergence dominates the outer status: a non-converged
        # inner solve invalidates the V-refresh step that follows it.
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

    assert last_info is not None  # weighting_iterations >= 1 enforced

    # The user-facing ``final_objective`` is the CU-fallback objective at
    # the final theta_hat, *not* the inner Fixed-weight objective at the
    # penultimate V_k. The two coincide when iterated GMM converges
    # (V == V_k at the fixed point) but differ when it does not, and the
    # CU-fallback value is the one consistent with how the rest of the
    # framework reports the IteratedWeighting objective downstream.
    # Match the optimiser-side convention of reporting 0.5 * ||y||^2 (so
    # this field is comparable across weighting strategies and across
    # backends).
    y_final = cu_residual_fn(theta_k_flat)
    final_objective_cu = 0.5 * jnp.sum(y_final * y_final)

    final_info = OptimizerInfo(
        steps=total_inner_steps,
        status=outer_status,
        final_objective=final_objective_cu,  # type: ignore[arg-type]
        backend=last_info.backend,
    )
    return theta_k_flat, final_info, outer_status


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

    # Residual closure: produces the whitened moment vector y. Built as
    # a factory so the iterated path can rebuild it per outer step with
    # a frozen :class:`Fixed`-weight closure over the current ``V_k``.
    #
    # When a penalty is supplied, append sqrt(p(theta)) as one extra row
    # so the LM ||y||^2 surface absorbs the penalty without touching the
    # solver. ``penalty=None`` keeps the residual shape and values
    # bitwise identical to v1.
    def _make_residual_fn(
        weighting_for_solve: WeightingStrategy,
    ) -> Any:
        if penalty is None:

            def residual_fn(
                theta_flat: Float[Array, " K"],
            ) -> Float[Array, " M"]:
                theta = params_mod.unflatten_params(theta_flat, treedef)
                if _cache_method is not None:
                    cached = _cache_method(model, theta)
                    m = cached[0]
                    # The minimal :class:`CovarianceStrategy` protocol
                    # does not advertise ``cached_intermediates``;
                    # concrete IID / Clustered / Synthetic strategies
                    # extend the signature with the kwarg and the
                    # signature probe above gates this call. ``Any``
                    # cast bypasses mypy's protocol-narrow check.
                    V = cast(Any, covariance).covariance(
                        model, theta, measure, cached_intermediates=cached
                    )
                else:
                    m = measure.expectation(model, theta)
                    V = covariance.covariance(model, theta, measure)
                V_star = _apply_anchored(V)
                y = weighting_for_solve.whitening_residual(m, V_star, theta)
                return y

        else:

            def residual_fn(
                theta_flat: Float[Array, " K"],
            ) -> Float[Array, " M"]:
                theta = params_mod.unflatten_params(theta_flat, treedef)
                if _cache_method is not None:
                    cached = _cache_method(model, theta)
                    m = cached[0]
                    V = cast(Any, covariance).covariance(
                        model, theta, measure, cached_intermediates=cached
                    )
                else:
                    m = measure.expectation(model, theta)
                    V = covariance.covariance(model, theta, measure)
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

    # Optimise. Iterated weighting drives an outer Python loop of
    # Fixed-weight inner solves; all other strategies dispatch through
    # the residual_fn directly.
    iterated_status: str | None = None
    if isinstance(weighting, IteratedWeighting):
        # The CU-fallback residual is the user-facing whitened residual
        # of ``IteratedWeighting`` itself (its ``whitening_residual``
        # falls back to the CU form). The iterated driver uses it to
        # report ``final_objective`` consistent with that fallback.
        cu_residual_fn = _make_residual_fn(weighting)
        (
            theta_hat_flat,
            optimizer_info,
            iterated_status,
        ) = _run_iterated_weighting(
            weighting=weighting,
            make_residual_fn=_make_residual_fn,
            cu_residual_fn=cu_residual_fn,
            model=model,
            measure=measure,
            covariance=covariance,
            apply_ridge=_apply_anchored,
            optimizer=optimizer,
            theta_init_flat=theta_init_flat,
            treedef=treedef,
        )
    else:
        residual_fn = _make_residual_fn(weighting)
        theta_hat_flat, optimizer_info = optimizer(residual_fn, theta_init_flat)
    theta_hat = params_mod.unflatten_params(theta_hat_flat, treedef)
    # The downstream inference still uses the user-facing weighting
    # strategy. For IteratedWeighting that means re-running its
    # CU-equivalent ``whitening_residual`` on the final ``V_hat``;
    # the result is the same as the last inner Fixed-weight residual
    # at convergence because ``V_hat == V_k`` for the last accepted
    # ``theta_k`` (CU and Fixed coincide once V is held fixed at the
    # current theta).
    residual_fn = _make_residual_fn(weighting)

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
        # ``jax.grad(half_obj)`` retrace. We compute ``0.5 * ||r||^2``
        # by re-using the exact ``residual_fn`` the optimiser saw, so
        # the reported gradient norm matches the LM convergence
        # criterion bit-for-bit (and matches what users get when they
        # reconstruct the residual by hand). When ``penalty`` is
        # supplied the residual includes the appended sqrt(p+eps) row
        # automatically; with ``penalty=None`` this reduces to the v1
        # unpenalised data-only gradient norm.
        def _half(tf):
            r = residual_fn(tf)
            return 0.5 * jnp.sum(r * r)

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

    # Split the final objective into the *data* criterion and the
    # *full* criterion (data + penalty). With penalty=None the two
    # coincide and equal J_stat. With penalty supplied,
    # final_objective_full = J_stat + p(theta_hat) and matches what
    # ``optimizer_info.final_objective`` (== ||r||^2 at the optimum)
    # reports.
    if penalty is None:
        final_objective_data = J_stat
        final_objective_full = J_stat
        penalty_hessian = None
    else:
        p_hat = penalty.penalty(theta_hat)
        final_objective_data = J_stat
        final_objective_full = J_stat + p_hat

        # Penalty Hessian on the *flat* parameter axes: take the
        # Hessian of theta_flat -> p(unflatten(theta_flat)). This is
        # the natural (K, K) matrix to slot into the information-matrix
        # split inside ``compute_cond_info``. The closure captures
        # ``penalty`` and ``treedef`` from the outer scope; under jit
        # this is a normal trace-time constant.
        def _p_flat(tf: Float[Array, " K"]) -> Float[Array, ""]:
            return penalty.penalty(params_mod.unflatten_params(tf, treedef))

        penalty_hessian = jax.hessian(_p_flat)(theta_hat_flat)

    N_j_arr = _effective_n_per_moment(measure, theta_hat, M)

    # Hessian-condition trio (issue #10). G' Lambda G with
    # Lambda = (V_star_hat)^{-1} is the v1 information matrix
    # (CLAUDE.md commitment 5); cond_info reports its condition number
    # along three views (raw / data_only / exclude_gauge). When a
    # ``penalty`` is supplied (issue #7), ``'data_only'`` excludes the
    # penalty Hessian contribution while ``'raw'`` includes it; without
    # a penalty all three views coincide. ``compute_cond_info`` operates
    # on the *already-computed* ``G_hat`` and ``V_star_hat`` returned
    # from the jit'd block, so the cost is a small extra Cholesky +
    # matmul --- not a full residual pipeline retrace.
    cond_info = compute_cond_info(G_hat, V_star_hat, penalty_hessian=penalty_hessian)

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

    # ``converged`` and ``iterations`` are derived from the optimiser's
    # info. Under ``jit`` / ``vmap`` the status is the literal string
    # ``"traced"`` and steps are a 0-d JAX array; both must avoid
    # ``int()`` / Python branches that touch traced values.
    converged = optimizer_info.status in ("converged", "traced")
    iterations = optimizer_info.steps
    # Iterated weighting overrides convergence status when either:
    #   (a) the outer loop exhausted its iteration budget without
    #       reaching ``weighting_tol`` --- the inner solves may still
    #       have converged individually, but the V-refresh fixed point
    #       did not, or
    #   (b) any inner Fixed-weight solve itself failed to certify
    #       convergence; the inner divergence invalidates the outer
    #       V-refresh that builds on it.
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
        theta_init=theta_init,
        measure=measure,
        covariance=covariance,
        weighting=weighting,
        regularization=regularization,
        diagnostics=diagnostics,
        labels=label_context,
    )


__all__ = ["estimate"]
