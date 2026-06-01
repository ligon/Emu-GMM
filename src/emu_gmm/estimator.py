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
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats
import numpy as np
from jaxtyping import Array, Float

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import cholesky as cho
from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal import params as params_mod
from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule
from emu_gmm.diagnostics import (
    build_diagnostics,
    build_optimizer_health,
    compute_cond_info,
    regularization_adjusted_pvalue,
)
from emu_gmm.manifolds.euclidean import Euclidean
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.spec import ManifoldSpec
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.penalty import PenaltyStrategy
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


def _all_euclidean(manifold_spec: ManifoldSpec) -> bool:
    """True when every leaf is Euclidean and there is no gauge structure."""
    if manifold_spec.total_gauge_dim != 0:
        return False
    return all(isinstance(ls.manifold, Euclidean) for ls in manifold_spec.leaf_specs)


def _is_riemannian_optimizer(optimizer: Any) -> bool:
    """Detect a :class:`RiemannianOptimizer` by its ``manifold_spec`` param.

    A v1 :class:`~emu_gmm.types.Optimizer` has ``__call__(residual_fn,
    theta_init)``; a v2 :class:`~emu_gmm.manifolds.optimizer.RiemannianOptimizer`
    adds a third ``manifold_spec`` parameter. The signature is the
    distinguishing surface (plan §2.6 / §7).
    """
    try:
        sig = inspect.signature(optimizer.__call__)
    except (TypeError, ValueError):
        return False
    return "manifold_spec" in sig.parameters


def _resolve_optimizer(
    manifold_spec: ManifoldSpec,
    user_optimizer: Optimizer | None,
) -> tuple[Any, str]:
    """Resolve the optimiser and the dispatch mode (``"v1"`` / ``"v2"``).

    Rules (plan §2.6 / §7):

    1. ``None`` + all-Euclidean -> ``optimistix_lm()``, ``"v1"``.
    2. ``None`` + any non-Euclidean leaf -> ``riemannian_lm()``, ``"v2"``.
    3. v1 optimiser + all-Euclidean -> ``"v1"`` (no adapter).
    4. v1 optimiser + any non-Euclidean leaf -> :class:`TypeError`.
    5. :class:`RiemannianOptimizer` -> ``"v2"``.
    """
    all_euc = _all_euclidean(manifold_spec)
    if user_optimizer is None:
        if all_euc:
            return optimistix_lm(), "v1"
        return riemannian_lm(), "v2"
    if _is_riemannian_optimizer(user_optimizer):
        return user_optimizer, "v2"
    # User supplied a v1-style Optimizer.
    if all_euc:
        return user_optimizer, "v1"
    raise TypeError(
        "estimate(): the supplied optimizer satisfies the v1 Optimizer "
        "protocol (residual_fn, theta_init) but theta_init has a "
        "non-Euclidean manifold leaf (e.g. Positive). Use "
        "emu_gmm.manifolds.riemannian_lm.riemannian_lm() (or pass "
        "optimizer=None to auto-dispatch) for manifold parameters."
    )


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

    # Manifold spec + optimiser dispatch (plan §2.6 / §7). For v1-style
    # all-Euclidean trees this resolves to optimistix_lm() and mode
    # "v1"; a non-Euclidean leaf (e.g. Positive) routes to RiemannianLM
    # with mode "v2".
    manifold_spec = params_mod.manifold_spec_from_params(theta_init)
    optimizer, dispatch_mode = _resolve_optimizer(manifold_spec, optimizer)

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
    # Under-identification guard on the *identified ambient* dimension
    # (Phase 4 / INT-01 / R5/R9). The information matrix is the ambient
    # horizontal-projected sandwich G' Lambda G of size
    # ``total_dimension`` with an exact ``total_gauge_dim`` gauge
    # nullspace; the rank available to identify parameters is therefore
    # ``total_dimension - total_gauge_dim``. For v1 / all-Euclidean /
    # scalar-Positive trees ``total_dimension == K_probe`` and
    # ``total_gauge_dim == 0``, so this reduces to ``K_probe > M`` exactly
    # (bitwise v1 non-regression). For Product(PSDFixedRank(5,2),
    # Euclidean(1)): identified == 11 - 1 == 10, so M >= 10 is required.
    identified_dim = manifold_spec.total_dimension - manifold_spec.total_gauge_dim
    if identified_dim > M:
        raise Emu_GMM_DimensionError(
            f"estimate() requires M >= K_id, the identified parameter "
            f"dimension (over-/just-identified); got M={M} moments and "
            f"K_id={identified_dim} (= total_dimension "
            f"{manifold_spec.total_dimension} - gauge "
            f"{manifold_spec.total_gauge_dim}) (under-identified). "
            "For v1 / scalar trees K_id == K (the parameter count). "
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
    # used for unflatten on every call. For v1 / all-Euclidean trees the
    # v1 2-tuple ``flatten_params`` produces a scalar-only treedef and
    # ``unflatten_params`` reindexes scalars (manifold_spec=None). For a
    # v2 manifold tree (non-scalar ManifoldLeaf blocks) we take the
    # manifold-aware 3-tuple ``flatten_params_with_spec`` so the treedef
    # descends into each ManifoldLeaf and every later unflatten passes
    # ``manifold_spec`` to reshape the ambient blocks (Phase 4 / R19).
    # ``unflatten_spec`` is None on the v1 path -> v1 unflatten is bitwise
    # unchanged.
    if dispatch_mode == "v2":
        _, treedef, _ = params_mod.flatten_params_with_spec(theta_init)
        unflatten_spec: ManifoldSpec | None = manifold_spec
    else:
        _, treedef = params_mod.flatten_params(theta_init)
        unflatten_spec = None
    # ``total_dimension`` is the ambient tangent dimension that the
    # inference / label / axes path keys on. For v1 / all-Euclidean /
    # scalar-Positive trees it equals the field count ``K_probe``, so the
    # 226 v1 tests are bitwise unchanged.
    total_dimension = manifold_spec.total_dimension

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
                theta = params_mod.unflatten_params(
                    theta_flat, treedef, manifold_spec=unflatten_spec
                )
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
                theta = params_mod.unflatten_params(
                    theta_flat, treedef, manifold_spec=unflatten_spec
                )
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
    # Identified-parameter count: ambient total_dimension minus the
    # gauge nullspace. For the v1 all-Euclidean path total_gauge_dim==0
    # and total_dimension==K, so this recovers J_dof = M - K. For a
    # Positive(1,1) leaf gauge_dim==0 too, so J_dof = M - 1.
    dim_info = manifold_spec.total_dimension - manifold_spec.total_gauge_dim
    half_M_minus_K_overidentified = (M - dim_info) > 0
    J_dof = max(M - dim_info, 0)

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
            # theta_flat has length total_ambient_dim == total_dimension
            # (the ambient tangent dim D), which equals the field count K
            # only for v1 / all-scalar trees.
            theta_flat: Float[Array, " D"],
        ) -> tuple[
            Float[Array, "D D"],  # Sigma_theta_arr
            Float[Array, "M M"],  # V_star_hat
            Float[Array, ""],  # J_stat
            Float[Array, ""],  # kappa_V
            Float[Array, ""],  # tau_hat
            Float[Array, "D D"],  # info_matrix
            Float[Array, " M"],  # m_hat
            Float[Array, "M M"],  # L_hat
            Float[Array, " M"],  # y_hat
            Float[Array, "M D"],  # G_hat
            Float[Array, "M M"],  # V_hat
            Float[Array, ""],  # cholesky_pivot_min
            Float[Array, ""],  # final_gradient_norm
            Float[Array, ""],  # J_pvalue
            Float[Array, ""],  # J_pvalue_adjusted
        ]:
            theta_local = params_mod.unflatten_params(
                theta_flat, treedef, manifold_spec=unflatten_spec
            )
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
            # G = d m / d theta_flat at the FIXED theta_hat (commitment 5 /
            # delta method; AD of the MOMENT function, never through the
            # solver). For the v1 path this is exactly ``measure.jacobian``
            # (which itself ``jacfwd``s the unflatten->expectation closure),
            # kept verbatim so v1 stays bitwise. For the v2 manifold path
            # ``measure.jacobian`` would route through the v1 scalar-only
            # flatten and raise on a non-scalar ManifoldLeaf block, so we
            # AD a manifold-aware unflatten->expectation closure inline; the
            # result is the same (M, total_dimension) ambient Jacobian.
            if dispatch_mode == "v2":

                def _moment_of_flat(tf: Float[Array, " K"]) -> Float[Array, " M"]:
                    th = params_mod.unflatten_params(
                        tf, treedef, manifold_spec=unflatten_spec
                    )
                    return jnp.asarray(measure_local.expectation(model, th))

                G_local = jax.jacfwd(_moment_of_flat)(theta_flat)
            else:
                G_local_raw = measure_local.jacobian(model, theta_local)
                if hasattr(G_local_raw, "array"):
                    G_local_raw = G_local_raw.array
                G_local = jnp.asarray(G_local_raw)
            # Per-leaf G_riem assembly (Phase 4 / BUG-A / R4/R7). Iterate
            # over the manifold spec's leaves -- NOT range(K) over field
            # count -- slicing each leaf's ambient column block
            # ``G_local[:, offset:offset+size]``, scaling by the leaf's
            # retraction differential (the unit Convention-B differential
            # at v=0 for every native retraction, so this is the identity),
            # and applying the leaf's HORIZONTAL projection (a Lyapunov
            # solve for PSDFixedRank; identity for Euclidean / Positive) so
            # the gauge / vertical directions carry no information. ALL
            # ambient columns flow through -> G_riem is (M, total_dimension)
            # with no silent column drop. For v1 / all-scalar trees each
            # block is one column, the differential is 1, and the
            # projection is the identity, so G_riem == G_local bitwise.
            riem_blocks = []
            for ls in manifold_spec.leaf_specs:
                size = int(np.prod(ls.ambient_shape)) if ls.ambient_shape != () else 1
                block = G_local[:, ls.offset : ls.offset + size]  # (M, size)
                diff = ls.manifold.retraction_differential(
                    theta_flat[ls.offset : ls.offset + size]
                )
                block = jnp.asarray(diff) * block
                if ls.ambient_shape == () or ls.manifold.gauge_dim == 0:
                    # Scalar / Euclidean / Positive leaf: horizontal
                    # projection is the identity (bitwise v1 path).
                    riem_blocks.append(block)
                else:
                    # Non-scalar gauge-bearing leaf (PSDFixedRank): reshape
                    # the point block to ambient_shape and project each
                    # of the M Jacobian rows row-by-row onto the horizontal
                    # subspace, then ravel back to (M, size).
                    pt = jnp.reshape(
                        theta_flat[ls.offset : ls.offset + size], ls.ambient_shape
                    )
                    manifold = ls.manifold
                    shape = ls.ambient_shape

                    def _proj_row(
                        row: Float[Array, " size"],
                        _pt: Any = pt,
                        _m: Any = manifold,
                        _shape: Any = shape,
                        _size: int = size,
                    ) -> Float[Array, " size"]:
                        row_m = jnp.reshape(row, _shape)
                        proj = _m.projection(_pt, row_m)
                        return jnp.reshape(proj, (_size,))

                    riem_blocks.append(jax.vmap(_proj_row)(block))
            G_riem = jnp.concatenate(riem_blocks, axis=1)  # (M, total_dimension)
            Z_local = jax.scipy.linalg.solve_triangular(L_local, G_riem, lower=True)
            info_local = Z_local.T @ Z_local
            # Gauge-aware pseudo-inverse (Phase 4 / BUG-B / R6/R8): drop the
            # ``total_gauge_dim`` smallest eigenvalues BY COUNT. For
            # ``total_gauge_dim == 0`` (v1 / Euclidean / Positive) this is
            # exactly ``inv()`` (bitwise v1 non-regression).
            Sigma_local = pinv_eigvalrule(
                info_local, drop_smallest=manifold_spec.total_gauge_dim
            )
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
        if dispatch_mode == "v2":
            # Manifold-aware flatten: the v1 2-tuple raises on non-scalar
            # ManifoldLeaf blocks (R19). The optimiser path below ignores
            # ``theta_init_flat`` for v2 (it consumes the pytree directly),
            # but the buffer must still flatten without raising.
            theta_init_flat, _, _ = params_mod.flatten_params_with_spec(theta_init_call)
        else:
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
        elif dispatch_mode == "v2":
            # RiemannianOptimizer owns the flat<->pytree round-trip; it
            # takes the ORIGINAL pytree plus the manifold_spec. The
            # residual closure is the same flat-vector closure (scalar
            # leaves keep flat coords).
            theta_hat_pytree, optimizer_info = cast(Any, optimizer)(
                residual_fn, theta_init_call, manifold_spec
            )
            # Manifold-aware flatten of the recovered pytree: the v1
            # 2-tuple raises on non-scalar ManifoldLeaf blocks (R19).
            theta_hat_flat, _, _ = params_mod.flatten_params_with_spec(theta_hat_pytree)
        else:
            theta_hat_flat, optimizer_info = optimizer(residual_fn, theta_init_flat)

        theta_hat = params_mod.unflatten_params(
            theta_hat_flat, treedef, manifold_spec=unflatten_spec
        )

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

        # Param axes are sized by the ambient tangent dimension
        # ``total_dimension`` (Phase 4 / BUG-D / R14/R17), matching the
        # (total_dimension, total_dimension) Sigma_theta the inference
        # block now returns. For v1 / all-scalar trees
        # ``total_dimension == K``, so these are bitwise unchanged.
        Params = axes_mod.params_axis(total_dimension)
        ParamsDual = axes_mod.params_dual_axis(total_dimension)
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
                return penalty.penalty(
                    params_mod.unflatten_params(
                        tf, treedef, manifold_spec=unflatten_spec
                    )
                )

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
            gauge_nullspace_dim=manifold_spec.total_gauge_dim,
        )

        # #78: prefer the optimiser's REAL traced ``done`` flag when the
        # backend supplies it (the Riemannian-LM path). ``done`` is True
        # only when the while_loop exited on a convergence criterion --- it
        # is False when ``max_steps`` was hit --- so it does NOT suffer the
        # ``status == "traced"`` always-converged hazard under jit. Backends
        # that omit it (``done is None``: optimistix / scipy / iterated)
        # keep the original status-string semantics unchanged (no v1
        # regression). When ``done`` is a traced array under an outer jit,
        # ``converged`` rides as a traced bool just as ``optimizer_info``
        # already did.
        done_flag = getattr(optimizer_info, "done", None)
        if done_flag is None:
            converged: Any = optimizer_info.status in ("converged", "traced")
        else:
            converged = jnp.asarray(done_flag)
            try:
                converged = bool(converged)
            except (jax.errors.TracerBoolConversionError, TypeError):
                pass
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
            # ``unflatten_spec`` is the manifold_spec for the manifold-aware
            # (v2) path and ``None`` for v1 / all-scalar trees (set above
            # alongside the treedef). Threading it onto the result drives the
            # Phase-5 readout: ``components()``, the manifold-aware
            # ``coef_table`` flatten, and positional tangent labels. For v1
            # it is ``None`` so every result-path method takes the v1 branch
            # bitwise (R5/R10/R28).
            manifold_spec=unflatten_spec,
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
        Starting parameters as a ``@jdc.pytree_dataclass``. Scalar (0-d)
        fields are estimated on the (Euclidean) real line as in v1. **The
        parameter geometry is declared here, not via an ``estimate()``
        argument:** a non-scalar or constrained leaf is expressed by wrapping
        its array in ``ManifoldLeaf(array, manifold)`` (e.g.
        ``ManifoldLeaf(A, PSDFixedRank(n, K))`` for a rank-``K`` PSD block,
        ``ManifoldLeaf(v, Euclidean(d))`` for a vector). A tree containing any
        such leaf auto-routes to ``RiemannianLM`` and yields gauge-aware
        inference; mix scalar and ``ManifoldLeaf`` fields freely (the product
        geometry is the pytree itself). The user's dataclass type is preserved
        in the returned ``EstimationResult.theta_hat``.
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
