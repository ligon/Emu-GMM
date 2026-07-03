"""Public-facing protocols and result types for emu-gmm.

This module defines the contracts that concrete implementations must
honour:

- :class:`Measure` --- integration over a (possibly empirical) measure.
- :class:`CovarianceStrategy` --- variance-of-the-moment-estimator.
- :class:`WeightingStrategy` --- whitening / weighting policy.
- :class:`RegularizationStrategy` --- adaptive PD-restoration.
- :class:`Optimizer` --- NLLS solver callable.

Plus the dataclasses that :func:`emu_gmm.estimate` returns:

- :class:`OptimizerInfo` --- backend-specific solver info.
- :class:`Diagnostics` --- numerical and labelled per-moment diagnostics.
- :class:`EstimationResult` --- estimate, inference, provenance, labels.

All protocols are ``@runtime_checkable`` so ``isinstance(impl, Protocol)``
works in user code. :class:`OptimizerInfo` is a
``@jdc.pytree_dataclass`` so that the ``(theta_opt, info)`` tuple an
:class:`Optimizer` returns can be threaded through ``jax.jit`` and
``jax.vmap`` --- this is what the ``optimistix_lm`` adapter advertises.
The two string-typed fields (``status``, ``backend``) ride on the
pytree treedef as ``jdc.static_field()``. :class:`Diagnostics` and
:class:`EstimationResult` remain plain :func:`dataclasses.dataclass`
instances: they are constructed once at the end of an estimation and
not threaded through ``jit`` boundaries directly (the surrounding
:func:`emu_gmm.estimate` returns scalar fields, not the result object,
to anything inside a jit boundary).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pandas as pd
from jaxtyping import Array, Float

from emu_gmm._internal.labels import LabelContext as LabelContext  # re-export
from emu_gmm._internal.labels import tangent_basis_names

# ``flatten_params`` is the v1 scalar-only flatten; ``flatten_params_with_spec``
# is the manifold-aware ambient flatten. ``coef_table`` routes through the
# latter when a non-scalar ``manifold_spec`` is present on the result (Phase 5,
# manifold epic #12) and falls back to the former for v1 / all-scalar trees so
# the v1 output is bitwise unchanged.
from emu_gmm._internal.params import flatten_params, flatten_params_with_spec

# A user's parameter PyTree: typically a @jdc.pytree_dataclass. We use
# Any in the protocol signatures because users define their own types;
# the framework only assumes the value is a valid JAX PyTree.
ParamsLike = Any


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class Emu_GMM_DimensionError(ValueError):
    """Raised when ``estimate()`` is given a degenerate problem dimension.

    The framework requires ``M >= 1`` moments, ``K >= 1`` parameters, and
    ``M >= K`` (no under-identified problems). Each of the three failures
    has a distinct silent-fail mode in lower layers --- empty array
    operations, ``jnp.stack`` of an empty list, or rank-deficient
    inversion producing inf/nan ``Sigma_theta`` --- and surfacing them
    as a typed error at the top of :func:`emu_gmm.estimate` lets users
    distinguish "I mis-specified my model" from "I hit a JAX edge case".
    """


# A StructuralModel is any callable taking (x, theta) and returning
# either a plain (M,) JAX array or a haliax NamedArray with a Moments
# axis. The label adapter handles both.
StructuralModel = Callable[[Float[Array, " D"], ParamsLike], Any]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Measure(Protocol):
    """Integration operator: ``E_mu[psi]`` and its Jacobian.

    Implementations live in :mod:`emu_gmm.measures`.
    """

    def expectation(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, " M"]: ...

    def jacobian(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, "M K"]: ...


@runtime_checkable
class CovarianceStrategy(Protocol):
    """Constructor for V_mu(theta), the variance of the moment estimator.

    Implementations live in :mod:`emu_gmm.covariance`.
    """

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Measure,
    ) -> Float[Array, "M M"]: ...


@runtime_checkable
class WeightingStrategy(Protocol):
    """Whitening / weighting policy.

    Returns ``y = L^{-1} m`` where ``V = L L^T``, with the strategy
    deciding whether ``L`` is recomputed per call (CU) or held fixed
    (Identity, Fixed). Implementations live in :mod:`emu_gmm.weighting`.

    Outer-loop hook
    ---------------
    Most weightings (Identity / Fixed / CU) feed straight into the
    residual function and ride the standard
    :class:`Optimizer` path inside :func:`emu_gmm.estimate`. The
    :class:`~emu_gmm.weighting.IteratedWeighting` strategy, by contrast,
    requires an *outer* Python-level loop alternating Fixed-weight inner
    solves with variance refreshes; the estimator dispatches to that
    custom driver when ``requires_outer_loop`` is ``True``.

    The :attr:`requires_outer_loop` flag and the
    :meth:`outer_loop_driver` method are *optional* extensions to the
    protocol. Any strategy that omits them is treated as a standard
    residual-path strategy (``requires_outer_loop = False``). Third-party
    authors of custom outer-loop weightings should set
    ``requires_outer_loop = True`` and implement
    :meth:`outer_loop_driver` with the signature documented on
    :class:`~emu_gmm.weighting.IteratedWeighting.outer_loop_driver`.
    """

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]: ...


@runtime_checkable
class RegularizationStrategy(Protocol):
    """Adaptive PD-restoration on V.

    Returns ``(V_star, tau)``: the regularised matrix and the realised
    ridge magnitude (for diagnostics). ``tau`` may be a Python float or
    a JAX scalar; the inference engine converts to a Python float when
    building the :class:`Diagnostics` record. Implementations live in
    :mod:`emu_gmm.regularization`.
    """

    def apply(
        self, V: Float[Array, "M M"]
    ) -> tuple[Float[Array, "M M"], Float[Array, ""]]: ...


@runtime_checkable
class Optimizer(Protocol):
    """Non-linear least-squares solver callable.

    Solves ``min_theta || residual_fn(theta) ||^2`` from a starting
    point. Implementations live in :mod:`emu_gmm.optimizer`.

    Optional ``args`` channel (#124)
    --------------------------------
    The built-in optimisers additionally accept a keyword-only
    ``args=None``: when supplied, ``residual_fn`` is a TWO-argument
    kernel ``residual_fn(theta, args)`` and ``args`` is an arbitrary
    traced pytree (the estimator threads the measure through it so
    fresh same-structure data reuses one trace). Third-party
    optimisers may ignore this channel entirely: the estimator probes
    for a keyword-capable ``args`` parameter (``inspect.Parameter.kind``
    aware) and serves two-argument optimisers via the legacy closure
    path. An optimiser that *declares* ``args`` must actually pass it
    through to ``residual_fn`` -- declaring and dropping it would
    evaluate the kernel without data and fail loudly.
    """

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Float[Array, " K"],
    ) -> tuple[Float[Array, " K"], "OptimizerInfo"]: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class OptimizerInfo:
    """Backend-specific info from one optimiser run.

    Registered as a JAX PyTree via :func:`jax_dataclasses.pytree_dataclass`
    so that ``(theta_opt, info)`` tuples returned by an :class:`Optimizer`
    are valid JAX values --- traceable through :func:`jax.jit` and
    :func:`jax.vmap`. The ``steps`` and ``final_objective`` fields are
    traced (typically 0-d JAX scalars under jit, Python ints/floats
    eagerly); ``status`` and ``backend`` are strings and so are marked
    as static fields, mirroring the pattern in
    :mod:`emu_gmm.measures.synthetic` for callable / hyperparameter
    fields.
    """

    steps: Any  # traced; typically int (eager) or 0-d JAX int array (jit)
    final_objective: Any  # traced; Python float (eager) or 0-d JAX float (jit)
    status: str = jdc.static_field()  # type: ignore[attr-defined]
    backend: str = jdc.static_field()  # type: ignore[attr-defined]
    # #78: the optimiser's REAL convergence flag (traced 0-d bool array).
    # Defaults to ``None`` so every pre-existing backend that omits it stays
    # backward-compatible: the estimator falls back to the status string
    # when ``done is None`` and uses the traced ``done`` when supplied (the
    # Riemannian-LM path). Carried as a traced child of the PyTree.
    done: Any = None

    # #152 (Riemannian trust region): ADDITIVE, backward-compatible fields.
    # Every pre-existing backend leaves these at their ``None`` default, so
    # nothing about the LM / optimistix / scipy paths changes (they remain
    # traced children of the pytree, but ``None`` rides as an empty subtree).
    #
    # ``final_gradient_norm`` is the HORIZONTAL Riemannian gradient norm at
    # the returned iterate (gauge-invariant); RTR surfaces it so a caller can
    # read the stationarity certificate without re-deriving it. Traced 0-d.
    final_gradient_norm: Any = None
    # ``max_tcg_steps`` / ``max_radius`` are the RESOLVED tCG / trust-radius
    # defaults the run actually used (derived from the intrinsic quotient
    # dimension, not the ambient nk). Surfaced for the intrinsic-dimension
    # contract (#152). Static-ish ints/floats; carried traced for jit safety.
    max_tcg_steps: Any = None
    max_radius: Any = None
    # ``n_negcurv`` counts the outer steps whose inner tCG reported a
    # negative-curvature exit -- the meta-check that RTR genuinely used its
    # second-order machinery (non-convex navigation), not a vacuous GN path.
    n_negcurv: Any = None
    # ``tr_trace`` is the per-outer-step trace (Delta, rho, rhonum, rho_reg,
    # grad_norm, accepted, stop-reason, conv_threshold, proposed_full_rank).
    # A mapping of stacked 0-d arrays keyed by name; ``None`` for non-RTR
    # backends. Read via the tests' ``_info_get`` helper (which also probes
    # an ``extra`` mapping; we expose the same dict under both names).
    tr_trace: Any = None
    extra: Any = None

    # #152 advisory (Riemannian-LM curvature probe): set ONLY on the EAGER
    # ``riemannian_lm`` path, and only when a gauge-bearing solve converges to a
    # genuine STATIONARY point (a #156 ftol cost-stagnation stop drifts at large
    # gradient and is excluded). ``min_curvature`` is the smallest eigenvalue of
    # the horizontal true Hessian at the optimum (the ``k(k-1)/2`` gauge
    # directions map to ~0 under the projected HVP, so they do NOT masquerade as
    # curvature); ``stalled_indefinite`` is the Python bool
    # ``min_curvature < -tol`` -- True flags a saddle where ``riemannian_tr`` may
    # do better. Both stay ``None`` on every other backend, on the
    # vmapped/replicate MC path (the probe is eager-only -- warnings cannot fire
    # inside ``vmap``), and at non-stationary (ftol-drift) stops, by design.
    stalled_indefinite: Any = None
    min_curvature: Any = None


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """Numerical diagnostics from one estimation run.

    Scalar fields capture the regularisation choice and convergence
    metrics. They are stored as 0-d JAX arrays so that ``estimate()``
    traces through ``jit`` and ``vmap``; ``float(diagnostic_field)`` at
    the eager boundary recovers a Python scalar. Labelled fields
    (``N_j``, ``moment_residual``) carry moment-axis coordinates and are
    usable in pandas-style inspection via
    :meth:`EstimationResult.to_pandas`.

    The ``cond_info`` and ``optimizer_health`` dicts surface the
    Hessian condition trio and lightweight optimiser-health metrics
    discussed in issue #10 (parity with ManifoldGMM's
    ``compute_hessian_cond`` / ``optimizer_health``).
    """

    # Regularisation
    tau_realised: Float[Array, ""]
    kappa_V: Float[Array, ""]
    binding_ridge: Any  # 0-d bool JAX array

    # Cholesky
    cholesky_pivot_min: Float[Array, ""]

    # Optimisation.
    #
    # ``final_objective_data`` is the *data-only* criterion value
    # :math:`Q_\\mu(\\hat\\theta) = \\|L^{-1}\\, m_\\mu(\\hat\\theta)\\|^2`
    # at the optimum --- equivalently :math:`J_{\\mathrm{stat}}`. This is
    # what the GMM literature and ``J_stat`` use. Reported regardless of
    # whether a :class:`PenaltyStrategy` is supplied.
    #
    # ``final_objective_full`` is the *full* criterion the optimiser
    # actually minimised, including any in-objective penalty contribution
    # :math:`p(\\hat\\theta)` (issue #7 hook). With ``penalty=None`` it
    # equals ``final_objective_data``; with a penalty supplied it is
    # strictly :math:`\\geq` ``final_objective_data``.
    #
    # Note that :data:`OptimizerInfo.final_objective` reports the
    # *half* norm :math:`\\tfrac{1}{2}\\|r\\|^2` (the standard NLLS
    # convention for the LM-minimised value), so under a penalty
    # ``optimizer_info.final_objective == 0.5 * final_objective_full``.
    # ``final_objective_full`` is reported *unhalved* so it stays on
    # the same scale as ``J_stat`` and ``final_objective_data``.
    #
    # ``final_objective`` is retained as an alias for
    # ``final_objective_data`` for backwards compatibility with code
    # written against earlier versions; new code should prefer the
    # explicit split.
    #
    # ``final_gradient_norm`` is :math:`\\|\\nabla_\\theta
    # \\tfrac{1}{2} \\|r(\\hat\\theta)\\|^2\\|` where ``r`` is the
    # *residual the optimiser saw*. When ``penalty`` is supplied this
    # includes the penalty contribution from the appended
    # :math:`\\sqrt{p(\\theta)}` row (so the reported norm matches the
    # convergence criterion the LM solver actually used); when
    # ``penalty=None`` it is the unpenalised data-only gradient norm.
    final_objective: Float[Array, ""]
    final_gradient_norm: Float[Array, ""]

    # Labelled per-moment
    N_j: ha.NamedArray  # axis [Moments]
    moment_residual: ha.NamedArray  # axis [Moments]; m_hat at theta_hat

    # Provenance
    optimizer_info: OptimizerInfo

    # Split of the optimised criterion into data-only and full
    # components. With ``penalty=None`` both equal ``final_objective``;
    # with a penalty supplied ``final_objective_data == J_stat`` while
    # ``final_objective_full == J_stat + p(theta_hat)``. Defaults to
    # NaN so the dataclass can be constructed from older callsites that
    # only supply ``final_objective``; the framework's
    # ``build_diagnostics`` always populates them explicitly.
    final_objective_data: Float[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.asarray(jnp.nan)
    )
    final_objective_full: Float[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.asarray(jnp.nan)
    )

    # Hessian condition trio at theta_hat. See ``docs/design.org`` and
    # CLAUDE.md commitment 5: the information matrix is ``G' Lambda G``
    # (never numerical Hessian); ``cond_info`` reports the condition
    # number of that matrix.
    #
    # Keys:
    #   - ``'raw'``: cond(G' Lambda G + (1/2) H_p) at theta_hat with
    #     Lambda = (V*)^{-1}. When no :class:`PenaltyStrategy` is
    #     supplied H_p == 0 and this reduces to cond(G' Lambda G).
    #   - ``'data_only'``: cond(G' Lambda G) with the penalty Hessian
    #     contribution excluded. This is the asymptotic-inference
    #     identifier (delta-method variance is built from the data
    #     Hessian alone) and what you want when the penalty is a
    #     stabiliser rather than a prior.
    #   - ``'exclude_gauge'``: the gauge-aware QUOTIENT condition
    #     number (#20). For a gauge-bearing tree (PSDFixedRank(n, K):
    #     gauge_dim = K(K-1)/2) the information matrix carries exactly
    #     ``gauge_nullspace_dim`` spectrally-zero directions by
    #     construction, so ``'raw'`` is meaningless for K >= 2; this
    #     entry is cond over the spectrum EXCLUDING the gauge_dim
    #     smallest eigenvalues BY COUNT (the pinv_eigvalrule rule).
    #     Additional near-zeros beyond the dropped count blow it up
    #     too -- genuine structural rank-deficiency stays visible.
    #     For gauge_nullspace_dim == 0 (v1 / all-Euclidean) it is the
    #     bitwise alias of ``'raw'``, as before.
    cond_info: dict[str, float] = dataclasses.field(default_factory=dict)

    # Lightweight optimiser-health summary at termination. Keys:
    #   - ``'iters'``: iteration / step count
    #     (mirrors ``optimizer_info.steps``).
    #   - ``'grad_norm'``: ``||grad (1/2) ||r||^2||`` at theta_hat,
    #     where ``r`` is the *full* residual vector the optimiser saw.
    #     When ``penalty`` is supplied this includes the penalty
    #     contribution from the appended ``sqrt(p+eps)`` row; when
    #     ``penalty=None`` it reduces to ``||grad (1/2) ||y||^2||``.
    #     Mirrors ``final_gradient_norm``.
    #   - ``'step_norm'``: norm of the last accepted step, if the
    #     backend exposes it; otherwise ``None``.
    #   - ``'accepted_step_count'``: number of accepted (vs rejected)
    #     LM steps, if the backend exposes it; otherwise ``None``.
    optimizer_health: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Gauge nullspace dimension of the parameter manifold (Phase 4, #12).
    # Equal to the manifold spec's ``total_gauge_dim``: the number of
    # exact-zero gauge directions of the information matrix. ``0`` for
    # every Euclidean / scalar-Positive (v1) tree; ``k(k-1)/2`` for a
    # ``PSDFixedRank(n, k)`` leaf (the tangent dimension of the O(k) gauge
    # group). ``Sigma_theta`` is reported at rank ``total_dimension -
    # gauge_nullspace_dim``: the ``gauge_nullspace_dim`` smallest
    # eigenvalues of the info matrix are pinned to exact zero by
    # ``pinv_eigvalrule``, so users can distinguish these *expected* gauge
    # zeros from genuine near-zero (weakly-identified) directions.
    gauge_nullspace_dim: int = 0
    #: #138 (diagnose-loudly policy): True when the weighting-aware
    #: sandwich produced a NEGATIVE diagonal entry in ``Sigma_theta`` --
    #: i.e. the unregularized meat ``G' Lambda V Lambda G`` was
    #: indefinite, which happens when the raw ``V(theta_hat)`` is itself
    #: indefinite (the binding-ridge regime, #111/#133). The affected
    #: ``standard_errors`` entries are NaN BY DESIGN (the honest answer
    #: when V is not a covariance matrix); use the bootstrap for SEs in
    #: this regime, and see the studies summarizers' ``n_valid_se``
    #: accounting (#140; see design.org). Traced 0-d bool under jit,
    #: Python bool eagerly. Recorded per-rep as the stackable 0/1 float
    #: ``FitRecord.sigma_meat_indefinite`` (#143), so the event is
    #: auditable in committed MC records.
    sigma_meat_indefinite: Any = False
    #: Diagnose-loudly policy: True when the FINAL regularised covariance
    #: ``V*`` at ``theta_hat`` is not strictly positive-definite --- i.e.
    #: the diagonal-ridge family could not repair ``V`` and the tau
    #: bisection saturated at its upper bound (e.g. an exactly-zero
    #: diagonal entry from a zero-support moment: ``(1 + tau) * 0 == 0``
    #: at every tau). Downstream ``cholesky(V*)`` returns NaN BY DESIGN,
    #: so the criterion, ``J_stat``, and the standard errors are expected
    #: NaN. Inspect ``Diagnostics.N_j`` (an all-zero-support moment is
    #: the common cause) and ``tau_realised``. NaN eigenvalues of ``V*``
    #: count as flagged. Traced 0-d bool under jit, Python bool eagerly.
    v_star_indefinite: Any = False


def _is_non_scalar_spec(manifold_spec: Any) -> bool:
    """True iff any leaf of ``manifold_spec`` is non-scalar (ambient ndim>0).

    For an all-Euclidean / scalar-Positive (v1) tree every leaf has
    ``ambient_shape == ()`` so this is ``False`` and every result-path
    method takes the bitwise-v1 branch (R5/R28). ``None`` (no spec threaded:
    a v1 estimate) is also ``False``.
    """
    if manifold_spec is None:
        return False
    return any(
        tuple(int(s) for s in ls.ambient_shape) != () for ls in manifold_spec.leaf_specs
    )


def _walk_components(theta_hat: Any) -> tuple[Any, ...]:
    """Per-leaf array tuple of ``theta_hat`` in dataclass field order.

    Treats :class:`~emu_gmm.manifolds.manifold_leaf.ManifoldLeaf` as opaque
    so each wrapped block is one leaf, unwrapping it to its raw ambient
    ``array``; bare (scalar) leaves are returned verbatim. The ordering is
    the PyTree leaf-walk order, which for a flat ``@jdc.pytree_dataclass``
    is declaration / field order --- the same order
    :func:`flatten_params_with_spec` and ``manifold_spec.leaf_specs`` use,
    so ``components()`` round-trips a warm start (Phase 5, #12).
    """
    from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf

    leaves = jax.tree_util.tree_leaves(
        theta_hat, is_leaf=lambda x: isinstance(x, ManifoldLeaf)
    )
    return tuple(
        leaf.array if isinstance(leaf, ManifoldLeaf) else leaf for leaf in leaves
    )


@dataclasses.dataclass(frozen=True)
class ManifoldPoint:
    """Thin, inspectable view of an estimated parameter PyTree (Phase 5, #12).

    Wraps the raw ``theta_hat`` PyTree (left untouched: ``result.theta_hat``
    is still the user's dataclass, R19) and exposes the K-Aggregators
    structural readout contract:

    * :meth:`components` returns the per-leaf ambient array tuple
      ``(A, phi, ...)`` in dataclass field order, so a caller computes
      ``Gamma_hat = A @ A.T``, ``theta = exp(phi)``, ``eigvalsh(Gamma)``;
    * a warm start reads ``prev.theta.components()`` straight back into a
      new ``estimate()``'s ``theta_init`` (it round-trips, since the tuple
      is in leaf-walk order).

    Pure and immutable: :meth:`components` returns the same per-leaf arrays
    (identity-stable) on every call. This object is **not** a JAX PyTree and
    never replaces ``theta_hat`` in the tree; it is a readout convenience.
    """

    theta_hat: Any
    manifold_spec: Any = None

    def components(self) -> tuple[Any, ...]:
        """Per-leaf ambient array tuple in dataclass field order."""
        return _walk_components(self.theta_hat)


@jdc.pytree_dataclass
class FitRecord:
    """Slim, stackable per-fit summary pytree (#125).

    The atom of repeated estimation: the canonical batching gesture is

    .. code-block:: python

       records = [fit_record(run(theta0, dgp(jax.random.fold_in(key, r))))
                  for r in range(n_reps)]
       stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)

    after which every field carries a leading replication axis and the
    Monte Carlo summarizers (bias, MC SD, coverage, size/power, J
    calibration) are cheap reductions. Registered as a JAX pytree so it
    also rides ``jit`` / ``vmap`` (the "inference results are pytrees"
    invariant; api-sketch.org Section 3) --- :class:`EstimationResult`
    itself deliberately remains a host-side leaf, and this is its
    derived, traced-world projection.

    ``theta_flat`` / ``se`` are on the **ambient tangent axis** (length
    ``total_dimension``): for v1 / all-scalar parameters this is the
    dataclass field order; for manifold parameters it is the
    manifold-aware ambient flatten that ``Sigma_theta`` is sized by, in
    which case the raw per-entry values of a gauge-bearing leaf are
    gauge-arbitrary --- compute invariant functionals via
    :meth:`EstimationResult.functional_se` instead. ``param_names``
    rides on the treedef as a static field (the
    ``ClusterBootstrapResult`` pattern), so stacking across fits with
    identical parameter structure needs no configuration.

    ``sigma_meat_indefinite`` is the #138 diagnose-loudly event as a
    stackable 0/1 float (the ``binding_ridge`` pattern): ``1.0`` when
    the sandwich meat was indefinite at this fit, i.e. the rep whose
    ``se`` entries are NaN BY DESIGN. Carrying it here (#143) makes the
    NaN-SE event auditable in committed MC records rather than only on
    the live ``EstimationResult.diagnostics``.
    """

    theta_flat: Float[Array, " D"]
    se: Float[Array, " D"]
    J_stat: Float[Array, ""]
    J_pvalue: Float[Array, ""]
    J_pvalue_adjusted: Float[Array, ""]
    converged: Float[Array, ""]  # 0/1; jnp.stack-able and mean()-able
    tau_realised: Float[Array, ""]
    binding_ridge: Float[Array, ""]  # 0/1
    sigma_meat_indefinite: Float[Array, ""]  # 0/1; the #138 NaN-SE event
    J_dof: int = jdc.static_field()  # type: ignore[attr-defined]
    param_names: tuple[str, ...] = jdc.static_field()  # type: ignore[attr-defined]


def _result_param_names(result: "OptimizationResult") -> tuple[str, ...]:
    """Ambient tangent labels of ``result.theta_hat`` (length ``D``).

    Manifold-aware and robust to a missing :class:`LabelContext` (a
    hand-built artificial result): non-scalar spec -> positional tangent
    labels; a label context -> its parameter names; otherwise positional
    ``theta_i`` sized by the flattened estimate. The single source of truth
    for the ambient axis labels shared by the law and :func:`fit_record`.
    """
    spec = result.manifold_spec
    labels = result.labels
    if _is_non_scalar_spec(spec):
        fallback = tuple(labels.param_names) if labels is not None else ()
        return tuple(tangent_basis_names(spec, fallback_param_names=fallback))
    if labels is not None:
        return tuple(labels.param_names)
    flat, _treedef = flatten_params(result.theta_hat)
    n = int(jnp.asarray(flat).shape[0])
    return tuple(f"theta_{i}" for i in range(n))


def _flat_theta(theta_hat: Any, manifold_spec: Any) -> Float[Array, " D"]:
    """The ambient flat parameter vector of ``theta_hat`` (the ``Sigma`` axis).

    Manifold-aware: the ambient (``flatten_params_with_spec``) flatten for a
    non-scalar leaf, the v1 2-tuple ``flatten_params`` for an all-scalar tree
    (bitwise the v1 axis). Shared by the law's ``coef_table`` / ``mean`` and
    :func:`fit_record`.
    """
    if _is_non_scalar_spec(manifold_spec):
        flat, _treedef, _spec = flatten_params_with_spec(theta_hat)
    else:
        flat, _treedef = flatten_params(theta_hat)
    return jnp.asarray(flat)


@dataclasses.dataclass
class OptimizationResult:
    """The output of :func:`emu_gmm.estimate` --- the *optimization* surface.

    An estimator is two things wearing one coat: an *optimization* (find the
    argmin of a criterion and characterise the objective near it) and an
    *inference* (turn that local characterisation into a statement about the
    sampling distribution of the estimator). :class:`OptimizationResult` is the
    first: the argmin, the objective value there, the gradient and the
    Gauss--Newton curvature / moment Jacobian, the optimiser's trace, and the
    numerical conditioning --- objects intrinsic to the optimization, with **no
    statistical interpretation** (docs/optimization-result-law-split.org).

    The inference surface --- ``se`` / ``cov`` / ``coef_table`` / functional
    SEs / the J-test p-values --- lives on an :class:`~emu_gmm.law.EstimatorLaw`
    that *names the assumption it adds*. Get the strong-assumption (Gaussian /
    CLT) grade via :meth:`asymptotic` (``AsymptoticLaw(result)``); it assembles
    the ridge-correct #133 sandwich from the ingredients on this result
    (:attr:`moment_jacobian`, :attr:`weighting_matrix`, :attr:`moment_covariance`).

    Directly constructible: every field but :attr:`theta_hat` has a default, so
    a test can hand-build an artificial result
    (``OptimizationResult(theta_hat=..., moment_jacobian=G,
    weighting_matrix=Lambda, moment_covariance=V, manifold_spec=...)``) and feed
    it to :class:`~emu_gmm.law.AsymptoticLaw` to unit-test the inference
    assembly in isolation.

    ``EstimationResult`` is a deprecated back-compat alias for this class.
    """

    # Estimate (in the user's parameter dataclass type)
    theta_hat: Any

    # -- optimization scalars / vectors ----------------------------------
    #: ``Q(theta_hat) = m_bar' Lambda m_bar`` at the optimum (was ``J_stat``).
    #: A 0-d JAX array so ``estimate`` is jit / vmap compatible;
    #: ``float(result.objective_value)`` outside trace recovers a Python scalar.
    objective_value: Any = None
    #: ``grad Q(theta_hat)`` --- the FOC residual vector at convergence, on the
    #: ambient tangent axis (length ``D``). ``None`` on a hand-built result.
    gradient: Any = None
    #: ``G = d m_bar / d theta`` at theta_hat (the horizontal-projected moment
    #: Jacobian ``G_riem`` for a manifold parameter), shape ``(M, D)``. The
    #: sandwich's outer factor; the Law reads it for the bread and meat.
    moment_jacobian: Any = None
    #: ``B = G' Lambda G`` --- the Gauss--Newton curvature of the criterion
    #: (the information matrix; was ``info_matrix``), shape ``(D, D)``. A
    #: curvature object (result-side); ``identification_strength`` reads it.
    gn_hessian: Any = None
    #: ``Lambda`` --- the realised, frozen, ridged metric in ``m_bar' Lambda
    #: m_bar`` the objective used, shape ``(M, M)``. ``V* = Lambda^{-1}`` is the
    #: regularised moment covariance the fit whitened by.
    weighting_matrix: Any = None
    #: The RAW (unregularised) moment covariance ``V`` at theta_hat --- the
    #: sandwich *meat*, shape ``(M, M)``. Storing it (rather than recomputing
    #: from the model) keeps the result self-contained and directly
    #: constructible.
    moment_covariance: Any = None
    #: ``M - p_id`` --- the number of overidentifying restrictions (was
    #: ``J_dof``). A static Python int.
    n_overid: int = 0

    # -- optimiser status / provenance -----------------------------------
    #: ``converged`` is a Python bool derived from the optimiser's discrete
    #: status enum (or the sentinel ``"traced"`` under jit/vmap).
    converged: Any = True
    #: Whatever the backend supplied: a Python int from SciPy, a 0-d JAX int
    #: array from optimistix (so it traces under jit).
    iterations: Any = 0

    # Provenance (echoed from the call site)
    theta_init: Any = None
    measure: Measure | None = None
    covariance: CovarianceStrategy | None = None
    weighting: WeightingStrategy | None = None
    regularization: RegularizationStrategy | None = None

    # Diagnostics (optimization conditioning only)
    diagnostics: Diagnostics | None = None

    # Labels collected during input normalisation.
    labels: LabelContext | None = None

    # Manifold metadata describing ``theta_hat``'s leaf geometry (Phase 5,
    # manifold epic #12). ``None`` for a v1 / all-scalar estimate; a
    # :class:`emu_gmm.manifolds.spec.ManifoldSpec` (== the estimate's
    # ``unflatten_spec``) for the manifold-aware path. Drives the
    # ``components()`` readout, the law's manifold-aware ``coef_table`` flatten,
    # and the positional tangent labels (INT-12/R5).
    manifold_spec: Any = None

    def asymptotic(self) -> Any:
        """The asymptotic :class:`~emu_gmm.law.EstimatorLaw` for this fit.

        Adds the asymptotic *assumption* (a CLT / exact limiting behaviour) to
        this optimization result and returns an
        :class:`~emu_gmm.law.AsymptoticLaw` that assembles the ridge-correct
        sandwich from the result's ingredients. Convenience for
        ``AsymptoticLaw(result)``; the statistical surface (``se`` / ``cov`` /
        ``coef_table`` / functional SEs / ``j_test``) lives on the returned law,
        not on the result --- an :class:`OptimizationResult` foregoes any
        statistical interpretation (docs/optimization-result-law-split.org).
        """
        from emu_gmm.law import AsymptoticLaw

        return AsymptoticLaw(self)

    @property
    def theta(self) -> ManifoldPoint:
        """Inspectable view of ``theta_hat`` exposing :meth:`components`.

        ``result.theta.components()`` returns the per-leaf ambient array
        tuple ``(A, phi, ...)`` in dataclass field order (Phase 5, #12);
        ``result.theta_hat`` remains the raw user dataclass (R19). Equivalent
        to :meth:`components`.
        """
        return ManifoldPoint(self.theta_hat, self.manifold_spec)

    def components(self) -> tuple[Any, ...]:
        """Per-leaf ambient array tuple of ``theta_hat`` in field order.

        For a ``Product(PSDFixedRank(n, k), Euclidean(1))`` estimate this is
        ``(A, phi)`` with ``A.shape == (n, k)`` and ``phi`` shape ``(1,)``;
        callers compute ``Gamma_hat = A @ A.T`` and ``eigvalsh(Gamma_hat)``.
        For a v1 / all-scalar tree it is the tuple of 0-d scalar leaves in
        leaf-walk order. A warm start feeds ``prev.components()`` back into
        a new ``estimate()`` (it round-trips). Convenience alias for
        ``self.theta.components()``.
        """
        return _walk_components(self.theta_hat)

    def to_pandas(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Materialise the optimization's labelled fields as pandas objects.

        The *statistical* surface (``coefficients`` / ``Sigma_theta`` / ``V_X``)
        moved to the :class:`~emu_gmm.law.AsymptoticLaw`: use
        ``result.asymptotic().coef_table`` and ``.cov()``. What remains here is
        the optimization readout --- the labelled per-moment diagnostics and a
        scalar summary of the fit:

        - ``"N_j"``: :class:`pandas.Series` indexed by moment names.
        - ``"moment_residual"``: :class:`pandas.Series` indexed by moment names.
        - ``"summary"``: :class:`pandas.Series` of scalar optimization fields
          (``objective_value``, ``n_overid``, ``converged``, ``iterations``,
          ``tau_realised``, ``kappa_V``, ``final_objective``).
        """
        labels = self.labels
        assert labels is not None
        assert self.diagnostics is not None
        moment_names = list(labels.moment_names)

        n_j = pd.Series(
            jnp.asarray(self.diagnostics.N_j.array),
            index=moment_names,
            name="N_j",
        )
        m_res = pd.Series(
            jnp.asarray(self.diagnostics.moment_residual.array),
            index=moment_names,
            name="moment_residual",
        )
        # Summary is the eager-only consumer boundary: cast 0-d JAX
        # arrays to Python floats so the resulting Series is ergonomic.
        summary = pd.Series(
            {
                "objective_value": float(jnp.asarray(self.objective_value)),
                "n_overid": int(self.n_overid),
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "tau_realised": float(jnp.asarray(self.diagnostics.tau_realised)),
                "kappa_V": float(jnp.asarray(self.diagnostics.kappa_V)),
                "final_objective": float(jnp.asarray(self.diagnostics.final_objective)),
                "final_objective_data": float(
                    jnp.asarray(self.diagnostics.final_objective_data)
                ),
                "final_objective_full": float(
                    jnp.asarray(self.diagnostics.final_objective_full)
                ),
            }
        )

        return {
            "N_j": n_j,
            "moment_residual": m_res,
            "summary": summary,
        }

    def _main_namespace_hazards(self) -> list[str]:
        """Names of provenance objects whose classes/callables live in
        ``__main__`` --- the pickle-portability hazard (#23).

        Pickle stores classes and functions *by reference* (module +
        qualname). Anything resolved through ``__main__`` unpickles only
        in a process whose ``__main__`` happens to define the same
        names --- the K-Aggregators scripts ended up installing shim
        attributes on ``__main__`` to work around exactly this. The
        durable fix is to define parameter dataclasses, samplers, and
        closed-form callables in an importable module.
        """
        candidates: list[tuple[str, Any]] = [
            ("theta_hat", type(self.theta_hat)),
            ("theta_init", type(self.theta_init)),
            ("measure", type(self.measure)),
            ("covariance", type(self.covariance)),
            ("weighting", type(self.weighting)),
        ]
        # Callables carried as static fields on the provenance objects
        # (SyntheticMeasure.sampler, AnalyticalMeasure.expectation_fn /
        # jacobian_fn, AnalyticalCovariance.covariance_fn).
        for owner_name, owner in (
            ("measure", self.measure),
            ("covariance", self.covariance),
        ):
            for attr in ("sampler", "expectation_fn", "jacobian_fn", "covariance_fn"):
                fn = getattr(owner, attr, None)
                if callable(fn):
                    candidates.append((f"{owner_name}.{attr}", fn))
        hazards = []
        for name, obj in candidates:
            module = getattr(obj, "__module__", None)
            if module == "__main__":
                hazards.append(name)
        return hazards

    def to_pickle(self, path: Any) -> None:
        """Pickle this result to ``path`` (the K-Aggregators idiom, #23).

        Thin wrapper over :func:`pickle.dump`. Two portability caveats,
        both inherent to pickle rather than to this method:

        - Classes and callables are stored *by reference*: the user's
          parameter dataclass (and any sampler / closed-form callables
          riding the provenance fields) must be importable under the
          same module path at load time. Objects defined in
          ``__main__`` (a script / notebook top level) trigger a
          :class:`UserWarning` here at *save* time --- when you can
          still move them into an importable module --- rather than a
          confusing :class:`AttributeError` at load time in another
          process. Lambdas do not pickle at all and raise immediately.
        - The provenance fields carry the full measure (data arrays
          included), so the file scales with the dataset.

        Parameters
        ----------
        path : str | os.PathLike
            Destination file path; opened in binary-write mode.
        """
        import pickle
        import warnings

        hazards = self._main_namespace_hazards()
        if hazards:
            warnings.warn(
                "OptimizationResult.to_pickle: the following provenance "
                f"objects resolve through __main__: {hazards}. Pickle "
                "stores classes/functions by reference, so this file will "
                "only load in a process whose __main__ defines the same "
                "names. Define parameter dataclasses, samplers, and "
                "closed-form callables in an importable module for a "
                "portable pickle.",
                UserWarning,
                stacklevel=2,
            )
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_pickle(cls, path: Any) -> "OptimizationResult":
        """Load an :class:`OptimizationResult` pickled by :meth:`to_pickle`.

        Thin wrapper over :func:`pickle.load` with a type check, so a
        wrong-file mistake surfaces as a clear :class:`TypeError` rather
        than an :class:`AttributeError` three lines later. The usual
        pickle trust caveat applies: only load files you wrote.

        Parameters
        ----------
        path : str | os.PathLike
            Source file path; opened in binary-read mode.
        """
        import pickle

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"OptimizationResult.from_pickle: {path!r} contains a "
                f"{type(obj).__name__}, not an OptimizationResult. If this "
                "is a ManifoldGMM GMMResult pickle, re-estimate with "
                "emu_gmm (see docs/migration/manifoldgmm-to-emu-gmm.org); "
                "pickles do not migrate across libraries."
            )
        return obj


#: Deprecated back-compat alias for :class:`OptimizationResult`. The
#: estimation/inference split (docs/optimization-result-law-split.org) renamed
#: the class: ``estimate()`` returns the leaner optimization surface and the
#: statistical surface lives on ``result.asymptotic()``. New code should use
#: ``OptimizationResult``; this name is retained so existing imports keep
#: working.
EstimationResult = OptimizationResult


__all__ = [
    "ParamsLike",
    "StructuralModel",
    "Measure",
    "CovarianceStrategy",
    "WeightingStrategy",
    "RegularizationStrategy",
    "Optimizer",
    "OptimizerInfo",
    "Diagnostics",
    "OptimizationResult",
    "EstimationResult",
    "FitRecord",
    "ManifoldPoint",
    "Emu_GMM_DimensionError",
    # Re-exported from emu_gmm._internal.labels so the type that
    # annotates ``EstimationResult.labels`` has a public home (#56).
    "LabelContext",
]
