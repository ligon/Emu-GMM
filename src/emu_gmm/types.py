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
import functools
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pandas as pd
import scipy.stats
from jaxtyping import Array, Float

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal.labels import LabelContext as LabelContext  # re-export
from emu_gmm._internal.labels import label_vector, tangent_basis_names

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
    #   - ``'exclude_gauge'``: alias to ``'raw'`` for v1. Once the v2
    #     manifold support lands, this will project out the
    #     K*(K-1)/2 PSDFixedRank gauge nullspace before computing the
    #     condition number.
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

       records = [run(theta0, dgp(jax.random.fold_in(key, r))).record()
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
    """

    theta_flat: Float[Array, " D"]
    se: Float[Array, " D"]
    J_stat: Float[Array, ""]
    J_pvalue: Float[Array, ""]
    J_pvalue_adjusted: Float[Array, ""]
    converged: Float[Array, ""]  # 0/1; jnp.stack-able and mean()-able
    tau_realised: Float[Array, ""]
    binding_ridge: Float[Array, ""]  # 0/1
    J_dof: int = jdc.static_field()  # type: ignore[attr-defined]
    param_names: tuple[str, ...] = jdc.static_field()  # type: ignore[attr-defined]


@dataclasses.dataclass
class EstimationResult:
    """The output of :func:`emu_gmm.estimate`.

    Holds the estimate, inference quantities, provenance, and a label
    context for materialising labelled outputs into pandas.
    """

    # Estimate (in the user's parameter dataclass type)
    theta_hat: Any
    # Asymptotic covariance, axes [Params, ParamsDual]
    Sigma_theta: ha.NamedArray
    # Variance at theta_hat, axes [Moments, MomentsDual]
    V_X: ha.NamedArray

    # J-test. ``J_stat`` and ``J_pvalue`` are 0-d JAX arrays so
    # ``estimate`` is jit / vmap compatible; ``float(result.J_stat)``
    # outside trace recovers a Python scalar. ``J_dof`` is a static int.
    #
    # ``J_pvalue`` is the *nominal* chi^2_{M-K} survival function value.
    # ``J_pvalue_adjusted`` is the regularisation-adjusted survival
    # function under the weighted-chi^2 limit of
    # mcar-asymptotics.org Theorem 6; it equals ``J_pvalue`` when the
    # ridge is not binding (tau <= tau_threshold) and differs (via a
    # Welch-Satterthwaite approximation to the generalised chi-squared)
    # when it binds.
    J_stat: Float[Array, ""]
    J_dof: int
    J_pvalue: Float[Array, ""]
    J_pvalue_adjusted: Float[Array, ""]

    # Status. ``converged`` is a Python bool derived from the optimiser's
    # discrete status enum (or the sentinel ``"traced"`` under jit/vmap).
    # ``iterations`` is whatever the backend supplied: a Python int from
    # SciPy, a 0-d JAX int array from optimistix (so it traces under jit).
    converged: bool
    iterations: Any

    # Provenance (echoed from the call site)
    theta_init: Any
    measure: Measure
    covariance: CovarianceStrategy
    weighting: WeightingStrategy
    regularization: RegularizationStrategy | None

    # Diagnostics
    diagnostics: Diagnostics

    # Labels collected during input normalisation; threads through to
    # to_pandas() for DataFrame construction.
    labels: LabelContext

    # Manifold metadata describing ``theta_hat``'s leaf geometry (Phase 5,
    # manifold epic #12). ``None`` for a v1 / all-scalar estimate; a
    # :class:`emu_gmm.manifolds.spec.ManifoldSpec` (== the estimate's
    # ``unflatten_spec``) for the manifold-aware path. Drives the
    # ``components()`` readout, the manifold-aware ``coef_table`` flatten,
    # and the positional tangent labels (so a non-scalar leaf's ambient
    # coordinates are NOT mislabelled as scalar field-names; INT-12/R5).
    # Defaults to ``None`` so every existing v1 callsite constructs the
    # result unchanged.
    manifold_spec: Any = None

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

    @functools.cached_property
    def standard_errors(self) -> ha.NamedArray:
        """Asymptotic standard errors of ``theta_hat``.

        Computed as ``sqrt(diag(Sigma_theta))``, returned as a
        :class:`haliax.NamedArray` on the ``parameters`` axis so
        downstream code can index by parameter name. Cached on first
        access.

        Negative diagonal entries (which can arise in pathological
        finite-sample regimes when ``info_matrix`` was numerically
        non-PD) propagate as :data:`numpy.nan` rather than complex
        values, matching the convention in :func:`numpy.sqrt` with
        ``where=arr>=0``.
        """
        diag = jnp.diag(jnp.asarray(self.Sigma_theta.array))
        # Guard against tiny negatives from finite-precision round-off:
        # clip exactly to 0 before sqrt.
        se = jnp.sqrt(jnp.where(diag >= 0.0, diag, jnp.nan))
        Params = axes_mod.params_axis(int(se.shape[0]))
        return label_vector(se, Params)

    def functional_se(
        self, f: Callable[[tuple[Any, ...]], Any]
    ) -> tuple[Float[Array, " p"], Float[Array, "p p"]]:
        r"""Gauge-invariant delta-method SE / covariance of ``f(components)``.

        The general Phase-7 (#42) primitive. For a functional
        ``f(components) -> R^p`` that depends on ``theta_hat`` only through
        gauge invariants, returns ``(se, cov)`` where
        ``cov = J_f @ Sigma_theta @ J_f.T`` (the delta method) and
        ``se = sqrt(diag(cov))``. ``J_f`` is the Jacobian of ``f`` w.r.t. the
        ambient flat parameter, taken at the **fixed** ``theta_hat`` (never
        through the solver; commitment 5).

        Parameters
        ----------
        f
            A JAX-AD-able callable mapping the components tuple
            ``(A, phi, ...)`` -- exactly what :meth:`components` returns, same
            order -- to a 1-D JAX array of length ``p`` (a scalar is treated
            as ``p == 1``). ``f`` must be **gauge-invariant**: it must depend
            on each gauge-bearing leaf (a ``PSDFixedRank`` factor ``A``) only
            through that leaf's gauge invariants (e.g. ``Gamma = A @ A.T``).

            Example::

                def gamma00(comps):
                    A, phi = comps
                    return (A @ A.T)[0, 0]      # one Gamma entry, gauge-inv.

                se, cov = result.functional_se(gamma00)

        Returns
        -------
        se : ``(p,)`` ``jax`` array
            ``sqrt(diag(cov))``; negative diagonal entries from round-off
            clip to ``nan`` (matching :attr:`standard_errors`).
        cov : ``(p, p)`` ``jax`` array
            The symmetrised delta-method covariance.

        Notes
        -----
        Gauge-invariance is automatic for a gauge-invariant ``f`` (its ``J_f``
        annihilates the gauge nullspace already pinned out of
        ``Sigma_theta``), so ``Y0`` and ``Y0 @ Q`` give identical SEs. A
        gauge-VIOLATING ``f`` (one that leaks raw ``Y``) returns a valid but
        gauge-dependent SE -- the routine cannot detect the violation; it is
        the caller's responsibility. **v1 / scalar reduction:** for an
        all-scalar tree ``components()`` is the tuple of scalar leaves and
        ``Sigma_theta`` is the ordinary ``(K, K)`` covariance, so this reduces
        to the ordinary delta-method SE; for ``f = lambda c: c[i]`` it agrees
        with ``standard_errors[i]``.

        Eager-only: call outside any ``jax.jit`` boundary.
        """
        from emu_gmm.inference.functional_se import functional_se as _fse

        return _fse(f, self.components(), jnp.asarray(self.Sigma_theta.array))

    def gamma_covariance(self) -> Float[Array, "q q"]:
        r"""Delta-method covariance of ``vech(Gamma)``, ``Gamma = A @ A.T``.

        ``A`` is the first (``PSDFixedRank``) component. The ``q = n(n+1)/2``
        entries follow the row-major lower-triangular ``vech`` order
        (``Gamma[0,0], Gamma[1,0], Gamma[1,1], ...``; see
        :func:`emu_gmm.inference.functional_se.vech_indices`). Gauge-invariant:
        ``Gamma`` is unchanged under ``A -> A @ Q`` for ``Q in O(K)``.
        """
        from emu_gmm.inference.functional_se import gamma_se as _gse

        idx, _ls = self._gamma_leaf()
        _se, cov = _gse(
            self.components(), jnp.asarray(self.Sigma_theta.array), index=idx
        )
        return cov

    def gamma_se(self) -> Float[Array, " q"]:
        r"""Delta-method SE of each ``vech(Gamma)`` entry, ``Gamma = A @ A.T``.

        Returns a ``(q,)`` vector of standard errors for the
        ``q = n(n+1)/2`` unique entries of the symmetric ``Gamma`` in
        row-major lower-triangular ``vech`` order (``Gamma[0,0]``,
        ``Gamma[1,0]``, ``Gamma[1,1]``, ...; see
        :func:`emu_gmm.inference.functional_se.vech_indices`). Thin wrapper
        over :meth:`functional_se`; gauge-invariant.

        Note this axis (``n(n+1)/2`` Gamma-functional entries) is **distinct**
        from the ``coef_table`` / ``standard_errors`` axis (the
        ``total_dimension`` ambient tangent coordinates, whose raw per-entry
        ``Y`` SEs are gauge-arbitrary and not interpretable). They are
        different spaces (R29).
        """
        from emu_gmm.inference.functional_se import gamma_se as _gse

        idx, _ls = self._gamma_leaf()
        se, _cov = _gse(
            self.components(), jnp.asarray(self.Sigma_theta.array), index=idx
        )
        return se

    def eigenvalue_se(self, rank: int | None = None) -> Float[Array, " k"]:
        r"""Delta-method SE of the nonzero eigenvalues of ``Gamma = A @ A.T``.

        The K-Aggregators primary: SEs of the eigenvalues of the cross-price
        substitution matrix ``Gamma``. For a rank-``k`` ``Gamma in R^{n x n}``
        (``A`` a ``PSDFixedRank(n, k)`` factor) this returns a length-``k``
        vector of SEs for the ``k`` **nonzero** eigenvalues, ordered ascending
        to match :func:`jax.numpy.linalg.eigvalsh`.

        The ``n - k`` structural zeros are **not** returned: the zero block is
        a repeated eigenvalue, so its eigenvalue Jacobian is degenerate /
        undefined; the consumer cares only about the ``k`` nonzero eigenvalues.

        Parameters
        ----------
        rank
            The number ``k`` of nonzero eigenvalues. Defaults to the
            ``PSDFixedRank`` rank read from the first leaf of
            :attr:`manifold_spec` when present; otherwise the
            numerically-nonzero eigenvalue count of ``Gamma_hat``
            (``|lambda| > 1e-10 * max|lambda|``).

        Returns
        -------
        ``(k,)`` ``jax`` array of standard errors.

        Degenerate eigenvalues
        ----------------------
        If two of the ``k`` nonzero eigenvalues coincide the individual
        eigenvalues are not smooth functions of ``Gamma`` (only symmetric
        functions of the degenerate block are), so the per-eigenvalue SE is
        not well-defined at exact degeneracy: ``eigvalsh`` returns a finite
        but eigenbasis-dependent derivative there. This is non-generic (the
        generic case is distinct eigenvalues, where the SEs are exact); a
        near-degenerate spectrum yields large but finite SEs. With
        (near-)repeated eigenvalues, prefer a symmetric functional of the
        block (e.g. its sum) via :meth:`functional_se`. Gauge-invariant: the
        eigenvalues of ``Gamma`` do not depend on the O(K) representative of
        ``A``.
        """
        from emu_gmm.inference.functional_se import eigenvalue_se as _evse

        comps = self.components()
        idx, _ls = self._gamma_leaf()
        if rank is None:
            rank = self._gamma_rank(comps)
        se, _cov = _evse(
            comps, jnp.asarray(self.Sigma_theta.array), int(rank), index=idx
        )
        return se

    def _gamma_leaf(self) -> tuple[int, Any | None]:
        """Locate the ``PSDFixedRank`` factor: ``(component_index, leaf_spec)``.

        The single source of truth for which component the Gamma readouts
        (:meth:`gamma_se`, :meth:`gamma_covariance`, :meth:`eigenvalue_se`)
        and the default-``rank`` inference operate on (#117). Previously the
        readouts hard-coded ``components[0]`` while the rank default scanned
        the spec for the first 2-D leaf; a dataclass declared ``(phi, A)``
        got the right rank but the wrong ``Gamma``.

        Rules:

        * spec present: the **unique** leaf whose manifold is a
          :class:`~emu_gmm.manifolds.psd_fixed_rank.PSDFixedRank` (a 2-D
          *Euclidean* matrix leaf is not a Gamma factor and is skipped).
          Zero such leaves, or more than one, is a typed error -- with
          several factors there is no canonical ``Gamma``; use
          :meth:`functional_se` with an explicit functional instead.
        * no spec (a hand-rolled / v1 result): legacy ``components[0]``
          contract; :func:`_gamma_from_components` validates that the
          component is 2-D.

        ``leaf_specs`` and :meth:`components` share leaf-walk (dataclass
        field) order, so the spec index is the component index.
        """
        from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank

        spec = self.manifold_spec
        if spec is None:
            return 0, None
        psd_indices = [
            i
            for i, ls in enumerate(spec.leaf_specs)
            if isinstance(ls.manifold, PSDFixedRank)
        ]
        if not psd_indices:
            raise TypeError(
                "Gamma readout: the manifold spec has no PSDFixedRank leaf, "
                "so Gamma = A @ A.T is undefined for this estimate. The "
                "gamma_se / gamma_covariance / eigenvalue_se conveniences "
                "apply only to PSDFixedRank factors; for other functionals "
                "use result.functional_se(f)."
            )
        if len(psd_indices) > 1:
            raise TypeError(
                f"Gamma readout: the manifold spec has {len(psd_indices)} "
                "PSDFixedRank leaves (component indices "
                f"{psd_indices}); there is no canonical Gamma. Use "
                "result.functional_se(f) with a functional that selects "
                "the intended factor explicitly."
            )
        idx = psd_indices[0]
        return idx, spec.leaf_specs[idx]

    def _gamma_rank(self, components: tuple[Any, ...]) -> int:
        """Infer the rank ``k`` of the ``PSDFixedRank`` factor ``A``.

        Reads the rank off the *same* leaf :meth:`_gamma_leaf` locates for
        the Gamma readouts (#117); falls back to the numerically-nonzero
        eigenvalue count of ``Gamma_hat`` when no spec is present.
        """
        from emu_gmm.inference.functional_se import count_nonzero_eigenvalues

        idx, ls = self._gamma_leaf()
        if ls is not None:
            rank_attr = getattr(ls.manifold, "rank", None)
            if rank_attr is None:
                rank_attr = getattr(ls.manifold, "k", None)
            if rank_attr is not None:
                return int(rank_attr)
            amb = tuple(int(s) for s in ls.ambient_shape)
            return int(amb[1])  # (n, k) ambient shape -> k columns
        return count_nonzero_eigenvalues(components, index=idx)

    @functools.cached_property
    def coef_table(self) -> pd.DataFrame:
        """Coefficient table: estimate, std error, t-stat, p-value.

        A :class:`pandas.DataFrame` indexed by parameter name with four
        columns:

        - ``estimate``: ``theta_hat`` (flattened in PyTree-traversal
          order to align with ``Sigma_theta``'s ``parameters`` axis).
        - ``std_error``: ``sqrt(diag(Sigma_theta))`` (see
          :attr:`standard_errors`).
        - ``t_stat``: ``estimate / std_error`` (NaN where std_error is
          0 or NaN).
        - ``p_value``: two-sided large-sample p-value under standard
          normal reference (``2 * (1 - Phi(|t_stat|))``), matching the
          asymptotic-normality of the GMM estimator.

        Cached on first access.

        Flattening of ``theta_hat`` is manifold-aware. For a v1 / all-scalar
        tree (no ``manifold_spec`` or a spec whose leaves are all scalar)
        the estimate column uses :func:`flatten_params` and is indexed by
        the dataclass field names --- bitwise unchanged from v1. For a
        non-scalar manifold leaf (e.g. ``PSDFixedRank``) the estimate column
        uses :func:`flatten_params_with_spec` (the ambient flatten the
        ``Sigma_theta`` / ``standard_errors`` axis is sized by), and the
        table rows carry **positional tangent labels** (``Y[0,0]``,
        ``Y[0,1]`` ... ``phi[0]``) rather than scalar field names: the raw
        per-entry ambient coordinates of a manifold leaf are gauge-arbitrary
        and not individually interpretable (INT-12/R5). Gauge-invariant
        functionals of ``Gamma = A @ A.T`` (issue #42) must be computed from
        :meth:`components` and their SEs estimated via the delta method.
        """
        import numpy as _np

        if _is_non_scalar_spec(self.manifold_spec):
            estimate, _treedef, _spec = flatten_params_with_spec(self.theta_hat)
            param_names = list(
                tangent_basis_names(
                    self.manifold_spec,
                    fallback_param_names=tuple(self.labels.param_names),
                )
            )
        else:
            estimate, _treedef = flatten_params(self.theta_hat)
            param_names = list(self.labels.param_names)
        estimate_arr = jnp.asarray(estimate)
        # The estimate column, the SE column, and the row index must all
        # describe the same ambient tangent axis: size == total_dimension
        # for a manifold leaf, == field count for v1. A mismatch is a
        # routing bug, not a user error (R7).
        if len(param_names) != int(estimate_arr.shape[0]):
            raise ValueError(
                "coef_table: parameter label count "
                f"{len(param_names)} does not match the flattened estimate "
                f"length {int(estimate_arr.shape[0])}; this indicates a "
                "manifold-spec / flatten routing mismatch"
            )
        se_arr = jnp.asarray(self.standard_errors.array)
        # Where se_arr is non-positive or NaN, t_stat and p_value go
        # to NaN rather than dividing by zero. ``jnp.where`` evaluates
        # both branches under jit, so the divide itself is harmless;
        # we just mask the result.
        safe_se = jnp.where((se_arr > 0.0) & jnp.isfinite(se_arr), se_arr, jnp.nan)
        t_stat = estimate_arr / safe_se
        # Two-sided p-value under N(0,1). scipy is already a dep (see
        # estimator.py); evaluate on the host side via numpy.
        t_host = _np.asarray(t_stat)
        p_value = 2.0 * scipy.stats.norm.sf(_np.abs(t_host))
        return pd.DataFrame(
            {
                "estimate": _np.asarray(estimate_arr),
                "std_error": _np.asarray(se_arr),
                "t_stat": t_host,
                "p_value": p_value,
            },
            index=param_names,
        )

    def to_pandas(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Materialise labelled fields as pandas objects.

        Returns a dict with keys:

        - ``"coefficients"``: :class:`pandas.DataFrame` with columns
          ``estimate, std_error, t_stat, p_value``, indexed by
          parameter name. Same object as :attr:`coef_table`.
        - ``"Sigma_theta"``: :class:`pandas.DataFrame` indexed by
          parameter names on both axes.
        - ``"V_X"``: :class:`pandas.DataFrame` indexed by moment names
          on both axes.
        - ``"N_j"``: :class:`pandas.Series` indexed by moment names.
        - ``"moment_residual"``: :class:`pandas.Series` indexed by
          moment names.
        - ``"summary"``: :class:`pandas.Series` of scalar fields
          (J_stat, J_dof, J_pvalue, converged, iterations,
          tau_realised, kappa_V, final_objective).

        Useful for pandas-centric reporting workflows; the labelled
        :class:`haliax.NamedArray` fields remain available on ``self``
        for users who prefer to stay in the JAX/Haliax stack.
        """
        # ``Sigma_theta`` is sized by the ambient tangent dimension
        # (== field count for v1; > field count for a non-scalar manifold
        # leaf). Index it by the positional tangent labels so the rows /
        # columns match the matrix shape and are not mislabelled as scalar
        # field-names (INT-12/R5). For v1 these labels coincide with the
        # field names, so the DataFrame is unchanged.
        if _is_non_scalar_spec(self.manifold_spec):
            param_names = list(
                tangent_basis_names(
                    self.manifold_spec,
                    fallback_param_names=tuple(self.labels.param_names),
                )
            )
        else:
            param_names = list(self.labels.param_names)
        moment_names = list(self.labels.moment_names)

        sigma_df = pd.DataFrame(
            jnp.asarray(self.Sigma_theta.array),
            index=param_names,
            columns=param_names,
        )
        v_df = pd.DataFrame(
            jnp.asarray(self.V_X.array),
            index=moment_names,
            columns=moment_names,
        )
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
                "J_stat": float(jnp.asarray(self.J_stat)),
                "J_dof": int(self.J_dof),
                "J_pvalue": float(jnp.asarray(self.J_pvalue)),
                "J_pvalue_adjusted": float(jnp.asarray(self.J_pvalue_adjusted)),
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
            "coefficients": self.coef_table,
            "Sigma_theta": sigma_df,
            "V_X": v_df,
            "N_j": n_j,
            "moment_residual": m_res,
            "summary": summary,
        }

    def record(self) -> FitRecord:
        """The slim, stackable per-fit summary pytree (#125).

        Extracts exactly the fields repeated-estimation consumers need
        --- ``theta_flat`` (manifold-aware ambient flatten, same axis as
        ``Sigma_theta``), ``se``, the J triple, ``converged``,
        ``tau_realised``, ``binding_ridge`` --- as a
        :class:`FitRecord` pytree ready for
        ``tree_map(jnp.stack, *records)``. Replaces the hand-rolled
        ``_internal.params.flatten_params`` extraction every MC harness
        previously re-invented (which silently broke for manifold
        parameters; the dispatch here is the same one ``coef_table``
        uses).
        """
        if _is_non_scalar_spec(self.manifold_spec):
            theta_flat, _treedef, _spec = flatten_params_with_spec(self.theta_hat)
            param_names = tuple(
                tangent_basis_names(
                    self.manifold_spec,
                    fallback_param_names=tuple(self.labels.param_names),
                )
            )
        else:
            theta_flat, _treedef = flatten_params(self.theta_hat)
            param_names = tuple(self.labels.param_names)
        theta_arr = jnp.asarray(theta_flat)
        se_arr = jnp.asarray(self.standard_errors.array)
        if int(se_arr.shape[0]) != int(theta_arr.shape[0]):
            raise ValueError(
                "record(): flattened estimate length "
                f"{int(theta_arr.shape[0])} does not match the SE axis "
                f"{int(se_arr.shape[0])}; this indicates a manifold-spec "
                "routing bug (mirrors the coef_table guard)."
            )
        return FitRecord(
            theta_flat=theta_arr,
            se=se_arr,
            J_stat=jnp.asarray(self.J_stat),
            J_pvalue=jnp.asarray(self.J_pvalue),
            J_pvalue_adjusted=jnp.asarray(self.J_pvalue_adjusted),
            converged=jnp.asarray(self.converged, dtype=jnp.float64),
            tau_realised=jnp.asarray(self.diagnostics.tau_realised),
            binding_ridge=jnp.asarray(
                self.diagnostics.binding_ridge, dtype=jnp.float64
            ),
            J_dof=int(self.J_dof),
            param_names=param_names,
        )

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
                "EstimationResult.to_pickle: the following provenance "
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
    def from_pickle(cls, path: Any) -> "EstimationResult":
        """Load an :class:`EstimationResult` pickled by :meth:`to_pickle`.

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
                f"EstimationResult.from_pickle: {path!r} contains a "
                f"{type(obj).__name__}, not an EstimationResult. If this "
                "is a ManifoldGMM GMMResult pickle, re-estimate with "
                "emu_gmm (see docs/migration/manifoldgmm-to-emu-gmm.org); "
                "pickles do not migrate across libraries."
            )
        return obj


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
    "EstimationResult",
    "FitRecord",
    "ManifoldPoint",
    "Emu_GMM_DimensionError",
    # Re-exported from emu_gmm._internal.labels so the type that
    # annotates ``EstimationResult.labels`` has a public home (#56).
    "LabelContext",
]
