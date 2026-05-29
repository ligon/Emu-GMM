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
import jax.numpy as jnp
import jax_dataclasses as jdc
import pandas as pd
import scipy.stats
from jaxtyping import Array, Float

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal.labels import LabelContext, label_vector
from emu_gmm._internal.params import flatten_params

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

    # Optimisation
    final_objective: Float[Array, ""]
    final_gradient_norm: Float[Array, ""]

    # Labelled per-moment
    N_j: ha.NamedArray  # axis [Moments]
    moment_residual: ha.NamedArray  # axis [Moments]; m_hat at theta_hat

    # Provenance
    optimizer_info: OptimizerInfo

    # Hessian condition trio at theta_hat. See ``docs/design.org`` and
    # CLAUDE.md commitment 5: the information matrix is ``G' Lambda G``
    # (never numerical Hessian); ``cond_info`` reports the condition
    # number of that matrix.
    #
    # Keys:
    #   - ``'raw'``: cond(G' Lambda G), Lambda = (V*)^{-1} at theta_hat.
    #   - ``'data_only'``: cond(G' Lambda G) with the penalty
    #     contribution suppressed. In v1 no ``PenaltyStrategy`` is
    #     wired, so this equals ``'raw'``; once #7 (penalty hook) lands,
    #     subtract the penalty Hessian contribution and recompute.
    #   - ``'exclude_gauge'``: alias to ``'raw'`` for v1. Once the v2
    #     manifold support lands, this will project out the
    #     K*(K-1)/2 PSDFixedRank gauge nullspace before computing the
    #     condition number.
    cond_info: dict[str, float] = dataclasses.field(default_factory=dict)

    # Lightweight optimiser-health summary at termination. Keys:
    #   - ``'iters'``: iteration / step count
    #     (mirrors ``optimizer_info.steps``).
    #   - ``'grad_norm'``: ``||grad (1/2)||y||^2||`` at theta_hat
    #     (mirrors ``final_gradient_norm``).
    #   - ``'step_norm'``: norm of the last accepted step, if the
    #     backend exposes it; otherwise ``None``.
    #   - ``'accepted_step_count'``: number of accepted (vs rejected)
    #     LM steps, if the backend exposes it; otherwise ``None``.
    optimizer_health: dict[str, Any] = dataclasses.field(default_factory=dict)


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

        Cached on first access. The flattening of ``theta_hat`` uses
        :func:`emu_gmm._internal.params.flatten_params`, which requires
        all parameter leaves to be 0-d scalars (the v1 contract).
        """
        import numpy as _np

        param_names = list(self.labels.param_names)
        estimate, _treedef = flatten_params(self.theta_hat)
        estimate_arr = jnp.asarray(estimate)
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
    "Emu_GMM_DimensionError",
]
