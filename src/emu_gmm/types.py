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
works in user code. Result dataclasses are plain
:func:`dataclasses.dataclass` instances (not JAX PyTrees) since they are
constructed once at the end of an estimation and not threaded through
``jit`` boundaries.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import haliax as ha
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float

from emu_gmm._internal.labels import LabelContext

# A user's parameter PyTree: typically a @jdc.pytree_dataclass. We use
# Any in the protocol signatures because users define their own types;
# the framework only assumes the value is a valid JAX PyTree.
ParamsLike = Any

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
    ridge magnitude (for diagnostics). Implementations live in
    :mod:`emu_gmm.regularization`.
    """

    def apply(self, V: Float[Array, "M M"]) -> tuple[Float[Array, "M M"], float]: ...


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


@dataclasses.dataclass(frozen=True)
class OptimizerInfo:
    """Backend-specific info from one optimiser run."""

    steps: int
    status: str  # "converged" | "max_iterations" | "diverged" | other
    final_objective: float
    backend: str  # "optimistix" | "scipy" | other


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """Numerical diagnostics from one estimation run.

    Scalar fields capture the regularisation choice and convergence
    metrics. Labelled fields (``N_j``, ``moment_residual``) carry
    moment-axis coordinates and are usable in pandas-style inspection
    via :meth:`EstimationResult.to_pandas`.
    """

    # Regularisation
    tau_realised: float
    kappa_V: float
    binding_ridge: bool

    # Cholesky
    cholesky_pivot_min: float

    # Optimisation
    final_objective: float
    final_gradient_norm: float

    # Labelled per-moment
    N_j: ha.NamedArray  # axis [Moments]
    moment_residual: ha.NamedArray  # axis [Moments]; m_hat at theta_hat

    # Provenance
    optimizer_info: OptimizerInfo


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

    # J-test
    J_stat: float
    J_dof: int
    J_pvalue: float

    # Status
    converged: bool
    iterations: int

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

    def to_pandas(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Materialise labelled fields as pandas objects.

        Returns a dict with keys:

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
        summary = pd.Series(
            {
                "J_stat": self.J_stat,
                "J_dof": self.J_dof,
                "J_pvalue": self.J_pvalue,
                "converged": self.converged,
                "iterations": self.iterations,
                "tau_realised": self.diagnostics.tau_realised,
                "kappa_V": self.diagnostics.kappa_V,
                "final_objective": self.diagnostics.final_objective,
            }
        )

        return {
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
]
