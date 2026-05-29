"""Diagnostics builders and logging hooks for :func:`emu_gmm.estimate`.

The :class:`emu_gmm.types.Diagnostics` dataclass is constructed once at
the end of an estimation. This module provides:

- :func:`build_diagnostics`: assemble a :class:`Diagnostics` from the
  raw arrays and scalars computed by the estimator pipeline, wrapping
  the per-moment fields in labelled :class:`haliax.NamedArray` instances.
- :func:`log_to_stdout`: a simple console logger usable as a per-step
  hook during optimisation; prints :math:`\\tau`, :math:`\\kappa(V^\\star)`,
  and the current objective.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax.numpy as jnp
from jaxtyping import Array, Float

from emu_gmm._internal import labels as labels_mod
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
        :class:`jax.Array`.
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

    Returns
    -------
    :class:`Diagnostics`
    """
    return Diagnostics(
        tau_realised=jnp.asarray(tau_realised),
        kappa_V=jnp.asarray(kappa_V),
        binding_ridge=jnp.asarray(binding_ridge),
        cholesky_pivot_min=jnp.asarray(cholesky_pivot_min),
        final_objective=jnp.asarray(final_objective),
        final_gradient_norm=jnp.asarray(final_gradient_norm),
        N_j=labels_mod.label_vector(jnp.asarray(N_j_array), moments_axis),
        moment_residual=labels_mod.label_vector(
            jnp.asarray(moment_residual_array), moments_axis
        ),
        optimizer_info=optimizer_info,
    )


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


__all__ = ["build_diagnostics", "log_to_stdout"]
