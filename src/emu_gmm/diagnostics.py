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
import jax
import jax.numpy as jnp
import jax.scipy.stats
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


def regularization_adjusted_pvalue(
    J_stat: Float[Array, ""],
    V: Float[Array, "M M"],
    V_star: Float[Array, "M M"],
    G: Float[Array, "M K"],
) -> Float[Array, ""]:
    """Regularisation-adjusted J-test p-value (weighted-chi^2 limit).

    Per :file:`docs/mcar-asymptotics.org` Theorem 6, when ``V`` is
    replaced by :math:`V^\\star = V + \\tau \\operatorname{diag}(V)`
    with :math:`\\tau` held fixed across the sample path, the J-statistic
    limit is a generalised chi-squared --- a weighted sum of independent
    :math:`\\chi^2_1` variates whose weights are the eigenvalues of
    :math:`(V^\\star)^{-1} V` projected onto the orthogonal complement
    of the column space of :math:`G(\\theta_0)`. When :math:`\\tau = 0`
    the weights coincide at 1, recovering :math:`\\chi^2_{M-K}`.

    For v1, the weighted-chi^2 survival function is approximated via the
    Welch--Satterthwaite scheme [cite:@satterthwaite1946]: match the
    first two moments of :math:`\\sum_i w_i Z_i^2` (mean :math:`\\sum w_i`,
    variance :math:`2 \\sum w_i^2`) to a scaled chi-squared
    :math:`c \\cdot \\chi^2_v` with :math:`c = \\sum w_i^2 / \\sum w_i`
    and :math:`v = (\\sum w_i)^2 / \\sum w_i^2`. This is the standard
    approximation in the linear-model literature; Davies' / Imhof's
    exact CDF would be more accurate but requires non-trivial code that
    falls outside v1's scope.

    Concretely, working in the whitened coordinates
    :math:`\\tilde y = L_\\star^{-1} m` with
    :math:`L_\\star L_\\star^\\top = V^\\star`,

    - :math:`\\tilde G = L_\\star^{-1} G` is the whitened Jacobian;
    - :math:`\\tilde V = L_\\star^{-1} V L_\\star^{-\\top}` is the
      whitened variance under the unregularised limit;
    - :math:`P = I - \\tilde G (\\tilde G^\\top \\tilde G)^{-1}
      \\tilde G^\\top` projects onto the orthogonal complement of the
      column space of :math:`\\tilde G`;
    - the weights are the (M-K) non-zero eigenvalues of
      :math:`P \\tilde V P`.

    Parameters
    ----------
    J_stat : 0-d scalar array
        Realised J-statistic :math:`\\|y\\|^2`.
    V : (M, M) array
        Unregularised covariance estimate at :math:`\\hat\\theta`.
    V_star : (M, M) array
        Regularised covariance :math:`V + \\tau \\operatorname{diag}(V)`
        used by the weighting strategy.
    G : (M, K) array
        Moment Jacobian :math:`E_\\mu[\\nabla_\\theta \\psi]` at
        :math:`\\hat\\theta`.

    Returns
    -------
    p : 0-d scalar array
        Approximate survival function value
        :math:`P(\\sum_i w_i Z_i^2 \\geq J_{\\mathrm{stat}})`.

    Notes
    -----
    Traceable under ``jit`` and ``vmap``: the eigendecomposition is
    via :func:`jax.numpy.linalg.eigh` and the chi-squared survival
    function via :func:`jax.scipy.stats.chi2.sf`. Both accept tracers.
    """
    M, K = G.shape
    # Cholesky-whiten V into the L_star coordinates.
    L_star = jnp.linalg.cholesky(V_star)
    # ~G = L_star^{-1} G
    G_tilde = jax.scipy.linalg.solve_triangular(L_star, G, lower=True)
    # ~V = L_star^{-1} V L_star^{-T}: this is what (V_star)^{-1} V's
    # eigenvalues live as in whitened space.
    Vinv_V_root_left = jax.scipy.linalg.solve_triangular(L_star, V, lower=True)
    V_tilde = jax.scipy.linalg.solve_triangular(
        L_star, Vinv_V_root_left.T, lower=True
    ).T
    # Symmetrise (cancel out roundoff).
    V_tilde = 0.5 * (V_tilde + V_tilde.T)

    # Orthogonal projector onto the orthogonal complement of col(~G).
    # P = I_M - ~G (~G' ~G)^{-1} ~G'.
    GtG = G_tilde.T @ G_tilde
    GtG_inv = jnp.linalg.inv(GtG)
    P = jnp.eye(M) - G_tilde @ GtG_inv @ G_tilde.T

    # Eigenvalues of P ~V P. Of the M eigenvalues, K are exactly zero
    # (projector kills the col(~G) directions); the remaining M-K are
    # the weights for the weighted-chi^2 limit. We keep all of them and
    # sum: zeros contribute nothing.
    PVP = P @ V_tilde @ P
    PVP = 0.5 * (PVP + PVP.T)
    weights = jnp.linalg.eigvalsh(PVP)
    # Clamp small negative eigenvalues (numerical noise) to zero so the
    # Satterthwaite ratio is well defined.
    weights = jnp.where(weights > 0.0, weights, 0.0)

    # Welch-Satterthwaite: ~ c * chi^2_v, with c = sum w^2 / sum w,
    # v = (sum w)^2 / sum w^2. Guard against the degenerate
    # all-weights-zero case (would only happen with M=K, which the
    # dimension guard already prevents from reaching here in practice).
    s1 = jnp.sum(weights)
    s2 = jnp.sum(weights**2)
    safe = s1 > 0.0
    c = jnp.where(safe, s2 / jnp.where(safe, s1, 1.0), 1.0)
    v = jnp.where(safe, (s1**2) / jnp.where(safe, s2, 1.0), float(M - K))

    return jax.scipy.stats.chi2.sf(J_stat / c, v)


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


__all__ = [
    "build_diagnostics",
    "log_to_stdout",
    "regularization_adjusted_pvalue",
]
