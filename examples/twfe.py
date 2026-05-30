"""Two-way fixed effects panel: GMM with iterated weighting + cluster-robust SE.

Pedagogical port of the textbook panel-data example into the emu-gmm
framework. The model is

    y_{i,t} = a_i + b_t + c * x_{i,t} + e_{i,t},          (*)

with i = 0..N-1 individuals, t = 0..T-1 periods, a_i and b_t additive
fixed effects, x iid normal independent of (a, b, e), and e iid normal.
The structural object of interest is the scalar slope ``c``.

The within-transformation
=========================

Subtracting unit, time, and grand means from both sides of (*) eliminates
the fixed effects:

    y_{i,t} - y_bar_i - y_bar_t + y_bar
        = c * (x_{i,t} - x_bar_i - x_bar_t + x_bar)
          + (e_{i,t} - e_bar_i - e_bar_t + e_bar).

Write the within-transformed variables as ``y_w`` and ``x_w``. Because
``x`` is independent of the fixed effects and the error, the population
moment condition

    E[x_w * (y_w - c * x_w)] = 0                          (m_0)

uniquely identifies ``c``. With M=1, K=1 this is the just-identified case
and the GMM estimator coincides with the within-OLS (TWFE) estimator.

For over-identification (and to give the J-test and the iterated-weighting
loop something nontrivial to do) this example adds a second moment

    E[x_w^3 * (y_w - c * x_w)] = 0,                       (m_1)

which holds population-wise whenever ``x`` and ``e`` are independent
with finite moments (the structural error orthogonality is sharper than
mere first-moment orthogonality; cubed-instrument moments are a standard
device when the literal OLS moment is just-identified). With M=2, K=1
the system is over-identified and the J-statistic is asymptotically
chi^2_1.

Within-cluster correlation
==========================

The within-transformed residual ``e_w_{i,t}`` is *not* iid across
``(i, t)``: subtracting unit and time means induces dependence. The
canonical defence in applied panel work is to cluster the sandwich on
the individual ``i``, treating each unit's full time series as one
correlated block. The example wires this through
:class:`emu_gmm.ClusteredCovariance` with one cluster per individual.

What Emu-GMM API surface this exercises
=======================================

- :class:`emu_gmm.EmpiricalMeasure` --- the within-transformed panel
  flattened to an ``(N*T, 2)`` data array of ``(y_w, x_w)`` rows.
- :class:`emu_gmm.ClusteredCovariance` --- cluster-robust sandwich on
  the individual index ``i``, the standard panel-clustering choice.
- :class:`emu_gmm.IteratedWeighting` --- the classical Hansen-style
  two-step / k-step GMM weighting scheme. With M > K the iterated loop
  refreshes the weight matrix at each updated ``c`` and stops when
  ``c`` stabilises within ``weighting_tol``.
- :func:`emu_gmm.estimate` --- the entry point that ties them all
  together and produces an :class:`EstimationResult` with cluster-robust
  standard errors on ``c_hat`` and the J-statistic of the
  over-identifying restriction.

Run directly::

    poetry run python examples/twfe.py
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float

from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    IteratedWeighting,
    estimate,
    optimistix_lm,
)
from emu_gmm.types import EstimationResult

# ---------------------------------------------------------------------------
# Panel dimensions and true structural parameter.
# ---------------------------------------------------------------------------
N_UNITS: int = 20  # number of individuals
N_PERIODS: int = 15  # number of time periods
C_TRUE: float = 1.0  # true slope on x
SIGMA_A: float = 1.0  # individual-FE std dev
SIGMA_B: float = 1.0  # time-FE std dev
SIGMA_E: float = 1.0  # error std dev

# Seeds. The fixed effects are drawn once and held constant across calls
# of :func:`make_panel`; the per-call ``seed`` drives ``x`` and ``e``.
FE_SEED: int = 1729


# ---------------------------------------------------------------------------
# Parameter dataclass.
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class TWFEParams:
    """The lone structural parameter: the slope ``c`` on ``x``."""

    c: float


# ---------------------------------------------------------------------------
# DGP.
# ---------------------------------------------------------------------------
def _draw_fixed_effects(seed: int = FE_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Draw individual and time fixed effects, held constant across panels."""
    rng = np.random.default_rng(seed)
    a = SIGMA_A * rng.standard_normal(size=N_UNITS)
    b = SIGMA_B * rng.standard_normal(size=N_PERIODS)
    return a, b


def make_panel(
    seed: int,
    *,
    n_units: int = N_UNITS,
    n_periods: int = N_PERIODS,
    c_true: float = C_TRUE,
) -> tuple[
    Float[Array, "N 2"],
    Float[Array, " N"],
    int,
]:
    """Generate one TWFE panel and within-transform it.

    Returns a tuple ``(data, cluster_ids, n_clusters)`` where

    - ``data`` is an ``(N*T, 2)`` JAX array whose columns are the
      within-transformed ``(y_w, x_w)``;
    - ``cluster_ids`` is the per-row individual index ``i`` (as a float
      array, cast internally to int by :class:`ClusteredCovariance`);
    - ``n_clusters`` is the number of distinct individuals.

    The fixed effects ``a_i`` and ``b_t`` are drawn once at module load
    (seeded by ``FE_SEED``) and reused across all calls; only ``x`` and
    ``e`` vary with ``seed``. This mirrors the "fixed effects as
    structural parameters" convention used in
    :mod:`DGP_Protocol.examples.twfe`.
    """
    a_vec, b_vec = _draw_fixed_effects()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n_units, n_periods))
    e = SIGMA_E * rng.standard_normal(size=(n_units, n_periods))
    y = a_vec[:, None] + b_vec[None, :] + c_true * x + e

    # Within-transform: subtract unit, time, and grand means.
    y_mean_i = y.mean(axis=1, keepdims=True)
    y_mean_t = y.mean(axis=0, keepdims=True)
    y_grand = y.mean()
    x_mean_i = x.mean(axis=1, keepdims=True)
    x_mean_t = x.mean(axis=0, keepdims=True)
    x_grand = x.mean()
    y_w = y - y_mean_i - y_mean_t + y_grand
    x_w = x - x_mean_i - x_mean_t + x_grand

    # Flatten to long format: row index is i * T + t, columns are (y_w, x_w).
    data = np.stack([y_w.ravel(), x_w.ravel()], axis=1)
    # Cluster IDs: per-row individual index ``i``. JAX prefers floats for
    # traced arrays; :class:`ClusteredCovariance` casts to int internally.
    cluster_ids = np.repeat(np.arange(n_units, dtype=np.float64), n_periods)

    return jnp.asarray(data), jnp.asarray(cluster_ids), n_units


# ---------------------------------------------------------------------------
# StructuralModel.
# ---------------------------------------------------------------------------
def twfe_residual(x: Float[Array, " 2"], theta: TWFEParams) -> Float[Array, " 2"]:
    """Per-observation residual vector psi(x, theta).

    The data row ``x`` packs the within-transformed observables:
    ``x[0] = y_w`` and ``x[1] = x_w``. The two moment conditions are

        psi_0 = x_w * (y_w - c * x_w),
        psi_1 = x_w**3 * (y_w - c * x_w).

    Both have zero population expectation when ``c == c_true`` and the
    DGP satisfies (x, e) independence; see the module docstring.
    """
    y_w = x[0]
    x_w = x[1]
    e_w = y_w - theta.c * x_w
    return jnp.array([x_w * e_w, x_w**3 * e_w])


# ---------------------------------------------------------------------------
# Entry point used by both the runnable demo and the smoke test.
# ---------------------------------------------------------------------------
def run_twfe(
    *,
    seed: int = 0,
    c_init: float = 0.5,
    weighting_iterations: int = 5,
    weighting_tol: float = 1e-6,
    optimizer_rtol: float = 1e-8,
    optimizer_atol: float = 1e-8,
) -> EstimationResult:
    """Run the TWFE estimation pipeline once and return the result.

    Exposed as a module-level entry point so the recovery smoke test can
    call it without importing or executing the ``__main__`` block.
    """
    data, cluster_ids, n_clusters = make_panel(seed=seed)
    measure = EmpiricalMeasure(
        x=data,
        mask=jnp.ones((data.shape[0], 2)),
        weights=jnp.ones(data.shape[0]),
    )
    covariance = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=n_clusters)
    weighting = IteratedWeighting(
        weighting_iterations=weighting_iterations,
        weighting_tol=weighting_tol,
    )
    return estimate(
        model=twfe_residual,
        measure=measure,
        covariance=covariance,
        weighting=weighting,
        optimizer=optimistix_lm(rtol=optimizer_rtol, atol=optimizer_atol),
        theta_init=TWFEParams(c=c_init),
    )


# ---------------------------------------------------------------------------
# Runnable demo.
# ---------------------------------------------------------------------------
def main() -> None:
    print(
        f"Two-way fixed effects panel: N={N_UNITS} individuals, "
        f"T={N_PERIODS} periods, true c = {C_TRUE}."
    )
    print("Estimating via IteratedWeighting + ClusteredCovariance(by individual i).\n")

    result = run_twfe(seed=0)

    print(result.coef_table.to_string())
    c_hat = float(result.theta_hat.c)
    se = float(result.coef_table["std_error"].iloc[0])
    print(
        f"\n  c_hat = {c_hat:.6f}   (truth {C_TRUE:.2f}, "
        f"|err| = {abs(c_hat - C_TRUE):.2e}, cluster-robust SE = {se:.4f})"
    )
    print(
        f"  J-stat = {float(result.J_stat):.4e}   "
        f"(dof = {result.J_dof}, p = {float(result.J_pvalue):.3f}, "
        f"p_adj = {float(result.J_pvalue_adjusted):.3f})"
    )
    print(
        f"  converged = {bool(result.converged)}   "
        f"(outer iterations = {int(result.iterations)}, "
        f"final_grad_norm = {float(result.diagnostics.final_gradient_norm):.2e})"
    )

    # Recovery sanity: c_hat within 2 cluster-robust SE of the truth.
    assert (
        abs(c_hat - C_TRUE) < 2.0 * se
    ), f"c_hat off by {abs(c_hat - C_TRUE):.4f}, > 2 * SE = {2 * se:.4f}"


if __name__ == "__main__":
    main()
