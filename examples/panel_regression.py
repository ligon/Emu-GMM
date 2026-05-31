#!/usr/bin/env python
"""Linear panel (fixed-effects) regression run through the emu-gmm harness.

This demonstrates that the GMM interface is *not* Euler-shaped: an ordinary
linear panel regression with unit fixed effects falls right out of the same
``psi`` + ``Measure`` + ``CovarianceStrategy`` pipeline, with no special
casing. (Adapted from the Seasonality consumer's full-insurance proof of
concept, TaimakaSeasonality PR #24 / emu-gmm #82, made self-contained on
synthetic data.)

The idiom
---------
A within (fixed-effects) regression of ``y`` on regressors ``X`` with unit
fixed effects ``alpha_i`` is

    y_it = alpha_i + X_it . beta + u_it.

Sweep the fixed effects *outside* the moment by within-unit demeaning, so the
``alpha_i`` never become parameters:

    y~_it = y_it - mean_i(y),   X~_it = X_it - mean_i(X).

The OLS normal equation of the residualized regression is the moment:

    psi_i(beta) = X~_i * (y~_i - X~_i . beta)        (an M = K vector)

so M == K (just-identified) and the over-identification statistic J is
identically zero. GMM with this moment *is* OLS, so:

  * the point estimate must equal the within-OLS estimator to machine
    precision;
  * ``IIDCovariance`` reproduces the heteroskedasticity-robust (HC0) sandwich
    standard errors;
  * ``ClusteredCovariance(unit)`` reproduces the cluster-robust (CRVE)
    standard errors -- the panel-appropriate inference -- which are larger
    here because the errors are serially correlated within unit.

Run:
    poetry run python examples/panel_regression.py
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    IIDCovariance,
    estimate,
    optimistix_lm,
)

# ---------------------------------------------------------------------------
# Parameter container: two slope coefficients (K = 2). v1 leaves are scalars.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class PanelParams:
    b_x: float
    b_z: float


# ---------------------------------------------------------------------------
# Synthetic panel with unit fixed effects and within-unit (serially
# correlated) errors, so the cluster-robust SE genuinely exceeds the robust SE.
# ---------------------------------------------------------------------------


def make_panel(n_units=60, n_periods=8, beta=(1.5, -0.8), seed=0):
    rng = np.random.default_rng(seed)
    N = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)

    # Regressors: a unit-level level plus within-unit variation (so the within
    # estimator is identified after demeaning).
    x = rng.normal(rng.normal(size=n_units)[unit], 1.0)
    z = rng.normal(rng.normal(size=n_units)[unit], 1.0)

    # Errors: AR(1) within unit (survives demeaning -> clustering matters) with
    # mild heteroskedasticity in x (so HC0 differs from the classical SE).
    u = np.empty(N)
    for g in range(n_units):
        idx = np.where(unit == g)[0]
        e = rng.normal(size=idx.size) * (0.5 + 0.6 * np.abs(x[idx]))
        prev = 0.0
        for t, i in enumerate(idx):
            prev = 0.6 * prev + e[t]
            u[i] = prev

    alpha = rng.normal(scale=2.0, size=n_units)[unit]  # unit fixed effects
    y = alpha + beta[0] * x + beta[1] * z + u
    return y, np.column_stack([x, z]), unit


def within_demean(values, unit):
    """Subtract the within-unit mean (sweep out unit fixed effects)."""
    out = np.asarray(values, dtype=float).copy()
    for g in np.unique(unit):
        m = unit == g
        out[m] -= out[m].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Independent numpy references: within-OLS + HC0 + CRVE (no dof correction,
# matching emu-gmm's cluster-totals convention; see emu-gmm #82 item 1).
# ---------------------------------------------------------------------------


def within_ols(y, X):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    u = y - X @ beta
    return beta, u, XtX


def hc0_se(X, u, XtX):
    bread = np.linalg.inv(XtX)
    meat = (X * u[:, None]).T @ (X * u[:, None])  # sum_i x_i x_i' u_i^2
    return np.sqrt(np.diag(bread @ meat @ bread))


def crve_se(X, u, XtX, cluster):
    bread = np.linalg.inv(XtX)
    K = X.shape[1]
    meat = np.zeros((K, K))
    for g in np.unique(cluster):
        idx = cluster == g
        s = (X[idx] * u[idx, None]).sum(0)  # sum_{i in g} x_i u_i
        meat += np.outer(s, s)
    return np.sqrt(np.diag(bread @ meat @ bread))


# ---------------------------------------------------------------------------
# The emu-gmm wiring of the OLS normal equation.
# ---------------------------------------------------------------------------


def emu_panel_fit(y_t, X_t, unit):
    """Cast the residualized within-regression as emu-gmm and return the IID
    and clustered results. ``y_t``, ``X_t`` are already within-demeaned."""
    N = y_t.shape[0]
    # x carries [y~, x~, z~] per observation (D = 3); M = K = 2.
    data = jnp.asarray(np.column_stack([y_t, X_t]))
    measure = EmpiricalMeasure(
        x=data,
        mask=jnp.ones((N, 2)),  # both moments observed for every row
        weights=jnp.ones(N),
    )

    def psi(row, theta):
        resid = row[0] - theta.b_x * row[1] - theta.b_z * row[2]
        return jnp.array([row[1] * resid, row[2] * resid])

    # Densify unit ids into [0, n_clusters) floats for ClusteredCovariance.
    _, codes = np.unique(unit, return_inverse=True)
    n_clusters = int(codes.max()) + 1

    common = dict(
        model=psi,
        measure=measure,
        optimizer=optimistix_lm(),
        theta_init=PanelParams(b_x=0.0, b_z=0.0),
        moment_names=("score_x", "score_z"),
    )
    res_iid = estimate(covariance=IIDCovariance(), **common)
    res_cl = estimate(
        covariance=ClusteredCovariance(
            cluster_ids=jnp.asarray(codes, dtype=jnp.float64),
            n_clusters=n_clusters,
        ),
        **common,
    )
    return res_iid, res_cl


def main():
    y, X, unit = make_panel()
    y_t = within_demean(y, unit)
    X_t = within_demean(X, unit)

    # Reference within-OLS + sandwich SEs.
    beta_ref, u_ref, XtX = within_ols(y_t, X_t)
    se_hc0 = hc0_se(X_t, u_ref, XtX)
    se_crve = crve_se(X_t, u_ref, XtX, unit)

    # emu-gmm.
    res_iid, res_cl = emu_panel_fit(y_t, X_t, unit)
    beta_emu = np.array([float(res_iid.theta_hat.b_x), float(res_iid.theta_hat.b_z)])
    se_emu_iid = np.asarray(res_iid.standard_errors.array, dtype=float)
    se_emu_cl = np.asarray(res_cl.standard_errors.array, dtype=float)

    names = ["b_x", "b_z"]
    print("Linear panel (within / fixed-effects) regression via emu-gmm")
    print("=" * 64)
    print(f"{'coef':>6} {'within-OLS':>12} {'emu-gmm':>12} {'|diff|':>10}")
    for j, nm in enumerate(names):
        print(
            f"{nm:>6} {beta_ref[j]:>12.6f} {beta_emu[j]:>12.6f} "
            f"{abs(beta_ref[j] - beta_emu[j]):>10.2e}"
        )
    print(f"\nJ stat (just-identified, should be ~0): {float(res_iid.J_stat):.2e}")
    print(f"converged: {res_iid.converged}; iterations: {res_iid.iterations}")

    print("\nStandard errors")
    print("-" * 64)
    print(
        f"{'coef':>6} {'HC0(ref)':>12} {'emu IID':>12} "
        f"{'CRVE(ref)':>12} {'emu cluster':>12}"
    )
    for j, nm in enumerate(names):
        print(
            f"{nm:>6} {se_hc0[j]:>12.6f} {se_emu_iid[j]:>12.6f} "
            f"{se_crve[j]:>12.6f} {se_emu_cl[j]:>12.6f}"
        )
    print(
        "\nemu IIDCovariance == HC0 sandwich; emu ClusteredCovariance == CRVE.\n"
        "The clustered SEs are larger (within-unit serial correlation) -- the\n"
        "panel-appropriate inference, obtained by swapping ONE argument."
    )


if __name__ == "__main__":
    main()
