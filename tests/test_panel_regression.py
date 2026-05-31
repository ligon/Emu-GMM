"""Linear panel (fixed-effects) regression via emu-gmm: a known-answer lock.

A within (fixed-effects) regression cast as GMM with the OLS normal-equation
moment ``psi_i(beta) = X~_i (y~_i - X~_i . beta)`` (M = K, just-identified)
must reproduce OLS exactly, and its design-aware standard errors must match
the textbook sandwiches:

  * point estimate == within-OLS to machine precision, J == 0;
  * ``IIDCovariance`` SE == heteroskedasticity-robust (HC0) sandwich;
  * ``ClusteredCovariance(unit)`` SE == cluster-robust (CRVE) sandwich, and is
    larger here because the errors are serially correlated within unit.

This guards the empirical path on a closed-form problem far from the Euler
example -- i.e. that the interface is general, not Euler-shaped. The numpy
references are independent (the class is never used to check itself). See
``examples/panel_regression.py`` and emu-gmm #82.
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


@jdc.pytree_dataclass
class _Params:
    b_x: float
    b_z: float


def _make_panel(n_units=30, n_periods=6, beta=(1.5, -0.8), seed=3):
    rng = np.random.default_rng(seed)
    N = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    x = rng.normal(rng.normal(size=n_units)[unit], 1.0)
    z = rng.normal(rng.normal(size=n_units)[unit], 1.0)
    # AR(1) within-unit errors (survive demeaning) -> CRVE > HC0.
    u = np.empty(N)
    for g in range(n_units):
        idx = np.where(unit == g)[0]
        e = rng.normal(size=idx.size) * (0.5 + 0.6 * np.abs(x[idx]))
        prev = 0.0
        for t, i in enumerate(idx):
            prev = 0.6 * prev + e[t]
            u[i] = prev
    alpha = rng.normal(scale=2.0, size=n_units)[unit]
    y = alpha + beta[0] * x + beta[1] * z + u
    return y, np.column_stack([x, z]), unit


def _within_demean(values, unit):
    out = np.asarray(values, dtype=float).copy()
    for g in np.unique(unit):
        m = unit == g
        out[m] -= out[m].mean(axis=0)
    return out


def _within_ols(y, X):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return beta, y - X @ beta, XtX


def _hc0_se(X, u, XtX):
    bread = np.linalg.inv(XtX)
    meat = (X * u[:, None]).T @ (X * u[:, None])
    return np.sqrt(np.diag(bread @ meat @ bread))


def _crve_se(X, u, XtX, cluster):
    bread = np.linalg.inv(XtX)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(cluster):
        idx = cluster == g
        s = (X[idx] * u[idx, None]).sum(0)
        meat += np.outer(s, s)
    return np.sqrt(np.diag(bread @ meat @ bread))


def _emu_fit(y_t, X_t, unit, covariance):
    N = y_t.shape[0]
    measure = EmpiricalMeasure(
        x=jnp.asarray(np.column_stack([y_t, X_t])),
        mask=jnp.ones((N, 2)),
        weights=jnp.ones(N),
    )

    def psi(row, theta):
        resid = row[0] - theta.b_x * row[1] - theta.b_z * row[2]
        return jnp.array([row[1] * resid, row[2] * resid])

    return estimate(
        model=psi,
        measure=measure,
        covariance=covariance,
        optimizer=optimistix_lm(),
        theta_init=_Params(b_x=0.0, b_z=0.0),
    )


def _fixture():
    y, X, unit = _make_panel()
    y_t = _within_demean(y, unit)
    X_t = _within_demean(X, unit)
    _, codes = np.unique(unit, return_inverse=True)
    return y_t, X_t, unit, codes


def test_point_estimate_matches_within_ols():
    y_t, X_t, unit, codes = _fixture()
    beta_ref, _, _ = _within_ols(y_t, X_t)
    res = _emu_fit(y_t, X_t, unit, IIDCovariance())
    beta_emu = np.array([float(res.theta_hat.b_x), float(res.theta_hat.b_z)])
    assert np.max(np.abs(beta_emu - beta_ref)) < 1e-9
    assert float(res.J_stat) < 1e-18  # just-identified -> J identically 0


def test_iid_covariance_equals_hc0_sandwich():
    y_t, X_t, unit, codes = _fixture()
    _, u_ref, XtX = _within_ols(y_t, X_t)
    se_ref = _hc0_se(X_t, u_ref, XtX)
    res = _emu_fit(y_t, X_t, unit, IIDCovariance())
    se_emu = np.asarray(res.standard_errors.array, dtype=float)
    assert np.allclose(se_emu, se_ref, rtol=1e-8, atol=1e-12)


def test_clustered_covariance_equals_crve_sandwich():
    y_t, X_t, unit, codes = _fixture()
    _, u_ref, XtX = _within_ols(y_t, X_t)
    se_ref = _crve_se(X_t, u_ref, XtX, unit)
    res = _emu_fit(
        y_t,
        X_t,
        unit,
        ClusteredCovariance(
            cluster_ids=jnp.asarray(codes, dtype=jnp.float64),
            n_clusters=int(codes.max()) + 1,
        ),
    )
    se_emu = np.asarray(res.standard_errors.array, dtype=float)
    assert np.allclose(se_emu, se_ref, rtol=1e-8, atol=1e-12)


def test_clustering_inflates_standard_errors():
    y_t, X_t, unit, codes = _fixture()
    se_iid = np.asarray(
        _emu_fit(y_t, X_t, unit, IIDCovariance()).standard_errors.array, dtype=float
    )
    se_cl = np.asarray(
        _emu_fit(
            y_t,
            X_t,
            unit,
            ClusteredCovariance(
                cluster_ids=jnp.asarray(codes, dtype=jnp.float64),
                n_clusters=int(codes.max()) + 1,
            ),
        ).standard_errors.array,
        dtype=float,
    )
    # AR(1) within-unit errors -> the panel-robust SE strictly exceeds HC0.
    assert np.all(se_cl > se_iid)
