"""Independent NumPy cell projection versus mixed and pure design covariance.

Run from the repository root after installing development dependencies::

    nice -n 19 taskset -c 36-39 poetry run python \
        scripts/validation/design_aware_fixed_cell_pointwise.py

Ten #130 ladder draws use seed 20260610, the fixed 0.7 stratum profile,
and theta=BETA_TRUE. The original design and estimand are unchanged.
"""

import sys

sys.path.insert(0, "scripts/validation")

import jax
import ladder_mc as ladder
import numpy as np
from emu_gmm import DesignAwareCovariance

design = ladder.make_design(ladder.DesignSpec(psu_per_cell=5, sigma_strat=0.7))
base = ladder.covariance_arm(design, "design_aware_nodof")
fixed = DesignAwareCovariance.from_design_mask(
    base.design, base.sampling, base.design_moment_mask, fixed_strata=True
)
strat = ladder.covariance_arm(design, "stratified")
t = np.asarray(base.design_moment_mask > 0)
Ptt = np.outer(t, t)
Pss = np.outer(~t, ~t)
Pts = np.outer(t, ~t) + np.outer(~t, t)


def cell_projection(measure):
    mask = np.asarray(measure.mask)
    weights = np.asarray(measure.weights)
    psi = np.asarray(jax.vmap(lambda row: ladder.psi(row, ladder.BETA_TRUE))(measure.x))
    cluster = design.psu_ids
    cell = design.cell_ids
    n_cluster, n_moment = design.spec.n_psu, psi.shape[1]
    totals = np.zeros((n_cluster, n_moment))
    exposure = np.zeros_like(totals)
    support = np.zeros_like(totals, dtype=bool)
    np.add.at(totals, cluster, psi * mask * weights[:, None])
    np.add.at(exposure, cluster, mask * weights[:, None])
    np.logical_or.at(support, cluster, mask > 0)
    cluster_cell = cell[np.unique(cluster, return_index=True)[1]]
    N = (mask * weights[:, None]).sum(axis=0)
    V = np.zeros((n_moment, n_moment))
    for j in range(n_moment):
        for k in range(n_moment):
            numerator = 0.0
            for c in range(design.spec.n_cells):
                ids = np.flatnonzero(
                    (cluster_cell == c) & support[:, j] & support[:, k]
                )
                H = len(ids)
                if H < 2:
                    continue
                ej, ek = exposure[ids, j], exposure[ids, k]
                Pj = np.eye(H) - np.outer(ej, ej) / (ej @ ej)
                Pk = np.eye(H) - np.outer(ek, ek) / (ek @ ek)
                trace = np.trace(Pj @ Pk)
                numerator += H / trace * ((Pj @ totals[ids, j]) @ (Pk @ totals[ids, k]))
            if N[j] * N[k] > 0:
                V[j, k] = numerator / (N[j] * N[k])
    return V


max_abs = []
max_fixed_strat = []
max_fixed_numpy = []
fixed_ss = []
fixed_ts = []
for r in range(10):
    measure = ladder.draw_measure(
        design, jax.random.fold_in(jax.random.PRNGKey(20260610), r)
    )
    V_cell = cell_projection(measure)
    V_strat = np.asarray(strat.covariance(ladder.psi, ladder.BETA_TRUE, measure))
    V_fixed = np.asarray(fixed.covariance(ladder.psi, ladder.BETA_TRUE, measure))
    max_abs.append(float(np.max(np.abs(V_cell - V_strat))))
    max_fixed_strat.append(float(np.max(np.abs(V_fixed - V_strat))))
    max_fixed_numpy.append(float(np.max(np.abs(V_fixed - V_cell))))
    fixed_ss.append(float(np.linalg.norm((V_fixed - V_strat) * Pss)))
    fixed_ts.append(float(np.linalg.norm((V_fixed - V_strat) * Pts)))
    if r == 0:
        np.set_printoptions(precision=7, suppress=True)
        print("rep0_cell_V=", V_cell)
        print("rep0_strat_V=", V_strat)
        print("rep0_fixed_V=", V_fixed)
        print("rep0_cell_minus_strat=", V_cell - V_strat)
        print("rep0_fixed_minus_strat=", V_fixed - V_strat)
        print("rep0_TT_exact=", np.array_equal(V_fixed * Ptt, V_strat * Ptt))
print("n_reps=10 seed=20260610 theta=BETA_TRUE")
print("max_abs_numpy_cell_vs_strat=", max(max_abs))
print("max_abs_fixed_vs_strat=", max(max_fixed_strat))
print("max_abs_fixed_vs_numpy=", max(max_fixed_numpy))
print("mean_fixed_SS_diff_norm=", np.mean(fixed_ss))
print("mean_fixed_TS_diff_norm=", np.mean(fixed_ts))
