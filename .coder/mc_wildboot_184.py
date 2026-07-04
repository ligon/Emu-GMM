#!/usr/bin/env python
"""MC study for issue #184: moment_wild_bootstrap calibration at theta_hat.

Design
------
Over-identified mean model: x_ij = theta0 + eps_ij, eps ~ N(0,1) iid,
psi_j(x_i, theta) = x_ij - theta  (M = 5 moments, K = 1, J_dof = 4).
Arbitrary balanced cluster assignment (G = 20 x 10) -- no within-cluster
correlation, so both the cluster-wild bootstrap and the clustered V are
valid under H0.

Per MC replicate: fit theta_hat via the package estimator (CU weighting),
then run the refit-free wild bootstrap twice -- at theta_hat (the
documented-but-suspect usage, issue #184) and at the true theta0 (the
theoretically clean usage). Under the estimation-effect hypothesis:
  J_observed(theta_hat) ~ chi2_{M-K} (mean ~ 4) while the sign-flipped
  J_boot draws target ~ chi2_M (mean ~ 5), so p-values at theta_hat are
  pushed toward 1 (conservative rejection rates); at theta0 both sides
  target chi2_M and p is ~ Uniform(0,1).

Usage: python mc_wildboot_184.py [R] [B]   (defaults R=500, B=999)
"""

import sys

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import ClusteredCovariance, EmpiricalMeasure, build_estimator
from emu_gmm.inference import moment_wild_bootstrap

N, M, G = 200, 5, 20
THETA0 = 1.0
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
B = int(sys.argv[2]) if len(sys.argv) > 2 else 999


@jdc.pytree_dataclass
class Mu:
    mu: jax.Array


def psi(x, theta):
    return x - theta.mu  # (M,)


CLUSTER_IDS = jnp.repeat(jnp.arange(G), N // G)


def draw_measure(key):
    x = THETA0 + jax.random.normal(key, (N, M), dtype=jnp.float64)
    return EmpiricalMeasure(x=x, mask=jnp.ones((N, M)), weights=jnp.ones(N))


def main() -> None:
    cov = ClusteredCovariance(cluster_ids=CLUSTER_IDS, n_clusters=G)
    key = jax.random.PRNGKey(20260702)

    meas0 = draw_measure(jax.random.fold_in(key, 10**6))
    run = build_estimator(
        model=psi,
        measure=meas0,
        covariance=cov,
        theta_init=Mu(mu=jnp.asarray(0.0)),
    )

    rows = []
    for r in range(R):
        if r > 0 and r % 50 == 0:
            jax.clear_caches()  # moment_wild_bootstrap builds fresh closures per call
            print(f"rep {r}/{R}", flush=True)
        kr = jax.random.fold_in(key, r)
        meas = draw_measure(jax.random.fold_in(kr, 1))
        res = run(Mu(mu=jnp.asarray(0.0)), meas)
        wb_hat = moment_wild_bootstrap(
            psi, res.theta_hat, meas, cov, n_boot=B, key=jax.random.fold_in(kr, 2)
        )  # projected (post-#184 default)
        wb_hat_raw = moment_wild_bootstrap(
            psi,
            res.theta_hat,
            meas,
            cov,
            n_boot=B,
            key=jax.random.fold_in(kr, 2),
            project_estimation_effect=False,
        )  # unprojected (pre-#184 behaviour)
        wb_0 = moment_wild_bootstrap(
            psi,
            Mu(mu=jnp.asarray(THETA0)),
            meas,
            cov,
            n_boot=B,
            key=jax.random.fold_in(kr, 3),
            project_estimation_effect=False,
        )  # hypothesised theta_0: no estimation effect, projection off
        rows.append(
            (
                float(wb_hat.p_value),
                float(wb_hat_raw.p_value),
                float(wb_0.p_value),
                float(wb_hat.J_observed),
                float(wb_0.J_observed),
                float(jnp.mean(wb_hat.J_boot)),
                float(jnp.mean(wb_hat_raw.J_boot)),
                bool(res.converged),
            )
        )

    a = np.array([row[:7] for row in rows])
    conv = np.array([row[7] for row in rows])
    p_hat, p_raw, p_0, j_hat, j_0, jboot, jboot_raw = a.T
    print(f"\n=== issue #184 MC: R={R}, B={B}, N={N}, M={M}, K=1, G={G} ===")
    print(f"converged: {conv.sum()}/{R}")
    print(
        f"mean J_observed(theta_hat) = {j_hat.mean():.3f}   (chi2_{{M-K}} mean = {M - 1})"
    )
    print(f"mean J_observed(theta_0)   = {j_0.mean():.3f}   (chi2_M mean     = {M})")
    print(
        f"mean J_boot projected   = {jboot.mean():.3f}  (target chi2_{{M-K}} mean = {M - 1})"
    )
    print(f"mean J_boot unprojected = {jboot_raw.mean():.3f}  (chi2_M mean = {M})")
    for alpha in (0.01, 0.05, 0.10):
        print(
            f"reject@{alpha:.2f}:  hat/proj = {(p_hat <= alpha).mean():.4f}"
            f"   hat/raw = {(p_raw <= alpha).mean():.4f}"
            f"   theta0 = {(p_0 <= alpha).mean():.4f}"
        )
    print(
        f"mean p: hat/proj = {p_hat.mean():.4f}   hat/raw = {p_raw.mean():.4f}"
        f"   theta0 = {p_0.mean():.4f}"
    )
    # Kolmogorov-Smirnov distance from Uniform(0,1), no scipy needed.
    for name, p in (("hat/proj", p_hat), ("hat/raw", p_raw), ("theta_0", p_0)):
        s = np.sort(p)
        grid = (np.arange(1, R + 1)) / R
        ks = np.max(np.maximum(np.abs(grid - s), np.abs(s - (np.arange(R) / R))))
        print(f"KS vs U(0,1) at {name}: {ks:.4f}  (95% crit ~ {1.36 / np.sqrt(R):.4f})")


if __name__ == "__main__":
    main()
