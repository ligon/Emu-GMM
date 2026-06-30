"""Estimate BLP with emu-gmm and compare critically against pyblp.

Primary comparison ("fixed_W"): feed emu pyblp's *final optimal weight matrix*
as a Fixed weight, so emu minimises the identical GMM objective g'Wg. If emu's
moment construction and optimiser are correct it must land on pyblp's theta_hat
and reproduce its objective. This isolates correctness to the parts emu owns.

Secondary ("iterated"): emu's fully self-contained pipeline --- its own robust
covariance (IIDCovariance over product-market rows == pyblp's robust S) and its
own iterated optimal weighting --- starting from pyblp's estimate. Compares
emu's independently-constructed SEs and J against pyblp's.

Run:  .venv/bin/python examples/blp/run_emu.py [logit|rc|rc_demographics|all]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from emu_gmm import estimate
from emu_gmm.covariance import AnalyticalCovariance, IIDCovariance
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import Fixed, IteratedWeighting

from blp_data import load_data, load_estimates, load_updated_W, load_W
from blp_model import (
    BLPParams,
    LogitParams,
    RCParams,
    euclidean_leaf,
    make_psi,
    product_index_x,
)

REF = Path(__file__).parent / "reference"


# Nevo's pi sparsity pattern: only these (row, col) entries are free; the
# rest are fixed at zero (pyblp estimates only nonzero initial values). Taken
# from build_reference.INITIAL_PI.
INITIAL_PI = np.array(
    [
        [5.4819, 0.0, 0.2037, 0.0],
        [15.8935, -1.2000, 0.0, 2.6342],
        [-0.2506, 0.0, 0.0511, 0.0],
        [1.2650, 0.0, -0.8091, 0.0],
    ]
)
PI_FREE = np.nonzero(INITIAL_PI)  # (rows, cols) of the 10 estimated entries


def build_theta_init(spec, est, data):
    beta = euclidean_leaf(np.array(est["beta"]).reshape(-1))
    if spec == "logit":
        return LogitParams(beta=beta)
    sigma = euclidean_leaf(np.diag(np.array(est["sigma"])))
    if spec == "rc":
        return RCParams(beta=beta, sigma=sigma)
    # rc_demographics: pi holds only the free entries, in PI_FREE order.
    pi_full = np.array(est["pi"])
    pi = euclidean_leaf(pi_full[PI_FREE])
    return BLPParams(beta=beta, sigma=sigma, pi=pi)


def _leaf_arr(leaf):
    return np.asarray(getattr(leaf, "array", leaf)).reshape(-1)


def flat_theta(theta):
    """Flatten a param object to (beta, sigma_diag, pi) numpy for printing."""
    beta = _leaf_arr(theta.beta)
    sigma = _leaf_arr(theta.sigma) if hasattr(theta, "sigma") else np.zeros(0)
    pi = _leaf_arr(theta.pi) if hasattr(theta, "pi") else np.zeros(0)
    return beta, sigma, pi


def cold_theta_init(spec, data, W):
    """Nevo's documented initial values as theta_init, with beta profiled in.

    Uses pyblp's published initial sigma/pi (NOT the converged estimate) and
    sets beta to the linear-IV solution at the initial delta(theta2) --- a fair
    cold start that tests emu's optimiser, not just fixed-point confirmation.
    """
    from blp_model import solve_all_deltas

    INITIAL_SIGMA_DIAG = jnp.asarray([0.3302, 2.4526, 0.0163, 0.2441])
    N = data.T * data.J
    X1N = jnp.asarray(data.X1).reshape(N, data.K1)
    ZN = jnp.asarray(data.Z).reshape(N, data.M)
    if spec == "rc":
        delta = solve_all_deltas(INITIAL_SIGMA_DIAG, jnp.zeros((4, 0)), data).reshape(-1)
    else:
        pi0 = jnp.zeros((data.K2, data.D)).at[PI_FREE].set(jnp.asarray(INITIAL_PI[PI_FREE]))
        delta = solve_all_deltas(INITIAL_SIGMA_DIAG, pi0, data).reshape(-1)
    # beta = (X1'Z W Z'X1)^-1 X1'Z W Z'delta  (linear IV-GMM with weight W)
    XZ = X1N.T @ ZN  # (K1, M)
    A = XZ @ W @ XZ.T
    b = XZ @ W @ (ZN.T @ delta)
    beta0 = jnp.linalg.solve(A, b)
    beta = euclidean_leaf(beta0)
    sigma = euclidean_leaf(INITIAL_SIGMA_DIAG)
    if spec == "rc":
        return RCParams(beta=beta, sigma=sigma)
    return BLPParams(beta=beta, sigma=sigma, pi=euclidean_leaf(INITIAL_PI[PI_FREE]))


def run_spec(spec, *, mode="fixed_W", cold=False):
    est = load_estimates()[spec]
    data = load_data(spec)
    model_type = "logit" if spec == "logit" else "rc"
    pi_free = PI_FREE if spec == "rc_demographics" else None
    psi = make_psi(data, model_type=model_type, pi_free_positions=pi_free)
    N = data.T * data.J
    measure = EmpiricalMeasure.from_arrays(product_index_x(data.T, data.J), M=data.M)
    W = load_W(spec)  # pyblp final optimal weight (M, M)
    theta_init = cold_theta_init(spec, data, W) if cold else build_theta_init(spec, est, data)

    if mode == "fixed_W":
        # pyblp's objective = N * gbar'W gbar and its SE = sqrt((G'WG)^-1 / N)
        # both correspond to treating the moment-vector covariance as
        # V_X = W^-1 / N. Feeding emu that same V_X as BOTH the fixed weight
        # and the covariance strategy makes emu's own formulas reproduce
        # pyblp's point estimate, objective, J-stat, AND standard errors.
        weighting = Fixed(V0=jnp.linalg.inv(W) / N)
        # SEs: pyblp uses the weight it RE-optimises at theta_hat (updated_W),
        # not the objective weight W. Use it for the covariance so emu's
        # Sigma_theta = (G'V_X^-1 G)^-1 matches pyblp's reported SEs.
        V_se = jnp.linalg.inv(load_updated_W(spec)) / N
        covariance = AnalyticalCovariance(covariance_fn=lambda model, theta: V_se)
        regularization = None
    else:  # iterated: emu builds its own optimal weight + robust covariance
        weighting = IteratedWeighting(weighting_iterations=5, weighting_tol=1e-8)
        covariance = IIDCovariance()
        regularization = DiagonalTikhonov(kappa_target=1e12)

    result = estimate(
        model=psi,
        measure=measure,
        covariance=covariance,
        weighting=weighting,
        regularization=regularization,
        theta_init=theta_init,
    )

    # In fixed_W mode V_X = W^-1/N, so emu's objective m'V_X^-1 m = N*gbar'W gbar
    # already matches pyblp's reported objective directly.
    emu_obj = float(result.diagnostics.final_objective)
    se = np.asarray(result.standard_errors.array).reshape(-1)
    return {
        "spec": spec,
        "mode": mode,
        "N": N,
        "theta_hat": flat_theta(result.theta_hat),
        "se": se,
        "emu_objective": emu_obj,
        "J_stat": float(result.J_stat),
        "J_dof": int(result.J_dof),
        "J_pvalue": float(result.J_pvalue),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "pyblp_objective": est["objective"],
        "pyblp_beta": np.array(est["beta"]).reshape(-1),
        "pyblp_beta_se": np.array(est["beta_se"]).reshape(-1),
        "pyblp_sigma": np.diag(np.array(est["sigma"])) if est["sigma"] else None,
        "pyblp_pi_free": (
            np.array(est["pi"])[PI_FREE] if spec == "rc_demographics" else None
        ),
    }


def report(r):
    print(f"\n{'='*70}\n  SPEC: {r['spec']}   (mode={r['mode']})\n{'='*70}")
    print(f"  converged={r['converged']}  iterations={r['iterations']}  N={r['N']}")
    b, s, p = r["theta_hat"]
    print("\n  -- point estimates --")
    print(f"  beta (emu)  : {b}")
    print(f"  beta (pyblp): {r['pyblp_beta']}")
    if s.size:
        print(f"  |sigma| (emu)  : {np.abs(s)}")
        print(f"  |sigma| (pyblp): {np.abs(r['pyblp_sigma'])}")
    if p.size and r["pyblp_pi_free"] is not None:
        print(f"  pi free (emu)  : {np.round(p, 4)}")
        print(f"  pi free (pyblp): {np.round(r['pyblp_pi_free'], 4)}")
        print(f"  pi diff max|emu-pyblp| : "
              f"{np.max(np.abs(p - r['pyblp_pi_free'])):.3e}")
    print("\n  -- objective --")
    print(f"  emu objective     : {r['emu_objective']:.6g}")
    print(f"  pyblp objective   : {r['pyblp_objective']:.6g}")
    print("\n  -- inference --")
    print(f"  emu beta SE       : {r['se'][:1]}")
    print(f"  pyblp beta SE     : {r['pyblp_beta_se']}")
    print(f"  J = {r['J_stat']:.4f}  dof={r['J_dof']}  p={r['J_pvalue']:.4f}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    specs = ["logit", "rc", "rc_demographics"] if which == "all" else [which]
    out = {}
    for spec in specs:
        r = run_spec(spec, mode="fixed_W")
        report(r)
        out[spec] = {k: (v.tolist() if isinstance(v, np.ndarray)
                         else [x.tolist() if isinstance(x, np.ndarray) else x
                               for x in v] if isinstance(v, tuple) else v)
                     for k, v in r.items() if k != "pyblp_pi"}
    (REF / "emu_results.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
