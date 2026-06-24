"""De-risk the emu BLP model before wiring it into estimate().

Checks, against the saved pyblp reference:
  1. Contraction recovers pyblp's delta at pyblp's (sigma, pi).
  2. The moment scaling: emu's mean-over-products psi reproduces pyblp's
     sample moment g and GMM objective g'Wg.
  3. The custom_jvp IFT derivative matches finite differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

# This script does not import emu_gmm (which would enable x64 at import); the
# contraction needs float64 to converge to its 1e-13 tolerance.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyblp

from blp_data import load_data, load_W
from blp_model import (
    BLPParams,
    individual_mu,
    make_psi,
    market_index_x,
    predicted_shares,
    solve_delta,
)

REF = Path(__file__).parent / "reference"


def grouped_delta(spec):
    """pyblp delta grouped (T, J) by the same market_order as the arrays."""
    prod = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    mids = prod["market_ids"].to_numpy()
    order = pd.unique(mids)
    delta = np.load(REF / f"{spec}_delta.npy").reshape(-1)
    blocks = [delta[np.where(mids == m)[0]] for m in order]
    return jnp.asarray(np.stack(blocks, 0), dtype=jnp.float64)


def main():
    est = json.loads((REF / "estimates.json").read_text())
    spec = "rc_demographics"
    data = load_data(spec)
    e = est[spec]
    sigma = jnp.asarray(np.diag(np.array(e["sigma"])), dtype=jnp.float64)  # diag vec
    sigma = jnp.asarray(np.array(e["sigma"]), dtype=jnp.float64)
    sigma_diag = jnp.asarray(np.diag(np.array(e["sigma"])), dtype=jnp.float64)
    pi = jnp.asarray(np.array(e["pi"]), dtype=jnp.float64)
    beta = jnp.asarray(np.array(e["beta"]).reshape(-1), dtype=jnp.float64)

    print(f"dims: T={data.T} J={data.J} M={data.M} K1={data.K1} K2={data.K2} D={data.D}")

    # ── 1. contraction vs pyblp delta ──────────────────────────────────
    delta_pyblp = grouped_delta(spec)  # (T, J)

    def solve_one(t):
        mu = individual_mu(
            data.X2[t], data.nodes[t], data.demographics[t], sigma_diag, pi
        )
        return solve_delta(mu, jnp.log(data.shares[t]), data.agent_weights[t])

    delta_emu = jax.vmap(solve_one)(jnp.arange(data.T))  # (T, J)
    max_abs = float(jnp.max(jnp.abs(delta_emu - delta_pyblp)))
    print(f"\n[1] contraction vs pyblp delta: max|diff| = {max_abs:.3e}")

    # also confirm predicted shares match observed at the solution
    s0 = predicted_shares(delta_emu[0], individual_mu(
        data.X2[0], data.nodes[0], data.demographics[0], sigma_diag, pi),
        data.agent_weights[0])
    print(f"    market-0 share recovery max|diff| = "
          f"{float(jnp.max(jnp.abs(s0 - data.shares[0]))):.3e}")

    # ── 2. moment + objective scaling vs pyblp ─────────────────────────
    psi = make_psi(data, model_type="rc")
    theta = BLPParams(beta=beta, sigma=sigma_diag, pi=pi)
    x = market_index_x(data.T)
    psi_rows = jax.vmap(lambda xi: psi(xi, theta))(x)  # (T, M)
    g_emu = psi_rows.mean(axis=0)  # emu's sample moment (mean over markets)

    # pyblp's g from its own delta: (1/N) sum_jt Ztilde_jt xi_jt
    xi_pyblp = delta_pyblp - data.prices * beta[0]  # (T, J)
    g_pyblp = (data.Z * xi_pyblp[..., None]).reshape(-1, data.M).mean(axis=0)
    print(f"\n[2] g (emu)   first 3: {np.array(g_emu)[:3]}")
    print(f"    g (pyblp) first 3: {np.array(g_pyblp)[:3]}")
    print(f"    max|g_emu - g_pyblp| = {float(jnp.max(jnp.abs(g_emu - g_pyblp))):.3e}")

    W = load_W(spec)
    obj_emu = float(g_emu @ W @ g_emu)
    obj_pyblp_g = float(g_pyblp @ W @ g_pyblp)
    print(f"    pyblp reported objective : {e['objective']:.6g}")
    print(f"    g'Wg  (emu moment)       : {obj_emu:.6g}")
    print(f"    g'Wg  (pyblp delta)      : {obj_pyblp_g:.6g}")
    # try common pyblp scalings to identify the convention
    for scale_name, s in [("x1", 1.0), ("xN", data.T * data.J), ("xN^2", (data.T*data.J)**2)]:
        print(f"      scale {scale_name}: {obj_emu * s:.6g}")

    # ── 3. custom_jvp vs finite differences ────────────────────────────
    t0 = 0
    mu0 = individual_mu(data.X2[t0], data.nodes[t0], data.demographics[t0],
                        sigma_diag, pi)
    logS0 = jnp.log(data.shares[t0])
    aw0 = data.agent_weights[t0]
    key = jax.random.PRNGKey(0)
    dmu = jax.random.normal(key, mu0.shape)
    _, jvp_val = jax.jvp(lambda m: solve_delta(m, logS0, aw0), (mu0,), (dmu,))
    eps = 1e-6
    fd = (solve_delta(mu0 + eps * dmu, logS0, aw0)
          - solve_delta(mu0 - eps * dmu, logS0, aw0)) / (2 * eps)
    print(f"\n[3] custom_jvp vs finite-diff: max|diff| = "
          f"{float(jnp.max(jnp.abs(jvp_val - fd))):.3e}")


if __name__ == "__main__":
    main()
