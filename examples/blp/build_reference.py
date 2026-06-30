"""Build the pyblp Nevo reference and extract its internal arrays.

This script is the *ground truth* side of the emu-gmm vs pyblp BLP
comparison (see examples/blp/README for the full plan). It:

1. Builds the pyblp Nevo "fake cereal" Problem in three nested specs:
   - ``logit``           : plain multinomial logit (no random coefficients)
   - ``rc``              : random-coefficients logit, sigma only
   - ``rc_demographics`` : RC logit + demographic interactions (pi) --- the
                           headline Nevo tutorial spec.
2. Solves each with pyblp and records its estimates (beta, sigma, pi and
   their SEs, the GMM objective, the final optimal weight matrix W).
3. Extracts pyblp's *exact internal arrays* --- the demand instruments ZD,
   linear chars X1, nonlinear chars X2, observed shares, and the agent
   integration nodes / demographics / weights --- grouped by market into
   dense (T, J, .) / (T, I, .) tensors (the data is balanced: every market
   has J=24 products and I=20 agents).

The emu side (``run_emu.py``) loads these arrays and re-solves the SAME GMM
problem so any difference is isolated to the moment construction + optimiser,
not to data wrangling or the weight-matrix estimation step.

Run:  .venv/bin/python examples/blp/build_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

pyblp.options.verbose = False

OUT = Path(__file__).parent / "reference"
OUT.mkdir(exist_ok=True)

# Canonical Nevo tutorial initial values (pyblp docs). The optimiser starts
# here for the RC specs; logit has no nonlinear parameters.
INITIAL_SIGMA = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
INITIAL_PI = np.array(
    [
        [5.4819, 0.0, 0.2037, 0.0],
        [15.8935, -1.2000, 0.0, 2.6342],
        [-0.2506, 0.0, 0.0511, 0.0],
        [1.2650, 0.0, -0.8091, 0.0],
    ]
)

OPT = pyblp.Optimization("bfgs", {"gtol": 1e-6})


def _to_list(a):
    return None if a is None else np.asarray(a).tolist()


def group_by_market(values, market_ids, market_order):
    """Stack a (N, ...) array into (T, per_market, ...) by market.

    Assumes a balanced panel: every market has the same count. Preserves
    within-market row order (the order rows appear in the source frame).
    """
    market_ids = np.asarray(market_ids).reshape(-1)
    blocks = []
    for m in market_order:
        idx = np.where(market_ids == m)[0]
        blocks.append(values[idx])
    counts = {b.shape[0] for b in blocks}
    assert len(counts) == 1, f"unbalanced panel: per-market counts {counts}"
    return np.stack(blocks, axis=0)


def brand_demean(Z, product_ids):
    """Within-product (brand) demeaning: the FE-absorption projection A.

    pyblp's ``absorb='C(product_ids)'`` partials the 24 brand fixed effects
    out of the demand moment. Because A is symmetric idempotent, the moment
    Z'A(delta - X1 b) = (AZ)'(delta - X1 b), so demeaning the instruments
    alone reproduces the absorbed moment with the raw delta / prices and no
    explicit dummies. Returns AZ.
    """
    Z = np.asarray(Z, dtype=np.float64)
    pid = np.asarray(product_ids).reshape(-1)
    out = Z.copy()
    for p in np.unique(pid):
        m = pid == p
        out[m] = Z[m] - Z[m].mean(axis=0, keepdims=True)
    return out


def extract_arrays(problem, product_data, agent_data):
    """Pull pyblp's internal arrays into market-grouped dense tensors."""
    P = problem.products
    market_ids = np.asarray(P.market_ids).reshape(-1)
    market_order = pd.unique(market_ids)  # source order, stable

    shares = np.asarray(P.shares, dtype=np.float64)  # (N, 1)
    X1 = np.asarray(P.X1, dtype=np.float64)  # (N, K1) = prices
    X2 = np.asarray(P.X2, dtype=np.float64)  # (N, K2)
    ZD = np.asarray(P.ZD, dtype=np.float64)  # (N, MD)
    prices = np.asarray(P.prices, dtype=np.float64)  # (N, 1)
    product_ids = np.asarray(product_data["product_ids"]).reshape(-1)

    # Brand-demeaned instruments (the FE-absorption projection applied to Z).
    ZD_tilde = brand_demean(ZD, product_ids)

    arrays = {
        "market_order": np.asarray([str(m) for m in market_order]),
        "shares": group_by_market(shares, market_ids, market_order),  # (T,J,1)
        "X1": group_by_market(X1, market_ids, market_order),  # (T,J,K1)
        "X2": group_by_market(X2, market_ids, market_order),  # (T,J,K2)
        "ZD": group_by_market(ZD, market_ids, market_order),  # (T,J,MD)
        "ZD_tilde": group_by_market(ZD_tilde, market_ids, market_order),
        "prices": group_by_market(prices, market_ids, market_order),  # (T,J,1)
    }

    if agent_data is not None:
        A = problem.agents
        a_market = np.asarray(A.market_ids).reshape(-1)
        nodes = np.asarray(A.nodes, dtype=np.float64)  # (I_tot, K2)
        weights = np.asarray(A.weights, dtype=np.float64)  # (I_tot, 1)
        arrays["nodes"] = group_by_market(nodes, a_market, market_order)
        arrays["agent_weights"] = group_by_market(weights, a_market, market_order)
        demo = np.asarray(A.demographics, dtype=np.float64)  # (I_tot, D)
        if demo.size:
            arrays["demographics"] = group_by_market(demo, a_market, market_order)
    return arrays


def solve_spec(name, product_formulations, product_data, agent_formulation=None,
               agent_data=None, sigma=None, pi=None):
    print(f"\n=== solving spec: {name} ===")
    if agent_formulation is not None:
        problem = pyblp.Problem(
            product_formulations, product_data, agent_formulation, agent_data
        )
    elif agent_data is not None:
        problem = pyblp.Problem(product_formulations, product_data, agent_data=agent_data)
    else:
        problem = pyblp.Problem(product_formulations, product_data)

    print(problem)
    if sigma is None:
        results = problem.solve()  # logit: linear IV-GMM, no nonlinear params
    else:
        results = problem.solve(sigma=sigma, pi=pi, optimization=OPT, method="2s")
    print(results)

    estimates = {
        "spec": name,
        "N": int(problem.N),
        "T": int(problem.T),
        "K1": int(problem.K1),
        "K2": int(problem.K2),
        "D": int(problem.D),
        "MD": int(problem.MD),
        "beta": _to_list(results.beta),
        "beta_se": _to_list(results.beta_se),
        "X1_labels": [str(f) for f in problem._X1_formulations],
        "X2_labels": [str(f) for f in problem._X2_formulations],
        "demographics_labels": [str(f) for f in problem._demographics_formulations],
        "sigma": _to_list(results.sigma),
        "sigma_se": _to_list(results.sigma_se),
        "pi": _to_list(results.pi),
        "pi_se": _to_list(results.pi_se),
        "objective": float(np.asarray(results.objective).reshape(-1)[0]),
        "gradient_norm": (
            float(np.max(np.abs(np.asarray(results.gradient))))
            if results.gradient is not None and np.asarray(results.gradient).size
            else None
        ),
    }

    # Final optimal weight matrix (used in the objective) and the re-optimal
    # weight pyblp recomputes at theta_hat for its reported SEs, plus delta.
    np.save(OUT / f"{name}_W.npy", np.asarray(results.W, dtype=np.float64))
    if getattr(results, "updated_W", None) is not None:
        np.save(OUT / f"{name}_updated_W.npy",
                np.asarray(results.updated_W, dtype=np.float64))
    np.save(OUT / f"{name}_delta.npy", np.asarray(results.delta, dtype=np.float64))
    return problem, results, estimates


def main():
    product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)

    all_estimates = {}

    # --- Stage 1: plain logit (no random coefficients) ---
    problem, results, est = solve_spec(
        "logit",
        pyblp.Formulation("prices", absorb="C(product_ids)"),
        product_data,
    )
    arrays = extract_arrays(problem, product_data, None)
    np.savez(OUT / "logit_arrays.npz", **arrays)
    all_estimates["logit"] = est

    # --- Stage 2: RC logit, sigma only (no demographics) ---
    pf = (
        pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
        pyblp.Formulation("1 + prices + sugar + mushy"),
    )
    problem, results, est = solve_spec(
        "rc", pf, product_data, agent_data=agent_data, sigma=INITIAL_SIGMA, pi=None
    )
    arrays = extract_arrays(problem, product_data, agent_data)
    np.savez(OUT / "rc_arrays.npz", **arrays)
    all_estimates["rc"] = est

    # --- Stage 3: RC logit + demographics (the headline tutorial spec) ---
    af = pyblp.Formulation("0 + income + income_squared + age + child")
    problem, results, est = solve_spec(
        "rc_demographics", pf, product_data, agent_formulation=af,
        agent_data=agent_data, sigma=INITIAL_SIGMA, pi=INITIAL_PI,
    )
    arrays = extract_arrays(problem, product_data, agent_data)
    np.savez(OUT / "rc_demographics_arrays.npz", **arrays)
    all_estimates["rc_demographics"] = est

    (OUT / "estimates.json").write_text(json.dumps(all_estimates, indent=2))
    print("\nSaved reference to", OUT)
    for k, v in all_estimates.items():
        print(f"  {k}: objective={v['objective']:.6g}  beta={v['beta']}")


if __name__ == "__main__":
    main()
