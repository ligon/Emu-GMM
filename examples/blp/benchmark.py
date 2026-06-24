"""Indicative wall-clock benchmark: emu-gmm vs pyblp on the headline spec.

NOT a rigorously controlled benchmark --- the two solve the same GMM problem
with different optimisers (pyblp: gradient BFGS; emu: Gauss-Newton LM), and emu
pays a one-time JIT-compile cost pyblp does not. Reported with those caveats in
REPORT.md "Performance". Run after build_reference.py.

Run:  .venv/bin/python examples/blp/benchmark.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyblp

from blp_data import load_data, load_estimates, load_W
from blp_model import make_psi, product_index_x
from run_emu import PI_FREE, build_theta_init, cold_theta_init

from emu_gmm import estimate
from emu_gmm.covariance import AnalyticalCovariance
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.weighting import Fixed

pyblp.options.verbose = False
SPEC = "rc_demographics"


def time_pyblp():
    prod = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    ag = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
    pf = (
        pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
        pyblp.Formulation("1 + prices + sugar + mushy"),
    )
    af = pyblp.Formulation("0 + income + income_squared + age + child")
    problem = pyblp.Problem(pf, prod, af, ag)
    s0 = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
    p0 = np.array([[5.4819, 0, .2037, 0], [15.8935, -1.2, 0, 2.6342],
                   [-.2506, 0, .0511, 0], [1.2650, 0, -.8091, 0]])
    opt = pyblp.Optimization("bfgs", {"gtol": 1e-6})
    t = time.perf_counter()
    problem.solve(s0, p0, optimization=opt, method="2s")
    return time.perf_counter() - t


def emu_fit_factory():
    data = load_data(SPEC)
    est = load_estimates()[SPEC]
    W = load_W(SPEC)
    N = data.T * data.J
    psi = make_psi(data, model_type="rc", pi_free_positions=PI_FREE)
    measure = EmpiricalMeasure.from_arrays(product_index_x(data.T, data.J), M=data.M)
    Vx = jnp.linalg.inv(W) / N

    def fit(theta_init):
        r = estimate(
            model=psi, measure=measure,
            covariance=AnalyticalCovariance(covariance_fn=lambda m, t: Vx),
            weighting=Fixed(V0=Vx), regularization=None, theta_init=theta_init,
        )
        jax.block_until_ready(r.theta_hat.beta.array)
        return r

    return fit, build_theta_init(SPEC, est, data), cold_theta_init(SPEC, data, W)


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return time.perf_counter() - t, r


def main():
    pt = time_pyblp()
    print(f"pyblp solve (BFGS, 2s)          : {pt:6.2f} s")

    fit, warm_init, cold_init = emu_fit_factory()
    w1, _ = timed(lambda: fit(warm_init))       # includes JIT compile
    w2, rw = timed(lambda: fit(warm_init))       # compiled, steady
    ct, rc = timed(lambda: fit(cold_init))       # compiled, cold start
    print(f"emu warm (incl JIT compile)     : {w1:6.2f} s")
    print(f"emu warm (compiled, steady)     : {w2:6.2f} s   iters={int(rw.iterations)}")
    print(f"emu cold (compiled, Nevo init)  : {ct:6.2f} s   iters={int(rc.iterations)}")


if __name__ == "__main__":
    main()
