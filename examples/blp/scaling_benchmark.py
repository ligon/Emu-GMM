"""Scaling micro-benchmarks for the emu-gmm BLP path.

Two curves that make the §4 "Performance" scaling discussion concrete:

(A) Draws-scaling: per-evaluation cost (the share-inversion contraction and the
    LM Jacobian through it) as the number of Monte-Carlo integration draws I
    grows. This is the dimension that turns the per-market work from a tiny
    matmul into one big enough to saturate many cores / a GPU.

(B) vmap-batched throughput: time to evaluate the moment at R different
    parameter vectors, sequentially vs as one vmapped batch. This is the
    bootstrap / Monte-Carlo / replicate pattern --- the case that scales
    near-linearly with added hardware.

Both report the visible CPU count so a cores sweep is just::

    for c in 0 0-1 0-3; do taskset -c $c .venv/bin/python examples/blp/scaling_benchmark.py A; done

GPU: install a CUDA jaxlib and run unchanged (JAX picks up the device). Note BLP
needs float64 --- prefer a datacenter GPU; consumer cards throttle fp64.

Run:  .venv/bin/python examples/blp/scaling_benchmark.py [A|B|all]
"""

from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from blp_data import load_data
from blp_model import RCParams, euclidean_leaf, make_psi, product_index_x

from emu_gmm.measures import EmpiricalMeasure

DRAWS = [20, 100, 500, 2000]
BATCH = [1, 2, 4, 8, 16, 32]
WARMUP, REPS = 2, 5


def _ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _bench(fn, *args):
    """Median wall-clock (ms) of a JIT-compiled fn, after warmup."""
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def make_model_with_draws(base, I, seed=0):
    """Rebuild the rc BLPData with I standard-normal integration draws/market."""
    key = jax.random.PRNGKey(seed)
    T, K2 = base.T, base.K2
    nodes = jax.random.normal(key, (T, I, K2), dtype=jnp.float64)
    weights = jnp.full((T, I), 1.0 / I, dtype=jnp.float64)
    import dataclasses

    data = dataclasses.replace(
        base, nodes=nodes, agent_weights=weights,
        demographics=jnp.zeros((T, I, 0), dtype=jnp.float64),
    )
    psi = make_psi(data, model_type="rc")
    measure = EmpiricalMeasure.from_arrays(product_index_x(T, data.J), M=data.M)
    theta = RCParams(
        beta=euclidean_leaf(jnp.asarray([-30.0])),
        sigma=euclidean_leaf(jnp.asarray([0.2, 2.4, 0.01, 0.18])),
    )
    return psi, measure, theta


def bench_draws():
    base = load_data("rc")
    print(f"\n(A) draws-scaling   [cores visible: {_ncpu()}]")
    print(f"{'I (draws)':>10} {'agents N':>10} {'expectation ms':>16} {'jacobian ms':>14}")
    for I in DRAWS:
        psi, measure, theta = make_model_with_draws(base, I)
        exp_fn = jax.jit(lambda th: measure.expectation(psi, th))
        jac_fn = jax.jit(lambda th: measure.jacobian(psi, th))
        te = _bench(exp_fn, theta)
        tj = _bench(jac_fn, theta)
        print(f"{I:>10} {base.T * I:>10} {te:>16.2f} {tj:>14.2f}")


def bench_batched():
    base = load_data("rc")
    psi, measure, theta = make_model_with_draws(base, 100)
    exp_fn = jax.jit(lambda th: measure.expectation(psi, th))

    # R perturbed thetas (stacked along a leading axis for vmap).
    def perturbed(R, seed=1):
        key = jax.random.PRNGKey(seed)
        b = -30.0 + 0.5 * jax.random.normal(key, (R, 1))
        s = jnp.asarray([0.2, 2.4, 0.01, 0.18])[None] * (
            1 + 0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (R, 4))
        )
        return [RCParams(beta=euclidean_leaf(b[i]), sigma=euclidean_leaf(s[i]))
                for i in range(R)], (b, s)

    batched = jax.jit(
        jax.vmap(lambda b, s: measure.expectation(
            psi, RCParams(beta=euclidean_leaf(b), sigma=euclidean_leaf(s))))
    )
    print(f"\n(B) vmap-batched throughput   [cores visible: {_ncpu()}]  (I=100 draws)")
    print(f"{'R fits':>8} {'sequential ms':>14} {'vmapped ms':>12} {'speedup':>9} {'ms/fit (vmap)':>14}")
    for R in BATCH:
        thetas, (b, s) = perturbed(R)
        # sequential: R separate compiled calls
        for _ in range(WARMUP):
            for th in thetas:
                jax.block_until_ready(exp_fn(th))
        t = time.perf_counter()
        for _ in range(REPS):
            for th in thetas:
                jax.block_until_ready(exp_fn(th))
        seq = (time.perf_counter() - t) / REPS * 1e3
        # vmapped: one batched call
        tv = _bench(batched, b, s)
        print(f"{R:>8} {seq:>14.2f} {tv:>12.2f} {seq / tv:>8.1f}x {tv / R:>14.2f}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        bench_draws()
    if which in ("B", "all"):
        bench_batched()


if __name__ == "__main__":
    main()
