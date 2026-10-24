"""Seeded weak-null calibration for mixed K inference (ledger Section 19).

Run in an allocated compute job with affinity bounded to its requested CPUs:

    PYTHONPATH=src python scripts/validation/mixed_k_calibration.py --n-reps 600

Each original sampling bundle pairs one T PSU with one S PSU. Fixed mode
holds two opposite stratum profiles fixed; random mode uses no fixed profile.
The design moment is observed on T only, the sampling moments on S only.
The first two moments use the same instrument, so their population cross
covariance is the signed ``rho / (2 * bundles_per_stratum)``. Opposite fixed
profiles enter as additive moment offsets; multiplying profiles by the
shared random instrument would add a positive nuisance cross covariance.
The population mean Jacobian is ``[-0.02, -0.02, 0]``. A regressor shock
correlated with outcomes also makes the score/moment TS cross nonzero.
Paired outcome innovations share signed correlation. All methods use the
same data, moment covariance and public K-statistic; ablations replace only
its score/moment cross covariance. JSON records include finite denominators
and Monte Carlo uncertainty. A fixture is a calibration check, not a theorem.
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
    k_statistic,
)


@jdc.pytree_dataclass
class Theta:
    value: float


def moment(x, theta):
    outcome, regressor, instrument, profile = x
    return (outcome - theta.value * regressor) * jnp.asarray(
        [instrument, instrument, instrument**2 - 1.0]
    ) + profile


def fixture(n_bundles, fixed, rho, n_reps, seed):
    """Generate paired Gaussian innovations with fixed cell counts."""
    rng = np.random.default_rng(seed)
    e_t, independent, instrument, regressor_noise = rng.normal(
        size=(4, n_reps, 2, n_bundles)
    )
    e_s = rho * e_t + np.sqrt(1.0 - rho**2) * independent
    innovation = np.stack([e_t, e_s], axis=-1)
    z = np.repeat(instrument[..., None], 2, axis=-1)
    profile = np.broadcast_to(
        np.asarray([2.0, -2.0])[None, :, None, None] if fixed else 0.0,
        innovation.shape,
    )
    noise = np.repeat(regressor_noise[..., None], 2, axis=-1)
    regressor = (
        0.02 * z + 0.7 * innovation + np.sqrt(1 - 0.7**2) * noise + 0.75 * profile
    )
    data = np.stack([innovation, regressor, z, profile], axis=-1).reshape(n_reps, -1, 4)
    n_rows = 4 * n_bundles
    strata = np.repeat(np.arange(2), 2 * n_bundles)
    arm = np.tile([0, 1], 2 * n_bundles)
    cell = 2 * strata + arm
    design = StratifiedCovariance(
        psu_ids=jnp.arange(n_rows, dtype=float),
        cell_ids=jnp.asarray(cell),
        stratum_ids=jnp.asarray(strata),
        n_psu=n_rows,
        n_cells=4,
        n_strata=2,
        fpc=False,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.repeat(jnp.arange(2 * n_bundles, dtype=float), 2),
        n_clusters=2 * n_bundles,
        dof_correction=not fixed,
    )
    covariance = DesignAwareCovariance.from_design_mask(
        design,
        sampling,
        jnp.asarray([1.0, 0.0, 0.0]),
        fixed_strata=fixed,
    )
    mask = np.zeros((n_rows, 3))
    mask[arm == 0, 0] = 1.0
    mask[arm == 1, 1:] = 1.0
    return jnp.asarray(data), jnp.asarray(mask), covariance


def evaluate(data, mask, covariance):
    theta = Theta(0.0)
    weights = jnp.ones(data.shape[0])
    measure = EmpiricalMeasure(x=data, mask=mask, weights=weights)
    matched = k_statistic(theta, measure, covariance, moment)
    g = measure.moment_contributions(moment, theta)
    D = measure.jacobian_contributions(moment, theta)
    N = jnp.sum(mask, axis=0)
    # Deliberately wrong dependence: observations replace paired bundles.
    iid_cross = jnp.einsum("nja,nk->ajk", D, g) / jnp.outer(N, N)[None]
    wrong = k_statistic(
        theta,
        measure,
        covariance,
        moment,
        score_cov_fn=lambda *_: iid_cross,
    )
    zero = k_statistic(
        theta,
        measure,
        covariance,
        moment,
        score_cov_fn=lambda *_: jnp.zeros_like(iid_cross),
    )
    cross = covariance.cross_block(moment, theta, measure)[0, 1]
    return jnp.asarray([matched.p_K, wrong.p_K, zero.p_K, cross])


def summarize(values, method, mode, rho, requested):
    valid = np.isfinite(values)
    used = values[valid]
    n = int(valid.sum())
    reject = int(np.count_nonzero(used < 0.05))
    rate = reject / n if n else None
    return {
        "event": "summary",
        "mode": mode,
        "rho": rho,
        "method": method,
        "n_requested": requested,
        "n_finite": n,
        "n_invalid": requested - n,
        "mean_p": float(used.mean()) if n else None,
        "mean_p_mcse": float(used.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
        "n_reject_5pct": reject,
        "rejection_rate": rate,
        "rejection_mcse": math.sqrt(rate * (1 - rate) / n) if n else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=600)
    parser.add_argument("--bundles-per-stratum", type=int, default=60)
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--mode", choices=["both", "fixed", "random"], default="both")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.8, -0.8])
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.n_reps < 2 or args.bundles_per_stratum < 2 or args.batch_size < 1:
        parser.error("reps/bundles must be >=2 and batch size must be positive")
    if any(not np.isfinite(rho) or abs(rho) >= 1 for rho in args.rho):
        parser.error("rho must be finite and strictly between -1 and 1")
    print(
        json.dumps(
            {
                "event": "config",
                "columns": ["matched", "iid_cross", "zero_cross", "moment_TS_01"],
                **vars(args),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    modes = ["fixed", "random"] if args.mode == "both" else [args.mode]
    run = jax.jit(jax.vmap(evaluate, in_axes=(0, None, None)))
    for mode in modes:
        for rho in args.rho:
            data, mask, covariance = fixture(
                args.bundles_per_stratum,
                mode == "fixed",
                rho,
                args.n_reps,
                args.seed,
            )
            blocks = []
            for start in range(0, args.n_reps, args.batch_size):
                block = np.asarray(
                    run(data[start : start + args.batch_size], mask, covariance)
                )
                blocks.append(block)
                print(
                    json.dumps(
                        {
                            "event": "batch",
                            "mode": mode,
                            "rho": rho,
                            "start": start,
                            "n_requested": len(block),
                            "n_finite": np.isfinite(block).sum(axis=0).tolist(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            values = np.concatenate(blocks)
            for column, method in enumerate(["matched", "iid_cross", "zero_cross"]):
                print(
                    json.dumps(
                        summarize(values[:, column], method, mode, rho, args.n_reps),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            cross = values[:, 3]
            finite = cross[np.isfinite(cross)]
            G = 2 * args.bundles_per_stratum
            # Random-mode dof correction multiplies the uncentered TS
            # estimate by G/(G-1); the population covariance is unchanged.
            target = rho / G
            expected = target * (G / (G - 1) if mode == "random" else 1.0)
            print(
                json.dumps(
                    {
                        "event": "cross_block_evidence",
                        "mode": mode,
                        "rho": rho,
                        "n_requested": args.n_reps,
                        "n_finite": len(finite),
                        "n_invalid": args.n_reps - len(finite),
                        "population_cross_target": target,
                        "expected_cross_estimate": expected,
                        "mean_cross_estimate": (
                            float(finite.mean()) if len(finite) else None
                        ),
                        "cross_estimate_mcse": (
                            float(finite.std(ddof=1) / math.sqrt(len(finite)))
                            if len(finite) > 1
                            else None
                        ),
                        "n_sign_matches": int(np.count_nonzero(rho * finite > 0)),
                        "population_mean_jacobian": [-0.02, -0.02, 0.0],
                        "population_score_cross_target": -0.7 * rho / G,
                        "population_moment_covariance": (
                            np.asarray(
                                [[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 2.0]]
                            )
                            / G
                        ).tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            # Release each case's compiled kernels before the next layout.
            jax.clear_caches()


if __name__ == "__main__":
    main()
