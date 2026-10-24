"""Bounded whole-stratum refit pilot, grounded in ledger Section 27.

Run one immutable-source shard per rho in an allocated job, for example::

    PYTHONPATH=src python scripts/validation/random_stratum_bootstrap_251.py \
        --rho 0.8 --outer-index 0 --nouter 6 --nboot 20 --seed 251

Independent iid strata contain q paired observations T=eps and
S=alpha+rho*eps+sqrt(1-rho**2)*eta, with Var(alpha)=tau**2. The two means
are just identified. Identity weighting makes the residual affine, permitting
linear_solver(verify=False); this is not general CUE calibration. All fit,
sandwich, delta-method and refit-SE calculations use public package APIs.
Regularization is reported as matrix diagnostics. Only the independent
population and empirical-stratum oracles use formulas.

JSONL records preserve invalid events and all denominators. The default
6 outer samples and 20 draws per sign are a feasibility pilot, not evidence
for broad coverage. The two signs share innovations and bootstrap indices.
Outer indices determine random streams independently of sharding/order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    DiagonalTikhonov,
    EmpiricalLaw,
    EmpiricalMeasure,
    Identity,
    StratifiedCovariance,
    estimate,
    linear_solver,
    mixed_bootstrap,
)


@jdc.pytree_dataclass
class Means:
    theta_t: float
    theta_s: float


def moment(row, theta):
    return jnp.asarray([row[0] - theta.theta_t, row[1] - theta.theta_s])


def contrast(components):
    return components[0] - components[1]


def emit(record):
    """Invalid numbers remain named events and serialize as JSON null."""

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, float | np.floating):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer | np.bool_):
            return value.item()
        return value

    print(json.dumps(clean(record), sort_keys=True, allow_nan=False), flush=True)


def rate_record(successes, denominator):
    """Explicit counts, binomial MCSE and Wilson95 interval (pilot summary)."""
    if denominator == 0:
        return {
            "numerator": successes,
            "denominator": 0,
            "rate": None,
            "mcse": None,
            "wilson95": None,
        }
    z = 1.959963984540054
    p = successes / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    half = (
        z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    )
    return {
        "numerator": successes,
        "denominator": denominator,
        "rate": p,
        "mcse": math.sqrt(p * (1 - p) / denominator),
        "wilson95": [max(0.0, center - half), min(1.0, center + half)],
    }


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    return {
        "n_requested": int(values.size),
        "n_finite": n,
        "n_invalid": int(values.size) - n,
        "mean": float(finite.mean()) if n else None,
        "mean_mcse": float(finite.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
    }


def fixture(args, outer_index, rho):
    """One observed PSU per row; paired shocks and arm masks are preserved."""
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, outer_index]))
    eps, eta = rng.normal(size=(2, args.nstrata, args.q))
    alpha = args.tau * rng.normal(size=(args.nstrata, 1))
    t = eps
    s = alpha + rho * eps + math.sqrt(1 - rho**2) * eta
    n_pairs = args.nstrata * args.q
    n = 2 * n_pairs
    data = np.zeros((n, 2))
    data[0::2, 0] = t.ravel()
    data[1::2, 1] = s.ravel()
    mask = np.tile([[1.0, 0.0], [0.0, 1.0]], (n_pairs, 1))
    strata = np.repeat(np.arange(args.nstrata), 2 * args.q)
    cell = 2 * strata + np.tile([0, 1], n_pairs)
    measure = EmpiricalMeasure(
        x=jnp.asarray(data), mask=jnp.asarray(mask), weights=jnp.ones(n)
    )
    design = StratifiedCovariance(
        psu_ids=jnp.arange(n, dtype=float),
        cell_ids=jnp.asarray(cell),
        stratum_ids=jnp.asarray(strata),
        n_psu=n,
        n_cells=2 * args.nstrata,
        n_strata=args.nstrata,
        fpc=False,
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(strata),
        n_clusters=args.nstrata,
        centered=False,
        dof_correction=not args.no_dof_correction,
    )
    covariance = DesignAwareCovariance.from_design_mask(
        design,
        sampling,
        jnp.asarray([1.0, 0.0]),
        couple=not args.no_couple,
        fixed_strata=False,
    )
    # Independent balanced empirical-stratum oracle (Section27). It does not
    # call DesignAwareCovariance and includes the finite-S empirical centering.
    bundles = np.column_stack([t.mean(axis=1), s.mean(axis=1)])
    deviations = bundles - bundles.mean(axis=0)
    conditional = deviations.T @ deviations / args.nstrata**2
    return measure, covariance, conditional


def one_case(args, outer_index, rho):
    started = time.monotonic()
    measure, covariance, conditional = fixture(args, outer_index, rho)
    theta0 = Means(0.0, 0.0)
    regularization = DiagonalTikhonov()
    optimizer, weighting = linear_solver(verify=False), Identity()
    fitted = estimate(
        moment,
        measure,
        covariance=covariance,
        theta_init=theta0,
        optimizer=optimizer,
        weighting=weighting,
        regularization=regularization,
    )
    fit_converged = bool(fitted.converged)
    law = fitted.asymptotic()
    value = float(law.mean(contrast)[0])
    raw_se = float(law.functional_se(contrast)[0][0])
    raw = np.asarray(fitted.moment_covariance)
    # V* is a matrix diagnostic only. The actual #133 sandwich above always
    # retains the original raw meat, including when the ridge binds.
    vstar, diagnostic_tau = regularization.apply(jnp.asarray(raw))
    bootstrap_seed = int(
        np.random.SeedSequence([args.seed, outer_index, 251]).generate_state(1)[0]
    )
    boot = mixed_bootstrap(
        moment,
        fitted.theta_hat,
        measure,
        covariance,
        n_boot=args.nboot,
        key=jax.random.PRNGKey(bootstrap_seed),
        weighting=weighting,
        optimizer=optimizer,
        regularization=regularization,
    )
    names = tuple(boot.param_names)
    draws = np.asarray(boot.theta_boot.array)
    ordered = draws[:, [names.index("theta_t"), names.index("theta_s")]]
    converged = np.asarray(boot.convergence, dtype=bool)
    finite = np.all(np.isfinite(ordered), axis=1)
    valid = converged & finite
    empirical = EmpiricalLaw.from_draws(
        np.where(valid[:, None], ordered, np.nan),
        names=("theta_t", "theta_s"),
        component_shapes=((), ()),
        events={
            "converged": converged,
            "zero_support": np.asarray(boot.zero_support),
            "tau_saturated": np.asarray(boot.tau_saturated),
        },
    )
    refit_se = float(empirical.se(contrast)[0])
    refit_cov = np.asarray(empirical.cov())
    S, q = args.nstrata, args.q
    target = np.asarray(
        [[1 / (S * q), rho / (S * q)], [rho / (S * q), args.tau**2 / S + 1 / (S * q)]]
    )
    # These are independent target-oracle formulas authorized by ledger27;
    # neither formula reconstructs a package estimator or its sandwich.
    oracle_se = math.sqrt(args.tau**2 / S + 2 * (1 - rho) / (S * q))
    conditional_se = math.sqrt(
        float(conditional[0, 0] + conditional[1, 1] - 2 * conditional[0, 1])
    )
    se = {
        "analytic_raw": raw_se,
        "whole_stratum_refit": refit_se,
        "population_oracle": oracle_se,
        "empirical_stratum_oracle": conditional_se,
    }
    valid_se = {
        name: fit_converged
        and np.isfinite(value)
        and np.isfinite(number)
        and number > 0
        for name, number in se.items()
    }
    coverage = {
        name: bool(abs(value) <= 1.959963984540054 * number) if valid_se[name] else None
        for name, number in se.items()
    }
    record = {
        "event": "case",
        "outer_index": outer_index,
        "rho": rho,
        "fit_converged": fit_converged,
        "fit_finite": bool(np.isfinite(value)),
        "fit_tau": float(fitted.diagnostics.tau_realised),
        "fit_tau_saturated": bool(fitted.diagnostics.tau_saturated),
        "fit_zero_support": bool(np.any(np.asarray(fitted.diagnostics.N_j.array) == 0)),
        "contrast_hat": value,
        "se": se,
        "valid_se": valid_se,
        "coverage95": coverage,
        "raw_covariance": raw,
        "regularized_covariance": np.asarray(vstar),
        "diagnostic_tau_at_fitted_raw_covariance": float(diagnostic_tau),
        "diagnostic_ridge_binding": bool(diagnostic_tau > regularization.tau_threshold),
        "diagnostic_tau_saturated": bool(
            regularization.tau_saturated(jnp.asarray(raw), diagnostic_tau)
        ),
        "raw_indefinite": bool(np.linalg.eigvalsh(raw)[0] < 0),
        "refit_covariance": refit_cov,
        "population_covariance": target,
        "empirical_stratum_covariance": conditional,
        "cross": {
            "analytic_raw": float(raw[0, 1]),
            "whole_stratum_refit": float(refit_cov[0, 1]),
            "empirical_stratum_oracle": float(conditional[0, 1]),
            "population_target": rho / (S * q),
        },
        "n_refit_requested": args.nboot,
        "n_refit_converged": boot.n_valid,
        "n_refit_nonconverged": boot.n_nonconverged,
        "n_refit_finite": int(finite.sum()),
        "n_refit_valid": int(valid.sum()),
        "n_refit_converged_nonfinite": int((converged & ~finite).sum()),
        "n_refit_zero_support": boot.n_zero_support,
        "n_refit_tau_saturated": boot.n_tau_saturated,
        "refit_contrasts": ordered[:, 0] - ordered[:, 1],
        "refit_convergence": converged,
        "seconds": time.monotonic() - started,
    }
    return record


def summarize(records, rho):
    usable = [r for r in records if r["fit_converged"] and r["fit_finite"]]
    result = {
        "event": "summary",
        "rho": rho,
        "n_outer_requested": len(records),
        "n_fit_converged": sum(r["fit_converged"] for r in records),
        "n_fit_valid": len(usable),
        "n_fit_invalid": len(records) - len(usable),
        "n_raw_indefinite": sum(r["raw_indefinite"] for r in records),
        "se": {},
        "cross": {},
    }
    for name in records[0]["se"]:
        used = [r for r in usable if r["valid_se"][name]]
        result["se"][name] = {
            **finite_summary(
                [r["se"][name] if r["valid_se"][name] else np.nan for r in records]
            ),
            "n_valid_se": len(used),
            "coverage95": rate_record(
                sum(r["coverage95"][name] for r in used), len(used)
            ),
        }
    for name in ("analytic_raw", "whole_stratum_refit", "empirical_stratum_oracle"):
        values = np.asarray(
            [
                r["cross"][name] if r["fit_converged"] and r["fit_finite"] else np.nan
                for r in records
            ]
        )
        finite = values[np.isfinite(values)]
        result["cross"][name] = {
            **finite_summary(values),
            "population_target": records[0]["cross"]["population_target"],
            "correct_sign": rate_record(int(np.sum(finite * rho > 0)), len(finite)),
        }
    estimates = [r["contrast_hat"] for r in usable]
    result["outer_dispersion"] = {
        "n_valid": len(estimates),
        "sd": (
            float(EmpiricalLaw.from_draws(estimates).se()[0])
            if len(estimates) >= 2
            else None
        ),
    }
    for field in (
        "n_refit_requested",
        "n_refit_converged",
        "n_refit_nonconverged",
        "n_refit_finite",
        "n_refit_valid",
        "n_refit_converged_nonfinite",
        "n_refit_zero_support",
        "n_refit_tau_saturated",
    ):
        result[field] = sum(r[field] for r in records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outer-index", type=int, default=0, help="first global outer index, inclusive"
    )
    parser.add_argument("--nouter", type=int, default=6)
    parser.add_argument("--nboot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=251)
    parser.add_argument("--nstrata", type=int, default=24)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.8, -0.8])
    parser.add_argument("--no-dof-correction", action="store_true")
    parser.add_argument("--no-couple", action="store_true")
    args = parser.parse_args()
    if args.outer_index < 0 or args.nouter < 1 or args.nboot < 2:
        parser.error("outer-index must be nonnegative, nouter>=1, nboot>=2")
    if not 0 <= args.seed < 2**32 or args.nstrata < 2 or args.q < 2:
        parser.error("require uint32 seed, nstrata>=2 and q>=2")
    if not np.isfinite(args.tau) or args.tau < 0:
        parser.error("tau must be finite and nonnegative")
    if any(not np.isfinite(r) or not 0 < abs(r) < 1 for r in args.rho):
        parser.error("rho must be finite with 0<abs(rho)<1")
    if len(set(args.rho)) != len(args.rho):
        parser.error("rho cases must be unique")
    import emu_gmm

    emit(
        {
            "event": "config",
            **vars(args),
            "package_file": emu_gmm.__file__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "balanced iid random-stratum two-mean feasibility pilot; not general CUE calibration",
            "weighting": "Identity",
            "optimizer": "linear_solver(verify=False); residual affine",
            "sampling_centered": False,
            "dof_correction": not args.no_dof_correction,
            "fixed_strata": False,
            "fpc": False,
            "couple": not args.no_couple,
            "coverage": "normal95 intervals, true contrast0; Wilson+MCSE use valid outer denominator",
            "empirical_oracle": "sum centered stratum means outer products / S**2; bootstrap finite-S target",
            "population_contrast_variance": "tau**2/S + 2*(1-rho)/(S*q)",
            "finite_S": "conditional bootstrap covariance has expectation (S-1)/S times population covariance; analytic SS/TS dof selected separately",
        }
    )
    for rho in args.rho:
        records = []
        for index in range(args.outer_index, args.outer_index + args.nouter):
            try:
                record = one_case(args, index, rho)
                emit(record)
                records.append(record)
            finally:
                # Bare public estimate/refits own fresh closures. The shared
                # refitter also clears every25 draws (ledger27 Section2).
                jax.clear_caches()
        emit(summarize(records, rho))


if __name__ == "__main__":
    main()
