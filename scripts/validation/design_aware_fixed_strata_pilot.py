"""Issue 145: fixed-profile estimator pilot and paired-cluster cross check.

Run from the repository root with a bounded CPU affinity, for example::

    nice -n 19 taskset -c 32-35 poetry run python \
        scripts/validation/design_aware_fixed_strata_pilot.py \
        --fit-reps 6 --paired-reps 256 --seed 20260610

The small fit run is a feasibility check, not a coverage claim. Larger fit
runs use the same seed and per-rep anchoring as ladder_mc study 3.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import ladder_mc as ladder
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
)


def _identity(x: Any, theta: Any) -> Any:
    del theta
    return x


@jdc.pytree_dataclass
class _ExtraGlobalAblation:
    """Validation-only counterfactual applying #119 after the local correction."""

    base: DesignAwareCovariance

    def covariance(
        self, psi: Any, theta: Any, measure: Any, cached_intermediates: Any = None
    ) -> Any:
        V = self.base.covariance(psi, theta, measure, cached_intermediates)
        factor = self.base.sampling._finite_cluster_correction(measure.mask)
        t = (self.base.design_moment_mask > 0.0).astype(V.dtype)
        tt = jnp.outer(t, t)
        return V * tt + V * (1.0 - tt) * factor

    def cross_block(
        self, psi: Any, theta: Any, measure: Any, cached_intermediates: Any = None
    ) -> Any:
        cross = self.base.cross_block(psi, theta, measure, cached_intermediates)
        return cross * self.base.sampling._finite_cluster_correction(measure.mask)


def run_fit_pilot(n_reps: int, seed: int) -> None:
    """Compare the existing #130 arms with both opt-in dof conventions."""
    design = ladder.make_design(ladder.DesignSpec(psu_per_cell=5, sigma_strat=0.7))
    base_covariance_arm = ladder.covariance_arm

    def covariance_arm(d: ladder.Design, arm: str) -> Any:
        if arm not in ("fixed_local", "fixed_plus_global"):
            return base_covariance_arm(d, arm)
        base = base_covariance_arm(d, "design_aware_nodof")
        local = DesignAwareCovariance.from_design_mask(
            base.design,
            base.sampling,
            base.design_moment_mask,
            fixed_strata=True,
        )
        return local if arm == "fixed_local" else _ExtraGlobalAblation(local)

    ladder.covariance_arm = covariance_arm
    print(f"fit_pilot seed={seed} reps={n_reps} profile=0.7 per_rep_anchor=True")
    print(ladder.ORG_HEADER)
    for arm in ("design_aware", "fixed_local", "fixed_plus_global", "stratified"):
        result = ladder.run_arm_per_rep_anchor(design, arm, n_reps=n_reps, seed=seed)
        print(ladder.org_row(arm, result))
        print(f"validity {arm}: {json.dumps(result.validity(), sort_keys=True)}")


def run_paired_cluster_check(n_reps: int, seed: int) -> None:
    """Check a genuine cross-arm shock with nested paired sampling clusters."""
    n_strata, pairs_per_stratum = 3, 5
    n_pairs = n_strata * pairs_per_stratum
    pair_stratum = jnp.repeat(jnp.arange(n_strata), pairs_per_stratum)
    row_stratum = jnp.repeat(pair_stratum, 2)
    psu = jnp.arange(2 * n_pairs)
    cell = 2 * row_stratum + jnp.tile(jnp.arange(2), n_pairs)
    cluster = jnp.repeat(jnp.arange(n_pairs), 2)
    mask = jnp.tile(jnp.array([[1.0, 0.0], [0.0, 1.0]]), (n_pairs, 1))
    weights = jnp.ones(2 * n_pairs)
    design = StratifiedCovariance(
        psu_ids=psu,
        cell_ids=cell,
        stratum_ids=row_stratum,
        n_psu=2 * n_pairs,
        n_cells=2 * n_strata,
        n_strata=n_strata,
    )
    local_sampling = ClusteredCovariance(cluster_ids=cluster, n_clusters=n_pairs)
    local = DesignAwareCovariance.from_design_mask(
        design, local_sampling, jnp.array([1.0, 0.0]), fixed_strata=True
    )
    plus_global = _ExtraGlobalAblation(local)
    profile_t = jnp.array([-2.0, 1.0, 1.0])[pair_stratum]
    profile_s = jnp.array([1.0, -1.0, 0.0])[pair_stratum]

    def one_draw(key: Any, rho: Any) -> tuple[Any, Any, Any]:
        k_z, k_t, k_s = jax.random.split(key, 3)
        shock = jax.random.normal(k_z, (n_pairs,))
        t = profile_t + shock + 0.4 * jax.random.normal(k_t, (n_pairs,))
        s = profile_s + rho * shock + 0.4 * jax.random.normal(k_s, (n_pairs,))
        x = jnp.zeros((2 * n_pairs, 2))
        x = x.at[0::2, 0].set(t)
        x = x.at[1::2, 1].set(s)
        measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        means = jnp.sum(x, axis=0) / n_pairs
        cross_local = local.cross_block(_identity, None, measure)[0, 1]
        cross_global = plus_global.cross_block(_identity, None, measure)[0, 1]
        return means, cross_local, cross_global

    keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.PRNGKey(seed), r))(
        jnp.arange(n_reps)
    )
    batch = jax.jit(jax.vmap(one_draw, in_axes=(0, None)))
    print(
        f"paired_cluster seed={seed} reps={n_reps} "
        f"strata={n_strata} pairs_per_stratum={pairs_per_stratum}"
    )
    for rho in (0.8, -0.8, 0.0):
        means, cross_local, cross_global = batch(keys, rho)
        mean_np = np.asarray(means)
        local_np = np.asarray(cross_local)
        global_np = np.asarray(cross_global)
        print(
            json.dumps(
                {
                    "rho": rho,
                    "analytic_cross": rho / n_pairs,
                    "empirical_cross": float(np.cov(mean_np.T)[0, 1]),
                    "estimated_local": float(local_np.mean()),
                    "local_mcse": float(local_np.std(ddof=1) / np.sqrt(n_reps)),
                    "estimated_plus_global": float(global_np.mean()),
                    "plus_global_mcse": float(global_np.std(ddof=1) / np.sqrt(n_reps)),
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-reps", type=int, default=6)
    parser.add_argument("--paired-reps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()
    if args.fit_reps < 0 or args.fit_reps == 1 or args.paired_reps < 2:
        parser.error("fit-reps must be zero or at least two; paired-reps at least two")
    if args.fit_reps:
        run_fit_pilot(args.fit_reps, args.seed)
    run_paired_cluster_check(args.paired_reps, args.seed)


if __name__ == "__main__":
    main()
