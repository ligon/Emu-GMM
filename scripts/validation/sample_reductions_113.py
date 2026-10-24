"""Weighted unbalanced-panel oracle for issue 113's model contract.

Projected instruments preserve means and derivatives, but residual projection
changes cluster covariance. Grounding: .coder/design-113-reductions.org.
This uses synthetic data and the package covariance, not consumer data.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from emu_gmm import ClusteredCovariance, EmpiricalMeasure


def main() -> None:
    rng = np.random.default_rng(113)
    n = 90
    product = rng.integers(0, 7, n)
    market = rng.integers(0, 6, n)
    fixed_effects = np.column_stack([np.eye(7)[product], np.eye(6)[market]])
    weights = rng.uniform(0.5, 2, n)
    mask = (rng.uniform(size=(n, 2)) > 0.2).astype(float)
    instruments = rng.normal(size=(n, 2))
    outcome = jnp.asarray(rng.normal(size=n) + 3 * product - 2 * market)
    attributes = jnp.asarray(rng.normal(size=n))
    projectors = []
    projected_instruments = np.zeros_like(instruments)
    for j in range(2):
        exposure = weights * mask[:, j]
        residualizer = np.eye(n) - fixed_effects @ np.linalg.pinv(
            fixed_effects.T @ (exposure[:, None] * fixed_effects)
        ) @ (fixed_effects.T * exposure)
        projectors.append(residualizer)
        projected_instruments[:, j] = residualizer @ instruments[:, j]
    residualizers = jnp.asarray(np.stack(projectors))
    instruments_jax = jnp.asarray(projected_instruments)
    measure = EmpiricalMeasure(
        x=jnp.arange(n, dtype=float)[:, None],
        mask=jnp.asarray(np.tile(mask, (1, 2))),
        weights=jnp.asarray(weights),
    )

    def model(row, slope):
        i = row[0].astype(int)
        transformed = jnp.exp(-slope * attributes) * outcome
        projected = jnp.einsum("jkl,l->jk", residualizers, transformed)
        return jnp.concatenate(
            [instruments_jax[i] * projected[:, i], instruments_jax[i] * transformed[i]]
        )

    slope = jnp.asarray(0.25)
    moment = np.asarray(measure.expectation(model, slope))
    derivative = np.asarray(jax.jacfwd(lambda t: measure.expectation(model, t))(slope))
    covariance = np.asarray(
        ClusteredCovariance(
            cluster_ids=jnp.asarray(market, dtype=float), n_clusters=6
        ).covariance(model, slope, measure)
    )
    mean_gap = float(np.max(abs(moment[:2] - moment[2:])))
    derivative_gap = float(np.max(abs(derivative[:2] - derivative[2:])))
    covariance_gap = float(np.max(abs(covariance[:2, :2] - covariance[2:, 2:])))
    assert mean_gap < 1e-12 and derivative_gap < 1e-12
    assert covariance_gap > 0.1
    print(
        json.dumps(
            dict(
                seed=113,
                n=n,
                max_mean_gap=mean_gap,
                max_derivative_gap=derivative_gap,
                max_cluster_covariance_gap=covariance_gap,
                projected_diag=np.diag(covariance[:2, :2]).tolist(),
                raw_diag=np.diag(covariance[2:, 2:]).tolist(),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
