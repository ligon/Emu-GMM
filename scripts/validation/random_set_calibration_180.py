"""Known-DGP finite-grid coverage and persistence check (issue 180).

Grounding: .coder/design-180-calibration.org. This uses package K inference
and confidence-set assembly, not a parallel statistic implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import EmpiricalMeasure, IIDCovariance, RandomSetLaw, k_statistic
from emu_gmm.inference.confidence_set import _assemble_set


@jdc.pytree_dataclass
class Theta:
    value: float


def moment(row, theta):
    outcome, regressor = row
    residual = outcome - regressor * theta.value
    return residual * jnp.array([regressor, regressor**2, 1.0])


def p_at(data, value):
    measure = EmpiricalMeasure(
        x=data,
        mask=jnp.ones((data.shape[0], 3)),
        weights=jnp.ones(data.shape[0]),
    )
    return k_statistic(Theta(value), measure, IIDCovariance(), moment).p_K


def law_from_p(grid, p_values):
    sets = [
        _assemble_set(
            grid,
            [SimpleNamespace(p_K=float(p)) for p in row],
            "K",
            0.05,
            False,
        )
        for row in p_values
    ]
    return RandomSetLaw.from_k_confidence_sets(sets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=600)
    parser.add_argument("--n", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=180)
    args = parser.parse_args()
    if args.n_reps < 2 or args.n < 10 or args.batch_size < 1:
        parser.error("n-reps>=2, n>=10 and positive batch-size required")
    rng = np.random.default_rng(args.seed)
    regressor = rng.normal(size=(args.n_reps, args.n))
    outcome = regressor + rng.normal(size=regressor.shape)
    data = np.stack([outcome, regressor], axis=-1)
    coarse = np.linspace(0.6, 1.4, 5)
    fine = np.linspace(0.6, 1.4, 9)
    # Use exact common values to avoid confusing floating representation with
    # mathematical grid refinement; 1.0 is scanned in both grids.
    coarse = fine[::2].copy()
    evaluate = jax.jit(jax.vmap(jax.vmap(p_at, in_axes=(None, 0)), in_axes=(0, None)))
    values = []
    for grid in (coarse, fine):
        blocks = [
            np.asarray(
                evaluate(jnp.asarray(data[start : start + args.batch_size]), grid)
            )
            for start in range(0, args.n_reps, args.batch_size)
        ]
        values.append(np.concatenate(blocks))
    common_gap = float(np.max(np.abs(values[0] - values[1][:, ::2])))
    np.testing.assert_allclose(values[0], values[1][:, ::2], atol=1e-12, rtol=1e-12)
    reports = []
    for grid, p_values in zip((coarse, fine), values, strict=True):
        law = law_from_p(grid, p_values)
        report = law.coverage(1.0)
        assert report == law.capacity([1.0])
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "random-set.npz"
            law.save(artifact)
            restored = RandomSetLaw.load(artifact)
            assert restored.coverage(1.0) == report
            assert restored.capacity(coarse) == law.capacity(coarse)
            assert restored.event_share("grid_empty") == law.event_share("grid_empty")
        mcse = math.sqrt(0.95 * 0.05 / report.n_decidable)
        passed = abs(report.estimate - 0.95) <= 4 * mcse
        reports.append(
            {
                "grid_size": len(grid),
                **asdict(report),
                "nominal_coverage": 0.95,
                "null_mcse": mcse,
                "within_four_mcse": passed,
                "grid_empty": asdict(law.event_share("grid_empty")),
                "persistence_parity": True,
            }
        )
    assert reports[0]["n_hit"] == reports[1]["n_hit"]
    print(
        json.dumps(
            {
                "config": vars(args),
                "backend": jax.default_backend(),
                "x64": bool(jax.config.jax_enable_x64),
                "common_point_max_p_gap": common_gap,
                "reports": reports,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not all(report["within_four_mcse"] for report in reports):
        raise SystemExit(
            "known-DGP coverage lies outside the declared falsification band"
        )


if __name__ == "__main__":
    main()
