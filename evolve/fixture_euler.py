"""Known-truth fixtures for the emu-gmm evolve harness (Euler DGP).

A *fixture* is a frozen (master key, n_sim, R) triple on the bundled
multi-asset Euler economy, whose ground truth is exact by construction
(BETA_TRUE = 0.96, GAMMA_TRUE = 2.0; every population Euler moment
vanishes at the truth -- src/emu_gmm/examples/euler.py).  Replicate r
draws with jax.random.fold_in(master, r) (the package's CRN contract),
so fixtures are fully deterministic and nothing needs to be stored
except the registered baseline statistics.

Pattern imported from AggLC Computation/evolve_pilot (fixture cache +
truth-anchored evaluator + hold-outs); the gate here is GOLDEN-FIXTURE
truth recovery, not a certificate -- see README.org for the epistemic
difference (P4 refinement: this verifies instances of THIS DGP).
"""

import json
import os
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax  # noqa: E402

from emu_gmm import (  # noqa: E402
    SyntheticCovariance,
    SyntheticMeasure,
    build_estimator,
    optimistix_lm,
)
from emu_gmm.examples.euler import (  # noqa: E402
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)

THETA_INIT = EulerParams(beta=0.9, gamma=1.0)

REGISTRY = {
    # Fitness fixtures: both enter every candidate's score.
    "euler_n800_k0": dict(n_sim=800, key=0, n_reps=24, role="fitness"),
    "euler_n200_k1": dict(n_sim=200, key=1, n_reps=24, role="fitness"),
    # Hold-outs: NEVER scored during evolution (unseen keys AND unseen
    # sample sizes; the AggLC B3 lesson).
    "euler_n3000_k7": dict(n_sim=3000, key=7, n_reps=24, role="holdout"),
    "euler_n100_k11": dict(n_sim=100, key=11, n_reps=24, role="holdout"),
}


def cache_dir():
    d = Path(
        os.environ.get(
            "EMU_EVOLVE_CACHE",
            "/global/scratch/fsa/fc_jevons/ligon/sue-scratch/emu_gmm_evolve/"
            "fixtures",
        )
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_replicates(config_objects, spec):
    """Estimate on every CRN replicate of a fixture; returns raw stats.

    ``config_objects`` is the dict produced by
    evaluator_gmm.build_objects (weighting/optimizer instances).  Uses
    the build_estimator factory fast path: one trace per config, ~ms
    per replicate thereafter.
    """
    sampler = euler_sampler_factory(spec["n_sim"])
    master = jax.random.PRNGKey(spec["key"])

    def measure_at(r):
        return SyntheticMeasure(
            key=jax.random.fold_in(master, r), n_sim=spec["n_sim"], sampler=sampler
        )

    run = build_estimator(
        euler_residual,
        measure=measure_at(0),
        covariance=SyntheticCovariance(),
        weighting=config_objects.get("weighting"),
        optimizer=config_objects.get("optimizer") or optimistix_lm(),
        theta_init=THETA_INIT,
    )

    errs, iters, conv = [], [], []
    for r in range(spec["n_reps"]):
        res = run(THETA_INIT, measure_at(r))
        b = float(res.theta_hat.beta)
        g = float(res.theta_hat.gamma)
        errs.append(
            ((b - BETA_TRUE) / BETA_TRUE) ** 2 + ((g - GAMMA_TRUE) / GAMMA_TRUE) ** 2
        )
        iters.append(int(res.iterations))
        conv.append(bool(res.converged))
    n = float(len(errs))
    return dict(
        rmse=float((sum(errs) / n) ** 0.5),
        mean_iters=float(sum(iters) / n),
        converged_frac=float(sum(conv) / n),
        n_reps=spec["n_reps"],
    )


def build_baseline(name, force=False):
    """Registered baseline = all-defaults config (CUE + optimistix_lm)."""
    path = cache_dir() / (name + ".json")
    if path.exists() and not force:
        return json.loads(path.read_text())
    spec = REGISTRY[name]
    stats = run_replicates({}, spec)
    meta = dict(
        name=name,
        spec=spec,
        baseline=stats,
        theta_true=dict(beta=BETA_TRUE, gamma=GAMMA_TRUE),
    )
    path.write_text(json.dumps(meta, indent=2))
    return meta


def main(argv):
    if len(argv) >= 2 and argv[1] == "build":
        names = argv[2:] or [n for n, s in REGISTRY.items() if s["role"] == "fitness"]
        for name in names:
            print("building baseline", name, "...", flush=True)
            print(json.dumps(build_baseline(name), indent=2), flush=True)
        return 0
    print("usage: fixture_euler.py build [name ...]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
