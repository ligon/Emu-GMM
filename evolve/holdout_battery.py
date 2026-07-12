"""Post-run HOLD-OUT transfer battery (pre-registered; run AFTER evolution).

Evaluates a config on the hold-out fixtures -- unseen PRNG keys AND
unseen sample sizes, never scored during evolution -- and applies the
ratified transfer bar: the winner must retain >= half its fitness
improvement on EACH hold-out, with all gates (100 percent convergence,
mean iterations <= 3x that fixture's baseline) passing.

    .venv/bin/python evolve/holdout_battery.py \
        --config '{"weighting": "identity", ...}' \
        --fitness-improvement 0.004564

Builds (and caches) the hold-out baselines on first use.  Scores here
are golden-fixture evidence only; adoption additionally requires a
review of the winning config's statistical rationale (README.org).
"""

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import evaluator_gmm as ev  # noqa: E402
import fixture_euler as fx  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="config dict as JSON")
    ap.add_argument(
        "--fitness-improvement",
        type=float,
        required=True,
        help="the winner's combined fitness score (its improvement over "
        "the all-defaults baseline); the bar is >= half this on "
        "each hold-out",
    )
    args = ap.parse_args(argv)

    cfg = ev.validate_config(json.loads(args.config))
    objects = ev.build_objects(cfg)
    bar = 0.5 * args.fitness_improvement

    out = dict(
        config=cfg,
        fitness_improvement=args.fitness_improvement,
        transfer_bar=bar,
        holdouts=[],
        transfer="PASS",
    )
    for name, spec in fx.REGISTRY.items():
        if spec["role"] != "holdout":
            continue
        fx.build_baseline(name)  # ensure cached
        res = ev.score_on_fixture(objects, name)
        res["passes_bar"] = bool(res["status"] == "ok" and res["score"] >= bar)
        out["holdouts"].append(res)
        if not res["passes_bar"]:
            out["transfer"] = "FAIL"
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
