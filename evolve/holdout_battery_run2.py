"""Run-2 post-run HOLD-OUT battery (pre-registered; run AFTER evolution).

Two hold-outs, two bars (ratified 2026-07-11):

- fb_mid_eps_k41 (unseen BINDING spec, unseen key): transfer bar --
  the winner retains >= half its fitness improvement, i.e. its
  hold-out score = 1 - err/err_baseline(holdout) must be >= half the
  combined fitness score, with the same relative gates.
- fb_benign_k42 (benign no-binding arm): DO-NO-HARM bar -- the
  winner's calibration error may not exceed the baseline's by more
  than 0.05 absolute (this is what catches a candidate that "fixes"
  the binding regime by indiscriminate over-ridging).

    .venv/bin/python evolve/holdout_battery_run2.py \
        --config '{"kappa_target": 1e4, ...}' \
        --fitness-improvement 0.31

Scores here are golden-fixture evidence only; adoption additionally
requires a review of the winning config's statistical rationale.
"""

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import evaluator_ridge as ev  # noqa: E402
import fixture_binding as fx  # noqa: E402

DO_NO_HARM_MARGIN = 0.05

BINDING_HOLDOUT = "fb_mid_eps_k41"
BENIGN_HOLDOUT = "fb_benign_k42"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="config dict as JSON")
    ap.add_argument(
        "--fitness-improvement",
        type=float,
        required=True,
        help="the winner's combined fitness score; the binding hold-out "
        "bar is >= half this",
    )
    args = ap.parse_args(argv)

    cfg = ev.validate_config(json.loads(args.config))
    objects = ev.build_objects(cfg)
    bar = 0.5 * args.fitness_improvement

    out = dict(
        config=cfg,
        fitness_improvement=args.fitness_improvement,
        transfer_bar=bar,
        do_no_harm_margin=DO_NO_HARM_MARGIN,
        holdouts=[],
        transfer="PASS",
    )

    # Binding hold-out: same scored-and-gated path as a fitness fixture.
    res = ev.score_on_fixture(objects, BINDING_HOLDOUT)
    res["bar"] = bar
    res["passes_bar"] = bool(res["status"] == "ok" and res["score"] >= bar)
    out["holdouts"].append(res)
    if not res["passes_bar"]:
        out["transfer"] = "FAIL"

    # Benign hold-out: do-no-harm on err (gates still apply -- a
    # candidate that breaks convergence on the healthy arm fails).
    meta = fx.build_baseline(BENIGN_HOLDOUT)
    res_b = ev.score_on_fixture(objects, BENIGN_HOLDOUT)
    base_err = meta["baseline"]["err"]
    harm_ok = bool(
        res_b["status"] == "ok" and res_b["err"] <= base_err + DO_NO_HARM_MARGIN
    )
    res_b["bar"] = f"err <= {base_err:.4g} + {DO_NO_HARM_MARGIN}"
    res_b["passes_bar"] = harm_ok
    out["holdouts"].append(res_b)
    if not harm_ok:
        out["transfer"] = "FAIL"

    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
