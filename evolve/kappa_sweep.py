"""EXPLORATORY direct kappa_target sweep under CUE (NOT registered).

Run 2's GA never varied kappa_target under CUE weighting (it fixated
on the gated iterated family), so the run's central hypothesis --
does the DiagonalTikhonov condition-number target move calibration in
the binding regime? -- went untested by evolution.  This script tests
it directly: one (kappa, fixture) evaluation per invocation, using the
REGISTERED metric and fixtures, all other knobs at defaults.

    .venv/bin/python evolve/kappa_sweep.py --kappa 1e4 \
        --fixture fb_few_eps_k31 [--out DIR]

Results are exploratory reference points for the run-2 report; any
adoption-grade claim would need its own pre-registration.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--fixture", required=True, choices=list(fx.REGISTRY))
    ap.add_argument(
        "--out",
        default="/global/scratch/fsa/fc_jevons/ligon/sue-scratch/"
        "emu_gmm_evolve/kappa_sweep",
    )
    args = ap.parse_args(argv)

    cfg = ev.validate_config({"kappa_target": float(args.kappa)})
    objects = ev.build_objects(cfg)
    res = ev.score_on_fixture(objects, args.fixture)
    res["kappa_target"] = float(args.kappa)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"kappa_{args.kappa:.0e}_{args.fixture}.json"
    (out / name).write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {
                k: res.get(k)
                for k in (
                    "fixture",
                    "kappa_target",
                    "status",
                    "score",
                    "err",
                    "n_used",
                    "binding_frequency",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
