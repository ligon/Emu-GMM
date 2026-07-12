"""Truth-anchored evaluator for evolved GMM estimator configurations.

The candidate is a Python file defining ``propose_config(ctx) -> dict``
choosing among WHITELISTED estimator knobs (weighting strategy and its
iteration policy; optimizer tolerances/steps).  Score, per fixture:

    score = 1 - rmse(candidate) / rmse(baseline),

where rmse is the normalized theta-recovery error against the EXACT
known truth over the fixture's CRN replicates, GATED on (a) every
replicate converged and (b) mean iterations <= 3x baseline (accuracy
may not be bought with unbounded compute).  Combined score = mean over
the fitness fixtures; any gate failure or contract violation = -1.

Anti-gaming notes.  Candidates return DATA (a validated config dict),
never code the evaluator executes; the whitelist and bounds are the
contract.  The gate is truth-anchored, not certified: per the adopted
[t6:proc-verify-by-certificate] refinement, it verifies INSTANCES of
this DGP -- transfer to unseen keys/sample sizes is checked on
hold-out fixtures the candidate never sees, and any adoption claim is
capped at golden-fixture strength (no certificate exists here).
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fixture_euler as fx  # noqa: E402
from emu_gmm import Identity, IteratedWeighting, optimistix_lm  # noqa: E402

FAIL_SCORE = -1.0
ITER_BUDGET_FACTOR = 3.0

_BOUNDS = dict(
    weighting=("cue", "identity", "iterated"),
    weighting_iterations=(1, 30),
    weighting_tol=(1e-10, 1e-2),
    rtol=(1e-12, 1e-4),
    atol=(1e-12, 1e-4),
    max_steps=(10, 400),
)


class ContractViolation(ValueError):
    pass


class ProbeBudgetExceeded(RuntimeError):
    pass


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ContractViolation("config must be a dict")
    unknown = set(cfg) - set(_BOUNDS)
    if unknown:
        raise ContractViolation(f"unknown keys {sorted(unknown)}")
    w = cfg.get("weighting", "cue")
    if w not in _BOUNDS["weighting"]:
        raise ContractViolation(
            "weighting must be one of {}".format(_BOUNDS["weighting"])
        )
    for k in ("weighting_iterations", "max_steps"):
        if k in cfg:
            lo, hi = _BOUNDS[k]
            v = cfg[k]
            if not isinstance(v, int) or not lo <= v <= hi:
                raise ContractViolation(f"{k} must be int in [{lo}, {hi}]")
    for k in ("weighting_tol", "rtol", "atol"):
        if k in cfg:
            lo, hi = _BOUNDS[k]
            v = float(cfg[k])
            if not lo <= v <= hi:
                raise ContractViolation(f"{k} must be in [{lo}, {hi}]")
    return cfg


def build_objects(cfg):
    """Whitelisted config dict -> emu_gmm strategy objects."""
    w = cfg.get("weighting", "cue")
    if w == "cue":
        weighting = None  # estimate() default = CUE
    elif w == "identity":
        weighting = Identity()
    else:
        weighting = IteratedWeighting(
            weighting_iterations=int(cfg.get("weighting_iterations", 10)),
            weighting_tol=float(cfg.get("weighting_tol", 1e-6)),
        )
    optimizer = optimistix_lm(
        rtol=float(cfg.get("rtol", 1e-8)),
        atol=float(cfg.get("atol", 1e-8)),
        max_steps=int(cfg.get("max_steps", 200)),
    )
    return dict(weighting=weighting, optimizer=optimizer)


PROBE_FIXTURE = "euler_n800_k0"
PROBE_REPS = 6


def make_ctx(oracle_budget=8):
    metas = {
        n: fx.build_baseline(n)
        for n, s in fx.REGISTRY.items()
        if s["role"] == "fitness"
    }
    state = dict(calls=0)

    def probe(cfg):
        if state["calls"] >= oracle_budget:
            raise ProbeBudgetExceeded(f"probe budget of {oracle_budget} exhausted")
        state["calls"] += 1
        objects = build_objects(validate_config(cfg))
        spec = dict(fx.REGISTRY[PROBE_FIXTURE], n_reps=PROBE_REPS)
        return fx.run_replicates(objects, spec)

    ctx = dict(
        fixtures={
            n: dict(n_sim=m["spec"]["n_sim"], baseline=m["baseline"])
            for n, m in metas.items()
        },
        theta_true=dict(beta=fx.BETA_TRUE, gamma=fx.GAMMA_TRUE),
        bounds={k: v for k, v in _BOUNDS.items()},
        defaults=dict(
            weighting="cue",
            weighting_iterations=10,
            weighting_tol=1e-6,
            rtol=1e-8,
            atol=1e-8,
            max_steps=200,
        ),
        probe=probe,
        probe_calls_remaining=lambda: oracle_budget - state["calls"],
    )
    return ctx, state


def load_candidate(path):
    src = Path(path).read_text()
    ns = {"__name__": "candidate", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)
    if "propose_config" not in ns or not callable(ns["propose_config"]):
        raise ValueError("candidate must define propose_config(ctx)")
    return ns["propose_config"]


def score_on_fixture(objects, name):
    meta = fx.build_baseline(name)
    out = dict(fixture=name, score=FAIL_SCORE, baseline=meta["baseline"])
    try:
        stats = fx.run_replicates(objects, meta["spec"])
    except Exception:
        out.update(
            status="build_failed", feedback=traceback.format_exc(limit=3)[-1500:]
        )
        return out
    out.update(stats)
    base = meta["baseline"]
    if stats["converged_frac"] < 1.0:
        out.update(
            status="convergence_failed",
            feedback="{} of reps converged".format(stats["converged_frac"]),
        )
        return out
    if stats["mean_iters"] > ITER_BUDGET_FACTOR * base["mean_iters"]:
        out.update(
            status="compute_budget_exceeded",
            feedback="mean iters {} > {}x baseline {}".format(
                stats["mean_iters"], ITER_BUDGET_FACTOR, base["mean_iters"]
            ),
        )
        return out
    out.update(
        status="ok",
        score=float(1.0 - stats["rmse"] / base["rmse"]),
        feedback="rmse {:.6g} vs baseline {:.6g}".format(stats["rmse"], base["rmse"]),
    )
    return out


def evaluate_candidate(path, oracle_budget=8):
    propose_fn = load_candidate(path)
    ctx, state = make_ctx(oracle_budget)
    result = dict(
        score=FAIL_SCORE, min_fixture_score=FAIL_SCORE, probe_calls=0, fixtures=[]
    )
    try:
        cfg = validate_config(propose_fn(ctx))
        objects = build_objects(cfg)
    except Exception:
        result.update(
            status="candidate_error",
            feedback=traceback.format_exc(limit=4)[-2000:],
            probe_calls=state["calls"],
        )
        return result
    result["probe_calls"] = state["calls"]
    result["config"] = cfg
    per = [
        score_on_fixture(objects, n)
        for n, s in fx.REGISTRY.items()
        if s["role"] == "fitness"
    ]
    result["fixtures"] = per
    scores = [p["score"] for p in per]
    result.update(
        status="ok" if all(p["status"] == "ok" for p in per) else "fixture_failure",
        score=float(sum(scores) / len(scores)),
        min_fixture_score=float(min(scores)),
    )
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--oracle-budget", type=int, default=8)
    args = ap.parse_args(argv)
    json.dump(
        evaluate_candidate(args.candidate, args.oracle_budget), sys.stdout, indent=2
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
