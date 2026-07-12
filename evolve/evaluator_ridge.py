"""Truth-anchored evaluator for evolved RIDGE/WEIGHTING configurations.

Run 2 of the emu-gmm evolve harness (ratified 2026-07-11): candidates
choose among whitelisted estimator knobs -- run 1's surface (weighting
strategy + iteration policy + optimizer tolerances) PLUS the
``DiagonalTikhonov`` condition-number target ``kappa_target`` -- and
are scored on INFERENCE CALIBRATION in the binding regime, where the
run-1 theta-RMSE gate showed no transferable headroom but baseline
coverage/size calibration is visibly broken (baseline calibration
error ~0.4-0.5 on the fitness fixtures).

Score, per fixture:

    score = 1 - err(candidate) / err(baseline),

with ``err`` the adversarially-accounted calibration error defined in
fixture_binding.py (every registered replicate in the denominator:
non-converged and NaN-SE replicates count AGAINST coverage, NaN-J and
non-convergence count AS rejections -- no candidate can win by
breaking hard replicates).  Combined score = mean over the fitness
fixtures; min_fixture_score reported.  GATES, per fixture, all
RELATIVE to the registered baseline (the binding regime breaks some
replicates for everyone -- absolute gates are unusable here):

    n_used     >= baseline n_used        (no rescuing the score by
                                          dropping hard replicates)
    mean_iters <= 3x baseline mean_iters (over used replicates;
                                          accuracy may not be bought
                                          with unbounded compute)

Any gate failure or contract violation = -1.  Anti-gaming notes as in
run 1: candidates return DATA (a validated config dict); the whitelist
and bounds are the contract; the gate verifies INSTANCES of this DGP
family -- transfer is checked post-run on hold-out fixtures (an unseen
binding spec with the >= half-improvement bar, plus a benign
no-binding arm with a do-no-harm bar), and adoption claims cap at
golden-fixture strength.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fixture_binding as fx  # noqa: E402
from emu_gmm import (  # noqa: E402
    DiagonalTikhonov,
    Identity,
    IteratedWeighting,
    optimistix_lm,
)

FAIL_SCORE = -1.0
ITER_BUDGET_FACTOR = 3.0

_BOUNDS = dict(
    weighting=("cue", "identity", "iterated"),
    weighting_iterations=(1, 30),
    weighting_tol=(1e-10, 1e-2),
    rtol=(1e-12, 1e-4),
    atol=(1e-12, 1e-4),
    max_steps=(10, 400),
    kappa_target=(1e2, 1e12),
)

_DEFAULTS = dict(
    weighting="cue",
    weighting_iterations=10,
    weighting_tol=1e-6,
    rtol=1e-8,
    atol=1e-8,
    max_steps=200,
    kappa_target=1e6,  # DiagonalTikhonov's default
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
    for k in ("weighting_tol", "rtol", "atol", "kappa_target"):
        if k in cfg:
            lo, hi = _BOUNDS[k]
            v = float(cfg[k])
            if not lo <= v <= hi:
                raise ContractViolation(f"{k} must be in [{lo}, {hi}]")
    return cfg


def build_objects(cfg):
    """Whitelisted config dict -> emu_gmm strategy objects.

    Absent keys mean the package default (weighting None = CUE;
    regularization None = the estimator's default DiagonalTikhonov()).
    """
    w = cfg.get("weighting", "cue")
    if w == "cue":
        weighting = None
    elif w == "identity":
        weighting = Identity()
    else:
        weighting = IteratedWeighting(
            weighting_iterations=int(cfg.get("weighting_iterations", 10)),
            weighting_tol=float(cfg.get("weighting_tol", 1e-6)),
        )
    regularization = (
        DiagonalTikhonov(kappa_target=float(cfg["kappa_target"]))
        if "kappa_target" in cfg
        else None
    )
    optimizer = optimistix_lm(
        rtol=float(cfg.get("rtol", 1e-8)),
        atol=float(cfg.get("atol", 1e-8)),
        max_steps=int(cfg.get("max_steps", 200)),
    )
    return dict(weighting=weighting, regularization=regularization, optimizer=optimizer)


PROBE_FIXTURE = "fb_few_eps_k31"
PROBE_REPS = 12


def make_ctx(oracle_budget=6):
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
        fixture = dict(fx.REGISTRY[PROBE_FIXTURE], n_reps=PROBE_REPS)
        return fx.run_replicates(objects, fixture)

    ctx = dict(
        fixtures={
            n: dict(n_obs=m["n_obs"], baseline=m["baseline"]) for n, m in metas.items()
        },
        theta_true=dict(beta0=float(fx.TRUTH.beta0), beta1=float(fx.TRUTH.beta1)),
        bounds={k: v for k, v in _BOUNDS.items()},
        defaults=dict(_DEFAULTS),
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
    base = meta["baseline"]
    out = dict(fixture=name, score=FAIL_SCORE, baseline=base)
    try:
        stats = fx.run_replicates(objects, meta["fixture"])
    except Exception:
        out.update(
            status="build_failed", feedback=traceback.format_exc(limit=3)[-1500:]
        )
        return out
    out.update(stats)
    if stats["n_used"] < base["n_used"]:
        out.update(
            status="convergence_regressed",
            feedback="n_used {} < baseline {}".format(stats["n_used"], base["n_used"]),
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
        score=float(1.0 - stats["err"] / base["err"]),
        feedback="err {:.6g} vs baseline {:.6g}".format(stats["err"], base["err"]),
    )
    return out


def evaluate_candidate(path, oracle_budget=6):
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
    ap.add_argument("--oracle-budget", type=int, default=6)
    args = ap.parse_args(argv)
    json.dump(
        evaluate_candidate(args.candidate, args.oracle_budget), sys.stdout, indent=2
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
