"""Battery for the emu-gmm evolve harness.

    .venv/bin/python evolve/test_harness.py     (from the repo root)

G1  Seed (all defaults) scores exactly 0.0 on every fitness fixture
    (also builds/caches the baselines on first run).
G2  Contract violations fail cleanly: unknown key, out-of-bounds
    value, non-dict return.
G3  A non-default whitelisted config (identity weighting) runs end to
    end and produces a scored (> -1 or gated) result -- the full
    build_estimator path works off-default.
G4  Determinism: two full evaluations of the seed are identical.
G5  Probe budget enforced.
G6  Truth-anchor sanity: baseline rmse finite and positive; baseline
    converged_frac == 1.0 (the gate is meaningful).
"""

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import evaluator_gmm as ev
import fixture_euler as fx


def test_g1_seed_zero():
    r = ev.evaluate_candidate(_THIS_DIR / "initial_config.py")
    assert r["status"] == "ok", r
    assert abs(r["score"]) < 1e-12, r
    for p in r["fixtures"]:
        assert abs(p["score"]) < 1e-12, p
    print("G1 ok: seed exact 0.0 on", len(r["fixtures"]), "fixtures")
    return r


def test_g2_contract():
    for bad in (
        {"nonsense": 1},
        {"rtol": 1.0},
        {"weighting": "magic"},
        {"max_steps": 5},
        ["not", "a", "dict"],
    ):
        try:
            ev.validate_config(bad)
        except ev.ContractViolation:
            continue
        raise AssertionError("accepted bad config {}".format(bad))
    print("G2 ok: contract violations rejected")


def test_g3_nondefault_runs():
    objects = ev.build_objects(ev.validate_config({"weighting": "identity"}))
    out = ev.score_on_fixture(objects, "euler_n800_k0")
    assert out["status"] in ("ok", "convergence_failed", "compute_budget_exceeded"), out
    print(
        "G3 ok: identity-weighting config ran; status",
        out["status"],
        "score",
        round(out["score"], 4),
    )


def test_g4_determinism(first):
    r2 = ev.evaluate_candidate(_THIS_DIR / "initial_config.py")
    assert json.dumps(first, sort_keys=True) == json.dumps(
        r2, sort_keys=True
    ), "evaluations differ"
    print("G4 ok: bit-identical evaluations")


def test_g5_budget():
    ctx, state = ev.make_ctx(oracle_budget=2)
    ctx["probe"]({})
    ctx["probe"]({})
    try:
        ctx["probe"]({})
    except ev.ProbeBudgetExceeded:
        print("G5 ok: probe budget enforced at", state["calls"])
        return
    raise AssertionError("budget not enforced")


def test_g6_anchor_sanity():
    for n, s in fx.REGISTRY.items():
        if s["role"] != "fitness":
            continue
        b = fx.build_baseline(n)["baseline"]
        assert 0.0 < b["rmse"] < 1.0, (n, b)
        assert b["converged_frac"] == 1.0, (n, b)
    print("G6 ok: baselines finite, positive, fully converged")


if __name__ == "__main__":
    first = test_g1_seed_zero()
    test_g2_contract()
    test_g3_nondefault_runs()
    test_g4_determinism(first)
    test_g5_budget()
    test_g6_anchor_sanity()
    print("all harness tests passed")
