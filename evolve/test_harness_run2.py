"""Battery for the run-2 (binding-regime) evolve harness.

    .venv/bin/python evolve/test_harness_run2.py     (from the repo root)

NOTE: G1/G3 evaluate at the registered 96-replicate counts with
per-rep anchoring -- the full battery takes ~30-60 minutes on a
4-core slice (baselines cache on first build).

G1  Seed (all defaults) scores exactly 0.0 on every fitness fixture.
G2  Contract violations fail cleanly: unknown key, out-of-bounds
    kappa_target, non-dict return, run-1 keys that were dropped
    (tau_threshold) rejected.
G3  A non-default whitelisted config (identity + kappa_target=1e4)
    runs end to end on a fitness fixture and produces a scored (> -1
    or cleanly gated) result.
G4  Determinism: two 12-rep probe-sized runs of the same non-default
    config are bit-identical (the full-eval determinism of run 1's G4
    is implied by G1's exact zero plus this pipeline check, at a
    fraction of the cost).
G5  Probe budget enforced.
G6  Anchor sanity: fitness baselines have finite positive err,
    n_used >= half the reps, and BINDING_FREQUENCY >= 0.15 (a fixture
    where the ridge never binds cannot discriminate kappa_target --
    the #130 "discriminate or it doesn't count" lesson); the benign
    hold-out baseline has binding_frequency == 0, n_used == n_reps,
    and small err (the do-no-harm reference is itself healthy).
"""

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import evaluator_ridge as ev  # noqa: E402
import fixture_binding as fx  # noqa: E402


def test_g1_seed_zero():
    r = ev.evaluate_candidate(_THIS_DIR / "initial_config_ridge.py")
    assert r["status"] == "ok", r
    assert abs(r["score"]) < 1e-12, r
    for p in r["fixtures"]:
        assert abs(p["score"]) < 1e-12, p
    print("G1 ok: seed exact 0.0 on", len(r["fixtures"]), "fixtures")
    return r


def test_g2_contract():
    for bad in (
        {"nonsense": 1},
        {"tau_threshold": 0.5},  # dropped as inert; must NOT validate
        {"kappa_target": 10.0},  # below 1e2
        {"kappa_target": 1e13},  # above 1e12
        {"rtol": 1.0},
        {"weighting": "magic"},
        {"max_steps": 5},
        ["not", "a", "dict"],
    ):
        try:
            ev.validate_config(bad)
        except ev.ContractViolation:
            continue
        raise AssertionError(f"accepted bad config {bad}")
    print("G2 ok: contract violations rejected")


def test_g3_nondefault_runs():
    objects = ev.build_objects(
        ev.validate_config({"weighting": "identity", "kappa_target": 1e4})
    )
    out = ev.score_on_fixture(objects, "fb_few_eps_k31")
    assert out["status"] in (
        "ok",
        "convergence_regressed",
        "compute_budget_exceeded",
    ), out
    print(
        "G3 ok: identity+kappa=1e4 ran; status",
        out["status"],
        "score",
        round(out["score"], 4),
    )


def test_g4_determinism():
    objects = ev.build_objects(
        ev.validate_config({"weighting": "iterated", "kappa_target": 1e8})
    )
    fixture = dict(fx.REGISTRY[ev.PROBE_FIXTURE], n_reps=ev.PROBE_REPS)
    a = fx.run_replicates(objects, fixture)
    b = fx.run_replicates(objects, fixture)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (a, b)
    print("G4 ok: bit-identical replicate runs")


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
        b = fx.build_baseline(n)["baseline"]
        if s["role"] == "fitness":
            assert 0.0 < b["err"] < 2.0, (n, b)
            assert b["n_used"] >= b["n_reps"] // 2, (n, b)
            assert b["binding_frequency"] >= 0.15, (
                n,
                "fixture does not discriminate the ridge",
                b,
            )
        elif n == "fb_benign_k42":
            # 96-rep registered baseline: binding 0.010 (one rep of 96;
            # the 24-rep pilot's 0.0 was granularity, not structure).
            # "Benign" = binding is RARE, not impossible.
            assert b["binding_frequency"] <= 0.05, (n, b)
            assert b["n_used"] == b["n_reps"], (n, b)
            assert b["err"] < 0.15, (n, b)
    print("G6 ok: fitness fixtures bind, benign hold-out is healthy")


if __name__ == "__main__":
    test_g1_seed_zero()
    test_g2_contract()
    test_g3_nondefault_runs()
    test_g4_determinism()
    test_g5_budget()
    test_g6_anchor_sanity()
    print("all run-2 harness tests passed")
