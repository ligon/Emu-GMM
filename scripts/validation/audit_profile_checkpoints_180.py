"""Reclassify saved native profiles with a separately pinned numerical FOC guard.

No optimization, reference distribution or p-value is recomputed. The supplied
plan pins each diagnostic manifest and the original assembled artifact. Reuse
ledger Sections 29/30/33, the actual FOC helper, set assembler and law codec.
Strict FOC is the default; historical optimizer-or-FOC replay is explicit.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import native_projected_set_180 as native
import numpy as np
from emu_gmm.inference.confidence_set import _FOC_TOL_DEFAULT, _assemble_set
from emu_gmm.inference.k_statistic import KStatisticResult

_IDENTITY_KEYS = (
    "source_files_sha256",
    "driver_sha256",
    "installed_distributions",
    "python",
    "jax_enable_x64",
    "backend",
    "numerical_environment",
)


def load_guard(path, expected):
    if native.sha256(path) != expected:
        raise ValueError("guard module hash mismatch")
    spec = importlib.util.spec_from_file_location("_audited_foc_guard_180", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned guard module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module._FOC_TOL_DEFAULT != _FOC_TOL_DEFAULT:
        raise ValueError("FOC threshold changed")
    return module._foc_decrement


def audit(plan, guard, output, *, policy="strict_foc"):
    if policy not in ("strict_foc", "optimizer_or_foc"):
        raise ValueError("unknown nuisance admission policy")
    original_path = Path(plan["original_manifest"])
    if native.sha256(original_path) != plan["original_manifest_sha256"]:
        raise ValueError("original artifact manifest changed")
    original = json.loads(original_path.read_text())
    expected_source = original["plan"]["source_identity"]
    runtime = native.source_environment()
    if any(runtime[k] != expected_source[k] for k in _IDENTITY_KEYS):
        raise ValueError("scientific audit runtime differs from producer")
    expected_grid = original["confidence_set"]["grid"]
    if len(plan["diagnoses"]) != len(expected_grid):
        raise ValueError("incomplete diagnostic grid")
    by_s = {}
    for entry in plan["diagnoses"]:
        path = Path(entry["path"])
        if native.sha256(path) != entry["sha256"]:
            raise ValueError("diagnostic manifest changed")
        diagnosis = json.loads(path.read_text())
        if (
            any(
                diagnosis["source_environment"][k] != expected_source[k]
                for k in _IDENTITY_KEYS
            )
            or diagnosis["inputs"]["sha256"]
            != original["plan"]["input_identity"]["sha256"]
        ):
            raise ValueError("diagnostic source/environment/input mismatch")
        rows = diagnosis["evaluations"]
        if (
            diagnosis.get("execution_failure")
            or diagnosis.get("original_objective_and_public_K_parity") is not True
            or len(rows) != 1
            or rows[0]["c0_offset"] != 0
            or rows[0]["evaluation_converged"] is not False
            or rows[0]["evaluation_iterations"] != 0
        ):
            raise ValueError("diagnosis is not a successful endpoint-only evaluation")
        s = diagnosis["producer_record"]["grid_value"]
        if s in by_s:
            raise ValueError("duplicate diagnostic grid value")
        by_s[s] = (path, diagnosis)
    if set(by_s) != set(expected_grid):
        raise ValueError("diagnostic grid mismatch")
    records, results, nonconverged = [], [], []
    for i, s in enumerate(expected_grid):
        path, diagnosis = by_s[s]
        row = diagnosis["evaluations"][0]
        recorded = diagnosis["producer_record"]
        expected = original["per_point"][i]["source_record"]
        shard = original["per_point"][i]["source_shard"]
        if (
            diagnosis["producer_manifest_sha256"]
            != original["source_shards"][shard]["manifest_sha256"]
        ):
            raise ValueError("diagnostic producer manifest mismatch")
        if (
            recorded != expected
            or diagnosis["original_statistics"] != original["statistics"][i]
        ):
            raise ValueError("diagnosis does not match original point provenance")
        arrays = path.parent / row["arrays_file"]
        if (
            arrays.parent.resolve() != path.parent.resolve()
            or native.sha256(arrays) != row["arrays_sha256"]
        ):
            raise ValueError("diagnostic array path/hash mismatch")
        with np.load(arrays, allow_pickle=False) as z:
            current = guard(
                SimpleNamespace(gradient=z["gradient"], gn_hessian=z["gn_hessian"]),
                row["gauge_dim"],
            )
        info = recorded["optimizer_info"]
        certified = (
            info["done"]
            if info["done"] is not None
            else info["status"] in ("converged", "traced")
        )
        foc_ok = (
            current is not None and np.isfinite(current) and current <= _FOC_TOL_DEFAULT
        )
        # Reuse the production decrement; Section 33 requires it independently
        # of raw solver certification. Preserve explicit historical replay.
        failed = not (foc_ok or (policy == "optimizer_or_foc" and certified))
        nonconverged.append(failed)
        results.append(KStatisticResult(**diagnosis["original_statistics"]))
        records.append(
            {
                "grid_value": s,
                "original_profiler_nuisance_valid": recorded["profiler_nuisance_valid"],
                "original_optimizer_certified": certified,
                "original_optimizer_steps": info["steps"],
                "original_foc_checkpoint_evaluation": row["foc_squared_decrement"],
                "corrected_foc_decrement": current,
                "corrected_foc_available": current is not None
                and bool(np.isfinite(current)),
                "corrected_foc_satisfied": bool(foc_ok),
                "corrected_profiler_nuisance_valid": not failed,
                "diagnostic_manifest_sha256": native.sha256(path),
                "arrays_sha256": row["arrays_sha256"],
            }
        )
    cs = _assemble_set(
        np.asarray(expected_grid),
        results,
        original["confidence_set"]["statistic"],
        original["confidence_set"]["alpha"],
        True,
        nonconverged=np.asarray(nonconverged, dtype=bool),
    )
    persistence = native.persistence_check(cs, output / "corrected-guard-law.npz")
    return {
        "kind": "corrected-numerical-guard-audit-of-unchanged-saved-profile-points",
        "interpretation": "not a refit, historical reconstruction, native coverage or continuous-domain claim; strong-nuisance/interior assumptions remain unverified",
        "plan": plan,
        "foc_tol": _FOC_TOL_DEFAULT,
        "nuisance_admission_policy": policy,
        "points": records,
        "confidence_set": {
            f.name: getattr(cs, f.name)
            for f in dataclasses.fields(cs)
            if f.name != "results"
        },
        "persistence": persistence,
        "law_sha256": native.sha256(output / "corrected-guard-law.npz"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("strict_foc", "optimizer_or_foc"),
        default="strict_foc",
        help="Require passing FOC (default), or explicitly replay historical OR policy",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    guard = load_guard(args.guard, args.guard_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = audit(plan, guard, args.output_dir, policy=args.policy)
    report.update(
        plan_sha256=native.sha256(args.plan),
        guard_sha256=args.guard_sha256,
        audit_driver_sha256=native.sha256(__file__),
        source_environment=native.source_environment(),
    )
    native.write_json(args.output_dir / "audit.json", report)
    print(
        json.dumps(
            native.json_value(
                {
                    "event": "audit-complete",
                    "points": report["points"],
                    "persistence": report["persistence"],
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
