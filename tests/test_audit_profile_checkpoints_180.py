"""Controls for reclassifying saved profiles without changing their statistics."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_VALIDATION = Path(__file__).parents[1] / "scripts/validation"
for _name in ("native_projected_set_180", "audit_profile_checkpoints_180"):
    _spec = importlib.util.spec_from_file_location(_name, _VALIDATION / f"{_name}.py")
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
auditor = sys.modules["audit_profile_checkpoints_180"]
native = sys.modules["native_projected_set_180"]


def _plan(tmp_path):
    environment = native.source_environment()
    original = {
        "confidence_set": {"grid": [0.0, 1.0], "statistic": "K", "alpha": 0.05},
        "per_point": [],
        "statistics": [],
        "plan": {
            "source_identity": environment,
            "input_identity": {"sha256": {"data": "frozen"}},
        },
        "source_shards": [{"manifest_sha256": str(i) * 64} for i in range(2)],
    }
    diagnoses = []
    for i in range(2):
        directory = tmp_path / f"point-{i}"
        directory.mkdir()
        arrays = directory / "evaluation-00.npz"
        np.savez_compressed(arrays, gradient=np.ones(1), gn_hessian=np.eye(1))
        record = {
            "grid_value": float(i),
            "profiler_nuisance_valid": True,
            "optimizer_info": {
                "done": bool(i),
                "steps": 10,
                "status": "converged" if i else "max_iterations",
            },
        }
        statistics = {
            "K": 5.0,
            "S": 1.0,
            "J": 6.0,
            "p_K": 0.01,
            "p_S": 0.3,
            "p_J": 0.04,
            "df_K": 1,
            "df_S": 1,
            "df_J": 2,
            "invalid_input": False,
        }
        diagnosis = {
            "source_environment": environment,
            "inputs": {"sha256": {"data": "frozen"}},
            "producer_manifest_sha256": str(i) * 64,
            "original_objective_and_public_K_parity": True,
            "producer_record": record,
            "original_statistics": statistics,
            "evaluations": [
                {
                    "c0_offset": 0.0,
                    "evaluation_converged": False,
                    "evaluation_iterations": 0,
                    "arrays_file": arrays.name,
                    "arrays_sha256": native.sha256(arrays),
                    "gauge_dim": 0,
                    "foc_squared_decrement": 0.0,
                }
            ],
        }
        path = directory / "diagnosis.json"
        path.write_text(json.dumps(diagnosis))
        diagnoses.append({"path": str(path), "sha256": native.sha256(path)})
        original["per_point"].append({"source_record": record, "source_shard": i})
        original["statistics"].append(statistics)
    path = tmp_path / "original.json"
    path.write_text(json.dumps(original))
    return {
        "original_manifest": str(path),
        "original_manifest_sha256": native.sha256(path),
        "diagnoses": diagnoses,
    }


def test_reclassification_preserves_statistics_and_certified_superset(tmp_path):
    plan = _plan(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    report = auditor.audit(
        plan, lambda inner, gd: 4.0, output, policy="optimizer_or_foc"
    )
    assert report["nuisance_admission_policy"] == "optimizer_or_foc"
    cs = report["confidence_set"]
    np.testing.assert_array_equal(cs["p_grid"], [0.01, 0.01])
    assert cs["nonconverged_indices"] == (0,)
    assert report["points"][0]["corrected_foc_decrement"] == 4.0
    assert report["points"][1]["original_optimizer_certified"]
    assert report["points"][1]["corrected_profiler_nuisance_valid"]
    assert report["persistence"]["point_verdicts"] == [None, False]
    assert not report["persistence"]["known_empty_scanned_set"]
    assert report["persistence"]["capacity_queries_checked"] == 3


@pytest.mark.parametrize(
    "corruption",
    [
        "manifest",
        "arrays",
        "mapping",
        "duplicate",
        "incomplete",
        "source",
        "input",
        "producer",
    ],
)
def test_audit_rejects_changed_or_incomplete_evidence(tmp_path, corruption):
    plan = _plan(tmp_path)
    entry = plan["diagnoses"][0]
    path = Path(entry["path"])
    if corruption == "manifest":
        path.write_text(path.read_text() + " ")
    elif corruption == "arrays":
        arrays = path.parent / "evaluation-00.npz"
        arrays.write_bytes(arrays.read_bytes() + b"changed")
    elif corruption == "mapping":
        m = json.loads(path.read_text())
        m["producer_record"]["optimizer_info"]["steps"] = 99
        path.write_text(json.dumps(m))
        entry["sha256"] = native.sha256(path)
    elif corruption == "duplicate":
        plan["diagnoses"][1] = entry
    elif corruption in ("source", "input", "producer"):
        m = json.loads(path.read_text())
        if corruption == "source":
            m["source_environment"]["driver_sha256"] = "changed"
        elif corruption == "input":
            m["inputs"]["sha256"]["data"] = "changed"
        else:
            m["producer_manifest_sha256"] = "changed"
        path.write_text(json.dumps(m))
        entry["sha256"] = native.sha256(path)
    else:
        plan["diagnoses"].pop()
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(ValueError):
        auditor.audit(plan, lambda inner, gd: 4.0, output)


def test_audit_refuses_different_scientific_runtime(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    changed = native.source_environment()
    changed["driver_sha256"] = "changed"
    monkeypatch.setattr(native, "source_environment", lambda: changed)
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(ValueError, match="runtime"):
        auditor.audit(plan, lambda inner, gd: 4.0, output)


@pytest.mark.parametrize("decrement", [4.0, None, np.nan, np.inf])
def test_strict_policy_requires_finite_passing_foc_even_if_certified(
    tmp_path, decrement
):
    plan = _plan(tmp_path)
    output = tmp_path / "strict-audit"
    output.mkdir()
    report = auditor.audit(plan, lambda inner, gd: decrement, output)
    assert report["nuisance_admission_policy"] == "strict_foc"
    cs = report["confidence_set"]
    np.testing.assert_array_equal(cs["p_grid"], [0.01, 0.01])
    assert cs["nonconverged_indices"] == (0, 1)
    assert cs["n_nonconverged"] == 2
    assert cs["n_invalid"] == 0
    assert report["points"][1]["original_optimizer_certified"]
    assert not report["points"][1]["corrected_profiler_nuisance_valid"]
    assert report["persistence"]["point_verdicts"] == [None, None]
    assert not report["persistence"]["known_empty_scanned_set"]
    assert report["persistence"]["capacity_queries_checked"] == 3


def test_strict_policy_admits_passing_foc_regardless_of_certification(tmp_path):
    plan = _plan(tmp_path)
    output = tmp_path / "passing-audit"
    output.mkdir()
    report = auditor.audit(plan, lambda inner, gd: auditor._FOC_TOL_DEFAULT, output)
    assert report["confidence_set"]["n_nonconverged"] == 0
    assert all(p["corrected_profiler_nuisance_valid"] for p in report["points"])


def test_unknown_policy_is_rejected_before_creating_a_law(tmp_path):
    with pytest.raises(ValueError, match="admission policy"):
        auditor.audit({}, lambda inner, gd: 0.0, tmp_path, policy="typo")
    assert not list(tmp_path.iterdir())
