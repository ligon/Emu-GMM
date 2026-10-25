"""One bounded native restricted-CU trust-region probe (ledger Section 34).

This is a new optimizer experiment at one null, not a replay or a confidence
set. The job stages the original scientific source and exact dependency set;
this driver and the corrected FOC helper are separately pinned additions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import audit_profile_checkpoints_180 as auditor
import jax
import native_profile_diagnostics_180 as diagnostic
import native_projected_set_180 as native
import numpy as np
from emu_gmm import build_estimator, identification_strength, k_statistic, riemannian_tr
from emu_gmm._internal.models import map_model_parameters
from emu_gmm._internal.params import flatten_params_with_spec
from emu_gmm.inference.confidence_set import _FOC_TOL_DEFAULT, _ProfileReducer
from emu_gmm.manifolds.riemannian_lm import _horizontal_curvature_eigenvalues

NULL = 0.050088084721517094
GUARD_SHA256 = "1a6261dfc4b41d4be3ffee6aa4e2d88deda6ccd5e459ed19791c72f30b1d85f0"


class CapturingRecorder(native.RecordingOptimizer):
    """Save the genuine raw iterate before any postfit interpretation."""

    def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
        self.residual_fn = residual_fn
        self.residual_args = args
        self.manifold_spec = manifold_spec
        return super().__call__(residual_fn, theta_init, manifold_spec, args=args)


class CapturingEvaluation(diagnostic.CheckpointEvaluation):
    """Evaluate one saved iterate with zero optimization steps."""

    def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
        self.residual_fn = residual_fn
        self.residual_args = args
        self.manifold_spec = manifold_spec
        return super().__call__(residual_fn, theta_init, manifold_spec, args=args)


def protocol(max_steps: int, max_tcg_steps: int) -> dict:
    return {
        "kind": "single-null-native-physical-c0-restricted-rtr-probe",
        "null_s": NULL,
        "nuisance": ["A", "c0", "pi"],
        "model": "original theta-dependent joint product/market FE projection",
        "weighting": "original restricted ContinuouslyUpdated",
        "regularization": "original warm-anchored DiagonalTikhonov",
        "penalty": None,
        "optimizer": {
            "backend": "emu_gmm.riemannian_tr",
            "rtol": 1e-9,
            "atol": 1e-9,
            "max_steps": max_steps,
            "max_tcg_steps": max_tcg_steps,
        },
        "foc_tol": _FOC_TOL_DEFAULT,
        "interpretation": "sample local evidence only; no profile verdict, coverage, boundary optimum, or population identification claim",
    }


def _save_result_arrays(result, path: Path) -> str:
    """Retain estimator-owned ingredients without rebuilding any statistic."""
    arrays = {
        name: np.asarray(getattr(getattr(result, name), "array", getattr(result, name)))
        for name in (
            "objective_value",
            "gradient",
            "gn_hessian",
            "moment_jacobian",
            "weighting_matrix",
            "moment_covariance",
        )
    }
    for where in ("theta_hat", "theta_init"):
        for field in dataclasses.fields(getattr(result, where)):
            arrays[f"{where}_{field.name}"] = np.asarray(
                getattr(getattr(result, where), field.name).array
            )
    np.savez_compressed(path, **arrays)
    return native.sha256(path)


def _curvature(recorder: CapturingRecorder, theta) -> tuple[float, float]:
    flat, _, spec = flatten_params_with_spec(theta)
    if spec.total_gauge_dim != recorder.manifold_spec.total_gauge_dim:
        raise ValueError("optimizer and returned-point gauge dimensions differ")
    args = recorder.residual_args
    residual = (
        (lambda x: recorder.residual_fn(x))
        if args is None
        else (lambda x: recorder.residual_fn(x, args))
    )
    low, high = _horizontal_curvature_eigenvalues(
        residual, recorder.manifold_spec, flat
    )
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--guard-sha256", default=GUARD_SHA256)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--warm", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tcg-steps", type=int, default=10)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--replay-report-sha256")
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--replay-arrays", type=Path)
    parser.add_argument("--original-driver-sha256")
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 200 or not 1 <= args.max_tcg_steps <= 21:
        parser.error("bounded max-steps and max-tcg-steps required")
    plan = protocol(args.max_steps, args.max_tcg_steps)
    replay = args.replay_report is not None
    replay_inputs = (
        args.replay_report_sha256,
        args.replay_checkpoint,
        args.replay_arrays,
        args.original_driver_sha256,
    )
    if replay != all(value is not None for value in replay_inputs):
        parser.error(
            "replay requires its report hash, checkpoint, arrays and old driver hash"
        )
    if replay:
        plan["kind"] = "saved-native-rtr-checkpoint-evaluation-no-solve"
        plan["optimizer"] = {"backend": "diagnostic_checkpoint", "steps": 0}
    if args.print_protocol:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    producer_manifest = args.producer_dir / "manifest.json"
    if native.sha256(producer_manifest) != args.manifest_sha256:
        raise ValueError("original producer manifest hash mismatch")
    producer = json.loads(producer_manifest.read_text())
    environment = native.source_environment()
    diagnostic.validate_producer(producer, environment)
    matches = [r for r in producer["per_point"] if r["grid_value"] == NULL]
    if len(matches) != 1:
        raise ValueError("original producer must contain exactly one selected null")
    original = matches[0]
    checkpoint_path = args.producer_dir / original["parameter_checkpoint"]
    if checkpoint_path.parent.resolve() != args.producer_dir.resolve():
        raise ValueError("original checkpoint path escapes producer directory")
    guard = auditor.load_guard(args.guard, args.guard_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {
        "protocol": plan,
        "started_UTC": datetime.now(UTC).isoformat(),
        "driver_sha256": native.sha256(__file__),
        "source_environment": environment,
        "producer_manifest_sha256": args.manifest_sha256,
        "producer_record": original,
        "guard_sha256": args.guard_sha256,
        "utility_sha256": {
            Path(m.__file__).name: native.sha256(m.__file__)
            for m in (native, diagnostic, auditor)
        },
    }
    report_path = args.output_dir / "rtr-probe.json"
    native.write_json(report_path, report)
    try:
        model, warm, measure, covariance, inputs = native.load_inputs(
            args.data, args.report, args.warm
        )
        report["inputs"] = inputs
        if inputs["sha256"] != producer["inputs"]["sha256"]:
            raise ValueError("frozen native input identity differs from producer")
        full_init = native.point(warm.A.array, warm.c0.array, NULL, warm.pi.array)
        reducer = _ProfileReducer(full_init, ["s"])
        reduced_init = reducer.free_init(full_init)
        original_checkpoint = diagnostic.restore_checkpoint(
            checkpoint_path, original, reduced_init
        )
        reduced_model = map_model_parameters(
            model, lambda free: reducer.recombine(full_init, free)
        )

        # Preflight: the same restricted model, original anchor and retained
        # checkpoint must reproduce the original package objective/statistic.
        evaluate = build_estimator(
            reduced_model,
            measure=measure,
            covariance=covariance,
            theta_init=reduced_init,
            optimizer=diagnostic.CheckpointEvaluation(original_checkpoint),
        )
        old = evaluate(reduced_init, measure)
        np.testing.assert_allclose(
            old.objective_value,
            2 * original["optimizer_info"]["final_objective"],
            rtol=1e-10,
            atol=1e-9,
        )
        old_k = k_statistic(
            reducer.recombine(full_init, old.theta_hat),
            measure,
            covariance,
            model,
            interest=["s"],
        )
        for name in ("K", "S", "J", "p_K", "p_S", "p_J"):
            np.testing.assert_allclose(
                getattr(old_k, name),
                producer["statistics"][original["index"]][name],
                rtol=1e-7,
                atol=1e-9,
            )
        report["preflight"] = {
            "original_objective_and_k_parity": True,
            "old_objective": old.objective_value,
            "old_corrected_foc": guard(old, old.manifold_spec.total_gauge_dim),
        }
        native.write_json(report_path, report)

        if replay:
            if native.sha256(args.replay_report) != args.replay_report_sha256:
                raise ValueError("saved RTR report hash mismatch")
            prior = json.loads(args.replay_report.read_text())
            raw = prior["raw_optimizer_record"]
            optimizer_log = args.replay_report.parent / "optimizer.jsonl"
            if prior["outputs_sha256"][optimizer_log.name] != native.sha256(
                optimizer_log
            ) or [
                json.loads(line) for line in optimizer_log.read_text().splitlines()
            ] != [
                raw
            ]:
                raise ValueError("saved RTR optimizer record differs from raw log")
            if (
                prior["driver_sha256"] != args.original_driver_sha256
                or prior["producer_manifest_sha256"] != args.manifest_sha256
                or any(
                    prior["source_environment"][key] != environment[key]
                    for key in (
                        "source_files_sha256",
                        "driver_sha256",
                        "installed_distributions",
                        "python",
                        "jax_enable_x64",
                        "backend",
                        "numerical_environment",
                    )
                )
                or prior["inputs"]["sha256"] != inputs["sha256"]
                or prior["outputs_sha256"][args.replay_checkpoint.name]
                != native.sha256(args.replay_checkpoint)
                or raw["parameter_checkpoint_sha256"]
                != native.sha256(args.replay_checkpoint)
                or prior["outputs_sha256"][args.replay_arrays.name]
                != native.sha256(args.replay_arrays)
            ):
                raise ValueError("saved RTR source/input/checkpoint identity mismatch")
            target = diagnostic.restore_checkpoint(
                args.replay_checkpoint, raw, reduced_init
            )
            optimizer = CapturingEvaluation(target)
            report["replay_of"] = {
                "report_sha256": args.replay_report_sha256,
                "raw_record": raw,
                "arrays_sha256": native.sha256(args.replay_arrays),
            }
        else:
            optimizer = CapturingRecorder(
                riemannian_tr(
                    rtol=1e-9,
                    atol=1e-9,
                    max_steps=args.max_steps,
                    max_tcg_steps=args.max_tcg_steps,
                ),
                args.output_dir / "optimizer.jsonl",
            )
        run = build_estimator(
            reduced_model,
            measure=measure,
            covariance=covariance,
            theta_init=reduced_init,
            optimizer=optimizer,
        )
        result = run(reduced_init, measure)
        if replay:
            if optimizer.calls != 1:
                raise RuntimeError("expected exactly one checkpoint evaluation")
            report["checkpoint_evaluations"] = optimizer.calls
            report["new_optimizer_solves"] = 0
        else:
            if len(optimizer.records) != 1:
                raise RuntimeError("expected exactly one genuine restricted solve")
            report["raw_optimizer_record"] = optimizer.records[0]
            report["new_optimizer_solves"] = 1
        report["postfit_arrays_sha256"] = _save_result_arrays(
            result, args.output_dir / "postfit-arrays.npz"
        )
        report["postfit_arrays_file"] = "postfit-arrays.npz"
        if replay:
            with (
                np.load(args.replay_arrays) as previous,
                np.load(args.output_dir / "postfit-arrays.npz") as current,
            ):
                if set(previous.files) != set(current.files):
                    raise ValueError("saved RTR postfit ingredient keys changed")
                for name in current.files:
                    np.testing.assert_allclose(
                        current[name], previous[name], rtol=1e-10, atol=1e-9
                    )
            report["postfit_parity"] = True
        report["objective_value"] = result.objective_value
        report["diagnostics"] = result.diagnostics
        c0 = np.asarray(result.theta_hat.c0.array).item()
        amplitude = float(np.exp(-c0))
        report["amplitude"] = amplitude
        report["finite_interior"] = bool(
            all(
                np.isfinite(np.asarray(x)).all()
                for x in jax.tree_util.tree_leaves(result.theta_hat)
            )
            and np.isfinite(amplitude)
            and amplitude > 0
        )
        report["factor_full_rank"] = bool(
            np.linalg.matrix_rank(np.asarray(result.theta_hat.A.array)) == 2
        )
        report["foc_decrement"] = guard(result, result.manifold_spec.total_gauge_dim)
        report["foc_passes"] = report["foc_decrement"] is not None and bool(
            np.isfinite(report["foc_decrement"])
            and report["foc_decrement"] <= _FOC_TOL_DEFAULT
        )
        native.write_json(report_path, report)

        try:
            low, high = _curvature(optimizer, result.theta_hat)
            report["true_horizontal_curvature"] = {"minimum": low, "maximum": high}
        except Exception as exc:
            report["curvature_unavailable"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        try:
            report["sample_nuisance_strength"] = identification_strength(
                result,
                reduced_model,
                blocks={
                    "restricted_nuisance": range(result.manifold_spec.total_dimension)
                },
            )
        except Exception as exc:
            report["strength_unavailable"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        report["sample_status"] = (
            "interior_stationary_candidate"
            if report["finite_interior"]
            and report["factor_full_rank"]
            and report["foc_passes"]
            and "true_horizontal_curvature" in report
            and report["true_horizontal_curvature"]["minimum"]
            >= -(1e-9 + 1e-6 * abs(report["true_horizontal_curvature"]["maximum"]))
            else "unresolved_or_failed_local_conditions"
        )
    except Exception as exc:
        report["execution_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finished_UTC"] = datetime.now(UTC).isoformat()
        report["max_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["outputs_sha256"] = {
            p.name: native.sha256(p)
            for p in sorted(args.output_dir.iterdir())
            if p.is_file() and p != report_path
        }
        native.write_json(report_path, report)


if __name__ == "__main__":
    main()
