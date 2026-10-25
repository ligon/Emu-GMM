"""Single-null positive-amplitude optimization experiment (ledger Section 31).

Stage the original scientific package/native driver and pinned dependencies.
Only the corrected FOC helper is loaded from a separately pinned module.
This changes optimization coordinates, not the model or inference policy.
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
import jax.numpy as jnp
import jax_dataclasses as jdc
import native_profile_diagnostics_180 as diagnostic
import native_projected_set_180 as native
import numpy as np
from emu_gmm import (
    ManifoldLeaf,
    Positive,
    build_estimator,
    identification_strength,
    k_statistic,
    riemannian_lm,
)
from emu_gmm._internal.models import map_model_parameters
from emu_gmm.inference.confidence_set import _FOC_TOL_DEFAULT, _ProfileReducer
from emu_gmm.types import OptimizerInfo

NULL = 0.050088084721517094
GUARD_SHA256 = "1a6261dfc4b41d4be3ffee6aa4e2d88deda6ccd5e459ed19791c72f30b1d85f0"


@jdc.pytree_dataclass
class AmplitudePoint:
    A: ManifoldLeaf
    delta: ManifoldLeaf
    s: ManifoldLeaf
    pi: ManifoldLeaf


def to_amplitude(theta):
    return AmplitudePoint(
        theta.A,
        ManifoldLeaf(jnp.exp(-theta.c0.array[0]), Positive()),
        theta.s,
        theta.pi,
    )


def to_physical(theta):
    # No clipping or stopped derivatives. This adapter also runs under AD.
    return native.point(
        theta.A.array, -jnp.log(theta.delta.array), theta.s.array, theta.pi.array
    )


def interior_events(theta):
    """Host-side domain accounting; never quietly map a boundary to c0=inf."""
    delta = np.asarray(theta.delta.array)
    leaves = jax.tree_util.tree_leaves(theta)
    events = {
        "delta_nonfinite": not bool(np.isfinite(delta).all()),
        "delta_underflow_or_zero": bool(np.any(delta == 0)),
        "delta_negative": bool(np.any(delta < 0)),
        "parameter_nonfinite": not all(
            np.isfinite(np.asarray(x)).all() for x in leaves
        ),
    }
    events["finite_interior"] = not any(events.values())
    return events


class FiniteInteriorError(RuntimeError):
    pass


class ExperimentOptimizer:
    """A mutable v2 protocol wrapper: evaluations first, exactly one solve."""

    def __init__(self, target, fixed_s=None):
        self.target = target
        self.fixed_s = fixed_s
        self.recorder = None
        self.solve_calls = 0

    def enable_solve(self, delegate, path):
        if self.recorder is not None or self.solve_calls:
            raise RuntimeError("the experiment permits only one solve")
        self.recorder = native.RecordingOptimizer(delegate, path)

    def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
        if self.recorder is None:
            return self.target, OptimizerInfo(
                steps=0,
                final_objective=jnp.asarray(jnp.nan),
                status="amplitude_parity_evaluation_no_solve",
                backend="diagnostic_checkpoint",
                done=jnp.asarray(False),
            )
        if self.solve_calls:
            raise RuntimeError("a second optimization is outside this experiment")
        self.solve_calls += 1
        theta, info = self.recorder(residual_fn, theta_init, manifold_spec, args=args)
        events = interior_events(theta)
        events["physical_mapping_performed"] = False
        if events["finite_interior"] and self.fixed_s is not None:
            physical = native.point(
                theta.A.array, -jnp.log(theta.delta.array), self.fixed_s, theta.pi.array
            )
            mapped = self.recorder.path.parent / "returned-physical-state.npz"
            np.savez_compressed(
                mapped,
                **{
                    f.name: np.asarray(getattr(physical, f.name).array)
                    for f in dataclasses.fields(physical)
                },
            )
            events.update(
                physical_mapping_performed=True,
                physical_checkpoint=mapped.name,
                physical_checkpoint_sha256=native.sha256(mapped),
            )
        native.write_json(
            self.recorder.path.parent / "returned-domain-events.json", events
        )
        if not events["finite_interior"]:
            # RecordingOptimizer has already durably saved the raw returned
            # leaves/info. Do not evaluate log(delta), inference, or K here.
            raise FiniteInteriorError(
                "optimizer returned outside finite positive interior"
            )
        return theta, info


def protocol():
    return {
        "kind": "single-null-bijective-amplitude-optimization-experiment",
        "interest": {"s": NULL},
        "nuisance": ["A", "delta", "pi"],
        "mapping": "delta=exp(-c0)>0; c0=-log(delta)",
        "geometry": "PSDFixedRank(5,2) x Positive() x Euclidean(10); scalar delta",
        "optimizer": {
            "backend": "riemannian_lm",
            "rtol": 1e-9,
            "atol": 1e-9,
            "max_steps": 1500,
        },
        "initialization": "same original physical warm nuisance; no checkpoint restart",
        "weighting": "ContinuouslyUpdated; unchanged original warm anchor",
        "regularization": "package default DiagonalTikhonov",
        "penalty": None,
        "parity_controls": [
            "warm",
            "saved-checkpoint",
            "checkpoint-c0-minus1",
            "checkpoint-c0-plus1",
        ],
        "interpretation": "changed optimization geometry; not an original replay, confidence set, boundary-valid inference or strong-nuisance certificate",
    }


def result_arrays(result):
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
    arrays["means"] = np.asarray(result.diagnostics.moment_residual.array)
    for where in ("theta_hat", "theta_init"):
        for field in dataclasses.fields(getattr(result, where)):
            arrays[f"{where}_{field.name}"] = np.asarray(
                getattr(getattr(result, where), field.name).array
            )
    return arrays


def summarize(result, model, path, guard):
    arrays = result_arrays(result)
    np.savez_compressed(path, **arrays)
    spec = result.manifold_spec
    decrement = guard(result, spec.total_gauge_dim)
    return {
        "objective_value": result.objective_value,
        "optimizer_certified": bool(result.converged),
        "iterations": int(result.iterations),
        "diagnostics": result.diagnostics,
        "corrected_foc_decrement": decrement,
        "corrected_foc_available": decrement is not None
        and bool(np.isfinite(decrement)),
        "corrected_foc_satisfied": decrement is not None
        and bool(np.isfinite(decrement) and decrement <= _FOC_TOL_DEFAULT),
        "foc_tol": _FOC_TOL_DEFAULT,
        "leaf_strength": identification_strength(result, model),
        "joint_nuisance_strength": identification_strength(
            result, model, blocks={"joint_nuisance": range(spec.total_dimension)}
        ),
        "manifold_spec": {
            "total_dimension": spec.total_dimension,
            "total_gauge_dim": spec.total_gauge_dim,
            "leaves": [
                {
                    "field_name": leaf.field_name,
                    "ambient_shape": leaf.ambient_shape,
                    "manifold": repr(leaf.manifold),
                }
                for leaf in spec.leaf_specs
            ],
        },
        "arrays_file": path.name,
        "arrays_sha256": native.sha256(path),
    }


def chain_diagonal(physical_spec, amplitude_spec, delta):
    """Actual registered leaf ownership supplies the derivative column order."""
    if (
        physical_spec.total_dimension != amplitude_spec.total_dimension
        or physical_spec.total_gauge_dim != amplitude_spec.total_gauge_dim
    ):
        raise ValueError("coordinate change altered dimension or gauge")
    parts = []
    for old, new in zip(
        physical_spec.leaf_specs, amplitude_spec.leaf_specs, strict=True
    ):
        if old.field_name != {"delta": "c0"}.get(new.field_name, new.field_name):
            raise ValueError("coordinate leaf ownership/order changed")
        size = int(np.prod(old.ambient_shape))
        if size != int(np.prod(new.ambient_shape)):
            raise ValueError("coordinate leaf size changed")
        parts.append(np.full(size, -1.0 / delta if new.field_name == "delta" else 1.0))
    return np.concatenate(parts)


class PreparedExperiment:
    def __init__(self, model, warm, measure, covariance):
        self.model, self.measure, self.covariance = model, measure, covariance
        self.warm = warm
        self.amplitude_warm = to_amplitude(warm)
        self.physical_reducer = _ProfileReducer(warm, ["s"])
        self.amplitude_reducer = _ProfileReducer(self.amplitude_warm, ["s"])
        self.physical_initial = self.physical_reducer.free_init(warm)
        self.amplitude_initial = self.amplitude_reducer.free_init(self.amplitude_warm)
        self.physical_model = map_model_parameters(
            model, lambda free: self.physical_reducer.recombine(warm, free)
        )
        self.amplitude_model = map_model_parameters(
            model,
            lambda free: to_physical(
                self.amplitude_reducer.recombine(self.amplitude_warm, free)
            ),
        )
        self.physical_optimizer = ExperimentOptimizer(self.physical_initial)
        self.amplitude_optimizer = ExperimentOptimizer(
            self.amplitude_initial, warm.s.array
        )
        # Both factories are anchored exactly once at the same physical warm
        # point. Calls below always pass their original warm initial tree.
        self.physical_run = build_estimator(
            self.physical_model,
            measure=measure,
            covariance=covariance,
            theta_init=self.physical_initial,
            optimizer=self.physical_optimizer,
        )
        self.amplitude_run = build_estimator(
            self.amplitude_model,
            measure=measure,
            covariance=covariance,
            theta_init=self.amplitude_initial,
            optimizer=self.amplitude_optimizer,
        )
        self.preflight_passed = False

    def parity(self, physical_point, output, name, guard, amplitude_result=None):
        amplitude = to_amplitude(physical_point)
        if not interior_events(amplitude)["finite_interior"]:
            raise FiniteInteriorError(
                "parity control is outside finite positive interior"
            )
        self.physical_optimizer.target = self.physical_reducer.free_init(physical_point)
        old = self.physical_run(self.physical_initial, self.measure)
        if amplitude_result is None:
            self.amplitude_optimizer.target = self.amplitude_reducer.free_init(
                amplitude
            )
            new = self.amplitude_run(self.amplitude_initial, self.measure)
        else:
            # Reuse the genuine solve result; do not call its delegate twice.
            new = amplitude_result
        a, b = result_arrays(old), result_arrays(new)
        diagonal = chain_diagonal(
            old.manifold_spec, new.manifold_spec, float(amplitude.delta.array)
        )
        for key in (
            "means",
            "moment_covariance",
            "objective_value",
        ):
            np.testing.assert_allclose(b[key], a[key], rtol=1e-8, atol=1e-9)
        # Inverting an ill-conditioned V amplifies log/exp roundoff. A matrix
        # norm avoids treating a near-cancelled entry of Lambda as its scale.
        metric_error = np.linalg.norm(b["weighting_matrix"] - a["weighting_matrix"])
        metric_scale = np.linalg.norm(a["weighting_matrix"])
        if not np.isfinite(metric_error) or metric_error > 1e-8 * metric_scale:
            raise AssertionError("anchored weighting-matrix norm parity failed")
        for key, expected in (
            ("gradient", diagonal * a["gradient"]),
            ("moment_jacobian", a["moment_jacobian"] * diagonal[None, :]),
            ("gn_hessian", (diagonal[:, None] * a["gn_hessian"]) * diagonal[None, :]),
        ):
            np.testing.assert_allclose(b[key], expected, rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(
            old.diagnostics.tau_realised, new.diagnostics.tau_realised
        )
        row = {
            "name": name,
            "parity_passed": True,
            "physical": summarize(
                old, self.physical_model, output / f"{name}-physical.npz", guard
            ),
            "amplitude": summarize(
                new, self.amplitude_model, output / f"{name}-amplitude.npz", guard
            ),
            "chain_diagonal": diagonal,
            "weighting_matrix_relative_norm_error": metric_error / metric_scale,
        }
        return row

    def preflight(self, checkpoint, output, guard):
        controls = [
            ("warm", self.warm),
            ("checkpoint", checkpoint),
            (
                "checkpoint-minus1",
                native.point(
                    checkpoint.A.array,
                    checkpoint.c0.array - 1,
                    checkpoint.s.array,
                    checkpoint.pi.array,
                ),
            ),
            (
                "checkpoint-plus1",
                native.point(
                    checkpoint.A.array,
                    checkpoint.c0.array + 1,
                    checkpoint.s.array,
                    checkpoint.pi.array,
                ),
            ),
        ]
        rows = [self.parity(point, output, name, guard) for name, point in controls]
        tau = rows[0]["physical"]["diagnostics"].tau_realised
        for row in rows:
            for chart in ("physical", "amplitude"):
                np.testing.assert_array_equal(
                    row[chart]["diagnostics"].tau_realised, tau
                )
        self.preflight_passed = True
        return rows

    def solve(self, output, guard, delegate=None):
        if not self.preflight_passed:
            raise RuntimeError("all parity controls must pass before optimization")
        self.amplitude_optimizer.enable_solve(
            (
                riemannian_lm(rtol=1e-9, atol=1e-9, max_steps=1500)
                if delegate is None
                else delegate
            ),
            output / "optimizer.jsonl",
        )
        result = self.amplitude_run(self.amplitude_initial, self.measure)
        full = self.amplitude_reducer.recombine(self.amplitude_warm, result.theta_hat)
        events = interior_events(full)
        if not events["finite_interior"]:
            raise FiniteInteriorError(
                "no finite physical interpretation of returned point"
            )
        physical = to_physical(full)
        if not all(
            np.isfinite(np.asarray(x)).all()
            for x in jax.tree_util.tree_leaves(physical)
        ):
            raise FiniteInteriorError("mapped physical point is nonfinite")
        amplitude_summary = summarize(
            result, self.amplitude_model, output / "fit-amplitude.npz", guard
        )
        raw_record = self.amplitude_optimizer.recorder.records[0]
        returned_control = self.parity(
            physical, output, "returned-fit-control", guard, amplitude_result=result
        )
        return {
            "finite_interior_events": events,
            "amplitude_result": amplitude_summary,
            "physical_checkpoint_evaluation_not_solve": returned_control["physical"],
            "returned_fit_control": returned_control,
            "physical_point": {
                f.name: getattr(physical, f.name).array
                for f in dataclasses.fields(physical)
            },
            "raw_optimizer_record": raw_record,
            "interpretation": "no confidence-set verdict; optimizer certification and FOC evidence remain separate",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--guard", type=Path)
    parser.add_argument("--guard-sha256", default=GUARD_SHA256)
    parser.add_argument("--data", type=Path, default=native.DATA)
    parser.add_argument("--report", type=Path, default=native.REPORT)
    parser.add_argument("--warm", type=Path, default=native.WARM)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-protocol", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--raw-k",
        action="store_true",
        help="optional package K/S/J at returned physical point, without a profile-validity claim",
    )
    args = parser.parse_args()
    if args.print_protocol:
        print(json.dumps(protocol(), indent=2, sort_keys=True))
        return
    if any(
        x is None
        for x in (args.producer_dir, args.manifest_sha256, args.guard, args.output_dir)
    ):
        parser.error("producer-dir, manifest-sha256, guard and output-dir are required")
    producer_path = args.producer_dir / "manifest.json"
    if native.sha256(producer_path) != args.manifest_sha256:
        raise ValueError("producer manifest hash mismatch")
    producer = json.loads(producer_path.read_text())
    environment = native.source_environment()
    diagnostic.validate_producer(producer, environment)
    matches = [r for r in producer["per_point"] if r["grid_value"] == NULL]
    if len(matches) != 1:
        raise ValueError("producer must contain exactly the original failed null")
    record = matches[0]
    if producer["protocol"]["grid"][record["index"]] != NULL:
        raise ValueError("producer point mapping mismatch")
    guard = auditor.load_guard(args.guard, args.guard_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {
        "protocol": protocol(),
        "started_UTC": datetime.now(UTC).isoformat(),
        "driver_sha256": native.sha256(__file__),
        "source_environment": environment,
        "utility_sha256": {
            Path(m.__file__).name: native.sha256(m.__file__)
            for m in (native, diagnostic, auditor)
        },
        "guard_sha256": args.guard_sha256,
        "producer_manifest_sha256": args.manifest_sha256,
        "producer_record": record,
        "preflight_only": args.preflight_only,
    }
    try:
        model, original_warm, measure, covariance, inputs = native.load_inputs(
            args.data, args.report, args.warm
        )
        report["inputs"] = inputs
        if inputs["sha256"] != producer["inputs"]["sha256"]:
            raise ValueError("producer inputs mismatch")
        warm = native.point(
            original_warm.A.array, original_warm.c0.array, NULL, original_warm.pi.array
        )
        reducer = _ProfileReducer(warm, ["s"])
        checkpoint_path = args.producer_dir / record["parameter_checkpoint"]
        if checkpoint_path.parent.resolve() != args.producer_dir.resolve():
            raise ValueError("checkpoint path escapes producer directory")
        checkpoint = reducer.recombine(
            warm,
            diagnostic.restore_checkpoint(
                checkpoint_path, record, reducer.free_init(warm)
            ),
        )
        prepared = PreparedExperiment(model, warm, measure, covariance)
        report["preflight"] = prepared.preflight(checkpoint, args.output_dir, guard)
        np.testing.assert_allclose(
            report["preflight"][1]["physical"]["objective_value"],
            2 * record["optimizer_info"]["final_objective"],
            rtol=1e-10,
            atol=1e-9,
        )
        report["preflight_passed"] = True
        native.write_json(args.output_dir / "amplitude.json", report)
        if not args.preflight_only:
            report["fit"] = prepared.solve(args.output_dir, guard)
            if args.raw_k:
                report["raw_k_without_profile_validity"] = k_statistic(
                    native.point(**report["fit"]["physical_point"]),
                    measure,
                    covariance,
                    model,
                    interest=["s"],
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
            if p.is_file() and p.name != "amplitude.json"
        }
        native.write_json(args.output_dir / "amplitude.json", report)


if __name__ == "__main__":
    main()
