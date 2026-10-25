"""Evaluate a frozen native checkpoint with its original initial-point anchor.

This is diagnostic evaluation, not a solve or a new convergence certificate.
Reuses ledger Sections 7/28/29 and the actual profiler's model restriction.
Run against the producer's exact package/driver source and wheel environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import native_projected_set_180 as native
import numpy as np
from emu_gmm import build_estimator, identification_strength, k_statistic
from emu_gmm._internal.models import map_model_parameters
from emu_gmm.inference.confidence_set import (
    _FOC_TOL_DEFAULT,
    _foc_decrement,
    _ProfileReducer,
)
from emu_gmm.types import OptimizerInfo


def restore_checkpoint(path, record, initial):
    """Reject altered bytes or geometry before reconstructing the actual tree."""
    if native.sha256(path) != record["parameter_checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    pairs, tree = jax.tree_util.tree_flatten_with_path(initial)
    if repr(tree) != record["parameter_treedef_repr"]:
        raise ValueError("checkpoint parameter tree mismatch")
    expected = [
        {
            "key": f"leaf_{i:03d}",
            "path": [str(k) for k in keys],
            "shape": list(np.shape(leaf)),
            "dtype": str(np.asarray(leaf).dtype),
        }
        for i, (keys, leaf) in enumerate(pairs)
    ]
    if expected != record["parameter_leaf_order"]:
        raise ValueError("checkpoint leaf metadata mismatch")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {e["key"] for e in expected}:
            raise ValueError("checkpoint leaf keys mismatch")
        leaves = []
        for e in expected:
            leaf = z[e["key"]]
            if (
                list(leaf.shape) != e["shape"]
                or str(leaf.dtype) != e["dtype"]
                or not np.isfinite(leaf).all()
            ):
                raise ValueError("checkpoint leaf shape/dtype/finiteness mismatch")
            leaves.append(jnp.asarray(leaf))
    result = jax.tree_util.tree_unflatten(tree, leaves)
    if np.linalg.matrix_rank(np.asarray(result.A.array)) != 2:
        raise ValueError("checkpoint factor lost the declared rank")
    return result


class CheckpointEvaluation:
    """Supply a diagnostic point; explicitly perform and certify no solve."""

    def __init__(self, target):
        self.target = target
        self.calls = 0

    def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
        self.calls += 1
        return self.target, OptimizerInfo(
            steps=0,
            final_objective=jnp.asarray(jnp.nan),
            status="checkpoint_evaluation_no_solve",
            backend="diagnostic_checkpoint",
            done=jnp.asarray(False),
        )


def c0_section(target, offset):
    """Move only the level in its existing Euclidean leaf geometry."""
    return dataclasses.replace(
        target, c0=type(target.c0)(target.c0.array + offset, target.c0.manifold)
    )


def summarize(result, model, path):
    """Read package-owned quantities; never re-anchor or invert a new metric."""
    spec = result.manifold_spec
    leaves = identification_strength(result, model)
    joint = identification_strength(
        result, model, blocks={"joint_nuisance": range(spec.total_dimension)}
    )
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
    for name in ("A", "c0", "pi"):
        arrays[f"theta_hat_{name}"] = np.asarray(getattr(result.theta_hat, name).array)
        arrays[f"theta_init_{name}"] = np.asarray(
            getattr(result.theta_init, name).array
        )
    np.savez_compressed(path, **arrays)
    decrement = _foc_decrement(result, spec.total_gauge_dim)
    return {
        "kind": "diagnostic_checkpoint_evaluation_not_a_solve",
        "objective_value": result.objective_value,
        "foc_squared_decrement": decrement,
        "foc_decrement_finite": decrement is not None and bool(np.isfinite(decrement)),
        "original_foc_test_satisfied_at_checkpoint": decrement is not None
        and bool(np.isfinite(decrement) and decrement <= _FOC_TOL_DEFAULT),
        "original_profiler_foc_tol": _FOC_TOL_DEFAULT,
        "evaluation_converged": bool(result.converged),
        "evaluation_iterations": int(result.iterations),
        "diagnostics": result.diagnostics,
        "leaf_strength": leaves,
        "joint_nuisance_strength": joint,
        "gauge_dim": spec.total_gauge_dim,
        "c0": result.theta_hat.c0.array,
        "A_singular_values": np.linalg.svd(
            np.asarray(result.theta_hat.A.array), compute_uv=False
        ),
        "pi_norm": np.linalg.norm(np.asarray(result.theta_hat.pi.array)),
        "arrays_file": path.name,
        "arrays_sha256": native.sha256(path),
    }


def validate_producer(manifest, environment):
    """The new analysis script is separate; scientific source/env stay frozen."""
    original = manifest["source_environment"]
    for key in (
        "source_files_sha256",
        "driver_sha256",
        "installed_distributions",
        "python",
        "jax_enable_x64",
        "backend",
        "numerical_environment",
    ):
        if environment[key] != original[key]:
            raise ValueError(f"producer environment/source mismatch: {key}")
    protocol = manifest["protocol"]
    if (
        protocol["kind"] != "new-projected-native-current-CU-subvector-artifact"
        or protocol["interest"] != ["s"]
        or protocol["nuisance"] != ["A", "c0", "pi"]
        or protocol["penalty"] is not None
        or protocol["regularization"] != "package default DiagonalTikhonov"
        or protocol["weighting"] != "public profiler default ContinuouslyUpdated"
    ):
        raise ValueError("unsupported producer protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c0-offsets", default="-4,-1,1,4,8")
    args = parser.parse_args()
    offsets = [0.0] + [float(x) for x in args.c0_offsets.split(",") if x]
    if len(set(offsets)) != len(offsets) or not np.isfinite(offsets).all():
        parser.error("c0 offsets must be finite, unique and exclude zero")
    path = args.producer_dir / "manifest.json"
    if native.sha256(path) != args.manifest_sha256:
        raise ValueError("producer manifest hash mismatch")
    producer = json.loads(path.read_text())
    if not 0 <= args.index < len(producer["per_point"]):
        parser.error("point index is outside producer grid")
    environment = native.source_environment()
    validate_producer(producer, environment)
    record = producer["per_point"][args.index]
    s = record["grid_value"]
    if s != producer["protocol"]["grid"][args.index] or record["index"] != args.index:
        raise ValueError("producer point mapping mismatch")
    checkpoint = args.producer_dir / record["parameter_checkpoint"]
    if checkpoint.parent.resolve() != args.producer_dir.resolve():
        raise ValueError("checkpoint path escapes producer directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {
        "kind": "checkpoint_diagnosis_not_refit_or_updated_confidence_set",
        "started_UTC": datetime.now(UTC).isoformat(),
        "analysis_driver_sha256": native.sha256(__file__),
        "producer_manifest_sha256": args.manifest_sha256,
        "producer_record": record,
        "original_statistics": producer["statistics"][args.index],
        "source_environment": environment,
        "c0_offsets": offsets,
        "path_interpretation": "holds A/pi/s fixed; not reprofiled, not a global-optimum proof",
        "evaluations": [],
    }
    try:
        model, warm, measure, covariance, inputs = native.load_inputs(
            args.data, args.report, args.warm
        )
        report["inputs"] = inputs
        if inputs["sha256"] != producer["inputs"]["sha256"]:
            raise ValueError("producer inputs mismatch")
        full_init = native.point(warm.A.array, warm.c0.array, s, warm.pi.array)
        reducer = _ProfileReducer(full_init, ["s"])
        initial = reducer.free_init(full_init)
        target = restore_checkpoint(checkpoint, record, initial)
        reduced_model = map_model_parameters(
            model, lambda free: reducer.recombine(full_init, free)
        )
        optimizer = CheckpointEvaluation(target)
        # Ledger 29: factory and calls share the ORIGINAL warm start. Only the
        # diagnostic optimizer's returned target varies, so tau never reanchors.
        run = build_estimator(
            reduced_model,
            measure=measure,
            covariance=covariance,
            theta_init=initial,
            optimizer=optimizer,
        )
        for i, offset in enumerate(offsets):
            optimizer.target = c0_section(target, offset)
            result = run(initial, measure)
            row = summarize(
                result, reduced_model, args.output_dir / f"evaluation-{i:02d}.npz"
            )
            row["c0_offset"] = offset
            if i == 0:
                np.testing.assert_allclose(
                    result.objective_value,
                    2 * record["optimizer_info"]["final_objective"],
                    rtol=1e-10,
                    atol=1e-9,
                )
                ks = k_statistic(
                    reducer.recombine(full_init, result.theta_hat),
                    measure,
                    covariance,
                    model,
                    interest=["s"],
                )
                for name in (
                    "K",
                    "S",
                    "J",
                    "p_K",
                    "p_S",
                    "p_J",
                    "df_K",
                    "df_S",
                    "df_J",
                ):
                    np.testing.assert_allclose(
                        getattr(ks, name),
                        producer["statistics"][args.index][name],
                        rtol=1e-7,
                        atol=1e-9,
                    )
                report["original_objective_and_public_K_parity"] = True
                report["recomputed_statistics"] = ks
            report["evaluations"].append(row)
            native.write_json(args.output_dir / "diagnosis.json", report)
            print(
                json.dumps(
                    native.json_value(
                        {
                            "event": "evaluated",
                            "c0_offset": offset,
                            "objective": result.objective_value,
                            "foc": row["foc_squared_decrement"],
                            "leaf_strength": row["leaf_strength"],
                        }
                    )
                ),
                flush=True,
            )
    except Exception as exc:
        report["execution_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finished_UTC"] = datetime.now(UTC).isoformat()
        native.write_json(args.output_dir / "diagnosis.json", report)


if __name__ == "__main__":
    main()
