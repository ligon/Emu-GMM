"""Checkpoint identity and original-anchor controls for native issue 180."""

import copy
import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from emu_gmm import build_estimator
from emu_gmm._internal.models import map_model_parameters
from emu_gmm.inference.confidence_set import _ProfileReducer

_VALIDATION = Path(__file__).parents[1] / "scripts/validation"
for _name in ("native_projected_set_180", "native_profile_diagnostics_180"):
    _spec = importlib.util.spec_from_file_location(_name, _VALIDATION / f"{_name}.py")
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)

driver = sys.modules["native_profile_diagnostics_180"]
native = sys.modules["native_projected_set_180"]


def _checkpoint(tmp_path):
    full = native.point(
        np.arange(10).reshape(5, 2) + np.eye(5, 2), 0.1, 0.2, np.arange(10.0)
    )
    initial = _ProfileReducer(full, ["s"]).free_init(full)
    pairs, tree = jax.tree_util.tree_flatten_with_path(initial)
    path = tmp_path / "checkpoint.npz"
    np.savez_compressed(
        path, **{f"leaf_{i:03d}": np.asarray(a) for i, (_, a) in enumerate(pairs)}
    )
    record = {
        "parameter_checkpoint_sha256": native.sha256(path),
        "parameter_treedef_repr": repr(tree),
        "parameter_leaf_order": [
            {
                "key": f"leaf_{i:03d}",
                "path": [str(k) for k in p],
                "shape": list(a.shape),
                "dtype": str(a.dtype),
            }
            for i, (p, a) in enumerate(pairs)
        ],
    }
    return path, record, initial


def test_checkpoint_roundtrip_and_no_solve_status(tmp_path):
    path, record, initial = _checkpoint(tmp_path)
    target = driver.restore_checkpoint(path, record, initial)
    for a, b in zip(
        jax.tree_util.tree_leaves(target),
        jax.tree_util.tree_leaves(initial),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    optimizer = driver.CheckpointEvaluation(target)
    actual, info = optimizer(
        lambda *a: pytest.fail("must not run a residual or solve"), initial, None
    )
    assert actual is target and optimizer.calls == 1
    assert not bool(info.done) and info.steps == 0
    assert info.status == "checkpoint_evaluation_no_solve"


@pytest.mark.parametrize(
    "corruption", ["hash", "tree", "metadata", "dtype", "rank", "keys"]
)
def test_checkpoint_refuses_corruption(tmp_path, corruption):
    path, record, initial = _checkpoint(tmp_path)
    if corruption == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
    elif corruption == "tree":
        record["parameter_treedef_repr"] += "changed"
    elif corruption == "metadata":
        record["parameter_leaf_order"][0]["path"] = ["wrong owner"]
    else:
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        if corruption == "dtype":
            arrays["leaf_000"] = arrays["leaf_000"].astype(np.float32)
        elif corruption == "rank":
            arrays["leaf_000"] = np.ones((5, 2))
        else:
            arrays["unexpected"] = np.zeros(1)
        np.savez_compressed(path, **arrays)
        record["parameter_checkpoint_sha256"] = native.sha256(path)
    with pytest.raises(ValueError, match="checkpoint"):
        driver.restore_checkpoint(path, record, initial)


@pytest.mark.parametrize(
    "changed", ["source_files_sha256", "driver_sha256", "numerical_environment"]
)
def test_source_and_environment_changes_are_rejected(changed):
    environment = {k: {} for k in ("source_files_sha256", "installed_distributions")}
    environment.update(python="frozen Python", jax_enable_x64=True, backend="cpu")
    environment.update(driver_sha256="frozen driver", numerical_environment={})
    manifest = {
        "source_environment": copy.deepcopy(environment),
        "protocol": {
            "kind": "new-projected-native-current-CU-subvector-artifact",
            "interest": ["s"],
            "nuisance": ["A", "c0", "pi"],
            "penalty": None,
            "regularization": "package default DiagonalTikhonov",
            "weighting": "public profiler default ContinuouslyUpdated",
        },
    }
    driver.validate_producer(manifest, environment)
    environment[changed] = "changed"
    with pytest.raises(ValueError, match="source"):
        driver.validate_producer(manifest, environment)


def test_checkpoint_evaluation_retains_initial_anchor_and_current_target(tmp_path):
    model, full, measure, covariance, _ = native.synthetic_smoke_inputs()
    reducer = _ProfileReducer(full, ["s"])
    initial = reducer.free_init(full)
    reduced = map_model_parameters(model, lambda free: reducer.recombine(full, free))
    optimizer = driver.CheckpointEvaluation(initial)
    run = build_estimator(
        reduced,
        measure=measure,
        covariance=covariance,
        theta_init=initial,
        optimizer=optimizer,
    )
    first = run(initial, measure)
    optimizer.target = driver.c0_section(initial, 1.0)
    second = run(initial, measure)
    assert not first.converged and not second.converged
    np.testing.assert_array_equal(second.theta_init.c0.array, initial.c0.array)
    np.testing.assert_array_equal(second.theta_hat.c0.array, initial.c0.array + 1.0)
    np.testing.assert_array_equal(
        first.diagnostics.tau_realised, second.diagnostics.tau_realised
    )
    assert float(first.objective_value) != float(second.objective_value)
    report = driver.summarize(second, reduced, tmp_path / "genuine-result.npz")
    assert report["gauge_dim"] == 1 and not report["evaluation_converged"]
    with np.load(tmp_path / "genuine-result.npz") as z:
        np.testing.assert_array_equal(z["gn_hessian"], np.asarray(second.gn_hessian))
        np.testing.assert_array_equal(z["theta_init_c0"], initial.c0.array)
