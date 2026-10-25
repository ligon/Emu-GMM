"""Tiny coordinate/AD/anchor and domain controls, never native-panel fits."""

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import ManifoldLeaf, Positive, riemannian_lm
from emu_gmm.inference.confidence_set import _foc_decrement
from emu_gmm.types import OptimizerInfo

_VALIDATION = Path(__file__).parents[1] / "scripts/validation"
for _name in (
    "native_projected_set_180",
    "native_profile_diagnostics_180",
    "audit_profile_checkpoints_180",
    "native_amplitude_diagnosis_180",
):
    _spec = importlib.util.spec_from_file_location(_name, _VALIDATION / f"{_name}.py")
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)

driver = sys.modules["native_amplitude_diagnosis_180"]
native = sys.modules["native_projected_set_180"]


def test_mapping_is_bijective_and_differentiated_without_frozen_aux():
    model, theta, measure, _covariance, _ = native.synthetic_smoke_inputs()
    amplitude = driver.to_amplitude(theta)
    assert amplitude.delta.array.shape == ()
    assert isinstance(amplitude.delta.manifold, Positive)
    restored = driver.to_physical(amplitude)
    for a, b in zip(
        jax.tree_util.tree_leaves(theta),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)

    def amplitude_mean(delta):
        point = type(amplitude)(
            amplitude.A, ManifoldLeaf(delta, Positive()), amplitude.s, amplitude.pi
        )
        return measure.expectation(
            driver.map_model_parameters(model, driver.to_physical), point
        )

    def physical_mean(c0):
        return measure.expectation(
            model, native.point(theta.A.array, c0, theta.s.array, theta.pi.array)
        )

    expected = jax.jacfwd(physical_mean)(theta.c0.array[0]) * (
        -1 / amplitude.delta.array
    )
    actual = jax.jacfwd(amplitude_mean)(amplitude.delta.array)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_preflight_and_one_tiny_native_solve_keep_original_warm_anchor(tmp_path):
    model, warm, measure, covariance, _ = native.synthetic_smoke_inputs()
    prepared = driver.PreparedExperiment(model, warm, measure, covariance)
    with pytest.raises(RuntimeError, match="parity controls"):
        prepared.solve(tmp_path, _foc_decrement)
    checkpoint = native.point(
        warm.A.array, warm.c0.array + 0.2, warm.s.array, warm.pi.array
    )
    rows = prepared.preflight(checkpoint, tmp_path, _foc_decrement)
    assert len(rows) == 4 and all(row["parity_passed"] for row in rows)
    assert prepared.amplitude_optimizer.solve_calls == 0
    assert all(row["physical"]["manifold_spec"]["total_gauge_dim"] == 1 for row in rows)
    result = prepared.solve(
        tmp_path,
        _foc_decrement,
        delegate=riemannian_lm(rtol=1e-9, atol=1e-9, max_steps=2),
    )
    assert prepared.amplitude_optimizer.solve_calls == 1
    assert result["finite_interior_events"]["finite_interior"]
    assert result["amplitude_result"]["iterations"] <= 2
    assert result["returned_fit_control"]["parity_passed"]
    with np.load(tmp_path / "fit-amplitude.npz") as arrays:
        np.testing.assert_array_equal(
            arrays["theta_init_delta"], prepared.amplitude_initial.delta.array
        )
        np.testing.assert_array_equal(arrays["theta_init_A"], warm.A.array)
    np.testing.assert_array_equal(
        rows[0]["physical"]["diagnostics"].tau_realised,
        result["amplitude_result"]["diagnostics"].tau_realised,
    )
    assert (tmp_path / "point-000-nuisance.npz").is_file()
    assert (tmp_path / "returned-physical-state.npz").is_file()
    domain = json.loads((tmp_path / "returned-domain-events.json").read_text())
    assert domain["physical_mapping_performed"]
    native.write_json(tmp_path / "summary.json", result)
    assert (
        "corrected_foc_decrement"
        in json.loads((tmp_path / "summary.json").read_text())["amplitude_result"]
    )
    with pytest.raises(RuntimeError, match="one solve"):
        prepared.solve(tmp_path, _foc_decrement)


@pytest.mark.parametrize(
    "delta,event",
    [
        (0.0, "delta_underflow_or_zero"),
        (-1.0, "delta_negative"),
        (np.inf, "delta_nonfinite"),
        (np.nan, "delta_nonfinite"),
    ],
)
def test_invalid_amplitude_is_checkpointed_without_physical_mapping(
    tmp_path, delta, event
):
    _model, physical, _measure, _covariance, _ = native.synthetic_smoke_inputs()
    point = driver.to_amplitude(physical)
    bad = type(point)(
        point.A, ManifoldLeaf(jnp.asarray(delta), Positive()), point.s, point.pi
    )
    info = OptimizerInfo(
        steps=1, final_objective=jnp.nan, status="stagnated", backend="test", done=False
    )

    def fake(fn, initial, spec, *, args=None):
        assert initial is point and args == "sentinel"
        return bad, info

    wrapper = driver.ExperimentOptimizer(point, point.s.array)
    wrapper.enable_solve(fake, tmp_path / "optimizer.jsonl")
    with pytest.raises(driver.FiniteInteriorError):
        wrapper(
            lambda *a: pytest.fail("no post-solve evaluation"),
            point,
            None,
            args="sentinel",
        )
    events = json.loads((tmp_path / "returned-domain-events.json").read_text())
    assert events[event] and not events["finite_interior"]
    assert not events["physical_mapping_performed"]
    assert not (tmp_path / "returned-physical-state.npz").exists()
    assert (tmp_path / "point-000-nuisance.npz").exists()
    assert wrapper.recorder.records[0]["optimizer_info"] is info


def test_protocol_print_does_not_read_inputs_or_create_outputs(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["amplitude", "--print-protocol"])
    monkeypatch.setattr(native, "load_inputs", lambda *a: pytest.fail("no data reads"))
    driver.main()
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["interest"] == {"s": driver.NULL}
    assert protocol["optimizer"]["max_steps"] == 1500
    assert protocol["penalty"] is None
