"""Small independent FE/complete-D oracles for the native validation driver."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from emu_gmm import ClusteredCovariance, EmpiricalMeasure
from emu_gmm._internal.params import flatten_params_for_ad
from emu_gmm.inference.confidence_set import KConfidenceSet
from emu_gmm.types import OptimizerInfo

_PATH = Path(__file__).parents[1] / "scripts/validation/native_projected_set_180.py"
_SPEC = importlib.util.spec_from_file_location("native_projected_set_180", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


def _setup():
    # Connected, incomplete, unbalanced product/market incidence, including
    # repeated cells; no special orthogonality is imposed on instruments.
    product = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    market = np.array([0, 1, 1, 1, 2, 3, 0, 2, 3, 3])
    rng = np.random.default_rng(180)
    x = rng.normal(size=(len(product), 85))
    x[:, 0] += 2
    direction = np.array([0.2, -0.4, 0.1, 0.7, -0.5])
    model = driver.ProjectedModel(direction, 3, 4)
    measure = EmpiricalMeasure(
        jnp.array(np.column_stack([x, product, market])),
        jnp.ones((len(product), 43)),
        jnp.ones(len(product)),
    )
    theta = driver.point(rng.normal(size=(5, 2)), 0.1, 0.2, rng.normal(size=10))
    F = np.column_stack([np.eye(3)[product], np.eye(4)[market]])
    P = np.eye(len(product)) - F @ np.linalg.pinv(F)
    return model, measure, theta, P


def test_schur_aux_matches_independent_joint_numpy_projection():
    model, measure, theta, P = _setup()
    x = np.asarray(measure.x)
    product, market = x[:, 85].astype(int), x[:, 86].astype(int)
    y = np.asarray(model.transformed_outcome(measure.x, theta))
    a, b = model.aux(measure, theta)
    actual = y - np.asarray(a)[product] - np.asarray(b)[market]
    np.testing.assert_allclose(actual, P @ y, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(np.sum(a), 0.0, atol=1e-14)
    # Changing current theta recomputes the effects, rather than freezing them.
    changed = driver.point(theta.A.array, 0.3, -0.15, theta.pi.array)
    a2, b2 = model.aux(measure, changed)
    assert np.max(np.abs(np.asarray(a2) - a)) > 0.01
    y2 = np.asarray(model.transformed_outcome(measure.x, changed))
    np.testing.assert_allclose(
        y2 - np.asarray(a2)[product] - np.asarray(b2)[market],
        P @ y2,
        atol=2e-14,
        rtol=2e-14,
    )


def test_complete_row_D_and_G_V_match_independent_numpy_chain_rule():
    model, measure, theta, P = _setup()
    x = np.asarray(measure.x)
    A = np.asarray(theta.A.array)
    y = np.asarray(model.transformed_outcome(measure.x, theta))
    dgamma = []
    for j in range(10):
        tangent = np.zeros((5, 2))
        tangent.flat[j] = 1
        dgamma.append((tangent @ A.T + A @ tangent.T).ravel())
    de = np.column_stack(
        [
            -x[:, 2:27] @ np.array(dgamma).T,
            P @ (-y),
            P @ (-(x[:, 27:32] @ np.asarray(model.direction)) * y),
            -x[:, 32:42],
        ]
    )
    expected_D = x[:, 42:85, None] * de[:, None, :]
    D = np.asarray(measure.jacobian_contributions(model, theta))
    assert D.shape == (len(x), 43, 22)  # Appended keys cannot become moments.
    np.testing.assert_allclose(D, expected_D, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(
        measure.jacobian(model, theta), expected_D.mean(0), atol=1e-13, rtol=1e-13
    )
    aux = model.aux(measure, theta)
    frozen_D = np.asarray(
        measure.jacobian_contributions(lambda row, th: model.psi(row, th, aux), theta)
    )
    assert np.max(np.abs(D[:, :, 10:12] - frozen_D[:, :, 10:12])) > 0.1
    y_within = P @ y
    residual = (
        y_within
        - x[:, 1]
        - x[:, 2:27] @ (A @ A.T).ravel()
        - x[:, 32:42] @ np.asarray(theta.pi.array)
    )
    contributions = residual[:, None] * x[:, 42:85]
    np.testing.assert_allclose(
        measure.expectation(model, theta), contributions.mean(0), atol=1e-13, rtol=1e-13
    )
    clusters = x[:, 86].astype(int)
    totals = np.zeros((4, 43))
    np.add.at(totals, clusters, contributions)
    covariance = ClusteredCovariance(jnp.asarray(clusters, dtype=float), 4)
    np.testing.assert_allclose(
        covariance.covariance(model, theta, measure),
        totals.T @ totals / len(x) ** 2,
        atol=1e-12,
        rtol=1e-12,
    )
    flat, _, _ = flatten_params_for_ad(theta)
    gauge = np.zeros(len(flat))
    gauge[:10] = (A @ np.array([[0.0, -1.0], [1.0, 0.0]])).ravel()
    np.testing.assert_allclose(D @ gauge, 0, atol=1e-12)


def test_jitted_aux_uses_current_measure_group_keys_and_outcomes():
    model, measure, theta, _ = _setup()
    evaluate = jax.jit(lambda mu, th: mu.moment_contributions(model, th))
    first = np.asarray(evaluate(measure, theta))
    x = np.asarray(measure.x).copy()
    x[0, 85] = 1  # Changes counts and incidence while retaining connectedness.
    x[:, 0] += np.linspace(-0.1, 0.3, len(x))
    fresh = EmpiricalMeasure(jnp.asarray(x), measure.mask, measure.weights)
    second = np.asarray(evaluate(fresh, theta))
    F = np.column_stack(
        [np.eye(3)[x[:, 85].astype(int)], np.eye(4)[x[:, 86].astype(int)]]
    )
    P = np.eye(len(x)) - F @ np.linalg.pinv(F)
    y = np.asarray(model.transformed_outcome(fresh.x, theta))
    e = (
        P @ y
        - x[:, 1]
        - x[:, 2:27] @ np.asarray(theta.A.array @ theta.A.array.T).ravel()
        - x[:, 32:42] @ np.asarray(theta.pi.array)
    )
    np.testing.assert_allclose(second, e[:, None] * x[:, 42:85], atol=1e-13, rtol=1e-13)
    assert np.max(np.abs(first - second)) > 0.1


def test_unknown_grid_roundtrip_is_not_reported_as_known_empty(tmp_path):
    cs = KConfidenceSet(
        grid=np.array([0.0, 0.1]),
        p_grid=np.array([np.nan, 0.8]),
        in_set=np.array([False, False]),
        intervals=[],
        topology="empty",
        open_left=False,
        open_right=False,
        alpha=0.05,
        statistic="K",
        n_invalid=1,
        invalid_indices=(0,),
        n_nonconverged=1,
        nonconverged_indices=(1,),
    )
    report = driver.persistence_check(cs, tmp_path / "law.npz")
    assert report["parity"] and report["capacity_queries_checked"] == 3
    assert report["point_verdicts"] == [None, None]
    assert not report["known_empty_scanned_set"]
    assert report["capacity_full_grid"].n_decidable == 0


def test_recording_optimizer_forwards_protocol_and_raw_flags(tmp_path):
    marker = object()
    theta = jnp.ones(2)
    info = OptimizerInfo(
        steps=2,
        final_objective=jnp.array(3.0),
        status="stagnated",
        backend="test",
        done=False,
        stagnated=True,
    )

    def fake(fn, initial, spec, *, args=None):
        assert initial is theta and spec is marker and args == (4,)
        assert fn(2) == 6
        return initial, info

    optimizer = driver.RecordingOptimizer(fake, tmp_path / "optimizer.jsonl")
    actual, raw = optimizer(lambda x: x * 3, theta, marker, args=(4,))
    assert actual is theta and raw is info
    assert optimizer.records[0]["optimizer_info"] is info
    assert "profiler_nuisance_valid" not in optimizer.records[0]
    with np.load(tmp_path / "point-000-nuisance.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["leaf_000"], theta)


def test_cli_synthetic_public_profile_and_codec(tmp_path):
    output = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, str(_PATH), "--synthetic-smoke", "--output-dir", str(output)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"]["kind"] == "synthetic-api-smoke-not-scientific"
    assert manifest["protocol"]["optimizer"]["max_steps"] == 2
    assert len(manifest["per_point"]) == 2
    assert manifest["persistence"]["parity"]
    assert manifest["persistence"]["capacity_queries_checked"] == 3
    assert (output / "profile-grid.npz").exists()
    assert len(list(output.glob("point-*-nuisance.npz"))) == 2
    cs = manifest["confidence_set"]
    for i, record in enumerate(manifest["per_point"]):
        assert record["profiler_nuisance_valid"] == (
            i not in cs["nonconverged_indices"]
        )
        assert "parameter_leaf_order" in record
