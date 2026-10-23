"""Opt-in numerical bread truncation and its separate sensitivity contract."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    NumericalRankPolicy,
    OptimizationResult,
    load_law,
    save_law,
)
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.manifolds import ManifoldLeaf, PSDFixedRank


@jdc.pytree_dataclass
class _Theta:
    a: jax.Array
    b: jax.Array


@jdc.pytree_dataclass
class _FactorTheta:
    A: ManifoldLeaf


def _result(
    first_gradient: float = 1e-6,
    *,
    weighting: jax.Array | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        theta_hat=_Theta(jnp.array(0.0), jnp.array(2.0)),
        objective_value=jnp.array(0.0),
        moment_jacobian=jnp.diag(jnp.array([first_gradient, 1.0])),
        weighting_matrix=jnp.eye(2) if weighting is None else weighting,
        moment_covariance=jnp.eye(2),
        n_overid=1,
        diagnostics=SimpleNamespace(  # type: ignore[arg-type]  # focused law fixture
            binding_ridge=False,
            gauge_nullspace_dim=0,
            tau_realised=0.0,
            kappa_V=1.0,
        ),
    )


def _factor_result(A: np.ndarray, G: np.ndarray) -> OptimizationResult:
    theta = _FactorTheta(ManifoldLeaf(jnp.asarray(A), PSDFixedRank(2, 2)))
    return OptimizationResult(
        theta_hat=theta,
        objective_value=jnp.array(0.0),
        moment_jacobian=jnp.asarray(G),
        weighting_matrix=jnp.eye(4),
        moment_covariance=jnp.eye(4),
        n_overid=1,
        diagnostics=SimpleNamespace(  # type: ignore[arg-type]  # focused law fixture
            binding_ridge=False,
            gauge_nullspace_dim=1,
            tau_realised=0.0,
            kappa_V=1.0,
        ),
        manifold_spec=manifold_spec_from_params(theta),
    )


def test_opt_in_truncates_weak_positive_bread_and_reports_sensitivity():
    result = _result()
    plain = result.asymptotic()
    law = result.asymptotic(rank_policy=NumericalRankPolicy(1e-10))
    np.testing.assert_allclose(law.cov(), np.diag([0.0, 1.0]), atol=1e-12)
    assert plain.rank_report is None
    assert plain.se()[0] > 1e5
    report = law.rank_report
    assert report is not None
    assert report.status == "ok"
    assert (report.effective_rank, report.nominal_gauge_dim, report.extra_excluded) == (
        1,
        0,
        1,
    )
    np.testing.assert_array_equal(report.retained_mask, [False, True])
    np.testing.assert_allclose(report.excluded_projector, np.diag([1.0, 0.0]))
    assert report.signed_eigenvalues[0] > 0
    assert report.cutoff == pytest.approx(1e-10)

    linear = law.functional_sensitivity(lambda c: c[0])
    np.testing.assert_allclose(linear.first_order_norm, [1.0])
    np.testing.assert_allclose(linear.second_order_norm, [0.0])
    quadratic = law.functional_sensitivity(lambda c: c[0] ** 2)
    np.testing.assert_allclose(quadratic.first_order_norm, [0.0])
    np.testing.assert_allclose(quadratic.second_order_norm, [2.0])
    np.testing.assert_allclose(law.functional_se(lambda c: c[0] ** 2)[0], [0.0])
    assert quadratic.higher_orders_unassessed
    cubic = law.functional_sensitivity(lambda c: c[0] ** 3)
    np.testing.assert_allclose(cubic.first_order_norm, [0.0])
    np.testing.assert_allclose(cubic.second_order_norm, [0.0])

    table = law.coef_table
    assert table.loc["theta_0", "rank_policy"] == "retained_subspace"
    assert table.loc["theta_0", "excluded_loading"] == pytest.approx(1.0)
    assert np.isnan(table.loc["theta_0", "t_stat"])
    assert np.isnan(table.loc["theta_0", "p_value"])
    assert np.isfinite(table.loc["theta_1", "t_stat"])


def test_rotated_excluded_direction_flags_both_raw_coordinates():
    weak = np.array([1.0, 1.0]) / np.sqrt(2.0)
    strong = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    basis = np.column_stack((weak, strong))
    result = replace(
        _result(),
        moment_jacobian=jnp.asarray(np.diag([1e-6, 1.0]) @ basis.T),
    )
    default = result.asymptotic()
    law = result.asymptotic(rank_policy=NumericalRankPolicy(1e-10))
    report = law.rank_report
    assert report is not None
    np.testing.assert_allclose(report.excluded_projector, np.outer(weak, weak))
    np.testing.assert_allclose(law.cov(), np.outer(strong, strong), atol=1e-12)
    conditional_se, _ = law.functional_se(lambda c: c[0] + c[1])
    np.testing.assert_allclose(conditional_se, [0.0], atol=1e-7)
    sensitivity = law.functional_sensitivity(lambda c: c[0] + c[1])
    np.testing.assert_allclose(sensitivity.first_order_norm, [np.sqrt(2.0)])
    table = law.coef_table
    np.testing.assert_allclose(table["excluded_loading"], [1 / np.sqrt(2)] * 2)
    assert np.isnan(table["t_stat"]).all()
    assert np.isnan(table["p_value"]).all()
    assert law.j_test().n_overid == default.j_test().n_overid == result.n_overid
    assert law.j_test().J_stat == default.j_test().J_stat


def test_sensitivity_includes_mixed_curvature_and_singular_psd_geometry():
    zero_result = replace(_result(), theta_hat=_Theta(jnp.array(0.0), jnp.array(0.0)))
    law = zero_result.asymptotic(rank_policy=NumericalRankPolicy(1e-10))
    mixed = law.functional_sensitivity(lambda c: c[0] * c[1])
    np.testing.assert_allclose(mixed.first_order_norm, [0.0])
    np.testing.assert_allclose(mixed.second_order_norm, [1.0])

    psd_result = _factor_result(np.zeros((2, 2)), np.diag([0.0, 0.0, 1.0, 1.0]))
    psd_law = psd_result.asymptotic(rank_policy=NumericalRankPolicy(1e-10))
    report = psd_law.rank_report
    assert report is not None
    assert report.geometry_status == "singular"
    assert report.status == "ok"
    assert report.nominal_gauge_dim == 1
    assert report.extra_excluded == 1
    assert report.factor_geometries[0].singular_values == (0.0, 0.0)
    gamma = psd_law.functional_sensitivity(lambda c: jnp.reshape(c[0] @ c[0].T, (-1,)))
    np.testing.assert_allclose(gamma.first_order_norm, np.zeros(4))
    assert np.any(gamma.second_order_norm > 0.0)


def test_tied_zeros_boundary_and_full_rank_psd_gauge_rotation(tmp_path):
    gauge = np.array([0.0, -1.0, 1.0, 0.0]) / np.sqrt(2.0)
    weak = np.array([1.0, 0.0, 0.0, 0.0])
    other = np.array([0.0, 0.0, 0.0, 1.0])
    symmetric = np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0)
    basis = np.column_stack((gauge, weak, other, symmetric))
    A = np.eye(2)
    angle = np.pi / 5.0
    Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotation = np.kron(np.eye(2), Q.T)
    policy = NumericalRankPolicy(1e-10)

    # Two exact zeros remain a subspace even though the eigensolver may rotate
    # their individual vectors; the nominal gauge count is only one.
    tied_G = np.diag([0.0, 0.0, 1.0, np.sqrt(2.0)]) @ basis.T
    tied = _factor_result(A, tied_G).asymptotic(rank_policy=policy)
    rotated = _factor_result(A @ Q, tied_G @ rotation.T).asymptotic(rank_policy=policy)
    first_report, second_report = tied.rank_report, rotated.rank_report
    assert first_report is not None and second_report is not None
    assert first_report.geometry_status == second_report.geometry_status == "regular"
    assert first_report.nominal_gauge_dim == second_report.nominal_gauge_dim == 1
    assert first_report.extra_excluded == second_report.extra_excluded == 1
    np.testing.assert_allclose(
        second_report.excluded_projector,
        rotation @ first_report.excluded_projector @ rotation.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(rotated.cov(), rotation @ tied.cov() @ rotation.T)

    def gamma(c):
        return jnp.reshape(c[0] @ c[0].T, (-1,))

    np.testing.assert_allclose(
        tied.functional_se(gamma)[0], rotated.functional_se(gamma)[0], atol=1e-8
    )
    np.testing.assert_allclose(
        tied.functional_sensitivity(gamma).first_order_norm,
        rotated.functional_sensitivity(gamma).first_order_norm,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tied.functional_sensitivity(gamma).second_order_norm,
        rotated.functional_sensitivity(gamma).second_order_norm,
        atol=1e-12,
    )
    path = tmp_path / "full-rank-psd.npz"
    save_law(tied, path)
    loaded = load_law(path)
    np.testing.assert_array_equal(
        loaded.leaf("A").se("gamma"), tied.leaf("A").se("gamma")
    )
    assert loaded.rank_report is not None
    assert loaded.rank_report.geometry_status == "regular"

    # A retained/excluded split almost tied at the policy boundary is marked
    # ambiguous; it is still a retained-subspace computation.
    near = np.array([0.0, 1e-10 * (1 - 1e-8), 1e-10 * (1 + 1e-8), 1.0])
    near_G = np.diag(np.sqrt(near)) @ basis.T
    boundary = _factor_result(A, near_G).asymptotic(rank_policy=policy)
    assert boundary.rank_report is not None
    assert boundary.rank_report.status == "ok"
    assert boundary.rank_report.projector_boundary_ambiguous
    assert boundary.rank_report.extra_excluded == 1


def test_invalid_bread_statuses_are_loud():
    policy = NumericalRankPolicy(1e-10)
    for result, status in (
        (_result(0.0, weighting=jnp.zeros((2, 2))), "all_zero_bread"),
        (
            _result(weighting=jnp.diag(jnp.array([-1.0, 1.0]))),
            "materially_negative_bread",
        ),
    ):
        law = result.asymptotic(rank_policy=policy)
        assert law.rank_report is not None
        assert law.rank_report.status == status
        assert np.isnan(law.cov()).all()
        assert np.isnan(law.se()).all()
        with pytest.raises(ValueError, match=status):
            law.quantile(0.95)
        with pytest.raises(ValueError, match=status):
            law.sample(jax.random.PRNGKey(0), 2)


def test_policy_and_projector_survive_load_and_resave(tmp_path):
    law = _result().asymptotic(rank_policy=NumericalRankPolicy(1e-10))
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_law(law, first)
    loaded = load_law(first)
    save_law(loaded, second)
    reloaded = load_law(second)
    for other in (loaded, reloaded):
        assert other.rank_policy == law.rank_policy
        assert other.rank_report is not None
        np.testing.assert_array_equal(other.cov(), law.cov())
        np.testing.assert_array_equal(
            other.rank_report.excluded_projector,
            law.rank_report.excluded_projector,
        )
        np.testing.assert_array_equal(
            other.functional_sensitivity(lambda c: c[0] ** 2).second_order_norm,
            law.functional_sensitivity(lambda c: c[0] ** 2).second_order_norm,
        )
        assert np.isnan(other.coef_table.loc["theta_0", "p_value"])
        with pytest.raises(RuntimeError, match="after reload"):
            other.with_rank_policy(NumericalRankPolicy(1e-8))

    with np.load(first, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["rank_excluded_projector"] = np.diag([0.2, 0.0])
    np.savez(tmp_path / "tampered.npz", **arrays)
    with pytest.raises(ValueError, match="projector must be idempotent"):
        load_law(tmp_path / "tampered.npz")

    arrays["rank_excluded_projector"] = law.rank_report.excluded_projector
    manifest = json.loads(str(arrays["__manifest__"]))
    manifest["component_shapes"] = [[2]]
    arrays["__manifest__"] = np.asarray(json.dumps(manifest))
    np.savez(tmp_path / "wrong-shapes.npz", **arrays)
    with pytest.raises(ValueError, match="rank labels or leaf shapes disagree"):
        load_law(tmp_path / "wrong-shapes.npz")


def test_schema_one_law_loads_with_count_only_behavior(tmp_path):
    law = _result().asymptotic()
    path = tmp_path / "legacy.npz"
    save_law(law, path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    manifest = json.loads(str(arrays["__manifest__"]))
    manifest["schema_version"] = 1
    manifest.pop("numerical_rank", None)
    arrays["__manifest__"] = np.asarray(json.dumps(manifest))
    np.savez(path, **arrays)
    loaded = load_law(path)
    assert loaded.rank_policy is None
    assert loaded.rank_report is None
    np.testing.assert_array_equal(loaded.cov(), law.cov())
