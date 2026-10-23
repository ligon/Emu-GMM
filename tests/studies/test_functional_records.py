"""Fit-time functionals retain joint-covariance SEs and the fit's events."""

from __future__ import annotations

import dataclasses

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import OptimizationResult
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.studies import bias_sd, coverage, fit_record, replicate
from emu_gmm.types import Diagnostics, OptimizerInfo
from emu_gmm.weighting import ContinuouslyUpdated


@jdc.pytree_dataclass
class _Location:
    a: float
    b: float


def _diagnostics(m):
    moments = ha.Axis("Moments", m)
    return Diagnostics(
        tau_realised=jnp.asarray(0.0),
        kappa_V=jnp.asarray(1.0),
        binding_ridge=False,
        cholesky_pivot_min=jnp.asarray(1.0),
        final_objective=jnp.asarray(1.5),
        final_gradient_norm=jnp.asarray(0.0),
        N_j=ha.named(jnp.full(m, 100.0), moments),
        moment_residual=ha.named(jnp.zeros(m), moments),
        optimizer_info=OptimizerInfo(
            steps=1, status="converged", final_objective=0.75, backend="fixture"
        ),
    )


@pytest.fixture
def result():
    # This directly constructible result isolates the inference projection.
    # G selects the first two moments, so Sigma is V's upper-left block.
    return OptimizationResult(
        theta_hat=_Location(a=1.0, b=2.0),
        objective_value=jnp.asarray(1.5),
        moment_jacobian=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        weighting_matrix=jnp.eye(3),
        moment_covariance=jnp.array([[4.0, 1.5, 0.0], [1.5, 9.0, 0.0], [0, 0, 1]]),
        n_overid=1,
        diagnostics=_diagnostics(3),
        weighting=ContinuouslyUpdated(),
    )


def _sum(components):
    return components[0] + components[1]


class TestFunctionalRecords:
    def test_scalar_se_uses_cross_covariance(self, result):
        record = fit_record(result, f=_sum, names=("sum",))
        np.testing.assert_array_equal(record.theta_flat, [3.0])
        # Var(a+b) = 4 + 9 + 2*1.5 = 16; marginal SEs alone give 13.
        np.testing.assert_allclose(record.se, [4.0], rtol=1e-14)
        np.testing.assert_array_equal(
            record.se, result.asymptotic().functional_se(_sum)[0]
        )
        assert record.param_names == ("sum",)

    def test_vector_functional_and_default_names(self, result):
        def functional(components):
            a, b = components
            return jnp.stack([a + b, a * b])

        record = fit_record(result, f=functional)
        np.testing.assert_array_equal(record.theta_flat, [3.0, 2.0])
        np.testing.assert_array_equal(
            record.se, result.asymptotic().functional_se(functional)[0]
        )
        assert record.param_names == ("f_0", "f_1")
        assert fit_record(result, f=_sum).param_names == ("f_0",)

    def test_original_j_and_events_are_preserved(self, result):
        flagged = dataclasses.replace(
            result,
            converged=False,
            diagnostics=dataclasses.replace(
                result.diagnostics,
                tau_realised=jnp.asarray(0.5),
                binding_ridge=True,
                sigma_meat_indefinite=True,
                tau_saturated=True,
            ),
        )
        original = fit_record(flagged)
        functional = fit_record(flagged, f=_sum)
        for field in (
            "J_stat",
            "J_pvalue",
            "J_pvalue_adjusted",
            "converged",
            "tau_realised",
            "binding_ridge",
            "sigma_meat_indefinite",
            "tau_saturated",
        ):
            np.testing.assert_array_equal(
                getattr(functional, field), getattr(original, field)
            )
        assert functional.J_dof == original.J_dof == 1

    @pytest.mark.parametrize(
        "names, message",
        [(("only_one",), "names length"), (("x", "x"), "distinct")],
    )
    def test_bad_names_raise(self, result, names, message):
        with pytest.raises(ValueError, match=message):
            fit_record(result, f=lambda c: jnp.stack(c), names=names)

    def test_non_string_name_raises(self, result):
        with pytest.raises(TypeError, match="only strings"):
            fit_record(result, f=_sum, names=(1,))

    def test_matrix_output_requires_explicit_flattening(self, result):
        with pytest.raises(ValueError, match="scalar or a 1-D array"):
            fit_record(result, f=lambda c: jnp.stack(c)[:, None])

    def test_replication_and_existing_summaries_use_functional_truth(self, result):
        key = jax.random.PRNGKey(4)

        def dgp(rep_key):
            return {"shift": jax.random.normal(rep_key, ())}

        def run(theta_init, dataset):
            return dataclasses.replace(
                result,
                theta_hat=_Location(a=1.0 + dataset["shift"], b=2.0),
            )

        records = replicate(
            run,
            dgp,
            n_reps=4,
            key=key,
            theta_init=None,
            record=lambda fit, dataset: fit_record(fit, f=_sum, names=("sum",)),
        )
        shifts = np.array(
            [float(dgp(jax.random.fold_in(key, r))["shift"]) for r in range(4)]
        )
        np.testing.assert_array_equal(records.records.theta_flat[:, 0], 3.0 + shifts)
        summary = bias_sd(records, theta0=jnp.array([3.0]))
        np.testing.assert_allclose(summary.bias, [shifts.mean()], atol=1e-15)
        np.testing.assert_allclose(summary.mc_sd, [shifts.std(ddof=1)], rtol=1e-14)
        np.testing.assert_allclose(summary.mean_se, [4.0], rtol=1e-14)
        covered = coverage(records, theta0=jnp.array([3.0]), level=0.95)
        expected = np.mean(np.abs(shifts) <= 1.959963984540054 * 4.0)
        np.testing.assert_array_equal(covered.coverage, [expected])
        assert summary.param_names == covered.param_names == ("sum",)

    def test_nan_se_and_nonconvergence_remain_counted(self, result):
        indefinite = dataclasses.replace(
            result,
            moment_covariance=-result.moment_covariance,
            diagnostics=dataclasses.replace(
                result.diagnostics, sigma_meat_indefinite=True
            ),
        )
        nonconverged = dataclasses.replace(result, converged=False)
        records = [
            fit_record(fit, f=_sum) for fit in (result, indefinite, nonconverged)
        ]
        assert np.isnan(records[1].se[0])
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)
        covered = coverage(stacked, theta0=jnp.array([3.0]))
        summary = bias_sd(stacked, theta0=jnp.array([3.0]))
        for value in (covered, summary):
            assert value.n_used == 2
            assert value.n_excluded == 1
            np.testing.assert_array_equal(value.n_valid_se, [1])
        np.testing.assert_array_equal(covered.coverage, [1.0])
        np.testing.assert_allclose(summary.mean_se, [4.0], rtol=1e-14)


@jdc.pytree_dataclass
class _ProductParams:
    A: ManifoldLeaf
    phi: ManifoldLeaf


def _psd_result(A):
    params = _ProductParams(
        A=ManifoldLeaf(A, PSDFixedRank(3, 2)),
        phi=ManifoldLeaf(jnp.array([0.7]), Euclidean(1)),
    )
    rows, columns = jnp.triu_indices(3)

    def moments(flat):
        factor = flat[:6].reshape(3, 2)
        gamma = factor @ factor.T
        return jnp.concatenate([gamma[rows, columns], flat[6:]])

    flat = jnp.concatenate([A.ravel(), jnp.array([0.7])])
    return OptimizationResult(
        theta_hat=params,
        objective_value=jnp.asarray(1.5),
        moment_jacobian=jax.jacrev(moments)(flat),
        weighting_matrix=jnp.eye(7),
        moment_covariance=0.01 * jnp.eye(7),
        n_overid=1,
        diagnostics=_diagnostics(7),
        weighting=ContinuouslyUpdated(),
        manifold_spec=manifold_spec_from_params(params),
    )


def test_psd_functional_records_are_gauge_invariant():
    A = jnp.array([[2.0, 0.0], [0.0, 1.0], [0.3, -0.2]])
    angle = 0.7
    Q = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])

    def functional(components):
        factor, phi = components
        gamma = factor @ factor.T
        # Distinct nonzero eigenvalues are smooth at this interior point.
        return jnp.concatenate([jnp.linalg.eigvalsh(gamma)[-2:], gamma[0, :2], phi])

    original = _psd_result(A)
    rotated = _psd_result(A @ Q)
    left = fit_record(original, f=functional)
    right = fit_record(rotated, f=functional)
    np.testing.assert_allclose(
        left.theta_flat, right.theta_flat, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(left.se, right.se, rtol=1e-11, atol=1e-14)
    assert np.all(np.asarray(left.se) > 0)
    np.testing.assert_array_equal(
        left.se, original.asymptotic().functional_se(functional)[0]
    )
    assert left.param_names == tuple(f"f_{i}" for i in range(5))
