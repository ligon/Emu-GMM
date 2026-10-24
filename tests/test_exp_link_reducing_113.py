"""The exp-link two-way-FE consumer shape through the public native factory."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import EmpiricalMeasure, IIDCovariance, build_estimator, estimate
from emu_gmm.inference import k_statistic
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.riemannian_tr import riemannian_tr

_TRIU = np.triu_indices(3)
_A = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])


@jdc.pytree_dataclass
class _Point:
    A: ManifoldLeaf
    c: ManifoldLeaf


def _point():
    return _Point(
        ManifoldLeaf(jnp.asarray(1.01 * _A), PSDFixedRank(3, 2)),
        ManifoldLeaf(jnp.array([0.15]), Euclidean(1)),
    )


def _panel(seed):
    rng = np.random.default_rng(1133)
    n = 120
    product = rng.integers(0, 6, n)
    market = rng.integers(0, 5, n)
    F = np.column_stack([np.eye(6)[product], np.eye(5)[market]])
    residualizer = np.eye(n) - F @ np.linalg.pinv(F)
    X = rng.normal(size=(n, 6))
    zeta = rng.normal(size=n)
    structural = F @ rng.normal(size=11) + X @ (_A @ _A.T)[_TRIU]
    instruments = residualizer @ np.column_stack([X, zeta * structural])
    error = np.random.default_rng(seed).normal(size=n) * 0.1
    Y = np.exp(0.2 * zeta) * (structural + error)
    rows = np.column_stack(
        [Y, zeta, residualizer @ X, instruments, product, market, np.arange(n)]
    )
    return EmpiricalMeasure.from_arrays(jnp.asarray(rows), M=7), jnp.asarray(
        residualizer
    )


class _ExpWithin:
    def __init__(self):
        self.aux_calls = 0
        self.psi_calls = 0

    def aux(self, measure, theta):
        self.aux_calls += 1
        x = measure.x
        F = jnp.concatenate(
            [
                jax.nn.one_hot(x[:, 15].astype(int), 6),
                jax.nn.one_hot(x[:, 16].astype(int), 5),
            ],
            axis=1,
        )
        outcome = jnp.exp(-theta.c.array[0] * x[:, 1]) * x[:, 0]
        # The present consumer population is the unweighted retained panel.
        return jnp.linalg.pinv(F.T @ F) @ (F.T @ outcome)

    def psi(self, row, theta, aux):
        self.psi_calls += 1
        f = jnp.concatenate(
            [
                jax.nn.one_hot(row[15].astype(int), 6),
                jax.nn.one_hot(row[16].astype(int), 5),
            ]
        )
        outcome = jnp.exp(-theta.c.array[0] * row[1]) * row[0] - f @ aux
        gamma = theta.A.array @ theta.A.array.T
        return (outcome - row[2:8] @ gamma[_TRIU]) * row[8:15]


@pytest.mark.parametrize(
    "optimizer", [riemannian_lm(max_steps=200), riemannian_tr(max_steps=200)]
)
def test_exp_link_projected_native_fit_k_and_fresh_panel_reuse(
    optimizer, lm_advisory_trace_counts
):
    first_mu, residualizer = _panel(1134)
    second_mu, _ = _panel(1135)
    model = _ExpWithin()
    observed = lm_advisory_trace_counts(lambda: (model.aux_calls, model.psi_calls))
    run = build_estimator(
        model,
        measure=first_mu,
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=optimizer,
    )
    first = run(_point(), first_mu)
    assert bool(first.converged)
    calls, _ = observed()
    second = run(_point(), second_mu)
    assert bool(second.converged)
    assert observed()[0] == calls
    assert observed()[1] <= 1
    assert not np.allclose(first.theta_hat.c.array, second.theta_hat.c.array, atol=1e-5)

    def closure_oracle(row, theta):
        outcome = jnp.exp(-theta.c.array[0] * first_mu.x[:, 1]) * first_mu.x[:, 0]
        within = residualizer @ outcome
        gamma = theta.A.array @ theta.A.array.T
        return (within[row[17].astype(int)] - row[2:8] @ gamma[_TRIU]) * row[8:15]

    reference = estimate(
        closure_oracle,
        first_mu,
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=optimizer,
    )
    assert bool(reference.converged)
    np.testing.assert_allclose(
        first.theta_hat.A.array @ first.theta_hat.A.array.T,
        reference.theta_hat.A.array @ reference.theta_hat.A.array.T,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        first.theta_hat.c.array, reference.theta_hat.c.array, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        first.objective_value, reference.objective_value, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        first_mu.jacobian_contributions(model, _point()),
        first_mu.jacobian_contributions(closure_oracle, _point()),
        atol=1e-12,
    )
    actual_k = k_statistic(_point(), first_mu, IIDCovariance(), model)
    oracle_k = k_statistic(_point(), first_mu, IIDCovariance(), closure_oracle)
    np.testing.assert_allclose(actual_k.K, oracle_k.K, rtol=1e-8, atol=1e-10)
