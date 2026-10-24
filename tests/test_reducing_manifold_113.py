"""Combined reducing-model AD and native manifold reuse (ledger Section 26)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import EmpiricalMeasure, IIDCovariance, build_estimator
from emu_gmm._internal.params import flatten_params_for_ad, unflatten_params
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.riemannian_tr import riemannian_tr

_A = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
_TRIU = np.triu_indices(3)


@jdc.pytree_dataclass
class Point:
    factor: ManifoldLeaf
    level: ManifoldLeaf


def _point():
    return Point(
        ManifoldLeaf(jnp.asarray(1.01 * _A), PSDFixedRank(3, 2)),
        ManifoldLeaf(jnp.array([1.8]), Euclidean(1)),
    )


def _measure(*, masses=0, outcome_shift=0.0):
    # Antithetic noise and pair-constant exposure retain known moment roots
    # while allowing unequal moment masks and changed within-group masses.
    pair = np.repeat(np.arange(16), 2)
    group = pair // 8
    sign = np.tile([1.0, -1.0], 16)
    noise = np.random.default_rng(113).normal(size=(16, 7))[pair] * sign[:, None]
    target = np.r_[(_A @ _A.T)[_TRIU], 2.0]
    y = target + 0.03 * noise
    y[:, 6] += outcome_shift
    z = 0.3 + 0.7 * group + 0.08 * (pair % 8) + 0.03 * sign
    mask = np.ones((32, 7))
    for j in range(7):
        mask[(pair + j) % 4 == 3, j] = 0.0
    weights = 1.0 + 0.1 * (pair % 3)
    if masses:
        weights *= 0.7 + 0.15 * (pair % 8) + 0.4 * group
    return EmpiricalMeasure(
        jnp.asarray(np.column_stack([y, z, group])),
        jnp.asarray(mask),
        jnp.asarray(weights),
    )


def _group_means_numpy(measure):
    x = np.asarray(measure.x)
    group = x[:, 8].astype(int)
    mass = np.asarray(measure.weights) * np.asarray(measure.mask)[:, 6]
    den = np.bincount(group, weights=mass, minlength=2)
    mu = np.bincount(group, weights=mass * x[:, 7], minlength=2) / den
    return mu, mass.sum(), np.dot(mass, x[:, 7]) / mass.sum()


class Reduction:
    def __init__(self):
        self.aux_calls = 0
        self.psi_calls = 0

    def aux(self, measure, theta):
        self.aux_calls += 1
        group = measure.x[:, 8].astype(jnp.int32)
        mass = measure.weights * measure.mask[:, 6]
        den = jax.ops.segment_sum(mass, group, num_segments=2)
        numer = jax.ops.segment_sum(mass * measure.x[:, 7], group, num_segments=2)
        return theta.level.array[0] * numer / den

    def psi(self, row, theta, aux):
        self.psi_calls += 1
        gamma = theta.factor.array @ theta.factor.array.T
        level = theta.level.array[0] * (1.0 + 0.2 * row[7])
        level -= 0.1 * aux[row[8].astype(jnp.int32)]
        return jnp.r_[gamma[_TRIU], level] - row[:7]

    def __call__(self, row, theta):
        raise AssertionError("the complete reducing contract takes precedence")


def test_complete_ambient_row_derivatives_include_aux_and_gauge_nullspace():
    measure = _measure(masses=1)
    point = _point()
    model = Reduction()
    D = np.asarray(measure.jacobian_contributions(model, point))
    mass = np.asarray(measure.mask) * np.asarray(measure.weights)[:, None]
    A = np.asarray(point.factor.array)
    expected = np.zeros((32, 7, 7))
    # Independent analytic derivative of Gamma_ab = sum_c A_ac A_bc.
    for j, (a, b) in enumerate(zip(*_TRIU, strict=True)):
        derivative = np.zeros((3, 2))
        derivative[a, :] += A[b, :]
        derivative[b, :] += A[a, :]
        expected[:, j, :6] = mass[:, j, None] * derivative.ravel()
    mu, _, _ = _group_means_numpy(measure)
    x = np.asarray(measure.x)
    expected[:, 6, 6] = mass[:, 6] * (
        1.0 + 0.2 * x[:, 7] - 0.1 * mu[x[:, 8].astype(int)]
    )
    np.testing.assert_allclose(D, expected, rtol=1e-13, atol=1e-13)
    incomplete = mass[:, 6] * (1.0 + 0.2 * x[:, 7])
    assert np.max(abs(D[:, 6, 6] - incomplete)) > 0.05
    flat, tree, spec = flatten_params_for_ad(point)
    G = jax.jacfwd(
        lambda value: measure.expectation(model, unflatten_params(value, tree, spec))
    )(flat)
    np.testing.assert_allclose(
        np.asarray(measure.jacobian(model, point)),
        expected.sum(axis=0) / mass.sum(axis=0)[:, None],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        G, expected.sum(axis=0) / mass.sum(axis=0)[:, None], atol=1e-12
    )
    vertical = np.r_[(A @ np.array([[0.0, -1.0], [1.0, 0.0]])).ravel(), 0.0]
    np.testing.assert_allclose(D @ vertical, 0.0, atol=1e-13)


@pytest.mark.parametrize(
    "optimizer", [riemannian_lm(max_steps=100), riemannian_tr(max_steps=100)]
)
def test_aux_native_factory_reuses_fresh_outcomes_and_group_masses(
    optimizer, lm_advisory_trace_counts
):
    model = Reduction()
    observed = lm_advisory_trace_counts(lambda: (model.aux_calls, model.psi_calls))
    original = _measure()
    changed_mass = _measure(masses=1)
    changed_y = _measure(masses=1, outcome_shift=0.02)
    old_mu, old_mass, _ = _group_means_numpy(original)
    new_mu, new_mass, _ = _group_means_numpy(changed_mass)
    assert not np.array_equal(old_mu, new_mu)
    assert old_mass != new_mass
    run = build_estimator(
        model,
        measure=original,
        covariance=IIDCovariance(),
        parameters=_point(),
        optimizer=optimizer,
    )
    first = run(_point(), original)
    assert bool(first.converged)
    calls, _ = observed()
    second = run(_point(), changed_mass)
    assert bool(second.converged)
    assert observed()[0] == calls
    assert observed()[1] <= 1
    _, _, old_zmean = _group_means_numpy(original)
    _, _, new_zmean = _group_means_numpy(changed_mass)
    # The first six means have an exact feasible Gamma root in both worlds.
    # The final root changes solely because the auxiliary population changes.
    np.testing.assert_allclose(
        first.theta_hat.level.array, [2.0 / (1.0 + 0.1 * old_zmean)], atol=1e-7
    )
    np.testing.assert_allclose(
        second.theta_hat.level.array, [2.0 / (1.0 + 0.1 * new_zmean)], atol=1e-7
    )
    assert (
        abs(float(first.theta_hat.level.array[0] - second.theta_hat.level.array[0]))
        > 1e-3
    )
    np.testing.assert_allclose(
        second.theta_hat.factor.array @ second.theta_hat.factor.array.T,
        _A @ _A.T,
        atol=1e-7,
    )
    third = run(_point(), changed_y)
    assert bool(third.converged)
    assert observed()[0] == calls
    assert observed()[1] <= 1
    assert not np.allclose(
        third.theta_hat.level.array, second.theta_hat.level.array, atol=1e-3
    )
