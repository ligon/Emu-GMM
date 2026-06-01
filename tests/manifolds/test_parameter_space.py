r"""Acceptance tests for the ergonomic :class:`ParameterSpace` layer (#107).

This is the END-TO-END proof of the additive sugar built on #107:

* a user declares field -> manifold ONCE in a ``ParameterSpace`` subclass via
  the :func:`on` descriptor;
* ``Space.point()`` / ``Space.point(seed)`` materialise a bound instance that
  is an ordinary ``ManifoldLeaf`` PyTree (a valid ``theta_init``);
* ``estimate(..., parameters=...)`` is polymorphic over a ``ParameterSpace``
  class, a bound instance, or any existing theta PyTree (warm start);
* ``theta_init=`` keeps working as a deprecated, bitwise-identical alias.

These tests add NO src math: they CALL the validated manifold core
(``ManifoldLeaf`` / ``manifold_spec_from_params`` / ``riemannian_lm`` / the
inference block) through the ordinary :func:`emu_gmm.estimate` entry point.

The worked DGP is an exactly-identified multivariate-normal method-of-moments
fit: the moment vector stacks the K mean moments and the K(K+1)/2 lower-
triangular second-central-moment moments, so the estimator recovers the
sample mean (in ``mu``) and the sample covariance (in ``Gamma = A @ A.T``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    Euclidean,
    ManifoldLeaf,
    ParameterSpace,
    PSDFixedRank,
    estimate,
    on,
)
from emu_gmm.covariance import IIDCovariance
from emu_gmm.measures import EmpiricalMeasure

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared MVN method-of-moments DGP.
# ---------------------------------------------------------------------------
def _tril_idx(K: int) -> jnp.ndarray:
    """Row-major lower-triangular (i, j) index pairs, length K(K+1)/2."""
    return jnp.array(np.tril_indices(K)).T


def _moment_count(K: int) -> int:
    return K + K * (K + 1) // 2


def _make_model(K: int):
    """psi(x, theta) = [x - mu ; vech((x-mu)(x-mu)^T) - vech(A A^T)].

    Exactly identified: the K mean moments pin ``mu`` to the sample mean and
    the K(K+1)/2 second-central-moment moments pin ``Gamma = A A^T`` to the
    sample covariance taken about ``mu``.
    """
    tril = _tril_idx(K)
    ii, jj = tril[:, 0], tril[:, 1]

    def model(x, theta):
        mu = theta.mu.array
        A = theta.A.array
        Gamma = A @ A.T
        d = x - mu
        outer = jnp.outer(d, d)
        second = outer[ii, jj] - Gamma[ii, jj]
        return jnp.concatenate([d, second])

    return model


def _make_data(K: int, *, n: int, seed: int):
    rng = np.random.default_rng(seed)
    mu_true = rng.normal(size=(K,))
    L = rng.normal(size=(K, K))
    Sigma_true = L @ L.T + K * np.eye(K)  # well-conditioned PD
    X = rng.multivariate_normal(mu_true, Sigma_true, size=n)
    return jnp.asarray(X), jnp.asarray(mu_true), jnp.asarray(Sigma_true)


def _sample_moments(X: jnp.ndarray):
    """Sample mean and (about-the-sample-mean) sample covariance (1/N)."""
    mean = jnp.mean(X, axis=0)
    d = X - mean
    cov = (d.T @ d) / X.shape[0]
    return mean, cov


# ---------------------------------------------------------------------------
# Test 1 -- MVN end-to-end via ParameterSpace recovers mean + covariance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [2, 3])
def test_mvn_recovery_via_parameter_space_class(K):
    X, _mu_true, _Sig_true = _make_data(K, n=4000, seed=100 + K)
    M = _moment_count(K)
    model = _make_model(K)
    mean_hat, cov_hat = _sample_moments(X)

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K), default=jnp.linalg.cholesky(cov_hat))
        mu: jax.Array = on(Euclidean(K), default=mean_hat)

    result = estimate(
        model,
        EmpiricalMeasure.from_arrays(X, M=M),
        covariance=IIDCovariance(),
        parameters=Normal,
    )
    assert bool(result.converged)

    A_hat, mu_hat = result.components()
    Gamma_hat = A_hat @ A_hat.T

    # Recovers the SAMPLE mean and SAMPLE covariance (the MoM targets).
    assert bool(jnp.allclose(mu_hat, mean_hat, atol=1e-6))
    assert bool(jnp.allclose(Gamma_hat, cov_hat, atol=1e-6))


# ---------------------------------------------------------------------------
# Test 2 -- .point() determinism; .point(seed) random + valid on-manifold.
# ---------------------------------------------------------------------------
def test_point_default_is_deterministic_and_equals_declared_defaults():
    K = 3
    A0 = jnp.eye(K) * 1.3
    mu0 = jnp.arange(K, dtype=jnp.float64)

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K), default=A0)
        mu: jax.Array = on(Euclidean(K), default=mu0)

    p1 = Normal.point()
    p2 = Normal.point()
    # Deterministic and equal to the declared defaults.
    assert bool(jnp.allclose(p1.A.array, A0))
    assert bool(jnp.allclose(p1.mu.array, mu0))
    assert bool(jnp.allclose(p1.A.array, p2.A.array))
    assert bool(jnp.allclose(p1.mu.array, p2.mu.array))
    # The bound instance carries the declared manifolds.
    assert isinstance(p1.A, ManifoldLeaf)
    assert p1.A.manifold == PSDFixedRank(K, K)
    assert p1.mu.manifold == Euclidean(K)


def test_point_no_default_requires_seed():
    K = 2

    class Bare(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K))  # no default
        mu: jax.Array = on(Euclidean(K), default=jnp.zeros(K))

    with pytest.raises(ValueError, match="no default"):
        Bare.point()
    # A seed is fine (random_point needs no default).
    p = Bare.point(0)
    assert p.A.array.shape == (K, K)


def test_point_seed_is_random_valid_and_seed_deterministic():
    K = 3

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K), default=jnp.eye(K))
        mu: jax.Array = on(Euclidean(K), default=jnp.zeros(K))

    p_a = Normal.point(0)
    p_b = Normal.point(1)
    p_a2 = Normal.point(0)

    # Correct shapes / on-manifold validity (PSDFixedRank: any (K,K) real).
    assert p_a.A.array.shape == (K, K) and p_a.mu.array.shape == (K,)
    assert bool(jnp.all(jnp.isfinite(p_a.A.array)))
    # Differs from the deterministic default and across seeds.
    assert not bool(jnp.allclose(p_a.A.array, jnp.eye(K)))
    assert not bool(jnp.allclose(p_a.A.array, p_b.A.array))
    # Same seed -> identical raw Y (determinism).
    assert bool(jnp.allclose(p_a.A.array, p_a2.A.array))


def test_two_random_seeds_recover_same_gauge_invariant_gamma():
    """Two seeds give different raw Y but, after a full fit, the SAME
    gauge-invariant Gamma_hat (random multi-start = free gauge-robustness)."""
    K = 2
    X, _mu, _Sig = _make_data(K, n=4000, seed=222)
    M = _moment_count(K)
    model = _make_model(K)
    _mean_hat, cov_hat = _sample_moments(X)

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K))
        mu: jax.Array = on(Euclidean(K))

    p1 = Normal.point(1)
    p2 = Normal.point(2)
    # Different raw starting Y.
    assert not bool(jnp.allclose(p1.A.array, p2.A.array))

    r1 = estimate(
        model,
        EmpiricalMeasure.from_arrays(X, M=M),
        covariance=IIDCovariance(),
        parameters=p1,
    )
    r2 = estimate(
        model,
        EmpiricalMeasure.from_arrays(X, M=M),
        covariance=IIDCovariance(),
        parameters=p2,
    )
    assert bool(r1.converged) and bool(r2.converged)
    G1 = r1.components()[0] @ r1.components()[0].T
    G2 = r2.components()[0] @ r2.components()[0].T
    # Gauge-invariant Gamma matches across starts (and equals the target).
    assert bool(jnp.allclose(G1, G2, atol=1e-6))
    assert bool(jnp.allclose(G1, cov_hat, atol=1e-6))


# ---------------------------------------------------------------------------
# Test 3 -- warm start: parameters=prev_result.theta round-trips.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warm_source", ["theta", "theta_hat", "point"])
def test_warm_start_round_trips(warm_source):
    K = 2
    X, _mu, _Sig = _make_data(K, n=3000, seed=333)
    M = _moment_count(K)
    model = _make_model(K)
    mean_hat, cov_hat = _sample_moments(X)
    measure = EmpiricalMeasure.from_arrays(X, M=M)

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K), default=jnp.linalg.cholesky(cov_hat))
        mu: jax.Array = on(Euclidean(K), default=mean_hat)

    first = estimate(model, measure, covariance=IIDCovariance(), parameters=Normal)
    assert bool(first.converged)

    if warm_source == "theta":
        warm = first.theta  # a ManifoldPoint view
    elif warm_source == "theta_hat":
        warm = first.theta_hat  # the raw ManifoldLeaf pytree
    else:
        warm = Normal.point()  # a fresh bound instance

    second = estimate(model, measure, covariance=IIDCovariance(), parameters=warm)
    assert bool(second.converged)
    A2, mu2 = second.components()
    assert bool(jnp.allclose(mu2, mean_hat, atol=1e-6))
    assert bool(jnp.allclose(A2 @ A2.T, cov_hat, atol=1e-6))


# ---------------------------------------------------------------------------
# Test 4 -- back-compat: theta_init= == parameters= (bitwise identical).
# ---------------------------------------------------------------------------
def test_theta_init_alias_is_bitwise_identical_to_parameters():
    K = 2
    X, _mu, _Sig = _make_data(K, n=3000, seed=444)
    M = _moment_count(K)
    model = _make_model(K)
    mean_hat, cov_hat = _sample_moments(X)
    measure = EmpiricalMeasure.from_arrays(X, M=M)

    # A hand-built ManifoldLeaf pytree (the established v2 pattern).
    @jdc.pytree_dataclass
    class HandBuilt:
        A: ManifoldLeaf
        mu: ManifoldLeaf

    theta = HandBuilt(
        A=ManifoldLeaf(jnp.linalg.cholesky(cov_hat), PSDFixedRank(K, K)),
        mu=ManifoldLeaf(mean_hat, Euclidean(K)),
    )

    r_old = estimate(model, measure, covariance=IIDCovariance(), theta_init=theta)
    r_new = estimate(model, measure, covariance=IIDCovariance(), parameters=theta)

    assert bool(r_old.converged) and bool(r_new.converged)
    A_old, mu_old = r_old.components()
    A_new, mu_new = r_new.components()
    # Bitwise identical (same start, same path).
    assert bool(jnp.array_equal(A_old, A_new))
    assert bool(jnp.array_equal(mu_old, mu_new))
    assert float(r_old.J_stat) == float(r_new.J_stat)


def test_passing_both_parameters_and_theta_init_errors():
    K = 2

    class Normal(ParameterSpace):
        A: jax.Array = on(PSDFixedRank(K, K), default=jnp.eye(K))
        mu: jax.Array = on(Euclidean(K), default=jnp.zeros(K))

    X, _mu, _Sig = _make_data(K, n=500, seed=1)
    measure = EmpiricalMeasure.from_arrays(X, M=_moment_count(K))
    with pytest.raises(TypeError, match="exactly one"):
        estimate(
            _make_model(K),
            measure,
            covariance=IIDCovariance(),
            parameters=Normal,
            theta_init=Normal.point(),
        )


# ---------------------------------------------------------------------------
# Test 5 -- v1 reduction: an all-scalar ParameterSpace behaves like a scalar
# theta_init dataclass.
# ---------------------------------------------------------------------------
def test_all_scalar_parameter_space_reduces_to_v1():
    # Linear-in-theta moment model: m = (a, b, a + b) - x. Exactly identified.
    target = jnp.array([0.3, -0.7, 0.3 - 0.7])
    n = 2000
    rng = np.random.default_rng(99)
    X = jnp.asarray(target[None, :] + 0.05 * rng.normal(size=(n, 3)))
    measure = EmpiricalMeasure.from_arrays(X, M=3)

    class Scalars(ParameterSpace):
        a: jax.Array = on(Euclidean(), default=jnp.asarray(0.0))
        b: jax.Array = on(Euclidean(), default=jnp.asarray(0.0))

    @jdc.pytree_dataclass
    class P:
        a: jax.Array
        b: jax.Array

    theta_v1 = P(a=jnp.asarray(0.0), b=jnp.asarray(0.0))

    # The v1 model reads bare scalar fields; the ParameterSpace model reads
    # the ManifoldLeaf-wrapped scalar via ``.array``. Both lower to the SAME
    # flat-array layout / criterion, so the numerics are identical.
    def model_v1(x, theta):
        return jnp.array([theta.a, theta.b, theta.a + theta.b]) - x

    def model_space(x, theta):
        a, b = theta.a.array, theta.b.array
        return jnp.array([a, b, a + b]) - x

    r_space = estimate(
        model_space, measure, covariance=IIDCovariance(), parameters=Scalars
    )
    r_v1 = estimate(model_v1, measure, covariance=IIDCovariance(), theta_init=theta_v1)

    assert bool(r_space.converged) and bool(r_v1.converged)
    a_s, b_s = r_space.components()
    a_v, b_v = r_v1.components()
    # The all-scalar (Euclidean) ParameterSpace matches the v1 scalar path.
    assert float(a_s) == pytest.approx(float(a_v), abs=1e-9)
    assert float(b_s) == pytest.approx(float(b_v), abs=1e-9)
    assert float(r_space.J_stat) == pytest.approx(float(r_v1.J_stat), abs=1e-9)
    # An all-Euclidean ParameterSpace takes the v1 dispatch (no Riemannian
    # routing): like the bare scalar dataclass, manifold_spec is None.
    assert r_space.manifold_spec is None
    assert r_v1.manifold_spec is None
