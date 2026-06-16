r"""Regression test for #146: IteratedWeighting on a manifold space.

Surfaced porting K-Aggregators' ``Product(PSDFixedRank(5, K), Euclidean(1))``
GMM (label ``from-k-aggregators``): ``IteratedWeighting`` raised
``unflatten_params: flat array has N elements but treedef expects M
leaves`` on any non-scalar (manifold) parameter space, because
``outer_loop_driver`` unflattened without the ``manifold_spec``. Fixing
that exposed a second gap -- the inner solve was called with the v1
two-argument convention, but a manifold space dispatches to
``RiemannianLM`` whose ``__call__`` needs the pytree init plus
``manifold_spec`` and returns a pytree. The driver now threads
``manifold_spec`` and, for a Riemannian optimiser, round-trips
pytree<->flat for the inner solve.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated, IteratedWeighting

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side
K = 2  # PSDFixedRank
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)

_RNG = np.random.default_rng(7)
_A_TRUE = jnp.asarray(_RNG.normal(size=(N, K)))
_GAMMA_TRUE = _A_TRUE @ _A_TRUE.T
_PHI_TRUE = 0.7
_TARGET = jnp.concatenate(
    [_GAMMA_TRUE[_TRIU[:, 0], _TRIU[:, 1]], jnp.reshape(jnp.asarray(_PHI_TRUE), (1,))]
)
_M = int(_TARGET.shape[0])  # 15 + 1


@jdc.pytree_dataclass
class ProductParams:
    """A PSDFixedRank ``Y`` leaf plus a Euclidean(1) ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi):
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, K)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _model(x, theta):
    Y = theta.Y.array
    phi = theta.phi.array[0]
    Gamma = Y @ Y.T
    g = Gamma[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _sampler(key, theta):
    return _TARGET[None, :] + 0.1 * jax.random.normal(key, (200, _M))


def _run(weighting):
    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=200, sampler=_sampler)
    Y0 = jnp.asarray(_A_TRUE + 0.05 * _RNG.normal(size=(N, K)))
    with warnings.catch_warnings():
        # Iterated-on-misspecification can warn about non-convergence; this
        # synthetic DGP is well-specified, but keep the run quiet.
        warnings.simplefilter("ignore")
        return estimate(
            _model,
            measure,
            covariance=SyntheticCovariance(),
            weighting=weighting,
            optimizer=riemannian_lm(max_steps=400),
            theta_init=_make_params(Y0, 0.65),
        )


def test_iterated_runs_and_recovers_gamma():
    # The pre-fix failure was a ValueError before the optimizer ran.
    result = _run(IteratedWeighting(weighting_iterations=10, weighting_tol=1e-8))
    A = result.theta_hat.Y.array
    assert A.shape == (N, K)
    Gamma_hat = A @ A.T
    # Gauge-invariant recovery near the truth (0.1 i.i.d. moment noise).
    assert jnp.allclose(Gamma_hat, _GAMMA_TRUE, atol=5e-2)
    assert result.converged


def test_iterated_matches_continuously_updated():
    # Iterated and CU are asymptotically equivalent and, on this
    # well-specified synthetic problem, agree to numerical tolerance. The
    # comparison is on Gamma = A A^T (gauge-invariant), not raw A.
    r_it = _run(IteratedWeighting(weighting_iterations=10, weighting_tol=1e-10))
    r_cu = _run(ContinuouslyUpdated())
    A_it, A_cu = r_it.theta_hat.Y.array, r_cu.theta_hat.Y.array
    G_it, G_cu = A_it @ A_it.T, A_cu @ A_cu.T
    assert jnp.allclose(G_it, G_cu, atol=1e-6)
    assert jnp.allclose(
        float(r_it.theta_hat.phi.array[0]),
        float(r_cu.theta_hat.phi.array[0]),
        atol=1e-6,
    )
    assert jnp.allclose(float(r_it.J_stat), float(r_cu.J_stat), atol=1e-6)
