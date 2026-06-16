r"""Regression test for #147: pickle round-trip of a manifold-valued result.

Surfaced porting K-Aggregators' ``Product(PSDFixedRank(5, K), Euclidean(1))``
GMM (label ``from-k-aggregators``): ``EstimationResult.from_pickle`` raised
``ManifoldLeaf is immutable`` because the frozen leaf rejected pickle's
slot-state reconstruction. ``ManifoldLeaf`` now defines ``__getstate__`` /
``__setstate__``; the unit-level round-trip lives in
``tests/_internal/test_params_manifold_leaf.py``, and this is the
end-to-end ``EstimationResult`` round-trip exercising the reported symptom.

Everything the pickle path touches (the params dataclass, the model, the
sampler) is module-level here so ``to_pickle`` resolves it by reference
without the ``__main__`` portability warning.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side
K = 2  # PSDFixedRank
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)

# Module-level truth so the sampler below is importable (picklable).
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


def _sampler(key, theta):  # module-level -> picklable result.measure
    return _TARGET[None, :] + 0.1 * jax.random.normal(key, (200, _M))


def _run():
    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=200, sampler=_sampler)
    Y0 = jnp.asarray(_A_TRUE + 0.05 * _RNG.normal(size=(N, K)))
    return estimate(
        _model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=400),
        theta_init=_make_params(Y0, 0.65),
    )


def test_estimation_result_pickle_round_trip(tmp_path):
    result = _run()
    path = tmp_path / "manifold_result.pkl"
    # to_pickle must not warn (all provenance is importable here) ...
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.to_pickle(path)
    # ... and from_pickle must not raise "ManifoldLeaf is immutable".
    loaded = EstimationResult.from_pickle(path)
    assert isinstance(loaded, EstimationResult)
    assert isinstance(loaded.theta_hat, ProductParams)
    assert isinstance(loaded.theta_hat.Y, ManifoldLeaf)
    np.testing.assert_array_equal(
        np.asarray(loaded.theta_hat.Y.array),
        np.asarray(result.theta_hat.Y.array),
    )
    assert loaded.theta_hat.Y.manifold == result.theta_hat.Y.manifold
    assert float(loaded.J_stat) == float(result.J_stat)
    # The reconstructed leaf is still immutable.
    with pytest.raises(AttributeError):
        loaded.theta_hat.Y.array = jnp.zeros((N, K))
