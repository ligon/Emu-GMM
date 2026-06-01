r"""Phase-4 acceptance: gauge-aware Sigma_theta for a non-scalar Product.

Drives the **inference block** of ``estimate()`` for a recovered
``Product(PSDFixedRank(5, K), Euclidean(1))`` solve and checks the four
Phase-4 contract items:

* ``info_matrix.shape == (5K+1, 5K+1)`` and ``G_riem`` uses ALL ambient
  columns (BUG-A fixed: per-leaf assembly, no silent ``range(K)`` drop);
* ``pinv_eigvalrule`` drops EXACTLY ``total_gauge_dim == K(K-1)/2`` smallest
  eigenvalues by count, and ``Sigma_theta`` is finite (BUG-B fixed);
* ``J_dof == M - (total_dimension - total_gauge_dim)`` flows from the
  populated ``total_gauge_dim`` (NOT hard-coded 0);
* ``gauge_nullspace_dim == total_gauge_dim`` is surfaced in Diagnostics.

The model depends on ``theta`` only through the gauge-invariant
``Gamma = Y Y^T`` (and ``phi``), so the unique gauge-invariant minimiser is
the truth and the gauge directions of the info matrix are exactly null.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side


@jdc.pytree_dataclass
class ProductParams:
    """A PSDFixedRank ``Y`` leaf plus a Euclidean(1) ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k):
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _moment_count(k):
    # M must satisfy M >= identified dim == 5K + 1 - K(K-1)/2.
    # We use the n(n+1)/2 unique upper-triangular entries of Gamma plus 1
    # (phi), giving M = 15 + 1 = 16 for n=5 -- over-identified for K=2,3.
    return N * (N + 1) // 2 + 1


_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2) index pairs


def _model(x, theta):
    """Gauge-invariant residual: Gamma=YY^T upper-tri entries + phi.

    ``x`` carries the (noiseless) target moments as a length-M vector;
    the residual is (model_moment - x), so E_mu[psi] == model_moment -
    mean(x). The sampler returns the SAME target at every draw, so the
    synthetic expectation is exactly the model moment minus the target.
    """
    Y = theta.Y.array
    phi = theta.phi.array[0]
    Gamma = Y @ Y.T
    g = Gamma[_TRIU[:, 0], _TRIU[:, 1]]
    model_m = jnp.concatenate([g, jnp.reshape(phi, (1,))])
    return model_m - x


def _run_estimate(k, seed=0):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    M = _moment_count(k)
    assert int(target.shape[0]) == M

    n_sim = 200

    def sampler(key, theta):
        # Exogenous per-draw target moments: the truth plus i.i.d. noise so
        # the moment-variance V is well-conditioned (a noiseless / identical
        # draw makes V singular and the whitening blow up). CRN-frozen key
        # -> deterministic objective surface in theta; independent of theta,
        # so E_mu[psi] = model_m(theta) - mean(draws) ~ model_m - target.
        noise = 0.1 * jax.random.normal(key, (n_sim, M))
        return target[None, :] + noise

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)

    # Warm start near the truth so the solver lands on the gauge-invariant
    # minimiser; recovery itself is the Phase-3 contract, here we only need
    # a converged theta_hat to exercise the inference block.
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
    theta_init = _make_params(Y0, 0.65, k)

    result = estimate(
        _model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=400),
        theta_init=theta_init,
    )
    spec = manifold_spec_from_params(theta_init)
    return result, spec, M, k


@pytest.mark.parametrize("k", [2, 3])
class TestPhase4Inference:
    def test_info_and_sigma_shape_uses_all_ambient_columns(self, k):
        result, spec, M, _ = _run_estimate(k, seed=10 + k)
        D = spec.total_dimension
        assert D == N * k + 1  # 11 for k=2, 16 for k=3
        # Sigma_theta is the ambient (D, D) sandwich -> ALL ambient columns
        # flowed through G_riem (BUG-A). A range(K)=2-column drop would
        # give (2, 2).
        assert result.Sigma_theta.array.shape == (D, D)

    def test_sigma_finite(self, k):
        result, _, _, _ = _run_estimate(k, seed=20 + k)
        assert bool(jnp.all(jnp.isfinite(result.Sigma_theta.array)))

    def test_exact_gauge_eigenvalue_drop_count(self, k):
        result, spec, M, _ = _run_estimate(k, seed=30 + k)
        gauge = k * (k - 1) // 2
        assert spec.total_gauge_dim == gauge
        assert result.diagnostics.gauge_nullspace_dim == gauge
        # Sigma_theta rank == total_dimension - gauge_dim: exactly the
        # gauge_dim directions are pinned to zero (the dropped eigenpairs).
        D = spec.total_dimension
        evals = jnp.linalg.eigvalsh(
            0.5 * (result.Sigma_theta.array + result.Sigma_theta.array.T)
        )
        n_zero = int(jnp.sum(jnp.abs(evals) < 1e-10 * jnp.max(jnp.abs(evals))))
        assert n_zero == gauge
        assert int(jnp.sum(jnp.abs(evals) >= 1e-10 * jnp.max(jnp.abs(evals)))) == (
            D - gauge
        )

    def test_J_dof_calibrates_from_populated_gauge_dim(self, k):
        result, spec, M, _ = _run_estimate(k, seed=40 + k)
        gauge = k * (k - 1) // 2
        D = spec.total_dimension
        expected = max(M - (D - gauge), 0)
        assert result.J_dof == expected
        # And NOT the hard-coded-0-gauge value (M - D), which would be 1
        # smaller for k=2 and 3 smaller for k=3.
        assert result.J_dof == M - (D - gauge)
        assert result.J_dof != (M - D) or gauge == 0


class TestUnderIdentificationGuard:
    def test_guard_uses_identified_ambient_dimension(self):
        # M = 5 < identified dim (11 - 1 = 10) for K=2: must raise.
        from emu_gmm.types import Emu_GMM_DimensionError

        k = 2
        rng = np.random.default_rng(7)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        M_small = 5

        def sampler(key, theta):
            return jnp.zeros((8, M_small))

        def model(x, theta):
            Y = theta.Y.array
            g = (Y @ Y.T)[_TRIU[:M_small, 0], _TRIU[:M_small, 1]]
            return g - x

        measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=8, sampler=sampler)
        theta_init = _make_params(A_true, 0.5, k)
        with pytest.raises(Emu_GMM_DimensionError):
            estimate(
                model,
                measure,
                covariance=SyntheticCovariance(),
                optimizer=riemannian_lm(max_steps=10),
                theta_init=theta_init,
            )
