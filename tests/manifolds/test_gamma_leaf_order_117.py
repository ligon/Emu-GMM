r"""Regression gate for #117: Gamma readouts locate the PSDFixedRank leaf
via the manifold spec, not by hard-coding ``components[0]``.

Before the fix, ``gamma_se`` / ``gamma_covariance`` / ``eigenvalue_se``
took ``components[0]`` as the factor ``A`` while the default-``rank``
inference *scanned* the spec for the first 2-D leaf. A parameter
dataclass declared ``(phi, Y)`` -- scalar-ish leaf first, PSD factor
second -- therefore got the right default rank but built ``Gamma`` from
``phi`` (a wrong answer or an opaque shape error). The fix routes both
through ``EstimationResult._gamma_leaf()``.

Gates here:

1. End-to-end: the same DGP estimated under ``(Y, phi)`` and ``(phi, Y)``
   field orders yields the same ``gamma_se`` / ``eigenvalue_se`` /
   ``gamma_covariance`` (the readouts are functions of the gauge
   invariants, which do not depend on dataclass field order).
2. The locator's typed errors: no PSDFixedRank leaf, and more than one.
3. The ``components[0]`` legacy fallback (no spec) validates that the
   selected component is 2-D rather than silently contracting a vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.inference.functional_se import gamma_vech
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5
K = 2
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)


@jdc.pytree_dataclass
class ParamsYFirst:
    """The canonical ordering: PSD factor first (K-Aggregators contract)."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


@jdc.pytree_dataclass
class ParamsPhiFirst:
    """The adversarial ordering: Euclidean leaf first, PSD factor second."""

    phi: ManifoldLeaf
    Y: ManifoldLeaf


def _model(x, theta):
    """psi = concat(triu(Y Y'), phi) - x; gauge-invariant in Y."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _make_measure(data_seed: int, n_sim: int = 200, noise: float = 0.01):
    rng = np.random.default_rng(data_seed)
    A_true = jnp.asarray(rng.normal(size=(N, K)))
    Gamma_true = A_true @ A_true.T
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.asarray([0.7])])
    M = int(target.shape[0])
    noise_key = jax.random.PRNGKey(data_seed)

    def sampler(key, theta):
        del key
        return target[None, :] + noise * jax.random.normal(noise_key, (n_sim, M))

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, K)))
    return measure, Y0


def _leaves(Y0):
    return (
        ManifoldLeaf(jnp.asarray(Y0), PSDFixedRank(N, K)),
        ManifoldLeaf(jnp.asarray([0.65]), Euclidean(1)),
    )


def _fit(theta_init, measure) -> EstimationResult:
    return estimate(
        _model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=400),
        theta_init=theta_init,
    )


# ---------------------------------------------------------------------------
# Gate 1 -- field-order agreement (the #117 headline).
# ---------------------------------------------------------------------------
class TestFieldOrderAgreement:
    def test_gamma_readouts_agree_across_field_order(self):
        measure, Y0 = _make_measure(data_seed=1170)
        Y_leaf, phi_leaf = _leaves(Y0)

        res_y_first = _fit(ParamsYFirst(Y=Y_leaf, phi=phi_leaf), measure)
        res_phi_first = _fit(ParamsPhiFirst(phi=phi_leaf, Y=Y_leaf), measure)
        assert bool(res_y_first.converged) and bool(res_phi_first.converged)

        # The default rank is read off the SAME leaf the readouts use.
        ev_a = res_y_first.eigenvalue_se()
        ev_b = res_phi_first.eigenvalue_se()
        assert ev_a.shape == (K,) and ev_b.shape == (K,)
        np.testing.assert_allclose(np.asarray(ev_a), np.asarray(ev_b), rtol=1e-5)

        g_a = res_y_first.gamma_se()
        g_b = res_phi_first.gamma_se()
        assert g_a.shape == (N * (N + 1) // 2,)
        np.testing.assert_allclose(np.asarray(g_a), np.asarray(g_b), rtol=1e-5)

        c_a = res_y_first.gamma_covariance()
        c_b = res_phi_first.gamma_covariance()
        np.testing.assert_allclose(
            np.asarray(c_a), np.asarray(c_b), rtol=1e-4, atol=1e-12
        )

    def test_locator_points_at_the_psd_leaf(self):
        measure, Y0 = _make_measure(data_seed=1171)
        Y_leaf, phi_leaf = _leaves(Y0)
        res = _fit(ParamsPhiFirst(phi=phi_leaf, Y=Y_leaf), measure)
        idx, ls = res._gamma_leaf()
        assert idx == 1  # Y is the SECOND component under (phi, Y)
        assert isinstance(ls.manifold, PSDFixedRank)
        # And the component at that index really is the (N, K) factor.
        comps = res.components()
        assert tuple(int(s) for s in jnp.asarray(comps[idx]).shape) == (N, K)


# ---------------------------------------------------------------------------
# Gate 2 -- locator typed errors (unit-level; dummy results suffice).
# ---------------------------------------------------------------------------
def _dummy_result(theta_hat) -> EstimationResult:
    """An EstimationResult with only the fields _gamma_leaf touches."""
    from emu_gmm._internal.params import manifold_spec_from_params

    return EstimationResult(
        theta_hat=theta_hat,
        Sigma_theta=None,
        V_X=None,
        J_stat=None,
        J_dof=0,
        J_pvalue=None,
        J_pvalue_adjusted=None,
        converged=True,
        iterations=0,
        theta_init=theta_hat,
        measure=None,
        covariance=None,
        weighting=None,
        regularization=None,
        diagnostics=None,
        labels=None,
        manifold_spec=manifold_spec_from_params(theta_hat),
    )


@jdc.pytree_dataclass
class _NoPSDParams:
    phi: ManifoldLeaf


@jdc.pytree_dataclass
class _TwoPSDParams:
    Y1: ManifoldLeaf
    Y2: ManifoldLeaf


class TestLocatorTypedErrors:
    def test_no_psd_leaf_raises(self):
        theta = _NoPSDParams(phi=ManifoldLeaf(jnp.asarray([0.5]), Euclidean(1)))
        res = _dummy_result(theta)
        with pytest.raises(TypeError, match="no PSDFixedRank leaf"):
            res._gamma_leaf()

    def test_multiple_psd_leaves_raises(self):
        theta = _TwoPSDParams(
            Y1=ManifoldLeaf(jnp.ones((3, 2)), PSDFixedRank(3, 2)),
            Y2=ManifoldLeaf(jnp.ones((4, 2)), PSDFixedRank(4, 2)),
        )
        res = _dummy_result(theta)
        with pytest.raises(TypeError, match="no canonical Gamma"):
            res._gamma_leaf()


# ---------------------------------------------------------------------------
# Gate 3 -- legacy components[0] fallback validates dimensionality.
# ---------------------------------------------------------------------------
class TestLegacyFallbackValidation:
    def test_gamma_vech_rejects_non_2d_component(self):
        """Without a spec the legacy contract is components[0]; a non-2-D
        component must raise a typed error rather than silently
        contracting a vector into a scalar 'Gamma'."""
        with pytest.raises(TypeError, match="2-D ambient factor"):
            gamma_vech((jnp.asarray([1.0, 2.0]), jnp.ones((3, 2))))
