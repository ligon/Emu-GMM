"""PR B (B2) spike gates: RiemannianLM with the measure as a traced arg.

The #124 spike assessed the v2 manifold path by READING the code; these
tests establish it by RUNNING it. A two-argument kernel closing over the
manifold machinery (manifold-aware flatten/unflatten + the moment
expectation) is handed to :func:`riemannian_lm` with the measure riding
``args=``:

(1) Recovery parity: the args path matches the legacy one-argument
    closure path on the SAME data to 1e-8 (same deterministic iteration;
    only the jit boundary differs).
(2) Retrace gate: two fresh same-structure :class:`SyntheticMeasure`
    instances (via ``with_key``) share ONE trace -- the counting-psi
    counter freezes after the first solve.
(3) The legacy one-argument contract is untouched (also pinned by the
    whole pre-existing tests/manifolds suite).

DGP: the small ``Product(PSDFixedRank(5, 2), Euclidean(1))`` problem from
``test_gamma_leaf_order_117.py``, with the sampler keyed on the measure's
PRNG key so ``with_key`` genuinely yields fresh draws.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure

jax.config.update("jax_enable_x64", True)

N = 5
K = 2
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)
_N_SIM = 200


@jdc.pytree_dataclass
class _Params:
    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _psi(x, theta):
    """psi = concat(triu(Y Y'), phi) - x; gauge-invariant in Y."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


class _CountingPsi:
    """psi wrapper counting Python-level executions (== trace events)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, theta):
        self.calls += 1
        return _psi(x, theta)


def _make_measure(data_seed: int, noise: float = 0.01):
    """Target + key-driven noise, so ``with_key`` yields fresh draws."""
    rng = np.random.default_rng(data_seed)
    A_true = jnp.asarray(rng.normal(size=(N, K)))
    Gamma_true = A_true @ A_true.T
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.asarray([0.7])])
    M = int(target.shape[0])

    def sampler(key, theta):
        del theta
        return target[None, :] + noise * jax.random.normal(key, (_N_SIM, M))

    measure = SyntheticMeasure(
        key=jax.random.PRNGKey(data_seed), n_sim=_N_SIM, sampler=sampler
    )
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, K)))
    return measure, Y0, Gamma_true


def _theta0(Y0):
    return _Params(
        Y=ManifoldLeaf(jnp.asarray(Y0), PSDFixedRank(N, K)),
        phi=ManifoldLeaf(jnp.asarray([0.65]), Euclidean(1)),
    )


def _make_kernel(psi, spec, treedef):
    """Two-argument kernel: manifold machinery closed over, measure as arg."""

    def kernel(theta_flat, measure):
        theta = params_mod.unflatten_params(theta_flat, treedef, manifold_spec=spec)
        return jnp.asarray(measure.expectation(psi, theta))

    return kernel


def _gamma(theta_hat):
    Y = jnp.asarray(theta_hat.Y.array)
    return Y @ Y.T


class TestArgsPathParity:
    def test_args_path_matches_closure_path_on_same_data(self):
        measure, Y0, _ = _make_measure(data_seed=1240)
        theta0 = _theta0(Y0)
        spec = params_mod.manifold_spec_from_params(theta0)
        _, treedef, _ = params_mod.flatten_params_with_spec(theta0)
        kernel = _make_kernel(_psi, spec, treedef)
        rlm = riemannian_lm(max_steps=400)

        # Legacy contract: one-argument closure binding the measure.
        th_closure, info_c = rlm(lambda tf: kernel(tf, measure), theta0, spec)
        # PR B contract: the measure rides args=.
        th_args, info_a = rlm(kernel, theta0, spec, args=measure)

        assert bool(info_c.done) and bool(info_a.done)
        assert info_a.status == "converged"

        flat_c, _, _ = params_mod.flatten_params_with_spec(th_closure)
        flat_a, _, _ = params_mod.flatten_params_with_spec(th_args)
        np.testing.assert_allclose(
            np.asarray(flat_a), np.asarray(flat_c), atol=1e-8, rtol=0
        )
        # And on the gauge invariant (the quantity that is actually
        # identified) the agreement must hold too.
        np.testing.assert_allclose(
            np.asarray(_gamma(th_args)),
            np.asarray(_gamma(th_closure)),
            atol=1e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            float(info_a.final_objective),
            float(info_c.final_objective),
            rtol=1e-8,
            atol=1e-12,
        )

    def test_args_path_recovers_truth(self):
        measure, Y0, Gamma_true = _make_measure(data_seed=1241)
        theta0 = _theta0(Y0)
        spec = params_mod.manifold_spec_from_params(theta0)
        _, treedef, _ = params_mod.flatten_params_with_spec(theta0)
        kernel = _make_kernel(_psi, spec, treedef)

        th_args, info = riemannian_lm(max_steps=400)(kernel, theta0, spec, args=measure)
        assert bool(info.done)
        # noise=0.01 on the moment targets -> loose recovery bound.
        np.testing.assert_allclose(
            np.asarray(_gamma(th_args)), np.asarray(Gamma_true), atol=0.05
        )
        np.testing.assert_allclose(float(th_args.phi.array[0]), 0.7, atol=0.05)


class TestArgsPathNoRetrace:
    def test_fresh_same_structure_measures_share_one_trace(self):
        base, Y0, _ = _make_measure(data_seed=1242)
        theta0 = _theta0(Y0)
        spec = params_mod.manifold_spec_from_params(theta0)
        _, treedef, _ = params_mod.flatten_params_with_spec(theta0)
        counting = _CountingPsi()
        kernel = _make_kernel(counting, spec, treedef)
        rlm = riemannian_lm(max_steps=400)

        m1 = base.with_key(jax.random.PRNGKey(11))
        m2 = base.with_key(jax.random.PRNGKey(12))

        th1, info1 = rlm(kernel, theta0, spec, args=m1)
        assert bool(info1.done)
        calls_after_first = counting.calls
        assert calls_after_first > 0  # tracing happened

        th2, info2 = rlm(kernel, theta0, spec, args=m2)
        assert bool(info2.done)
        assert counting.calls == calls_after_first, (
            f"psi re-traced on a fresh same-structure measure: "
            f"{counting.calls} != {calls_after_first}"
        )
        # Fresh draws -> a genuinely different (but nearby) optimum.
        g1 = np.asarray(_gamma(th1))
        g2 = np.asarray(_gamma(th2))
        assert np.max(np.abs(g1 - g2)) > 0.0
        np.testing.assert_allclose(g1, g2, atol=0.1)


class TestLegacyContractUntouched:
    def test_one_argument_closure_still_eager_per_call(self):
        """The legacy path is NOT routed through the memoised jit: each
        call traces (the documented pre-#124 behaviour), pinning that
        the args channel did not change the default contract."""
        base, Y0, _ = _make_measure(data_seed=1243)
        theta0 = _theta0(Y0)
        spec = params_mod.manifold_spec_from_params(theta0)
        _, treedef, _ = params_mod.flatten_params_with_spec(theta0)
        counting = _CountingPsi()
        kernel = _make_kernel(counting, spec, treedef)
        rlm = riemannian_lm(max_steps=400)

        rlm(lambda tf: kernel(tf, base), theta0, spec)
        calls_first = counting.calls
        rlm(lambda tf: kernel(tf, base), theta0, spec)
        assert counting.calls > calls_first
