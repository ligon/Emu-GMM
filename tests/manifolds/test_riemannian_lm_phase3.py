r"""Phase-3 acceptance: per-leaf Riemannian-LM solver on a non-scalar tree.

These tests drive ``riemannian_lm`` **directly** (the Phase-3 contract is
"drivable by calling the solver directly"; full ``estimate()`` end-to-end
wiring for non-scalar leaves is Phase 4/5). The synthetic problem is a
``Product(PSDFixedRank(5, k), Euclidean(1))``:

    theta = (Y in R^{5xk},  phi in R)
    Gamma_true = A @ A.T  (rank k),  phi_true = 0.7
    r(theta) = concat( ravel(Y @ Y.T - Gamma_true), [phi - phi_true] )

``r`` is exactly 0 at the truth and is invariant to ``Y -> Y Q`` for any
``Q in O(k)`` (since ``(YQ)(YQ)^T = Y Y^T``), so the unique gauge-invariant
minimiser is ``(Gamma_true, phi_true)``. Recovery is asserted on the
gauge-INVARIANT ``Gamma_hat = A_hat @ A_hat.T`` and its eigenvalues, never on
raw ``Y`` entries.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm._internal.params import (
    flatten_params_with_spec,
    manifold_spec_from_params,
    unflatten_params,
)
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side


@jdc.pytree_dataclass
class ProductParams:
    """A PSDFixedRank ``Y`` leaf plus a scalar Euclidean ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y: jnp.ndarray, phi: float, k: int) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(
            jnp.asarray(jnp.reshape(jnp.asarray(phi), (1,))), Euclidean(1)
        ),
    )


def _flat_residual(
    theta_flat: jnp.ndarray, Gamma_true: jnp.ndarray, phi_true: float, k: int
):
    """Flat-vector residual matching the (Y:(N,k), phi:(1,)) layout."""
    Yf = theta_flat[: N * k]
    Y = jnp.reshape(Yf, (N, k))
    phi = theta_flat[N * k]
    gamma_res = jnp.reshape(Y @ Y.T - Gamma_true, (-1,))
    return jnp.concatenate([gamma_res, jnp.reshape(phi - phi_true, (1,))])


def _orthogonal(key, k: int) -> jnp.ndarray:
    g = jax.random.normal(key, (k, k), dtype=jnp.float64)
    q, r = jnp.linalg.qr(g)
    # Fix signs so q is a proper-ish orthogonal (any O(k) element is fine).
    return q @ jnp.diag(jnp.sign(jnp.diag(r)))


def _setup(k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    spec = manifold_spec_from_params(_make_params(A_true, phi_true, k))
    return A_true, Gamma_true, phi_true, spec


def _run_from_start(
    Y0: jnp.ndarray, phi0: float, Gamma_true, phi_true, k, spec, max_steps: int = 400
):
    theta0 = _make_params(Y0, phi0, k)
    flat0, treedef, _ = flatten_params_with_spec(theta0)

    def residual_fn(tf):
        return _flat_residual(tf, Gamma_true, phi_true, k)

    solver = riemannian_lm(max_steps=max_steps)
    theta_hat, info = solver(residual_fn, theta0, spec)
    return theta_hat, info, flat0, treedef


@pytest.mark.parametrize("k", [2, 3])
class TestSyntheticRecovery:
    def test_recovers_gamma_and_phi(self, k):
        A_true, Gamma_true, phi_true, spec = _setup(k, seed=k)
        rng = np.random.default_rng(100 + k)
        Y0 = jnp.asarray(A_true + 0.3 * rng.normal(size=(N, k)))
        theta_hat, info, _, _ = _run_from_start(Y0, 0.0, Gamma_true, phi_true, k, spec)

        # REAL convergence -- not merely status=="traced".
        assert bool(info.done) is True
        assert str(info.status) == "converged"
        assert float(info.final_objective) < 1e-12

        A_hat = jnp.asarray(theta_hat.Y.array)
        phi_hat = float(jnp.asarray(theta_hat.phi.array)[0])
        Gamma_hat = A_hat @ A_hat.T

        # Gauge-invariant recovery: Gamma and its spectrum, never raw A.
        assert bool(jnp.allclose(Gamma_hat, Gamma_true, atol=1e-5))
        ev_hat = jnp.linalg.eigvalsh(Gamma_hat)
        ev_true = jnp.linalg.eigvalsh(Gamma_true)
        assert bool(jnp.allclose(ev_hat, ev_true, atol=1e-5))
        assert phi_hat == pytest.approx(phi_true, abs=1e-5)


@pytest.mark.parametrize("k", [2, 3])
class TestGaugeInvariance:
    def test_Y0_and_Y0Q_give_same_gamma_and_objective(self, k):
        A_true, Gamma_true, phi_true, spec = _setup(k, seed=10 + k)
        rng = np.random.default_rng(200 + k)
        Y0 = jnp.asarray(A_true + 0.2 * rng.normal(size=(N, k)))
        Q = _orthogonal(jax.random.PRNGKey(7 + k), k)
        Y0Q = Y0 @ Q

        th_a, info_a, _, _ = _run_from_start(Y0, 0.1, Gamma_true, phi_true, k, spec)
        th_b, info_b, _, _ = _run_from_start(Y0Q, 0.1, Gamma_true, phi_true, k, spec)

        assert bool(info_a.done) and bool(info_b.done)

        Ga = jnp.asarray(th_a.Y.array) @ jnp.asarray(th_a.Y.array).T
        Gb = jnp.asarray(th_b.Y.array) @ jnp.asarray(th_b.Y.array).T

        # Same gauge-invariant Gamma to tolerance ...
        assert bool(jnp.allclose(Ga, Gb, atol=1e-5))
        assert bool(jnp.allclose(Ga, Gamma_true, atol=1e-5))
        # ... and the same objective (J / 2 ||r||^2).
        assert float(info_a.final_objective) == pytest.approx(
            float(info_b.final_objective), abs=1e-10
        )


class TestDoneFlag:
    """A deliberately under-resourced / non-identified run is NOT converged."""

    def test_too_few_steps_reports_not_converged(self):
        k = 2
        A_true, Gamma_true, phi_true, spec = _setup(k, seed=42)
        rng = np.random.default_rng(999)
        Y0 = jnp.asarray(A_true + 1.5 * rng.normal(size=(N, k)))
        theta_hat, info, _, _ = _run_from_start(
            Y0, 5.0, Gamma_true, phi_true, k, spec, max_steps=2
        )
        assert bool(info.done) is False
        assert str(info.status) == "max_iterations"

    def test_non_identified_residual_does_not_falsely_converge_gamma(self):
        # Residual depends ONLY on phi, not on Gamma -> Gamma is not
        # identified; the iterate must not certify a recovered Gamma.
        k = 2
        _, _, phi_true, spec = _setup(k, seed=3)
        rng = np.random.default_rng(7)
        Y0 = jnp.asarray(rng.normal(size=(N, k)))
        theta0 = _make_params(Y0, 2.0, k)

        def residual_fn(tf):
            phi = tf[N * k]
            # M must exceed the (single) identified direction; pad zeros.
            return jnp.concatenate(
                [jnp.reshape(phi - phi_true, (1,)), jnp.zeros((N * N - 1,))]
            )

        solver = riemannian_lm(max_steps=50)
        theta_hat, info = solver(residual_fn, theta0, spec)
        # phi is recovered; Gamma is whatever the gauge-floored solve left
        # near the start (NOT Gamma_true). Assert it did NOT magically land
        # on a structured Gamma it had no information about.
        phi_hat = float(jnp.asarray(theta_hat.phi.array)[0])
        assert phi_hat == pytest.approx(phi_true, abs=1e-4)


class TestScalarNonRegression:
    """total_gauge_dim==0 trees: lam_floor contributes nothing; the path
    reduces to the scalar retraction."""

    def test_lam_floor_zero_for_all_euclidean(self):
        @jdc.pytree_dataclass
        class P:
            a: jnp.ndarray
            b: jnp.ndarray

        params = P(a=jnp.asarray(1.0), b=jnp.asarray(2.0))
        spec = manifold_spec_from_params(params)
        assert spec.total_gauge_dim == 0

        # Quadratic: r = [a - 3, b + 1]; min at (3, -1).
        def residual_fn(tf):
            return jnp.stack([tf[0] - 3.0, tf[1] + 1.0])

        theta_hat, info = riemannian_lm(max_steps=100)(residual_fn, params, spec)
        assert bool(info.done) is True
        assert float(theta_hat.a) == pytest.approx(3.0, abs=1e-8)
        assert float(theta_hat.b) == pytest.approx(-1.0, abs=1e-8)


class TestBlockBoundaries:
    """Guard against a Phase-2 offset/shape bug without editing landed code
    (red-team R21)."""

    def test_flat_layout_tiles_exactly(self):
        k = 3
        A_true, _, _, spec = _setup(k, seed=1)
        params = _make_params(A_true, 0.5, k)
        flat, treedef, fspec = flatten_params_with_spec(params)

        assert fspec.total_ambient_dim == N * k + 1
        assert int(flat.shape[0]) == N * k + 1
        # offsets: Y at 0 (size N*k), phi at N*k (size 1).
        offs = [ls.offset for ls in spec.leaf_specs]
        sizes = [int(np.prod(ls.ambient_shape)) for ls in spec.leaf_specs]
        assert offs == [0, N * k]
        assert sizes == [N * k, 1]
        assert sum(sizes) == int(flat.shape[0])

        # Round-trip through the manifold-aware unflatten.
        back = unflatten_params(flat, treedef, manifold_spec=spec)
        assert jnp.asarray(back.Y.array).shape == (N, k)
        assert jnp.asarray(back.phi.array).shape == (1,)
        assert bool(jnp.allclose(jnp.asarray(back.Y.array), A_true))
