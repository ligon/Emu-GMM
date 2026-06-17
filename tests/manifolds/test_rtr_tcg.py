r"""Truncated-CG (Steihaug-Toint) unit tests for the Riemannian Trust Region
optimizer (#152, theme: ``tcg``).

TEST-FIRST. The module under test (``emu_gmm.manifolds.riemannian_tr``) does
**not exist yet** -- every test here is RED (ImportError, then assertion)
until Phase 2 lands the implementation. That is intended: this file is the
executable specification of the inner solver, written against the INTENDED
Phase-2 API (issue #152 revision):

    from emu_gmm.manifolds.riemannian_tr import (
        riemannian_tr,        # factory -> RiemannianOptimizer
        _riemannian_hvp,      # unit-level HVP (retraction-pullback Hessian)
        _truncated_cg,        # unit-level Steihaug-Toint inner solve
    )

These tests pin the five ``tcg``-lens red-team risks (Phase-0 register,
#152). Each class docstring names the risk it pins. The assertions are
deliberately STRONG -- a Euclidean-sum metric bug, an output-only (non
self-adjoint) HVP, a dropped re-tangentialization, an unguarded ``z_r/d_Hd``
divide, or a flat-rtol stopping rule must each make a concrete test FAIL,
not pass vacuously.

Helper-contract assumptions (flag in ``shared_helpers_needed`` for Phase 2):

* ``_riemannian_hvp(residual_fn, theta_flat, manifold_spec, eta_flat) ->
  hvp_flat`` is the retraction-pullback Euclidean Hessian of
  ``Q(R_Y(eta))`` at ``eta=0`` applied to ``eta``, projected to horizontal
  per leaf. It accepts and returns FLAT ambient vectors (the same layout as
  ``flatten_params_with_spec``). It must be self-adjoint in the manifold
  metric on the horizontal/tangent subspace.

* ``_truncated_cg(hvp, grad_flat, manifold_spec, point_flat, Delta, *,
  theta, kappa, min_inner, max_tcg_steps) -> (eta_flat, Heta_flat, info)``
  is the Steihaug-Toint inner solve. ``hvp`` is a one-arg callable
  ``eta_flat -> H[eta]_flat`` (already closed over the point). ``info`` is a
  mapping/namespace exposing at least ``num_inner`` (inner iterations
  actually run, == pymanopt's returned ``j``) and ``stop_reason`` (one of
  the strings below). Inner products / norms / the tau boundary solve use
  the per-leaf ``manifold_spec`` metric, NOT Euclidean sums.

  ``stop_reason in {"negative_curvature", "exceeded_tr",
  "reached_target_linear", "reached_target_superlinear",
  "max_inner", "model_increased"}`` -- mirroring pymanopt's six codes.

If Phase 2 picks different names/shapes, update THIS contract (one place),
not the math: the math here is the specification.
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
)
from emu_gmm.manifolds import Euclidean, Positive, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf

# RED until Phase 2: SKIP cleanly until the module lands (then go live).
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
_riemannian_hvp = riemannian_tr_mod._riemannian_hvp
_truncated_cg = riemannian_tr_mod._truncated_cg

jax.config.update("jax_enable_x64", True)

N = 4  # ambient PSD side for the unit fixtures (small, fast, k/n <= 0.75)


# ===========================================================================
# Shared fixtures / helpers (candidates for a shared conftest -- see
# shared_helpers_needed).
# ===========================================================================
@jdc.pytree_dataclass
class _PSDParams:
    """A single ``PSDFixedRank(N, k)`` ``Y`` leaf."""

    Y: ManifoldLeaf


@jdc.pytree_dataclass
class _PSDPosParams:
    """``Product(PSDFixedRank(N, k), Positive())`` -- mixed-metric tree."""

    Y: ManifoldLeaf
    s: ManifoldLeaf


def _psd_params(Y: jnp.ndarray, k: int) -> _PSDParams:
    return _PSDParams(Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)))


def _psd_pos_params(Y: jnp.ndarray, s: float, k: int) -> _PSDPosParams:
    # Positive's ambient_shape is () (0-d scalar); ManifoldLeaf carries it.
    return _PSDPosParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        s=ManifoldLeaf(jnp.asarray(jnp.float64(s)), Positive()),  # type: ignore[arg-type]
    )


_TRIU4 = jnp.array(np.triu_indices(N)).T  # (10, 2): unique entries of a 4x4 sym


def _gauge_invariant_residual(target: jnp.ndarray, k: int):
    """``psi(theta_flat) = triu(Y Y^T) - target`` -- depends on ``Y`` ONLY
    through ``Gamma = Y Y^T`` (so ``Q`` is gauge-invariant; egrad horizontal).
    Flat layout: the whole flat buffer is ``vec(Y)`` for a pure-PSD tree."""

    def residual_fn(theta_flat: jnp.ndarray) -> jnp.ndarray:
        Y = jnp.reshape(theta_flat[: N * k], (N, k))
        g = (Y @ Y.T)[_TRIU4[:, 0], _TRIU4[:, 1]]
        return g - target

    return residual_fn


def _objective(residual_fn):
    """Q(theta_flat) = 1/2 ||r||^2 (the unweighted LS criterion)."""

    def Q(theta_flat: jnp.ndarray) -> jnp.ndarray:
        r = residual_fn(theta_flat)
        return 0.5 * jnp.sum(r * r)

    return Q


def _horizontal_project_psd(Y: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Reference horizontal projection on PSDFixedRank, independent of src.

    Solves the Sylvester/Lyapunov system ``YtY Omega + Omega YtY = Yt V -
    Vt Y`` for skew ``Omega`` and returns ``V - Y Omega``. Mirrors
    pymanopt psd.py but recomputed here so the test does not merely re-run
    the code it is checking."""
    YtY = Y.T @ Y
    AS = Y.T @ V - V.T @ Y
    k = Y.shape[1]
    eye = jnp.eye(k)
    M = jnp.kron(eye, YtY) + jnp.kron(YtY, eye)
    omega = jnp.linalg.solve(M, AS.reshape(-1)).reshape(k, k)
    return V - Y @ omega


def _reference_pullback_hvp(residual_fn, manifold, Y: jnp.ndarray, V: jnp.ndarray):
    r"""Reference retraction-pullback Hessian-vector product for a PSD leaf.

    ``H[V] = Proj_h( d/dt grad_amb Q(R_Y(t W)) |_{t=0} )`` evaluated at
    ``W = Proj_h(V)``, where ``R`` is the manifold retraction and
    ``grad_amb`` the ambient (Euclidean) gradient of ``Q``. Computed by an
    independent autodiff path (jvp of the ambient grad through the
    retraction), then projected to horizontal. This is the metric-exact
    Riemannian Hessian for the additive PSD retraction; the test compares
    the src ``_riemannian_hvp`` against it."""
    Q = _objective(residual_fn)
    W = _horizontal_project_psd(Y, V)

    def amb_grad_at(t):
        Yt = manifold.retraction(Y, t * W)
        # ambient gradient of Q wrt the (N,k) point
        return jax.grad(lambda Z: Q(Z.reshape(-1)))(Yt)

    # d/dt grad_amb(R_Y(t W))|_{t=0}  (directional derivative along W).
    _, dgrad = jax.jvp(amb_grad_at, (0.0,), (1.0,))
    return _horizontal_project_psd(Y, dgrad)


# ===========================================================================
# Risk tcg-1 (blocker): HVP self-adjoint on the horizontal subspace.
# ===========================================================================
@pytest.mark.parametrize("k", [2, 3])
class TestHVPSelfAdjoint:
    r"""tcg / 'HVP projected only at output, never symmetrized'.

    Steihaug-Toint CG is valid only for a self-adjoint operator: conjugacy,
    the monotone model decrease, and the negative-curvature test
    ``d_Hd = <delta, H delta>`` all assume ``<u, H v> = <v, H u>``. An
    output-only projection ``P @ H_amb`` acquires a skew part at a
    non-critical ``Y`` (where ``S = sum r_i grad^2 r_i`` couples to the
    vertical space). We assert self-adjointness in the MANIFOLD metric on
    two random HORIZONTAL tangents at a non-critical ``Y`` -- the exact
    regime RTR targets. An output-only port fails this to ~1e-3, not 1e-10.
    """

    def test_hvp_symmetric_on_horizontal(self, k):
        manifold = PSDFixedRank(N, k)
        rng = np.random.default_rng(7 + k)
        # A NON-critical Y: residual target != triu(Y Y^T), so S != 0 and the
        # ambient Hessian does NOT preserve the horizontal subspace.
        Y = jnp.asarray(rng.normal(size=(N, k)))
        target = jnp.asarray(rng.normal(size=(_TRIU4.shape[0],)))
        residual_fn = _gauge_invariant_residual(target, k)
        params = _psd_params(Y, k)
        spec = manifold_spec_from_params(params)
        flat, _, _ = flatten_params_with_spec(params)

        # Two random HORIZONTAL tangents (projection idempotent).
        u_amb = jnp.asarray(rng.normal(size=(N, k)))
        v_amb = jnp.asarray(rng.normal(size=(N, k)))
        u = _horizontal_project_psd(Y, u_amb).reshape(-1)
        v = _horizontal_project_psd(Y, v_amb).reshape(-1)

        def hvp(eta_flat):
            return _riemannian_hvp(residual_fn, flat, spec, eta_flat)

        Hu = hvp(u)
        Hv = hvp(v)
        # Manifold inner products (Frobenius on a pure PSD leaf == jnp.sum,
        # but we route through the manifold to keep the contract honest).
        uHv = float(manifold.inner_product(Y, v.reshape(N, k), Hu.reshape(N, k)))
        vHu = float(manifold.inner_product(Y, u.reshape(N, k), Hv.reshape(N, k)))
        scale = max(abs(uHv), abs(vHu), 1.0)
        # Self-adjoint to float64 precision; a non-symmetric S-coupling at a
        # non-critical Y makes this O(1e-2 * scale).
        assert abs(uHv - vHu) < 1e-9 * scale
        # And it must NOT be vacuous: the operator is genuinely non-trivial
        # (the ambient Hessian S-term is active at this non-critical Y).
        assert abs(uHv) > 1e-6

    def test_negative_curvature_sign_matches_dense_hessian(self, k):
        r"""Companion: the sign of ``<w, H w>`` for the most-negative
        horizontal eigenvector must match the dense symmetrized horizontal
        Hessian's smallest eigenvalue. A non-self-adjoint HVP mis-signs the
        negative-curvature test -- the very feature RTR exists to deliver."""
        manifold = PSDFixedRank(N, k)
        rng = np.random.default_rng(31 + k)
        Y = jnp.asarray(rng.normal(size=(N, k)))
        # Indefinite S: target placed so triu(YY^T) - target straddles 0.
        target = (Y @ Y.T)[_TRIU4[:, 0], _TRIU4[:, 1]] + jnp.asarray(
            rng.normal(size=(_TRIU4.shape[0],))
        )
        residual_fn = _gauge_invariant_residual(target, k)
        params = _psd_params(Y, k)
        spec = manifold_spec_from_params(params)
        flat, _, _ = flatten_params_with_spec(params)

        def hvp(eta_flat):
            return _riemannian_hvp(residual_fn, flat, spec, eta_flat)

        # Dense horizontal Hessian via the src HVP on a horizontal basis.
        d = N * k
        cols = []
        basis = []
        for i in range(d):
            e = jnp.zeros((d,)).at[i].set(1.0)
            eh = _horizontal_project_psd(Y, e.reshape(N, k)).reshape(-1)
            basis.append(eh)
            cols.append(hvp(eh))
        Bm = jnp.stack(basis, axis=1)  # (d, d) horizontal-projected basis
        Hm = jnp.stack(cols, axis=1)  # (d, d) H applied to each basis col
        # Symmetrized dense operator restricted to the basis' column space.
        dense = Bm.T @ Hm
        dense = 0.5 * (dense + dense.T)
        evals, evecs = jnp.linalg.eigh(dense)
        # The most-negative eigenpair -> a negative-curvature direction.
        lam_min = float(evals[0])
        if lam_min >= -1e-6:
            pytest.skip("fixture did not realise a negative-curvature direction")
        coeff = evecs[:, 0]
        w = (Bm @ coeff).reshape(-1)  # horizontal tangent
        w = _horizontal_project_psd(Y, w.reshape(N, k)).reshape(-1)
        Hw = hvp(w)
        wHw = float(manifold.inner_product(Y, w.reshape(N, k), Hw.reshape(N, k)))
        # The curvature along the negative eigenvector is negative AND close
        # to lam_min * ||w||^2 (the Rayleigh quotient). A skew HVP mis-signs.
        ww = float(manifold.inner_product(Y, w.reshape(N, k), w.reshape(N, k)))
        assert wHw < 0.0
        assert wHw == pytest.approx(lam_min * ww, rel=1e-6, abs=1e-8)


# ===========================================================================
# Risk tcg-2 (blocker): per-iteration re-tangentialization (gauge-drift guard).
# ===========================================================================
class TestRetangentialization:
    r"""tcg / 'tCG never re-projects the CG direction -> vertical leakage
    amplifies in the Krylov space'.

    pymanopt re-tangentializes ``delta`` every inner iteration
    (``to_tangent_space``) as a deliberate guard against accumulated drift
    off the horizontal space. We track the VERTICAL energy of the running
    ``eta`` across inner iterations on a ``PSDFixedRank(4, 3)`` problem and
    require it to stay below ``1e-10`` of ``||eta||``. A no-reprojection
    port shows monotonically growing vertical energy.

    Vertical energy is measured by an independent reference projection: the
    vertical part of ``eta`` is ``eta - Proj_h(eta)``.
    """

    def _vertical_energy(self, Y: jnp.ndarray, eta_flat: jnp.ndarray, k: int) -> float:
        eta = eta_flat.reshape(N, k)
        horiz = _horizontal_project_psd(Y, eta)
        vert = eta - horiz
        denom = float(jnp.linalg.norm(eta)) + 1e-300
        return float(jnp.linalg.norm(vert)) / denom

    def test_vertical_energy_stays_below_tol_across_inner_iters(self):
        k = 3
        manifold = PSDFixedRank(N, k)
        rng = np.random.default_rng(123)
        # Y deliberately OFF the canonical gauge so jax.grad(Q) carries a
        # measurable skew Y^T g - g^T Y != 0 -- the seed has a vertical part
        # that a no-reprojection CG would amplify.
        Y = jnp.asarray(rng.normal(size=(N, k)))
        target = (Y @ Y.T)[_TRIU4[:, 0], _TRIU4[:, 1]] + 0.5 * jnp.asarray(
            rng.normal(size=(_TRIU4.shape[0],))
        )
        residual_fn = _gauge_invariant_residual(target, k)
        params = _psd_params(Y, k)
        spec = manifold_spec_from_params(params)
        flat, _, _ = flatten_params_with_spec(params)

        Q = _objective(residual_fn)
        grad_flat = jax.grad(Q)(flat)
        # Confirm the seed gradient genuinely has a skew (vertical) component
        # in the ambient frame -- otherwise this test would be vacuous.
        g = grad_flat.reshape(N, k)
        skew = Y.T @ g - g.T @ Y
        assert float(jnp.linalg.norm(skew)) > 1e-3

        def hvp(eta_flat):
            return _riemannian_hvp(residual_fn, flat, spec, eta_flat)

        # Force several inner iterations: large Delta, low residual target,
        # min_inner high enough that the rule does not quit at step 1.
        eta, _Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            flat,
            Delta=1e6,
            theta=1.0,
            kappa=1e-12,
            min_inner=N * k - manifold.gauge_dim,
            max_tcg_steps=N * k - manifold.gauge_dim,
        )
        # The returned eta must be (essentially) horizontal.
        assert self._vertical_energy(Y, eta, k) < 1e-10
        # And it actually ran multiple inner iterations (not a 1-step Cauchy).
        assert int(info.num_inner) >= 2
        assert not bool(jnp.isnan(eta).any())


# ===========================================================================
# Risk tcg-4 (high): d_Hd == 0 / NaN division guard.
# ===========================================================================
class TestZeroCurvatureGuard:
    r"""tcg / 'negative/zero-curvature branch must replicate pymanopt's
    d_Hd guards or eager JAX produces NaN steps'.

    pymanopt checks ``d_Hd != 0`` before ``alpha = z_r / d_Hd`` and routes
    ``d_Hd <= 0`` to the step-to-boundary branch BEFORE dividing. An eager
    port that divides unconditionally yields inf/NaN that propagates through
    ``eta`` and the retraction, silently NaN-ing the iterate ( rho compares
    false -> reject -> shrink forever: a silent stall).

    We construct an HVP with a KNOWN nullspace direction (exact zero
    curvature) and seed CG with it. We assert: no NaN/inf in ``eta`` or
    ``Heta``; the solver classifies it as a boundary/neg-curvature exit and
    steps to the boundary (so ``||eta|| == Delta`` in the manifold metric).
    """

    def test_zero_curvature_direction_no_nan_and_steps_to_boundary(self):
        k = 2
        manifold = PSDFixedRank(N, k)
        spec_params = _psd_params(jnp.eye(N, k), k)
        spec = manifold_spec_from_params(spec_params)
        d = N * k

        # Synthetic HVP with an exact 1-D nullspace: H = sum_{i>=1} lam_i v_i v_i^T,
        # lam_0 = 0. Built from an orthonormal basis; v_0 is the null vector.
        rng = np.random.default_rng(5)
        Araw = jnp.asarray(rng.normal(size=(d, d)))
        Qb, _ = jnp.linalg.qr(Araw)
        lams = jnp.asarray(np.concatenate([[0.0], rng.uniform(0.5, 2.0, size=d - 1)]))
        Hmat = (Qb * lams) @ Qb.T  # symmetric, PSD, exact null along Qb[:,0]
        null_vec = Qb[:, 0]

        def hvp(eta_flat):
            return Hmat @ eta_flat

        # Seed: gradient ANTIPARALLEL to the null direction so the first CG
        # delta IS the zero-curvature direction (d_Hd == 0 exactly).
        grad_flat = -null_vec
        point = jnp.eye(N, k).reshape(-1)  # used only for the metric (Frobenius)

        Delta = 0.5
        eta, Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            point,
            Delta=Delta,
            theta=1.0,
            kappa=0.1,
            min_inner=1,
            max_tcg_steps=d,
        )
        # (a) no NaN/inf anywhere.
        assert not bool(jnp.isnan(eta).any())
        assert not bool(jnp.isinf(eta).any())
        assert not bool(jnp.isnan(Heta).any())
        # (b) zero curvature is treated as a boundary/neg-curvature exit, NOT
        # an interior residual-target stop.
        assert str(info.stop_reason) in ("negative_curvature", "exceeded_tr")
        # (c) stepped to the trust-region boundary in the MANIFOLD metric.
        eta_norm = float(manifold.norm(point.reshape(N, k), eta.reshape(N, k)))
        assert eta_norm == pytest.approx(Delta, rel=1e-8, abs=1e-10)

    def test_tiny_curvature_does_not_blow_up(self):
        r"""Fuzz the branch order: ``d_Hd`` seeded ~1e-300 must NOT produce a
        huge ``alpha = z_r/d_Hd`` step (pymanopt routes ``d_Hd <= 0`` to the
        boundary branch first; a tiny-but-positive value is interior but the
        guard must keep ``e_Pe_new`` finite). We assert finiteness only."""
        k = 2
        spec = manifold_spec_from_params(_psd_params(jnp.eye(N, k), k))
        d = N * k
        rng = np.random.default_rng(9)
        Qb, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(d, d))))
        lams = jnp.asarray(
            np.concatenate([[1e-300], rng.uniform(0.5, 2.0, size=d - 1)])
        )
        Hmat = (Qb * lams) @ Qb.T

        def hvp(eta_flat):
            return Hmat @ eta_flat

        grad_flat = -Qb[:, 0]
        eta, Heta, _info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            jnp.eye(N, k).reshape(-1),
            Delta=1.0,
            theta=1.0,
            kappa=0.1,
            min_inner=1,
            max_tcg_steps=d,
        )
        assert bool(jnp.all(jnp.isfinite(eta)))
        assert bool(jnp.all(jnp.isfinite(Heta)))


# ===========================================================================
# Risk tcg-3 (high): inner products use the MANIFOLD metric (mixed Product).
# ===========================================================================
class TestManifoldMetricBoundary:
    r"""tcg / 'tau solve and e_Pe boundary test must use manifold
    inner_product, not Euclidean sums, on mixed-metric Products'.

    A pure-PSD leaf has the Frobenius (identity) metric, so a naive
    ``jnp.sum`` happens to be correct -- the trap a PSD-only fixture cannot
    catch. On ``Product(PSDFixedRank(4, k), Positive())`` with a SMALL ``s``,
    the Positive leaf carries the ``1/s^2`` metric (``>> 1``). The boundary
    hit ``eta`` must satisfy ``norm_manifold(eta) == Delta`` -- NOT the
    Frobenius norm. A Euclidean-sum port reports a Frobenius norm at the
    boundary and mis-scales the Positive block.

    We use a synthetic HVP whose only descent/curvature is on the Positive
    leaf, forcing the step to live there where the metric matters, and pick
    ``Delta`` small enough that CG hits the trust boundary.
    """

    @pytest.mark.parametrize("k", [2])
    def test_boundary_eta_has_manifold_norm_equal_delta(self, k):
        s_val = 0.05  # 1/s^2 = 400 >> 1: metric mismatch is loud here
        params = _psd_pos_params(jnp.eye(N, k), s_val, k)
        manifold_prod = None  # built below from the spec's factors if needed
        spec = manifold_spec_from_params(params)
        point_flat, _, _ = flatten_params_with_spec(params)
        d = int(point_flat.shape[0])  # N*k + 1

        # Indices: Y block = [0, N*k), s scalar at N*k.
        s_idx = N * k

        # Synthetic NEGATIVE-curvature HVP isolated to the s leaf so the step
        # is driven onto the Positive block. (Negative curvature -> immediate
        # step-to-boundary; the tau solve then runs in the manifold metric.)
        Hmat = jnp.zeros((d, d)).at[s_idx, s_idx].set(-1.0)

        def hvp(eta_flat):
            return Hmat @ eta_flat

        # Gradient pushes along the s direction (ambient).
        grad_flat = jnp.zeros((d,)).at[s_idx].set(1.0)

        Delta = 0.3
        eta, _Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            point_flat,
            Delta=Delta,
            theta=1.0,
            kappa=0.1,
            min_inner=1,
            max_tcg_steps=d,
        )
        assert str(info.stop_reason) in ("negative_curvature", "exceeded_tr")

        # MANIFOLD norm of eta == Delta. Build the per-leaf metric explicitly
        # from the spec so the reference is independent of the src tCG.
        eta_Y = eta[:s_idx].reshape(N, k)
        eta_s = eta[s_idx]
        # Frobenius part (PSD leaf) + (1/s^2) * eta_s^2 (Positive leaf).
        man_sq = float(jnp.sum(eta_Y * eta_Y)) + float(eta_s**2) / (s_val**2)
        man_norm = float(jnp.sqrt(man_sq))
        assert man_norm == pytest.approx(Delta, rel=1e-8, abs=1e-10)

        # And it is NOT the Frobenius norm (which would be the bug's value):
        frob = float(jnp.sqrt(jnp.sum(eta * eta)))
        # With a step on the s leaf and 1/s^2 = 400, the manifold norm and
        # Frobenius norm must differ by a large factor -> non-vacuous.
        assert abs(frob - Delta) > 1e-3
        del manifold_prod

    def test_pure_psd_metric_is_frobenius(self):
        r"""Control: on a pure PSD leaf the manifold metric IS Frobenius, so
        the boundary eta's manifold norm and Frobenius norm coincide. This
        rules out a 'metric-weighting always applied' over-correction."""
        k = 2
        spec = manifold_spec_from_params(_psd_params(jnp.eye(N, k), k))
        d = N * k
        Hmat = jnp.zeros((d, d)).at[0, 0].set(-1.0)  # neg curvature on coord 0

        def hvp(eta_flat):
            return Hmat @ eta_flat

        grad_flat = jnp.zeros((d,)).at[0].set(1.0)
        Delta = 0.4
        eta, _Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            jnp.eye(N, k).reshape(-1),
            Delta=Delta,
            theta=1.0,
            kappa=0.1,
            min_inner=1,
            max_tcg_steps=d,
        )
        assert str(info.stop_reason) in ("negative_curvature", "exceeded_tr")
        frob = float(jnp.sqrt(jnp.sum(eta * eta)))
        assert frob == pytest.approx(Delta, rel=1e-8, abs=1e-10)


# ===========================================================================
# Risk tcg-5 (medium): two-regime stopping rule reproduces pymanopt counts.
# ===========================================================================
class TestStoppingRuleParityWithPymanopt:
    r"""tcg / 'residual-tolerance termination and the kappa/theta superlinear
    rule must be ported exactly, not approximated by a single rtol'.

    pymanopt's inner stop is ``norm_r <= norm_r0 * min(norm_r0**theta,
    kappa)`` gated by ``j >= mininner``, with the LINEAR (kappa) vs
    SUPERLINEAR (theta) classification. Collapsing to a flat
    ``norm_r < rtol*||r0||`` changes the inner-iteration count and the
    classification. We run BOTH our ``_truncated_cg`` and pymanopt's
    ``_truncated_conjugate_gradient`` on the SAME symmetric PD operator with
    identical ``Delta/theta/kappa/mininner`` and require:

      * the inner-iteration count agrees within +/-1, AND
      * the linear-vs-superlinear classification agrees.

    Import-gated: pymanopt is dev-only.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_inner_counts_and_classification_match_pymanopt(self, seed):
        pytest.importorskip("pymanopt")
        from pymanopt.manifolds import Euclidean as PymEuclidean
        from pymanopt.optimizers.trust_regions import TrustRegions

        k = 2
        d = N * k
        # Pure Euclidean problem of the same dimension: the inner solve is a
        # plain Steihaug-Toint CG, so pymanopt's Euclidean manifold is the
        # exact reference (identity metric, identity to_tangent_space). This
        # isolates the STOPPING RULE from the gauge/metric machinery.
        rng = np.random.default_rng(seed)
        Qb, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(d, d))))
        lams = jnp.asarray(rng.uniform(0.5, 5.0, size=d))  # SPD, well-conditioned
        Hmat = (Qb * lams) @ Qb.T
        grad_np = np.asarray(rng.normal(size=(d,)))
        grad_flat = jnp.asarray(grad_np)

        def hvp(eta_flat):
            return Hmat @ eta_flat

        theta, kappa, min_inner = 1.0, 0.1, 1
        Delta = 1e6  # huge: force the residual-target stop, not the boundary

        spec = manifold_spec_from_params(_euclidean_params(jnp.zeros((d,))))
        _eta, _Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            jnp.zeros((d,)),
            Delta=Delta,
            theta=theta,
            kappa=kappa,
            min_inner=min_inner,
            max_tcg_steps=2 * d,
        )

        # --- pymanopt reference inner solve on a matching problem. ----------
        pym_manifold = PymEuclidean(d)
        Hmat_np = np.asarray(Hmat)

        class _Prob:
            manifold = pym_manifold

            @staticmethod
            def riemannian_hessian(x, u):  # noqa: ARG004
                return Hmat_np @ np.asarray(u)

            @staticmethod
            def preconditioner(x, u):  # noqa: ARG004
                return np.asarray(u)

        tr = TrustRegions(verbosity=0)
        tr.use_rand = False
        x0 = np.zeros((d,))
        eta0 = pym_manifold.zero_vector(x0)
        _eta_p, _Heta_p, j_pym, stop_pym = tr._truncated_conjugate_gradient(
            _Prob,
            x0,
            grad_np,
            eta0,
            Delta,
            theta,
            kappa,
            min_inner,
            2 * d,
        )

        # pymanopt returns j == last index; num_inner is the count run. Match
        # within +/-1 (off-by-one conventions on the loop index).
        assert abs(int(info.num_inner) - int(j_pym)) <= 1

        # Classification parity: linear vs superlinear.
        linear = stop_pym == tr.REACHED_TARGET_LINEAR
        superlinear = stop_pym == tr.REACHED_TARGET_SUPERLINEAR
        if linear:
            assert str(info.stop_reason) == "reached_target_linear"
        elif superlinear:
            assert str(info.stop_reason) == "reached_target_superlinear"
        else:  # pragma: no cover - fixture is tuned to hit a residual stop
            pytest.skip(f"pymanopt stopped on {stop_pym!r}, not a residual target")

    def test_min_inner_is_enforced(self):
        r"""``min_inner`` must gate the residual-target stop: with
        ``min_inner = m``, tCG runs at least ``m`` inner iterations even if
        the residual is already tiny. A port that omits the ``j >= mininner``
        gate would quit at iteration 1 (near-Cauchy step)."""
        k = 2
        d = N * k
        rng = np.random.default_rng(17)
        Qb, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(d, d))))
        lams = jnp.asarray(rng.uniform(0.8, 1.2, size=d))  # near-identity SPD
        Hmat = (Qb * lams) @ Qb.T

        def hvp(eta_flat):
            return Hmat @ eta_flat

        grad_flat = jnp.asarray(rng.normal(size=(d,)))
        spec = manifold_spec_from_params(_euclidean_params(jnp.zeros((d,))))
        m = 3
        _eta, _Heta, info = _truncated_cg(
            hvp,
            grad_flat,
            spec,
            jnp.zeros((d,)),
            Delta=1e6,
            theta=1.0,
            kappa=0.5,  # loose: residual target reachable early
            min_inner=m,
            max_tcg_steps=2 * d,
        )
        assert int(info.num_inner) >= m


@jdc.pytree_dataclass
class _EucParams:
    v: ManifoldLeaf


def _euclidean_params(v: jnp.ndarray) -> _EucParams:
    v = jnp.asarray(v)
    return _EucParams(v=ManifoldLeaf(v, Euclidean(int(v.shape[0]))))
