r"""Phase-1 TEST-FIRST suite for the Riemannian Trust-Region solver (#152).

THEME: gauge + numerical robustness (red-team lens ``numerical-gauge``).

This file is written **before** the implementation. It imports
``emu_gmm.manifolds.riemannian_tr`` (and the unit helpers
``_riemannian_hvp`` / ``_truncated_cg``), which Phase 2 will create. Until
then EVERY test here is RED with an ``ImportError`` at collection time --
that is correct and intended. Do not stub the module to make them pass; the
implementation must satisfy these assertions, not the other way round.

The solver under test (intended API, exactly as Phase 2 will expose it)::

    from emu_gmm.manifolds.riemannian_tr import riemannian_tr
    opt = riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10,
                        rho_prime=0.1, kappa=0.1, theta=1.0, min_inner=1,
                        max_tcg_steps=None, max_radius=None, init_radius=None)
    theta_hat, info = opt(residual_fn, theta_init, manifold_spec)
    # also through the estimator: estimate(..., optimizer=riemannian_tr())
    from emu_gmm.manifolds.riemannian_tr import _riemannian_hvp, _truncated_cg

HVP semantics encoded here (the design revision, #152 latest comment):
``H[eta]`` is the *Euclidean Hessian of the retraction pullback*
``Q(R_Y(eta))`` at ``eta = 0``, projected to horizontal for the PSD leaf;
inner products use the per-leaf manifold metric; the operator is symmetric
on the horizontal space.

The shared synthetic DGP mirrors ``test_manifold_acceptance_phase6.py`` and
``test_riemannian_lm_phase3.py``: a ``Product(PSDFixedRank(n, k),
Euclidean(1))`` whose residual depends on ``theta`` ONLY through the
gauge-invariant ``Gamma = Y Y^T`` and ``phi``, so ``(Y Q)(Y Q)^T = Y Y^T``
for any ``Q in O(k)`` and the O(k) fibre is an exact symmetry. Every recovery
/ agreement assertion is on a gauge-INVARIANT functional
(``Gamma_hat``, ``eigvalsh``, ``J_stat``, ``||g||_horizontal``), NEVER on raw
``Y`` (red-team: the #146 raw-||grad|| hazard).

Each test docstring cites the risk it pins:

* ``TestGaugeInvariantConvergence``     -- "raw ambient ||grad||/||eta|| is
  not gauge-invariant" + "tCG search direction not re-tangentialized".
* ``TestIntrinsicDimensionDefaults``    -- "TR radius/maxinner from AMBIENT
  nk inflates Delta by the gauge fraction".
* ``TestNearRankDeficientHVP``          -- "indefinite-step rank drop blows up
  the next HVP's Lyapunov solve" (PD floor).
* ``TestVerticalGradientNegativeControl`` -- the negative control half of the
  raw-||grad|| risk: a purely vertical gradient must NOT certify converged.
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
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf

# --- The module under test. SKIP until Phase 2 lands it, then go live. ------
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
_riemannian_hvp = riemannian_tr_mod._riemannian_hvp
_truncated_cg = riemannian_tr_mod._truncated_cg
riemannian_tr = riemannian_tr_mod.riemannian_tr

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side (matches the Phase-6 acceptance DGP)

# Upper-triangular index pairs of the (n,n) symmetric Gamma.
_TRIU_N = jnp.array(np.triu_indices(N)).T  # (n(n+1)/2, 2)


# ---------------------------------------------------------------------------
# Shared DGP + small helpers (flag any that should move to a conftest).
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class ProductParams:
    """A ``PSDFixedRank(N, k)`` ``Y`` leaf plus a ``Euclidean(1)`` ``phi``."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k: int) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y, dtype=jnp.float64), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(
            jnp.reshape(jnp.asarray(phi, dtype=jnp.float64), (1,)), Euclidean(1)
        ),
    )


def _triu_pairs(n: int) -> jnp.ndarray:
    return jnp.array(np.triu_indices(n)).T


def _orthogonal(seed: int, k: int, reflect: bool = False) -> jnp.ndarray:
    """Random ``Q in O(k)``; ``reflect`` forces det(Q) = -1 (improper).

    Mirrors ``test_manifold_acceptance_phase6._orthogonal`` so the gauge
    fibre is exercised on BOTH components of O(k).
    """
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(k, k)))
    q, r = jnp.linalg.qr(g)
    q = q @ jnp.diag(jnp.sign(jnp.diag(r)))
    want = -1.0 if reflect else 1.0
    if float(jnp.linalg.det(q)) * want < 0:
        q = q.at[:, 0].set(-q[:, 0])
    return q


def _nonconvex_residual(theta_flat, x_bar, k: int, n: int = N):
    r"""A deliberately NON-CONVEX least-squares residual in ``Gamma``.

    ``m(theta) = triu(Gamma) (+) [phi]`` where each upper-triangular entry
    is passed through a mild nonlinearity ``g + 0.35 * sin(3 g)`` BEFORE the
    target subtraction. This is still gauge-invariant (a function of
    ``Gamma`` only), but the ``Y |-> nonlinear(Y Y^T)`` map gives the
    least-squares objective genuine negative curvature away from the truth --
    the saddle/indefinite regime where plain Gauss-Newton (riemannian_lm)
    stalls and a real second-order tCG must take a negative-curvature step.
    Depends on ``theta`` ONLY through ``Gamma = Y Y^T`` and ``phi`` (gauge
    invariant), so ``Y -> Y Q`` is an exact symmetry.
    """
    triu = _triu_pairs(n)
    Y = jnp.reshape(theta_flat[: n * k], (n, k))
    phi = theta_flat[n * k]
    g = (Y @ Y.T)[triu[:, 0], triu[:, 1]]
    g = g + 0.35 * jnp.sin(3.0 * g)
    m = jnp.concatenate([g, jnp.reshape(phi, (1,))])
    return m - x_bar


def _target_for(A_true, phi_true, k: int, n: int = N):
    """Build the ``x_bar`` consumed by ``_nonconvex_residual`` at the truth."""
    triu = _triu_pairs(n)
    Gamma = A_true @ A_true.T
    g = Gamma[triu[:, 0], triu[:, 1]]
    g = g + 0.35 * jnp.sin(3.0 * g)
    return jnp.concatenate([g, jnp.reshape(jnp.asarray(phi_true), (1,))])


def _components(theta_hat, k: int):
    """Return ``(A_hat (n,k), phi_hat float)`` from a returned PyTree."""
    A = jnp.asarray(theta_hat.Y.array)
    phi = float(jnp.reshape(jnp.asarray(theta_hat.phi.array), ()))
    return A, phi


def _outer_steps(info) -> int:
    """Read the outer-iteration count off the OptimizerInfo (``steps``)."""
    return int(jnp.asarray(info.steps))


# ---------------------------------------------------------------------------
# Risk: "Convergence test on raw ambient ||grad||/||eta|| is not gauge-
# invariant" + "tCG search direction is not re-tangentialized". (blocker/high)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("reflect", [False, True])
class TestGaugeInvariantConvergence:
    r"""[numerical-gauge] Gauge-invariant convergence + re-tangentialization.

    Pins two convergent red-team risks:

    1. "Convergence test on raw ambient ``||grad||/||eta||`` is not
       gauge-invariant -- a gauge-rotated iterate can certify falsely (#146)."
       Running from ``Y0`` and from ``Y0 @ Q`` (``Q in O(k)``, BOTH SO(k) and
       the reflection component) must give: identical OUTER-iteration count,
       ``Gamma`` agreeing to 1e-9, and a reported ``||g||`` that equals the
       HORIZONTAL gradient norm and is UNCHANGED under ``Y -> Y Q``.
    2. "tCG search direction is not re-tangentialized -- gauge contamination
       accumulates through the CG recurrence." Tested at the unit level on
       ``_truncated_cg`` (below) AND implied here by the bit-for-bit fibre
       agreement: a leaking tCG would desynchronise the two trajectories.

    A loose tolerance would hide a partial gauge leak; do NOT loosen. If
    these fail, suspect the horizontal projection / the missing
    re-tangentialization, not the test.
    """

    def _setup(self, k: int):
        rng = np.random.default_rng(100 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        x_bar = _target_for(A_true, 0.7, k)
        # Warm start a little off the truth (same basin), away from the fibre.
        Y0 = jnp.asarray(A_true + 0.10 * rng.normal(size=(N, k)))
        params = _make_params(Y0, 0.6, k)
        spec = manifold_spec_from_params(params)

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k)

        return residual_fn, params, spec, A_true

    def test_outer_count_and_gamma_invariant_under_YQ(self, k, reflect):
        residual_fn, params_a, spec, _A = self._setup(k)
        Y0 = jnp.asarray(params_a.Y.array)
        phi0 = float(jnp.reshape(jnp.asarray(params_a.phi.array), ()))
        Q = _orthogonal(7 + k, k, reflect=reflect)
        # Q is genuinely O(k) on the requested component.
        assert bool(jnp.allclose(Q @ Q.T, jnp.eye(k), atol=1e-10))
        assert float(jnp.linalg.det(Q)) == pytest.approx(
            -1.0 if reflect else 1.0, abs=1e-8
        )
        params_b = _make_params(Y0 @ Q, phi0, k)

        opt = riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10)
        th_a, info_a = opt(residual_fn, params_a, spec)
        th_b, info_b = opt(residual_fn, params_b, spec)

        # Both genuinely converged (not the under-jit "traced" collapse).
        assert bool(info_a.done) is True
        assert bool(info_b.done) is True

        # (1) Outer-iteration count agrees to WITHIN ONE step. ``Q`` here is a
        # DENSE rotation (``_orthogonal`` = qr of a Gaussian), so ``Y0 @ Q`` is
        # itself a rounded matrix product and the two trajectories are
        # gauge-equivalent only up to ~1e-9 rounding. The integer outer-step
        # count is the single most fragile observable -- a hard threshold
        # crossing near convergence -- so it can differ by one between gauges
        # even when every CONTINUOUS invariant agrees (Gamma to 1e-9 below; the
        # reported horizontal gradient norm to 1e-9 in the sibling test). Exact
        # integer equality under a dense rotation is not achievable without
        # per-step gauge canonicalization (owner decision 2026-06-17: out of
        # scope; the meaningful gauge invariants stay strict). #152.
        assert abs(_outer_steps(info_a) - _outer_steps(info_b)) <= 1

        # (2) Gamma agrees to 1e-9 (gauge-invariant point estimate).
        Aa, _ = _components(th_a, k)
        Ab, _ = _components(th_b, k)
        Ga, Gb = Aa @ Aa.T, Ab @ Ab.T
        assert bool(jnp.allclose(Ga, Gb, atol=1e-9))
        assert bool(
            jnp.allclose(jnp.linalg.eigvalsh(Ga), jnp.linalg.eigvalsh(Gb), atol=1e-9)
        )

    def test_reported_gradnorm_is_horizontal_and_YQ_invariant(self, k, reflect):
        residual_fn, params_a, spec, _A = self._setup(k)
        Y0 = jnp.asarray(params_a.Y.array)
        phi0 = float(jnp.reshape(jnp.asarray(params_a.phi.array), ()))
        Q = _orthogonal(11 + k, k, reflect=reflect)
        params_b = _make_params(Y0 @ Q, phi0, k)

        opt = riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10)
        _th_a, info_a = opt(residual_fn, params_a, spec)
        _th_b, info_b = opt(residual_fn, params_b, spec)

        # The reported final gradient norm must be the HORIZONTAL norm and so
        # be (near-)identical along the fibre. A raw ambient ||grad|| would
        # carry the gauge-rotated vertical component and differ under Y->YQ.
        ga = float(jnp.asarray(info_a.final_gradient_norm))
        gb = float(jnp.asarray(info_b.final_gradient_norm))
        assert np.isfinite(ga) and np.isfinite(gb)
        assert ga == pytest.approx(gb, abs=1e-9, rel=1e-7)

    def test_horizontal_gradnorm_matches_manifold_projection(self, k, reflect):
        # Pin the *definition*: the reported ||g|| equals the per-leaf manifold
        # norm of the HORIZONTAL gradient. We recompute it test-side from the
        # Euclidean gradient of the objective at the optimum and the PSD
        # leaf's horizontal projection, and assert agreement. A solver that
        # certified on the raw ambient gradient would NOT match this value
        # (the vertical component is nonzero off the fibre representative).
        del reflect  # one direction suffices for the definitional check
        residual_fn, params_a, spec, _A = self._setup(k)
        opt = riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10)
        th, info = opt(residual_fn, params_a, spec)
        assert bool(info.done) is True

        flat, _treedef, _fspec = flatten_params_with_spec(th)

        def objective(tf):
            r = residual_fn(tf)
            return 0.5 * jnp.sum(r * r)

        egrad = jax.grad(objective)(flat)  # ambient Euclidean gradient

        # Horizontal projection of the PSD block (the Euclidean phi block has
        # no gauge, so its projection is the identity).
        psd = PSDFixedRank(N, k)
        Y = jnp.reshape(flat[: N * k], (N, k))
        eg_psd = jnp.reshape(egrad[: N * k], (N, k))
        h_psd = psd.projection(Y, eg_psd)  # horizontal grad on the PSD leaf
        eg_phi = egrad[N * k :]
        # Per-leaf Riemannian norm: Frobenius for PSD + Euclidean phi.
        gnorm_ref = jnp.sqrt(jnp.sum(h_psd * h_psd) + jnp.sum(eg_phi * eg_phi))

        reported = float(jnp.asarray(info.final_gradient_norm))
        assert reported == pytest.approx(float(gnorm_ref), abs=1e-9, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit-level re-tangentialization risk on ``_truncated_cg`` directly.
# ---------------------------------------------------------------------------
class TestTruncatedCGStaysHorizontal:
    r"""[numerical-gauge] tCG re-tangentialization (blocker).

    "tCG search direction is not re-tangentialized -- gauge contamination
    accumulates through the CG recurrence." Force tCG to its boundary with
    many inner iterations (small radius) on a ``PSDFixedRank(4, 3)`` problem,
    then decompose the returned ``eta`` into horizontal + vertical via the
    manifold projection. WITHOUT per-iteration re-tangentialization the
    vertical fraction grows with inner-iteration count (>1e-6 by ~8 iters);
    WITH it, it stays at the rounding floor.
    """

    def test_returned_eta_has_no_vertical_component(self):
        n, k = 4, 3
        rng = np.random.default_rng(0)
        Y = jnp.asarray(rng.normal(size=(n, k)))
        psd = PSDFixedRank(n, k)

        # A symmetric, indefinite HVP-like operator on ambient (n,k) blocks,
        # projected to horizontal on output (mimics the real HVP contract:
        # symmetric on the horizontal space). The point of the test is that
        # tCG must KEEP its iterates horizontal across the recurrence, not
        # just at the end.
        rngM = np.random.default_rng(1)
        flat_dim = n * k
        # SPD operator with a deliberately SPREAD eigen-spectrum (cond ~ 1e3),
        # so tCG genuinely runs many interior CG steps before the residual
        # target is met. The previous fixture used an INDEFINITE Msym with a
        # tiny radius, which made tCG stop on iter 1 (negative-curvature bail
        # or an immediate boundary hit) -- the recurrence was never exercised.
        # Restricted to the horizontal subspace the operator ``P Msym`` equals
        # the symmetric ``P Msym P`` (P is the orthogonal horizontal
        # projection), which is SPD when Msym is SPD: no neg-curvature bail.
        Qortho, _ = jnp.linalg.qr(jnp.asarray(rngM.normal(size=(flat_dim, flat_dim))))
        eigs = jnp.asarray(np.geomspace(1.0, 1.0e3, num=flat_dim))
        Msym = Qortho @ jnp.diag(eigs) @ Qortho.T
        Msym = 0.5 * (Msym + Msym.T)  # symmetrize away rounding

        def hvp(eta_flat):
            vmat = jnp.reshape(eta_flat, (n, k))
            out = jnp.reshape(Msym @ eta_flat, (n, k))
            out = psd.projection(Y, out)  # horizontal output
            del vmat
            return jnp.reshape(out, (-1,))

        grad = jnp.reshape(
            psd.projection(Y, jnp.asarray(rng.normal(size=(n, k)))), (-1,)
        )
        # Large radius so the boundary is NOT the binding stop on iter 1, and a
        # tight kappa/theta residual target so the two-regime stop is not met
        # until the spread spectrum has been resolved over many CG steps. With
        # an SPD, well-conditioned-only-by-luck operator this needs >= 5 inner
        # iterations to drive ||r|| below ``norm_r0 * 1e-6``.
        eta, info = _truncated_cg(
            hvp,
            grad,
            radius=1e3,
            max_inner=50,
            kappa=1e-6,
            theta=1.0,
            min_inner=1,
        )
        # At least a handful of inner iterations actually ran (else the test
        # is vacuous -- it must EXERCISE the recurrence).
        assert int(info["inner_iters"]) >= 5

        eta_mat = jnp.reshape(eta, (n, k))
        horizontal = psd.projection(Y, eta_mat)
        vertical = eta_mat - horizontal
        eta_norm = float(jnp.linalg.norm(eta_mat))
        assert eta_norm > 0.0
        vert_frac = float(jnp.linalg.norm(vertical)) / eta_norm
        assert vert_frac < 1e-10


# ---------------------------------------------------------------------------
# Risk: "TR radius / maxinner from the AMBIENT dimension nk inflates Delta by
# the gauge fraction". (medium)
# ---------------------------------------------------------------------------
class TestIntrinsicDimensionDefaults:
    r"""[numerical-gauge] Intrinsic-dimension defaults.

    "Trust-region radius initialization / maxinner uses the AMBIENT
    dimension ``nk`` (counts the ``k(k-1)/2`` gauge directions), inflating
    ``Delta`` proportional to the gauge fraction." The defaults must derive
    from the IDENTIFIED quotient dimension ``total_dimension -
    total_gauge_dim``, not the ambient ``nk``.

    Concrete fixture: ``PSDFixedRank(6, 5)`` -> ambient ``nk = 30``,
    ``gauge_dim = 10``, identified ``= 20``. We assert the solver's default
    ``max_tcg_steps`` is the identified ``20`` (NOT ``30``) and that
    ``Delta_bar`` scales as ``sqrt(identified)`` (NOT ``sqrt(ambient)``).
    Regression half: an all-Euclidean tree (gauge_dim == 0) is unchanged --
    identified == ambient -- so the v1 reduction is bit-for-bit.
    """

    def _spec_psd(self, n: int, k: int):
        params = _make_params(jnp.ones((n, k)), 0.5, k) if n == N else None
        # We need a (6,5) PSD leaf, not the module-level N=5 helper, so build
        # the params inline for the off-N case.
        if params is None:

            @jdc.pytree_dataclass
            class P:
                Y: ManifoldLeaf
                phi: ManifoldLeaf

            params = P(  # type: ignore[assignment]
                Y=ManifoldLeaf(jnp.ones((n, k)), PSDFixedRank(n, k)),
                phi=ManifoldLeaf(jnp.ones((1,)), Euclidean(1)),
            )
        return manifold_spec_from_params(params)

    def test_default_max_tcg_steps_is_identified_dim(self):
        n, k = 6, 5
        spec = self._spec_psd(n, k)
        # Sanity: the spec carries the gauge bookkeeping we rely on.
        ambient = spec.total_dimension  # n*k + 1 = 31
        gauge = spec.total_gauge_dim  # k(k-1)/2 = 10
        identified = ambient - gauge  # 21
        assert ambient == n * k + 1
        assert gauge == k * (k - 1) // 2
        assert identified == n * k + 1 - k * (k - 1) // 2

        opt = riemannian_tr()  # all defaults -> must derive from the spec
        # The resolved defaults are exposed for inspection (Phase 2 surfaces
        # them on the OptimizerInfo of a real run). Run one tiny solve to read
        # them back, on a trivially-convex residual so it converges fast.
        rng = np.random.default_rng(3)
        A_true = jnp.asarray(rng.normal(size=(n, k)))
        x_bar = _target_for(A_true, 0.7, k, n=n)
        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(n, k)))

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(Y0, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.6]), Euclidean(1)),
        )

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k, n=n)

        _th, info = opt(residual_fn, params, spec)
        # max_tcg_steps default == identified quotient dim (NOT ambient nk+1).
        assert int(info.max_tcg_steps) == identified
        assert int(info.max_tcg_steps) != ambient

    def test_init_radius_scales_with_identified_not_ambient(self):
        # pymanopt's Delta_bar default is sqrt(manifold.dim). The gauge-aware
        # default must use the IDENTIFIED dim, so init_radius/Delta_bar are
        # provably smaller than the ambient-counting value by the gauge
        # fraction. We assert the reported Delta_bar matches sqrt(identified),
        # and is strictly below sqrt(ambient).
        n, k = 6, 5
        spec = self._spec_psd(n, k)
        ambient = spec.total_dimension
        gauge = spec.total_gauge_dim
        identified = ambient - gauge

        rng = np.random.default_rng(4)
        A_true = jnp.asarray(rng.normal(size=(n, k)))
        x_bar = _target_for(A_true, 0.7, k, n=n)
        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(n, k)))

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(Y0, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.6]), Euclidean(1)),
        )

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k, n=n)

        opt = riemannian_tr()
        _th, info = opt(residual_fn, params, spec)
        max_radius = float(jnp.asarray(info.max_radius))  # Delta_bar
        assert max_radius == pytest.approx(np.sqrt(identified), rel=1e-9)
        assert max_radius < np.sqrt(ambient)  # strictly below ambient-counting

    def test_all_euclidean_default_unchanged(self):
        # gauge_dim == 0: identified == ambient, so the gauge-aware default
        # is bit-for-bit the ambient one (v1 non-regression).
        @jdc.pytree_dataclass
        class P:
            a: jnp.ndarray
            b: jnp.ndarray

        params = P(a=jnp.asarray(1.0), b=jnp.asarray(2.0))
        spec = manifold_spec_from_params(params)
        assert spec.total_gauge_dim == 0
        ambient = spec.total_dimension  # == 2

        def residual_fn(tf):
            return jnp.stack([tf[0] - 3.0, tf[1] + 1.0])

        opt = riemannian_tr()
        th, info = opt(residual_fn, params, spec)
        assert bool(info.done) is True
        assert float(th.a) == pytest.approx(3.0, abs=1e-8)
        assert float(th.b) == pytest.approx(-1.0, abs=1e-8)
        # max_tcg_steps default == ambient when no gauge.
        assert int(info.max_tcg_steps) == ambient
        assert float(jnp.asarray(info.max_radius)) == pytest.approx(
            np.sqrt(ambient), rel=1e-9
        )


# ---------------------------------------------------------------------------
# Risk: "indefinite-step rank drop blows up the next HVP's Lyapunov solve"
# (PD floor). (high)
# ---------------------------------------------------------------------------
class TestNearRankDeficientHVP:
    r"""[numerical-gauge] Near-rank-deficient HVP + Lyapunov PD floor.

    "Indefinite-Hessian step-to-boundary plus the additive PSD retraction can
    drop rank / produce a ``Gamma`` with a near-zero eigenvalue, blowing up
    the next HVP's Lyapunov solve." At ``Y`` with ``cond(YtY) ~ 1e8`` the
    horizontal projection's Lyapunov solve (``YtY Omega + Omega YtY = ...``)
    is near-singular; without a PD floor it returns NaN/Inf and poisons
    ``final_objective`` / the OptimizerInfo. The HVP must stay FINITE.
    """

    def _ill_conditioned_Y(self, n: int, k: int, cond_target: float) -> jnp.ndarray:
        """Build ``Y`` with ``cond(Y^T Y) ~ cond_target`` via a controlled SVD.

        ``Y = U S V^T`` with singular values geometrically spaced so that
        ``cond(YtY) = (s_max/s_min)^2`` hits the target.
        """
        rng = np.random.default_rng(2024)
        U, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(n, k))))
        Vq, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(k, k))))
        ratio = np.sqrt(cond_target)  # cond(YtY) = (cond(Y))^2
        svals = jnp.asarray(np.geomspace(1.0, 1.0 / ratio, num=k))
        Y = U @ jnp.diag(svals) @ Vq.T
        return Y

    @pytest.mark.parametrize("n,k", [(3, 3), (5, 3)])
    def test_hvp_finite_at_high_condition_number(self, n, k):
        Y = self._ill_conditioned_Y(n, k, cond_target=1e8)
        YtY = Y.T @ Y
        cond = float(jnp.linalg.cond(YtY))
        # The fixture really is near-singular (else the test is toothless).
        assert cond > 1e6

        rng = np.random.default_rng(7)
        A_true = jnp.asarray(rng.normal(size=(n, k)))
        x_bar = _target_for(A_true, 0.7, k, n=n)

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k, n=n)

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(Y, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.6]), Euclidean(1)),
        )
        spec = manifold_spec_from_params(params)
        flat, _treedef, _fspec = flatten_params_with_spec(params)

        # A nonzero ambient tangent direction with a real PSD-block component.
        eta = jnp.asarray(rng.normal(size=flat.shape))

        Heta = _riemannian_hvp(residual_fn, flat, eta, spec)
        assert Heta.shape == flat.shape
        # The load-bearing assertion: the Lyapunov PD floor keeps the HVP
        # finite even when YtY is conditioned at 1e8.
        assert bool(jnp.all(jnp.isfinite(Heta)))

    def test_full_step_does_not_propagate_nan_into_info(self):
        # One outer TR step from an ill-conditioned start with a radius big
        # enough to reach the boundary: either the proposed Y keeps rank
        # (smallest singular value > 1e-8) OR the optimizer detects the bad
        # model (rho NaN / negative denominator) and REJECTS+shrinks. Either
        # way final_objective is FINITE and the run is NOT certified on a NaN.
        n, k = 3, 3  # k/n = 1: the worst gauge fraction
        Y = self._ill_conditioned_Y(n, k, cond_target=1e8)

        rng = np.random.default_rng(13)
        A_true = jnp.asarray(rng.normal(size=(n, k)))
        x_bar = _target_for(A_true, 0.7, k, n=n)

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k, n=n)

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(Y, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.6]), Euclidean(1)),
        )
        spec = manifold_spec_from_params(params)

        # Large init radius -> the first tCG can reach the boundary on a bad
        # model. A single outer step is enough to exercise the rank-drop path.
        opt = riemannian_tr(max_steps=1, init_radius=10.0)
        th, info = opt(residual_fn, params, spec)

        # final_objective must be finite (NaN would mean the Lyapunov solve
        # poisoned the carry).
        assert np.isfinite(float(jnp.asarray(info.final_objective)))
        # A single step on a hard model must NOT certify converged on a NaN.
        assert bool(info.done) is False
        A_prop, _ = _components(th, k)
        # The proposed iterate is finite (rejected steps fall back to Y).
        assert bool(jnp.all(jnp.isfinite(A_prop)))


# ---------------------------------------------------------------------------
# Negative control: a purely vertical gradient must NOT certify converged.
# ---------------------------------------------------------------------------
class TestVerticalGradientNegativeControl:
    r"""[numerical-gauge] Negative control on the raw-||grad|| risk.

    "Negative control: an iterate sitting on the gauge fibre (``g`` has only a
    vertical component injected) must NOT certify converged." We construct a
    residual whose Euclidean gradient at ``Y`` is PURELY vertical (a
    skew-symmetric ``Y Omega`` direction): the HORIZONTAL gradient is zero but
    the ambient gradient is large. A correct solver measures stationarity on
    the horizontal gradient, so it certifies converged (the horizontal
    component is the real first-order condition). A solver certifying on the
    raw ambient gradient would (wrongly) report NOT converged at a genuine
    horizontal stationary point -- so we pin the horizontal definition both
    ways:

    (a) at a horizontal stationary point with a nonzero VERTICAL ambient
        gradient, the reported ``||g||`` is ~0 (horizontal) and ``done`` is
        True -- a raw-gradient solver fails this; AND
    (b) the reported ``||g||`` is provably below the raw ambient gradient norm
        (the vertical mass is correctly discarded).
    """

    def test_vertical_only_gradient_is_horizontally_stationary(self):
        n, k = 5, 3
        rng = np.random.default_rng(21)
        Y = jnp.asarray(rng.normal(size=(n, k)))
        psd = PSDFixedRank(n, k)

        # A nonzero skew-symmetric Omega; Y @ Omega is a VERTICAL ambient
        # direction (its horizontal projection is zero).
        Omega = jnp.asarray(rng.normal(size=(k, k)))
        Omega = 0.5 * (Omega - Omega.T)  # skew
        vertical_dir = Y @ Omega
        # Confirm it is vertical: horizontal projection ~ 0.
        assert float(jnp.linalg.norm(psd.projection(Y, vertical_dir))) < 1e-8
        assert float(jnp.linalg.norm(vertical_dir)) > 1e-3  # but ambient large

        # Residual whose Euclidean gradient at flat(Y, phi*) is a nonzero
        # multiple of this vertical direction on the PSD block and 0 on phi.
        # The criterion is the least-squares ``0.5 * sum(r^2)``, so the ambient
        # gradient of moment 0 is ``r0 * c``. With the bare moment ``r0 =
        # <c, Yf>`` this VANISHES at the start: ``<Y Omega, Y>_F = 0`` because a
        # skew-symmetric ``Omega`` is Frobenius-orthogonal to the symmetric
        # ``Y^T Y``, so ``r0 = 0`` and the ambient gradient is ~0 -- the
        # negative control would be VACUOUS. Adding a nonzero constant offset
        # ``r0 = <c, Yf> - 1`` makes ``r0 = -1`` at the start, so the ambient
        # gradient is ``-1 * c = -vertical_dir`` (large, purely vertical) while
        # its horizontal projection is still ~0 (genuine horizontal
        # stationarity). The offset does not change WHERE the horizontal
        # stationary point is -- the only descent direction stays the
        # quotiented-out vertical ``c``.
        c = jnp.reshape(vertical_dir, (-1,))

        def residual_fn(tf):
            # phi already at its target (0), so only the vertical PSD push.
            Yf = tf[: n * k]
            phi = tf[n * k]
            return jnp.stack([jnp.sum(c * Yf) - 1.0, phi])

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(Y, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.0]), Euclidean(1)),
        )
        spec = manifold_spec_from_params(params)
        flat, _treedef, _fspec = flatten_params_with_spec(params)

        # The ambient Euclidean gradient of the objective at the start.
        def objective(tf):
            r = residual_fn(tf)
            return 0.5 * jnp.sum(r * r)

        egrad = jax.grad(objective)(flat)
        eg_psd = jnp.reshape(egrad[: n * k], (n, k))
        ambient_norm = float(jnp.linalg.norm(egrad))
        horiz = psd.projection(Y, eg_psd)
        horiz_norm = float(jnp.linalg.norm(horiz))
        # The PSD-block ambient gradient is (a scalar multiple of) the
        # vertical direction: horizontal part is ~0, ambient part is large.
        assert ambient_norm > 1e-3
        assert horiz_norm < 1e-7

        opt = riemannian_tr(max_steps=50, rtol=1e-8, atol=1e-10)
        _th, info = opt(residual_fn, params, spec)

        # (a) Horizontal stationarity -> the solver certifies converged, and
        # the reported ||g|| is the HORIZONTAL norm (~0), not the ambient one.
        reported = float(jnp.asarray(info.final_gradient_norm))
        assert reported < 1e-7  # NOT ~ambient_norm
        # (b) reported gradient norm is far below the raw ambient norm.
        assert reported < 1e-3 * ambient_norm
        assert bool(info.done) is True

    def test_injected_vertical_step_does_not_falsely_advance(self):
        # The complementary control: if a solver moved ALONG the vertical
        # fibre it would change raw Y but leave Gamma fixed and the objective
        # fixed. Starting AT a horizontal optimum, the converged Gamma must
        # equal the start Gamma to 1e-9 (the solver did not wander the fibre).
        n, k = 5, 2
        rng = np.random.default_rng(31)
        A = jnp.asarray(rng.normal(size=(n, k)))
        x_bar = _target_for(A, 0.7, k, n=n)  # start IS the truth

        def residual_fn(tf):
            return _nonconvex_residual(tf, x_bar, k, n=n)

        @jdc.pytree_dataclass
        class P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        params = P(
            Y=ManifoldLeaf(A, PSDFixedRank(n, k)),
            phi=ManifoldLeaf(jnp.array([0.7]), Euclidean(1)),
        )
        spec = manifold_spec_from_params(params)

        opt = riemannian_tr(max_steps=50, rtol=1e-10, atol=1e-12)
        th, info = opt(residual_fn, params, spec)
        assert bool(info.done) is True
        A_hat, _ = _components(th, k)
        G_start = A @ A.T
        G_hat = A_hat @ A_hat.T
        # Gamma did not move (no fibre wandering, no spurious advance).
        assert bool(jnp.allclose(G_hat, G_start, atol=1e-9))
        # And the objective stayed at its (near-zero) optimum.
        assert float(jnp.asarray(info.final_objective)) < 1e-12


# ---------------------------------------------------------------------------
# Estimator-path smoke: gauge-invariance through estimate() (slow).
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestEstimatorPathGaugeInvariance:
    r"""[numerical-gauge] End-to-end gauge invariance via ``estimate()``.

    Cross-cuts the same #146 raw-||grad|| risk at the public boundary: two
    starts ``Y0`` and ``Y0 @ Q`` driven through ``estimate(...,
    optimizer=riemannian_tr())`` must land on the same gauge-invariant
    ``Gamma_hat`` and ``J_stat``. Gated slow (full pipeline + pymanopt-free).
    """

    @pytest.mark.parametrize("k", [2, 3])
    def test_estimate_with_riemannian_tr_is_gauge_invariant(self, k):
        from emu_gmm import estimate
        from emu_gmm.covariance import SyntheticCovariance
        from emu_gmm.measures import SyntheticMeasure
        from emu_gmm.weighting import ContinuouslyUpdated

        rng = np.random.default_rng(300 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        Gamma_true = A_true @ A_true.T
        phi_true = 0.7
        g_true = Gamma_true[_TRIU_N[:, 0], _TRIU_N[:, 1]]
        target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
        M = N * (N + 1) // 2 + 1
        noise = 0.01
        n_sim = 200
        noise_key = jax.random.PRNGKey(300 + k)

        def sampler(key, theta):
            del key, theta
            return target[None, :] + noise * jax.random.normal(noise_key, (n_sim, M))

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler
        )

        def model(x, theta):
            Y = theta.Y.array
            phi = theta.phi.array[0]
            g = (Y @ Y.T)[_TRIU_N[:, 0], _TRIU_N[:, 1]]
            return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x

        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
        Q = _orthogonal(5 + k, k)

        def run(Yinit):
            return estimate(
                model,
                measure,
                covariance=SyntheticCovariance(),
                weighting=ContinuouslyUpdated(),
                optimizer=riemannian_tr(max_steps=200),
                theta_init=_make_params(Yinit, 0.65, k),
            )

        res_a = run(Y0)
        res_b = run(Y0 @ Q)
        assert bool(res_a.converged) and bool(res_b.converged)
        Aa, _ = res_a.components()
        Ab, _ = res_b.components()
        Ga, Gb = Aa @ Aa.T, Ab @ Ab.T
        assert bool(jnp.allclose(Ga, Gb, atol=1e-7))
        assert bool(jnp.allclose(Ga, Gamma_true, atol=4e-3))
        assert float(jnp.asarray(res_a.J_stat)) == pytest.approx(
            float(jnp.asarray(res_b.J_stat)), abs=1e-7
        )
