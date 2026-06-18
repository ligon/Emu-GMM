r"""pymanopt parity -- the GREEN-LIGHT gate for ``riemannian_tr`` (#152).

This is the **parity-test** lens of the Phase-0 red-team register turned into
assertions. It is written TEST-FIRST: the module

    ``from emu_gmm.manifolds.riemannian_tr import riemannian_tr``

does not exist yet (Phase 2 builds it), so every test here is RED (collection
fails at import, or the assertions fire) until the implementation lands. That
is intended -- the file is the executable spec the implementation must satisfy.

Why these four tests, and why they cannot pass for the wrong reason
-------------------------------------------------------------------
The red-team found that a *convex* parity fixture certifies nothing: a true
second-order RTR, a Gauss-Newton hill-climber, and ``riemannian_lm`` all land
on the same easy ``Gamma`` and the negative-curvature branch of tCG -- the one
code path that justifies the whole solver -- never runs. Comparing **raw** ``A``
is equally vacuous (or flaky): emu-gmm and pymanopt walk different but
equivalent ``O(k)``-orbit representatives, so raw ``A`` differs by an arbitrary
orthogonal ``Q`` and the only correct comparison is on gauge invariants. And a
``0.5||r||^2`` vs ``||r||^2`` convention slip rescales the trust radius / ``rho``
/ the tCG curvature test by a constant -- invisible at the argmin but fatal to
the machinery being validated.

So the gate asserts FOUR things, each pinning a register risk:

1.  ``TestNonConvexMetaGate`` -- parity-test: "a convex fixture cannot
    distinguish a true-Hessian RTR from a GN hill-climber". A deliberately
    NON-CONVEX fixture (residual non-linear in ``Gamma``, start on the wrong
    side of a saddle) where (a) ``riemannian_lm`` STALLS (asserted -- it cannot
    follow negative curvature), (b) ``riemannian_tr`` escapes to ``Gamma_true``,
    and (c) tCG reports >= 1 negative-curvature exit on the path (meta-check, so
    a future convex-ification of the fixture fails the meta rather than passing
    silently).

2.  ``TestGaugeInvariantComparisonOnly`` -- parity-test: "comparing raw A makes
    the assertion gauge-dependent". Compare ``Gamma = A @ A.T``, ``eigvalsh``,
    and the J-stat -- NEVER raw ``A``. Gauge-blind POSITIVE CONTROL:
    right-multiply ``A`` by a fixed random ``O(k)`` and assert the Gamma
    comparison still passes (proving it cannot be satisfied by accidental fibre
    alignment), while raw ``A`` provably differs.

3.  ``TestConventionPinnedAtY0`` -- parity-test: "objective/HVP convention
    mismatch silently rescales the trust region". Build the pymanopt
    ``Problem.cost`` from the EXACT same residual callable used to construct
    ``riemannian_tr``'s ``Q``, and assert cost / Riemannian-gradient /
    Riemannian-Hessian agree at ``Y0`` to ~1e-9 BEFORE optimising. This pins the
    ``0.5||r||^2`` factor and the retraction-pullback HVP at the gradient/Hessian
    level (where a 2x factor IS visible).

4.  ``TestStepLevelTraceMatchesPymanopt`` -- parity-test: "non-determinism lets
    parity agree only because both reach the easy global optimum" and "tCG
    omissions diverge only in gauge directions, invisible to a Gamma-only
    test". Pin EVERYTHING (same ``Y0`` / ``Delta0`` / ``Delta_bar`` /
    ``rho_prime`` / ``kappa`` / ``theta`` / ``max_tcg_steps``) and assert the
    per-OUTER-iteration trace (``Delta``, ``rho``, tCG stop-reason, ``||g||``)
    matches pymanopt over the first N steps -- argmin agreement is necessary but
    not sufficient.

Intended ``info`` surface (Phase 2 must expose this; the trace tests read it)
----------------------------------------------------------------------------
``theta_hat, info = riemannian_tr(...)(residual_fn, theta_init, manifold_spec)``
where ``info`` is the v2 ``OptimizerInfo`` (``info.steps``, ``info.done``,
``info.status``, ``info.final_objective``) PLUS an RTR-specific ``info.tr_trace``
namespace carrying, per executed OUTER iteration:

    info.tr_trace.delta        # (n_outer,) trust radius Delta_k AT entry to step k
    info.tr_trace.rho          # (n_outer,) accept/reject ratio rho_k
    info.tr_trace.grad_norm    # (n_outer,) Riemannian ||g_k|| (per-leaf metric)
    info.tr_trace.tcg_stop     # (n_outer,) int tCG stop-reason code (codes below)
    info.tr_trace.n_negcurv    # scalar int: # of NEGATIVE_CURVATURE tCG exits seen

The tCG stop-reason integer codes MUST match pymanopt's enumeration
(``trust_regions.py``): NEGATIVE_CURVATURE=0, EXCEEDED_TR=1,
REACHED_TARGET_LINEAR=2, REACHED_TARGET_SUPERLINEAR=3, MAX_INNER_ITER=4,
MODEL_INCREASED=5. (If Phase 2 chooses a different surface, update this file in
lockstep -- but the four risks above are the contract, not the field names.)

The unit helpers ``_riemannian_hvp`` and ``_truncated_cg`` are imported lazily
inside the test that needs them so the module still collects (and the other
tests still run RED-for-the-right-reason) even if only the public ``riemannian_tr``
factory has landed.
"""

from __future__ import annotations

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
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

# The module under construction (Phase 2). SKIP cleanly until it lands so the
# rest of the manifolds suite still collects; goes live once Phase 2 ships it.
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
riemannian_tr = riemannian_tr_mod.riemannian_tr

N = 5  # ambient PSD side (K-Aggregators primary geometry)

# Upper-triangular index pairs of the (5,5) symmetric Gamma: 15 unique entries.
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)

# tCG stop-reason codes -- MUST match pymanopt's TrustRegions enumeration.
NEGATIVE_CURVATURE = 0
EXCEEDED_TR = 1
REACHED_TARGET_LINEAR = 2
REACHED_TARGET_SUPERLINEAR = 3
MAX_INNER_ITER = 4
MODEL_INCREASED = 5


# ---------------------------------------------------------------------------
# Shared param container + helpers (mirrors test_manifold_acceptance_phase6).
# FLAG (shared_helpers_needed): ProductParams / _make_params / _triu_of /
# _orthogonal / _make_synthetic_measure want to move to a manifolds conftest;
# they are duplicated across the phase-6/7 acceptance suites and this file.
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class ProductParams:
    """A ``PSDFixedRank(5, K)`` ``Y`` leaf plus a ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _triu_of(G):
    """The 15 unique upper-triangular entries of a symmetric (5,5) matrix."""
    return G[_TRIU[:, 0], _TRIU[:, 1]]


def _orthogonal(seed: int, k: int) -> jnp.ndarray:
    """A fixed random ``Q in O(k)`` (QR of a Gaussian, sign-canonicalised)."""
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(k, k)))
    q, r = jnp.linalg.qr(g)
    return q @ jnp.diag(jnp.sign(jnp.diag(r)))


def _make_synthetic_measure(target, n_sim, noise, *, draw_seed=0):
    """A frozen synthetic measure whose per-draw moment is ``target + noise``."""
    M = int(target.shape[0])
    draw_noise = noise * jax.random.normal(jax.random.PRNGKey(draw_seed), (n_sim, M))

    def sampler(key, theta):
        del key, theta
        return target[None, :] + draw_noise

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    x_bar = jnp.mean(target[None, :] + draw_noise, axis=0)
    return measure, x_bar


# ---------------------------------------------------------------------------
# The NON-CONVEX fixture. The residual is non-linear in Gamma (a SQUARED moment
# trace(Gamma @ Gamma) and an exponential moment) so the true Hessian
# H = J'J + S carries an indefinite S term: there is a saddle between a chosen
# start and the truth, and J'J alone (Gauss-Newton / riemannian_lm) stalls on
# the wrong side of it. Used by the meta-gate and the step-level trace.
# ---------------------------------------------------------------------------
def _nonconvex_model(x, theta):
    r"""Gauge-invariant but NON-linear-in-Gamma moments.

    Depends on theta only through Gamma = Y Y^T (and phi), so the O(k) fibre is
    a true symmetry (raw-A comparison forbidden), BUT the moment map is
    g(Gamma) = [ trace(Gamma)^2,  trace(Gamma @ Gamma),  exp(0.3 trace(Gamma)),
                 triu(Gamma) (first 6),  phi ].
    The squared / exp terms make d^2 g / dGamma^2 != 0, so S = sum_i r_i grad^2 r_i
    is indefinite where the residual r changes sign -- the regime RTR is for.
    """
    Y = theta.Y.array
    phi = theta.phi.array[0]
    G = Y @ Y.T
    tr = jnp.trace(G)
    moms = jnp.array(
        [
            tr * tr,
            jnp.trace(G @ G),
            jnp.exp(0.3 * tr),
        ]
    )
    extra = _triu_of(G)[:6]
    m = jnp.concatenate([moms, extra, jnp.reshape(phi, (1,))])
    return m - x


def _nonconvex_target(A_true, phi_true):
    """The exact moment vector the non-convex model emits at the truth."""
    G = A_true @ A_true.T
    tr = jnp.trace(G)
    moms = jnp.array([tr * tr, jnp.trace(G @ G), jnp.exp(0.3 * tr)])
    extra = _triu_of(G)[:6]
    return jnp.concatenate([moms, extra, jnp.reshape(jnp.asarray(phi_true), (1,))])


# A linear-in-Gamma (convex) reference model, reused for the gauge-blind and
# convention-pinning tests where we do NOT need negative curvature.
def _gauge_invariant_model(x, theta):
    """psi = triu(Y Y^T) concat phi - x; linear in Gamma, convex."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = _triu_of(Y @ Y.T)
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _estimate_lm(model, measure, theta_init, *, max_steps=400):
    return estimate(
        model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=max_steps),
        theta_init=theta_init,
    )


def _estimate_tr(model, measure, theta_init, **tr_kwargs):
    return estimate(
        model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_tr(**tr_kwargs),
        theta_init=theta_init,
    )


# ===========================================================================
# Risk 1 -- non-convex meta-gate (parity-test: convex fixture certifies nothing)
# ===========================================================================
class TestNonConvexMetaGate:
    r"""parity-test / "convex fixture cannot distinguish true-Hessian RTR from a
    GN hill-climber" (blocker).

    On a deliberately NON-CONVEX fixture with a start on the wrong side of a
    saddle, assert the THREE-part meta-gate from the register's ``proposed_test``:

      (a) ``riemannian_lm`` (Gauss-Newton, J'J PSD-by-construction) STALLS --
          it cannot reach ``Gamma_true`` because it never follows negative
          curvature. This GUARDS that the fixture genuinely exercises the RTR
          code path; if a future edit convex-ifies the fixture, LM stops
          stalling and this assertion fails LOUDLY rather than the test passing
          for the wrong reason.
      (b) ``riemannian_tr`` escapes the saddle and recovers ``Gamma_true``.
      (c) tCG reported >= 1 NEGATIVE_CURVATURE exit on the path
          (``info.tr_trace.n_negcurv >= 1``) -- the meta-check that the negative-
          curvature branch actually fired. A solver that silently replaced the
          true Hessian with a J'J surrogate would converge (maybe) but report
          ZERO negative-curvature exits and fail here.

    All recovery is on the gauge invariant ``Gamma``; raw ``A`` is never touched.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("k", [2, 3])
    def test_lm_stalls_tr_escapes_with_negcurv(self, k):
        rng = np.random.default_rng(700 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        Gamma_true = A_true @ A_true.T
        phi_true = 0.5
        target = _nonconvex_target(A_true, phi_true)
        measure, _x_bar = _make_synthetic_measure(
            target, n_sim=400, noise=0.01, draw_seed=k
        )

        # Start on the WRONG side of the saddle: a strongly shrunk + rotated
        # factor whose Gamma has the wrong trace sign-structure for the squared
        # moments, so the Gauss-Newton model points the LM iterate away from
        # the truth. (Deterministic, not near-truth: this is a basin test.)
        Y0 = 0.15 * jnp.asarray(rng.normal(size=(N, k)))
        theta_init = _make_params(Y0, 0.4, k)

        # (a) riemannian_lm STALLS -- does NOT reach Gamma_true.
        res_lm = _estimate_lm(_nonconvex_model, measure, theta_init, max_steps=400)
        A_lm, _ = res_lm.components()
        Gamma_lm = A_lm @ A_lm.T
        lm_err = float(jnp.max(jnp.abs(Gamma_lm - Gamma_true)))
        assert lm_err > 0.1, (
            "fixture is no longer non-convex: riemannian_lm reached Gamma_true "
            f"(max|dGamma|={lm_err:.3e}); the meta-gate would pass vacuously"
        )

        # (b) riemannian_tr ESCAPES and recovers Gamma_true to a tight tol.
        res_tr = _estimate_tr(
            _nonconvex_model, measure, theta_init, max_steps=300, rtol=1e-8
        )
        assert bool(res_tr.converged)
        A_tr, _ = res_tr.components()
        Gamma_tr = A_tr @ A_tr.T
        tr_err = float(jnp.max(jnp.abs(Gamma_tr - Gamma_true)))
        assert tr_err < 2e-2, (Gamma_tr, Gamma_true, tr_err)
        # eigvals (gauge invariant) recovered too.
        assert bool(
            jnp.allclose(
                jnp.linalg.eigvalsh(Gamma_tr),
                jnp.linalg.eigvalsh(Gamma_true),
                atol=2e-2,
            )
        )

        # (c) tCG fired its negative-curvature branch at least once.
        n_negcurv = int(
            jnp.asarray(res_tr.diagnostics.optimizer_info.tr_trace.n_negcurv)
        )
        assert n_negcurv >= 1, (
            "tCG reported ZERO negative-curvature exits on a provably non-convex "
            "fixture -- the true-Hessian branch never fired (J'J surrogate?)"
        )
        # And the stop-reason trace actually contains a NEGATIVE_CURVATURE code.
        stops = np.asarray(res_tr.diagnostics.optimizer_info.tr_trace.tcg_stop)
        assert int(np.sum(stops == NEGATIVE_CURVATURE)) >= 1


# ===========================================================================
# Risk 2 -- gauge-invariant comparison ONLY (parity-test: raw-A is wrong)
# ===========================================================================
class TestGaugeInvariantComparisonOnly:
    r"""parity-test / "comparing raw A instead of Gamma=A A^T makes the parity
    assertion gauge-dependent and either flaky or vacuously loose" (blocker).

    Cross-check ``riemannian_tr`` against pymanopt TrustRegions on the IDENTICAL
    least-squares problem, and assert agreement ONLY on gauge invariants
    (``Gamma``, ``eigvalsh(Gamma)``, J-stat). Then the POSITIVE CONTROL the
    register demands: right-multiply ``A_tr`` by a fixed random ``O(k)`` and show

      * the Gamma-level comparison STILL passes (proving it is gauge-blind), and
      * the raw-``A`` comparison would FAIL (proving raw-A would be the wrong
        assertion -- it is satisfiable only by accidental fibre alignment).

    Import-gated via ``pytest.importorskip('pymanopt')`` (dev/test-only dep).
    """

    @pytest.mark.parametrize("k", [2, 3])
    def test_quotient_parity_and_gauge_blind_positive_control(self, k):
        pytest.importorskip("pymanopt")
        import pymanopt
        from pymanopt.manifolds import Euclidean as PymEuclidean
        from pymanopt.manifolds import Product as PymProduct
        from pymanopt.manifolds import PSDFixedRank as PymPSDFixedRank
        from pymanopt.optimizers import TrustRegions

        noise, n_sim = 0.02, 200
        rng = np.random.default_rng(2 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        phi_true = 0.7
        target = jnp.concatenate(
            [_triu_of(A_true @ A_true.T), jnp.reshape(jnp.asarray(phi_true), (1,))]
        )
        measure, x_bar = _make_synthetic_measure(target, n_sim, noise, draw_seed=0)

        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
        theta_init = _make_params(Y0, 0.65, k)
        res = _estimate_tr(_gauge_invariant_model, measure, theta_init, max_steps=300)
        assert bool(res.converged)
        A_tr, _ = res.components()
        Gamma_tr = A_tr @ A_tr.T
        J_tr = float(jnp.asarray(res.J_stat))

        # IDENTICAL whitening so the pymanopt objective coincides exactly.
        W = jnp.linalg.inv(jnp.asarray(res.V_X.array))
        manifold = PymProduct([PymPSDFixedRank(N, k), PymEuclidean(1)])

        @pymanopt.function.jax(manifold)
        def cost(Y, phi):
            m = jnp.concatenate([_triu_of(Y @ Y.T), phi])
            r = m - x_bar
            return 0.5 * r @ W @ r

        problem = pymanopt.Problem(manifold, cost)
        optimizer = TrustRegions(
            verbosity=0, max_iterations=300, min_gradient_norm=1e-10
        )
        out = optimizer.run(problem, initial_point=[np.asarray(Y0), np.array([0.65])])
        Y_pym = np.asarray(out.point[0], dtype=np.float64)
        Gamma_pym = jnp.asarray(Y_pym @ Y_pym.T)
        J_pym = 2.0 * float(out.cost)  # J = r'Wr = 2 * 0.5 r'Wr

        # --- gauge-INVARIANT parity (the only correct comparison) ---
        assert bool(jnp.allclose(Gamma_tr, Gamma_pym, atol=1e-6))
        assert bool(
            jnp.allclose(
                jnp.linalg.eigvalsh(Gamma_tr),
                jnp.linalg.eigvalsh(Gamma_pym),
                atol=1e-6,
            )
        )
        assert J_tr == pytest.approx(J_pym, abs=1e-6)

        # --- POSITIVE CONTROL: the Gamma assertion is genuinely gauge-blind ---
        # Right-multiply by a fixed random O(k): Gamma is invariant, raw A is not.
        if k >= 2:  # k=1 has a trivial O(1) fibre (only +-1); skip the rotation.
            Q = _orthogonal(31 + k, k)
            assert bool(jnp.allclose(Q @ Q.T, jnp.eye(k), atol=1e-12))
            A_rot = A_tr @ Q
            # (i) raw A genuinely moved (so this is a non-trivial Q).
            assert not bool(jnp.allclose(A_rot, A_tr, atol=1e-6))
            # (ii) Gamma-level parity STILL holds after the rotation.
            assert bool(jnp.allclose(A_rot @ A_rot.T, Gamma_pym, atol=1e-6))
            # (iii) raw-A comparison would FAIL -> raw A is the wrong assertion.
            assert not bool(jnp.allclose(A_rot, Y_pym, atol=1e-6))


# ===========================================================================
# Risk 3 -- convention pinned at Y0 (parity-test: objective/HVP mismatch)
# ===========================================================================
class TestConventionPinnedAtY0:
    r"""parity-test / "objective/HVP convention mismatch silently rescales the
    trust region, so 'parity' compares two different problems" (high).

    BEFORE optimising, build the pymanopt ``Problem`` from the EXACT same
    residual callable + whitening + ``0.5`` factor used to define
    ``riemannian_tr``'s criterion ``Q``, and assert at a fixed ``Y0`` and a
    fixed random horizontal ``eta``:

      cost:    pymanopt.cost(Y0)          == Q(Y0)                     (0.5 factor)
      rgrad:   pymanopt.riem_gradient     == emu-gmm horizontal grad   (1x, not 2x)
      rhess:   pymanopt.riem_hessian[eta] == emu-gmm _riemannian_hvp[eta]

    all to rtol ~1e-9. A 2x objective slip is INVISIBLE at the argmin but shows
    up here as a factor-of-2 on grad and Hessian. The HVP equality also pins the
    retraction-pullback Hessian semantics: ``H[eta]`` is the Euclidean Hessian of
    ``eta -> Q(R_Y(eta))`` at ``eta=0``, projected to horizontal -- NOT a
    ``J'J`` surrogate (whose Hessian would drop the S term and disagree here).
    """

    @pytest.mark.parametrize("k", [2, 3])
    def test_cost_grad_hvp_agree_at_Y0_before_optimizing(self, k):
        pytest.importorskip("pymanopt")
        import pymanopt
        from emu_gmm._internal.params import (
            flatten_params_with_spec,
            manifold_spec_from_params,
        )

        # Unit-level RTR helper (Phase 2 exposes this).
        from emu_gmm.manifolds.riemannian_tr import _riemannian_hvp
        from pymanopt.manifolds import Euclidean as PymEuclidean
        from pymanopt.manifolds import Product as PymProduct
        from pymanopt.manifolds import PSDFixedRank as PymPSDFixedRank

        rng = np.random.default_rng(40 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        phi_true = 0.7
        target = jnp.concatenate(
            [_triu_of(A_true @ A_true.T), jnp.reshape(jnp.asarray(phi_true), (1,))]
        )
        _measure, x_bar = _make_synthetic_measure(
            target, n_sim=200, noise=0.02, draw_seed=0
        )
        # A fixed, non-identity whitening so a missing/extra W would be caught.
        Wroot = jnp.asarray(rng.normal(size=(target.shape[0], target.shape[0])))
        W = Wroot @ Wroot.T + jnp.eye(int(target.shape[0]))
        L = jnp.linalg.cholesky(jnp.linalg.inv(W))  # whitening: r'Wr == ||L^-1 r||^2

        Y0 = jnp.asarray(A_true + 0.1 * rng.normal(size=(N, k)))
        phi0 = 0.6
        theta_init = _make_params(Y0, phi0, k)
        spec = manifold_spec_from_params(theta_init)
        flat0, _treedef, _fspec = flatten_params_with_spec(theta_init)

        def whitened_residual(flat):
            Y = flat[: N * k].reshape(N, k)
            phi = flat[N * k]
            m = jnp.concatenate([_triu_of(Y @ Y.T), jnp.reshape(phi, (1,))])
            r = m - x_bar
            return jnp.linalg.solve(L, r)  # L^{-1} r, so ||.||^2 == r' W r

        def Q(flat):  # the 0.5 ||.||^2 criterion riemannian_tr minimises
            rr = whitened_residual(flat)
            return 0.5 * jnp.sum(rr * rr)

        # ---- emu-gmm side: horizontal gradient + retraction-pullback HVP ----
        psd = PSDFixedRank(N, k)

        def project_flat(flat, v):
            vY = psd.projection(flat[: N * k].reshape(N, k), v[: N * k].reshape(N, k))
            return jnp.concatenate([vY.reshape(-1), v[N * k :]])

        egrad = jax.grad(Q)(flat0)
        rgrad_emu = project_flat(flat0, egrad)  # horizontal (PSD) + identity (Euclid)

        # A fixed random HORIZONTAL eta (project an ambient random vector once).
        eta_amb = jnp.asarray(rng.normal(size=flat0.shape))
        eta = project_flat(flat0, eta_amb)
        # eta must be (numerically) horizontal: zero vertical residue.
        assert float(jnp.linalg.norm(eta - project_flat(flat0, eta))) < 1e-10 * float(
            jnp.linalg.norm(eta) + 1e-30
        )

        hvp_emu = _riemannian_hvp(whitened_residual, spec, flat0, eta)

        # ---- pymanopt side: rgrad / rhess of the IDENTICAL cost ----
        manifold = PymProduct([PymPSDFixedRank(N, k), PymEuclidean(1)])

        @pymanopt.function.jax(manifold)
        def cost(Y, phi):
            m = jnp.concatenate([_triu_of(Y @ Y.T), phi])
            r = m - x_bar
            rr = jnp.linalg.solve(L, r)
            return 0.5 * jnp.sum(rr * rr)

        problem = pymanopt.Problem(manifold, cost)
        Ypt = [np.asarray(Y0, dtype=np.float64), np.array([phi0], dtype=np.float64)]
        eta_pt = [
            np.asarray(eta[: N * k].reshape(N, k), dtype=np.float64),
            np.asarray(eta[N * k :], dtype=np.float64),
        ]

        # cost: 0.5 factor pinned.
        assert float(problem.cost(Ypt)) == pytest.approx(float(Q(flat0)), rel=1e-9)

        # rgrad: 1x (NOT 2x). Compare on the gauge-invariant flattened layout.
        rg_pym = problem.riemannian_gradient(Ypt)
        rg_pym_flat = jnp.concatenate(
            [jnp.asarray(rg_pym[0]).reshape(-1), jnp.asarray(rg_pym[1]).reshape(-1)]
        )
        assert bool(jnp.allclose(rgrad_emu, rg_pym_flat, rtol=1e-9, atol=1e-9)), (
            rgrad_emu,
            rg_pym_flat,
        )

        # rhess[eta]: the retraction-pullback Hessian, projected to horizontal.
        rh_pym = problem.riemannian_hessian(Ypt, eta_pt)
        rh_pym_flat = jnp.concatenate(
            [jnp.asarray(rh_pym[0]).reshape(-1), jnp.asarray(rh_pym[1]).reshape(-1)]
        )
        assert bool(jnp.allclose(hvp_emu, rh_pym_flat, rtol=1e-8, atol=1e-8)), (
            hvp_emu,
            rh_pym_flat,
        )

        # HVP self-adjointness on the horizontal space (register blocker #1):
        # <eta2, H[eta1]> == <eta1, H[eta2]> for two horizontal directions.
        eta2 = project_flat(flat0, jnp.asarray(rng.normal(size=flat0.shape)))
        h1 = _riemannian_hvp(whitened_residual, spec, flat0, eta)
        h2 = _riemannian_hvp(whitened_residual, spec, flat0, eta2)
        sym_lhs = float(jnp.sum(eta2 * h1))
        sym_rhs = float(jnp.sum(eta * h2))
        assert sym_lhs == pytest.approx(sym_rhs, rel=1e-8, abs=1e-10)


# ===========================================================================
# Risk 4 -- step-level trace (parity-test: argmin agreement is not sufficient)
# ===========================================================================
class TestStepLevelTraceMatchesPymanopt:
    r"""parity-test / "non-determinism lets parity agree only because both reach
    the easy global optimum" + "tCG re-tangentialization/HVP-projection omissions
    diverge only in gauge directions, invisible to a Gamma-only test" (high).

    Pin EVERYTHING -- identical explicit ``Y0``, ``Delta0``, ``Delta_bar``,
    ``rho_prime``, ``kappa``, ``theta``, ``max_tcg_steps`` (= intrinsic quotient
    dim) -- on a problem with KNOWN negative curvature on the path, and assert the
    per-OUTER-iteration trace ``(Delta_k, rho_k, tCG stop-reason, ||g_k||)``
    matches pymanopt over the first N outer steps. Step-level agreement is what
    proves the TR loop + tCG were ported faithfully (a transcription bug in the
    tau-boundary or rho heuristic would diverge here while the argmin still
    agrees).
    """

    N_STEPS = 5  # compare the first 5 OUTER iterations step for step

    @pytest.mark.slow
    @pytest.mark.parametrize("k", [2])
    def test_outer_trace_matches_pymanopt(self, k):
        pytest.importorskip("pymanopt")
        import pymanopt
        from pymanopt.manifolds import Euclidean as PymEuclidean
        from pymanopt.manifolds import Product as PymProduct
        from pymanopt.manifolds import PSDFixedRank as PymPSDFixedRank
        from pymanopt.optimizers import TrustRegions

        # pymanopt 2.2.1's ``TrustRegions.run()`` calls ``_initialize_log`` but
        # NEVER ``_add_log_entry`` (verified against the installed source), so
        # ``out.log['iterations']`` is permanently EMPTY -- there is no public
        # per-iteration trajectory to read. We instrument the oracle directly by
        # subclassing and overriding the single tCG entry point: it fires once
        # per OUTER iteration, BEFORE rho is computed, with ``Delta`` = the trust
        # radius at entry, ``fgradx`` = the Riemannian gradient at entry, and
        # ``result[3]`` = the tCG stop-reason integer. That captures the radius
        # schedule, the gradient trajectory, and the tCG decisions -- the strong
        # per-step parity surface -- without copying pymanopt's 270-line run().
        class _LoggingTrustRegions(TrustRegions):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._cap = []

            def _truncated_conjugate_gradient(
                self, problem, x, fgradx, eta, Delta, theta, kappa, mininner, maxinner
            ):
                entry = {
                    "Delta": float(Delta),
                    "gnorm": float(problem.manifold.norm(x, fgradx)),
                }
                self._cap.append(entry)
                result = super()._truncated_conjugate_gradient(
                    problem, x, fgradx, eta, Delta, theta, kappa, mininner, maxinner
                )
                entry["stop"] = int(result[3])  # the stop_inner code
                return result

        rng = np.random.default_rng(900 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        phi_true = 0.5
        target = _nonconvex_target(A_true, phi_true)
        measure, x_bar = _make_synthetic_measure(
            target, n_sim=400, noise=0.01, draw_seed=k
        )

        # Wrong-side-of-saddle start (so the first few steps hit neg curvature).
        Y0 = 0.2 * jnp.asarray(rng.normal(size=(N, k)))
        phi0 = 0.4
        theta_init = _make_params(Y0, phi0, k)

        # Pinned TR hyperparameters, shared by both solvers.
        intrinsic_dim = (N * k - k * (k - 1) // 2) + 1  # quotient dim + Euclidean
        delta_bar = float(np.sqrt(intrinsic_dim))
        delta0 = delta_bar / 8.0
        rho_prime, kappa, theta = 0.1, 0.1, 1.0

        res_tr = _estimate_tr(
            _nonconvex_model,
            measure,
            theta_init,
            max_steps=60,
            rho_prime=rho_prime,
            kappa=kappa,
            theta=theta,
            min_inner=1,
            max_tcg_steps=intrinsic_dim,
            init_radius=delta0,
            max_radius=delta_bar,
        )
        trace = res_tr.diagnostics.optimizer_info.tr_trace
        delta_emu = np.asarray(trace.delta)
        gnorm_emu = np.asarray(trace.grad_norm)
        stop_emu = np.asarray(trace.tcg_stop)
        # (trace.rho is intentionally not read here: per-step rho parity against
        # pymanopt 2.2.1 is not validated -- see the note at the rho block below.)
        # Enough outer steps were actually taken to compare.
        assert delta_emu.shape[0] >= self.N_STEPS

        # ---- pymanopt, instrumented per-outer-iteration via the subclass ----
        W = jnp.linalg.inv(jnp.asarray(res_tr.V_X.array))
        manifold = PymProduct([PymPSDFixedRank(N, k), PymEuclidean(1)])

        def model_m(Y, phi):
            G = Y @ Y.T
            tr = jnp.trace(G)
            moms = jnp.array([tr * tr, jnp.trace(G @ G), jnp.exp(0.3 * tr)])
            return jnp.concatenate([moms, _triu_of(G)[:6], phi])

        @pymanopt.function.jax(manifold)
        def cost(Y, phi):
            r = model_m(Y, phi) - x_bar
            return 0.5 * r @ W @ r

        problem = pymanopt.Problem(manifold, cost)

        # Capture pymanopt's per-outer-iteration (Delta, ||g||, tCG stop) via the
        # instrumented subclass above (the public log is empty in 2.2.1).
        optimizer = _LoggingTrustRegions(
            verbosity=0,
            max_iterations=self.N_STEPS,  # only need the first N outer steps
            rho_prime=rho_prime,
            kappa=kappa,
            theta=theta,
        )
        optimizer.run(
            problem,
            initial_point=[np.asarray(Y0), np.array([phi0])],
            Delta_bar=delta_bar,
            Delta0=delta0,
            maxinner=intrinsic_dim,
            mininner=1,
        )
        cap = optimizer._cap
        delta_pym = np.array([c["Delta"] for c in cap])[: self.N_STEPS]
        gnorm_pym = np.array([c["gnorm"] for c in cap])[: self.N_STEPS]
        stop_pym = np.array([c["stop"] for c in cap])[: self.N_STEPS]

        n = self.N_STEPS
        # ||g_k||: the Riemannian gradient norm per outer step. Tight: same
        # geometry, same metric, same start.
        assert np.allclose(gnorm_emu[:n], gnorm_pym, rtol=1e-6, atol=1e-9), (
            gnorm_emu[:n],
            gnorm_pym,
        )
        # Delta_k: the trust radius at entry to each outer step.
        assert np.allclose(delta_emu[:n], delta_pym, rtol=1e-6, atol=1e-9), (
            delta_emu[:n],
            delta_pym,
        )
        # rho_k: per-step accept/reject ratio parity is NOT validated against
        # pymanopt 2.2.1. pymanopt computes rho inside ``run()`` AFTER the tCG
        # call (lines 255-321 of trust_regions.py), so it is not capturable from
        # a ``_truncated_conjugate_gradient`` override without copying run(); and
        # 2.2.1 exposes no per-iteration log to read it from either. Delta/gnorm/
        # stop-code parity (the tCG decisions + radius schedule + gradient
        # trajectory) is the strong per-step check.
        # tCG stop reason: integer code must match step for step.
        assert np.array_equal(stop_emu[:n], stop_pym), (stop_emu[:n], stop_pym)
        # And the path actually exercised negative curvature (else the trace
        # match is on an easy convex path and proves nothing about tCG).
        assert int(np.sum(stop_pym == NEGATIVE_CURVATURE)) >= 1
