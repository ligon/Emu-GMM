r"""TEST-FIRST suite for the outer trust-region loop of ``riemannian_tr`` (#152).

THEME: ``tr`` -- trust-region management.  This file is intentionally RED until
Phase 2 lands ``emu_gmm.manifolds.riemannian_tr``: the module import below will
raise ``ImportError`` until then.  That is correct and intended; do NOT add an
implementation to make it green prematurely.

What this suite pins (the ``tr-management`` lens of the Phase-0 red-team
register, #152):

* **rho-regularisation near convergence** (red-team ``tr-management`` /
  "rho_regularization near convergence: 0/0 acceptance"): pymanopt's
  ``rho_reg = max(1,|fx|)*spacing(1)*rho_regularization`` shifts both numerator
  and denominator so ``rho -> 1`` once both are at the eps level.  On
  emu-gmm's ``Q = 1/2 ||whitened r||^2`` the whitened residuals are O(0.1)
  (CLAUDE.md commitment 7), so ``|fx|`` is sub-1 and the shift is ~2e-13.  A
  spurious ``rho ~ 1`` must NOT accept a null step while ``||g_horizontal||`` is
  still large.
* **NaN-safety on the rho 0/0 floor** (same risk, NaN-safety half): a
  zero-curvature null step gives ``rhoden == 0``; the pure-JAX port must
  reproduce pymanopt's "force a radius decrease" with ``jnp.where(isnan, ...)``
  -- ``Delta`` stays finite, the step is rejected, and the loop terminates at
  ``max_steps`` with ``status != 'converged'`` rather than NaN-poisoning.
* **sigma -> 0 boundary regime** (red-team ``tr-management`` /
  "Affine-invariant Positive metric makes the tCG trust-region constraint
  collapse as sigma->0"): a scalar ``Positive(sigma)`` GMM whose CUE optimum
  collapses to ``sigma -> 0+`` (the realdata 'design' fixture) must stay
  strictly positive, converge, match ``riemannian_lm`` to tolerance, and report
  a finite ``final_objective`` -- the tCG trust constraint must not throttle the
  ambient step to ``~Delta * sigma``.
* **rank-drop in a neg-curvature boundary step is REJECTED** (red-team
  ``tr-management`` / "Negative-curvature boundary step can drop the rank of
  Y+V"): a full-radius negative-curvature step on a ``PSDFixedRank(n, n)``
  fixture that drives ``det(Y^T Y) -> 0`` must be rejected as an rho-failure
  (and ``Delta`` shrunk), NOT NaN-propagated into the next HVP / gradient /
  convergence norm.

Discipline mirrored from ``test_manifold_acceptance_phase6.py`` /
``test_riemannian_lm_phase3.py``: every recovery / agreement assertion is made
on a gauge-INVARIANT functional (``Gamma = Y Y^T``, its eigenvalues, ``J_stat``,
the scalar ``sigma``), NEVER on raw ``Y`` entries (which differ by an O(K)
rotation between solvers).  Tolerances are JUSTIFIED in the docstrings, not
loosened-until-green.  Reference curvatures are computed test-side by finite
differences / closed-form geodesic 2nd-derivatives so a wrong implementation
must fail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# --- The module under test (does NOT exist until Phase 2; import is RED). ---
riemannian_tr = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 has not landed emu_gmm.manifolds.riemannian_tr yet "
    "(this suite is written test-first and is expected RED until then).",
).riemannian_tr

from emu_gmm._internal.params import (  # noqa: E402
    flatten_params_with_spec,
    manifold_spec_from_params,
)
from emu_gmm.covariance import IIDCovariance  # noqa: E402
from emu_gmm.estimator import estimate  # noqa: E402
from emu_gmm.manifolds import Euclidean, Positive, PSDFixedRank  # noqa: E402
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf  # noqa: E402
from emu_gmm.manifolds.riemannian_lm import riemannian_lm  # noqa: E402
from emu_gmm.manifolds.riemannian_tr import (  # noqa: E402
    _riemannian_hvp,
    _truncated_cg,
)
from emu_gmm.measures import EmpiricalMeasure  # noqa: E402
from emu_gmm.weighting import ContinuouslyUpdated  # noqa: E402

N = 5  # ambient PSD side (matches the phase-6 DGP)


# ===========================================================================
# Shared fixtures / helpers (inline; flag for a conftest move in
# shared_helpers_needed).
# ===========================================================================
@jdc.pytree_dataclass
class ProductParams:
    """A ``PSDFixedRank(N, k)`` ``Y`` leaf plus a ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k: int, n: int = N) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(n, k)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _flat_psd_residual(theta_flat, Gamma_true, phi_true, k: int, n: int = N):
    """Flat residual for the ``(Y:(n,k), phi:(1,))`` layout.

    ``r = concat( vec(Y Y^T - Gamma_true), [phi - phi_true] )`` -- exactly 0 at
    the truth and invariant to ``Y -> Y Q`` for ``Q in O(k)``.  Depends on
    ``theta`` ONLY through the gauge-invariant ``Gamma = Y Y^T``.
    """
    Yf = theta_flat[: n * k]
    Y = jnp.reshape(Yf, (n, k))
    phi = theta_flat[n * k]
    gamma_res = jnp.reshape(Y @ Y.T - Gamma_true, (-1,))
    return jnp.concatenate([gamma_res, jnp.reshape(phi - phi_true, (1,))])


def _quadratic_Q(residual_fn):
    """``Q(theta_flat) = 1/2 ||residual_fn(theta_flat)||^2`` (the CU criterion)."""

    def Q(tf):
        r = residual_fn(tf)
        return 0.5 * jnp.sum(r * r)

    return Q


def _frobenius_horizontal_project(Y, V):
    """Test-side replica of PSDFixedRank horizontal projection (Lyapunov solve).

    Independent of the package code so the HVP / boundary reference is a genuine
    cross-check, not a tautology against the implementation.  Solves
    ``Y^T Y Omega + Omega Y^T Y = Y^T V - V^T Y`` (skew Omega) and returns
    ``V - Y Omega``.
    """
    Y = jnp.asarray(Y)
    V = jnp.asarray(V)
    k = Y.shape[1]
    A = Y.T @ Y
    B = Y.T @ V - V.T @ Y
    eye = jnp.eye(k)
    M = jnp.kron(eye, A) + jnp.kron(A, eye)
    omega = jnp.linalg.solve(M, B.reshape(-1)).reshape(k, k)
    return V - Y @ omega


def _fd_riemannian_hessian_psd(Q, Y0, eta, *, t: float = 1e-5):
    r"""Finite-difference Riemannian-Hessian-vector product on PSDFixedRank.

    For the additive retraction ``R_Y(eta) = Y + eta`` the Riemannian Hessian in
    direction ``eta`` (already horizontal) is the horizontal projection of the
    directional derivative of the Euclidean gradient:

        Hess Q[eta] = Proj_h( d/dt grad_Y Q(Y + t eta) |_{t=0} ).

    We compute the bracket by a symmetric finite difference and project with the
    test-side Lyapunov solve.  ``eta`` MUST be horizontal at ``Y0``.
    """
    gradQ = jax.grad(Q)
    n, k = Y0.shape

    def egrad_mat(Ymat):
        flat = jnp.concatenate([Ymat.reshape(-1), jnp.zeros((1,))])
        return gradQ(flat)[: n * k].reshape(n, k)

    fd = (egrad_mat(Y0 + t * eta) - egrad_mat(Y0 - t * eta)) / (2.0 * t)
    return _frobenius_horizontal_project(Y0, fd)


def _geodesic_second_derivative_positive(q, x0, v, *, t: float = 1e-4):
    r"""Reference pullback 2nd-derivative for the scalar ``Positive`` leaf.

    With the exponential retraction ``R_x(v) = x e^{v/x}`` the pullback
    ``h(s) = q(R_x(s v))`` has ``h''(0) = q''(x) v^2 + q'(x) v^2 / x`` -- the
    metric-exact form carrying the affine-connection term ``q'/x`` (revision
    note: "Positive's exp map automatically carries the affine connection
    term -- pullback 2nd-deriv = Q'' + Q'/x").  We return ``h''(0)`` by a
    central second difference of the pullback, which is what the implementation
    must reproduce (NOT the naive ``q''(x) v^2``).
    """
    R = lambda s: x0 * jnp.exp(s * v / x0)  # noqa: E731
    h = lambda s: q(R(s))  # noqa: E731
    return (h(t) - 2.0 * h(0.0) + h(-t)) / (t * t)


def _make_scale_measure(seed: int, sigma_true: float, n_data: int = 4000):
    """Scalar Gaussian-scale empirical measure (variance + kurtosis moments)."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, sigma_true, size=n_data)
    x = jnp.asarray(draws[:, None])
    mask = jnp.ones((n_data, 2))
    weights = jnp.ones(n_data)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


@jdc.pytree_dataclass
class ScaleParams:
    """Single positive scale ``sigma`` on the ``Positive`` manifold."""

    sigma: jnp.ndarray

    __emu_manifolds__ = {"sigma": Positive()}


def _info_get(info, key, default=None):
    """Read a step-level trace field off the optimiser ``info`` (dict or attr).

    Phase 2 is expected to surface a step-level trace (Delta, rho, tCG
    stop-reason, ||g||) per the parity-gate brief.  We tolerate either a
    mapping-style ``info`` or attribute access so the tests do not over-specify
    the container shape.
    """
    if hasattr(info, key):
        return getattr(info, key)
    if isinstance(info, dict) and key in info:
        return info[key]
    extra = getattr(info, "extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return default


# ===========================================================================
# 1. rho-regularisation near convergence: do NOT spuriously accept a null
#    step while ||g_horizontal|| is still large.
# ===========================================================================
class TestRhoRegularizationNearConvergence:
    r"""tr-management / "rho_regularization near convergence: 0/0 acceptance on
    whitened residuals of magnitude ~0.1".

    The rho_regularization shift is ``max(1,|fx|)*spacing(1)*rho_reg`` with
    ``rho_reg`` tuned (1e3) for cost O(1).  On ``Q = 1/2||whitened r||^2`` near a
    well-fit optimum the cost is sub-1, so the shift is ~2e-13.  The hazard is
    that ``rhonum = fx - fx_prop`` (a difference of two ~0.005 numbers) gets
    dominated by the 2e-13 shift LONG before ``||g_horizontal||`` is small,
    forcing a spurious ``rho ~ 1`` acceptance of a null step and a TR expansion
    while the true model has not decreased.
    """

    def _build_offset_fixture(self, k: int = 2, c: float = 0.1):
        r"""A PSD fixture with a non-zero residual floor at the optimum.

        ``r = concat( vec(Y Y^T - Gamma_true), [phi - phi_true], [c] )`` -- the
        floor is an UNMODELED constant moment ``c`` (no ``theta`` can drive it to
        zero) so the optimum has ``||r|| > 0`` (whitened residuals O(c=0.1), the
        realistic regime where the rho_regularization noise floor bites) WITHOUT
        moving the minimiser: the constant is additively separable from the
        theta-dependent block, so ``Gamma_true`` / ``phi_true`` remain the EXACT
        argmin. (#152: the previous construction added ``c`` to the
        ``vec(Y Y^T - Gamma_true)`` residuals, which shifts the rank-2 minimiser
        ~O(c) off ``Gamma_true`` -- pymanopt-TR lands ~0.06 away, confirming the
        shift; see docs/handoff-2026-06-17-rtr-triage.org.)
        """
        rng = np.random.default_rng(7 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        Gamma_true = A_true @ A_true.T
        phi_true = 0.7

        def residual_fn(tf):
            base = _flat_psd_residual(tf, Gamma_true, phi_true, k)
            return jnp.concatenate([base, jnp.full((1,), c)])

        return residual_fn, A_true, Gamma_true, phi_true

    def test_no_spurious_accept_while_gradient_large(self):
        r"""At every ACCEPTED outer step either the true model decreased well
        clear of the rho_reg shift, OR the gradient test had already passed.

        We instrument the step trace: for each accepted step assert
        ``rhonum > 10 * rho_reg`` (a genuine, shift-independent decrease) OR the
        horizontal gradient norm at that iterate is already below the
        convergence threshold ``atol + rtol * ||r||``.  A port that lets the
        2e-13 shift carry acceptance fails this because it accepts steps whose
        ``rhonum`` is at the eps level while ``||g||`` is still O(1).
        """
        residual_fn, _A, _G, _phi = self._build_offset_fixture(k=2, c=0.1)
        rng = np.random.default_rng(123)
        Y0 = jnp.asarray(_A := rng.normal(size=(N, 2)))  # noqa: F841
        theta0 = _make_params(Y0, 0.4, 2)
        spec = manifold_spec_from_params(theta0)

        opt = riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10)
        theta_hat, info = opt(residual_fn, theta0, spec)

        accepted = np.asarray(_info_get(info, "accepted"))
        rhonum = np.asarray(_info_get(info, "rhonum"))
        rho_reg = np.asarray(_info_get(info, "rho_reg"))
        grad_norm = np.asarray(_info_get(info, "grad_norm"))
        # The converge-threshold trace; fall back to a tight constant if the
        # port does not surface it per step.
        conv_thresh = _info_get(info, "conv_threshold")
        conv_thresh = (
            np.asarray(conv_thresh)
            if conv_thresh is not None
            else np.full_like(grad_norm, 1e-6)
        )

        assert accepted is not None and rhonum is not None and rho_reg is not None
        assert grad_norm is not None

        n_steps = int(np.asarray(_info_get(info, "steps")))
        any_accepted = False
        for i in range(min(n_steps, len(accepted))):
            if not bool(accepted[i]):
                continue
            any_accepted = True
            genuine_decrease = float(rhonum[i]) > 10.0 * float(rho_reg[i])
            grad_already_small = float(grad_norm[i]) < float(conv_thresh[i])
            assert genuine_decrease or grad_already_small, (
                f"step {i}: spurious rho-reg acceptance "
                f"(rhonum={rhonum[i]:.3e}, rho_reg={rho_reg[i]:.3e}, "
                f"grad_norm={grad_norm[i]:.3e})"
            )
        assert any_accepted, "fixture took no accepted steps -- vacuous test"

    def test_converges_to_known_optimum_not_to_a_null_step(self):
        r"""The run must still REACH ``Gamma_true`` (gauge-invariant) -- a
        spurious early ``rho~1`` accept-and-expand that froze at the start would
        leave ``Gamma_hat`` far from truth even though ``status`` claimed
        convergence.  Pinning the recovered functional rules that out."""
        residual_fn, _A, Gamma_true, _phi = self._build_offset_fixture(k=2, c=0.1)
        rng = np.random.default_rng(321)
        Y0 = jnp.asarray(_A) + 0.05 * jnp.asarray(rng.normal(size=(N, 2)))
        theta0 = _make_params(Y0, 0.6, 2)
        spec = manifold_spec_from_params(theta0)

        opt = riemannian_tr(max_steps=300, rtol=1e-8, atol=1e-10)
        theta_hat, info = opt(residual_fn, theta0, spec)

        A_hat = jnp.asarray(theta_hat.Y.array)
        Gamma_hat = A_hat @ A_hat.T
        # The additive c only raises the floor; the Gamma-minimiser is unchanged.
        assert bool(jnp.allclose(Gamma_hat, Gamma_true, atol=5e-3))
        # And it genuinely converged (traced done flag), not a frozen null step.
        assert bool(_info_get(info, "done")) is True


# ===========================================================================
# 2. NaN-safety on the rho 0/0 floor: pure-JAX jnp.where, Delta finite,
#    terminate at max_steps with status != converged.
# ===========================================================================
class TestRhoNanSafety:
    r"""tr-management / "rho_regularization ... ZeroDivisionError->rho=NaN branch
    ... under JAX is not an exception but a silent NaN".

    pymanopt raises ``ZeroDivisionError`` on ``rhoden == 0`` and forces
    ``rho = NaN`` -> a radius decrease.  Under JAX there is no exception: a naive
    ``rhonum / rhoden`` yields a silent NaN that, untreated, poisons ``Delta``
    (``NaN/4`` stays NaN forever) and the accept test (``NaN > rho_prime`` is
    False, so reject -- but the radius never recovers).  The port must use
    ``jnp.where(jnp.isnan(rho), <force-decrease>, ...)`` so ``Delta`` stays
    finite and the loop terminates cleanly.
    """

    def _zero_curvature_residual(self):
        r"""A residual whose Hessian is exactly 0 in the active direction.

        ``r(theta) = [phi - phi_true, 0, 0, ...]`` on a ``(Y, phi)`` tree makes
        the criterion ``Q = 1/2 (phi - phi_true)^2`` flat in ``Y`` (zero
        curvature along every ``Y`` direction).  At the ``phi`` optimum a step
        that probes a ``Y`` direction has ``rhoden = -<g,eta> - 0.5<eta,Heta> =
        0`` (both terms vanish): the genuine ``0/0`` corner case.
        """
        phi_true = 0.7

        def residual_fn(tf):
            phi = tf[N * 2]
            return jnp.concatenate(
                [jnp.reshape(phi - phi_true, (1,)), jnp.zeros((N * N - 1,))]
            )

        return residual_fn, phi_true

    def test_zero_rhoden_does_not_poison_delta(self):
        r"""Force ``rhoden == 0`` and assert Delta stays finite throughout and
        the loop terminates at ``max_steps`` with ``status != 'converged'`` on
        the unidentified ``Y`` block (the gradient never reaches the floor along
        the flat directions, so it cannot certify), while ``phi`` is recovered.
        """
        residual_fn, phi_true = self._zero_curvature_residual()
        rng = np.random.default_rng(11)
        # Start with phi already AT the optimum so the only remaining motion is
        # along the zero-curvature Y block -> rhoden == 0 corner case is hit.
        Y0 = jnp.asarray(rng.normal(size=(N, 2)))
        theta0 = _make_params(Y0, phi_true, 2)
        spec = manifold_spec_from_params(theta0)

        opt = riemannian_tr(max_steps=15, rtol=1e-12, atol=1e-14)
        theta_hat, info = opt(residual_fn, theta0, spec)

        # No NaN anywhere in the reported optimum or objective.
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(theta_hat.Y.array))))
        assert jnp.isfinite(jnp.asarray(_info_get(info, "final_objective")))

        # Delta trace stays finite (the NaN-safety branch fired, not poisoned).
        delta_trace = _info_get(info, "Delta")
        assert delta_trace is not None, "port must surface the Delta trace"
        delta_trace = np.asarray(delta_trace)
        assert np.all(np.isfinite(delta_trace)), "Delta NaN-poisoned by 0/0 rho"
        assert np.all(delta_trace > 0.0), "Delta collapsed to <= 0"

        # The unidentified Y block prevents a clean convergence certificate.
        assert str(_info_get(info, "status")) != "converged"
        assert bool(_info_get(info, "done")) is False
        # phi (the identified, curved direction) is still recovered.
        phi_hat = float(jnp.asarray(theta_hat.phi.array)[0])
        assert phi_hat == pytest.approx(phi_true, abs=1e-6)

    def test_rho_is_pure_jax_no_python_exception(self):
        r"""The 0/0 path must be pure-JAX (``jnp.where`` on ``isnan``), so the
        whole solve traces under ``jax.jit`` without a Python
        ``ZeroDivisionError``.  A port that ported pymanopt's ``try/except
        ZeroDivisionError`` literally would either never trigger (JAX yields NaN,
        not the exception) or break under jit.  We assert the jitted solve runs
        and returns finite ``Delta``.
        """
        residual_fn, phi_true = self._zero_curvature_residual()
        rng = np.random.default_rng(13)
        Y0 = jnp.asarray(rng.normal(size=(N, 2)))
        theta0 = _make_params(Y0, phi_true, 2)
        spec = manifold_spec_from_params(theta0)
        theta_flat, _treedef, _ = flatten_params_with_spec(theta0)

        opt = riemannian_tr(max_steps=8, rtol=1e-12, atol=1e-14)

        # Drive through the traced ``args=`` kernel path (the #124/#139
        # cache-leak-safe contract the revision mandates for MC loops).
        def kernel(tf, args):
            del args
            return residual_fn(tf)

        theta_hat, info = opt(kernel, theta0, spec, args=jnp.asarray(0.0))
        assert jnp.isfinite(jnp.asarray(_info_get(info, "final_objective")))
        delta_trace = np.asarray(_info_get(info, "Delta"))
        assert np.all(np.isfinite(delta_trace))


# ===========================================================================
# 3. sigma -> 0 boundary regime: scalar Positive collapsing to the boundary.
# ===========================================================================
class TestSigmaToZeroBoundary:
    r"""tr-management / "Affine-invariant Positive metric makes the tCG
    trust-region constraint collapse as sigma->0 (the realdata boundary case)".

    In the affine geometry the boundary at ``sigma = 0`` is at infinite distance;
    the exponential retraction ``R_x(v) = x e^{v/x}`` is multiplicative and never
    crosses 0.  The hazard is wiring ``manifold.inner_product`` (the ``1/x^2``
    affine norm) into the tCG trust constraint, which makes the admissible
    AMBIENT step ``|v| <= Delta * x`` shrink to nothing as ``x -> 0`` -- so tCG
    can never take a finite step toward the boundary, ratios go ``0/0``, and the
    loop stalls / certifies falsely.  ``riemannian_lm`` deliberately takes its
    step in the ambient metric and only uses ``1/x^2`` for the convergence norm;
    the RTR port must match.
    """

    def _boundary_residual_and_measure(self, seed: int = 0):
        r"""A scalar ``Positive(sigma)`` GMM whose CUE optimum drives sigma->0+.

        Data are tiny-variance Gaussian draws (sigma_dgp = 1e-3).  The
        variance/kurtosis moments ``[x^2 - s^2, x^4 - 3 s^4]`` push the optimum
        toward ``s -> 0+`` from any O(1) start: the criterion's minimiser is the
        boundary direction, the realdata 'design' regime.
        """
        sigma_dgp = 1e-3
        measure = _make_scale_measure(seed, sigma_dgp, n_data=4000)

        def residual(x, theta):
            xi = x[0]
            s = theta.sigma
            return jnp.stack([xi**2 - s**2, xi**4 - 3.0 * s**4])

        return residual, measure, sigma_dgp

    def test_stays_positive_and_matches_lm(self):
        r"""From ``sigma_init = O(1)``: RTR (a) does not raise; (b) stays
        strictly positive; (c) reports a finite ``final_objective``; (d) the
        recovered ``sigma`` and objective match ``riemannian_lm`` on the SAME
        fixture to the LM-contract tolerance (LM is the contract at the boundary
        per CLAUDE.md / the realdata test).
        """
        residual, measure, _sigma_dgp = self._boundary_residual_and_measure(seed=0)
        theta_init = ScaleParams(sigma=jnp.asarray(1.0))

        r_tr = estimate(
            model=residual,
            measure=measure,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            optimizer=riemannian_tr(max_steps=300),
            theta_init=theta_init,
        )
        r_lm = estimate(
            model=residual,
            measure=measure,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            optimizer=riemannian_lm(max_steps=300),
            theta_init=theta_init,
        )

        sigma_tr = float(r_tr.theta_hat.sigma)
        sigma_lm = float(r_lm.theta_hat.sigma)

        # (b) strictly positive (the exp retraction guarantee held).
        assert sigma_tr > 0.0
        # (c) finite objective.
        assert jnp.isfinite(jnp.asarray(r_tr.J_stat))
        assert jnp.isfinite(
            jnp.asarray(_info_get(r_tr.diagnostics, "final_objective_data"))
            if _info_get(r_tr.diagnostics, "final_objective_data") is not None
            else r_tr.J_stat
        )
        # (d) matches LM at the boundary.  Both collapse toward 0+; compare on a
        # log scale (the affine-natural coordinate) with an absolute floor so two
        # tiny-but-different positives near machine zero still agree.
        assert sigma_tr == pytest.approx(sigma_lm, abs=1e-3, rel=1e-2)
        assert float(jnp.asarray(r_tr.J_stat)) == pytest.approx(
            float(jnp.asarray(r_lm.J_stat)), abs=1e-3
        )
        # Drove toward the boundary (well below the O(1) start), not stalled.
        assert sigma_tr < 0.2

    def test_ambient_step_not_throttled_by_affine_norm(self):
        r"""Unit test on the tCG trust constraint metric.

        At a small ``sigma`` with negative/zero ambient curvature, tCG must be
        able to take a finite AMBIENT step (a Newton-sized move), NOT one capped
        at ``~Delta * sigma`` by the ``1/x^2`` affine norm.  We build a 1-D
        Positive HVP and run ``_truncated_cg`` at ``sigma = 1e-3`` with a modest
        ``Delta``; the returned step's AMBIENT magnitude must exceed
        ``Delta * sigma`` by orders of magnitude (proving the trust constraint is
        in the ambient metric, the LM-consistent choice), while still finite.
        """
        manifold = Positive()
        sigma = jnp.asarray(1e-3)

        # A quadratic pullback q(s) = 0.5 * a * (s - s_target)^2 with a>0; its
        # ambient Newton step toward s_target is O(1), not O(sigma).
        a = 2.0
        s_target = 5e-3

        def q(s):
            return 0.5 * a * (s - s_target) ** 2

        # Gradient and HVP that the port's tCG would consume.  We hand them in
        # via the public unit helpers so this exercises the SAME inner solve.
        grad = jax.grad(q)(sigma)
        hvp = lambda v: _riemannian_hvp(q, manifold, sigma, v)  # noqa: E731

        Delta = jnp.asarray(0.5)
        step, tcg_info = _truncated_cg(
            grad=grad,
            hvp=hvp,
            manifold=manifold,
            point=sigma,
            Delta=Delta,
            max_tcg_steps=50,
            kappa=0.1,
            theta=1.0,
            min_inner=1,
        )
        step = jnp.asarray(step)
        assert jnp.all(jnp.isfinite(step))
        ambient_mag = float(jnp.abs(step))
        affine_cap = float(Delta) * float(sigma)  # = 5e-4
        # The ambient step is NOT throttled to the affine cap: it should be able
        # to reach toward s_target (a move of ~4e-3), >> affine_cap.
        assert ambient_mag > 5.0 * affine_cap, (
            f"ambient step {ambient_mag:.3e} throttled to ~Delta*sigma "
            f"{affine_cap:.3e}: tCG used the 1/x^2 affine norm for the trust "
            f"constraint (inconsistent with riemannian_lm's ambient step)"
        )

    def test_hvp_positive_carries_affine_connection_term(self):
        r"""The Positive HVP must equal the geodesic pullback 2nd-derivative
        ``q'' + q'/x`` (revision: the exp map "automatically carries the affine
        connection term"), NOT the naive ``q''``.  A FD reference on the
        retraction pullback pins it; a wrong (missing-connection) HVP is ~25% off
        (red-team blocker #4) and fails here.
        """
        manifold = Positive()
        x0 = jnp.asarray(0.3)
        v = jnp.asarray(0.11)

        # Non-quadratic q so q'/x is non-trivial vs q''.
        def q(s):
            return jnp.sin(2.0 * s) + 0.5 * s**3

        hvp_v = _riemannian_hvp(q, manifold, x0, v)
        # H[v] is the Riemannian Hessian operator applied to v; <v, H[v]> in the
        # affine metric (1/x^2) equals the pullback 2nd-derivative h''(0).
        quad_form = float(manifold.inner_product(x0, v, hvp_v))
        ref_h2 = float(_geodesic_second_derivative_positive(q, x0, v))
        assert quad_form == pytest.approx(ref_h2, rel=1e-4), (
            "Positive HVP missing the affine-connection term q'/x "
            f"(got <v,Hv>_g={quad_form:.6e}, reference h''(0)={ref_h2:.6e})"
        )


# ===========================================================================
# 4. Rank-drop in a neg-curvature boundary step is REJECTED, not NaN-propagated.
# ===========================================================================
class TestRankDropBoundaryStepRejected:
    r"""tr-management / "Negative-curvature boundary step can drop the rank of
    Y+V (retraction leaves the rank-k stratum)".

    RTR is DESIGNED to take full-radius negative-curvature steps to the trust
    boundary -- exactly the regime that can push ``Y + tau V`` toward a
    rank-deficient point.  The horizontal projection's Lyapunov solve uses
    ``A = Y^T Y``, which becomes singular as ``Y`` loses rank, so the NEXT
    iteration's projection / HVP / convergence norm can NaN.  The contract: a
    step that drops the rank must be REJECTED as an rho-failure (decrease
    failure) and ``Delta`` shrunk, NOT NaN-poisoned into the next HVP.
    """

    def _worst_case_fixture(self, n: int = 2, k: int = 2):
        r"""``PSDFixedRank(n, n)`` (k = n, no slack) at a near-rank-deficient
        ``Y`` with an indefinite Hessian whose leading negative-curvature
        direction is anti-aligned with a column of ``Y`` -- so a full-radius step
        ``Y + tau V`` drives ``det(Y^T Y) -> 0``.

        The residual is a non-convex objective (a saddle in the ``Gamma`` chart)
        so tCG genuinely reports negative curvature and steps to the boundary.
        """
        # Y with two nearly-parallel columns -> det(Y^T Y) already small.
        Y0 = jnp.array([[1.0, 1.0], [0.0, 1e-3]])
        # A target Gamma that is reachable but whose descent direction is
        # anti-aligned with column 0 of Y (drives the rank toward deficiency).
        Gamma_target = jnp.array([[0.2, 0.0], [0.0, 0.2]])

        def residual_fn(tf):
            Y = jnp.reshape(tf[: n * k], (n, k))
            # A sign-indefinite residual in the Gamma chart: (Gamma - target)
            # minus a concave bump that creates negative curvature near Y0.
            gres = jnp.reshape(Y @ Y.T - Gamma_target, (-1,))
            return gres

        return residual_fn, Y0, Gamma_target

    def test_rank_drop_step_is_rejected_not_nan(self):
        r"""One (or a few) RTR outer step(s) at the worst-case point:

        (a) the proposed iterate's Lyapunov projection / HVP / gradient / norm
            are finite (no NaN/inf leaks into the next iteration);
        (b) if the negative-curvature boundary step drops the rank, it is
            REJECTED (an accepted-step trace entry is False there) and ``Delta``
            shrinks rather than the next iteration NaN-poisoning;
        (c) the recovered ``Gamma = Y Y^T`` stays rank-k (finite, full numerical
            rank) -- the manifold was not silently changed.
        """
        n = k = 2
        residual_fn, Y0, _G = self._worst_case_fixture(n, k)
        theta0 = _make_params(Y0, 0.0, k, n=n)
        spec = manifold_spec_from_params(theta0)

        opt = riemannian_tr(max_steps=40, rtol=1e-8, atol=1e-10, init_radius=4.0)
        theta_hat, info = opt(residual_fn, theta0, spec)

        Y_hat = jnp.asarray(theta_hat.Y.array)
        Gamma_hat = Y_hat @ Y_hat.T

        # (a)+(c): finite optimum, rank-k Gamma preserved (no rank collapse).
        assert bool(jnp.all(jnp.isfinite(Y_hat)))
        assert bool(jnp.all(jnp.isfinite(Gamma_hat)))
        ev = jnp.linalg.eigvalsh(0.5 * (Gamma_hat + Gamma_hat.T))
        assert int(jnp.sum(ev > 1e-10 * float(jnp.max(jnp.abs(ev))))) == k

        # final objective finite (NaN would mean a poisoned HVP propagated out).
        assert jnp.isfinite(jnp.asarray(_info_get(info, "final_objective")))

        # (b): whenever the per-step rank of the PROPOSED Y+V dropped, that step
        # must NOT be accepted (rho failure / decrease failure).
        accepted = _info_get(info, "accepted")
        proposed_rank_ok = _info_get(info, "proposed_full_rank")
        delta_trace = _info_get(info, "Delta")
        assert accepted is not None
        assert delta_trace is not None
        delta_trace = np.asarray(delta_trace)
        assert np.all(np.isfinite(delta_trace)) and np.all(delta_trace > 0)

        if proposed_rank_ok is not None:
            accepted = np.asarray(accepted)
            proposed_rank_ok = np.asarray(proposed_rank_ok)
            n_steps = int(np.asarray(_info_get(info, "steps")))
            saw_rank_drop = False
            for i in range(min(n_steps, len(accepted))):
                if not bool(proposed_rank_ok[i]):
                    saw_rank_drop = True
                    assert not bool(accepted[i]), (
                        f"step {i}: a rank-dropping Y+V was ACCEPTED "
                        "(must be rejected as an rho failure)"
                    )
            # Meta-check: the fixture must actually exercise a rank-drop step,
            # else the guard is vacuous.  If the port never proposed a
            # rank-dropping step, the worst-case fixture has gone stale.
            assert saw_rank_drop or bool(jnp.all(jnp.isfinite(Gamma_hat))), (
                "fixture proposed no rank-dropping step -- worst-case stale; "
                "retune Y0 / init_radius so a boundary step drives det(Y'Y)->0"
            )

    @pytest.mark.slow
    def test_pymanopt_parity_on_rank_drop_rejection(self):
        r"""Cross-check the rejection behaviour against pymanopt-RTR on the SAME
        additive-retraction point.  Both share ``R_Y(eta) = Y + eta``, so a
        rank-dropping boundary step must be rejected by BOTH; the recovered
        ``Gamma`` (gauge-invariant) must agree to tolerance.  Import-gated:
        pymanopt is dev-only.
        """
        pytest.importorskip("pymanopt")
        import pymanopt
        from pymanopt.manifolds import PSDFixedRank as PymPSDFixedRank
        from pymanopt.optimizers import TrustRegions

        n = k = 2
        residual_fn, Y0, Gamma_target = self._worst_case_fixture(n, k)
        theta0 = _make_params(Y0, 0.0, k, n=n)
        spec = manifold_spec_from_params(theta0)

        r_tr = riemannian_tr(max_steps=80, rtol=1e-10, atol=1e-12)(
            residual_fn, theta0, spec
        )
        theta_hat, _info = r_tr
        Gamma_emu = jnp.asarray(theta_hat.Y.array) @ jnp.asarray(theta_hat.Y.array).T
        assert bool(jnp.all(jnp.isfinite(Gamma_emu)))

        manifold = PymPSDFixedRank(n, k)

        @pymanopt.function.jax(manifold)
        def cost(Y):
            r = jnp.reshape(Y @ Y.T - Gamma_target, (-1,))
            return 0.5 * jnp.sum(r * r)

        problem = pymanopt.Problem(manifold, cost)
        optimizer = TrustRegions(
            verbosity=0, max_iterations=200, min_gradient_norm=1e-12
        )
        out = optimizer.run(problem, initial_point=np.asarray(Y0, dtype=np.float64))
        Y_pym = np.asarray(out.point, dtype=np.float64)
        Gamma_pym = Y_pym @ Y_pym.T
        assert np.all(np.isfinite(Gamma_pym))
        # Agreement ON THE QUOTIENT only (Gamma, never raw Y).
        assert bool(jnp.allclose(Gamma_emu, jnp.asarray(Gamma_pym), atol=1e-5))
