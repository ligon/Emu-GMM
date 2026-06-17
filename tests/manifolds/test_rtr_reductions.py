r"""TEST-FIRST reductions + Product-structure suite for the JAX-native
Riemannian Trust Region optimiser (#152, theme: ``reductions``).

This file is written BEFORE the implementation exists. It imports
``emu_gmm.manifolds.riemannian_tr`` (the Phase-2 deliverable) via
:func:`pytest.importorskip`, so the WHOLE module SKIPs cleanly (it does NOT
error at collection) until Phase 2 lands the module. That is the intended
Phase-1 state: this suite is the executable contract Phase 2 builds against.

Theme: **reductions + Product structure**. RTR is a strict generalisation of
the existing Euclidean / Positive / Riemannian-LM machinery, so on the
sub-problems where those simpler tools are already correct, RTR must REDUCE
to them exactly. And the Product machinery -- the full-tree HVP and the
per-leaf metric -- must not silently degrade to a block-diagonal /
flat-metric approximation. The risks pinned (one strong, non-vacuous test
per risk; each class docstring names its risk):

* R1 all-Euclidean reduction: on a smooth NON-convex (Rosenbrock-shaped)
  objective, RTR's final point matches
  ``scipy.optimize.minimize(method='trust-ncg')`` to 1e-8. A genuinely
  non-convex fixture forces the negative-curvature / tCG machinery to be
  exercised (a convex quadratic would pass vacuously).
* R2 scalar Positive reduction: ``theta_hat`` matches ``riemannian_lm`` to
  recovery tolerance (same optimum, NOT bitwise -- RTR has no lambda-floor),
  and the post-fit ``Sigma_theta`` / ``J_stat`` / ``J_dof`` are identical;
  plus a pin that the affine ``1/x^2`` metric enters the reported gradient
  norm at ``sigma_hat != 1``.
* R3 Positive-leaf HVP carries the affine connection term ``Q'' + Q'/x``
  (NOT ``Q''``): asserted equal to the finite-difference geodesic second
  derivative of ``x e^{t v / x}``, distinguishable from the naive ``Q''``
  recipe, with a negative-curvature SIGN gate where ``sign(Q'')`` and
  ``sign(Q'' + Q'/x)`` disagree.
* R4 (blocker) ``manifold_spec`` leaf-ordering: a Product whose factor order
  differs from the PyTree-leaf order must round-trip and its HVP must match
  the dense full-tree reference. An EQUAL-dim heterogeneous fixture
  (``PSDFixedRank(2, 1)`` = 2 next to ``Euclidean(2)`` = 2) makes a
  shape-only check blind -- only a value check catches mis-pairing.
* R5 (blocker) block-diagonal HVP must NOT drop cross-leaf coupling ``S``:
  an explicit ``Gamma`` x Euclidean cross term, ``eta`` on BOTH leaves,
  HVP == dense reference; and a PSD-only ``eta`` produces a NONZERO
  Euclidean-block HVP (which a block-diagonal slice would zero).
* R6 (blocker) tCG metric is per-leaf non-identity (Positive ``1/x^2``):
  ``_truncated_cg`` radius / curvature must use ``Product.inner_product``,
  not flat Frobenius. The Positive coordinate sits at ``x = 5``
  (``1/x^2 = 0.04``) so the step differs both from the flat-metric step and
  from the ``x = 1`` step.

Intended Phase-2 API (issue #152 revision; mirrored from the sibling RTR
test files ``test_rtr_hvp.py`` / ``test_rtr_tcg.py`` / ``test_rtr_integration.py``):

    opt = riemannian_tr(max_steps=..., rtol=..., atol=..., rho_prime=...,
                        kappa=..., theta=..., min_inner=..., max_tcg_steps=...,
                        max_radius=..., init_radius=...)
    theta_hat_pytree, info = opt(residual_fn, theta_init, manifold_spec)
    # unit HVP (pytree-native): Q maps a manifold point -> scalar.
    _riemannian_hvp(Q, manifold, point, eta) -> H[eta]
    # unit tCG (flat): -> (eta_flat, Heta_flat, info)
    _truncated_cg(hvp, grad_flat, manifold_spec, point_flat, Delta, *,
                  theta, kappa, min_inner, max_tcg_steps)

If Phase 2 picks a different unit-helper signature, edit the thin ``_hvp`` /
``_tcg`` shims here in ONE place -- the reference computations and the
assertions stay.
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
from emu_gmm.covariance import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.manifolds import Euclidean, Positive, Product, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

# RED-until-Phase-2: the implementation module does not exist yet. This
# importorskip makes the WHOLE file SKIP (not error) at collection, with the
# module name in the reason -- the intended Phase-1 signal. Once Phase 2 lands
# ``emu_gmm.manifolds.riemannian_tr`` these tests go live.
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
riemannian_tr = riemannian_tr_mod.riemannian_tr
_riemannian_hvp = riemannian_tr_mod._riemannian_hvp
_truncated_cg = riemannian_tr_mod._truncated_cg


# ---------------------------------------------------------------------------
# Thin shims around the Phase-2 unit helpers. If Phase 2 picks a different
# signature, edit ONLY these two functions. Everything below builds its
# reference from first principles and compares against these.
# ---------------------------------------------------------------------------
def _hvp(Q, manifold, point, eta):
    """``H[eta]`` = Riemannian HVP of scalar ``Q`` at ``point`` along ``eta``.

    ``H[eta]`` is the Euclidean Hessian of the retraction-pullback
    ``Q(R_point(eta))`` at ``eta = 0`` (projected to horizontal for the PSD
    leaf), per the #152 revision.
    """
    return _riemannian_hvp(Q, manifold, point, eta)


def _tcg(hvp, grad_flat, spec, point_flat, Delta, **kw):
    """Steihaug-Toint inner solve -> ``(eta_flat, Heta_flat, info)``."""
    return _truncated_cg(hvp, grad_flat, spec, point_flat, Delta, **kw)


# ===========================================================================
# R1 -- all-Euclidean reduction vs scipy trust-ncg on a NON-convex objective.
# ===========================================================================
@jdc.pytree_dataclass
class _EucParams:
    """A single ``Euclidean(d)`` leaf (the all-Euclidean reduction case)."""

    v: ManifoldLeaf


def _euc_params(v: jnp.ndarray) -> _EucParams:
    v = jnp.asarray(v)
    return _EucParams(v=ManifoldLeaf(v, Euclidean(int(v.shape[0]))))


def _rosenbrock_residual(d: int):
    r"""Residual whose ``1/2 ||r||^2`` is the (generalised) Rosenbrock function.

    The 2-D Rosenbrock ``f(x, y) = (a - x)^2 + b (y - x^2)^2`` is the squared
    norm of ``r = [a - x, sqrt(b)(y - x^2)]``; the N-D chained generalisation
    stacks ``[a - x_i, sqrt(b)(x_{i+1} - x_i^2)]``. This is SMOOTH but
    strongly NON-convex (a curved valley with indefinite Hessian off the
    valley floor) -- the regime where tCG must report negative curvature and
    RTR's trust-region globalisation matters. A convex quadratic would let a
    broken negative-curvature branch pass vacuously.
    """
    a, b = 1.0, 100.0

    def residual_fn(theta_flat):
        x = theta_flat
        r_lin = a - x[:-1]
        r_curv = jnp.sqrt(b) * (x[1:] - x[:-1] ** 2)
        return jnp.concatenate([r_lin, r_curv])

    return residual_fn


class TestEuclideanReductionVsScipy:
    r"""R1 -- on an all-Euclidean tree RTR reduces to a textbook trust-region
    Newton-CG solve, so its minimiser matches ``scipy.optimize.minimize(
    method='trust-ncg')`` to 1e-8 on a NON-convex (Rosenbrock) objective.

    The reduction is exact: for ``Euclidean`` the retraction is ``x + v``
    (so the pullback Hessian IS the ambient Hessian) and the metric is the
    identity (so tCG IS ordinary Steihaug-Toint CG). Matching scipy's
    independent trust-ncg implementation to 1e-8 on a curved, non-convex
    valley confirms the whole stack -- HVP, tCG, trust-radius updates --
    reduces correctly. The objective is non-convex by construction so the
    negative-curvature path is genuinely exercised (asserted via a separate
    indefinite-Hessian check so the fixture is not silently convex).
    """

    @pytest.mark.parametrize("d", [2, 4])
    def test_matches_scipy_trust_ncg(self, d):
        scipy_opt = pytest.importorskip("scipy.optimize")
        residual_fn = _rosenbrock_residual(d)

        def f(x):
            r = residual_fn(jnp.asarray(x))
            return 0.5 * jnp.sum(r * r)

        # A start OFF the valley floor (and away from the minimiser at all-ones)
        # so the path crosses the strongly-curved region.
        x0 = jnp.asarray(np.linspace(-1.2, 1.0, d))

        # Fixture sanity: the Hessian at x0 is genuinely INDEFINITE (so the
        # non-convex / negative-curvature machinery is actually exercised).
        H0 = np.asarray(jax.hessian(lambda x: f(x))(x0))
        evals0 = np.linalg.eigvalsh(0.5 * (H0 + H0.T))
        assert evals0.min() < -1e-6, "fixture is locally convex -- R1 vacuous"

        theta_init = _euc_params(x0)
        spec = manifold_spec_from_params(theta_init)
        opt = riemannian_tr(max_steps=500, rtol=1e-10, atol=1e-12)
        theta_hat, info = opt(residual_fn, theta_init, spec)
        assert bool(info.done), "RTR did not converge on the Rosenbrock fixture"
        x_rtr = np.asarray(theta_hat.v.array)

        # scipy trust-ncg reference: same objective, gradient + Hessian-vector
        # product from jax so the two solve the identical problem.
        grad = jax.grad(f)

        def hessp(x, p):
            return np.asarray(jax.jvp(grad, (jnp.asarray(x),), (jnp.asarray(p),))[1])

        out = scipy_opt.minimize(
            lambda x: float(f(jnp.asarray(x))),
            np.asarray(x0),
            method="trust-ncg",
            jac=lambda x: np.asarray(grad(jnp.asarray(x))),
            hessp=hessp,
            options={"gtol": 1e-12, "maxiter": 2000},
        )
        x_scipy = np.asarray(out.x)

        # Both land on the unique Rosenbrock minimiser (all-ones) and agree to
        # 1e-8 -- the all-Euclidean reduction is exact.
        np.testing.assert_allclose(x_rtr, np.ones(d), atol=1e-7)
        np.testing.assert_allclose(x_rtr, x_scipy, atol=1e-8)


# ===========================================================================
# R2 -- scalar Positive reduction: same optimum + identical post-fit inference.
# ===========================================================================
SIGMA_TRUE = 1.5
N_DATA = 5000


@jdc.pytree_dataclass
class _ScaleParams:
    """A single positive scale ``sigma`` (the ``test_estimator_positive`` DGP)."""

    sigma: jnp.ndarray

    __emu_manifolds__ = {"sigma": Positive()}


def _scale_residual(x, theta):
    """``m_0 = x^2 - sigma^2``, ``m_1 = x^4 - 3 sigma^4`` (Gaussian moments)."""
    xi = x[0]
    s = theta.sigma
    return jnp.stack([xi**2 - s**2, xi**4 - 3.0 * s**4])


def _scale_measure(seed: int = 0) -> EmpiricalMeasure:
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, SIGMA_TRUE, size=N_DATA)
    x = jnp.asarray(draws[:, None])
    mask = jnp.ones((N_DATA, 2))
    weights = jnp.ones(N_DATA)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


def _run_scale(optimizer):
    return estimate(
        model=_scale_residual,
        measure=_scale_measure(seed=0),
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimizer,
        theta_init=_ScaleParams(sigma=jnp.asarray(0.5)),
    )


class TestPositiveReductionMatchesLM:
    r"""R2 -- on the scalar ``Positive`` DGP RTR reaches the SAME optimum as
    ``riemannian_lm`` and reports identical post-fit inference.

    The two solvers differ in their inner mechanics (LM has a lambda-floor;
    RTR has a trust region + tCG), so we do NOT demand bitwise agreement on
    the iterate path. But at an interior optimum both converge to the same
    ``sigma_hat`` (recovery tolerance), and the post-fit
    ``Sigma_theta`` / ``J_stat`` / ``J_dof`` are functionals of the OPTIMUM
    only (the direct-form ``G'Lambda G`` information matrix, CLAUDE.md
    commitment 5) -- so they must be IDENTICAL up to the optimum-agreement
    floor. A separate pin checks the affine ``1/x^2`` metric enters the
    reported gradient norm (``sigma_hat != 1`` so ``1/x^2 != 1``).
    """

    def test_sigma_matches_lm(self):
        r_tr = _run_scale(riemannian_tr(max_steps=200))
        r_lm = _run_scale(riemannian_lm(max_steps=200))
        assert bool(r_tr.converged) and bool(r_lm.converged)
        s_tr = float(r_tr.theta_hat.sigma)
        s_lm = float(r_lm.theta_hat.sigma)
        # Both recover the truth, and agree with each other to the optimum
        # floor (same minimiser, different inner solver).
        assert s_tr == pytest.approx(SIGMA_TRUE, abs=0.1)
        assert s_tr == pytest.approx(s_lm, abs=1e-6)
        assert s_tr > 0.0  # positivity preserved by the exponential retraction

    def test_post_fit_inference_identical_to_lm(self):
        r"""``Sigma_theta`` / ``J_stat`` / ``J_dof`` are functionals of the
        optimum, so RTR and LM must report them identically (up to the
        same-optimum floor). A wrong RTR readout (e.g. a metric-rescaled
        Sigma or a mis-counted dof) diverges here even when sigma_hat agrees."""
        r_tr = _run_scale(riemannian_tr(max_steps=200))
        r_lm = _run_scale(riemannian_lm(max_steps=200))
        assert bool(r_tr.converged) and bool(r_lm.converged)

        # J_dof is static (M - dim_info); must be exactly equal.
        assert r_tr.J_dof == r_lm.J_dof == 1
        # J_stat is a scalar functional of the optimum.
        assert float(r_tr.J_stat) == pytest.approx(float(r_lm.J_stat), rel=1e-6)
        # Sigma_theta is the 1x1 ambient natural-scale variance (Convention B).
        sig_tr = float(r_tr.Sigma_theta.array[0, 0])
        sig_lm = float(r_lm.Sigma_theta.array[0, 0])
        assert sig_tr > 0.0
        assert sig_tr == pytest.approx(sig_lm, rel=1e-6)

    def test_affine_metric_enters_gradient_norm(self):
        r"""The reported gradient norm is the RIEMANNIAN norm under the affine
        ``1/x^2`` metric, NOT the flat ``|g|``. At the optimum ``sigma_hat``
        (!= 1, so ``1/x^2 != 1``) RTR's reported gradient norm must match what
        the LM solver reports (which also uses the affine metric via
        ``riemannian_lm.riem_norm``). A flat-metric port reports a
        systematically different (by the factor ``1/sigma_hat``) gradient norm.

        We confirm the metric is load-bearing by checking ``1/sigma_hat^2``
        genuinely differs from 1, so "the affine metric enters" is a real
        constraint here, not an identity.
        """
        r_tr = _run_scale(riemannian_tr(max_steps=200))
        assert bool(r_tr.converged)
        sigma_hat = float(r_tr.theta_hat.sigma)
        # sigma_hat is genuinely off 1, so the metric weighting is non-trivial.
        assert abs(sigma_hat - 1.0) > 0.3
        assert abs(1.0 / sigma_hat**2 - 1.0) > 0.1

        # The estimator's reported final gradient norm (affine metric).
        g_norm_reported = float(jnp.asarray(r_tr.diagnostics.final_gradient_norm))

        # The LM run uses the same affine norm, so at the shared optimum its
        # reported gradient norm must match RTR's. A flat-metric RTR would
        # report a different (1/sigma_hat-scaled) value.
        r_lm = _run_scale(riemannian_lm(max_steps=200))
        g_norm_lm = float(jnp.asarray(r_lm.diagnostics.final_gradient_norm))
        assert g_norm_reported == pytest.approx(g_norm_lm, rel=1e-4, abs=1e-8)


# ===========================================================================
# R3 -- Positive-leaf HVP carries the affine connection term Q'' + Q'/x.
# ===========================================================================
class TestPositiveHvpConnectionTerm:
    r"""R3 -- the ``Positive`` leaf HVP must equal ``Q'' + Q'/x``, NOT ``Q''``.

    The exponential retraction ``R_x(v) = x e^{v/x}`` is a SECOND-order
    retraction, so the Euclidean Hessian of the pullback ``Q(R_x(v))`` at
    ``v = 0`` automatically carries the affine connection term: the geodesic
    second derivative ``d^2/dt^2 Q(x e^{t v / x})|_0 = (Q'' + Q'/x) v^2``
    (the metric quadratic form ``g_x(H[v], v)`` with ``g_x(u, w) = uw/x^2``).
    The naive ``projection(jvp(grad Q))`` recipe (projection is identity for
    Positive) returns ``Q''`` and is WRONG -- the ~25% blocker. We pin three
    things: equality to the FD geodesic 2nd derivative, distinctness from the
    naive ``Q''`` recipe, and a sign gate where ``sign(Q'')`` and
    ``sign(Q'' + Q'/x)`` disagree (so a ``Q''`` port mis-signs curvature).
    """

    # (x, v) pairs spanning x below / at / above 1 so the 1/x weighting bites.
    CASES = [(1.7, 0.9), (0.4, -0.6), (3.5, 1.3), (0.8, 0.2)]

    @staticmethod
    def _Q_connection():
        # Q(x) = 0.5 (x - c)^2 + a log(x): Q' = (x - c) + a/x,
        # Q'' = 1 - a/x^2;  Q'' + Q'/x = 1 + (x - c)/x  (a-independent).
        c, a = 2.3, 0.9

        def Q(x):
            x = jnp.asarray(x)
            return 0.5 * (x - c) ** 2 + a * jnp.log(x)

        def Qp(x):
            return (x - c) + a / x

        def Qpp(x):
            return 1.0 - a / x**2

        return Q, Qp, Qpp

    @pytest.mark.parametrize("x,v", CASES)
    def test_hvp_equals_fd_geodesic_second_derivative(self, x, v):
        r"""``g_x(H[v], v) == d^2/dt^2 Q(R_x(t v))|_0`` (FD geodesic ref)."""
        manifold = Positive()
        Q, Qp, Qpp = self._Q_connection()
        x = jnp.asarray(float(x))
        v = jnp.asarray(float(v))

        # Reference 1: closed-form affine-invariant quadratic form.
        ref_form = (Qpp(x) + Qp(x) / x) * v**2

        # Reference 2 (independent): autodiff 2nd derivative of the geodesic
        # gamma(t) = x e^{t v / x} -- the exact "geodesic 2nd derivative".
        def along(t):
            return Q(manifold.retraction(x, t * v))

        d2_ad = float(jax.grad(jax.grad(along))(0.0))

        # Reference 3 (fully test-side FD, no autodiff): central difference of
        # the geodesic second derivative.
        eps = 1e-4
        fp = float(Q(manifold.retraction(x, eps * v)))
        f0 = float(Q(manifold.retraction(x, 0.0 * v)))
        fm = float(Q(manifold.retraction(x, -eps * v)))
        d2_fd = (fp - 2.0 * f0 + fm) / (eps**2)

        assert float(ref_form) == pytest.approx(d2_ad, rel=1e-9, abs=1e-9)
        assert float(ref_form) == pytest.approx(d2_fd, rel=1e-5, abs=1e-6)

        # The implemented HVP, read as the metric quadratic form.
        Hv = _hvp(Q, manifold, x, v)
        form_impl = float(manifold.inner_product(x, Hv, v))
        assert form_impl == pytest.approx(float(ref_form), rel=1e-7, abs=1e-9)

    @pytest.mark.parametrize("x,v", CASES)
    def test_hvp_distinguishable_from_naive_Qpp(self, x, v):
        r"""The HVP quadratic form must be ``(Q'' + Q'/x) v^2``, DISTINGUISHABLE
        from the naive Euclidean ``Q'' v^2`` (the rejected recipe). The
        connection term ``Q'/x`` must be a meaningful fraction here (fixture
        sanity) -- otherwise the test is vacuous."""
        manifold = Positive()
        Q, Qp, Qpp = self._Q_connection()
        x = jnp.asarray(float(x))
        v = jnp.asarray(float(v))

        wrong = float(Qpp(x) * v**2)  # naive projection(jvp(grad Q)) form
        right = float((Qpp(x) + Qp(x) / x) * v**2)
        gap = abs(right - wrong) / (abs(right) + 1e-12)
        assert gap > 0.05, "fixture: connection term negligible -- R3 vacuous"

        Hv = _hvp(Q, manifold, x, v)
        form_impl = float(manifold.inner_product(x, Hv, v))
        assert form_impl == pytest.approx(right, rel=1e-7, abs=1e-9)
        assert abs(form_impl - wrong) > 0.05 * abs(right), (
            "Positive HVP collapsed to the plain Euclidean Hessian Q'' "
            "(missing the affine connection term Q'/x)"
        )

    def test_negative_curvature_sign_gate(self):
        r"""A case where ``sign(Q'')`` and ``sign(Q'' + Q'/x)`` DISAGREE: the
        connection term flips the curvature sign. A naive ``Q''`` HVP reports
        the WRONG sign here, mis-driving the negative-curvature branch RTR
        exists to detect.

        Construct ``Q(x) = -log(x) + beta x`` -> ``Q' = -1/x + beta``,
        ``Q'' = 1/x^2 > 0`` (always convex in the Euclidean sense). The affine
        form ``Q'' + Q'/x = 1/x^2 + (beta - 1/x)/x = beta/x``. Pick ``beta < 0``
        -> ``Q'' > 0`` but ``Q'' + Q'/x = beta/x < 0`` (NEGATIVE curvature in
        the affine geometry). The two signs disagree.
        """
        manifold = Positive()
        beta = -0.5

        def Q(x):
            x = jnp.asarray(x)
            return -jnp.log(x) + beta * x

        x = jnp.asarray(2.0)
        v = jnp.asarray(1.0)

        qpp = 1.0 / float(x) ** 2  # Euclidean Hessian: strictly POSITIVE
        affine = beta / float(x)  # affine form: strictly NEGATIVE
        assert qpp > 0.0 and affine < 0.0, "fixture: signs do not disagree"

        Hv = _hvp(Q, manifold, x, v)
        form_impl = float(manifold.inner_product(x, Hv, v))
        # The implemented HVP must report NEGATIVE curvature (the affine sign),
        # NOT the positive Euclidean one. This is the curvature-sign blocker.
        assert form_impl < 0.0, (
            "Positive HVP reports the WRONG (Euclidean) curvature sign -- "
            "negative-curvature detection is broken"
        )
        assert form_impl == pytest.approx(affine * float(v) ** 2, rel=1e-7, abs=1e-9)


# ===========================================================================
# Shared Product helpers (PSD <-> Euclidean coupled residual).
# ===========================================================================
def _triu(n):
    return jnp.array(np.triu_indices(n)).T


def _project_tree(manifold, point, vtree):
    return tuple(
        f.projection(p, v)
        for f, p, v in zip(manifold.factors, point, vtree, strict=True)
    )


def _random_horizontal_tree(manifold, point, key):
    keys = jax.random.split(key, len(manifold.factors))
    amb = tuple(
        jax.random.normal(kk, jnp.asarray(p).shape, dtype=jnp.float64)
        for kk, p in zip(keys, point, strict=True)
    )
    return _project_tree(manifold, point, amb)


def _dense_reference_hvp(manifold, Q, point, eta):
    r"""Full-tree projected-HVP oracle: ``jvp(grad Q)`` on the FULL eta (so the
    cross-leaf block survives) then project per leaf -- the metric-exact HVP on
    an additive/Frobenius tree. Built independently of the production code via
    ``jax.jvp`` on the pytree gradient."""
    _, hv = jax.jvp(jax.grad(Q), (point,), (eta,))
    return _project_tree(manifold, point, hv)


# ===========================================================================
# R4 -- manifold_spec leaf-ordering: equal-dim heterogeneous round-trip + HVP.
# ===========================================================================
@jdc.pytree_dataclass
class _EucThenPSD:
    """PyTree-leaf order: ``phi`` (Euclidean) FIRST, then ``Y`` (PSD).

    The user's Product manifold lists factors as ``(PSDFixedRank, Euclidean)``
    -- i.e. the FACTOR order differs from the PyTree-leaf order. Both leaves
    are 2-dimensional (``PSDFixedRank(2, 1)`` = 2x1 = 2 ambient;
    ``Euclidean(2)`` = 2) so a shape-only / size-only check cannot tell them
    apart -- only a VALUE check catches a mis-pairing of the spec to the
    wrong leaf.
    """

    phi: ManifoldLeaf  # Euclidean(2) -- listed first in the PyTree
    Y: ManifoldLeaf  # PSDFixedRank(2, 1) -- listed second


class TestLeafOrderingRoundTrip:
    r"""R4 (blocker) -- a Product whose factor order differs from the
    PyTree-leaf order must round-trip and the HVP must match the dense
    full-tree reference.

    The EQUAL-dim heterogeneous fixture (PSD(2,1) = 2 next to Euclidean(2) = 2)
    is the trap: every block is width 2, so an implementation that pairs the
    spec to leaves by SIZE (or by a fixed factor order) silently swaps the PSD
    and Euclidean manifolds and a shape-only assertion still passes. Only a
    value check -- the HVP under the CORRECT per-leaf manifolds vs a swapped
    one -- catches it.
    """

    NPSD = 2  # PSDFixedRank(2, 1): ambient (2, 1) -> 2 entries
    KPSD = 1
    DEUC = 2  # Euclidean(2): 2 entries -- EQUAL to the PSD block width

    def _make(self, seed=0):
        rng = np.random.default_rng(seed)
        Y = jnp.asarray(rng.normal(size=(self.NPSD, self.KPSD)))
        phi = jnp.asarray(rng.normal(size=(self.DEUC,)))
        params = _EucThenPSD(
            phi=ManifoldLeaf(phi, Euclidean(self.DEUC)),
            Y=ManifoldLeaf(Y, PSDFixedRank(self.NPSD, self.KPSD)),
        )
        return params, Y, phi

    def test_spec_round_trips_in_pytree_leaf_order(self):
        """The spec's leaf_specs follow the PyTree-leaf walk (``phi`` then
        ``Y``), each carrying its OWN manifold -- not the Product factor order
        (``Y`` then ``phi``). The flat buffer tiles ``[phi(2) | Y(2)]``."""
        params, _Y, _phi = self._make(seed=1)
        spec = manifold_spec_from_params(params)
        flat, _treedef, _fspec = flatten_params_with_spec(params)

        # Two leaves, each width 2; offsets tile [0, 2).
        assert len(spec.leaf_specs) == 2
        assert [ls.offset for ls in spec.leaf_specs] == [0, self.DEUC]
        assert int(flat.shape[0]) == self.DEUC + self.NPSD * self.KPSD

        # The CRUCIAL value check (shape is identical for both leaves): the
        # FIRST leaf is the Euclidean one, the SECOND is the PSD one -- i.e.
        # the spec is paired to leaves by PyTree order, NOT by factor order.
        m0 = spec.leaf_specs[0].manifold
        m1 = spec.leaf_specs[1].manifold
        assert isinstance(m0, Euclidean), "first leaf mis-paired (not Euclidean)"
        assert isinstance(m1, PSDFixedRank), "second leaf mis-paired (not PSD)"

    def test_hvp_matches_dense_reference_under_reordered_factors(self):
        r"""Build the Product with factors in PyTree-leaf order (Euclidean, PSD)
        while a SWAPPED order (PSD, Euclidean) would mis-apply the per-leaf
        projections. The implemented HVP must match the dense full-tree
        reference computed with leaves in their TRUE per-leaf manifolds; a
        swapped pairing (PSD horizontal projection on the Euclidean block)
        gives a different, WRONG answer, which we compute test-side to prove
        the equal-dim fixture is non-vacuous."""
        params, Y, phi = self._make(seed=2)
        # Product point in PyTree-leaf order: (phi, Y). The manifold lists the
        # factors in that SAME order so the tuple positions line up.
        manifold = Product(Euclidean(self.DEUC), PSDFixedRank(self.NPSD, self.KPSD))
        point = (phi, Y)

        gtarget = jnp.asarray(np.random.default_rng(3).normal(size=(3,)))  # triu 2x2
        tri = _triu(self.NPSD)

        def Q(pt):
            phil, Yl = pt
            g = (Yl @ Yl.T)[tri[:, 0], tri[:, 1]]
            r_psd = jnp.tanh(g) - jnp.tanh(gtarget)
            r_euc = phil - 0.2
            # A coupling so the cross block is real (also pins R5-style mixing).
            s = jnp.sum(jnp.tanh(g))
            return 0.5 * jnp.sum(r_psd**2) + 0.5 * jnp.sum(r_euc**2) + s * jnp.sum(phil)

        eta = _random_horizontal_tree(manifold, point, jax.random.PRNGKey(11))

        h_impl = _hvp(Q, manifold, point, eta)
        h_ref = _dense_reference_hvp(manifold, Q, point, eta)
        for hi, hr in zip(h_impl, h_ref, strict=True):
            scale = float(jnp.linalg.norm(jnp.asarray(hr))) + 1e-12
            rel = float(jnp.linalg.norm(jnp.asarray(hi) - jnp.asarray(hr))) / scale
            assert rel < 1e-6, f"HVP != dense reference (rel={rel:.2e})"

        # Non-vacuity: a SWAPPED pairing (Product factors in the WRONG order,
        # so the PSD projection hits the Euclidean block and vice versa) gives a
        # materially different HVP -- only a value check separates them.
        swapped = Product(PSDFixedRank(self.NPSD, self.KPSD), Euclidean(self.DEUC))
        _, hv_full = jax.jvp(jax.grad(Q), (point,), (eta,))
        h_swapped = tuple(
            f.projection(p, hv)
            for f, p, hv in zip(swapped.factors, point, hv_full, strict=True)
        )
        diff = max(
            float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
            for a, b in zip(h_ref, h_swapped, strict=True)
        )
        scale = sum(float(jnp.linalg.norm(jnp.asarray(a))) for a in h_ref) + 1e-12
        assert diff / scale > 1e-3, (
            "swapped-factor HVP coincides with the correct one -- the "
            "equal-dim fixture failed to expose mis-pairing (R4 vacuous)"
        )


# ===========================================================================
# R5 -- block-diagonal HVP must NOT drop cross-leaf coupling S.
# ===========================================================================
class TestCrossLeafCoupling:
    r"""R5 (blocker) -- the Product HVP must run ``jvp`` on the FULL eta so the
    cross-leaf Hessian block ``S`` survives; a per-leaf-sliced (block-diagonal)
    HVP drops it and RTR stalls at the saddles it exists to navigate.

    Fixture: ``Product(PSDFixedRank(N, 2), Euclidean(D))`` with an explicit
    ``Gamma``-times-Euclidean cross term ``s(Gamma) * <w, phi>``, so the true
    Hessian has a nonzero PSD<->Euclidean off-diagonal block. Two pins:

      (1) ``eta`` on BOTH leaves: implemented HVP == dense full-tree reference,
          AND != the block-diagonal sliced HVP (computed test-side).
      (2) ``eta`` on the PSD leaf ONLY: the Euclidean-block HVP is NONZERO --
          pure cross coupling that a block-diagonal slice would zero exactly.
    """

    N = 5
    K = 2
    D = 3

    def _setup(self, seed=0):
        manifold = Product(PSDFixedRank(self.N, self.K), Euclidean(self.D))
        rng = np.random.default_rng(seed)
        Y = jnp.asarray(rng.normal(size=(self.N, self.K)))
        phi = jnp.asarray(0.5 + rng.normal(size=(self.D,)))
        w = jnp.asarray(rng.normal(size=(self.D,)))
        tri = _triu(self.N)
        gtarget = jnp.asarray(rng.normal(size=(tri.shape[0],)))

        def Q(pt):
            Yl, phil = pt
            g = (Yl @ Yl.T)[tri[:, 0], tri[:, 1]]
            s = jnp.sum(jnp.tanh(g))  # scalar functional of Gamma
            cross = s * jnp.sum(w * phil)  # explicit Gamma x Euclidean coupling
            r_psd = jnp.tanh(g) - jnp.tanh(gtarget)
            r_euc = phil - 0.3
            return 0.5 * jnp.sum(r_psd**2) + 0.5 * jnp.sum(r_euc**2) + cross

        return manifold, (Y, phi), Q

    def _sliced_reference_hvp(self, manifold, Q, point, eta):
        r"""The WRONG per-leaf-sliced HVP: differentiate each leaf's gradient
        only w.r.t. a perturbation in THAT leaf (zeroing the other leaves'
        eta) -- drops the off-diagonal cross block. Used only to PROVE the
        coupling is real and that a block-diagonal port would be caught."""
        out = []
        for i in range(len(manifold.factors)):
            eta_i = tuple(
                eta[j] if j == i else jnp.zeros_like(jnp.asarray(eta[j]))
                for j in range(len(manifold.factors))
            )
            _, hv = jax.jvp(jax.grad(Q), (point,), (eta_i,))
            out.append(manifold.factors[i].projection(point[i], hv[i]))
        return tuple(out)

    def test_both_leaf_eta_matches_dense_not_sliced(self):
        r"""eta on both leaves: HVP == coupled dense ref, != sliced ref."""
        manifold, point, Q = self._setup(seed=1)
        eta = _random_horizontal_tree(manifold, point, jax.random.PRNGKey(20))

        h_coupled = _dense_reference_hvp(manifold, Q, point, eta)
        h_sliced = self._sliced_reference_hvp(manifold, Q, point, eta)
        scale = sum(float(jnp.linalg.norm(jnp.asarray(a))) for a in h_coupled) + 1e-12

        # The two references genuinely differ (cross block != 0) -> non-vacuous.
        diff_refs = max(
            float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
            for a, b in zip(h_coupled, h_sliced, strict=True)
        )
        assert diff_refs / scale > 1e-3, "fixture: no cross coupling (R5 vacuous)"

        h_impl = _hvp(Q, manifold, point, eta)
        rel_coupled = (
            max(
                float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
                for a, b in zip(h_impl, h_coupled, strict=True)
            )
            / scale
        )
        rel_sliced = (
            max(
                float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
                for a, b in zip(h_impl, h_sliced, strict=True)
            )
            / scale
        )
        assert rel_coupled < 1e-6, "implemented HVP != coupled dense reference"
        assert rel_sliced > 1e-3, (
            "implemented HVP == block-diagonal sliced HVP -- cross-leaf "
            "coupling S was dropped"
        )

    def test_psd_only_eta_gives_nonzero_euclidean_block(self):
        r"""eta on the PSD leaf ONLY (Euclidean eta = 0): the Euclidean-block
        HVP is NONZERO -- pure cross coupling. A block-diagonal slice would
        zero this exactly (it differentiates the Euclidean gradient only w.r.t.
        the Euclidean eta, which is 0), so a nonzero Euclidean block is the
        crisp signature that the full-tree jvp preserved ``S``."""
        manifold, point, Q = self._setup(seed=2)
        # Horizontal PSD eta, ZERO Euclidean eta.
        amb = jax.random.normal(
            jax.random.PRNGKey(21), jnp.asarray(point[0]).shape, dtype=jnp.float64
        )
        eta_psd = manifold.factors[0].projection(point[0], amb)
        eta = (eta_psd, jnp.zeros_like(jnp.asarray(point[1])))

        h_impl = _hvp(Q, manifold, point, eta)
        euc_block = jnp.asarray(h_impl[1])
        # The Euclidean block must be NONZERO (it is exactly the cross term
        # s'(Gamma)[eta_psd] * w). A block-diagonal slice returns 0 here.
        assert float(jnp.linalg.norm(euc_block)) > 1e-6, (
            "Euclidean-block HVP is zero under a PSD-only perturbation -- the "
            "cross-leaf coupling was dropped (block-diagonal HVP)"
        )
        # And it matches the dense reference's Euclidean block.
        h_ref = _dense_reference_hvp(manifold, Q, point, eta)
        rel = float(jnp.linalg.norm(euc_block - jnp.asarray(h_ref[1]))) / (
            float(jnp.linalg.norm(jnp.asarray(h_ref[1]))) + 1e-12
        )
        assert rel < 1e-6, f"Euclidean cross-block HVP != reference (rel={rel:.2e})"


# ===========================================================================
# R6 -- tCG metric is per-leaf non-identity (Positive 1/x^2 at x = 5).
# ===========================================================================
@jdc.pytree_dataclass
class _EucPosParams:
    """``Product(Euclidean(d), Positive())`` -- a mixed-metric tree.

    The Euclidean leaf carries the identity metric; the Positive leaf carries
    the affine ``1/x^2`` metric. With the Positive coordinate at ``x = 5``,
    ``1/x^2 = 0.04`` -- a metric weight FAR from 1, so a flat-Frobenius tCG
    mis-scales the trust radius and curvature on the Positive block.
    """

    v: ManifoldLeaf  # Euclidean(d)
    s: ManifoldLeaf  # Positive() scalar


class TestTcgPerLeafMetric:
    r"""R6 (blocker) -- ``_truncated_cg`` must use ``Product.inner_product``
    (per-leaf metric), NOT flat Frobenius, for the trust-radius boundary test
    and the curvature ``<d, H d>``.

    A pure-PSD/Euclidean leaf has the identity metric, so a naive ``jnp.sum``
    happens to be correct -- the trap a same-metric fixture cannot catch. We
    put the Positive coordinate at ``x = 5`` (``1/x^2 = 0.04``): the
    boundary-hit ``eta`` must satisfy ``norm_manifold(eta) == Delta``, NOT the
    Frobenius norm, and the step differs both from the flat-metric step and
    from the ``x = 1`` step (where ``1/x^2 = 1`` and the two metrics coincide).
    """

    DEUC = 3

    def _spec(self, x_val):
        params = _EucPosParams(
            v=ManifoldLeaf(jnp.zeros((self.DEUC,)), Euclidean(self.DEUC)),
            s=ManifoldLeaf(jnp.asarray(jnp.float64(x_val)), Positive()),
        )
        spec = manifold_spec_from_params(params)
        point_flat, _treedef, _fspec = flatten_params_with_spec(params)
        return spec, point_flat

    def test_boundary_eta_has_manifold_norm_at_x5(self):
        r"""With ``x = 5`` and a negative-curvature HVP isolated to the Positive
        leaf, tCG steps to the boundary in the MANIFOLD metric: the affine norm
        of ``eta`` equals ``Delta``, while its Frobenius norm does NOT. A
        flat-metric port reports the Frobenius norm at the boundary."""
        x_val = 5.0
        spec, point_flat = self._spec(x_val)
        d = int(point_flat.shape[0])  # DEUC + 1
        s_idx = self.DEUC  # the Positive scalar sits last (after the Euclidean)

        # Negative-curvature HVP isolated to the Positive leaf -> immediate
        # step-to-boundary, and the tau solve runs in the manifold metric.
        Hmat = jnp.zeros((d, d)).at[s_idx, s_idx].set(-1.0)

        def hvp(eta_flat):
            return Hmat @ eta_flat

        grad_flat = jnp.zeros((d,)).at[s_idx].set(1.0)
        Delta = 0.3
        eta, _Heta, info = _tcg(
            hvp,
            grad_flat,
            spec,
            point_flat,
            Delta,
            theta=1.0,
            kappa=0.1,
            min_inner=1,
            max_tcg_steps=d,
        )
        assert str(info.stop_reason) in ("negative_curvature", "exceeded_tr")

        eta_s = float(eta[s_idx])
        eta_v = np.asarray(eta[:s_idx])
        # Affine manifold norm: Frobenius on the Euclidean block + (1/x^2) s^2.
        man_sq = float(np.sum(eta_v * eta_v)) + (eta_s**2) / (x_val**2)
        man_norm = float(np.sqrt(man_sq))
        assert man_norm == pytest.approx(Delta, rel=1e-8, abs=1e-10)

        # And it is NOT the Frobenius norm: with the step on the s leaf and
        # 1/x^2 = 0.04, the manifold and Frobenius norms differ a lot.
        frob = float(np.sqrt(np.sum(np.asarray(eta) ** 2)))
        assert abs(frob - Delta) > 1e-2, (
            "boundary eta's Frobenius norm equals Delta -> tCG used the flat "
            "metric instead of Product.inner_product (R6 blocker)"
        )

    def test_step_differs_from_flat_metric_and_from_x1(self):
        r"""The Positive-leaf step magnitude at ``x = 5`` differs BOTH from the
        flat-Frobenius step (same Delta, identity metric) AND from the ``x = 1``
        step (where ``1/x^2 = 1`` so the affine and flat metrics coincide).

        At ``x = 5`` the affine metric weight on the s leaf is ``0.04``, so to
        reach the SAME trust radius ``Delta`` the affine-metric step takes a
        LARGER ambient ``s`` displacement (``|s| = Delta * x``) than the flat
        step (``|s| = Delta``). The ``x = 1`` step coincides with the flat step.
        A flat-metric tCG would make all three equal -- a crisp discriminator."""
        Delta = 0.3
        results = {}
        for x_val in (5.0, 1.0):
            spec, point_flat = self._spec(x_val)
            d = int(point_flat.shape[0])
            s_idx = self.DEUC
            Hmat = jnp.zeros((d, d)).at[s_idx, s_idx].set(-1.0)

            def hvp(eta_flat, _H=Hmat):
                return _H @ eta_flat

            grad_flat = jnp.zeros((d,)).at[s_idx].set(1.0)
            eta, _Heta, _info = _tcg(
                hvp,
                grad_flat,
                spec,
                point_flat,
                Delta,
                theta=1.0,
                kappa=0.1,
                min_inner=1,
                max_tcg_steps=d,
            )
            results[x_val] = abs(float(eta[s_idx]))

        # Reference (affine geometry): |s| = Delta * x to hit manifold-norm Delta.
        assert results[1.0] == pytest.approx(Delta * 1.0, rel=1e-6, abs=1e-9)
        assert results[5.0] == pytest.approx(Delta * 5.0, rel=1e-6, abs=1e-9)

        # The flat-metric step (the bug) would give |s| = Delta for BOTH x.
        flat_step = Delta
        assert (
            abs(results[5.0] - flat_step) > 1e-2
        ), "x=5 step matches the flat-metric step -> Product metric not used"
        # x=5 and x=1 steps must differ (the metric weight is x-dependent).
        assert (
            abs(results[5.0] - results[1.0]) > 1e-2
        ), "x=5 and x=1 steps coincide -> tCG ignored the 1/x^2 weighting"
