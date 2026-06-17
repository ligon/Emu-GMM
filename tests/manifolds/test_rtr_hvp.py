r"""TEST-FIRST (Phase 1, RED until Phase 2) per-leaf HVP correctness for the
JAX-native Riemannian Trust Region optimiser (#152).

This module pins the **Riemannian Hessian-vector product** that tCG drives,
*before* any implementation exists. Every test here imports

    from emu_gmm.manifolds.riemannian_tr import _riemannian_hvp

which does NOT exist yet -- so this whole file is RED (``ImportError`` at
collection / a module-level ``importorskip`` failure) until Phase 2 ships the
implementation. That is the intended state: these are the executable spec.

HVP semantics under test (revised design, ``gh issue view 152``):

    H[eta] = Euclidean Hessian of the retraction-pullback  Q_hat(eta) = Q(R_Y(eta))
             at eta = 0,  projected to horizontal for the PSD leaf.

i.e. ``H[eta] = jvp(eta' |-> grad(Q_hat)(eta'), eta)|_0``, then for a quotient
leaf (PSDFixedRank) projected onto the horizontal space. Key consequences this
file asserts:

* For PSDFixedRank (additive retraction ``Y + V``, embedded Frobenius metric)
  the pullback Hessian reduces to the projected ambient Hessian; it must match
  a metric-exact reference (finite-difference of the *Riemannian* gradient,
  transported back).
* For Positive (exponential retraction ``x e^{v/x}``, affine metric ``uv/x^2``)
  the pullback automatically carries the affine connection term, so the second
  derivative of the geodesic is ``Q'' + Q'/x`` -- NOT ``Q''``. A plain
  ``projection(jvp(grad Q))`` recipe (projection is identity for Positive) would
  return ``Q''`` and is therefore WRONG; the test asserts the connection term is
  present.
* The operator is self-adjoint on the horizontal space (tCG requires it).
* Gauge-conditional exactness: gauge-invariant ``Q`` passes the additive recipe;
  a deliberately gauge-VARIANT ``psi`` that reads ``A_{ij}`` directly must FAIL
  the additive recipe (proving the recipe is conditional and guarding the
  projected-gradient fix).
* Product full-tree ``jvp`` preserves cross-leaf coupling ``S``: a per-leaf
  sliced HVP must FAIL the off-diagonal block.

Risks pinned (lenses hvp-psd / hvp-product-positive / tcg HVP-symmetry):

* hvp-psd: "Missing vertical/connection term ... HVP exactness assumes a
  gauge-invariant Q" -> ``test_psd_additive_recipe_is_gauge_conditional``.
* hvp-psd: "HVP matches the metric-exact Riemannian Hessian (FD of rgrad)" ->
  ``test_psd_hvp_matches_finite_difference_rgrad``.
* tcg / hvp-psd: "Symmetry of H not guaranteed -> tCG conjugacy void" ->
  ``test_psd_hvp_self_adjoint_on_horizontal``.
* hvp-product-positive (BLOCKER): "Positive-leaf Hessian is wrong under
  projection(jvp(grad Q)); must equal Q''+Q'/x via the geodesic 2nd
  derivative" -> ``test_positive_hvp_is_geodesic_second_derivative`` /
  ``test_positive_hvp_is_not_plain_euclidean_hessian``.
* hvp-product-positive: "Product HVP must run jvp on the FULL eta to preserve
  cross-leaf coupling; per-leaf slicing degenerates to block-diagonal" ->
  ``test_product_hvp_preserves_cross_leaf_coupling`` /
  ``test_product_sliced_hvp_would_drop_offdiagonal``.

Phase 2 must expose ``_riemannian_hvp`` with the signature assumed below:

    _riemannian_hvp(Q, manifold, point, eta) -> tangent_like(eta)

where ``Q`` maps a manifold point (an array for a single leaf, a tuple/pytree
for a Product) to a real scalar, ``manifold`` is the ``ManifoldParam`` for that
point, and ``point`` / ``eta`` share the manifold's ambient storage. If Phase 2
chooses a different unit-helper signature, update the thin ``_hvp`` shim here in
ONE place -- the reference computations and assertions stay.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.manifolds import Euclidean, Positive, Product, PSDFixedRank

jax.config.update("jax_enable_x64", True)

# RED-until-Phase-2: the implementation module does not exist yet. The
# importorskip below makes the whole file SKIP (not error) at collection with
# the module name in the reason -- the intended Phase-1 signal. Once Phase 2
# lands ``emu_gmm.manifolds.riemannian_tr`` with ``_riemannian_hvp``, these
# tests go live and are expected to be RED until the HVP is correct.
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
_riemannian_hvp = riemannian_tr_mod._riemannian_hvp

N = 5  # ambient PSD side (matches the Phase-6/7 acceptance fixtures)


# ---------------------------------------------------------------------------
# Thin shim around the Phase-2 unit helper. If Phase 2 picks a different
# signature, edit ONLY this function. Everything below builds its reference
# from first principles and compares against this.
# ---------------------------------------------------------------------------
def _hvp(Q, manifold, point, eta):
    """``H[eta]`` = Riemannian HVP of ``Q`` at ``point`` along ``eta``."""
    return _riemannian_hvp(Q, manifold, point, eta)


# ---------------------------------------------------------------------------
# Reference computations (metric-exact, implemented test-side).
# ---------------------------------------------------------------------------
def _project_psd(manifold, Y, V):
    """Horizontal projection at ``Y`` (the manifold's own operator)."""
    return manifold.projection(Y, V)


def _random_horizontal(manifold, Y, key):
    """A random tangent vector projected onto the horizontal space at ``Y``."""
    amb = jax.random.normal(key, Y.shape, dtype=jnp.float64)
    return _project_psd(manifold, Y, amb)


def _fd_riemannian_hessian_psd(Q, manifold, Y, eta, *, eps=1e-6):
    r"""Metric-exact reference Riemannian HVP for an EMBEDDED (PSD) leaf via a
    central finite difference of the *Riemannian gradient* along the geodesic.

    The Riemannian gradient of ``Q`` at a point ``Z`` is
    ``rgrad(Z) = projection(Z, grad Q(Z))`` (embedded Frobenius metric, so
    egrad2rgrad is the projection). The Riemannian Hessian-vector product is the
    covariant derivative of that field along ``eta``; for the additive
    retraction ``R_Y(t eta) = Y + t eta`` it is, to leading order,

        Hess Q[eta] = projection(Y, d/dt rgrad(Y + t eta)|_0).

    We evaluate ``d/dt rgrad`` by a central difference and then re-project to
    horizontal at ``Y`` (the trailing projection that the embedded-quotient
    connection prescribes). This is INDEPENDENT of the implementation's
    ``jvp(grad(pullback))`` route, so agreement is a genuine cross-check.
    """

    def rgrad(Z):
        return manifold.projection(Z, jax.grad(Q)(Z))

    plus = rgrad(Y + eps * eta)
    minus = rgrad(Y - eps * eta)
    dfield = (plus - minus) / (2.0 * eps)
    return manifold.projection(Y, dfield)


# ---------------------------------------------------------------------------
# Shared PSD fixtures: a gauge-invariant residual and a gauge-VARIANT one.
# ---------------------------------------------------------------------------
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2): unique entries of a 5x5 sym


def _make_psd_target(k, seed):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    return A_true, g_true


def _make_gauge_invariant_Q(k, seed, *, nonlinear=True):
    r"""``Q(Y) = 1/2 || w * (f(triu(YY^T)) - target) ||^2``.

    Depends on ``Y`` ONLY through ``Gamma = Y Y^T`` (gauge-invariant). When
    ``nonlinear=True`` the moment map ``f`` is a genuine nonlinearity in
    ``Gamma`` (so the residual-curvature term ``S = sum_i r_i grad^2 r_i`` is
    nonzero and the HVP is NOT a pure Gauss-Newton ``J'J`` -- this is what makes
    the FD cross-check non-vacuous and what RTR exists to navigate).
    """
    _A_true, g_true = _make_psd_target(k, seed)
    rng = np.random.default_rng(seed + 777)
    w = jnp.asarray(0.5 + rng.random(g_true.shape[0]))  # positive weights
    target = g_true + 0.3 * jnp.asarray(rng.normal(size=g_true.shape))

    def Q(Y):
        g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
        m = jnp.tanh(g) if nonlinear else g
        tt = jnp.tanh(target) if nonlinear else target
        r = w * (m - tt)
        return 0.5 * jnp.sum(r * r)

    return Q


def _make_gauge_variant_Q(k, seed):
    r"""A deliberately gauge-VARIANT ``Q`` that reads raw ``A_{ij}`` entries.

    ``Q`` depends on ``Y`` NOT only through ``Y Y^T``: it adds a term linear in
    a raw off-diagonal entry ``Y[0, 1]`` (requires ``k >= 2``). Then ``grad Q``
    has a vertical (skew) component at ``Y``, the egrad is no longer horizontal,
    and the additive recipe ``projection(jvp(grad Q))`` omits the connection
    term -- it must DISAGREE with the metric-exact FD reference.
    """
    base = _make_gauge_invariant_Q(k, seed, nonlinear=True)
    # Gauge-variant perturbation along a GENERIC fixed direction R: a linear
    # term (vertical egrad component) PLUS a quadratic term (a genuine
    # second-order / connection term the naive additive recipe drops). (#152)
    # A single raw entry c*Y[0,1] was tried first, but its gradient e_{0,1} is
    # nearly HORIZONTAL at the test Y, so the vertical egrad fraction SATURATED
    # at ~5e-4 for any coefficient (||g_bad|| grows with the coefficient too).
    # A generic R has vertical fraction ~ sqrt(gauge_dim/dim) = O(0.3) at any
    # full-rank Y, clearing the vert_frac > 1e-3 precondition for both k=2 and
    # k=3; the quadratic <Y,R>**2 keeps the recipe-disagreement assertion
    # (the naive recipe drops its connection term) non-vacuous.
    R = jnp.asarray(np.random.default_rng(seed + 99).normal(size=(N, k)))

    def Q(Y):
        s = jnp.vdot(Y, R)
        return base(Y) + 2.0 * s + 1.5 * s**2

    return Q


# ===========================================================================
# PSDFixedRank: metric-exact reference + self-adjointness + gauge-conditional.
# ===========================================================================
@pytest.mark.parametrize("k", [2, 3])
class TestPSDHvpCorrectness:
    r"""Per-leaf HVP correctness for the additive PSD leaf (lens hvp-psd)."""

    def _Y0(self, k, seed):
        A_true, _ = _make_psd_target(k, seed)
        rng = np.random.default_rng(seed + 11)
        # Off the truth so S != 0 and the iterate is NON-critical (the regime
        # RTR targets); a critical point would make the HVP vacuously GN.
        return A_true + 0.25 * jnp.asarray(rng.normal(size=(N, k)))

    def test_psd_hvp_matches_finite_difference_rgrad(self, k):
        r"""hvp-psd: 'HVP matches the metric-exact Riemannian Hessian'.

        At a NON-critical Y of a nonlinear (S != 0) gauge-invariant problem,
        the implemented HVP must equal the finite-difference of the Riemannian
        gradient (transported, re-projected) to ~1e-5 on random horizontal
        directions. A pure-Gauss-Newton surrogate (dropping S) or a misplaced
        projection fails this.
        """
        manifold = PSDFixedRank(N, k)
        Q = _make_gauge_invariant_Q(k, seed=k)
        Y = self._Y0(k, seed=k)
        keys = jax.random.split(jax.random.PRNGKey(100 + k), 4)
        for key in keys:
            eta = _random_horizontal(manifold, Y, key)
            h_impl = _hvp(Q, manifold, Y, eta)
            h_ref = _fd_riemannian_hessian_psd(Q, manifold, Y, eta)
            # Output must be horizontal (re-projection is a no-op).
            assert bool(
                jnp.allclose(
                    h_impl, manifold.projection(Y, h_impl), atol=1e-8, rtol=0.0
                )
            ), "implemented HVP output is not horizontal"
            scale = float(jnp.linalg.norm(h_ref)) + 1e-12
            rel = float(jnp.linalg.norm(h_impl - h_ref)) / scale
            assert rel < 1e-4, f"HVP != FD Riemannian Hessian (rel={rel:.2e})"

    def test_psd_hvp_self_adjoint_on_horizontal(self, k):
        r"""tcg / hvp-psd: 'Symmetry of H not guaranteed -> tCG conjugacy void'.

        Steihaug-Toint CG requires a self-adjoint operator under the manifold
        inner product. Assert |<u, H v> - <v, H u>| is at the rounding floor for
        random horizontal u, v -- an output-only (single-sided) projection of a
        Hessian that does not preserve the horizontal subspace would break this.
        """
        manifold = PSDFixedRank(N, k)
        Q = _make_gauge_invariant_Q(k, seed=k + 1)
        Y = self._Y0(k, seed=k + 1)
        ku, kv = jax.random.split(jax.random.PRNGKey(200 + k))
        u = _random_horizontal(manifold, Y, ku)
        v = _random_horizontal(manifold, Y, kv)
        Hu = _hvp(Q, manifold, Y, u)
        Hv = _hvp(Q, manifold, Y, v)
        uHv = float(manifold.inner_product(Y, u, Hv))
        vHu = float(manifold.inner_product(Y, v, Hu))
        denom = abs(uHv) + abs(vHu) + 1e-12
        asym = abs(uHv - vHu) / denom
        assert asym < 1e-8, f"HVP not self-adjoint on horizontal (asym={asym:.2e})"

    def test_psd_hvp_is_linear(self, k):
        r"""hvp-psd: ``H`` is a LINEAR operator (an HVP, not a finite step).

        ``H[a u + b v] == a H[u] + b H[v]`` to rounding. Guards against a port
        that secretly uses a finite-difference / retracted evaluation in place
        of a true jvp (which would be only first-order accurate and nonlinear).
        """
        manifold = PSDFixedRank(N, k)
        Q = _make_gauge_invariant_Q(k, seed=k + 2)
        Y = self._Y0(k, seed=k + 2)
        ku, kv = jax.random.split(jax.random.PRNGKey(300 + k))
        u = _random_horizontal(manifold, Y, ku)
        v = _random_horizontal(manifold, Y, kv)
        a, b = 1.7, -0.6
        lhs = _hvp(Q, manifold, Y, a * u + b * v)
        rhs = a * _hvp(Q, manifold, Y, u) + b * _hvp(Q, manifold, Y, v)
        scale = float(jnp.linalg.norm(rhs)) + 1e-12
        rel = float(jnp.linalg.norm(lhs - rhs)) / scale
        assert rel < 1e-9, f"HVP not linear (rel={rel:.2e})"


@pytest.mark.parametrize("k", [2, 3])
def test_psd_additive_recipe_is_gauge_conditional(k):
    r"""hvp-psd (BLOCKER): 'HVP exactness silently assumes gauge-invariant Q'.

    Two residuals at the SAME non-critical Y:
      (a) gauge-invariant Q (depends on Y only through Y Y^T) -- the implemented
          HVP must MATCH the metric-exact FD Riemannian Hessian;
      (b) a gauge-VARIANT Q reading raw A_{0,1} -- the *additive recipe*
          ``projection(jvp(grad Q))`` (which the implementation must NOT blindly
          use) must DISAGREE with the FD reference, proving the recipe is
          conditional on gauge-invariance.

    This pins that the implementation does the projected-gradient fix (or
    otherwise handles the vertical egrad component) rather than the naive
    output-only-projection recipe. We assert (a) passes and that the naive
    recipe (computed test-side) genuinely fails on (b) -- so the fixture is not
    vacuous.
    """
    manifold = PSDFixedRank(N, k)
    A_true, _ = _make_psd_target(k, seed=k + 5)
    rng = np.random.default_rng(k + 5)
    Y = A_true + 0.25 * jnp.asarray(rng.normal(size=(N, k)))

    Q_good = _make_gauge_invariant_Q(k, seed=k + 5)
    Q_bad = _make_gauge_variant_Q(k, seed=k + 5)

    # The naive "additive recipe" the design rejects: project ONLY the output of
    # jvp(grad Q). For a gauge-invariant Q this coincides with the true HVP; for
    # a gauge-variant Q it omits the connection term.
    def naive_recipe(Q, Y, eta):
        _, hv = jax.jvp(jax.grad(Q), (Y,), (eta,))
        return manifold.projection(Y, hv)

    key = jax.random.PRNGKey(400 + k)
    eta = _random_horizontal(manifold, Y, key)

    # (a) gauge-invariant: implemented HVP matches the metric-exact reference.
    h_impl_good = _hvp(Q_good, manifold, Y, eta)
    h_ref_good = _fd_riemannian_hessian_psd(Q_good, manifold, Y, eta)
    rel_good = float(jnp.linalg.norm(h_impl_good - h_ref_good)) / (
        float(jnp.linalg.norm(h_ref_good)) + 1e-12
    )
    assert rel_good < 1e-4, f"gauge-invariant HVP wrong (rel={rel_good:.2e})"

    # Sanity: the gauge-variant gradient genuinely has a vertical component,
    # i.e. egrad is NOT horizontal (otherwise the fixture proves nothing).
    g_bad = jax.grad(Q_bad)(Y)
    vert = g_bad - manifold.projection(Y, g_bad)
    vert_frac = float(jnp.linalg.norm(vert)) / (float(jnp.linalg.norm(g_bad)) + 1e-12)
    assert vert_frac > 1e-3, "gauge-variant fixture has no vertical egrad"

    # (b) The NAIVE additive recipe must DISAGREE with the metric-exact FD on
    # the gauge-variant Q -- proving the recipe is conditional. (We do not
    # require the *implemented* HVP to match the FD here -- the spec only
    # certifies the additive recipe for gauge-invariant Q; this branch's job is
    # to demonstrate the additive recipe is unsafe, motivating the fix.)
    h_naive_bad = naive_recipe(Q_bad, Y, eta)
    h_ref_bad = _fd_riemannian_hessian_psd(Q_bad, manifold, Y, eta)
    rel_bad = float(jnp.linalg.norm(h_naive_bad - h_ref_bad)) / (
        float(jnp.linalg.norm(h_ref_bad)) + 1e-12
    )
    assert rel_bad > 1e-3, (
        "naive additive recipe unexpectedly AGREED on a gauge-variant Q -- "
        f"fixture is vacuous (rel_bad={rel_bad:.2e})"
    )


# ===========================================================================
# Positive: geodesic second-derivative reference (the connection term).
# ===========================================================================
class TestPositiveHvpConnectionTerm:
    r"""hvp-product-positive (BLOCKER): the Positive leaf HVP must carry the
    affine connection term -- it must equal ``Q'' + Q'/x``, NOT ``Q''``.

    The metric-exact reference is the SECOND derivative of ``Q`` along the
    exponential geodesic ``gamma(t) = x e^{t v / x}``:

        d^2/dt^2 Q(gamma(t))|_0 = g_x(Hess Q[v], v) = (Q''(x) + Q'(x)/x) v^2,

    where ``g_x(u, w) = u w / x^2`` is the affine metric and ``Hess Q[v]`` is the
    Riemannian HVP. We pick a NON-quadratic ``Q`` so ``Q''`` and ``Q''+Q'/x``
    are clearly distinct, and recover the operator value via the metric.
    """

    # Several (x, v) pairs spanning x below/at/above 1 so 1/x weighting bites.
    CASES = [(1.7, 0.9), (0.4, -0.6), (3.5, 1.3), (0.8, 0.2)]

    @staticmethod
    def _Qpos():
        # A non-quadratic scalar criterion with closed-form derivatives.
        # Q(x) = 0.5 (x - c)^2 + a log(x): Q'(x) = (x - c) + a/x,
        # Q''(x) = 1 - a/x^2.  Q''+Q'/x = 1 - a/x^2 + (x-c)/x + a/x^2
        #                                = 1 + (x - c)/x   (a-independent!).
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
    def test_positive_hvp_is_geodesic_second_derivative(self, x, v):
        r"""``g_x(H[v], v) == Q''(x) + Q'(x)/x`` * v^2 (metric-exact)."""
        manifold = Positive()
        Q, Qp, Qpp = self._Qpos()
        x = jnp.asarray(float(x))
        v = jnp.asarray(float(v))

        # Reference 1: closed-form affine-invariant quadratic form.
        ref_form = (Qpp(x) + Qp(x) / x) * v**2

        # Reference 2 (independent): autodiff 2nd derivative of the geodesic.
        def along(t):
            return Q(manifold.retraction(x, t * v))

        d2 = jax.grad(jax.grad(along))(0.0)
        assert float(ref_form) == pytest.approx(float(d2), rel=1e-9, abs=1e-9)

        # Implemented HVP, evaluated as the metric quadratic form g_x(H[v], v).
        Hv = _hvp(Q, manifold, x, v)
        form_impl = float(manifold.inner_product(x, Hv, v))
        assert form_impl == pytest.approx(float(ref_form), rel=1e-7, abs=1e-9)

    @pytest.mark.parametrize("x,v", CASES)
    def test_positive_hvp_is_not_plain_euclidean_hessian(self, x, v):
        r"""The Positive HVP must DIFFER from the plain Euclidean Hessian.

        For Positive, ``projection`` is the identity, so the rejected recipe
        ``projection(jvp(grad Q))`` returns ``Q''(x) v`` and its quadratic form
        is ``Q'' v^2`` -- which OMITS the ``Q'/x`` connection term. Assert the
        implemented HVP's quadratic form is NOT ``Q'' v^2`` whenever ``Q'/x`` is
        non-negligible (it is, for these cases). This is the 25%-off blocker.
        """
        manifold = Positive()
        Q, Qp, Qpp = self._Qpos()
        x = jnp.asarray(float(x))
        v = jnp.asarray(float(v))

        wrong_form = float(Qpp(x) * v**2)  # plain Euclidean Hessian quad form
        right_form = float((Qpp(x) + Qp(x) / x) * v**2)
        # The connection term must be a meaningful fraction (fixture sanity).
        gap = abs(right_form - wrong_form) / (abs(right_form) + 1e-12)
        assert gap > 0.05, "fixture: connection term is negligible here"

        Hv = _hvp(Q, manifold, x, v)
        form_impl = float(manifold.inner_product(x, Hv, v))
        # Must match the connection-corrected form, NOT the Euclidean one.
        assert form_impl == pytest.approx(right_form, rel=1e-7, abs=1e-9)
        assert abs(form_impl - wrong_form) > 0.05 * abs(right_form), (
            "Positive HVP collapsed to the plain Euclidean Hessian Q'' "
            "(missing the affine connection term Q'/x)"
        )

    def test_positive_hvp_self_adjoint(self):
        r"""tcg: self-adjointness of the scalar Positive HVP under ``g_x``.

        With distinct tangent magnitudes ``u, v`` at the same ``x``,
        ``g_x(u, H[v]) == g_x(v, H[u])`` (trivially true for a 1-D operator, but
        a port that mismatches the metric on one side would break it).
        """
        manifold = Positive()
        Q, _, _ = self._Qpos()
        x = jnp.asarray(1.3)
        u = jnp.asarray(0.7)
        v = jnp.asarray(-0.4)
        Hu = _hvp(Q, manifold, x, u)
        Hv = _hvp(Q, manifold, x, v)
        uHv = float(manifold.inner_product(x, u, Hv))
        vHu = float(manifold.inner_product(x, v, Hu))
        assert uHv == pytest.approx(vHu, rel=1e-9, abs=1e-12)


# ===========================================================================
# Product: full-tree jvp preserves cross-leaf coupling S.
# ===========================================================================
class TestProductHvpCoupling:
    r"""hvp-product-positive: 'Product HVP must run jvp on the FULL eta to
    preserve cross-leaf coupling S; per-leaf slicing degenerates to
    block-diagonal and stalls at the saddles RTR targets'.

    Fixture: ``Product(PSDFixedRank(N, 2), Euclidean(2))`` with a residual that
    MIXES the two leaves multiplicatively, so the true Hessian has a nonzero
    off-diagonal PSD<->Euclidean block. We compare the implemented HVP against a
    dense full-tree reference (``jvp(grad Q)`` on the FULL eta, projected per
    leaf), and separately assert that a per-leaf-SLICED HVP (computed test-side)
    would DROP the off-diagonal -- so the test is non-vacuous.
    """

    K = 2
    D = 2  # Euclidean leaf width

    def _setup(self, seed=0):
        manifold = Product(PSDFixedRank(N, self.K), Euclidean(self.D))
        A_true, _ = _make_psd_target(self.K, seed)
        rng = np.random.default_rng(seed + 3)
        Y = A_true + 0.2 * jnp.asarray(rng.normal(size=(N, self.K)))
        # Coupled residual: a scalar from the PSD block TIMES the Euclidean
        # block, plus per-leaf terms. Mixing guarantees a nonzero cross block.
        w = jnp.asarray(rng.normal(size=(self.D,)))
        gtarget = jnp.asarray(rng.normal(size=(_TRIU.shape[0],)))

        def Q(point):
            Yl, phil = point
            g = (Yl @ Yl.T)[_TRIU[:, 0], _TRIU[:, 1]]
            s = jnp.sum(jnp.tanh(g))  # scalar functional of the PSD block
            coupling = s * jnp.sum(w * phil)  # PSD <-> Euclidean coupling
            r_psd = jnp.tanh(g) - jnp.tanh(gtarget)
            r_euc = phil - 0.3
            return (
                0.5 * jnp.sum(r_psd * r_psd) + 0.5 * jnp.sum(r_euc * r_euc) + coupling
            )

        phi0 = jnp.asarray(0.5 + rng.normal(size=(self.D,)))
        return manifold, (Y, phi0), Q

    def _project_tree(self, manifold, point, vtree):
        return tuple(
            f.projection(p, v)
            for f, p, v in zip(manifold.factors, point, vtree, strict=True)
        )

    def _random_horizontal_tree(self, manifold, point, key):
        keys = jax.random.split(key, len(manifold.factors))
        amb = tuple(
            jax.random.normal(kk, jnp.asarray(p).shape, dtype=jnp.float64)
            for kk, p in zip(keys, point, strict=True)
        )
        return self._project_tree(manifold, point, amb)

    def _dense_reference_hvp(self, manifold, Q, point, eta):
        r"""Full-tree reference: ``jvp(grad Q)`` on the FULL eta (preserving the
        cross block) then project per leaf -- the metric-exact HVP on this
        additive/Frobenius tree. Implemented independently of the production
        code via ``jax.jvp`` on the pytree gradient."""
        _, hv = jax.jvp(jax.grad(Q), (point,), (eta,))
        return self._project_tree(manifold, point, hv)

    def _sliced_reference_hvp(self, manifold, Q, point, eta):
        r"""The WRONG per-leaf-sliced HVP: differentiate each leaf's gradient
        only w.r.t. a perturbation in THAT leaf (zeroing the other leaves'
        eta), dropping the off-diagonal cross block. Used only to PROVE the
        cross-coupling is real (the implemented HVP must NOT equal this)."""
        out = []
        for i in range(len(manifold.factors)):
            eta_i = tuple(
                eta[j] if j == i else jnp.zeros_like(jnp.asarray(eta[j]))
                for j in range(len(manifold.factors))
            )
            _, hv = jax.jvp(jax.grad(Q), (point,), (eta_i,))
            out.append(manifold.factors[i].projection(point[i], hv[i]))
        return tuple(out)

    def test_product_hvp_preserves_cross_leaf_coupling(self):
        r"""Implemented Product HVP == full-tree (coupled) reference."""
        manifold, point, Q = self._setup(seed=1)
        key = jax.random.PRNGKey(500)
        eta = self._random_horizontal_tree(manifold, point, key)

        h_impl = _hvp(Q, manifold, point, eta)
        h_ref = self._dense_reference_hvp(manifold, Q, point, eta)

        for hi, hr in zip(h_impl, h_ref, strict=True):
            scale = float(jnp.linalg.norm(hr)) + 1e-12
            rel = float(jnp.linalg.norm(jnp.asarray(hi) - jnp.asarray(hr))) / scale
            assert rel < 1e-6, f"Product HVP != full-tree reference (rel={rel:.2e})"

    def test_product_sliced_hvp_would_drop_offdiagonal(self):
        r"""Meta-check: the coupled reference DIFFERS from a per-leaf-sliced HVP.

        Proves the cross block is genuinely nonzero, so
        ``test_product_hvp_preserves_cross_leaf_coupling`` is non-vacuous AND a
        sliced implementation would be caught. Also asserts the IMPLEMENTED HVP
        does not coincide with the sliced (block-diagonal) one.
        """
        manifold, point, Q = self._setup(seed=2)
        key = jax.random.PRNGKey(600)
        eta = self._random_horizontal_tree(manifold, point, key)

        h_coupled = self._dense_reference_hvp(manifold, Q, point, eta)
        h_sliced = self._sliced_reference_hvp(manifold, Q, point, eta)

        # The two references must differ on at least one leaf (cross block != 0).
        diffs = [
            float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
            for a, b in zip(h_coupled, h_sliced, strict=True)
        ]
        scale = sum(float(jnp.linalg.norm(jnp.asarray(a))) for a in h_coupled) + 1e-12
        assert max(diffs) / scale > 1e-3, (
            "fixture: coupled and sliced HVP coincide -> no cross coupling, "
            "test is vacuous"
        )

        # The IMPLEMENTED HVP must track the coupled reference, NOT the sliced.
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
        assert rel_coupled < 1e-6, "implemented HVP does not match coupled reference"
        assert rel_sliced > 1e-3, (
            "implemented HVP matches the block-diagonal SLICED HVP -- "
            "cross-leaf coupling was dropped"
        )

    def test_product_hvp_self_adjoint(self):
        r"""tcg: self-adjointness of the Product HVP under Product.inner_product.

        Cross-leaf coupling makes self-adjointness a genuine constraint here:
        the off-diagonal blocks must be transposes of each other under the
        per-leaf metrics summed by ``Product.inner_product``.
        """
        manifold, point, Q = self._setup(seed=3)
        ku, kv = jax.random.split(jax.random.PRNGKey(700))
        u = self._random_horizontal_tree(manifold, point, ku)
        v = self._random_horizontal_tree(manifold, point, kv)
        Hu = _hvp(Q, manifold, point, u)
        Hv = _hvp(Q, manifold, point, v)
        uHv = float(manifold.inner_product(point, u, Hv))
        vHu = float(manifold.inner_product(point, v, Hu))
        denom = abs(uHv) + abs(vHu) + 1e-12
        asym = abs(uHv - vHu) / denom
        assert asym < 1e-8, f"Product HVP not self-adjoint (asym={asym:.2e})"


# ===========================================================================
# Product with a Positive leaf: the connection term must survive composition.
# ===========================================================================
class TestProductWithPositiveLeaf:
    r"""hvp-product-positive: a ``Product`` containing a ``Positive`` leaf must
    apply the affine connection term to THAT leaf's block (not the additive
    PSD/Euclidean recipe) AND keep cross-leaf coupling.

    Fixture: ``Product(PSDFixedRank(N, 2), Positive())`` with a residual that
    couples the PSD block to the positive scalar. The Positive block's HVP
    quadratic form must reflect ``Q'' + Q'/x`` geometry; the dense full-tree
    reference (geodesic 2nd-derivative along the per-leaf retractions) is the
    truth.
    """

    K = 2

    def _setup(self, seed=0):
        manifold = Product(PSDFixedRank(N, self.K), Positive())
        A_true, _ = _make_psd_target(self.K, seed)
        rng = np.random.default_rng(seed + 9)
        Y = A_true + 0.2 * jnp.asarray(rng.normal(size=(N, self.K)))
        x = jnp.asarray(1.6)  # positive scalar away from 1 so 1/x bites
        gtarget = jnp.asarray(rng.normal(size=(_TRIU.shape[0],)))

        def Q(point):
            Yl, xl = point
            g = (Yl @ Yl.T)[_TRIU[:, 0], _TRIU[:, 1]]
            s = jnp.sum(jnp.tanh(g))
            r_psd = jnp.tanh(g) - jnp.tanh(gtarget)
            # Coupling + a genuinely non-quadratic term in x (so Q'/x != 0).
            return 0.5 * jnp.sum(r_psd * r_psd) + s * jnp.log(xl) + 0.5 * xl**2

        return manifold, (Y, x), Q

    def _geodesic_reference_form(self, manifold, Q, point, eta):
        r"""Metric-exact reference quadratic form per leaf: the 2nd derivative
        of Q along the per-leaf geodesics ``R(t eta)`` gives, summed,
        ``sum_leaf g_leaf(H[eta], eta)``. We isolate the Positive-leaf
        contribution by zeroing the PSD eta (cross term vanishes in the pure
        quadratic form when the other tangent is zero)."""

        def along_positive(t):
            pt = (point[0], manifold.factors[1].retraction(point[1], t * eta[1]))
            return Q(pt)

        return float(jax.grad(jax.grad(along_positive))(0.0))

    def test_positive_leaf_form_in_product(self):
        r"""The Positive leaf inside a Product carries the connection term."""
        manifold, point, Q = self._setup(seed=1)
        # Pure Positive-direction tangent (PSD eta = 0).
        eta = (jnp.zeros_like(point[0]), jnp.asarray(0.8))

        ref_form = self._geodesic_reference_form(manifold, Q, point, eta)
        Hv = _hvp(Q, manifold, point, eta)
        form_impl = float(manifold.inner_product(point, Hv, eta))
        assert form_impl == pytest.approx(ref_form, rel=1e-7, abs=1e-9)

        # And it must NOT be the plain Euclidean Hessian on the Positive block.
        x = float(point[1])

        def Qx(xl):
            return Q((point[0], xl))

        qpp = float(jax.grad(jax.grad(Qx))(point[1]))
        qp = float(jax.grad(Qx)(point[1]))
        wrong = qpp * float(eta[1]) ** 2  # Q'' v^2 (no connection term)
        right = (qpp + qp / x) * float(eta[1]) ** 2
        assert abs(right - wrong) / (abs(right) + 1e-12) > 0.05, "fixture vacuous"
        assert abs(form_impl - wrong) > 0.05 * abs(
            right
        ), "Positive leaf in Product collapsed to the Euclidean Hessian"

    def test_product_with_positive_self_adjoint(self):
        r"""tcg: self-adjointness across a PSD + Positive heterogeneous tree."""
        manifold, point, Q = self._setup(seed=2)
        ku, kv = jax.random.split(jax.random.PRNGKey(800))
        # Random horizontal PSD part + random scalar Positive part.
        amb_u = jax.random.normal(ku, jnp.asarray(point[0]).shape, dtype=jnp.float64)
        amb_v = jax.random.normal(kv, jnp.asarray(point[0]).shape, dtype=jnp.float64)
        u = (
            manifold.factors[0].projection(point[0], amb_u),
            jnp.asarray(0.5),
        )
        v = (
            manifold.factors[0].projection(point[0], amb_v),
            jnp.asarray(-0.9),
        )
        Hu = _hvp(Q, manifold, point, u)
        Hv = _hvp(Q, manifold, point, v)
        uHv = float(manifold.inner_product(point, u, Hv))
        vHu = float(manifold.inner_product(point, v, Hu))
        denom = abs(uHv) + abs(vHu) + 1e-12
        asym = abs(uHv - vHu) / denom
        assert asym < 1e-8, f"PSD+Positive HVP not self-adjoint (asym={asym:.2e})"
