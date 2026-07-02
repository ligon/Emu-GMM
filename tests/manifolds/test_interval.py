"""Unit tests for the bounded scalar :class:`Interval` manifold (#152).

The interval analogue of :class:`Positive`: the logit-pullback geometry on
``(lo, hi)`` whose exponential retraction never crosses either bound. Mirrors
``test_positive.py``. Motivation: a compact ``[lo, hi]`` scale parameter
restores the CUE regularity condition (``V_X`` bounded away from singular) that
fails as ``sigma -> 0``.

Plus the solver scalar-leaf dispatch regression suite (bottom of file): both
``riemannian_lm`` and ``riemannian_tr`` must route an Interval leaf through its
OWN retraction and metric -- the pre-fix ``_is_positive``-only dispatch sent it
down the additive Euclidean path, and accepted iterates escaped the bounds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal.params import (
    flatten_params_with_spec,
    manifold_spec_from_params,
    unflatten_params,
)
from emu_gmm.manifolds import Interval
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.manifolds.riemannian_tr import (
    _build_plan,
    _metric_diag,
    _raise_index_flat,
    _riemannian_hvp,
    riemannian_tr,
)

jax.config.update("jax_enable_x64", True)


class TestIntervalProtocol:
    def test_shape_dim_gauge(self):
        m = Interval(0.5, 3.0)
        assert m.ambient_shape == ()
        assert m.dimension == 1
        assert m.gauge_dim == 0

    def test_requires_lo_lt_hi(self):
        with pytest.raises(ValueError):
            Interval(2.0, 1.0)
        with pytest.raises(ValueError):
            Interval(1.0, 1.0)

    def test_frozen_hashable_equal(self):
        assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
        assert hash(Interval(0.0, 1.0)) == hash(Interval(0.0, 1.0))
        assert Interval(0.0, 1.0) != Interval(0.0, 2.0)


class TestIntervalOperators:
    LO, HI = 0.5, 4.0

    def test_retraction_never_crosses_bounds(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.7)
        # Never crosses either bound, for ANY v (sigmoid in [0,1] => R in [lo,hi]).
        for v in [-1e4, -100.0, -1.0, 0.0, 1.0, 100.0, 1e4]:
            xn = float(m.retraction(x, jnp.asarray(v)))
            assert self.LO <= xn <= self.HI, (v, xn)
        # Strictly interior for moderate steps (extreme v saturates to the bound
        # in float64, exactly as Positive's exp retraction reaches 0 at v -> -inf).
        for v in [-20.0, -1.0, 1.0, 20.0]:
            xn = float(m.retraction(x, jnp.asarray(v)))
            assert self.LO < xn < self.HI, (v, xn)

    def test_retraction_identity_at_zero(self):
        m = Interval(self.LO, self.HI)
        assert float(m.retraction(jnp.asarray(2.3), jnp.asarray(0.0))) == pytest.approx(
            2.3, abs=1e-12
        )

    def test_retraction_differential_is_identity(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(2.3)
        t = 1e-6
        fd = (
            float(m.retraction(x, jnp.asarray(t)))
            - float(m.retraction(x, jnp.asarray(-t)))
        ) / (2.0 * t)
        assert fd == pytest.approx(1.0, rel=1e-6)
        assert float(m.retraction_differential(x)) == pytest.approx(1.0)

    def test_inner_product_and_norm(self):
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.2)
        phip = (self.HI - self.LO) / ((x - self.LO) * (self.HI - x))
        u, v = jnp.asarray(0.7), jnp.asarray(-0.3)
        assert float(m.inner_product(x, u, v)) == pytest.approx(
            float(phip**2 * u * v), rel=1e-10
        )
        assert float(m.norm(x, u)) == pytest.approx(float(jnp.abs(u) * phip), rel=1e-10)

    def test_gradient_relation(self):
        # g_x(rgrad, v) == egrad * v  for all v.
        m = Interval(self.LO, self.HI)
        x = jnp.asarray(1.9)
        egrad = jnp.asarray(0.42)
        rgrad = m.euclidean_to_riemannian_gradient(x, egrad)
        for v in [0.5, -1.3, 2.0]:
            vv = jnp.asarray(v)
            assert float(m.inner_product(x, rgrad, vv)) == pytest.approx(
                float(egrad * vv), rel=1e-10
            )

    def test_distance_symmetric_and_zero_diagonal(self):
        m = Interval(self.LO, self.HI)
        a, b = jnp.asarray(1.0), jnp.asarray(3.0)
        assert float(m.distance(a, a)) == pytest.approx(0.0, abs=1e-12)
        assert float(m.distance(a, b)) == pytest.approx(
            float(m.distance(b, a)), rel=1e-12
        )
        assert float(m.distance(a, b)) > 0.0

    def test_retraction_realises_geodesic_distance(self):
        # ||v||_g == distance(x, R_x(v)): the exp map is a unit-speed geodesic.
        m = Interval(self.LO, self.HI)
        x, v = jnp.asarray(1.5), jnp.asarray(0.8)
        xn = m.retraction(x, v)
        assert float(m.distance(x, xn)) == pytest.approx(float(m.norm(x, v)), rel=1e-8)

    def test_random_point_in_bounds(self):
        m = Interval(self.LO, self.HI)
        for s in range(5):
            p = float(m.random_point(jax.random.PRNGKey(s)))
            assert self.LO < p < self.HI


class TestIntervalAsLeaf:
    def test_spec_resolves_interval_and_roundtrips(self):
        @jdc.pytree_dataclass
        class P:
            sigma: jnp.ndarray
            __emu_manifolds__ = {"sigma": Interval(0.5, 4.0)}

        p = P(sigma=jnp.asarray(1.3))
        # The parameter-space declaration layer resolves the annotated manifold.
        spec = manifold_spec_from_params(p)
        assert spec.leaf_specs[0].manifold == Interval(0.5, 4.0)
        assert int(spec.total_gauge_dim) == 0
        flat, treedef, _ = flatten_params_with_spec(p)
        assert int(flat.shape[0]) == 1
        p2 = unflatten_params(flat, treedef, manifold_spec=spec)
        assert float(p2.sigma) == pytest.approx(1.3, abs=1e-12)


# ===========================================================================
# Solver scalar-leaf dispatch regression: an Interval leaf must ride its OWN
# retraction and metric through BOTH Riemannian solvers.
#
# Bug: the scalar-leaf dispatch in ``riemannian_lm`` (retract / metric_diag)
# and ``riemannian_tr`` (``_retract_flat`` / ``_metric_diag`` /
# ``_raise_index_flat``) tested ``_is_positive`` ONLY, so an Interval leaf
# silently fell through to the additive Euclidean step with the identity
# metric -- accepted iterates could land on/past the bounds, defeating the
# manifold's entire purpose (the #152 CU regularity boundary motivation).
# ===========================================================================
LO_B, HI_B = 0.5, 4.0
# The unconstrained optimum sits OUTSIDE the bounds (just past ``hi``): the
# additive step exits the interval, the Interval retraction cannot. Keeping
# it only 1e-6 past ``hi`` keeps the per-step logit growth ~1, so a
# max_steps <= 8 truncation never reaches float64 sigmoid saturation and
# strict interiority is a robust assertion (see the truncation note below).
TARGET_B = HI_B + 1e-6


@jdc.pytree_dataclass
class _BoundedSigma:
    sigma: jnp.ndarray
    __emu_manifolds__ = {"sigma": Interval(LO_B, HI_B)}


@jdc.pytree_dataclass
class _FreeSigma:
    """The same scalar with the default Euclidean() geometry (the additive
    counterfactual -- bit-for-bit the buggy pre-fix Interval path)."""

    sigma: jnp.ndarray


def _pull_residual(x):
    """M=1 residual pulling sigma toward ``TARGET_B`` (outside the bounds)."""
    return jnp.reshape(x[0] - TARGET_B, (1,))


def _phi_prime(x: float) -> float:
    return (HI_B - LO_B) / ((x - LO_B) * (HI_B - x))


class TestIntervalSolverBoundPreservation:
    """Both solvers must keep every accepted Interval iterate inside (lo, hi).

    Truncation trick: both solvers are deterministic, and a solve stopped at
    ``max_steps=k`` returns the k-th accepted iterate -- so sweeping
    ``k = 1..8`` observes the whole accepted-iterate path, not just the
    final point.
    """

    @pytest.mark.parametrize("k", range(1, 9))
    def test_lm_accepted_iterates_strictly_inside_bounds(self, k):
        p0 = _BoundedSigma(sigma=jnp.asarray(1.0))
        spec = manifold_spec_from_params(p0)
        theta_hat, _info = riemannian_lm(max_steps=k)(_pull_residual, p0, spec)
        s = float(theta_hat.sigma)
        assert LO_B < s < HI_B, (k, s)

    @pytest.mark.parametrize("k", range(1, 9))
    def test_tr_accepted_iterates_strictly_inside_bounds(self, k):
        p0 = _BoundedSigma(sigma=jnp.asarray(1.0))
        spec = manifold_spec_from_params(p0)
        theta_hat, _info = riemannian_tr(max_steps=k)(_pull_residual, p0, spec)
        s = float(theta_hat.sigma)
        assert LO_B < s < HI_B, (k, s)

    def test_solvers_make_real_progress_toward_the_bound(self):
        """Guard: the bounds assertion must not pass via a stuck solver."""
        p0 = _BoundedSigma(sigma=jnp.asarray(1.0))
        spec = manifold_spec_from_params(p0)
        s_lm, _ = riemannian_lm(max_steps=8)(_pull_residual, p0, spec)
        s_tr, _ = riemannian_tr(max_steps=8)(_pull_residual, p0, spec)
        # Empirically ~3.99998 (LM) / ~3.90 (TR); anything past 3.5 shows the
        # solve genuinely chased the out-of-bounds pull.
        assert float(s_lm.sigma) > 3.5
        assert float(s_tr.sigma) > 3.5

    @pytest.mark.parametrize(
        "make_optimizer", [riemannian_lm, riemannian_tr], ids=["lm", "tr"]
    )
    def test_interval_retraction_exercised_vs_additive_counterfactual(
        self, make_optimizer
    ):
        """The Interval fit differs from the additive-step result.

        The Euclidean-leaf control IS the additive path (and bit-for-bit the
        buggy pre-fix Interval behaviour): it converges to ``TARGET_B``,
        OUTSIDE the bounds. The Interval leaf must stay strictly inside --
        proving its own retraction (not ``p0 + d0``) produced the iterates.
        """
        p_b = _BoundedSigma(sigma=jnp.asarray(1.0))
        p_f = _FreeSigma(sigma=jnp.asarray(1.0))
        opt = make_optimizer(max_steps=8)
        th_b, _ = opt(_pull_residual, p_b, manifold_spec_from_params(p_b))
        th_f, _ = opt(_pull_residual, p_f, manifold_spec_from_params(p_f))
        s_b, s_f = float(th_b.sigma), float(th_f.sigma)
        assert s_f > HI_B  # additive path exits the interval
        assert s_b < HI_B  # Interval retraction never does
        assert s_b != s_f


class TestIntervalSolverMetric:
    """The solvers' metric bookkeeping must carry Interval's phi'(x)^2 weight
    exactly the way it carries Positive's 1/x^2."""

    def test_tr_convergence_norm_uses_interval_metric(self):
        """``riemannian_tr``'s reported ``final_gradient_norm`` is
        ``riem_norm(x, g) = |g| * phi'(x)`` -- the Interval weight entering
        the convergence norm exactly as Positive's ``1/x`` does."""
        p0 = _BoundedSigma(sigma=jnp.asarray(1.0))
        spec = manifold_spec_from_params(p0)
        theta_hat, info = riemannian_tr(max_steps=8)(_pull_residual, p0, spec)
        s = float(theta_hat.sigma)
        g_ambient = abs(s - TARGET_B)  # dQ/dsigma of 0.5*(sigma - target)^2
        gnorm = float(jnp.asarray(info.final_gradient_norm))
        assert gnorm == pytest.approx(g_ambient * _phi_prime(s), rel=1e-10)
        # And it is genuinely NOT the identity-metric (ambient) norm: near the
        # bound phi' >> 1, so the metric weighting is load-bearing here.
        assert gnorm != pytest.approx(g_ambient, rel=1e-2, abs=0.0)

    def test_tr_metric_diag_and_raise_index_carry_interval_weights(self):
        """Unit pins on the TR flat-plan helpers: ``_metric_diag`` is
        ``phi'(x)^2`` and ``_raise_index_flat`` divides by it (the inverse
        metric), mirroring Positive's ``1/x^2`` / ``x^2`` pair."""
        p0 = _BoundedSigma(sigma=jnp.asarray(1.7))
        spec = manifold_spec_from_params(p0)
        plan = _build_plan(spec, 1)
        x = jnp.asarray([1.7])
        w2 = _phi_prime(1.7) ** 2
        w = _metric_diag(plan, x)
        assert float(w[0]) == pytest.approx(w2, rel=1e-12)
        raised = _raise_index_flat(plan, x, jnp.asarray([0.42]))
        assert float(raised[0]) == pytest.approx(0.42 / w2, rel=1e-12)

    def test_interval_hvp_carries_connection_term(self):
        """The retraction-pullback HVP consumes Interval's retraction: its
        metric form equals the geodesic second derivative
        ``d^2/dt^2 Q(R_x(t v))|_0 = (Q'' + Q' R''_x(0)) v^2`` -- and is
        measurably NOT the naive Euclidean ``Q'' v^2`` (the connection term
        of the non-additive retraction is real). Mirrors the Positive R3
        gate in ``test_rtr_reductions.py``."""
        m = Interval(LO_B, HI_B)
        x = jnp.asarray(1.7)
        v = jnp.asarray(0.6)

        def Q(z):
            return (z - 3.0) ** 2

        Hv = _riemannian_hvp(Q, m, x, v)
        metric_form = float(m.inner_product(x, Hv, v))

        def f(t: float) -> float:
            return float(Q(m.retraction(x, jnp.asarray(t) * v)))

        t = 1e-4
        fd = (f(t) - 2.0 * f(0.0) + f(-t)) / t**2
        assert metric_form == pytest.approx(fd, rel=1e-4)
        naive = 2.0 * float(v) ** 2  # Q'' = 2 for the quadratic Q
        assert abs(metric_form - naive) > 0.1
