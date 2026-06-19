r"""JAX-native Riemannian Trust Region (RTR) solver (#152).

A second :class:`~emu_gmm.manifolds.optimizer.RiemannianOptimizer`, a drop-in
alternative to :func:`~emu_gmm.manifolds.riemannian_lm.riemannian_lm` that
follows **negative curvature**. ``riemannian_lm`` builds its step from the
Gauss--Newton model Hessian ``J'J`` (PSD by construction), so it can only
describe a locally convex bowl and stalls at the saddles a non-convex GMM
criterion presents. RTR uses the *true* retraction-pullback Hessian
``H = J'J + S`` (the second-order residual term LM drops) and a truncated
(Steihaug--Toint) conjugate-gradient subproblem solver that detects
negative-curvature directions and steps to the trust boundary through them.

Algorithm port + attribution
-----------------------------
The truncated-CG inner solve (:func:`_truncated_cg`) and the trust-region
outer loop are a JAX reimplementation of pymanopt's
``pymanopt/optimizers/trust_regions.py`` (the Absil--Baker--Gallivan RTR /
GenRTR algorithm), including its tuned constants (``rho_prime=0.1``,
``rho_regularization=1e3``, the radius init/expand/shrink heuristics) and the
two-regime ``kappa``/``theta`` stopping rule. pymanopt is **BSD-2-Clause**
(GenRTR (c) 2004--2007 P.-A. Absil, C. G. Baker, K. A. Gallivan; adapted to
Manopt by N. Boumal; ported to pymanopt by J. Townsend). This file *ports* the
algorithm into JAX rather than importing pymanopt at runtime (honouring #3:
native constructors only; pymanopt stays a dev-only gated parity baseline).

HVP -- the retraction-pullback Hessian (revised design, ``gh issue view 152``)
------------------------------------------------------------------------------
``H[eta]`` is the Euclidean Hessian of the retraction pullback
``Q_hat(eta) = Q(R_Y(eta))`` at ``eta = 0`` applied to ``eta``, with its index
raised by the per-leaf inverse metric and projected to horizontal per leaf::

    H[eta] = R( P( jvp(grad(Q o R), 0)[eta_h] ) ),    eta_h = P(eta)

* ONE jvp on the FULL ``eta`` (never sliced before the jvp) so the cross-leaf
  Hessian coupling ``S`` survives for a ``Product``.
* ``P`` is the per-leaf horizontal projection (Lyapunov solve for PSDFixedRank;
  identity for Euclidean / Positive). Two-sided ``P`` makes ``H`` self-adjoint
  on the horizontal subspace -- the precondition tCG conjugacy relies on.
* ``R`` raises the index by the inverse leaf metric: ``x^2`` for a Positive
  leaf (the affine ``1/x^2`` metric), identity for Frobenius leaves. Because
  the exponential retraction ``x e^{v/x}`` is a *second-order* retraction the
  pullback automatically carries the affine connection term -- the metric form
  ``g_x(H[v], v)`` equals the geodesic second derivative ``Q'' + Q'/x``, NOT
  the bare Euclidean ``Q''`` (red-team blocker #4). No manual Christoffel
  algebra is needed.

Metric throughout
-----------------
The tCG inner products, the trust-radius boundary test, and the convergence
norm use the per-leaf *manifold* ``inner_product`` (Frobenius for
PSD/Euclidean; ``1/x^2`` for Positive). The trust-region constraint is applied
in this metric (required for pymanopt parity). For the scalar Positive
boundary case the affine geometry pushes the boundary to infinite distance, so
``sigma -> 0+`` is correct, not a stall (the exponential retraction is
multiplicative and never crosses 0). A pure-JAX ``jnp.where`` NaN-guard on the
``rho`` ``0/0`` floor keeps ``Delta`` finite (pymanopt raises
``ZeroDivisionError``; JAX yields a silent NaN we must trap).

REVERSE-MODE AD IS NOT SUPPORTED (matches ``riemannian_lm`` / #77): the outer
loop is a :func:`jax.lax.while_loop`, which has no reverse rule.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from emu_gmm._internal import params as params_mod
from emu_gmm._internal.fn_cache import FunctionKeyedCache
from emu_gmm.manifolds.positive import Positive
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec

if TYPE_CHECKING:
    from emu_gmm.types import OptimizerInfo


# pymanopt's tuned constants (ported, not imported).
_RHO_PRIME: float = 0.1
_RHO_REGULARIZATION: float = 1e3
_GAUGE_FLOOR: float = 1e-6
# #152/#156: ftol (cost-stagnation) convergence, mirroring riemannian_lm's #156
# stop. After ``_FTOL_PATIENCE`` consecutive ACCEPTED steps whose relative cost
# reduction is below ``_FTOL`` the iterate is at the achievable cost floor;
# certify. GATED on ``total_gauge_dim > 0`` (gauge-bearing leaves only) -- a
# gauge-free / scalar tree keeps its exact pre-#156 stopping point so a pinned
# boundary J is unchanged (the sigma->0 boundary is a consumer-model issue, not
# a solver one; see #159). Cost is gauge-invariant, so a fired ftol stop is
# identical along the O(k) fibre -- unlike a step-norm stop it never perturbs
# the gauge-equivariant step count.
_FTOL: float = 1e-8
_FTOL_PATIENCE: int = 8
# Trust-radius stagnation floor (RTR companion to ftol; see the converged-block
# note). Far below any useful radius, so a clean run -- which certifies on the
# gradient norm with Delta still O(1) -- never reaches it. Also gated on
# total_gauge_dim > 0.
_MIN_RADIUS: float = 1e-12

# tCG stop-reason codes (string mirror of pymanopt's six integer codes).
_NEG_CURV = "negative_curvature"
_EXCEEDED_TR = "exceeded_tr"
_REACHED_LINEAR = "reached_target_linear"
_REACHED_SUPERLINEAR = "reached_target_superlinear"
_MAX_INNER = "max_inner"
_MODEL_INCREASED = "model_increased"

# Integer encoding of the stop reasons for the traced inner loop (a string
# cannot ride a ``lax.while_loop`` carry). Decoded back to the string at the
# eager boundary.
_STOP_CODE = {
    _NEG_CURV: 0,
    _EXCEEDED_TR: 1,
    _REACHED_LINEAR: 2,
    _REACHED_SUPERLINEAR: 3,
    _MAX_INNER: 4,
    _MODEL_INCREASED: 5,
}
_STOP_NAME = {v: k for k, v in _STOP_CODE.items()}

# #124/#139 cache-leak-safe memoised jitted solve cores for the args= path.
_TRACED_SOLVE_CACHE = FunctionKeyedCache("_emu_gmm_traced_tr_solve")


def _is_positive(manifold: Any) -> bool:
    return isinstance(manifold, Positive)


def _is_manifold(obj: Any) -> bool:
    """Duck-type a ManifoldParam (it has projection + inner_product)."""
    return hasattr(obj, "projection") and hasattr(obj, "inner_product")


# ===========================================================================
# Per-leaf flat plan helpers (mirrors riemannian_lm's plan / project / metric).
# ===========================================================================
def _build_plan(
    manifold_spec: ManifoldSpec, K: int
) -> list[tuple[int, int, tuple, Any]]:
    leaf_specs: tuple[LeafSpec, ...] = manifold_spec.leaf_specs
    plan: list[tuple[int, int, tuple, Any]] = []
    running = 0
    for ls in leaf_specs:
        size = int(np.prod(ls.ambient_shape)) if ls.ambient_shape != () else 1
        plan.append((ls.offset, size, ls.ambient_shape, ls.manifold))
        running += size
    if not (int(manifold_spec.total_ambient_dim) == running == K):
        raise ValueError(
            "riemannian_tr: manifold_spec does not tile the flat parameter "
            f"buffer: total_ambient_dim={manifold_spec.total_ambient_dim}, "
            f"sum(block sizes)={running}, len(theta_flat)={K}"
        )
    return plan


def _project_flat(plan: list, x: jnp.ndarray, v_flat: jnp.ndarray) -> jnp.ndarray:
    """Per-leaf horizontal projection of an ambient flat vector at ``x``."""
    parts = []
    for offset, size, shape, manifold in plan:
        pt = x[offset : offset + size]
        vv = v_flat[offset : offset + size]
        if shape == ():
            parts.append(vv)  # scalar leaf: projection identity
            continue
        proj_m = manifold.projection(jnp.reshape(pt, shape), jnp.reshape(vv, shape))
        parts.append(jnp.reshape(proj_m, (size,)))
    return jnp.concatenate(parts)


def _raise_index_flat(plan: list, x: jnp.ndarray, v_flat: jnp.ndarray) -> jnp.ndarray:
    r"""Raise the index of an ambient cotangent flat vector by ``G^{-1}``.

    For a Positive leaf the inverse affine metric is ``x^2``; for Frobenius
    leaves (PSDFixedRank / Euclidean) it is the identity. This is what turns
    the Euclidean pullback Hessian ``\nabla^2\hat h`` into the Riemannian
    Hessian operator whose metric form is the geodesic second derivative.
    """
    parts = []
    for offset, size, shape, manifold in plan:
        vv = v_flat[offset : offset + size]
        if shape == () and _is_positive(manifold):
            pt = x[offset : offset + size]
            parts.append(vv * (pt[0] ** 2))
        else:
            parts.append(vv)
    return jnp.concatenate(parts)


def _metric_diag(plan: list, x: jnp.ndarray) -> jnp.ndarray:
    """Diagonal metric tensor ``G`` as a flat (K,) weight (1/x^2 for Positive)."""
    parts = []
    for offset, size, shape, manifold in plan:
        if shape == () and _is_positive(manifold):
            pt = x[offset : offset + size]
            parts.append(jnp.reshape(1.0 / (pt[0] ** 2), (1,)))
        else:
            parts.append(jnp.ones((size,), dtype=x.dtype))
    return jnp.concatenate(parts)


def _inner(plan: list, x: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Manifold inner product <u, v>_g in the flat ambient layout."""
    g = _metric_diag(plan, x)
    return jnp.sum(g * u * v)


def _retract_flat(plan: list, x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Per-leaf retraction of a flat step (additive / exponential / additive)."""
    parts = []
    for offset, size, shape, manifold in plan:
        pt = x[offset : offset + size]
        dd = d[offset : offset + size]
        if shape == ():
            p0, d0 = pt[0], dd[0]
            new = p0 * jnp.exp(d0 / p0) if _is_positive(manifold) else p0 + d0
            parts.append(jnp.reshape(new, (1,)))
            continue
        new_m = manifold.retraction(jnp.reshape(pt, shape), jnp.reshape(dd, shape))
        parts.append(jnp.reshape(new_m, (size,)))
    return jnp.concatenate(parts)


def _min_eig_YtY(plan: list, x: jnp.ndarray) -> jnp.ndarray:
    """Smallest eigenvalue of ``Y^T Y`` over PSD leaves (rank-drop sentinel).

    Returns ``+inf`` when there is no PSD leaf (no rank structure to lose).
    """
    from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank

    vals = []
    for offset, size, shape, manifold in plan:
        if isinstance(manifold, PSDFixedRank):
            Y = jnp.reshape(x[offset : offset + size], shape)
            ev = jnp.linalg.eigvalsh(Y.T @ Y)
            vals.append(jnp.min(ev))
    if not vals:
        # No PSD leaf -> no rank structure to lose; a large finite positive
        # sentinel so the rank-drop guard passes (NOT +inf, which would fail
        # an ``isfinite`` check downstream).
        return jnp.asarray(1.0)
    return jnp.min(jnp.stack(vals))


# ===========================================================================
# HVP -- canonical flat form + polymorphic dispatch.
# ===========================================================================
def _hvp_flat(
    residual_fn: Callable[..., jnp.ndarray],
    theta_flat: jnp.ndarray,
    manifold_spec: ManifoldSpec,
    eta_flat: jnp.ndarray,
    *,
    args: Any = None,
) -> jnp.ndarray:
    r"""The canonical retraction-pullback Riemannian HVP (flat layout).

    ``H[eta] = R(P( jvp(grad(Q o R_theta), 0)[ P(eta) ] ))`` where ``Q`` is the
    half-squared-norm criterion of ``residual_fn`` and ``R_theta`` the per-leaf
    retraction at ``theta_flat``. ONE jvp on the FULL eta (cross-leaf ``S``
    preserved); two-sided horizontal ``P`` (self-adjoint); index raise ``R``
    (affine connection term on Positive leaves).
    """
    K = int(theta_flat.shape[0])
    plan = _build_plan(manifold_spec, K)

    def Q_of_flat(tf: jnp.ndarray) -> jnp.ndarray:
        r = residual_fn(tf) if args is None else residual_fn(tf, args)
        return 0.5 * jnp.sum(r * r)

    def Q_hat(eta: jnp.ndarray) -> jnp.ndarray:
        # Pullback through the per-leaf retraction at theta_flat.
        return Q_of_flat(_retract_flat(plan, theta_flat, eta))

    eta_h = _project_flat(plan, theta_flat, eta_flat)
    grad_hat = jax.grad(Q_hat)
    _, hv = jax.jvp(grad_hat, (jnp.zeros_like(theta_flat),), (eta_h,))
    # Project output to horizontal, then raise the index (disjoint leaf types).
    return _raise_index_flat(plan, theta_flat, _project_flat(plan, theta_flat, hv))


def _hvp_per_leaf(Q: Callable[[Any], Any], manifold: Any, point: Any, eta: Any) -> Any:
    r"""Per-leaf HVP: ``Q`` maps a manifold point (array or tuple) -> scalar.

    Mirrors :func:`_hvp_flat` but pytree-native (the signature the
    ``test_rtr_hvp`` / ``test_rtr_reductions`` shims and ``test_rtr_trust_region``
    use directly). Builds a one-leaf or Product flat plan, flattens, delegates
    to :func:`_hvp_flat`, and unflattens to the input structure.
    """
    from emu_gmm.manifolds.product import Product

    if isinstance(manifold, Product):
        factors = manifold.factors
        points = tuple(jnp.asarray(p) for p in point)
        etas = tuple(jnp.asarray(e) for e in eta)
        shapes = [tuple(int(s) for s in p.shape) for p in points]
        sizes = [int(np.prod(s)) if s != () else 1 for s in shapes]
        offsets, off = [], 0
        for s in sizes:
            offsets.append(off)
            off += s
        plan = [
            (offsets[i], sizes[i], shapes[i], factors[i]) for i in range(len(factors))
        ]
        theta_flat = jnp.concatenate(
            [jnp.reshape(points[i], (sizes[i],)) for i in range(len(factors))]
        )
        eta_flat = jnp.concatenate(
            [jnp.reshape(etas[i], (sizes[i],)) for i in range(len(factors))]
        )
        spec = _plan_to_spec(plan)

        def Q_flat(tf: jnp.ndarray) -> jnp.ndarray:
            pt = tuple(
                jnp.reshape(tf[offsets[i] : offsets[i] + sizes[i]], shapes[i])
                for i in range(len(factors))
            )
            return Q(pt)

        h_flat = _hvp_flat_from_Q(Q_flat, theta_flat, spec, eta_flat, plan)
        return tuple(
            jnp.reshape(h_flat[offsets[i] : offsets[i] + sizes[i]], shapes[i])
            for i in range(len(factors))
        )

    # Single leaf.
    pt = jnp.asarray(point)
    ev = jnp.asarray(eta)
    shape = tuple(int(s) for s in pt.shape)
    size = int(np.prod(shape)) if shape != () else 1
    plan = [(0, size, shape, manifold)]
    spec = _plan_to_spec(plan)
    theta_flat = jnp.reshape(pt, (size,))
    eta_flat = jnp.reshape(ev, (size,))

    def Q_flat_leaf(tf: jnp.ndarray) -> jnp.ndarray:
        return Q(jnp.reshape(tf, shape) if shape != () else tf[0])

    h_flat = _hvp_flat_from_Q(Q_flat_leaf, theta_flat, spec, eta_flat, plan)
    return jnp.reshape(h_flat, shape) if shape != () else h_flat[0]


def _plan_to_spec(plan: list) -> ManifoldSpec:
    leaf_specs = tuple(
        LeafSpec(offset=off, ambient_shape=shape, manifold=manifold)
        for (off, size, shape, manifold) in plan
    )
    total = sum(size for (_o, size, _s, _m) in plan)
    gauge = sum(int(m.gauge_dim) for (_o, _sz, _s, m) in plan)
    return ManifoldSpec(
        leaf_specs=leaf_specs,
        total_ambient_dim=total,
        total_dimension=total,
        total_gauge_dim=gauge,
    )


def _hvp_flat_from_Q(
    Q_flat: Callable[[jnp.ndarray], jnp.ndarray],
    theta_flat: jnp.ndarray,
    manifold_spec: ManifoldSpec,
    eta_flat: jnp.ndarray,
    plan: list,
) -> jnp.ndarray:
    """HVP core taking ``Q_flat`` directly (used by the per-leaf path)."""

    def Q_hat(eta: jnp.ndarray) -> jnp.ndarray:
        return Q_flat(_retract_flat(plan, theta_flat, eta))

    eta_h = _project_flat(plan, theta_flat, eta_flat)
    _, hv = jax.jvp(jax.grad(Q_hat), (jnp.zeros_like(theta_flat),), (eta_h,))
    return _raise_index_flat(plan, theta_flat, _project_flat(plan, theta_flat, hv))


def _riemannian_hvp(*args: Any, **kwargs: Any) -> Any:
    """Polymorphic Riemannian HVP entry point (see module docstring).

    Three call conventions are dispatched by argument type:

    1. ``_riemannian_hvp(residual_fn, theta_flat, manifold_spec, eta_flat)``
       -- the canonical FLAT form (tcg / integration tests). ``args=`` kw
       optional for the two-argument kernel path.
    2. ``_riemannian_hvp(residual_fn, theta_flat, eta_flat, manifold_spec)``
       -- the gauge-numerical FLAT form (spec last). Disambiguated by which
       positional is the :class:`ManifoldSpec`.
    3. ``_riemannian_hvp(Q, manifold, point, eta)`` -- the per-leaf pytree
       form (hvp / reductions / trust_region tests), ``Q`` maps a manifold
       point to a real scalar.
    """
    pos = list(args)
    # Per-leaf form: second positional is a ManifoldParam instance.
    if len(pos) >= 2 and _is_manifold(pos[1]):
        Q, manifold, point, eta = pos[0], pos[1], pos[2], pos[3]
        return _hvp_per_leaf(Q, manifold, point, eta)
    # Flat forms: a ManifoldSpec sits at position 1, 2, or 3, disambiguating
    # the (residual_fn, spec, theta, eta) / canonical (..., spec, eta) /
    # gauge (..., eta, spec) orderings the sibling test files use.
    if len(pos) >= 4 and isinstance(pos[1], ManifoldSpec):
        residual_fn, manifold_spec, theta_flat, eta_flat = (
            pos[0],
            pos[1],
            pos[2],
            pos[3],
        )
    elif len(pos) >= 3 and isinstance(pos[2], ManifoldSpec):
        residual_fn, theta_flat, manifold_spec, eta_flat = (
            pos[0],
            pos[1],
            pos[2],
            pos[3],
        )
    elif len(pos) >= 4 and isinstance(pos[3], ManifoldSpec):
        residual_fn, theta_flat, eta_flat, manifold_spec = (
            pos[0],
            pos[1],
            pos[2],
            pos[3],
        )
    else:
        raise TypeError(
            "_riemannian_hvp: could not resolve the calling convention "
            f"(got {len(pos)} positional args); expected a ManifoldSpec at "
            "position 1, 2 or 3, or a ManifoldParam at position 1"
        )
    return _hvp_flat(
        residual_fn,
        jnp.asarray(theta_flat),
        manifold_spec,
        jnp.asarray(eta_flat),
        args=kwargs.get("args"),
    )


# ===========================================================================
# Truncated CG (Steihaug-Toint) -- canonical flat core + polymorphic dispatch.
# ===========================================================================
def _tcg_core(
    hvp: Callable[[jnp.ndarray], jnp.ndarray],
    grad_flat: jnp.ndarray,
    plan: list,
    point_flat: jnp.ndarray,
    Delta: jnp.ndarray,
    *,
    theta: float,
    kappa: float,
    min_inner: int,
    max_tcg_steps: int,
    ambient_metric: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    r"""Steihaug--Toint truncated CG (ported from pymanopt), flat layout.

    Solves ``min_eta <g, eta> + 1/2 <eta, H eta>  s.t. ||eta||_g <= Delta``
    using only ``hvp`` and the per-leaf manifold inner product (``_inner``).
    Returns ``(eta, Heta, info)`` with ``info['num_inner']`` and
    ``info['stop_reason']``.

    Eager Python loop with ``lax.cond``-free pure-JAX branch arithmetic so it
    runs under ``jax.jit`` (the outer solve jits it). The two-regime
    ``kappa``/``theta`` stop and the ``d_Hd``-sign / boundary tau handling are
    pymanopt-faithful; a ``d_Hd == 0`` division guard routes the zero-curvature
    case to the boundary branch BEFORE dividing.
    """
    Delta = jnp.asarray(Delta, dtype=grad_flat.dtype)
    x = point_flat
    dt = grad_flat.dtype

    def inner(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # ``ambient_metric`` selects the FROBENIUS (identity) metric for the
        # tCG trust constraint, the choice ``riemannian_lm`` makes so the
        # ambient step is not throttled to ``~Delta * sigma`` at small sigma
        # (the Positive boundary regime). The flat-spec path (canonical
        # convention) keeps the per-leaf manifold metric (pymanopt parity).
        if ambient_metric:
            return jnp.sum(u * v)
        return _inner(plan, x, u, v)

    def retang(v: jnp.ndarray) -> jnp.ndarray:
        return _project_flat(plan, x, v)

    r0 = grad_flat
    r_r0 = inner(r0, r0)
    norm_r0 = jnp.sqrt(r_r0)
    target = norm_r0 * jnp.minimum(norm_r0**theta, kappa)
    # Two-regime classification (traced): linear if kappa < ||r0||^theta.
    linear_regime = kappa < (norm_r0**theta)
    Delta2 = Delta * Delta

    # Fully-traced unrolled loop with a ``stopped`` mask: once a stop fires the
    # carry freezes (subsequent iterations are no-ops via jnp.where), so the
    # static-bound Python ``for`` traces cleanly under jit. ``num_inner`` and
    # ``stop_code`` advance only while not stopped.
    z0 = r0
    z_r0 = inner(z0, r0)
    carry = {
        "eta": jnp.zeros_like(grad_flat),
        "Heta": jnp.zeros_like(grad_flat),
        "r": r0,
        "z": z0,
        "z_r": z_r0,
        "d_Pd": z_r0,
        "delta": -z0,
        "e_Pe": jnp.asarray(0.0, dtype=dt),
        "e_Pd": jnp.asarray(0.0, dtype=dt),
        "model_value": jnp.asarray(0.0, dtype=dt),
        "stopped": jnp.asarray(False),
        "stop_code": jnp.asarray(_STOP_CODE[_MAX_INNER]),
        "num_inner": jnp.asarray(0),
    }

    def step(c: dict, j: int) -> dict:
        stopped = c["stopped"]
        run = jnp.logical_not(stopped)
        delta = c["delta"]
        eta = c["eta"]
        Heta = c["Heta"]
        e_Pe = c["e_Pe"]
        e_Pd = c["e_Pd"]
        d_Pd = c["d_Pd"]
        z_r = c["z_r"]

        Hdelta = hvp(delta)
        d_Hd = inner(delta, Hdelta)

        nonzero = jnp.abs(d_Hd) > 0.0
        alpha = jnp.where(nonzero, z_r / jnp.where(nonzero, d_Hd, 1.0), 0.0)
        e_Pe_new = jnp.where(nonzero, e_Pe + 2.0 * alpha * e_Pd + alpha**2 * d_Pd, e_Pe)

        neg_curv = d_Hd <= 0.0
        exceed = e_Pe_new >= Delta2
        bail = jnp.logical_or(neg_curv, exceed)

        # Boundary branch (tau step).
        disc = jnp.maximum(e_Pd * e_Pd + d_Pd * (Delta2 - e_Pe), 0.0)
        d_Pd_safe = jnp.where(jnp.abs(d_Pd) > 0.0, d_Pd, 1.0)
        tau = (-e_Pd + jnp.sqrt(disc)) / d_Pd_safe
        eta_bail = eta + tau * delta
        Heta_bail = Heta + tau * Hdelta
        code_bail = jnp.where(neg_curv, _STOP_CODE[_NEG_CURV], _STOP_CODE[_EXCEEDED_TR])

        # Interior branch.
        new_eta = eta + alpha * delta
        new_Heta = Heta + alpha * Hdelta
        new_model_value = inner(new_eta, grad_flat) + 0.5 * inner(new_eta, new_Heta)
        model_inc = new_model_value >= c["model_value"]

        r_new = c["r"] + alpha * Hdelta
        norm_r = jnp.sqrt(inner(r_new, r_new))
        residual_hit = jnp.logical_and(j >= min_inner, norm_r <= target)
        code_resid = jnp.where(
            linear_regime, _STOP_CODE[_REACHED_LINEAR], _STOP_CODE[_REACHED_SUPERLINEAR]
        )

        z_new = r_new
        z_r_new = inner(z_new, r_new)
        beta = z_r_new / jnp.where(jnp.abs(z_r) > 0.0, z_r, 1.0)
        delta_new = retang(-z_new + beta * delta)
        e_Pd_new = beta * (e_Pd + alpha * d_Pd)
        d_Pd_new = z_r_new + beta * beta * d_Pd

        # Decide this iteration's outcome (only meaningful when ``run``).
        # Priority: bail (boundary) > model_increased > residual stop > continue.
        fired_code = jnp.where(
            bail,
            code_bail,
            jnp.where(model_inc, _STOP_CODE[_MODEL_INCREASED], code_resid),
        )
        fired = jnp.logical_or(bail, jnp.logical_or(model_inc, residual_hit))

        # eta / Heta after this iteration.
        eta_out_interior = jnp.where(model_inc, eta, new_eta)
        Heta_out_interior = jnp.where(model_inc, Heta, new_Heta)
        eta_iter = jnp.where(bail, eta_bail, eta_out_interior)
        Heta_iter = jnp.where(bail, Heta_bail, Heta_out_interior)

        # Carry updates for a *continuing* interior step.
        eta_cont = new_eta
        Heta_cont = new_Heta

        # Compose the post-iteration carry, gated by ``run`` and ``fired``.
        def pick(new_val: Any, old_val: Any) -> Any:
            return jnp.where(run, new_val, old_val)

        eta_final = pick(eta_iter, eta)
        Heta_final = pick(Heta_iter, Heta)

        # State that only advances on a continuing interior step.
        continue_step = jnp.logical_and(run, jnp.logical_not(fired))
        eta_state = jnp.where(continue_step, eta_cont, eta_final)
        Heta_state = jnp.where(continue_step, Heta_cont, Heta_final)

        return {
            "eta": eta_state,
            "Heta": Heta_state,
            "r": jnp.where(continue_step, r_new, c["r"]),
            "z": jnp.where(continue_step, z_new, c["z"]),
            "z_r": jnp.where(continue_step, z_r_new, z_r),
            "d_Pd": jnp.where(continue_step, d_Pd_new, d_Pd),
            "delta": jnp.where(continue_step, delta_new, delta),
            "e_Pe": jnp.where(continue_step, e_Pe_new, e_Pe),
            "e_Pd": jnp.where(continue_step, e_Pd_new, e_Pd),
            "model_value": jnp.where(continue_step, new_model_value, c["model_value"]),
            "stopped": jnp.logical_or(stopped, jnp.logical_and(run, fired)),
            "stop_code": jnp.where(
                jnp.logical_and(run, fired), fired_code, c["stop_code"]
            ),
            "num_inner": c["num_inner"] + jnp.where(run, 1, 0),
        }

    for j in range(int(max_tcg_steps)):
        carry = step(carry, j)

    stop_code = int(carry["stop_code"]) if _is_concrete(carry["stop_code"]) else None
    info = {
        "num_inner": carry["num_inner"],
        "inner_iters": _maybe_int(carry["num_inner"]),
        "stop_reason": (
            _STOP_NAME[stop_code] if stop_code is not None else carry["stop_code"]
        ),
        "stop_code": carry["stop_code"],
    }
    return carry["eta"], carry["Heta"], info


def _is_concrete(x: Any) -> bool:
    try:
        int(x)
        return True
    except (
        jax.errors.TracerIntegerConversionError,
        TypeError,
        jax.errors.ConcretizationTypeError,
    ):
        return False


def _maybe_int(x: Any) -> Any:
    return int(x) if _is_concrete(x) else x


def _truncated_cg(*args: Any, **kwargs: Any) -> Any:
    """Polymorphic truncated-CG entry point.

    Three call conventions:

    1. ``_truncated_cg(hvp, grad_flat, spec, point_flat, Delta, *, theta,
       kappa, min_inner, max_tcg_steps)`` -> ``(eta, Heta, info)`` (canonical;
       tcg / reductions tests). ``info`` exposes ``num_inner`` /
       ``stop_reason`` as attributes.
    2. ``_truncated_cg(hvp, grad, radius=, max_inner=, kappa=, theta=,
       min_inner=)`` -> ``(eta, info)`` with ``info['inner_iters']`` (gauge).
       The point defaults to a zero vector (identity Frobenius metric).
    3. ``_truncated_cg(grad=, hvp=, manifold=, point=, Delta=, max_tcg_steps=,
       kappa=, theta=, min_inner=)`` -> ``(eta, info)`` (trust_region, per-leaf
       single-manifold form).
    """
    # ---- Convention 2: gauge-numerical (radius= / max_inner= kwargs). ----
    if "radius" in kwargs or "max_inner" in kwargs:
        hvp: Any = kwargs.get("hvp", args[0] if len(args) > 0 else None)
        grad = jnp.asarray(kwargs.get("grad", args[1] if len(args) > 1 else None))
        radius = jnp.asarray(kwargs["radius"])
        max_inner = int(kwargs["max_inner"])
        # Identity (Frobenius) metric: a single Euclidean leaf of full width.
        plan: list = [
            (0, int(grad.shape[0]), (int(grad.shape[0]),), _Euc(int(grad.shape[0])))
        ]
        eta, _Heta, info = _tcg_core(
            hvp,
            grad,
            plan,
            jnp.zeros_like(grad),
            radius,
            theta=float(kwargs.get("theta", 1.0)),
            kappa=float(kwargs.get("kappa", 0.1)),
            min_inner=int(kwargs.get("min_inner", 1)),
            max_tcg_steps=max_inner,
        )
        return eta, info

    # ---- Convention 3: trust_region (manifold= kwarg, per-leaf). ----
    if "manifold" in kwargs:
        manifold = kwargs["manifold"]
        hvp = kwargs.get("hvp", args[0] if len(args) > 0 else None)
        grad = jnp.asarray(kwargs["grad"])
        point = kwargs["point"]
        Delta = jnp.asarray(kwargs["Delta"])
        pt = jnp.asarray(point)
        shape = tuple(int(s) for s in pt.shape)
        size = int(np.prod(shape)) if shape != () else 1
        plan = [(0, size, shape, manifold)]
        point_flat = jnp.reshape(pt, (size,))
        grad_flat = jnp.reshape(grad, (size,))

        def hvp_flat(eta_flat: jnp.ndarray) -> jnp.ndarray:
            ev = jnp.reshape(eta_flat, shape) if shape != () else eta_flat[0]
            hv = hvp(ev)
            return jnp.reshape(hv, (size,))

        eta, _Heta, info = _tcg_core(
            hvp_flat,
            grad_flat,
            plan,
            point_flat,
            Delta,
            theta=float(kwargs.get("theta", 1.0)),
            kappa=float(kwargs.get("kappa", 0.1)),
            min_inner=int(kwargs.get("min_inner", 1)),
            max_tcg_steps=int(kwargs["max_tcg_steps"]),
            ambient_metric=True,
        )
        step = jnp.reshape(eta, shape) if shape != () else eta[0]
        return step, info

    # ---- Convention 1: canonical flat (spec + point + Delta; positional
    # or keyword). ``_truncated_cg(hvp, grad_flat, spec, point_flat, Delta,
    # *, theta, kappa, min_inner, max_tcg_steps)``. ----
    pos = list(args)

    def _take(name: str, idx: int) -> Any:
        if name in kwargs:
            return kwargs[name]
        if idx < len(pos):
            return pos[idx]
        raise TypeError(f"_truncated_cg: missing required argument {name!r}")

    hvp = _take("hvp", 0)
    grad_flat = _take("grad_flat", 1)
    spec = _take("spec", 2)
    point_flat = _take("point_flat", 3)
    Delta = _take("Delta", 4)
    grad_flat = jnp.asarray(grad_flat)
    point_flat = jnp.asarray(point_flat)
    plan = _build_plan(spec, int(point_flat.shape[0]))
    eta, Heta, info_d = _tcg_core(
        hvp,
        grad_flat,
        plan,
        point_flat,
        jnp.asarray(Delta),
        theta=float(kwargs.get("theta", 1.0)),
        kappa=float(kwargs.get("kappa", 0.1)),
        min_inner=int(kwargs.get("min_inner", 1)),
        max_tcg_steps=int(kwargs["max_tcg_steps"]),
    )
    return eta, Heta, _TCGInfo(info_d)


class _Euc:
    """Minimal Frobenius (identity-metric) leaf for the gauge-numerical path."""

    gauge_dim = 0

    def __init__(self, d: int) -> None:
        self.ambient_shape = (d,)
        self.dimension = d

    def projection(self, point: Any, v: Any) -> Any:  # noqa: ARG002
        return v

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:  # noqa: ARG002
        return jnp.sum(jnp.asarray(u) * jnp.asarray(v))


@dataclasses.dataclass(frozen=True)
class _TCGInfo:
    """Attribute view over the tCG info dict (``info.num_inner`` etc.)."""

    _d: dict

    @property
    def num_inner(self) -> Any:
        return self._d["num_inner"]

    @property
    def stop_reason(self) -> str:
        return self._d["stop_reason"]

    @property
    def inner_iters(self) -> int:
        return self._d["inner_iters"]

    def __getitem__(self, key: str) -> Any:
        return self._d[key]


@dataclasses.dataclass(frozen=True)
class _TRTrace:
    """Attribute view over the per-outer-step trust-region trace (#152).

    The pymanopt-parity gate (``test_rtr_pymanopt_parity.py``) reads the trace
    via ATTRIBUTE access on lowercase, partly-renamed fields
    (``tr_trace.delta`` / ``.rho`` / ``.grad_norm`` / ``.tcg_stop`` /
    ``.n_negcurv``). Internally the solve accumulates a raw dict keyed
    ``'Delta'`` / ``'rho'`` / ``'grad_norm'`` / ``'stop_code'`` / ... ; this
    frozen view maps the spec names onto that dict (still kept verbatim under
    ``OptimizerInfo.extra`` for the dict-style ``_info_get`` path) and carries
    the scalar negative-curvature count. ``stop_code`` is cast to int so
    ``tcg_stop`` matches pymanopt's integer stop-reason enumeration.
    """

    _d: dict
    _n_negcurv: Any

    @property
    def delta(self) -> Any:
        return self._d["Delta"]

    @property
    def rho(self) -> Any:
        return self._d["rho"]

    @property
    def grad_norm(self) -> Any:
        return self._d["grad_norm"]

    @property
    def tcg_stop(self) -> Any:
        return jnp.asarray(self._d["stop_code"], dtype=jnp.int32)

    @property
    def n_negcurv(self) -> Any:
        return self._n_negcurv

    def __getitem__(self, key: str) -> Any:
        return self._d[key]


# ===========================================================================
# The trust-region outer loop + RiemannianOptimizer adapter.
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class _RiemannianTR:
    """Callable Riemannian Trust Region solver (RiemannianOptimizer protocol)."""

    rtol: float
    atol: float
    max_steps: int
    rho_prime: float
    kappa: float
    theta: float
    min_inner: int
    max_tcg_steps: int | None
    max_radius: float | None
    init_radius: float | None
    gauge_floor: float
    ftol: float
    ftol_patience: int

    def __call__(
        self,
        residual_fn: Callable[..., Any],
        theta_init: Any,
        manifold_spec: ManifoldSpec,
        *,
        args: Any = None,
    ) -> tuple[Any, "OptimizerInfo"]:
        from emu_gmm.types import OptimizerInfo

        theta_flat, treedef, flat_spec = params_mod.flatten_params_with_spec(theta_init)
        K = int(theta_flat.shape[0])
        if manifold_spec is None:
            manifold_spec = flat_spec
        plan = _build_plan(manifold_spec, K)

        total_dim = int(manifold_spec.total_dimension)
        total_gauge = int(manifold_spec.total_gauge_dim)
        identified = total_dim - total_gauge
        # ftol (cost-stagnation) certification is gated on gauge structure
        # (#156): a gauge-free / scalar tree keeps its exact gradient-norm
        # stopping point (so a pinned boundary J is unchanged), while a
        # gauge-bearing leaf -- whose CU criterion gradient has a noise floor
        # the gradient test cannot reach -- certifies on cost stationarity.
        ftol_active = total_gauge > 0
        ftol = float(self.ftol)
        ftol_patience = int(self.ftol_patience)

        # Intrinsic-dimension defaults (NOT ambient nk): maxinner / Delta_bar
        # from the identified quotient dimension.
        max_tcg = (
            int(self.max_tcg_steps) if self.max_tcg_steps is not None else identified
        )
        Delta_bar = (
            float(self.max_radius)
            if self.max_radius is not None
            else float(np.sqrt(identified))
        )
        Delta0 = (
            float(self.init_radius) if self.init_radius is not None else Delta_bar / 8.0
        )

        rho_prime = float(self.rho_prime)
        kappa, theta_p, min_inner = (
            float(self.kappa),
            float(self.theta),
            int(self.min_inner),
        )
        rho_reg_coef = _RHO_REGULARIZATION

        two_arg = args is not None

        def Q_data(x: jnp.ndarray, args_in: Any) -> jnp.ndarray:
            r = residual_fn(x, args_in) if two_arg else residual_fn(x)
            return 0.5 * jnp.sum(r * r)

        def egrad(x: jnp.ndarray, args_in: Any) -> jnp.ndarray:
            return jax.grad(lambda z: Q_data(z, args_in))(x)

        def horiz_grad(x: jnp.ndarray, args_in: Any) -> jnp.ndarray:
            # AMBIENT horizontal gradient (project only, no index raise) --
            # the LM-consistent step direction: the trust-region step lives in
            # the ambient metric (riemannian_lm uses the ambient horizontal
            # Gram, not the affine x^2-raised gradient). The affine 1/x^2
            # metric enters only the reported convergence NORM below, matching
            # ``riemannian_lm.riem_norm``.
            eg = egrad(x, args_in)
            return _project_flat(plan, x, eg)

        def riem_norm(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.maximum(_inner(plan, x, v, v), 0.0))

        def hvp_at(
            x: jnp.ndarray, args_in: Any
        ) -> Callable[[jnp.ndarray], jnp.ndarray]:
            def Q_hat(eta: jnp.ndarray) -> jnp.ndarray:
                return Q_data(_retract_flat(plan, x, eta), args_in)

            def hvp(eta_flat: jnp.ndarray) -> jnp.ndarray:
                # AMBIENT horizontal pullback Hessian (project only, no raise):
                # consistent with the ambient step + the ``ambient_metric`` tCG
                # below. For PSD/Euclidean this is identical to the raised
                # operator (Frobenius metric); only Positive leaves differ, and
                # there the ambient choice is what matches riemannian_lm at the
                # sigma -> 0 boundary.
                eta_h = _project_flat(plan, x, eta_flat)
                _, hv = jax.jvp(jax.grad(Q_hat), (jnp.zeros_like(x),), (eta_h,))
                return _project_flat(plan, x, hv)

            return hvp

        max_steps = int(self.max_steps)

        def _solve(theta0: jnp.ndarray, args_in: Any) -> dict:
            x0 = theta0
            g0 = horiz_grad(x0, args_in)
            gnorm0 = riem_norm(x0, g0)
            f0 = Q_data(x0, args_in)
            r0 = residual_fn(x0, args_in) if two_arg else residual_fn(x0)
            rnorm0 = jnp.sqrt(jnp.sum(r0 * r0))

            # Per-step trace buffers (fixed length max_steps for jit).
            trace_keys = (
                "Delta",
                "rho",
                "rhonum",
                "rhoden",
                "rho_reg",
                "grad_norm",
                "conv_threshold",
                "accepted",
                "stop_code",
                "proposed_full_rank",
            )
            init_trace = {
                k: jnp.zeros((max_steps,), dtype=jnp.float64) for k in trace_keys
            }
            # The Delta trace must stay strictly positive on EVERY slot (the
            # NaN-safety / rank-drop guards assert ``all(Delta > 0)`` over the
            # full buffer). Unwritten tail slots therefore carry the positive
            # init radius, not 0.
            init_trace["Delta"] = jnp.full((max_steps,), Delta0, dtype=jnp.float64)
            init_trace["proposed_full_rank"] = jnp.ones((max_steps,), dtype=jnp.float64)

            def cond_fun(carry: Any) -> Any:
                _x, _g, _gn, _f, _rn, Delta, step, done, _nn, _tr, _stall = carry
                del _x, _g, _gn, _f, _rn, Delta, _nn, _tr
                return jnp.logical_and(step < max_steps, jnp.logical_not(done))

            def body_fun(carry: Any) -> Any:
                x, g, gnorm, fx, rnorm, Delta, step, done, n_negc, trace, stall = carry
                del done

                hvp = hvp_at(x, args_in)
                # tCG inner solve (eager-jit; static loop bound max_tcg).
                eta, Heta, tinfo = _tcg_core(
                    hvp,
                    g,
                    plan,
                    x,
                    Delta,
                    theta=theta_p,
                    kappa=kappa,
                    min_inner=min_inner,
                    max_tcg_steps=max_tcg,
                    ambient_metric=True,
                )
                stop_code = jnp.asarray(tinfo["stop_code"])
                is_negc = jnp.asarray(tinfo["stop_code"] == _STOP_CODE[_NEG_CURV])
                is_boundary = jnp.logical_or(
                    is_negc, jnp.asarray(tinfo["stop_code"] == _STOP_CODE[_EXCEEDED_TR])
                )

                x_prop = _retract_flat(plan, x, eta)
                # Rank-drop sentinel on the PROPOSED iterate.
                min_ev = _min_eig_YtY(plan, x_prop)
                rank_ok = jnp.logical_and(jnp.isfinite(min_ev), min_ev > 1e-12)
                prop_finite = jnp.all(jnp.isfinite(x_prop))
                prop_valid = jnp.logical_and(rank_ok, prop_finite)

                fx_prop = jnp.where(
                    prop_valid,
                    Q_data(jnp.where(prop_valid, x_prop, x), args_in),
                    jnp.inf,
                )

                rhonum = fx - fx_prop
                # Model denominator in the SAME (ambient) metric the tCG used,
                # so rho is a faithful actual/predicted ratio (pymanopt uses
                # manifold.inner_product, which here is the ambient choice).
                rhoden = -jnp.sum(g * eta) - 0.5 * jnp.sum(eta * Heta)
                rho_reg = (
                    jnp.maximum(1.0, jnp.abs(fx)) * jnp.spacing(1.0) * rho_reg_coef
                )
                rhonum_r = rhonum + rho_reg
                rhoden_r = rhoden + rho_reg

                model_decreased = rhoden_r >= 0.0
                # Pure-JAX rho with 0/0 NaN-safety -> force a radius decrease.
                rho_raw = jnp.where(
                    jnp.abs(rhoden_r) > 0.0,
                    rhonum_r / jnp.where(jnp.abs(rhoden_r) > 0.0, rhoden_r, 1.0),
                    jnp.nan,
                )
                rho_nan = jnp.isnan(rho_raw)
                # A rank-dropping proposal is treated as an rho failure.
                bad_model = jnp.logical_or(jnp.logical_not(model_decreased), rho_nan)
                bad_model = jnp.logical_or(bad_model, jnp.logical_not(prop_valid))

                # Radius update (pymanopt heuristics, NaN-safe).
                shrink = jnp.logical_or(rho_raw < 0.25, bad_model)
                expand = jnp.logical_and(
                    jnp.logical_and(rho_raw > 0.75, is_boundary),
                    jnp.logical_not(bad_model),
                )
                Delta_shrunk = Delta / 4.0
                Delta_expand = jnp.minimum(2.0 * Delta, Delta_bar)
                Delta_new = jnp.where(
                    shrink, Delta_shrunk, jnp.where(expand, Delta_expand, Delta)
                )
                Delta_new = jnp.where(jnp.isfinite(Delta_new), Delta_new, Delta_shrunk)
                Delta_new = jnp.maximum(Delta_new, 1e-300)

                accept = jnp.logical_and(
                    jnp.logical_not(bad_model), rho_raw > rho_prime
                )
                accept = jnp.logical_and(accept, prop_valid)

                x_new = jnp.where(accept, x_prop, x)
                g_new = horiz_grad(x_new, args_in)
                gnorm_new = riem_norm(x_new, g_new)
                fx_new = jnp.where(accept, fx_prop, fx)
                r_new = residual_fn(x_new, args_in) if two_arg else residual_fn(x_new)
                rnorm_new = jnp.sqrt(jnp.sum(r_new * r_new))

                conv_thresh = self.atol + self.rtol * rnorm_new
                # #152 parity: match pymanopt's stopping rule -- certify on a
                # pure (horizontal) gradient-norm floor
                # (pymanopt._check_stopping_criterion stops when
                # ||grad|| < min_gradient_norm). pymanopt has NO boundary-
                # progress hold-off, so we drop the emu-gmm-specific one that
                # could defer 'converged' on a boundary-riding trajectory all
                # the way to max_steps (the source of the non-convex meta-gate's
                # converged=False). The all-Euclidean linear case that rides the
                # trust boundary every step still certifies here exactly as
                # pymanopt does, the moment the gradient norm settles.
                grad_converged = gnorm_new < conv_thresh
                # #152/#156 ftol (cost-stagnation): an ACCEPTED step whose
                # relative cost reduction is below ftol is "stagnant" -- the
                # iterate is at the achievable cost floor. After ftol_patience
                # consecutive such steps, certify on cost stationarity. This
                # catches what grad_converged misses under continuously-updated
                # weighting, where the criterion gradient has an empirical noise
                # floor (V_X estimation + the d/dtheta W term) sitting above
                # atol+rtol*||r||, so the horizontal gradient norm never reaches
                # the floor (RTR reaches the right optimum but never certifies).
                # A REJECTED step resets the counter (the solver is still
                # exploring). GATED on gauge structure (ftol_active): a
                # gauge-free / scalar tree keeps its exact grad-norm stopping
                # point; and since the cost is gauge-invariant, a fired ftol stop
                # is identical along the O(k) fibre -- so, unlike a step-norm
                # stop, it never perturbs the gauge-equivariant step count.
                # Mirrors riemannian_lm's #156 ftol stop.
                actual_reduction = fx - fx_prop
                stagnant_accepted = jnp.logical_and(
                    accept, actual_reduction < ftol * fx
                )
                stall_next = jnp.where(stagnant_accepted, stall + 1, 0)
                # RTR-specific companion to ftol: at the CU noise floor RTR
                # OSCILLATES accept(rho~1, expand)/reject(rho<0 at the larger
                # radius, shrink), so the trust radius collapses toward ~1e-16
                # while the *consecutive-accepted* ftol counter keeps resetting
                # on the rejects and never reaches patience (whereas LM's
                # monotone-accepted collapse lets ftol fire -- #156). The radius
                # collapse is monotone through the oscillation, so certify when
                # Delta falls below _MIN_RADIUS: the solver is at the achievable
                # floor and cannot make further progress. Both criteria are gated
                # on gauge structure (scalar boundary keeps its grad-norm stop)
                # and are gauge-safe (cost and Delta are gauge-invariant).
                radius_collapsed = Delta_new < _MIN_RADIUS
                converged = grad_converged
                if ftol_active:
                    converged = jnp.logical_or(
                        converged,
                        jnp.logical_or(stall_next >= ftol_patience, radius_collapsed),
                    )

                n_negc_new = n_negc + jnp.where(is_negc, 1, 0)

                # Record the trace at index ``step``.
                idx = step
                upd = {
                    "Delta": Delta,
                    "rho": jnp.where(rho_nan, jnp.nan, rho_raw),
                    "rhonum": rhonum,
                    "rhoden": rhoden,
                    "rho_reg": rho_reg,
                    "grad_norm": gnorm,
                    "conv_threshold": conv_thresh,
                    "accepted": jnp.where(accept, 1.0, 0.0),
                    "stop_code": jnp.asarray(stop_code, dtype=jnp.float64),
                    "proposed_full_rank": jnp.where(prop_valid, 1.0, 0.0),
                }
                trace = {
                    k: trace[k].at[idx].set(jnp.asarray(upd[k], dtype=jnp.float64))
                    for k in trace
                }

                return (
                    x_new,
                    g_new,
                    gnorm_new,
                    fx_new,
                    rnorm_new,
                    Delta_new,
                    step + 1,
                    converged,
                    n_negc_new,
                    trace,
                    stall_next,
                )

            init_carry = (
                x0,
                g0,
                gnorm0,
                f0,
                rnorm0,
                jnp.asarray(Delta0),
                jnp.asarray(0),
                jnp.asarray(False),
                jnp.asarray(0),
                init_trace,
                jnp.asarray(0),
            )
            (
                x_final,
                g_final,
                gnorm_final,
                f_final,
                _rn,
                _Delta,
                steps,
                done,
                n_negc,
                trace,
                _stall,
            ) = jax.lax.while_loop(cond_fun, body_fun, init_carry)

            return {
                "x": x_final,
                "steps": steps,
                "done": done,
                "final_objective": f_final,
                "grad_norm": gnorm_final,
                "n_negc": n_negc,
                "trace": trace,
            }

        if not two_arg:
            out = _solve(theta_flat, None)
        else:
            solve_jit = _TRACED_SOLVE_CACHE.get_or_build(
                residual_fn, lambda: jax.jit(_solve), key=(self, manifold_spec)
            )
            out = solve_jit(theta_flat, args)

        x_final = out["x"]
        steps = out["steps"]
        done = out["done"]
        final_objective = out["final_objective"]
        gnorm_final = out["grad_norm"]
        n_negc = out["n_negc"]
        trace = out["trace"]

        try:
            done_concrete = bool(done)
            status = "converged" if done_concrete else "max_iterations"
        except (jax.errors.TracerBoolConversionError, TypeError):
            status = "traced"

        theta_hat = params_mod.unflatten_params(
            x_final, treedef, manifold_spec=manifold_spec
        )
        extra = {
            "steps": steps,
            "Delta": trace["Delta"],
            "rho": trace["rho"],
            "rhonum": trace["rhonum"],
            "rhoden": trace["rhoden"],
            "rho_reg": trace["rho_reg"],
            "grad_norm": trace["grad_norm"],
            "conv_threshold": trace["conv_threshold"],
            "accepted": trace["accepted"],
            "stop_code": trace["stop_code"],
            "proposed_full_rank": trace["proposed_full_rank"],
            "status": status,
            "final_objective": final_objective,
            "done": jnp.asarray(done),
        }
        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="emu_gmm.riemannian_tr",
            done=jnp.asarray(done),
            final_gradient_norm=gnorm_final,
            max_tcg_steps=jnp.asarray(max_tcg),
            max_radius=jnp.asarray(Delta_bar),
            n_negcurv=n_negc,
            tr_trace=_TRTrace(trace, n_negc),
            extra=extra,
        )
        return theta_hat, info


def riemannian_tr(
    *,
    max_steps: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    rho_prime: float = _RHO_PRIME,
    kappa: float = 0.1,
    theta: float = 1.0,
    min_inner: int = 1,
    max_tcg_steps: int | None = None,
    max_radius: float | None = None,
    init_radius: float | None = None,
    gauge_floor: float = _GAUGE_FLOOR,
    ftol: float = _FTOL,
    ftol_patience: int = _FTOL_PATIENCE,
) -> _RiemannianTR:
    """Build a JAX-native Riemannian Trust Region optimiser (#152).

    A second :class:`~emu_gmm.manifolds.optimizer.RiemannianOptimizer`, a
    drop-in alternative to :func:`riemannian_lm` that follows negative
    curvature via a truncated-CG trust-region subproblem on the true
    retraction-pullback Hessian. ``max_tcg_steps`` / ``max_radius`` default to
    the **intrinsic** quotient dimension (``total_dimension - total_gauge_dim``)
    rather than the ambient ``nk``.
    """
    return _RiemannianTR(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        rho_prime=rho_prime,
        kappa=kappa,
        theta=theta,
        min_inner=min_inner,
        max_tcg_steps=max_tcg_steps,
        max_radius=max_radius,
        init_radius=init_radius,
        gauge_floor=gauge_floor,
        ftol=ftol,
        ftol_patience=ftol_patience,
    )


__all__ = ["riemannian_tr", "_riemannian_hvp", "_truncated_cg"]
