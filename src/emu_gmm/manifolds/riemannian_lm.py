r"""Per-leaf Riemannian Gauss--Newton / Levenberg--Marquardt solver.

Phase 3 of the manifold epic (#12). Covers a mixed ``Product`` of native
leaves --- :class:`~emu_gmm.manifolds.euclidean.Euclidean`,
:class:`~emu_gmm.manifolds.positive.Positive` --- and the quotient
:class:`~emu_gmm.manifolds.psd_fixed_rank.PSDFixedRank` leaf. The step is
assembled **per :class:`~emu_gmm.manifolds.spec.LeafSpec`**: each leaf's
ambient block is sliced out of the flat tangent vector / Jacobian, reshaped
to the leaf's ``ambient_shape``, and dispatched to the leaf manifold's own
``retraction`` and (for the gauge-bearing PSD leaf) horizontal ``projection``.

Scalar non-regression (Phase-3 contract item 4)
------------------------------------------------
For an all-scalar tree (every leaf ``ambient_shape == ()``) the
``total_gauge_dim`` is ``0``, so the gauge lambda-floor contributes exactly
nothing (``gauge_floor * 0 == 0``) and the per-leaf projection is the
identity (Euclidean / Positive ``projection`` returns the ambient vector
unchanged). The per-leaf loop then reduces to the original coordinate-wise
retraction: ``x e^{v/x}`` for Positive, ``x + v`` for Euclidean. The scalar
``Positive(1,1)`` slice (``test_estimator_positive``) and the v1 estimator
suite are preserved.

Gauge correctness (Phase-3 contract items 1--2; red-team R2/R9/R11/R22)
-----------------------------------------------------------------------
For a :class:`PSDFixedRank` leaf the iterate must not wander the
:math:`O(k)` gauge fibre (the ``k(k-1)/2`` skew-symmetric directions along
which ``Y -> Y Q`` leaves ``Y Y^T`` fixed). Two mechanisms enforce this:

1. **Horizontal Jacobian.** Each leaf's Jacobian column block is projected
   row-by-row through the leaf manifold's horizontal ``projection`` (a
   Lyapunov solve for PSD; identity for Euclidean/Positive). The Gram
   ``A = J_h' J_h`` and tangent gradient ``g = J_h' r`` then live in the
   horizontal subspace, so the LM solve ``(A + lam I) d = -g`` returns a
   step ``d`` with **zero** gauge component (the only vertical contribution
   would come from ``lam I`` acting on a vertical ``g``, but ``g`` is
   horizontal by construction). ``d`` is re-projected once more as a
   numerical belt-and-suspenders before retraction. Because the embedded
   PSD metric coincides with the ambient Frobenius metric
   (``psd_fixed_rank.py`` ``inner_product``), the plain ambient Gram solve
   is the correct horizontal Gauss--Newton step --- no metric reweighting
   needed.
2. **Gauge lambda-floor.** ``lam_floor = gauge_floor * total_gauge_dim``
   keeps the damped Gram numerically PD even where the horizontal Jacobian
   is rank-deficient, so the solve cannot blow up to inf/nan along a
   marginally-singular gauge-adjacent direction. For ``total_gauge_dim == 0``
   it is exactly ``0`` (bitwise scalar path).

Convergence norms are computed on the **horizontal** step / gradient, so a
gauge-wandering iterate cannot certify falsely (red-team R9): the gauge
component is removed before the norm is taken. A third, ftol (cost-stagnation)
criterion (#156), **active only for a gauge-bearing tree
(``total_gauge_dim > 0``)**, certifies after ``ftol_patience`` consecutive
**accepted** steps whose relative cost reduction is below ``ftol`` -- the
standard MINPACK / scipy termination for an objective whose Gauss--Newton model
cannot drive the gradient to zero (the continuously-updated + clustered
criterion on the over-parameterised PSDFixedRank factor: the cost basin is
reached but the horizontal gradient plateaus while the iterate drifts at
constant cost). A *rejected* step resets the counter (the LM is still
exploring). For a gauge-free tree the criterion is gated off, so every scalar /
Euclidean / Positive solve -- including the ill-conditioned ``sigma -> 0``
boundary collapse -- keeps its exact stopping point. The cost ``||r||^2`` is
itself gauge-invariant.

#78 done-flag (Phase-3 contract item 3; red-team R6/R10/R16)
------------------------------------------------------------
The convergence ``done`` flag is a traced bool living in the
:func:`jax.lax.while_loop` carry. It is now propagated out via a new
``done`` field on :class:`~emu_gmm.types.OptimizerInfo` (traced, default
``None`` for backward compatibility), so the estimator can report **real**
convergence (``done and steps < max_steps``) instead of collapsing the
under-jit ``status == "traced"`` to always-converged.

Covariance / Jacobian convention (matches ``../ManifoldGMM``, Convention B):
every first-order retraction has unit differential at ``v = 0``
(``DR_x(0) = Id``): Positive's ``R_x(v) = x e^{v/x}`` gives ``dx/dv|_0 = 1``,
same as Euclidean's ``x + v`` and PSDFixedRank's ``Y + V``. So the
Gauss--Newton Jacobian in tangent coordinates is the *ambient* Jacobian
(``step_scale == 1``).

REVERSE-MODE AD IS NOT SUPPORTED THROUGH THIS SOLVER (#77)
----------------------------------------------------------
The solve loop is a :func:`jax.lax.while_loop`, which has **no reverse
differentiation rule**: ``jax.grad`` / ``jax.jacrev`` of any function
that calls this solver (e.g. hyperparameter gradients of
``theta_hat(data)``, differentiable-pipeline embeddings) raises JAX's
``Reverse-mode differentiation does not work for lax.while_loop``
error. This is a recorded deferral (re-deferred to v2.1, 2026-06-10),
not an oversight: nothing in the package differentiates THROUGH the
solver --- all post-fit inference uses the direct-form information
matrix :math:`G'\Lambda G` at the fixed optimum (CLAUDE.md commitment
5) and forward-mode :func:`jax.jacfwd` for moment Jacobians. Callers
needing solver-through gradients should use implicit differentiation
at the optimum (the stationarity conditions), not unrolled reverse
mode; that machinery is the #77 work item.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from emu_gmm._internal import params as params_mod
from emu_gmm._internal.fn_cache import FunctionKeyedCache
from emu_gmm.manifolds.euclidean import Euclidean
from emu_gmm.manifolds.positive import Positive
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec

if TYPE_CHECKING:
    from emu_gmm.types import OptimizerInfo


def _is_positive(manifold: Any) -> bool:
    return isinstance(manifold, Positive)


def _is_euclidean(manifold: Any) -> bool:
    return isinstance(manifold, Euclidean)


# Default gauge lambda-floor coefficient. Multiplied by ``total_gauge_dim``
# to form ``lam_floor`` (Phase-3 contract item 2). It is a conservative
# additive floor on the damped Gram; the horizontal-Jacobian projection
# (not this floor) is the primary mechanism that removes the gauge
# component of the step (red-team R11). For ``total_gauge_dim == 0`` the
# floor is exactly 0 and the scalar path is bitwise unchanged.
_GAUGE_FLOOR: float = 1e-6

# #152 advisory: an eigenvalue of the horizontal true Hessian is treated as
# genuine negative curvature only when it is below
# ``-(_CURV_ATOL + _CURV_RTOL * |lambda_max|)``. The relative term keeps the
# ``k(k-1)/2`` gauge directions (~eps * ||H|| under the projected HVP) and
# benign round-off from being mislabelled a saddle.
_CURV_RTOL: float = 1e-6
_CURV_ATOL: float = 1e-9


def _min_horizontal_curvature(
    residuals: Callable[[Float[Array, " K"]], Float[Array, " M"]],
    manifold_spec: ManifoldSpec,
    x_flat: Float[Array, " K"],
) -> tuple[float, float]:
    """Smallest / largest eigenvalue of the projected horizontal true Hessian.

    Assembles the ``(K, K)`` retraction-pullback Hessian of
    ``0.5 ||residuals(.)||^2`` at ``x_flat`` by applying RTR's projected
    Riemannian HVP to each ambient basis vector, then returns
    ``(lambda_min, lambda_max)`` of its symmetric part. The HVP is projected to
    the horizontal space, so the ``k(k-1)/2`` gauge directions map to ~0 and do
    NOT masquerade as curvature -- a ``lambda_min`` below
    ``-(_CURV_ATOL + _CURV_RTOL * |lambda_max|)`` is genuine horizontal negative
    curvature (a saddle), the regime ``riemannian_tr`` exists for.

    EAGER-only helper (a handful of HVPs at a single point). Reuses
    ``riemannian_tr._riemannian_hvp`` via a local import -- the TR module does
    not import this one, so there is no cycle.
    """
    from emu_gmm.manifolds.riemannian_tr import _riemannian_hvp

    K = int(x_flat.shape[0])
    eye = jnp.eye(K, dtype=x_flat.dtype)
    cols = [
        _riemannian_hvp(residuals, x_flat, manifold_spec, eye[:, j]) for j in range(K)
    ]
    H = jnp.stack(cols, axis=1)  # column j = H @ e_j
    H = 0.5 * (H + H.T)  # symmetrise numerical residue before eigvalsh
    evals = jnp.linalg.eigvalsh(H)
    return float(evals[0]), float(evals[-1])


# #124 (PR B): memoised jitted solve cores for the traced-``args`` path.
# Keyed per kernel OBJECT with secondary key ``(self, manifold_spec)`` --
# the kernel's identity is the trace key (same pattern as the
# estimator's factory-stable kernels), ``self`` pins the solver
# hyperparameters (frozen dataclass, hashable), and ``manifold_spec``
# pins the per-leaf plan (frozen dataclass hashing by structural
# identity, so a caller that rebuilds an equal spec still hits).
# #139: the jitted solve closes over the kernel, so the original
# id()-keyed module dict could never evict (its weakref.finalize was
# dead code; append-only cache, one immortal entry per (kernel, solver,
# spec) triple). The per-kernel table now rides as an attribute ON the
# kernel itself, so the traces survive exactly as long as the kernel
# does and object-identity keying forecloses the id-recycling stale-hit
# hazard. See ``_internal/fn_cache.py`` for the full design rationale.
_TRACED_SOLVE_CACHE = FunctionKeyedCache("_emu_gmm_traced_solve")


@dataclasses.dataclass(frozen=True)
class _RiemannianLM:
    """Callable Riemannian LM solver (per-leaf retraction / projection).

    Satisfies the :class:`~emu_gmm.manifolds.optimizer.RiemannianOptimizer`
    protocol: ``__call__(residual_fn, theta_init, manifold_spec)``.
    """

    rtol: float
    atol: float
    max_steps: int
    gauge_floor: float = _GAUGE_FLOOR
    ftol: float = 1e-8
    ftol_patience: int = 8
    # #152 advisory: when a gauge-bearing solve converges to a genuine
    # stationary point, probe the horizontal true Hessian at the optimum and --
    # if it is indefinite (a saddle) -- emit a non-convexity warning and set
    # ``OptimizerInfo.stalled_indefinite`` / ``.min_curvature``. EAGER-only (it
    # never fires under the vmapped/replicate MC path). Set ``False`` to silence
    # both the warning and the probe.
    advise_nonconvex: bool = True

    def __call__(
        self,
        residual_fn: Callable[..., Float[Array, " M"]],
        theta_init: Any,
        manifold_spec: ManifoldSpec,
        *,
        args: Any = None,
    ) -> tuple[Any, "OptimizerInfo"]:
        """Solve the Riemannian LM problem.

        With ``args is None`` (the v1/v2 contract, byte-identical to the
        pre-#124 behaviour): ``residual_fn`` is a one-argument closure
        over the flat ambient coordinates, evaluated eagerly with the
        ``lax.while_loop`` traced per call.

        With ``args`` supplied (#124 PR B): ``residual_fn`` is a
        two-argument kernel ``residual_fn(theta_flat, args)`` and
        ``args`` is an arbitrary traced pytree (e.g. a measure). The
        whole solve -- initial Jacobian, lambda anchor and the
        ``while_loop`` -- is compiled ONCE per ``(kernel identity,
        solver hyperparameters, manifold_spec)`` and memoised, so fresh
        same-structure ``args`` (the repeated-estimation case) are new
        leaf values on an existing trace: zero retrace.
        """
        # Local import to avoid an import cycle: emu_gmm.types ->
        # _internal.params -> manifolds.euclidean -> manifolds.__init__
        # -> riemannian_lm. By call time emu_gmm.types is fully loaded.
        from emu_gmm.types import OptimizerInfo

        # Manifold-aware flatten (red-team R20): the 3-tuple path ravels
        # non-scalar leaves (PSDFixedRank (n,k) blocks) into the ambient
        # buffer and returns the matching spec. For an all-scalar tree the
        # flat buffer / treedef are byte-identical to the v1 2-tuple
        # flatten, and the returned spec matches the ``manifold_spec`` the
        # caller passed (Phase-1 contract). We prefer the caller-supplied
        # ``manifold_spec`` for the geometry plan below (the estimator
        # builds it once via ``manifold_spec_from_params``); the locally
        # flattened spec is used only to obtain ``treedef`` for the
        # manifold-aware unflatten.
        theta_flat, treedef, flat_spec = params_mod.flatten_params_with_spec(theta_init)
        K = int(theta_flat.shape[0])
        if manifold_spec is None:
            manifold_spec = flat_spec

        # ----------------------------------------------------------------
        # Static per-leaf plan. Every field here is concrete at trace time
        # (offsets / shapes / manifold instances from the frozen
        # ManifoldSpec), so no traced value ever indexes leaf_specs
        # (red-team R3/R4/R12). For an all-scalar tree the offsets are the
        # leaf indices 0..K-1 and every block has size 1.
        # ----------------------------------------------------------------
        leaf_specs: tuple[LeafSpec, ...] = manifold_spec.leaf_specs
        total_ambient = int(manifold_spec.total_ambient_dim)
        total_gauge = int(manifold_spec.total_gauge_dim)

        # Block-boundary guard (red-team R5/R21): the spec must exactly
        # tile the flat buffer. Phase-1/2 build this; we only assert.
        plan: list[tuple[int, int, tuple[int, ...], Any]] = []
        running = 0
        for ls in leaf_specs:
            size = int(np.prod(ls.ambient_shape)) if ls.ambient_shape != () else 1
            plan.append((ls.offset, size, ls.ambient_shape, ls.manifold))
            running += size
        if not (total_ambient == running == K):
            raise ValueError(
                "riemannian_lm: manifold_spec does not tile the flat "
                f"parameter buffer: total_ambient_dim={total_ambient}, "
                f"sum(block sizes)={running}, len(theta_flat)={K}"
            )

        # Gauge lambda-floor (Phase-3 item 2). Exactly 0 when there is no
        # gauge structure -> scalar/v1 path is bitwise unchanged
        # (red-team R7/R13/R18).
        lam_floor = float(self.gauge_floor) * float(total_gauge)

        # ftol (cost-stagnation) certification is gated on gauge structure
        # (#156). It exists for the over-parameterised PSDFixedRank factor,
        # where the gauge redundancy + the continuously-updated/clustered
        # curvature stall the Gauss--Newton step at the cost floor. A
        # gauge-free leaf (scalar / Euclidean / Positive) has no such
        # redundancy: its solve converges via grad_ok / step_ok, and gating
        # ftol off keeps its stopping point -- and any pinned J / p-value
        # downstream (e.g. the ill-conditioned Positive sigma->0 boundary
        # collapse in test_estimator_realdata) -- bitwise unchanged.
        ftol_active = total_gauge > 0

        def _block(x: Float[Array, " K"], offset: int, size: int) -> Any:
            return x[offset : offset + size]

        # ----------------------------------------------------------------
        # Per-leaf horizontal projection of an ambient FLAT step / gradient.
        # For PSDFixedRank this removes the skew-symmetric O(k) component
        # via the leaf's Lyapunov solve; for Euclidean / Positive the leaf
        # projection is the identity, so this is a no-op on scalar trees
        # (red-team R8/R11/R12/R23).
        # ----------------------------------------------------------------
        def project_flat(
            x: Float[Array, " K"], v_flat: Float[Array, " K"]
        ) -> Float[Array, " K"]:
            parts = []
            for offset, size, shape, manifold in plan:
                pt = _block(x, offset, size)
                vv = _block(v_flat, offset, size)
                if shape == ():
                    # Scalar leaf: the tangent space of a 1-D manifold is all
                    # of R, so the projection is the identity for EVERY scalar
                    # leaf (Euclidean, Positive, Interval, future scalars);
                    # keep the 0-d-as-(1,) layout.
                    parts.append(vv)
                    continue
                pt_m = jnp.reshape(pt, shape)
                vv_m = jnp.reshape(vv, shape)
                proj_m = manifold.projection(pt_m, vv_m)
                parts.append(jnp.reshape(proj_m, (size,)))
            return jnp.concatenate(parts)

        # Horizontal projection of the Jacobian: project each leaf's column
        # block row-by-row (vmap over the M rows). Gives J_h whose row-space
        # is horizontal, so A = J_h' J_h and g = J_h' r carry no gauge
        # component (red-team R2/R9/R22). On scalar trees every leaf
        # projection is identity, so J_h == J bitwise.
        def project_jacobian(
            x: Float[Array, " K"], J: Float[Array, "M K"]
        ) -> Float[Array, "M K"]:
            # vmap project_flat over the M rows of J.
            return jax.vmap(lambda row: project_flat(x, row))(J)

        # ----------------------------------------------------------------
        # Per-leaf retraction of a (horizontal) FLAT step.
        # ----------------------------------------------------------------
        def retract(x: Float[Array, " K"], d: Float[Array, " K"]) -> Float[Array, " K"]:
            parts = []
            for offset, size, shape, manifold in plan:
                pt = _block(x, offset, size)
                dd = _block(d, offset, size)
                if shape == ():
                    # Native scalar leaf. Positive / Euclidean keep the
                    # original inline coordinate-wise retraction bitwise:
                    # Positive x*exp(d/x), Euclidean x+d (red-team R18/R23).
                    # Any OTHER scalar manifold (Interval today, future
                    # scalar geometries tomorrow) supplies its OWN
                    # retraction -- an additive fallback would defeat e.g.
                    # Interval's bound preservation.
                    p0 = pt[0]
                    d0 = dd[0]
                    if _is_positive(manifold):
                        new = p0 * jnp.exp(d0 / p0)
                    elif _is_euclidean(manifold):
                        new = p0 + d0
                    else:
                        new = manifold.retraction(p0, d0)
                    parts.append(jnp.reshape(new, (1,)))
                    continue
                pt_m = jnp.reshape(pt, shape)
                dd_m = jnp.reshape(dd, shape)
                new_m = manifold.retraction(pt_m, dd_m)
                parts.append(jnp.reshape(new_m, (size,)))
            return jnp.concatenate(parts)

        # ----------------------------------------------------------------
        # Diagonal metric for the convergence norm. Per-coordinate
        # 1/x^2 for a scalar Positive leaf; the leaf's own scalar metric
        # weight g_x(1, 1) for any other non-Euclidean scalar leaf (e.g.
        # phi'(x)^2 for Interval); 1 otherwise (PSDFixedRank uses the
        # embedded Frobenius metric == identity in ambient coords).
        # Built as a flat (K,) static-structured weight (red-team R29).
        # ----------------------------------------------------------------
        def metric_diag(x: Float[Array, " K"]) -> Float[Array, " K"]:
            parts = []
            for offset, size, shape, manifold in plan:
                pt = _block(x, offset, size)
                if shape == () and _is_positive(manifold):
                    parts.append(jnp.reshape(1.0 / (pt[0] ** 2), (1,)))
                elif shape == () and not _is_euclidean(manifold):
                    # Generic scalar leaf (e.g. Interval): its metric weight
                    # enters the convergence norm exactly as Positive's
                    # 1/x^2 does.
                    one = jnp.ones((), dtype=x.dtype)
                    parts.append(
                        jnp.reshape(manifold.inner_product(pt[0], one, one), (1,))
                    )
                else:
                    parts.append(jnp.ones((size,), dtype=x.dtype))
            return jnp.concatenate(parts)

        def riem_norm(x: Float[Array, " K"], v: Float[Array, " K"]) -> Float[Array, ""]:
            return jnp.sqrt(jnp.sum(metric_diag(x) * v * v))

        eye = jnp.eye(K)
        # #124 (PR B): static two-argument dispatch. ``args is None`` is
        # decided eagerly here, so the legacy one-argument contract is
        # byte-identical to the pre-#124 behaviour.
        two_arg = args is not None

        def _solve(
            theta0_flat: Float[Array, " K"], args_in: Any
        ) -> tuple[Any, Any, Any, Any]:
            """Pure solve core in ``(theta0_flat, args_in)``.

            On the legacy path it is called eagerly with ``args_in=None``
            (the empty pytree) -- the exact pre-#124 op sequence. On the
            args path one ``jax.jit(_solve)`` per kernel identity is
            memoised in ``_TRACED_SOLVE_CACHE``, so the residual kernel
            (and the user's psi inside it) traces once and fresh
            same-structure ``args`` ride the compiled solve.
            """

            def residuals(x: Float[Array, " K"]) -> Float[Array, " M"]:
                if two_arg:
                    return residual_fn(x, args_in)
                return residual_fn(x)

            # Initial Jacobian + lambda anchor (horizontal Gram scale).
            r0 = residuals(theta0_flat)
            J0 = jax.jacfwd(residuals)(theta0_flat)  # (M, K)
            Jh0 = project_jacobian(theta0_flat, J0)
            lam0 = 1e-3 * jnp.sqrt(jnp.sum((Jh0.T @ Jh0) ** 2))
            lam0 = jnp.maximum(lam0, 1e-12)
            lam0 = jnp.maximum(lam0, lam_floor)

            def cond_fun(carry: Any) -> Any:
                x, lam, step, done, stall = carry
                del x, lam, stall
                return jnp.logical_and(step < self.max_steps, jnp.logical_not(done))

            def body_fun(carry: Any) -> Any:
                x, lam, step, done, stall = carry
                del done
                r = residuals(x)
                cost = jnp.sum(r * r)
                J = jax.jacfwd(residuals)(x)  # (M, K) ambient
                Jh = project_jacobian(x, J)  # (M, K) horizontal
                g_tan = Jh.T @ r  # (K,) horizontal gradient
                A = Jh.T @ Jh  # (K, K) horizontal Gram

                # LM solve on the horizontal Gram with the gauge floor folded
                # into lam. The step d is horizontal by construction; project
                # once more to clean numerical residue (red-team R11).
                d_raw = jnp.linalg.solve(A + lam * eye, -g_tan)
                d = project_flat(x, d_raw)
                x_new = retract(x, d)
                r_new = residuals(x_new)
                cost_new = jnp.sum(r_new * r_new)

                improved = cost_new < cost
                x_out = jnp.where(improved, x_new, x)
                lam_decreased = jnp.maximum(jnp.maximum(1e-12, lam_floor), 0.5 * lam)
                lam_out = jnp.where(improved, lam_decreased, 2.0 * lam)

                # Convergence (two complementary criteria, both on the
                # HORIZONTAL geometry so a gauge-wandering iterate cannot
                # certify falsely; red-team R9):
                #   1. ||g_h||_g < atol + rtol * ||r||   (stationarity)
                #   2. accepted ||d_h||_g < atol + rtol * ||x||_g (small step)
                r_acc = jnp.where(improved, r_new, r)
                r_norm = jnp.sqrt(jnp.sum(r_acc * r_acc))
                g_norm = riem_norm(x_out, g_tan)
                grad_ok = g_norm < (self.atol + self.rtol * r_norm)

                step_norm = riem_norm(x, d)
                x_norm = riem_norm(x, x)
                step_ok = jnp.logical_and(
                    improved, step_norm < (self.atol + self.rtol * x_norm)
                )

                #   3. ftol (cost stagnation; #156). An **accepted** step
                #      (``improved``) that reduces the cost by less than
                #      ``ftol * cost`` is "stagnant" -- the iterate is moving
                #      but the cost is at its achievable floor. After
                #      ``ftol_patience`` *consecutive* such steps we certify on
                #      cost stationarity (the standard MINPACK / scipy ``ftol``
                #      termination), catching the case ``grad_ok`` / ``step_ok``
                #      miss: an objective whose Gauss--Newton model is imperfect
                #      (the continuously-updated + clustered criterion -- the
                #      cost basin is reached but the GN step can no longer drive
                #      the horizontal gradient to zero, and the iterate drifts
                #      at constant cost).
                #
                #      A **rejected** step does NOT count -- it resets the
                #      counter. A rejection means the LM is still exploring,
                #      ramping ``lam`` to find a descent direction (e.g. the
                #      transient stuck phase of a ``Positive`` solve climbing
                #      off a sub-true start, where the GN step overshoots and is
                #      rejected for a dozen-odd steps before it breaks free).
                #      Counting rejections would falsely certify that stuck
                #      start. Because a healthy solve makes a large relative
                #      reduction on every accepted step until it certifies via
                #      ``grad_ok``, the counter never accumulates and the
                #      iterate / step count of an already-converging solve are
                #      unchanged (the all-scalar / all-Euclidean path included).
                actual_reduction = cost - cost_new
                stagnant_accepted = jnp.logical_and(
                    improved, actual_reduction < self.ftol * cost
                )
                stall_next = jnp.where(stagnant_accepted, stall + 1, 0)

                converged = jnp.logical_or(grad_ok, step_ok)
                if ftol_active:
                    # Gauge-bearing leaf only (#156): OR in cost stagnation.
                    # For a gauge-free tree this branch is not taken, so
                    # ``converged`` is exactly ``grad_ok | step_ok`` and the
                    # stall counter is inert -- the iterate / step count are
                    # bitwise the pre-#156 behaviour.
                    converged = jnp.logical_or(
                        converged, stall_next >= self.ftol_patience
                    )
                return (x_out, lam_out, step + 1, converged, stall_next)

            init_carry = (
                theta0_flat,
                lam0,
                jnp.asarray(0),
                jnp.asarray(False),
                jnp.asarray(0),
            )
            x_final, _lam, steps, done, _stall = jax.lax.while_loop(
                cond_fun, body_fun, init_carry
            )

            r_final = residuals(x_final)
            final_objective = 0.5 * jnp.sum(r_final * r_final)
            del r0  # only used to seed the jacfwd trace shape
            return x_final, steps, done, final_objective

        if not two_arg:
            # Legacy contract: eager call, while_loop traced per call --
            # the exact pre-#124 behaviour.
            x_final, steps, done, final_objective = _solve(theta_flat, None)
        else:
            # Memoised on the kernel object itself (#139): the trace dies
            # with the kernel, and same-(solver, spec) calls on a live
            # kernel reuse one compiled solve.
            solve_jit = _TRACED_SOLVE_CACHE.get_or_build(
                residual_fn, lambda: jax.jit(_solve), key=(self, manifold_spec)
            )
            x_final, steps, done, final_objective = solve_jit(theta_flat, args)

        # Status: under jit these are traced; eagerly (including after the
        # memoised jitted solve, whose outputs are concrete) they are
        # concrete.
        try:
            done_concrete = bool(done)
            status = "converged" if done_concrete else "max_iterations"
        except (jax.errors.TracerBoolConversionError, TypeError):
            status = "traced"
        # Manifold-aware unflatten so non-scalar leaves are reshaped to
        # their ambient_shape (red-team R25). For all-scalar trees this is
        # byte-identical to the v1 unflatten (every ambient_shape == ()).
        theta_hat = params_mod.unflatten_params(
            x_final, treedef, manifold_spec=manifold_spec
        )

        # #152 advisory (EAGER-only): warn when an interactive gauge-bearing
        # solve converges to a genuine STATIONARY POINT whose horizontal true
        # Hessian is indefinite -- a saddle, where riemannian_tr (which follows
        # negative curvature) may do better. Three guards keep it precise:
        #   * ``status == "converged" and not two_arg`` -> only the eager v2 path
        #     (a single ``estimate()`` / interactive fit). The vmapped/replicate
        #     path is ``"traced"`` (warnings cannot fire inside vmap, fields stay
        #     None) and the #124 ``args`` channel is skipped so the user kernel
        #     is not re-traced (trace-sharing preserved).
        #   * a recomputed ``grad_ok`` stationarity test at ``theta_hat``. NOTE
        #     this REVISES the handoff's "fire iff the ftol stall path" trigger:
        #     empirically the #156 ftol (cost-stagnation) certification stops
        #     where the Gauss--Newton model can no longer drive the gradient to
        #     zero, so the iterate DRIFTS at a *large* gradient (||g|| ~ 1e-1) on
        #     a CORRECT estimate. Its Hessian is trivially indefinite but it is
        #     NOT a critical point, so a stall-based probe would warn on every
        #     CU+clustered solve (a false positive). Genuine saddles instead
        #     certify via ``grad_ok`` (||g|| ~ 0). Gating on real stationarity
        #     confines the warning to true saddles.
        # A gauge-free tree (``ftol_active`` False) never probes, so the scalar /
        # Euclidean / Positive path is bitwise unchanged.
        stalled_indefinite = None
        min_curvature = None
        if (
            self.advise_nonconvex
            and ftol_active
            and status == "converged"
            and not two_arg
        ):

            def _residuals_eager(xx: Float[Array, " K"]) -> Float[Array, " M"]:
                # two_arg is False in this branch (gated above): the v2 path
                # passes a one-argument flat residual closure.
                return residual_fn(xx)

            # Genuine-stationarity gate: the exact ``grad_ok`` test the solve
            # loop uses, recomputed at the returned iterate (horizontal gradient).
            r_fin = _residuals_eager(x_final)
            Jh_fin = project_jacobian(x_final, jax.jacfwd(_residuals_eager)(x_final))
            g_norm_fin = riem_norm(x_final, Jh_fin.T @ r_fin)
            r_norm_fin = jnp.sqrt(jnp.sum(r_fin * r_fin))
            is_stationary = bool(g_norm_fin < (self.atol + self.rtol * r_norm_fin))
            if is_stationary:
                lam_min, lam_max = _min_horizontal_curvature(
                    _residuals_eager, manifold_spec, x_final
                )
                min_curvature = lam_min
                neg_tol = _CURV_ATOL + _CURV_RTOL * abs(lam_max)
                stalled_indefinite = lam_min < -neg_tol
                if stalled_indefinite:
                    warnings.warn(
                        "riemannian_lm converged to a stationary point whose "
                        "horizontal true Hessian is indefinite (min eigenvalue "
                        f"~ {lam_min:.3e}, max ~ {lam_max:.3e}); the criterion "
                        "appears non-convex there -- it may be a saddle rather "
                        "than a minimum. Consider re-solving with "
                        "riemannian_tr(), which follows negative curvature.",
                        stacklevel=2,
                    )

        # #78: propagate the REAL done flag (traced bool) out so the
        # estimator reports genuine convergence, not the always-True
        # status=="traced" collapse (red-team R6/R10/R16/R28).
        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="riemannian_lm",
            done=jnp.asarray(done),
            stalled_indefinite=stalled_indefinite,
            min_curvature=min_curvature,
        )
        return theta_hat, info


def riemannian_lm(
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 200,
    gauge_floor: float = _GAUGE_FLOOR,
    ftol: float = 1e-8,
    ftol_patience: int = 8,
    advise_nonconvex: bool = True,
) -> _RiemannianLM:
    """Build a per-leaf Riemannian Levenberg--Marquardt optimiser.

    Covers :class:`~emu_gmm.manifolds.euclidean.Euclidean`,
    :class:`~emu_gmm.manifolds.positive.Positive` and
    :class:`~emu_gmm.manifolds.psd_fixed_rank.PSDFixedRank` leaves in any
    ``Product``. Defaults match :func:`emu_gmm.optimizer.optimistix_lm`.

    Parameters
    ----------
    gauge_floor
        Coefficient of the gauge lambda-floor
        ``lam_floor = gauge_floor * total_gauge_dim``. ``0`` for any
        all-scalar / all-Euclidean / all-Positive tree (no gauge
        structure), so the scalar path is bitwise unchanged.
    ftol, ftol_patience
        Cost-stagnation (MINPACK / scipy ``ftol``) termination (#156), gated
        on gauge structure: the solve certifies after ``ftol_patience``
        *consecutive accepted* steps whose relative cost reduction is below
        ``ftol``. A rejected step resets the counter (the LM is still
        exploring -- ramping ``lam`` to find a descent direction -- not
        stalled). This lets a solve whose Gauss--Newton model cannot drive the
        gradient to zero -- the continuously-updated + clustered criterion on
        the over-parameterised ``PSDFixedRank`` factor, where the cost basin
        is reached but the horizontal gradient plateaus while the iterate
        drifts -- still report ``converged=True`` once the cost is at its
        achievable floor.

        The criterion is **only active when ``total_gauge_dim > 0``**. A
        gauge-free leaf (scalar / ``Euclidean`` / ``Positive``) has no gauge
        redundancy: its solve converges via the gradient / step tests, so ftol
        is gated off and its stopping point -- and any pinned J / p-value
        downstream (e.g. the ill-conditioned ``Positive`` ``sigma -> 0``
        boundary collapse) -- stays **bitwise unchanged**. Even within a
        gauge-bearing solve, a healthy run certifies via the gradient test
        before the counter accumulates, so the iterate and step count of an
        already-converging solve are unchanged.
    advise_nonconvex
        When ``True`` (default), a gauge-bearing solve that converges to a
        genuine stationary point probes the horizontal true Hessian at the
        optimum and, if it is indefinite (a saddle), emits a non-convexity
        warning and sets ``OptimizerInfo.stalled_indefinite`` /
        ``.min_curvature``. The probe is eager-only -- it never fires under the
        vmapped/replicate MC path, where those fields stay ``None``. A
        cost-stagnation (#156 ftol) certification is NOT a stationary point (the
        iterate drifts at a large gradient on a correct estimate), so it does
        not trigger the probe. Set ``False`` to silence the warning and skip the
        probe entirely.
    """
    return _RiemannianLM(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        gauge_floor=gauge_floor,
        ftol=ftol,
        ftol_patience=ftol_patience,
        advise_nonconvex=advise_nonconvex,
    )


__all__ = ["riemannian_lm"]
