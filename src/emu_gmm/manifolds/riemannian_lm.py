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
component is removed before the norm is taken.

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from emu_gmm._internal import params as params_mod
from emu_gmm._internal.fn_cache import FunctionKeyedCache
from emu_gmm.manifolds.positive import Positive
from emu_gmm.manifolds.spec import LeafSpec, ManifoldSpec

if TYPE_CHECKING:
    from emu_gmm.types import OptimizerInfo


def _is_positive(manifold: Any) -> bool:
    return isinstance(manifold, Positive)


# Default gauge lambda-floor coefficient. Multiplied by ``total_gauge_dim``
# to form ``lam_floor`` (Phase-3 contract item 2). It is a conservative
# additive floor on the damped Gram; the horizontal-Jacobian projection
# (not this floor) is the primary mechanism that removes the gauge
# component of the step (red-team R11). For ``total_gauge_dim == 0`` the
# floor is exactly 0 and the scalar path is bitwise unchanged.
_GAUGE_FLOOR: float = 1e-6

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
                    # Scalar leaf: projection is identity for both
                    # Euclidean and Positive; keep the 0-d-as-(1,) layout.
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
                    # Native scalar leaf. Reproduce the original
                    # coordinate-wise retraction exactly: Positive
                    # x*exp(d/x), Euclidean x+d (red-team R18/R23).
                    p0 = pt[0]
                    d0 = dd[0]
                    if _is_positive(manifold):
                        new = p0 * jnp.exp(d0 / p0)
                    else:
                        new = p0 + d0
                    parts.append(jnp.reshape(new, (1,)))
                    continue
                pt_m = jnp.reshape(pt, shape)
                dd_m = jnp.reshape(dd, shape)
                new_m = manifold.retraction(pt_m, dd_m)
                parts.append(jnp.reshape(new_m, (size,)))
            return jnp.concatenate(parts)

        # ----------------------------------------------------------------
        # Diagonal metric for the convergence norm. Per-coordinate
        # 1/x^2 for a scalar Positive leaf, 1 otherwise (PSDFixedRank uses
        # the embedded Frobenius metric == identity in ambient coords).
        # Built as a flat (K,) static-structured weight (red-team R29).
        # ----------------------------------------------------------------
        def metric_diag(x: Float[Array, " K"]) -> Float[Array, " K"]:
            parts = []
            for offset, size, shape, manifold in plan:
                pt = _block(x, offset, size)
                if shape == () and _is_positive(manifold):
                    parts.append(jnp.reshape(1.0 / (pt[0] ** 2), (1,)))
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
                x, lam, step, done = carry
                del x, lam
                return jnp.logical_and(step < self.max_steps, jnp.logical_not(done))

            def body_fun(carry: Any) -> Any:
                x, lam, step, done = carry
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

                converged = jnp.logical_or(grad_ok, step_ok)
                return (x_out, lam_out, step + 1, converged)

            init_carry = (theta0_flat, lam0, jnp.asarray(0), jnp.asarray(False))
            x_final, _lam, steps, done = jax.lax.while_loop(
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
        # #78: propagate the REAL done flag (traced bool) out so the
        # estimator reports genuine convergence, not the always-True
        # status=="traced" collapse (red-team R6/R10/R16/R28).
        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="riemannian_lm",
            done=jnp.asarray(done),
        )
        return theta_hat, info


def riemannian_lm(
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 200,
    gauge_floor: float = _GAUGE_FLOOR,
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
    """
    return _RiemannianLM(
        rtol=rtol, atol=atol, max_steps=max_steps, gauge_floor=gauge_floor
    )


__all__ = ["riemannian_lm"]
