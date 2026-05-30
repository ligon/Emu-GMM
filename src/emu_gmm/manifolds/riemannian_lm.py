"""Minimal Riemannian Gauss--Newton / Levenberg--Marquardt solver.

Covers the native scalar leaves needed by the Phase-4 lite slice:
:class:`~emu_gmm.manifolds.euclidean.Euclidean` and
:class:`~emu_gmm.manifolds.positive.Positive`. No Lyapunov solve, no
``Product`` machinery (a ``Product`` of native scalar leaves still flows,
since the step works coordinate-wise on the flat ambient vector).

JIT-purity (plan §6 "JIT when all factors native"): every leaf here is
native, so the whole step is ``jax``-traceable. The LM iteration uses
:func:`jax.lax.while_loop` with a ``jnp.where`` / :func:`jax.lax.cond`
accept--reject so no Python-side control flow leaks into the trace.

Metric vs Jacobian-scaling (load-bearing; see the module spec):

* The Gauss--Newton *step* uses the retraction differential
  :math:`dx/dv|_{v=0}`. For :class:`Positive`, :math:`R_x(v) = x e^{v/x}`
  gives :math:`dx/dv|_0 = x`; for :class:`Euclidean`, :math:`1`. So the
  scaled Jacobian column is ``J_amb[:, j] * x_j`` (resp. unchanged). For
  the 1-D affine-invariant metric this scaled-Jacobian GN step is exactly
  the metric-correct natural step --- no separate Gram matrix needed.
* The *inference* information matrix (in the estimator) uses
  ``euclidean_to_riemannian_gradient = x**2 * egrad``. These two scalings
  are deliberately distinct; conflating them yields a wrong step or a
  wrong ``Sigma_theta``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from emu_gmm._internal import params as params_mod
from emu_gmm.manifolds.positive import Positive
from emu_gmm.manifolds.spec import ManifoldSpec

if TYPE_CHECKING:
    from emu_gmm.types import OptimizerInfo


def _is_positive(manifold: Any) -> bool:
    return isinstance(manifold, Positive)


@dataclasses.dataclass(frozen=True)
class _RiemannianLM:
    """Callable Riemannian LM solver (scalar Euclidean + Positive leaves).

    Satisfies the :class:`~emu_gmm.manifolds.optimizer.RiemannianOptimizer`
    protocol: ``__call__(residual_fn, theta_init, manifold_spec)``.
    """

    rtol: float
    atol: float
    max_steps: int

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Any,
        manifold_spec: ManifoldSpec,
    ) -> tuple[Any, "OptimizerInfo"]:
        # Local import to avoid an import cycle: emu_gmm.types ->
        # _internal.params -> manifolds.euclidean -> manifolds.__init__
        # -> riemannian_lm. By call time emu_gmm.types is fully loaded.
        from emu_gmm.types import OptimizerInfo

        theta_flat, treedef = params_mod.flatten_params(theta_init)
        K = int(theta_flat.shape[0])

        # Per-coordinate flag: positive (exp retraction, dx/dv = x) vs
        # euclidean (additive retraction, dx/dv = 1). All leaves are
        # scalar so offset == leaf index.
        is_pos = jnp.asarray(
            [_is_positive(ls.manifold) for ls in manifold_spec.leaf_specs],
            dtype=bool,
        )
        # Defensive: if no spec (shouldn't happen on this path) treat all
        # as Euclidean.
        if is_pos.shape[0] != K:
            is_pos = jnp.zeros((K,), dtype=bool)

        def step_scale(x: Float[Array, " K"]) -> Float[Array, " K"]:
            """``dx/dv|_0`` per coordinate: x for Positive, 1 for Euclidean."""
            return jnp.where(is_pos, x, jnp.ones_like(x))

        def retract(x: Float[Array, " K"], d: Float[Array, " K"]) -> Float[Array, " K"]:
            """Per-leaf retraction of the tangent step ``d`` at ``x``."""
            pos = x * jnp.exp(d / x)
            euc = x + d
            return jnp.where(is_pos, pos, euc)

        def metric_diag(x: Float[Array, " K"]) -> Float[Array, " K"]:
            """Diagonal metric ``g_jj``: 1/x^2 for Positive, 1 for Euclidean."""
            return jnp.where(is_pos, 1.0 / (x**2), jnp.ones_like(x))

        def riem_grad_norm(
            x: Float[Array, " K"], g_tan: Float[Array, " K"]
        ) -> Float[Array, ""]:
            """g-norm of the tangent-coordinate gradient ``g_tan``.

            ``sum_j g_jj * g_tan_j^2`` then sqrt. For Positive this is the
            affine-invariant metric ``g_tan^2 / x^2``.
            """
            return jnp.sqrt(jnp.sum(metric_diag(x) * g_tan * g_tan))

        def residuals(x: Float[Array, " K"]) -> Float[Array, " M"]:
            return residual_fn(x)

        # Initial Jacobian + lambda anchor.
        r0 = residuals(theta_flat)
        J0 = jax.jacfwd(residuals)(theta_flat)  # (M, K)
        Jr0 = J0 * step_scale(theta_flat)[None, :]
        lam0 = 1e-3 * jnp.sqrt(jnp.sum((Jr0.T @ Jr0) ** 2))
        lam0 = jnp.maximum(lam0, 1e-12)

        eye = jnp.eye(K)

        def cond_fun(carry: Any) -> Any:
            x, lam, step, done = carry
            del x, lam
            return jnp.logical_and(step < self.max_steps, jnp.logical_not(done))

        def body_fun(carry: Any) -> Any:
            x, lam, step, done = carry
            r = residuals(x)
            cost = jnp.sum(r * r)
            J = jax.jacfwd(residuals)(x)  # (M, K)
            Jr = J * step_scale(x)[None, :]  # tangent-coord Jacobian
            g_tan = Jr.T @ r  # (K,) tangent gradient
            A = Jr.T @ Jr  # (K, K) Gram

            # LM solve: (A + lam*I) d = -g_tan.
            d = jnp.linalg.solve(A + lam * eye, -g_tan)
            x_new = retract(x, d)
            r_new = residuals(x_new)
            cost_new = jnp.sum(r_new * r_new)

            improved = cost_new < cost
            x_out = jnp.where(improved, x_new, x)
            lam_out = jnp.where(improved, jnp.maximum(1e-12, 0.5 * lam), 2.0 * lam)

            # Convergence (two complementary criteria, both metric-aware):
            #
            # 1. Riemannian gradient norm small relative to the residual:
            #    ||g||_g < atol + rtol * ||r|| (the natural first-order
            #    stationarity test). This dominates far from the optimum.
            # 2. Accepted step small relative to the point in the g-metric:
            #    ||d||_g < atol + rtol * ||x||_g. Near the optimum the
            #    whitened residual has an O(1e-6) noise floor (CLAUDE.md
            #    commitment 7), so a pure gradient test never certifies; a
            #    vanishing step does --- matching optimistix LM's
            #    step-based certification.
            r_acc = jnp.where(improved, r_new, r)
            r_norm = jnp.sqrt(jnp.sum(r_acc * r_acc))
            g_norm = riem_grad_norm(x_out, g_tan)
            grad_ok = g_norm < (self.atol + self.rtol * r_norm)

            step_norm = riem_grad_norm(x, d)  # ||d||_g at the base point
            x_norm = riem_grad_norm(x, x)
            step_ok = jnp.logical_and(
                improved, step_norm < (self.atol + self.rtol * x_norm)
            )

            converged = jnp.logical_or(grad_ok, step_ok)

            return (x_out, lam_out, step + 1, converged)

        init_carry = (theta_flat, lam0, jnp.asarray(0), jnp.asarray(False))
        x_final, _lam, steps, done = jax.lax.while_loop(cond_fun, body_fun, init_carry)

        r_final = residuals(x_final)
        final_objective = 0.5 * jnp.sum(r_final * r_final)

        # Status: under jit these are traced; eagerly they are concrete.
        try:
            done_concrete = bool(done)
            status = "converged" if done_concrete else "max_iterations"
        except (jax.errors.TracerBoolConversionError, TypeError):
            status = "traced"

        del r0  # only used to seed the jacfwd trace shape
        theta_hat = params_mod.unflatten_params(x_final, treedef)
        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="riemannian_lm",
        )
        return theta_hat, info


def riemannian_lm(
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 200,
) -> _RiemannianLM:
    """Build a minimal Riemannian Levenberg--Marquardt optimiser.

    Covers scalar :class:`~emu_gmm.manifolds.euclidean.Euclidean` and
    :class:`~emu_gmm.manifolds.positive.Positive` leaves. Defaults match
    :func:`emu_gmm.optimizer.optimistix_lm`.
    """
    return _RiemannianLM(rtol=rtol, atol=atol, max_steps=max_steps)


__all__ = ["riemannian_lm"]
