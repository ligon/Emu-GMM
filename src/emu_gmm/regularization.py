"""PD-restoration via diagonal Tikhonov regularisation.

Pairwise-overlap variance estimators (and Monte Carlo variances at
small ``n_sim``) can produce ``V`` matrices that are numerically
non-PD --- close to singular or with a wildly large condition number.
The framework's response, per ``docs/design.org`` Section 5, is an
adaptive diagonal-Tikhonov ridge

.. math::
   V^\\star \\;=\\; V \\;+\\; \\tau \\cdot \\operatorname{diag}(V),

with :math:`\\tau \\geq 0` chosen as small as possible subject to
:math:`\\kappa(V^\\star) \\leq \\kappa_{\\mathrm{target}}`.

The :math:`\\tau` search is implemented via bisection over a fixed
number of iterations so the routine remains jit-compatible. The
"anchor-once-then-freeze" policy that pins :math:`\\tau` to the
:math:`\\theta_{\\mathrm{init}}` evaluation lives in
:mod:`emu_gmm.estimator`; the regulariser itself merely returns the
optimal :math:`(V^\\star, \\tau)` for the ``V`` it is handed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

# Number of bisection iterations. 30 steps shrink the interval by 2^30,
# i.e. by a factor of ~1e9; combined with a tau_max of order 1e3 this
# resolves tau to ~1e-6, far below any plausible tau_threshold.
_BISECT_ITERS: int = 30

# Upper bound on the bisection interval. A few hundred is enough to
# bring kappa below any reasonable target for inputs that aren't
# pathologically conditioned (e.g. exact zero diagonal entries).
_TAU_MAX: float = 1.0e3


def _kappa(V: Float[Array, "M M"]) -> Float[Array, ""]:
    """Condition number :math:`\\kappa(V) = \\sigma_{\\max}/\\sigma_{\\min}`."""
    return jnp.linalg.cond(V)


def _apply_tau(
    V: Float[Array, "M M"],
    tau: Float[Array, ""],
) -> Float[Array, "M M"]:
    """Return :math:`V + \\tau \\cdot \\operatorname{diag}(V)`."""
    return V + tau * jnp.diag(jnp.diag(V))


@jdc.pytree_dataclass
class DiagonalTikhonov:
    """Diagonal Tikhonov regulariser with adaptive :math:`\\tau`.

    Parameters
    ----------
    kappa_target : float (static, default 1e6)
        Upper bound on :math:`\\kappa(V^\\star)`.
    tau_threshold : float (static, default 0.01)
        Threshold for the ``binding_ridge`` diagnostic flag elsewhere in
        the pipeline. Not used inside :meth:`apply` itself; carried for
        downstream consumers.
    """

    kappa_target: float = jdc.static_field(default=1.0e6)  # type: ignore[attr-defined]
    tau_threshold: float = jdc.static_field(default=1.0e-2)  # type: ignore[attr-defined]

    def apply(
        self,
        V: Float[Array, "M M"],
    ) -> tuple[Float[Array, "M M"], Float[Array, ""]]:
        """Return :math:`(V^\\star, \\tau)` with :math:`\\kappa(V^\\star) \\leq \\kappa_{\\mathrm{target}}`.

        If ``V`` already satisfies the target, returns ``(V, 0.0)``.
        Otherwise bisects :math:`\\tau \\in [0, \\tau_{\\max}]` until the
        smallest interval-upper-bound satisfies the constraint, with a
        fixed iteration count so the routine traces under ``jit`` /
        ``vmap``.

        Parameters
        ----------
        V : (M, M) symmetric (typically PSD) array.

        Returns
        -------
        V_star : (M, M) array
            The regularised matrix :math:`V + \\tau \\cdot \\operatorname{diag}(V)`.
        tau : scalar array
            The realised :math:`\\tau`.
        """
        kappa_target = jnp.asarray(self.kappa_target)
        kappa_V = _kappa(V)

        # Bisection state: (lo, hi). Loop invariant: kappa(V_star(hi))
        # is always within the target (or hi is the explicit upper
        # bound, which we trust to be feasible for the inputs we
        # encounter).
        lo_init = jnp.asarray(0.0)
        hi_init = jnp.asarray(_TAU_MAX)

        def bisect_step(_: int, state: tuple) -> tuple:
            lo, hi = state
            mid = 0.5 * (lo + hi)
            kappa_mid = _kappa(_apply_tau(V, mid))
            feasible = kappa_mid <= kappa_target
            # If feasible, tighten the upper bound; otherwise raise lo.
            new_lo = jnp.where(feasible, lo, mid)
            new_hi = jnp.where(feasible, mid, hi)
            return (new_lo, new_hi)

        _, hi_final = jax.lax.fori_loop(
            0, _BISECT_ITERS, bisect_step, (lo_init, hi_init)
        )

        # The upper bound of the final interval is the smallest tau we
        # have verified to be feasible during the search.
        tau_search = hi_final

        # Short-circuit when V already meets the target: take tau = 0.
        already_ok = kappa_V <= kappa_target
        tau = jnp.where(already_ok, jnp.asarray(0.0), tau_search)
        V_star = _apply_tau(V, tau)
        return V_star, tau

    def apply_fixed_tau(
        self,
        V: Float[Array, "M M"],
        tau: Float[Array, ""],
    ) -> Float[Array, "M M"]:
        """Return :math:`V + \\tau \\cdot \\operatorname{diag}(V)` at a fixed,
        externally supplied :math:`\\tau`.

        This is the "anchor-once-then-freeze" application path: the
        :func:`emu_gmm.estimator.estimate` driver calls
        :meth:`apply` once at :math:`\\theta_{\\mathrm{init}}` to obtain a
        ``tau_anchor``; subsequent evaluations during optimisation route
        through :meth:`apply_fixed_tau` with that anchored ``tau`` so the
        residual surface is deterministic and :math:`C^1` in
        :math:`\\theta`. See ``docs/design.org`` Section 5 and CLAUDE.md
        commitment 3 for the policy.

        Parameters
        ----------
        V : (M, M) symmetric (typically PSD) array.
        tau : scalar array
            The anchored ridge magnitude, computed by an earlier
            :meth:`apply` call. May be a Python float or a 0-d JAX array.

        Returns
        -------
        V_star : (M, M) array
            The regularised matrix :math:`V + \\tau \\cdot \\operatorname{diag}(V)`.
        """
        tau_arr = jnp.asarray(tau)
        return _apply_tau(V, tau_arr)


__all__ = ["DiagonalTikhonov"]
