"""PD-restoration via diagonal Tikhonov regularisation.

Pairwise-overlap variance estimators (and Monte Carlo variances at
small ``n_sim``) can produce ``V`` matrices that are numerically
non-PD --- close to singular or with a wildly large condition number.
The framework's response, per ``docs/design.org`` Section 5, is an
adaptive ridge

.. math::
   V^\\star \\;=\\; V \\;+\\; \\tau \\cdot R(V),

with :math:`\\tau \\geq 0` chosen as small as possible subject to
:math:`\\kappa(V^\\star) \\leq \\kappa_{\\mathrm{target}}`. The reference
:math:`R(V)` is :math:`\\operatorname{diag}(V)` in the canonical case ---
this is the scale-equivariant form the design specifies, since per-moment
rescaling :math:`V \\to D V D` carries through:
:math:`D V D + \\tau \\operatorname{diag}(D V D) = D (V + \\tau \\operatorname{diag}(V)) D`.

When :math:`V` is (numerically) diagonal the canonical form degenerates:
:math:`V + \\tau \\operatorname{diag}(V) = (1 + \\tau) V`, which leaves
:math:`\\kappa(V^\\star) = \\kappa(V)` regardless of :math:`\\tau`. The
bisection silently saturates at :math:`\\tau_{\\max}` and the regulariser
returns a useless answer. Diagonal :math:`V` is not exotic --- it arises
naturally from :class:`IIDCovariance` and :class:`ClusteredCovariance` on
uncorrelated moments, and from :class:`SyntheticCovariance` on independent
simulation draws.

The fix is a dispatch: when :math:`V` is detected as (near-)diagonal we
switch to an additive eigenvalue shift

.. math::
   V^\\star_{\\text{diag-case}} \\;=\\; V \\;+\\; \\tau \\cdot \\bar{d} \\cdot I,
   \\qquad \\bar{d} = \\operatorname{tr}(V)/M,

which does change :math:`\\kappa`. The scaling by :math:`\\bar{d}` keeps
:math:`\\tau` in the same units as :math:`\\operatorname{diag}(V)` --- the
two formulas agree when :math:`V` is a multiple of the identity, and both
preserve uniform-scale equivariance (:math:`V \\to \\alpha V`). Per-moment
scale equivariance survives in the non-diagonal branch (the standard
case).

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
from jaxtyping import Array, Bool, Float

# Number of bisection iterations. 30 steps shrink the interval by 2^30,
# i.e. by a factor of ~1e9; combined with a tau_max of order 1e3 this
# resolves tau to ~1e-6, far below any plausible tau_threshold.
_BISECT_ITERS: int = 30

# Upper bound on the bisection interval. A few hundred is enough to
# bring kappa below any reasonable target for inputs that aren't
# pathologically conditioned (e.g. exact zero diagonal entries).
_TAU_MAX: float = 1.0e3

# Relative tolerance for the diagonal-V detector. ``V`` is treated as
# diagonal when ``max|off-diag| <= _DIAG_RTOL * max(mean|diag|, 1.0)``.
# Tight enough that a generic floating-point off-diagonal won't trip the
# branch; loose enough to absorb roundoff in symmetrised covariance
# constructions.
_DIAG_RTOL: float = 1.0e-12


def _kappa(V: Float[Array, "M M"]) -> Float[Array, ""]:
    """Condition number :math:`\\kappa(V) = \\sigma_{\\max}/\\sigma_{\\min}`."""
    return jnp.linalg.cond(V)


def _is_diagonal(V: Float[Array, "M M"]) -> Bool[Array, ""]:
    """Return a 0-d boolean array: ``True`` iff ``V`` is (near-)diagonal.

    A matrix is treated as diagonal when the largest off-diagonal entry
    is below ``_DIAG_RTOL`` times the typical diagonal scale. The
    comparison is jit-friendly (pure JAX, no Python branching) and
    handles all-zero ``V`` gracefully via the ``max(..., 1.0)`` floor.
    """
    diag = jnp.diag(V)
    off = V - jnp.diag(diag)
    diag_scale = jnp.maximum(jnp.mean(jnp.abs(diag)), 1.0)
    return jnp.max(jnp.abs(off)) <= _DIAG_RTOL * diag_scale


def _apply_tau(
    V: Float[Array, "M M"],
    tau: Float[Array, ""],
    is_diag: Bool[Array, ""],
) -> Float[Array, "M M"]:
    """Return the ridged ``V`` under the diagonal-aware dispatch.

    - Non-diagonal branch: :math:`V + \\tau \\operatorname{diag}(V)`
      (canonical, per-moment scale-equivariant).
    - Diagonal branch: :math:`V + \\tau (\\operatorname{tr}(V)/M) I`
      (additive eigenvalue shift; needed because the canonical form
      degenerates to :math:`(1 + \\tau) V` on diagonal ``V``).

    Both branches are evaluated under jit; the ``is_diag`` boolean
    selects one. This keeps the routine traceable and avoids data-
    dependent Python control flow.
    """
    M = V.shape[0]
    # Canonical (non-diagonal) branch: V + tau * diag(V).
    ridge_canonical = V + tau * jnp.diag(jnp.diag(V))
    # Diagonal branch: V + tau * (tr(V)/M) * I. Use the trace-mean as
    # the scale so the additive shift is in units of the average
    # diagonal entry --- units match the canonical branch in the
    # isotropic limit V = c*I.
    scale = jnp.trace(V) / jnp.asarray(M, dtype=V.dtype)
    ridge_diag = V + tau * scale * jnp.eye(M, dtype=V.dtype)
    return jnp.where(is_diag, ridge_diag, ridge_canonical)


@jdc.pytree_dataclass
class DiagonalTikhonov:
    """Adaptive Tikhonov regulariser with a diagonal-V dispatch.

    Parameters
    ----------
    kappa_target : float (static, default 1e6)
        Upper bound on :math:`\\kappa(V^\\star)`.
    tau_threshold : float (static, default 0.01)
        Threshold for the ``binding_ridge`` diagnostic flag elsewhere in
        the pipeline. Not used inside :meth:`apply` itself; carried for
        downstream consumers.

    Notes
    -----
    The regulariser applies an adaptive ridge

    .. math::
       V^\\star = V + \\tau \\cdot R(V),

    where :math:`R(V) = \\operatorname{diag}(V)` for generic non-diagonal
    ``V`` (the design's per-moment scale-equivariant form) and
    :math:`R(V) = (\\operatorname{tr}(V)/M)\\,I` when ``V`` is detected
    as (near-)diagonal. The diagonal-branch form is needed because
    :math:`V + \\tau \\operatorname{diag}(V) = (1+\\tau)V` for diagonal
    ``V``, which leaves :math:`\\kappa(V^\\star) = \\kappa(V)` regardless
    of :math:`\\tau`; the bisection would silently saturate at
    :math:`\\tau_{\\max}` without changing the condition number. See the
    module docstring for the full derivation.
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

        The ridge formula is dispatched based on ``V``'s structure:
        diagonal ``V`` uses :math:`V + \\tau (\\operatorname{tr}(V)/M) I`,
        general ``V`` uses :math:`V + \\tau \\operatorname{diag}(V)`. See
        the class docstring.

        Parameters
        ----------
        V : (M, M) symmetric (typically PSD) array.

        Returns
        -------
        V_star : (M, M) array
            The regularised matrix.
        tau : scalar array
            The realised :math:`\\tau`.
        """
        kappa_target = jnp.asarray(self.kappa_target)
        kappa_V = _kappa(V)
        is_diag = _is_diagonal(V)

        # Bisection state: (lo, hi). Loop invariant: kappa(V_star(hi))
        # is always within the target (or hi is the explicit upper
        # bound, which we trust to be feasible for the inputs we
        # encounter).
        lo_init = jnp.asarray(0.0)
        hi_init = jnp.asarray(_TAU_MAX)

        def bisect_step(_: int, state: tuple) -> tuple:
            lo, hi = state
            mid = 0.5 * (lo + hi)
            kappa_mid = _kappa(_apply_tau(V, mid, is_diag))
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
        V_star = _apply_tau(V, tau, is_diag)
        return V_star, tau

    def apply_fixed_tau(
        self,
        V: Float[Array, "M M"],
        tau: Float[Array, ""],
    ) -> Float[Array, "M M"]:
        """Return :math:`V + \\tau \\cdot R(V)` at a fixed, externally supplied :math:`\\tau`.

        :math:`R(V)` is dispatched diagonally vs. non-diagonally exactly
        as in :meth:`apply`. Detection happens on the supplied ``V``;
        for a covariance family that is structurally diagonal at every
        :math:`\\theta` (e.g. :class:`IIDCovariance` on uncorrelated
        moments), the detection is stable along the optimisation path
        and the residual surface stays :math:`C^1`.

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
            The regularised matrix.
        """
        tau_arr = jnp.asarray(tau)
        is_diag = _is_diagonal(V)
        return _apply_tau(V, tau_arr, is_diag)


__all__ = ["DiagonalTikhonov"]
