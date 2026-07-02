"""Adaptive diagonal-Tikhonov ridge plus Cholesky inverse.

A one-shot helper that mirrors the mechanics of
:class:`emu_gmm.regularization.DiagonalTikhonov` but exposes the
inverse directly. Given a symmetric (typically PSD) matrix :math:`M`
and a target condition number :math:`\\kappa_\\star`, return

.. math::
   M^{-1}_{\\mathrm{ridged}}, \\qquad
   M_\\star = M + \\tau \\cdot \\operatorname{diag}(M),

where :math:`\\tau \\geq 0` is the smallest value (bisected over a
fixed-size grid) such that :math:`\\kappa(M_\\star) \\leq \\kappa_\\star`,
together with a metadata dict.

The inverse is computed via the Cholesky factorisation of
:math:`M_\\star`; we never form ``jnp.linalg.inv(M)`` directly.

This helper exists primarily for callers that need an explicit
:math:`\\Omega^{-1}`-style ridge outside the main estimator pipeline
(see e.g. K-Aggregators). Inside the estimator, the
:class:`DiagonalTikhonov` strategy is the canonical entry point and
keeps the Cholesky factor around rather than materialising the
inverse.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float

from emu_gmm.regularization import DiagonalTikhonov

#: Default threshold for the ``binding`` diagnostic flag. ``tau`` above
#: this is treated as "the ridge had to work hard"; downstream callers
#: can use this to surface a warning. Matches
#: :attr:`DiagonalTikhonov.tau_threshold`.
_BINDING_TAU_THRESHOLD: float = 1.0e-2


def _strip(value: Any) -> tuple[Float[Array, "M M"], tuple[ha.Axis, ...] | None]:
    """Return ``(array, axes_or_None)`` from a NamedArray-or-plain input."""
    if isinstance(value, ha.NamedArray):
        return value.array, tuple(value.axes)
    return jnp.asarray(value), None


def ridge_inverse(
    M: Any,
    *,
    target_condition: float = 1.0e6,
) -> tuple[ha.NamedArray, dict[str, Any]]:
    """Return :math:`M_\\star^{-1}` and metadata for an adaptive ridge.

    Bisects :math:`\\tau \\geq 0` on the diagonal-Tikhonov form
    :math:`M_\\star = M + \\tau \\cdot \\operatorname{diag}(M)` until
    :math:`\\kappa(M_\\star) \\leq` ``target_condition``, then returns
    :math:`M_\\star^{-1}` computed via Cholesky.

    Parameters
    ----------
    M : :class:`haliax.NamedArray` or plain (M, M) array
        Symmetric (typically PSD) matrix to invert. A NamedArray's
        axes are preserved on the returned inverse. A plain array is
        wrapped on output with positional axes named ``("dim",
        "dim_dual")``.
    target_condition : float, keyword-only, default ``1e6``
        Upper bound on :math:`\\kappa(M_\\star)`. Must be ``> 1``.

    Returns
    -------
    M_inv_ridged : :class:`haliax.NamedArray`
        :math:`M_\\star^{-1}` wrapped with the input's axes (or
        positional fallbacks for plain-array input).
    info : dict
        Diagnostic metadata with keys:

        - ``'tau'`` (float): the realised :math:`\\tau`.
        - ``'kappa_before'`` (float): :math:`\\kappa(M)`.
        - ``'kappa_after'`` (float): :math:`\\kappa(M_\\star)`.
        - ``'binding'`` (bool): ``True`` if ``tau`` exceeds the
          binding-threshold (currently 1e-2); a signal that the
          ridge had to do non-trivial work.
        - ``'saturated'`` (bool): ``True`` if the realised
          :math:`M_\\star` is still **not** positive-definite — tested
          on the *signed* spectrum
          (``eigvalsh(sym(M_star)).min() <= 0``, NaN-safe: NaN
          eigenvalues count as saturated). The diagonal-Tikhonov ridge
          cannot repair e.g. an exact zero diagonal entry (the ridge
          adds ``tau * M[i, i] == 0`` there), so the Cholesky NaNs and
          the returned "inverse" is all-NaN. **A saturated result
          means the returned inverse is NaN/meaningless** — check this
          flag before consuming the inverse.

        ``'kappa_before'`` / ``'kappa_after'`` are SVD condition
        numbers (:func:`jnp.linalg.cond`, a ratio of *absolute*
        singular values) and are therefore blind to a small negative
        (or exactly zero) eigenvalue — see CLAUDE.md architectural
        commitment 3. Treat them as diagnostics only; ``'saturated'``
        is the positive-definiteness verdict.

    Notes
    -----
    Internally calls
    :meth:`emu_gmm.regularization.DiagonalTikhonov.apply` to perform
    the bisection, so the underlying mechanics are guaranteed
    identical to the in-estimator regularisation path.
    """
    if target_condition <= 1.0:
        raise ValueError(
            f"ridge_inverse: target_condition must be > 1, got {target_condition!r}"
        )

    arr, axes_in = _strip(M)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"ridge_inverse: expected a square 2-D matrix, got shape {arr.shape}"
        )

    kappa_before = jnp.linalg.cond(arr)

    # Delegate the bisection to DiagonalTikhonov so the two code paths
    # stay in lockstep. We pass kappa_target = target_condition; the
    # tau_threshold field of DiagonalTikhonov is unused inside apply()
    # and matters only for the downstream ``binding`` diagnostic, which
    # we compute here against our own _BINDING_TAU_THRESHOLD.
    reg = DiagonalTikhonov(
        kappa_target=float(target_condition),
        tau_threshold=_BINDING_TAU_THRESHOLD,
    )
    M_star, tau = reg.apply(arr)

    # Cholesky-based inverse: L L^T = M_star, so M_star^{-1} =
    # L^{-T} L^{-1}. cho_solve(... , I) does exactly that.
    L = jax.scipy.linalg.cholesky(M_star, lower=True)
    eye = jnp.eye(arr.shape[0], dtype=M_star.dtype)
    M_inv = jax.scipy.linalg.cho_solve((L, True), eye)

    # Symmetrise to wash out floating-point asymmetry from the solve.
    M_inv = 0.5 * (M_inv + M_inv.T)

    kappa_after = jnp.linalg.cond(M_star)

    # Saturation check on the SIGNED spectrum of the realised M_star:
    # jnp.linalg.cond is an SVD ratio of absolute singular values and is
    # blind to a zero/negative eigenvalue (CLAUDE.md commitment 3), so
    # it cannot detect that the delegated ridge failed to achieve PD
    # (e.g. an exact zero diagonal entry, which tau * diag(M) cannot
    # lift). In that case the Cholesky above NaNs and M_inv is all-NaN;
    # flag it rather than returning a silent NaN inverse. NaN-safe:
    # ``NaN > 0`` is False, so NaN eigenvalues count as saturated.
    min_eig = jnp.min(jnp.linalg.eigvalsh(0.5 * (M_star + M_star.T)))
    saturated = not bool(min_eig > 0.0)

    tau_f = float(tau)
    info: dict[str, Any] = {
        "tau": tau_f,
        "kappa_before": float(kappa_before),
        "kappa_after": float(kappa_after),
        "binding": tau_f > _BINDING_TAU_THRESHOLD,
        "saturated": saturated,
    }

    if axes_in is None:
        # Plain-array input: wrap with positional axes for the public
        # NamedArray return.
        dim = arr.shape[0]
        axes_out: tuple[ha.Axis, ...] = (
            ha.Axis(name="dim", size=dim),
            ha.Axis(name="dim_dual", size=dim),
        )
    else:
        axes_out = axes_in
    named_inv = ha.named(M_inv, axes_out)
    return named_inv, info


__all__ = ["ridge_inverse"]
