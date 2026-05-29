"""Reverse-mode-AD-safe NaN sentinel construction for masked aggregations.

The empirical-measure pipeline aggregates a user-supplied per-observation
residual :math:`\\psi(x, \\theta)` only over rows where the per-coordinate
mask is one. The canonical "double where" guard

.. code-block:: python

   x_safe = where(isnan(x), 0.0, x)
   psi_batch = vmap(psi)(x_safe)
   psi_safe = where(mask, psi_batch, 0.0)

protects against NaN cells in the *input* ``x``, but does **not** protect
against ``psi`` itself producing NaN/Inf at the sentinel value ``0.0``.
The typical failure mode is a residual that includes ``log(x[0])``,
``1.0 / x[1]``, or ``sqrt(x[2])``: substituting ``0.0`` at masked-out
cells produces ``-inf``, ``+inf``, or ``NaN`` respectively, and the
*reverse-mode* AD cotangent flow ignores the outer ``where`` branch
selection on the primal path --- the NaN gradient at the masked-out cell
poisons the accumulated gradient even though the masked-out cell
contributes zero to the primal value. The reported symptom is
``Diagnostics.final_gradient_norm`` returning ``NaN`` on a converged
solution; the underlying cause is the same poisoning pattern documented
in JAX's "common gotchas" entry on ``jnp.where``.

The fix here is to evaluate ``psi`` at a value that is *guaranteed* to
lie inside ``psi``'s domain at masked-out cells: the per-coordinate
column mean of the *observed* rows. Concretely, for each x-column
:math:`d`, we compute

.. math::
   \\mathrm{sentinel}_d
   \\;=\\;
   \\frac{\\sum_i \\mathbf{1}[x_{id}\\ \\mathrm{finite}]\\, x_{id}}
        {\\sum_i \\mathbf{1}[x_{id}\\ \\mathrm{finite}]},

and replace each NaN cell ``x[i, d]`` with ``sentinel_d``. The sentinel
is in :math:`\\psi`'s domain whenever any observed cell in column
:math:`d` is, so ``log``, ``1/x``, ``sqrt``, and similar partial
operations never see an out-of-domain argument inside the vmap. The mask
then zeroes the result, so the primal value is unchanged.

If a column has *no* observed cells, the sentinel falls back to ``0.0``
to avoid ``0 / 0``; in that degenerate case every row of that column is
masked-out for every moment that reads it, so the choice of sentinel is
immaterial to the primal aggregate, and the corresponding gradient
contribution is multiplied by zero via the mask.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def column_mean_sentinel(x: Float[Array, "N D"]) -> Float[Array, " D"]:
    """Per-column mean over finite cells; zero where a column is all-NaN.

    Parameters
    ----------
    x : (N, D) jax array
        Observations. May contain NaN cells.

    Returns
    -------
    sentinel : (D,) jax array
        ``sentinel[d] = mean(x[i, d] for i where x[i, d] is finite)``,
        or ``0.0`` if no row of column ``d`` is finite. The value is
        guaranteed not to introduce ``NaN`` or ``Inf`` of its own.
    """

    finite = jnp.isfinite(x).astype(x.dtype)  # (N, D) 0/1
    # Replace NaN with 0 inside the sum so the contribution is exactly
    # zero on non-finite cells without poisoning the accumulator.
    x_finite_zeroed = jnp.where(finite > 0.0, x, 0.0)
    numer = jnp.sum(x_finite_zeroed, axis=0)  # (D,)
    denom = jnp.sum(finite, axis=0)  # (D,)
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    mean = numer / safe_denom
    # Columns with no finite entries fall back to 0.0; the sentinel is
    # never used at those columns because every row is masked-out
    # everywhere they appear.
    return jnp.where(denom == 0.0, jnp.zeros_like(mean), mean)


def safe_x_for_psi(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
    """Replace NaN cells in ``x`` with the per-column observed mean.

    The returned array is guaranteed to be NaN-free and, at every cell,
    to lie within the convex hull of the observed values of its column.
    This is the strongest in-domain guarantee available without
    user-supplied bounds: ``log``, ``1/x``, and ``sqrt`` are all safe
    under it whenever any observed cell of the relevant column is in
    their domain.

    The mask must still be applied to the *output* of ``psi`` to zero
    out the masked-out contributions; this helper only ensures the
    ``psi`` evaluation itself does not introduce NaN/Inf at masked-out
    cells, which is exactly what is needed to keep reverse-mode AD
    well-defined (see module docstring for the AD argument).

    Parameters
    ----------
    x : (N, D) jax array
        Observations, possibly containing NaN cells.

    Returns
    -------
    x_eval : (N, D) jax array
        Same shape as ``x``; NaN cells replaced by the per-column
        observed mean, all other cells unchanged.
    """

    sentinel = column_mean_sentinel(x)  # (D,)
    return jnp.where(jnp.isnan(x), sentinel[None, :], x)


__all__ = ["column_mean_sentinel", "safe_x_for_psi"]
