"""Pairwise-overlap iid variance for empirical measures.

``IIDCovariance`` implements the pweighted, pairwise-overlap form of the
sample variance of the moment estimator. Per ``docs/design.org`` Section
2 and ``docs/mcar-asymptotics.org`` Section 5,

.. math::
   [V_X(\\theta)]_{jk}
   \\;=\\;
   \\frac{1}{N_j\\, N_k}\\,
   \\sum_{i=1}^N d_{ij}\\, d_{ik}\\, w_i^2\\,
   \\psi_j(x_i, \\theta)\\, \\psi_k(x_i, \\theta),

with :math:`N_j = \\sum_i d_{ij} w_i`. Each element uses the rows where
both moments are observable; listwise deletion is avoided at the cost of
finite-sample positive-definiteness (the framework's regularisation
layer handles that).

This is the "single-PSU" reduction of :class:`ClusteredCovariance`: with
each cluster of size one, the cluster-totals form collapses to the
expression above.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a haliax NamedArray wrapper, returning the underlying array."""
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


def _safe_outer_divide(
    numer: Float[Array, "M M"],
    denom_vec: Float[Array, " M"],
) -> Float[Array, "M M"]:
    """Return ``numer / (denom_vec outer denom_vec)`` with zero on degeneracy.

    Used to perform the ``1 / (N_j N_k)`` normalisation. When ``N_j`` or
    ``N_k`` is zero, the corresponding entry collapses to zero rather
    than ``inf`` / ``nan``; the estimator surfaces the degenerate
    coordinates separately through ``Diagnostics.N_j``.
    """
    denom = jnp.outer(denom_vec, denom_vec)
    safe = jnp.where(denom == 0.0, 1.0, denom)
    out = numer / safe
    return jnp.where(denom == 0.0, jnp.zeros_like(out), out)


@jdc.pytree_dataclass
class IIDCovariance:
    """Pairwise-overlap iid variance for an :class:`EmpiricalMeasure`.

    No configurable state in v1: the strategy reads ``x``, ``mask``, and
    ``weights`` off the measure and assembles the sample variance of the
    moment estimator under the pairwise-overlap rule.
    """

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
    ) -> Float[Array, "M M"]:
        """Construct :math:`V_X(\\theta)` for the supplied measure.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.
        measure
            An :class:`~emu_gmm.measures.empirical.EmpiricalMeasure`
            instance exposing ``x``, ``mask``, and ``weights``.

        Returns
        -------
        V : (M, M) jax array
            Symmetric PSD by construction.
        """

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(measure.x)  # (N, M)
        mask = measure.mask  # (N, M)
        weights = measure.weights  # (N,)

        # Per-coordinate effective sample size N_j = sum_i d_ij * w_i.
        N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)

        # NaN-mask semantics: drop NaN at masked positions before the sum.
        # 0 * NaN = NaN under IEEE arithmetic, so masked-out rows whose
        # residual evaluates to NaN (e.g. because an input column was NaN
        # under pandas missing-data convention) would otherwise poison V.
        # Replace NaN with 0 only at masked positions; observed-position
        # NaN still propagates and surfaces as a NaN moment in V.
        psi_safe = jnp.where(mask == 0.0, 0.0, psi_batch)

        # Pairwise overlap numerator: sum_i d_ij * d_ik * w_i^2 * psi_j * psi_k.
        # einsum: i is summed; j, k are kept.
        w2 = weights * weights  # (N,)
        weighted_psi = mask * psi_safe  # (N, M); zero out masked-out rows
        numer = jnp.einsum("i,ij,ik->jk", w2, weighted_psi, weighted_psi)

        return _safe_outer_divide(numer, N_j)


__all__ = ["IIDCovariance"]
