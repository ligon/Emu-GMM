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
expression above. The two share the :class:`_OuterProductCovariance`
template (the ``psi`` evaluation, ``mask`` / ``N_j`` bookkeeping, and the
``centered`` knob); ``IIDCovariance`` implements only its combination step
(:meth:`_assemble`).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from ._outer_product import _OuterProductCovariance, _safe_outer_divide


@jdc.pytree_dataclass
class IIDCovariance(_OuterProductCovariance):
    """Pairwise-overlap iid variance for an :class:`EmpiricalMeasure`.

    The strategy reads ``x``, ``mask``, and ``weights`` off the measure and
    assembles the sample variance of the moment estimator under the
    pairwise-overlap rule.

    The one knob is :attr:`centered`: whether to subtract the per-coordinate
    mean moment before forming the outer products. Both forms are PSD by
    construction and coincide asymptotically at :math:`\\theta_0`; they differ in
    finite samples at an over-identified :math:`\\hat\\theta`. ``centered=False``
    (the default) is the uncentered form; ``centered=True`` matches R's ``gmm``
    default and most textbooks (see
    ``docs/validation/r-reference-crosschecks.org``).
    """

    #: Center each moment coordinate by its mean :math:`m_j = (1/N_j) \\sum_i
    #: d_{ij} w_i \\psi_j` before the pairwise-overlap outer products, so the
    #: numerator is :math:`\\sum_i d_{ij} d_{ik} w_i^2 (\\psi_j - m_j)(\\psi_k -
    #: m_k)`. Default ``False`` (the uncentered :math:`\\sum_i w_i^2 \\psi_j
    #: \\psi_k` form). Both are PSD by construction --- a sum of outer products
    #: of the masked, optionally-centered row vectors --- and agree
    #: asymptotically at :math:`\\theta_0` (where :math:`E[\\psi]=0`); at an
    #: over-identified :math:`\\hat\\theta` they differ by :math:`O(\\bar m^2)`.
    #: ``centered=True`` reproduces the moment-covariance convention of R's
    #: ``gmm`` (``centeredVcov=TRUE``); the uncentered default is retained for
    #: back-compat and its marginally better finite-sample conditioning.
    #: A static field (recompile, not traced).
    centered: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def _assemble(
        self,
        *,
        centered_psi: Float[Array, "N M"],
        mask: Float[Array, "N M"],
        weights: Float[Array, " N"],
        weight_mask: Float[Array, "N M"],
        N_j: Float[Array, " M"],
        measure: Any,
    ) -> Float[Array, "M M"]:
        r"""Combine into :math:`V_X` via the pairwise-overlap :math:`w_i^2` sum.

        The IID estimator uses :math:`w_i^2` in the pairwise sum (not the
        :math:`(d_{ij} w_i)` product squared), so it derives :math:`w^2` from the
        raw ``weights`` and reapplies the plain ``mask`` to ``centered_psi``
        (``weight_mask`` and ``measure`` are unused here). The numerator is
        :math:`\sum_i d_{ij} d_{ik} w_i^2 v_j v_k` with :math:`v = \psi` (or
        :math:`\psi - m`); ``einsum`` sums :math:`i` and keeps :math:`j, k`.

        Returns
        -------
        V : (M, M) jax array
            Symmetric PSD by construction: a sum of outer products
            :math:`\sum_i w_i^2 v_i v_i'` under the :math:`1/(N_j N_k)` diagonal
            congruence (possibly singular, hence not guaranteed strictly *PD* ---
            the regularisation layer owns definiteness). Contrast the per-pair
            dof-corrected :class:`ClusteredCovariance`, which can be indefinite
            (issue #120).
        """
        del weight_mask, measure  # IID uses w_i^2 directly, not d_ij*w_i
        w2 = weights * weights  # (N,)
        weighted_psi = mask * centered_psi  # (N, M); zero masked-out rows
        numer = jnp.einsum("i,ij,ik->jk", w2, weighted_psi, weighted_psi)
        return _safe_outer_divide(numer, N_j)


__all__ = ["IIDCovariance"]
