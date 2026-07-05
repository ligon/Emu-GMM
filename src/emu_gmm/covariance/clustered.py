"""Cluster-robust variance via cluster-total outer products.

``ClusteredCovariance`` implements the cluster-totals form of the sample
variance of the moment estimator, suitable when within-cluster
correlation must be respected (households within villages, students
within schools, replicates within survey clusters). Per
``docs/design.org`` Section 2,

.. math::
   [V_X(\\theta)]_{jk}
   \\;=\\;
   \\frac{1}{N_j\\, N_k}\\,
   \\sum_c \\bigg(\\sum_{i \\in c} d_{ij}\\, w_i\\, \\psi_j(x_i, \\theta)\\bigg)
        \\bigg(\\sum_{i \\in c} d_{ik}\\, w_i\\, \\psi_k(x_i, \\theta)\\bigg),

with :math:`N_j = \\sum_i d_{ij} w_i`. With each cluster of size one
this collapses to :class:`emu_gmm.covariance.iid.IIDCovariance`; the
former is the cluster-aware generalisation of the latter. The two share
the :class:`_OuterProductCovariance` template (the ``psi`` evaluation,
``mask`` / ``N_j`` bookkeeping, and the ``centered`` knob); this class
implements only its combination step (:meth:`_assemble`).

The cluster IDs are kept as floats because JAX prefers a floating dtype
for traced values; the implementation casts to a 32-bit integer inside
:func:`jax.ops.segment_sum`. ``n_clusters`` is a static field so the
output dimension is concrete at trace time.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from ._outer_product import _OuterProductCovariance, _safe_outer_divide


@jdc.pytree_dataclass
class ClusteredCovariance(_OuterProductCovariance):
    """Cluster-totals variance for an :class:`EmpiricalMeasure`.

    Parameters
    ----------
    cluster_ids : (N,) jax array
        Per-observation cluster index in ``[0, n_clusters)``. Integer-valued;
        a float dtype is accepted (JAX prefers floats for traced arrays) and
        is **rounded** to the nearest 32-bit integer inside
        :func:`jax.ops.segment_sum` --- so a near-integer float id (e.g. from
        float32 round-off on a large code) bins to the right cluster rather
        than truncating into a neighbour. Integer-dtype arrays work directly.
    n_clusters : int (static)
        Number of distinct cluster IDs. Must satisfy
        ``max(cluster_ids) < n_clusters``. Treated as a static field so
        the segment-sum output dimension is concrete at trace time.
    dof_correction : bool (static), default ``False``
        Apply the finite-cluster degrees-of-freedom correction
        :math:`G_{jk}/(G_{jk}-1)`, where :math:`G_{jk}` is the number of
        clusters that hold at least one observed unit for **both** moments
        :math:`j` and :math:`k` (the per-pair effective cluster count). Off
        by default --- the bare cluster-totals sandwich underestimates the
        sampling variance with few clusters, and the correction is the
        standard small-sample inflation. With complete data every
        :math:`G_{jk}` equals the total cluster count :math:`G`, so the
        factor collapses to the textbook scalar :math:`G/(G-1)`; the
        per-pair form is the design-aware generalisation (CLAUDE.md
        commitment 10), mirroring :class:`StratifiedCovariance`'s own
        :math:`H/(H-1)` Bessel factor and the available-pairs
        :math:`N_j N_k` rule. A pair with :math:`G_{jk} < 2` is left
        uncorrected (factor 1; a single cluster furnishes no between-cluster
        variance). **Recommended whenever the cluster count is small.**
        (The secondary :math:`(N-1)/(N-K)` Stata adjustment is an
        inference-layer concern --- it needs :math:`K` --- and is not
        applied here.)
    centered : bool (static), default ``False``
        Subtract the per-coordinate mean moment :math:`m_j` before forming
        the per-cluster totals (the ``centered`` knob shared with
        :class:`IIDCovariance`, see :class:`_OuterProductCovariance`). Both
        forms coincide asymptotically at :math:`\\theta_0`; they differ by
        :math:`O(\\bar m^2)` at an over-identified :math:`\\hat\\theta`.
        ``centered=True`` matches R's ``gmm`` / ``sandwich`` centered
        convention. The single-PSU reduction to :class:`IIDCovariance` holds
        in both centering directions (the shared template subtracts the same
        :math:`m_j`; see the reduction test).
    """

    cluster_ids: Float[Array, " N"]
    n_clusters: int = jdc.static_field()  # type: ignore[attr-defined]
    dof_correction: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]
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
        r"""Combine into :math:`V_X` via per-cluster-total outer products.

        Each observation's contribution to moment :math:`j` is
        :math:`d_{ij} w_i v_j` (``weight_mask`` :math:`\times` ``centered_psi``,
        with :math:`v = \psi` or :math:`\psi - m`); ``segment_sum`` accumulates
        these into per-cluster totals, whose outer products (summed over
        clusters) form the numerator. The optional per-pair finite-cluster
        factor is applied before the :math:`1/(N_j N_k)` normalisation.
        ``weights`` and ``measure`` are unused here (``weight_mask`` already
        encodes :math:`d_{ij} w_i`, and the mask comes in directly).

        Returns
        -------
        V : (M, M) jax array
            Symmetric. PSD by construction **only when**
            ``dof_correction=False`` (a sum of outer products under a diagonal
            congruence). With ``dof_correction=True`` the entrywise multiply by
            :math:`G_{jk}/(G_{jk}-1)` is not a congruence and can leave the
            result indefinite under unequal support --- handled by the
            regularisation layer (:class:`DiagonalTikhonov`), not here
            (issue #120).
        """
        del weights, measure  # weight_mask already encodes d_ij * w_i
        # Per-observation contribution to moment j: d_ij * w_i * v_j.
        contrib = weight_mask * centered_psi  # (N, M)
        # segment_sum operates on the leading axis: sum the (N, M) contribution
        # along N grouped by cluster ID -> (n_clusters, M) cluster totals.
        segment_ids = jnp.round(self.cluster_ids).astype(jnp.int32)
        cluster_totals = jax.ops.segment_sum(
            contrib, segment_ids, num_segments=self.n_clusters
        )  # (n_clusters, M)
        # Outer product per cluster, then sum across clusters (c summed; j, k kept).
        numer = jnp.einsum("cj,ck->jk", cluster_totals, cluster_totals)
        if self.dof_correction:
            numer = numer * self._finite_cluster_correction(mask)
        return _safe_outer_divide(numer, N_j)

    def _finite_cluster_correction(
        self, mask: Float[Array, "N M"]
    ) -> Float[Array, "M M"]:
        r"""Per-pair finite-cluster factor :math:`G_{jk}/(G_{jk}-1)`.

        :math:`G_{jk}` is the number of clusters with at least one observed
        unit for **both** :math:`j` and :math:`k`. Pairs with
        :math:`G_{jk} < 2` get factor 1 (a single cluster furnishes no
        between-cluster variance). With complete data every :math:`G_{jk}`
        equals the total cluster count, so this is the scalar
        :math:`G/(G-1)`.
        """
        seg = jnp.round(self.cluster_ids).astype(jnp.int32)
        per_cluster_obs = jax.ops.segment_sum(
            (mask > 0.0).astype(mask.dtype), seg, num_segments=self.n_clusters
        )  # (n_clusters, M): observed-unit count per (cluster, coord)
        s = (per_cluster_obs > 0.0).astype(mask.dtype)  # support, (n_clusters, M)
        G = jnp.einsum("cj,ck->jk", s, s)  # (M, M) per-pair cluster counts
        return jnp.where(G >= 2.0, G / jnp.where(G >= 2.0, G - 1.0, 1.0), 1.0)


__all__ = ["ClusteredCovariance"]
