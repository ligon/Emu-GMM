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
former is the cluster-aware generalisation of the latter.

The cluster IDs are kept as floats because JAX prefers a floating dtype
for traced values; the implementation casts to a 32-bit integer inside
:func:`jax.ops.segment_sum`. ``n_clusters`` is a static field so the
output dimension is concrete at trace time.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.nan_safety import safe_x_for_psi
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
    """Return ``numer / (denom_vec outer denom_vec)`` with zero on degeneracy."""
    denom = jnp.outer(denom_vec, denom_vec)
    safe = jnp.where(denom == 0.0, 1.0, denom)
    out = numer / safe
    return jnp.where(denom == 0.0, jnp.zeros_like(out), out)


@jdc.pytree_dataclass
class ClusteredCovariance:
    """Cluster-totals variance for an :class:`EmpiricalMeasure`.

    Parameters
    ----------
    cluster_ids : (N,) jax array of floats
        Per-observation cluster index in ``[0, n_clusters)``. JAX prefers
        a float dtype for traced arrays; the implementation casts to a
        32-bit integer inside :func:`jax.ops.segment_sum`.
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
    """

    cluster_ids: Float[Array, " N"]
    n_clusters: int = jdc.static_field()  # type: ignore[attr-defined]
    dof_correction: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[
                Float[Array, " M"],
                Float[Array, "N M"],
                Float[Array, "N M"],
                Float[Array, " M"],
            ]
            | None
        ) = None,
    ) -> Float[Array, "M M"]:
        """Construct :math:`V_X(\\theta)` via cluster-total outer products.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.
        measure
            An :class:`~emu_gmm.measures.empirical.EmpiricalMeasure`
            instance exposing ``x``, ``mask``, and ``weights``.
        cached_intermediates : optional 4-tuple
            ``(m, psi_safe, weight_mask, N_j)`` produced by a previous
            call to
            :meth:`EmpiricalMeasure.expectation_and_contributions`. When
            supplied, this routine reuses the cached ``psi_safe``,
            ``weight_mask``, and ``N_j``, avoiding a redundant
            ``jax.vmap(psi)`` pass (see
            ``docs/reviews/v1x-performance-review.org`` finding #4).
            Back-compat: when ``None``, falls through to the
            self-computing path.

        Returns
        -------
        V : (M, M) jax array
            Symmetric PSD by construction.

        Notes
        -----
        With each cluster of size one
        (``cluster_ids = [0, 1, ..., N-1]`` and ``n_clusters = N``), the
        cluster-totals form reduces to :class:`IIDCovariance`. The unit
        test ``tests/covariance/test_clustered.py::test_singleton_clusters``
        verifies this special case.
        """
        if cached_intermediates is not None:
            _m, psi_safe, weight_mask, N_j = cached_intermediates
            # Per-observation contribution to moment j: d_ij * w_i * psi_j.
            # ``weight_mask`` already encodes d_ij * w_i, so multiply
            # directly by ``psi_safe``.
            contrib = weight_mask * psi_safe  # (N, M)
            segment_ids = self.cluster_ids.astype(jnp.int32)
            cluster_totals = jax.ops.segment_sum(
                contrib, segment_ids, num_segments=self.n_clusters
            )  # (n_clusters, M)
            numer = jnp.einsum("cj,ck->jk", cluster_totals, cluster_totals)
            if self.dof_correction:
                numer = numer * self._finite_cluster_correction(measure.mask)
            return _safe_outer_divide(numer, N_j)

        # Pre-sanitise data with the per-column observed-mean sentinel
        # so partial residuals (``log``, ``1/x``, ``sqrt``) cannot
        # introduce NaN/Inf at masked-out cells and poison reverse-mode
        # AD (see :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`
        # and :meth:`EmpiricalMeasure.expectation`).
        x_safe = safe_x_for_psi(measure.x)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
        mask = measure.mask  # (N, M)
        weights = measure.weights  # (N,)

        # NaN-safe contraction: replace psi_batch at masked-out cells
        # with zero before multiplying by the weighted mask. Mirrors
        # the guard in :class:`IIDCovariance` so that a user-supplied
        # psi which returns NaN at masked-out rows still yields a
        # finite cluster-totals covariance.
        mask_bool = mask > 0.0
        psi_safe = jnp.where(mask_bool, psi_batch, 0.0)  # (N, M)

        # Per-coordinate sample size N_j (same as IIDCovariance).
        N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)

        # Per-observation contribution to moment j: d_ij * w_i * psi_j.
        contrib = mask * weights[:, None] * psi_safe  # (N, M)

        # Segment-sum into cluster totals. jax.ops.segment_sum operates
        # on the leading axis only, so we sum the (N, M) contribution
        # along N grouped by cluster ID and end up with (n_clusters, M).
        segment_ids = self.cluster_ids.astype(jnp.int32)
        cluster_totals = jax.ops.segment_sum(
            contrib, segment_ids, num_segments=self.n_clusters
        )  # (n_clusters, M)

        # Outer product per cluster, then sum across clusters.
        # einsum: c is summed; j, k are kept.
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
        seg = self.cluster_ids.astype(jnp.int32)
        per_cluster_obs = jax.ops.segment_sum(
            (mask > 0.0).astype(mask.dtype), seg, num_segments=self.n_clusters
        )  # (n_clusters, M): observed-unit count per (cluster, coord)
        s = (per_cluster_obs > 0.0).astype(mask.dtype)  # support, (n_clusters, M)
        G = jnp.einsum("cj,ck->jk", s, s)  # (M, M) per-pair cluster counts
        return jnp.where(G >= 2.0, G / jnp.where(G >= 2.0, G - 1.0, 1.0), 1.0)


__all__ = ["ClusteredCovariance"]
