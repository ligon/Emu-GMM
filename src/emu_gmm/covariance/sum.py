r"""Signed sum of covariance strategies (#81), with multiway clustering.

``SumCovariance`` composes child :class:`CovarianceStrategy` objects into

.. math::
    V_X(\theta) = \sum_k s_k\, V_k(\theta), \qquad s_k \in \{+1, -1\},

the *additive* half of the composable-covariance algebra. (The *coupled*
half --- where the cross block is not zero --- is
:class:`~emu_gmm.covariance.stratified.DesignAwareCovariance`, which a plain
sum cannot express.)

The canonical consumer, and the reason this primitive exists rather than
being guessed at, is **multiway (two-way) clustering** (Cameron--Gelbach--
Miller): clustering on two cross-cutting dimensions :math:`g` and :math:`h`,

.. math::
    V = V_g + V_h - V_{g \cap h},

a /signed/ combination (the subtraction is inclusion--exclusion, removing
the double-counted within-:math:`(g \cap h)` covariance). Build it with
:meth:`SumCovariance.two_way_cluster`.

Contract
--------
- **Shared scale.** Every summand must be an empirical strategy on the
  *same* measure, so they share one :math:`N_j` and the
  :math:`1/(N_j N_k)` Var(mean) normalisation (CLAUDE.md commitment 9).
  Summing strategies that disagree on the mask / :math:`N_j` is incoherent.
- **Shared intermediates.** The ``cached_intermediates`` 4-tuple is threaded
  unchanged to every child, so a single ``vmap(psi)`` pass is reused across
  all terms.
- **PSD is not guaranteed.** Under subtraction the result can be indefinite
  --- the well-known multiway-clustering non-PD risk. No internal PD repair
  is performed; the regularization layer (:class:`DiagonalTikhonov`) handles
  it.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.types import ParamsLike, StructuralModel


@jdc.pytree_dataclass
class SumCovariance:
    r"""Signed sum :math:`V_X = \sum_k s_k V_k` of child covariance strategies.

    Parameters
    ----------
    terms : tuple of CovarianceStrategy
        The child strategies :math:`V_k`. Held as a pytree field (their
        traced leaves --- cluster ids etc. --- flatten through ``jit`` /
        ``vmap``). All must be empirical strategies on the same measure.
    signs : tuple of float (static)
        The per-term signs :math:`s_k` (``+1.0`` / ``-1.0``), one per entry
        of ``terms``. Static, since they define the combination's structure.
    """

    terms: tuple[Any, ...]
    signs: tuple[float, ...] = jdc.static_field()  # type: ignore[attr-defined]

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
        r"""Assemble :math:`V_X = \sum_k s_k V_k(\theta)`.

        Each child receives the *same* ``cached_intermediates`` (one
        ``vmap(psi)`` pass shared across terms). The signed sum is
        symmetrised once against round-off. PSD is not guaranteed (see the
        class docstring); the regularization layer handles repair.
        """
        acc: Any = None
        # strict=True: a terms/signs length mismatch is a construction error
        # and should fail loudly here, not silently drop a term.
        for sign, term in zip(self.signs, self.terms, strict=True):
            V_k = term.covariance(psi, theta, measure, cached_intermediates)
            contribution = sign * V_k
            acc = contribution if acc is None else acc + contribution
        return 0.5 * (acc + acc.T)  # symmetrise against round-off

    @classmethod
    def two_way_cluster(
        cls,
        cluster_ids_a: Any,
        n_clusters_a: int,
        cluster_ids_b: Any,
        n_clusters_b: int,
        *,
        dof_correction: bool = False,
    ) -> "SumCovariance":
        r"""Two-way clustering (Cameron--Gelbach--Miller): :math:`V_a + V_b - V_{a \cap b}`.

        Composes three :class:`ClusteredCovariance` passes: one clustered on
        dimension :math:`a`, one on :math:`b`, and one on the *intersection*
        :math:`a \cap b` (two observations share an intersection cluster iff
        they share both :math:`a` and :math:`b`). The intersection ids are
        densified as ``a * n_clusters_b + b`` over ``n_clusters_a *
        n_clusters_b`` segments; empty :math:`(a, b)` cells carry zero total
        and contribute nothing.

        Parameters
        ----------
        cluster_ids_a, cluster_ids_b : (N,) arrays
            Per-observation cluster indices for the two dimensions, in
            ``[0, n_clusters_a)`` / ``[0, n_clusters_b)``. Rounded to the
            nearest integer inside :class:`ClusteredCovariance`.
        n_clusters_a, n_clusters_b : int
            Distinct cluster counts for each dimension.
        dof_correction : bool, optional
            Forwarded to each :class:`ClusteredCovariance` term. Default
            ``False``. Note: the exact small-sample dof for two-way
            clustering is the *minimum* cluster count across dimensions (CGM)
            --- applying the per-term correction is an approximation, so it
            is off by default; leave it off unless you have checked the
            convention you need.

        Returns
        -------
        SumCovariance
            ``terms = (V_a, V_b, V_{a∩b})`` with ``signs = (+1, +1, -1)``.

        Notes
        -----
        For a very large ``n_clusters_a * n_clusters_b`` the dense
        intersection segmentation is memory-heavy; pre-densify the
        intersection ids and build the three terms by hand if that matters.
        With ``a == b`` this collapses to one-way clustering on ``a``
        (``V_a + V_a - V_a``), a useful sanity check.
        """
        a = jnp.asarray(cluster_ids_a, dtype=jnp.float64)
        b = jnp.asarray(cluster_ids_b, dtype=jnp.float64)
        cov_a = ClusteredCovariance(
            cluster_ids=a, n_clusters=n_clusters_a, dof_correction=dof_correction
        )
        cov_b = ClusteredCovariance(
            cluster_ids=b, n_clusters=n_clusters_b, dof_correction=dof_correction
        )
        # Intersection cluster id = (a, b) pair, densified into [0, n_a*n_b).
        ab = jnp.round(a) * n_clusters_b + jnp.round(b)
        cov_ab = ClusteredCovariance(
            cluster_ids=ab,
            n_clusters=n_clusters_a * n_clusters_b,
            dof_correction=dof_correction,
        )
        return cls(terms=(cov_a, cov_b, cov_ab), signs=(1.0, 1.0, -1.0))


__all__ = ["SumCovariance"]
