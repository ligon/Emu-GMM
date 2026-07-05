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


def _mean_moment(
    mask: Float[Array, "N M"],
    weights: Float[Array, " N"],
    psi_safe: Float[Array, "N M"],
    N_j: Float[Array, " M"],
) -> Float[Array, " M"]:
    """Per-coordinate mean moment ``m_j = (1/N_j) sum_i d_ij w_i psi_j``.

    The centre subtracted by ``IIDCovariance(centered=True)``. Degenerate
    coordinates (``N_j == 0``) get ``0`` rather than ``nan`` --- they are already
    surfaced through ``Diagnostics.N_j``. Matches the mean the estimator computes
    for the moment vector, so ``centered=True`` subtracts exactly the fitted
    moment at ``theta``.
    """
    numer_m = jnp.sum(mask * weights[:, None] * psi_safe, axis=0)  # (M,)
    safe_N = jnp.where(N_j == 0.0, 1.0, N_j)
    return jnp.where(N_j == 0.0, 0.0, numer_m / safe_N)


@jdc.pytree_dataclass
class IIDCovariance:
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
        cached_intermediates : optional 4-tuple
            ``(m, psi_safe, weight_mask, N_j)`` produced by a previous
            call to
            :meth:`EmpiricalMeasure.expectation_and_contributions`. When
            supplied, this routine reuses the cached ``psi_safe`` and
            ``N_j`` rather than running ``jax.vmap(psi)`` and rebuilding
            the weight mask --- the shared-primitive consolidation
            that halves ``vmap(psi)`` calls per ``residual_fn`` (see
            ``docs/reviews/v1x-performance-review.org`` finding #4).
            Back-compat: when ``None``, falls through to the
            self-computing path.

        Returns
        -------
        V : (M, M) jax array
            Symmetric PSD by construction: the numerator is a sum of outer
            products :math:`\\sum_i w_i^2 v_i v_i'` --- of :math:`v_i = g_i`
            (uncentered) or :math:`v_i = g_i - \\bar m` (``centered=True``),
            masked per row --- and the :math:`1/(N_j N_k)` scaling is a diagonal
            congruence, both of which preserve PSD (possibly singular, hence
            still not guaranteed *PD* --- the regularisation layer owns strict
            definiteness). Contrast the per-pair dof-corrected
            :class:`ClusteredCovariance`, which can be indefinite (issue #120).
        """
        if cached_intermediates is not None:
            m, psi_safe, weight_mask, N_j = cached_intermediates
            # The IID estimator uses w_i^2 in the pairwise sum (not
            # the d_ij * w_i product squared), so derive w^2 from the
            # measure weights directly. Cached weight_mask is sufficient
            # for the d_ij * d_ik mask combination.
            weights = measure.weights  # (N,)
            w2 = weights * weights  # (N,)
            # weighted_psi: d_ij * (psi_safe [- m]) with masked-out rows zeroed.
            # weight_mask already encodes d_ij * w_i; to recover the IID closed
            # form (sum_i d_ij * d_ik * w_i^2 * psi_j * psi_k), factor out w_i to
            # get d_ij and multiply back with w^2. When centered, subtract the
            # cached mean moment m; masked cells (psi_safe=0 -> -m_j) are re-zeroed
            # by the mask, so only observed deviations enter.
            mask = measure.mask  # (N, M)
            centered_psi = psi_safe - m[None, :] if self.centered else psi_safe
            weighted_psi = mask * centered_psi  # (N, M)
            numer = jnp.einsum("i,ij,ik->jk", w2, weighted_psi, weighted_psi)
            return _safe_outer_divide(numer, N_j)

        # Pre-sanitise data with the per-column observed-mean sentinel
        # so partial residuals (``log``, ``1/x``, ``sqrt``) cannot
        # introduce NaN/Inf at masked-out cells and poison reverse-mode
        # AD. See :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`
        # and the matching guard in
        # :meth:`EmpiricalMeasure.expectation` for the full rationale.
        x_safe = safe_x_for_psi(measure.x)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
        mask = measure.mask  # (N, M)
        weights = measure.weights  # (N,)

        # NaN-safe: substitute zero at masked-out cells via where(...) so
        # that 0 * NaN does not poison the sum. The standard
        # mask * psi_batch path silently produces NaN at every masked
        # cell whenever the user's psi returns NaN there, e.g., for the
        # "non-holder" rows in a seasonality / IMRS specification.
        mask_bool = mask > 0.0
        psi_safe = jnp.where(mask_bool, psi_batch, 0.0)  # (N, M)

        # Per-coordinate effective sample size N_j = sum_i d_ij * w_i.
        N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)

        # Pairwise overlap numerator: sum_i d_ij * d_ik * w_i^2 * psi_j * psi_k
        # (centered: psi_j -> psi_j - m_j). einsum: i is summed; j, k are kept.
        w2 = weights * weights  # (N,)
        if self.centered:
            m = _mean_moment(mask, weights, psi_safe, N_j)
            centered_psi = psi_safe - m[None, :]
        else:
            centered_psi = psi_safe
        weighted_psi = mask * centered_psi  # (N, M); zero masked-out rows
        numer = jnp.einsum("i,ij,ik->jk", w2, weighted_psi, weighted_psi)

        return _safe_outer_divide(numer, N_j)


__all__ = ["IIDCovariance"]
