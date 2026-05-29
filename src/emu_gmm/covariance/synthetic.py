"""Monte Carlo variance for simulator-backed estimators.

``SyntheticCovariance`` computes the variance of the simulated moment
/estimator/ :math:`\\bar m_{\\mathrm{sim}}(\\theta)`. Consistently with the
framework's convention (see ``docs/design.org`` Section 1 and Section 5,
"Architectural Core Highlights"), the returned matrix is

.. math::
   V_{\\mathrm{sim}}(\\theta) \\;=\\;
   \\frac{1}{n_{\\mathrm{sim}}^2}
   \\sum_{s=1}^{n_{\\mathrm{sim}}}
   (\\psi_s - \\bar m_{\\mathrm{sim}})(\\psi_s - \\bar m_{\\mathrm{sim}})^\\top,

which scales as :math:`\\operatorname{Var}[\\bar m_{\\mathrm{sim}}]` rather
than as the per-draw variance. When combined additively with an
empirical covariance in an SMM matching estimator (v2), the result
automatically picks up the canonical :math:`(1 + 1/S)` inflation factor
with :math:`S = n_{\\mathrm{sim}}/N`.

This is the biased / maximum-likelihood form (denominator :math:`n^2`)
rather than the unbiased :math:`n(n-1)` form, matching the framework's
empirical-side conventions.
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


@jdc.pytree_dataclass
class SyntheticCovariance:
    """Monte Carlo variance of the simulated moment estimator.

    No configurable state in v1: the strategy reads ``n_sim``, the
    frozen ``key``, and the ``sampler`` from the supplied
    :class:`SyntheticMeasure` and evaluates the variance of the per-draw
    residual averaged across draws.
    """

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[Float[Array, " M"], Float[Array, "n_sim M"]] | None
        ) = None,
    ) -> Float[Array, "M M"]:
        """Construct :math:`V_{\\mathrm{sim}}(\\theta)`.

        Parameters
        ----------
        psi
            Structural model. ``psi(x, theta)`` returns an (M,) array or
            a haliax NamedArray with a Moments axis.
        theta
            User parameter dataclass.
        measure
            A :class:`SyntheticMeasure` instance whose ``_draws`` method
            this strategy calls.
        cached_intermediates : optional 2-tuple
            ``(m, psi_batch)`` produced by a previous call to
            :meth:`SyntheticMeasure.moments_and_contributions`. When
            supplied, this routine reuses the cached ``psi_batch`` and
            skips a redundant ``_draws`` + ``vmap(psi)`` pass --- the
            SMM-dedup consolidation described in
            ``docs/reviews/v1x-performance-review.org`` finding #5.
            Back-compat: when ``None``, falls through to the
            self-computing path.

        Returns
        -------
        V : (M, M) jax array
            Symmetric PSD by construction.
        """
        if cached_intermediates is not None:
            _m, psi_batch = cached_intermediates
        else:
            x_batch = measure._draws(theta)

            def psi_at(x):
                return _to_plain(psi(x, theta))

            psi_batch = jax.vmap(psi_at)(x_batch)  # (n_sim, M)

        n = psi_batch.shape[0]
        m_bar = jnp.mean(psi_batch, axis=0)
        centered = psi_batch - m_bar
        # einsum: sum over the draw axis, leaving (M, M).
        return jnp.einsum("ij,ik->jk", centered, centered) / (n * n)


__all__ = ["SyntheticCovariance"]
