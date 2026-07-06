r"""Shared base for the empirical outer-product covariance family.

:class:`~emu_gmm.covariance.iid.IIDCovariance` and
:class:`~emu_gmm.covariance.clustered.ClusteredCovariance` are both sample
variances of the moment estimator formed from outer products of the
per-observation moment contributions --- they differ only in HOW the
contributions are combined: IID sums per-observation :math:`w_i^2` outer
products; Clustered sums the outer products of per-cluster totals, and IID is
the single-PSU (size-one cluster) reduction of Clustered. This base factors out
everything they share:

* the NaN-safe :math:`\psi` evaluation (:func:`safe_x_for_psi`);
* the ``mask`` / per-coordinate :math:`N_j` bookkeeping;
* the optional moment-covariance centering (the ``centered`` knob):
  subtracting the per-coordinate mean moment :math:`m_j` before the outer
  products, so both estimators expose it *uniformly* --- important because IID
  must reduce to size-one-cluster Clustered under centering as well.

Each concrete estimator implements only :meth:`_assemble` (its combination
step). The **design family** (:class:`~emu_gmm.covariance.stratified.StratifiedCovariance`
/ ``DesignAwareCovariance``) has a different shape --- design cells, per-pair dof,
and *intrinsic* Neyman centering (centering is definitional there, not a knob)
--- so it does **not** derive from this base. The universal interface remains the
:class:`~emu_gmm.types.CovarianceStrategy` protocol.

Implementation note: the base holds **no fields** and is **not** a dataclass, so
it cannot perturb a subclass's ``@jdc.pytree_dataclass`` field ordering (the
"non-default field after a default one" footgun). Concrete subclasses are the
pytree dataclasses and declare the shared static ``centered`` flag themselves.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from emu_gmm._internal.nan_safety import safe_x_for_psi
from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a :class:`haliax.NamedArray` wrapper, returning the underlying array."""
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


def _safe_outer_divide(
    numer: Float[Array, "M M"],
    denom_vec: Float[Array, " M"],
) -> Float[Array, "M M"]:
    """Return ``numer / (denom_vec outer denom_vec)`` with zero on degeneracy.

    The ``1 / (N_j N_k)`` normalisation. When ``N_j`` or ``N_k`` is zero the
    entry collapses to zero rather than ``inf`` / ``nan``; the degenerate
    coordinates are surfaced separately through ``Diagnostics.N_j``.
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

    The centre subtracted when ``centered=True``. Degenerate coordinates
    (``N_j == 0``) get ``0`` rather than ``nan`` (already surfaced via
    ``Diagnostics.N_j``). Equals the moment vector the estimator computes, so
    ``centered=True`` subtracts exactly the fitted moment at ``theta``.
    """
    numer_m = jnp.sum(mask * weights[:, None] * psi_safe, axis=0)  # (M,)
    safe_N = jnp.where(N_j == 0.0, 1.0, N_j)
    return jnp.where(N_j == 0.0, 0.0, numer_m / safe_N)


class _OuterProductCovariance:
    """Template base for the outer-product covariance family (see module docstring).

    Subclasses are ``@jdc.pytree_dataclass``, declare their fields (including the
    shared static ``centered`` flag), and implement :meth:`_assemble`.
    """

    #: Declared by each concrete subclass as a static field (default ``False``).
    #: When ``True`` the per-coordinate mean moment is subtracted before the
    #: outer products. A bare annotation here (not a dataclass field on this
    #: non-dataclass base) so the type-checker knows subclasses provide it.
    centered: bool

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
        r"""Construct :math:`V_X(\theta)`: shared preamble, then :meth:`_assemble`.

        Computes the NaN-safe ``psi_safe``, ``mask``, ``weights``, the weighted
        mask ``d_ij w_i``, per-coordinate ``N_j``, and (when ``centered``) the
        mean moment, then hands the masked, optionally-centered contributions to
        the estimator-specific :meth:`_assemble`. ``cached_intermediates``
        (``(m, psi_safe, weight_mask, N_j)`` from
        :meth:`EmpiricalMeasure.expectation_and_contributions`) skips the
        ``vmap(psi)`` pass; ``None`` takes the self-computing path.
        """
        m: Float[Array, " M"] | None
        if cached_intermediates is not None:
            m, psi_safe, weight_mask, N_j = cached_intermediates
            mask = measure.mask
            weights = measure.weights
        else:
            # Pre-sanitise data with the per-column observed-mean sentinel so
            # partial residuals (log, 1/x, sqrt) cannot introduce NaN/Inf at
            # masked-out cells and poison reverse-mode AD (see safe_x_for_psi
            # and EmpiricalMeasure.expectation).
            x_safe = safe_x_for_psi(measure.x)
            psi_batch = jax.vmap(lambda x: _to_plain(psi(x, theta)))(x_safe)  # (N,M)
            mask = measure.mask
            weights = measure.weights
            # NaN-safe: zero psi at masked-out cells before any contraction.
            psi_safe = jnp.where(mask > 0.0, psi_batch, 0.0)
            N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)
            weight_mask = mask * weights[:, None]  # d_ij * w_i
            m = _mean_moment(mask, weights, psi_safe, N_j) if self.centered else None

        # Centering subtracts the mean; masked cells (psi_safe=0 -> -m_j) are
        # re-zeroed by the mask inside _assemble, so only observed deviations
        # enter. Uncentered (default) is bit-for-bit the pre-refactor path.
        if self.centered:
            assert m is not None  # set on both paths when centered (mypy narrowing)
            centered_psi = psi_safe - m[None, :]
        else:
            centered_psi = psi_safe
        return self._assemble(
            centered_psi=centered_psi,
            mask=mask,
            weights=weights,
            weight_mask=weight_mask,
            N_j=N_j,
            measure=measure,
        )

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
        """Combine the (masked, optionally-centered) contributions into ``V_X``.

        Estimator-specific hook. ``centered_psi`` is ``psi_safe`` (or
        ``psi_safe - m``); ``mask`` is ``d_ij``; ``weight_mask`` is ``d_ij w_i``;
        ``N_j`` the per-coordinate effective sample size.
        """
        raise NotImplementedError  # pragma: no cover
