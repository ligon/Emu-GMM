"""Empirical (sample-backed) measure with pairwise-overlap missingness.

``EmpiricalMeasure`` exposes the framework's ``Measure`` protocol over a
materialised sample. Per-coordinate observability is tracked by a 0/1
mask matrix so that each moment is integrated against its own
coordinate-specific measure (see ``docs/design.org`` Section 2,
"Polymorphic Implementations / EmpiricalMeasure"). For coordinate
``j`` the expectation is

.. math::
   \\mathbb{E}_{\\mu_X}[\\psi]_j
   \\;=\\;
   \\frac{1}{N_j}\\,\\sum_{i=1}^N d_{ij}\\, w_i\\, \\psi_j(x_i, \\theta),
   \\qquad
   N_j = \\sum_{i=1}^N d_{ij}\\, w_i,

with the same per-coordinate normalisation applied to the Jacobian.

The class itself holds only JAX arrays --- the pandas column-name /
index-name labels are tracked at the estimator level via
:mod:`emu_gmm._internal.labels`. The :meth:`from_pandas` classmethod is
a convenience for constructing the JAX arrays from a
:class:`pandas.DataFrame`; downstream labelling is the caller's
responsibility.

NaN-as-missing semantics
------------------------

The hot path is fundamentally /mask-based/, not NaN-based: per
``docs/design.org`` Section 5, "Rather than relying on floating-point
NaN sentinels that contaminate the automatic differentiation tape, the
missingness layout is captured as a dedicated boolean structure". At
the I/O boundary, however, NaN is the natural sentinel for many
real-data workflows (an asset return is missing for a non-holder; a
seasonal contribution is missing for a household with no contemporary
holdings of asset :math:`j`). To bridge the two, two facilities are
provided:

1. :meth:`from_pandas` infers the per-coordinate mask from
   ``~df.isna()`` when no explicit mask is supplied, and replaces NaN
   cells in the stored ``x`` array with zero so they cannot poison the
   AD tape if a downstream model happens to read them.
2. :meth:`expectation` and :meth:`jacobian` use :func:`jax.numpy.where`
   to /zero out/ rows the mask excludes before multiplication, so a
   model :math:`\\psi` that returns NaN on excluded rows --- a natural
   pattern when the residual is only defined for holders --- still
   produces a finite moment vector and a finite gradient. Without this
   guard the JAX algebra ``0 * NaN = NaN`` poisons the sum.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal.params import flatten_params, unflatten_params
from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a haliax NamedArray wrapper, returning the underlying array.

    Plain arrays / scalars pass through unchanged.
    """
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


def _safe_divide(
    numer: Float[Array, "..."],
    denom: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return ``numer / denom``, falling back to zero where ``denom == 0``.

    Avoids the ``inf`` / ``nan`` propagation that ``numer / 0`` would
    produce when an empty coordinate has no observed rows. Coordinates
    with ``N_j = 0`` are degenerate and the estimator surfaces this via
    ``Diagnostics.N_j``; the residual itself simply records zero.
    """
    safe = jnp.where(denom == 0.0, 1.0, denom)
    out = numer / safe
    return jnp.where(denom == 0.0, jnp.zeros_like(out), out)


@jdc.pytree_dataclass
class EmpiricalMeasure:
    """Sample-backed measure with per-coordinate observability mask.

    Parameters
    ----------
    x : (N, D) jax array
        Observations.
    mask : (N, M) jax array
        Per-coordinate observability indicators (0/1 floats). Entry
        ``mask[i, j]`` is 1 if moment ``j`` is observable for unit
        ``i``, 0 otherwise.
    weights : (N,) jax array
        Per-observation weights, normalised so that they enter the
        moment estimator as ``d_ij * w_i`` summed over observations.
        Defaults to all-ones (unweighted) when the user passes
        ``None`` through :meth:`from_pandas`.
    """

    x: Float[Array, "N D"]
    mask: Float[Array, "N M"]
    weights: Float[Array, " N"]

    def expectation(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, " M"]:
        """Per-coordinate weighted mean of :math:`\\psi` under the mask.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function. May return a plain
            ``(M,)`` array or a :class:`haliax.NamedArray`; the wrapper
            is stripped internally. Cells where ``mask == 0`` may be
            NaN (e.g., a residual that is only defined for "holders"):
            the implementation zeroes them via :func:`jax.numpy.where`
            before multiplication so that the JAX algebra
            ``0 * NaN = NaN`` cannot poison the aggregate.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        m : (M,) jax array
            ``(1 / N_j) * sum_i d_ij * w_i * psi_j(x_i, theta)`` for
            each coordinate ``j``; coordinates with ``N_j = 0`` map to
            zero rather than NaN.
        """

        # Pre-sanitise the data array so that NaN cells (e.g., a
        # non-holder's return) cannot enter the user's psi or its
        # gradient. The "double where" pattern: evaluate psi on a
        # NaN-free surrogate, then mask the result. Without this
        # guard, reverse-mode AD would propagate the NaN through the
        # untaken branch of any jnp.where on the output side
        # (cotangent flow ignores branch selection).
        x_safe = jnp.where(jnp.isnan(self.x), 0.0, self.x)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
        # NaN-safe contraction: substitute zero wherever mask == 0 BEFORE
        # the weight multiplication. Without this guard, NaN at a
        # masked-out cell would propagate (0 * NaN = NaN in JAX).
        mask_bool = self.mask > 0.0  # (N, M)
        psi_safe = jnp.where(mask_bool, psi_batch, 0.0)  # (N, M)
        # Pairwise per-coordinate mass: d_ij * w_i, broadcast across moments.
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        numer = jnp.sum(weight_mask * psi_safe, axis=0)  # (M,)
        denom = jnp.sum(weight_mask, axis=0)  # (M,)
        return _safe_divide(numer, denom)

    def moment_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, "N M"]:
        """Per-observation, mask-weighted moment contributions ``g_i(theta)``.

        Returns the ``(N, M)`` matrix whose ``(i, j)`` entry is

        .. math::
           g_{ij}(\\theta) \\;=\\; d_{ij}\\, w_i\\, \\psi_j(x_i, \\theta),

        i.e. the per-observation contribution that summed-then-normalised
        produces :meth:`expectation`. Rows where ``mask[i, j] == 0`` are
        zeroed; rows where ``mask[i, j] == 1`` carry ``w_i * psi_j``.

        This is the building block downstream callers need for
        bootstrap, K-statistic, and other resampling-based inference
        routines: those procedures resample / reweight at the
        per-observation level and need access to the raw
        ``g_i(\\theta)`` matrix without normalisation. Combined with
        :meth:`jacobian` and a :class:`~emu_gmm.types.CovarianceStrategy`
        applied at ``theta`` (which gives the cluster- or iid-aware
        ``Omega_hat(theta)``), these three primitives match the surface
        that ManifoldGMM's ``MomentRestriction`` exposes for the same
        downstream machinery.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        g : (N, M) jax array
            ``g_ij = d_ij * w_i * psi_j(x_i, theta)``; the masked,
            weighted per-observation moment matrix. No ``1 / N_j``
            normalisation is applied --- the caller decides how to
            aggregate (sum, weighted sum, cluster-sum, bootstrap-resample).

        Notes
        -----
        Like :meth:`expectation` and :meth:`jacobian`, this method
        adopts the "double where" NaN-safe pattern: NaN cells in
        ``self.x`` are zeroed before invoking ``psi``, and masked-out
        ``(i, j)`` cells are zeroed in the output before the weight
        multiplication so that ``0 * NaN = NaN`` cannot poison the
        returned matrix or any downstream reverse-mode AD that
        traverses it.
        """

        # Pre-sanitise x (see :meth:`expectation` for the rationale).
        x_safe = jnp.where(jnp.isnan(self.x), 0.0, self.x)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
        # NaN-safe contraction: substitute zero wherever mask == 0 BEFORE
        # the weight multiplication (mirrors the pattern in
        # :meth:`expectation` and :meth:`jacobian`).
        mask_bool = self.mask > 0.0  # (N, M)
        psi_safe = jnp.where(mask_bool, psi_batch, 0.0)  # (N, M)
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        return weight_mask * psi_safe

    def jacobian(self, psi: StructuralModel, theta: ParamsLike) -> Float[Array, "M K"]:
        """Per-coordinate weighted mean of :math:`\\nabla_\\theta \\psi`.

        Uses :func:`jax.jacfwd` on the flattened ``theta`` (see
        :mod:`emu_gmm._internal.params`) at the per-observation level,
        then applies the same mask / weight aggregation as
        :meth:`expectation`. Returns a plain ``(M, K)`` JAX array.

        Like :meth:`expectation`, the implementation guards the sum
        against NaN gradients at masked-out cells: rows for which
        ``mask[i, j] == 0`` contribute zero to the moment-``j`` Jacobian
        regardless of what ``jacfwd`` produces there.
        """
        flat_theta, treedef = flatten_params(theta)

        # Pre-sanitise x (see :meth:`expectation` for the rationale).
        x_safe = jnp.where(jnp.isnan(self.x), 0.0, self.x)

        def psi_flat(x: Float[Array, " D"], flat: Float[Array, " K"]):
            params = unflatten_params(flat, treedef)
            return _to_plain(psi(x, params))

        def grad_at(x: Float[Array, " D"]) -> Float[Array, "M K"]:
            return jax.jacfwd(lambda flat: psi_flat(x, flat))(flat_theta)

        grad_batch = jax.vmap(grad_at)(x_safe)  # (N, M, K)
        # NaN-safe: zero the gradient at masked-out (i, j) cells before
        # weight multiplication so 0 * NaN cannot poison the (M, K) sum.
        mask_bool = (self.mask > 0.0)[:, :, None]  # (N, M, 1)
        grad_safe = jnp.where(mask_bool, grad_batch, 0.0)  # (N, M, K)
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        numer = jnp.sum(weight_mask[:, :, None] * grad_safe, axis=0)  # (M, K)
        denom = jnp.sum(weight_mask, axis=0)  # (M,)
        return _safe_divide(numer, denom[:, None])

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        weights: Any | None = None,
        mask: Any | None = None,
        *,
        nan_aware: bool = True,
    ) -> "EmpiricalMeasure":
        """Construct an :class:`EmpiricalMeasure` from a pandas DataFrame.

        The data is routed through
        :func:`emu_gmm._internal.labels.normalise_x`,
        :func:`emu_gmm._internal.labels.normalise_weights`, and
        :func:`emu_gmm._internal.labels.normalise_mask` to produce the
        JAX arrays stored on the measure. Pandas column-name / index-
        name labels are tracked at the estimator level; the measure
        itself carries only the arrays.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Observations. Each column becomes a coordinate of ``x``.
            When ``nan_aware`` is true (the default) and no explicit
            ``mask`` is supplied, NaN cells are treated as missing: the
            mask is inferred as ``~df.isna()`` and NaN cells in ``x``
            are replaced with zero so they cannot poison downstream
            JAX arithmetic.
        weights : :class:`pandas.Series` or array-like, optional
            Per-observation weights. Defaults to all-ones.
        mask : :class:`pandas.DataFrame` or array-like of shape ``(N, M)``,
            optional. Per-coordinate observability. When supplied, it
            takes precedence over NaN-inferred missingness. When omitted
            and ``nan_aware`` is true, ``~df.isna()`` is used; otherwise
            an all-ones mask is constructed.
        nan_aware : bool, keyword-only, default True
            When true, NaN cells in ``df`` indicate per-cell
            missingness and drive both the inferred mask (when no
            explicit ``mask`` is given) and the NaN-cleaning of ``x``.
            Set to false to preserve the legacy behaviour of all-ones
            masking and verbatim NaN propagation in ``x``.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`

        Notes
        -----
        The NaN-aware path supports the seasonality / non-holder
        pattern in :mod:`ManifoldGMM`: a per-asset return that is
        defined only for households that hold the asset arrives as NaN
        for non-holders, and the per-coordinate :math:`N_j` should
        reflect the per-asset holder count, not the row count of the
        ambient panel.
        """
        x_arr, _cols, _obs_name = labels_mod.normalise_x(df)
        n = int(x_arr.shape[0])
        w_arr = labels_mod.normalise_weights(weights, n)

        # Determine M and resolve the mask. Precedence is:
        # (1) explicit mask argument, (2) NaN-inferred mask when
        # nan_aware is true, (3) all-ones default.
        if mask is not None:
            # Probe shape of the supplied mask to determine M.
            if hasattr(mask, "shape"):
                m_shape = tuple(int(s) for s in mask.shape)
            else:
                m_shape = jnp.asarray(mask).shape
            if len(m_shape) != 2 or m_shape[0] != n:
                raise ValueError(
                    f"EmpiricalMeasure.from_pandas: mask must have shape "
                    f"(N={n}, M); got {m_shape}"
                )
            m = int(m_shape[1])
            mask_arr = labels_mod.normalise_mask(mask, n, m)
        elif nan_aware:
            # Infer mask from NaN cells: 1 where finite, 0 where NaN.
            finite_mask = jnp.where(jnp.isnan(x_arr), 0.0, 1.0)
            mask_arr = finite_mask.astype(jnp.float32)
            m = int(x_arr.shape[1])
        else:
            m = int(x_arr.shape[1])
            mask_arr = labels_mod.normalise_mask(None, n, m)

        # NaN-clean x so downstream JAX arithmetic / vmap of psi is
        # safe even where the user's psi happens to read masked-out
        # cells. The mask still controls aggregation; this is purely a
        # defensive substitution at the I/O boundary.
        if nan_aware:
            x_arr = jnp.where(jnp.isnan(x_arr), 0.0, x_arr)

        return cls(x=x_arr, mask=mask_arr, weights=w_arr)

    @classmethod
    def from_nan_aware(
        cls,
        x: Any,
        weights: Any | None = None,
    ) -> "EmpiricalMeasure":
        """Construct an :class:`EmpiricalMeasure` from an array containing NaN.

        Convenience wrapper for the NaN-as-missing semantics described
        in the module docstring. The per-coordinate mask is inferred
        from ``~jnp.isnan(x)``, and NaN cells in the stored ``x`` are
        replaced with zero so the hot path is NaN-free. Accepts any
        2-D array-like that :func:`jax.numpy.asarray` can coerce; for
        :class:`pandas.DataFrame` inputs use :meth:`from_pandas`
        instead (which preserves column-label semantics).

        Parameters
        ----------
        x : array-like, shape (N, D)
            Observations with NaN as the missing-cell sentinel.
        weights : array-like of length N, optional
            Per-observation weights. Defaults to all-ones.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`
        """
        x_arr = jnp.asarray(x)
        if x_arr.ndim != 2:
            raise ValueError(
                f"EmpiricalMeasure.from_nan_aware: expected a 2-D array, "
                f"got shape {tuple(x_arr.shape)}"
            )
        n = int(x_arr.shape[0])
        w_arr = labels_mod.normalise_weights(weights, n)
        mask_arr = jnp.where(jnp.isnan(x_arr), 0.0, 1.0).astype(jnp.float32)
        x_clean = jnp.where(jnp.isnan(x_arr), 0.0, x_arr)
        return cls(x=x_clean, mask=mask_arr, weights=w_arr)


__all__ = ["EmpiricalMeasure"]
