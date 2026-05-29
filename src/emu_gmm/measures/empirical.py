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
   cells in the stored ``x`` array with the per-column mean of the
   *observed* rows of that column (see
   :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`). The mean is a
   strictly stronger guarantee than the previous ``0.0`` constant: it
   lies inside the domain of ``log``, ``1/x``, ``sqrt`` and similar
   partial residuals whenever any observed cell does, and so prevents
   the user's :math:`\\psi` from producing NaN / Inf at masked-out
   cells. The mask still controls aggregation, so the primal value is
   unchanged.
2. :meth:`expectation` and :meth:`jacobian` use :func:`jax.numpy.where`
   to /zero out/ rows the mask excludes before multiplication, so a
   model :math:`\\psi` that returns NaN on excluded rows --- a natural
   pattern when the residual is only defined for holders --- still
   produces a finite moment vector and a finite gradient. Without this
   guard the JAX algebra ``0 * NaN = NaN`` poisons the sum, and
   reverse-mode AD propagates the NaN cotangent through the untaken
   ``where`` branch on the primal side regardless of the output mask
   (see ``docs/reviews/v1x-jax-ad-review.org`` and
   :mod:`emu_gmm._internal.nan_safety`).
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal.nan_safety import safe_x_for_psi
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


def _assert_finite_weights(
    weights: Float[Array, " N"],
    *,
    source: str,
) -> None:
    """Raise if ``weights`` contains NaN or +/-inf.

    The double-where NaN guard in :meth:`EmpiricalMeasure.expectation`
    and :meth:`jacobian` protects ``psi`` against NaN at masked-out
    cells, but it is applied to the residual --- not to the
    ``weights`` vector. The aggregator builds
    ``weight_mask = self.mask * self.weights[:, None]`` and a NaN
    weight propagates into ``weight_mask`` regardless of the mask
    (``mask * NaN = NaN``), poisoning the per-coordinate sum. The
    cheapest defence is to reject non-finite weights at the input
    boundary: real weights (frequency, sampling, inverse-probability)
    are always finite, and a NaN here is almost certainly an upstream
    bug worth surfacing.

    Parameters
    ----------
    weights : (N,) jax array
        The normalised weights vector.
    source : str
        Name of the calling constructor, used to disambiguate the
        error message (e.g. ``"from_pandas"`` vs ``"from_nan_aware"``).
    """
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError(
            f"EmpiricalMeasure.{source}: weights contain non-finite values "
            f"(NaN or +/-inf). A non-finite weight propagates through "
            f"mask * weights and poisons the per-coordinate sum, regardless "
            f"of any per-cell mask. Drop or impute the offending rows before "
            f"constructing the measure."
        )


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

    def expectation_and_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> tuple[
        Float[Array, " M"],
        Float[Array, "N M"],
        Float[Array, "N M"],
        Float[Array, " M"],
    ]:
        """Shared primitive: expectation + intermediates in one vmap pass.

        Returns the per-coordinate mean ``m`` together with the
        ``(N, M)`` mask-safe residual matrix ``psi_safe``, the pairwise
        weight mask ``weight_mask = mask * weights[:, None]``, and the
        per-coordinate effective sample size ``N_j``. This is the
        single source of truth for the empirical hot path: the
        :class:`~emu_gmm.covariance.iid.IIDCovariance` and
        :class:`~emu_gmm.covariance.clustered.ClusteredCovariance`
        strategies accept these as ``cached_intermediates`` and skip
        their own ``vmap(psi)`` pass, eliminating a ~72 MB transient
        rebuild and a redundant doubled ``psi`` evaluation per
        ``residual_fn`` call at typical empirical sizes
        (``N=60k, M=50, float64``).

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        m : (M,) jax array
            The per-coordinate mean
            ``(1 / N_j) * sum_i d_ij * w_i * psi_j(x_i, theta)``.
        psi_safe : (N, M) jax array
            Per-observation residual matrix with masked-out cells
            zeroed via :func:`jax.numpy.where`. NaN-safe under the
            "double where" pattern documented on :meth:`expectation`.
        weight_mask : (N, M) jax array
            ``mask * weights[:, None]``, the pairwise per-coordinate
            mass that combined with ``psi_safe`` reproduces both the
            moment sum and any downstream second-moment construction.
        N_j : (M,) jax array
            Per-coordinate effective sample size ``sum_i d_ij * w_i``,
            the same quantity used by both the numerator normalisation
            and the diagnostics layer.
        """
        # Pre-sanitise the data array so that NaN cells (e.g., a
        # non-holder's return) cannot enter the user's psi or its
        # gradient. The naive "double where" pattern --- replace NaN
        # with a fixed constant (e.g. 0.0), vmap psi, mask the result
        # --- is unsafe under *reverse-mode* AD whenever psi is partial
        # at the chosen constant: ``log(0) = -inf``, ``1 / 0 = inf``,
        # and ``sqrt(-x)`` returns NaN, and reverse-mode AD propagates
        # the cotangent through the untaken ``where`` branch on the
        # primal side. The result is a NaN gradient on a converged
        # solution (see ``docs/reviews/v1x-jax-ad-review.org``).
        # Instead we substitute the per-column observed mean: that
        # value lies inside psi's domain whenever any observed cell of
        # the column does, so log / division / sqrt all stay finite.
        # The output mask still zeroes the contribution to the primal,
        # so the aggregate is unchanged.
        x_safe = safe_x_for_psi(self.x)

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
        N_j = jnp.sum(weight_mask, axis=0)  # (M,)
        m = _safe_divide(numer, N_j)
        return m, psi_safe, weight_mask, N_j

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
        m, _psi_safe, _weight_mask, _N_j = self.expectation_and_contributions(
            psi, theta
        )
        return m

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

        # Pre-sanitise x with the per-column observed-mean sentinel
        # (see :meth:`expectation` for the reverse-mode AD rationale).
        x_safe = safe_x_for_psi(self.x)

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

        # Pre-sanitise x with the per-column observed-mean sentinel
        # (see :meth:`expectation` for the reverse-mode AD rationale).
        x_safe = safe_x_for_psi(self.x)

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
            takes precedence over NaN-inferred missingness, but it is
            an error to combine an explicit mask with a data array
            that still contains NaN (see "Raises" below). When omitted
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

        Raises
        ------
        ValueError
            If ``nan_aware`` is true, ``mask`` is supplied, and ``df``
            still contains NaN cells. The combination is ambiguous: the
            user's mask might mark a NaN cell observable, in which case
            silently rewriting it to zero would bias :math:`N_j` and
            the moment sum. Drop the explicit mask (let NaN-inference
            run), scrub NaN in ``df`` before calling, or pass
            ``nan_aware=False`` to opt back into NaN-passthrough.

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
        _assert_finite_weights(w_arr, source="from_pandas")

        # Determine M and resolve the mask. Precedence is:
        # (1) explicit mask argument, (2) NaN-inferred mask when
        # nan_aware is true, (3) all-ones default.
        if mask is not None:
            # Probe shape of the supplied mask to determine M. NamedArray
            # exposes ``shape`` as a {name: size} dict, so handle it
            # explicitly before the generic ``hasattr(mask, "shape")``
            # branch (which assumes a tuple-like).
            if isinstance(mask, ha.NamedArray):
                m_shape = tuple(int(s) for s in mask.array.shape)
            elif hasattr(mask, "shape"):
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
        #
        # The sentinel used is the per-column mean of the *observed*
        # rows of that column (see
        # :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`); this
        # lies inside the domain of partial residuals like ``log``,
        # ``1/x``, and ``sqrt`` whenever any observed cell does, and
        # so keeps reverse-mode AD well-defined at masked-out cells.
        # Substituting the previous fixed constant (``0.0``) silently
        # produced NaN cotangents for such residuals even though the
        # mask zeroed the primal contribution.
        #
        # Gating: scrub x only when ``nan_aware`` is true AND the mask
        # was inferred from NaN. If the user supplied an explicit mask
        # alongside NaN-laden x, silently rewriting NaN cells would
        # turn an unobserved value into a "real" observation at any
        # (i, j) the user marked observable, biasing N_j and the
        # moment sum. The conflict is almost always user error, so
        # raise loudly instead of guessing.
        if nan_aware and mask is None:
            x_arr = safe_x_for_psi(x_arr)
        elif nan_aware and mask is not None and bool(jnp.any(jnp.isnan(x_arr))):
            raise ValueError(
                "EmpiricalMeasure.from_pandas: an explicit mask was supplied "
                "alongside NaN values in the data. Silently rewriting NaN to "
                "zero would bias the per-coordinate sums at cells the mask "
                "marks observable. Either (a) drop the mask argument so "
                "nan_aware can infer it from ~df.isna(), or (b) scrub NaN in "
                "the data before calling from_pandas (e.g. df.fillna(0) or "
                "df.dropna()), or (c) pass nan_aware=False to keep the legacy "
                "all-ones-mask / NaN-passthrough behaviour."
            )

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
        _assert_finite_weights(w_arr, source="from_nan_aware")
        mask_arr = jnp.where(jnp.isnan(x_arr), 0.0, 1.0).astype(jnp.float32)
        # Substitute the per-column observed-mean sentinel at NaN cells
        # rather than ``0.0`` so that partial residuals like
        # ``log(x[0])`` or ``1.0 / x[1]`` cannot produce a NaN cotangent
        # at masked-out cells under reverse-mode AD (see
        # :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`).
        x_clean = safe_x_for_psi(x_arr)
        return cls(x=x_clean, mask=mask_arr, weights=w_arr)


__all__ = ["EmpiricalMeasure"]
