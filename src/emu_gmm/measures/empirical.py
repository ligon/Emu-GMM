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
holdings of asset :math:`j`). Non-finiteness is handled uniformly on
``isfinite``: a ``+/-inf`` cell is treated exactly like NaN (missing),
because a surviving ``inf`` poisons :math:`\\psi` and reverse-mode AD
the same way NaN does. To bridge the two, two facilities are provided:

1. :meth:`from_pandas` infers the per-coordinate mask from cell
   finiteness (``~df.isna()`` extended to ``+/-inf``) when no explicit
   mask is supplied, and replaces non-finite cells in the stored ``x``
   array with the per-column mean of the *observed* rows of that column
   (see :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`). The mean
   is a strictly stronger guarantee than the previous ``0.0`` constant:
   it lies inside the domain of ``log``, ``1/x``, ``sqrt`` and similar
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
from emu_gmm._internal.params import flatten_params_for_ad, unflatten_params
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


def _check_mask_psi_compatibility(
    mask_shape: tuple[int, ...],
    psi_shape: tuple[int, ...],
) -> None:
    """Raise a clear error if ``mask`` and ``psi(x_i)`` do not align.

    The aggregator combines a stored ``(N, M_mask)`` mask with a
    ``(N, M_psi)`` per-observation residual matrix via
    ``jnp.where(mask, psi_batch, 0.0)`` and ``mask * weights[:, None] *
    psi_batch``. Three failure classes are caught here, all at JAX
    trace-time (shapes are static during tracing):

    1. **psi returns a 0-d scalar per observation.** ``vmap`` then
       yields a 1-D ``(N,)`` batch, and JAX raises *no* error:
       ``jnp.where(mask_bool, psi_batch, 0.0)`` silently broadcasts the
       ``(N,)`` batch against the ``(N, 1)`` mask into an ``(N, N)``
       matrix, and ``expectation`` returns an ``(N,)`` vector of
       per-observation residuals as "moments". For ``K = 1`` the whole
       pipeline runs end-to-end on garbage with no error, so deferring
       to JAX here would let the defect through silently.
    2. **Any other non-2-D psi batch** (e.g. psi returning a 2-D block
       per observation, giving a 3-D batch): rejected with a generic
       message rather than deferred.
    3. **M mismatch** between a 2-D mask and a 2-D psi batch. The
       typical cause is constructing the measure via
       :meth:`EmpiricalMeasure.from_nan_aware` without passing ``M=``
       when the user's :math:`\\psi` returns ``M != D`` moments (e.g.
       Hansen-Singleton, where ``x`` has three columns but the moment
       is scalar); the message points the user at the ``M=`` kwarg.

    Parameters
    ----------
    mask_shape : tuple of int
        Shape of ``self.mask``; expected ``(N, M_mask)``.
    psi_shape : tuple of int
        Batch shape of psi's per-observation output; expected
        ``(N, M_psi)``. The Jacobian paths pass ``grad_batch.shape[:-1]``
        (the batch shape with the trailing parameter axis removed), so a
        0-d-scalar psi surfaces as a 1-D shape there too.

    Raises
    ------
    ValueError
        If psi's batch shape is not 2-D, or if the moment dimensions
        disagree.
    """
    if len(psi_shape) == 1 and len(mask_shape) == 2:
        raise ValueError(
            f"EmpiricalMeasure: psi must return a 1-D array of shape (M,) "
            f"per observation, but it returned a 0-d scalar (the vmapped "
            f"per-observation batch has 1-D shape {psi_shape} instead of "
            f"(N, M); the mask has shape {mask_shape}). JAX raises no "
            f"error for this: the scalar batch silently broadcasts "
            f"against the (N, M) mask (to (N, N) when M == 1) and the "
            f"'moments' become per-observation residuals -- garbage that "
            f"runs end-to-end for K = 1. Wrap psi's return value so that "
            f"a single moment has shape (1,): e.g. "
            f"'return jnp.array([expr])' or 'return jnp.atleast_1d(expr)'."
        )
    if len(psi_shape) != 2:
        raise ValueError(
            f"EmpiricalMeasure: expected psi's vmapped per-observation "
            f"batch to have shape (N, M) -- i.e. psi returns a 1-D array "
            f"of shape (M,) per observation -- but got shape {psi_shape}. "
            f"Return a 1-D (M,) array from psi (flatten any "
            f"multi-dimensional moment block, e.g. with jnp.ravel)."
        )
    if len(mask_shape) != 2:
        # The constructors all enforce a 2-D (N, M) mask; a hand-built
        # measure with an exotic mask falls through to JAX's own error.
        return
    n_mask, m_mask = mask_shape
    n_psi, m_psi = psi_shape
    if m_mask != m_psi:
        raise ValueError(
            f"EmpiricalMeasure: mask has shape (N={n_mask}, M={m_mask}) but "
            f"psi returns shape (N={n_psi}, M={m_psi}); the moment "
            f"dimensions disagree. The typical cause is constructing the "
            f"measure via EmpiricalMeasure.from_nan_aware(x) without "
            f"passing M=, which defaults to a (N, D)-shaped mask where "
            f"D is the number of columns in x. When psi returns M != D "
            f"moments (e.g. a Hansen-Singleton-style scalar moment built "
            f"from three data columns), pass M= explicitly: "
            f"EmpiricalMeasure.from_nan_aware(x, M={m_psi})."
        )


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

    Notes
    -----
    This is an *eager* value check on concrete inputs. When the array
    is a JAX tracer (construction inside ``jit`` / ``vmap``, possible
    via :meth:`EmpiricalMeasure.from_arrays`), the check is skipped so
    tracing still works --- a traced value cannot be inspected without
    aborting the trace.
    """
    if isinstance(weights, jax.core.Tracer):
        return
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError(
            f"EmpiricalMeasure.{source}: weights contain non-finite values "
            f"(NaN or +/-inf). A non-finite weight propagates through "
            f"mask * weights and poisons the per-coordinate sum, regardless "
            f"of any per-cell mask. Drop or impute the offending rows before "
            f"constructing the measure."
        )


def _assert_mask_columns_supported(
    mask: Float[Array, "N M"],
    *,
    source: str,
) -> None:
    """Raise if some mask column has no supported observation at all.

    An all-zero mask column means :math:`N_j = 0` for that moment: the
    moment mean is degenerate (``_safe_divide`` records zero), and every
    downstream statistic that touches coordinate ``j`` --- ``V_X``, the
    criterion, ``J_stat``, ``Sigma_theta`` --- is undefined, so the fit
    exits as a silent all-NaN result. A dead moment is always a
    construction-time defect (a fully-missing column, a mis-built
    mask), so surface it loudly at the input boundary with the dead
    indices named.

    Parameters
    ----------
    mask : (N, M) jax array
        The resolved 0/1 observability mask.
    source : str
        Name of the calling constructor, for the error message.

    Notes
    -----
    Eager value check; skipped when ``mask`` is a JAX tracer (see
    :func:`_assert_finite_weights`).
    """
    if isinstance(mask, jax.core.Tracer):
        return
    support = jnp.sum(mask > 0.0, axis=0)  # (M,)
    dead = [int(j) for j in jnp.nonzero(support == 0)[0]]
    if dead:
        raise ValueError(
            f"EmpiricalMeasure.{source}: mask column(s) {dead} have no "
            f"supported observations (every entry is zero), so N_j = 0 "
            f"for those moments and every statistic that touches them "
            f"(moment mean, V_X, J_stat, Sigma_theta) is undefined -- "
            f"the fit would exit as a silent all-NaN result. Drop the "
            f"dead moment(s) or fix the mask / missingness pattern "
            f"before constructing the measure."
        )


def _assert_complete_data_under_mask(
    x: Float[Array, "N D"],
    mask: Float[Array, "N M"],
    *,
    source: str,
) -> None:
    """Raise if a non-finite ``x`` cell sits at a mask-ON position.

    :meth:`EmpiricalMeasure.from_arrays` documents that it *asserts
    complete data* rather than inferring missingness from NaN; this is
    the enforcement. Cells the mask turns OFF may hold anything (the
    hot path scrubs them via
    :func:`emu_gmm._internal.nan_safety.safe_x_for_psi` before ``psi``
    ever sees them), but a non-finite cell at a mask-ON position would
    either poison the moment sum or --- worse, after the hot-path
    scrub --- be silently rewritten to the column mean and enter the
    sum as a fabricated observation, biasing ``N_j`` and the moment
    value.

    Two mask layouts are handled:

    - ``mask.shape == x.shape`` (``M == D``): cell-aligned check ---
      ``x[i, j]`` must be finite wherever ``mask[i, j]`` is ON.
    - otherwise (``M != D``): ``psi`` reads the full data row to build
      each moment, so every row supported by *any* moment must be fully
      finite.

    Parameters
    ----------
    x : (N, D) jax array
        The stored observation matrix.
    mask : (N, M) jax array
        The resolved 0/1 observability mask.
    source : str
        Name of the calling constructor, for the error message.

    Notes
    -----
    Eager value check; skipped when either input is a JAX tracer (see
    :func:`_assert_finite_weights`).
    """
    if isinstance(x, jax.core.Tracer) or isinstance(mask, jax.core.Tracer):
        return
    finite = jnp.isfinite(x)  # (N, D)
    if mask.shape == x.shape:
        bad_rows = jnp.any(~finite & (mask > 0.0), axis=1)  # (N,)
    else:
        supported = jnp.any(mask > 0.0, axis=1)  # (N,)
        bad_rows = supported & ~jnp.all(finite, axis=1)  # (N,)
    if bool(jnp.any(bad_rows)):
        rows = [int(i) for i in jnp.nonzero(bad_rows)[0][:10]]
        raise ValueError(
            f"EmpiricalMeasure.{source}: x contains non-finite cells "
            f"(NaN or +/-inf) at mask-ON positions (rows {rows}). "
            f"{source} asserts complete data: every cell the mask marks "
            f"observable must be finite. Use from_nan_aware / "
            f"from_pandas for NaN-as-missing semantics, or mask the "
            f"offending rows/cells off."
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
        # Trace-time shape guard: the mask must broadcast against psi's
        # output. The typical failure mode is constructing the measure
        # via ``EmpiricalMeasure.from_nan_aware(x)`` (which infers a
        # ``(N, D)`` mask from x's NaN pattern) when the user's psi
        # returns M != D moments -- the Hansen-Singleton case where x
        # carries ``(c_t, c_{t+1}, r_{t+1})`` and psi is scalar. The
        # downstream ``jnp.where(mask, psi_batch, 0.0)`` then raises a
        # generic JAX broadcast error that does not point back to the
        # mask layout. Catch it here with a helpful pointer to the
        # ``M=`` kwarg.
        _check_mask_psi_compatibility(self.mask.shape, psi_batch.shape)
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
        # Same trace-time shape check as :meth:`expectation_and_contributions`.
        _check_mask_psi_compatibility(self.mask.shape, psi_batch.shape)
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

        Manifold parameter trees (non-scalar :class:`ManifoldLeaf`
        blocks) are flattened to their AMBIENT coordinates via
        :func:`emu_gmm._internal.params.flatten_params_for_ad`, so ``K``
        is the total ambient dimension and gauge directions appear as an
        exact nullspace of the result (consumed gauge-aware by the
        K-statistic, #41). All-scalar trees take the v1 flatten verbatim.
        """
        flat_theta, treedef, mspec = flatten_params_for_ad(theta)

        # Pre-sanitise x with the per-column observed-mean sentinel
        # (see :meth:`expectation` for the reverse-mode AD rationale).
        x_safe = safe_x_for_psi(self.x)

        def psi_flat(x: Float[Array, " D"], flat: Float[Array, " K"]):
            params = unflatten_params(flat, treedef, manifold_spec=mspec)
            return _to_plain(psi(x, params))

        def grad_at(x: Float[Array, " D"]) -> Float[Array, "M K"]:
            return jax.jacfwd(lambda flat: psi_flat(x, flat))(flat_theta)

        grad_batch = jax.vmap(grad_at)(x_safe)  # (N, M, K)
        # Trace-time shape check; surface the same helpful message as
        # :meth:`expectation_and_contributions` if mask vs psi disagree.
        # Slice off the trailing parameter axis (K) rather than taking
        # the leading two axes: a psi that returns a 0-d scalar yields a
        # 2-D (N, K) grad batch whose ``shape[:2]`` masquerades as a
        # valid (N, M) pair, whereas ``shape[:-1] == (N,)`` exposes the
        # missing moment axis to the scalar-return check.
        _check_mask_psi_compatibility(self.mask.shape, grad_batch.shape[:-1])
        # NaN-safe: zero the gradient at masked-out (i, j) cells before
        # weight multiplication so 0 * NaN cannot poison the (M, K) sum.
        mask_bool = (self.mask > 0.0)[:, :, None]  # (N, M, 1)
        grad_safe = jnp.where(mask_bool, grad_batch, 0.0)  # (N, M, K)
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        numer = jnp.sum(weight_mask[:, :, None] * grad_safe, axis=0)  # (M, K)
        denom = jnp.sum(weight_mask, axis=0)  # (M,)
        return _safe_divide(numer, denom[:, None])

    def jacobian_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, "N M K"]:
        """Per-observation, mask-weighted Jacobian contributions ``D_i(theta)``.

        Returns the ``(N, M, K)`` tensor whose ``(i, j, k)`` entry is

        .. math::
           D_{ijk}(\\theta) \\;=\\; d_{ij}\\, w_i\\, \\partial_{\\theta_k} \\psi_j(x_i, \\theta),

        i.e. the per-observation contribution to the moment Jacobian
        before the ``1 / N_j`` normalisation. Combined with
        :meth:`moment_contributions`, this is the building block the
        Kleibergen K-statistic uses to estimate
        :math:`\\Sigma_{G_j, m}`, the cross-covariance between the
        :math:`j`-th column of the moment Jacobian and the moment vector
        (Kleibergen 2005 eq. 8). Bootstrap and other resampling routines
        consume the same primitive.

        The NaN-safe pattern mirrors :meth:`jacobian` and
        :meth:`moment_contributions`: NaN cells in ``self.x`` are scrubbed
        before invoking ``psi``, and gradients at masked-out ``(i, j)``
        cells are zeroed before the weight multiplication so the
        ``0 * NaN`` algebra cannot poison the result.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        D : (N, M, K) jax array
            ``D[i, j, k] = d_ij * w_i * (d psi_j / d theta_k)(x_i, theta)``.
            No ``1 / N_j`` normalisation is applied. ``K`` is the total
            ambient dimension for manifold trees (see :meth:`jacobian`).
        """
        flat_theta, treedef, mspec = flatten_params_for_ad(theta)

        # Pre-sanitise x with the per-column observed-mean sentinel
        # (see :meth:`expectation` for the reverse-mode AD rationale).
        # A fixed 0.0 sentinel here (the prior behaviour) sits outside
        # the domain of partial residuals like ``log`` / ``1/x``; the
        # resulting inf/NaN per-row Jacobian cells are zeroed in the
        # primal by the output mask below, but reverse-mode AD through
        # this method still produces ``0 * inf == NaN`` cotangents.
        x_safe = safe_x_for_psi(self.x)

        def psi_flat(x: Float[Array, " D"], flat: Float[Array, " K"]):
            params = unflatten_params(flat, treedef, manifold_spec=mspec)
            return _to_plain(psi(x, params))

        def grad_at(x: Float[Array, " D"]) -> Float[Array, "M K"]:
            return jax.jacfwd(lambda flat: psi_flat(x, flat))(flat_theta)

        grad_batch = jax.vmap(grad_at)(x_safe)  # (N, M, K)
        # Same trace-time shape check as the other methods; slice off
        # the trailing K axis (not ``shape[:2]``) so a 0-d-scalar psi's
        # 2-D (N, K) grad batch is caught by the scalar-return check
        # (see :meth:`jacobian`).
        _check_mask_psi_compatibility(self.mask.shape, grad_batch.shape[:-1])
        mask_bool = (self.mask > 0.0)[:, :, None]  # (N, M, 1)
        grad_safe = jnp.where(mask_bool, grad_batch, 0.0)  # (N, M, K)
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        return weight_mask[:, :, None] * grad_safe  # (N, M, K)

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
            ``mask`` is supplied, non-finite cells (NaN or ``+/-inf``)
            are treated as missing: the mask is 1 exactly where the
            cell is finite, and non-finite cells in ``x`` are replaced
            with the per-column mean of the observed rows
            (see :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`).
        weights : :class:`pandas.Series` or array-like, optional
            Per-observation weights. Defaults to all-ones.
        mask : :class:`pandas.DataFrame` or array-like of shape ``(N, M)``,
            optional. Per-coordinate observability. When supplied, it
            takes precedence over NaN-inferred missingness, but it is
            an error to combine an explicit mask with a data array
            that still contains non-finite values (see "Raises"
            below). When omitted and ``nan_aware`` is true, per-cell
            finiteness is used; otherwise an all-ones mask is
            constructed.
        nan_aware : bool, keyword-only, default True
            When true, non-finite cells in ``df`` indicate per-cell
            missingness and drive both the inferred mask (when no
            explicit ``mask`` is given) and the cleaning of ``x``.
            Set to false to preserve the legacy behaviour of all-ones
            masking and verbatim NaN / inf propagation in ``x``.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`

        Raises
        ------
        ValueError
            If ``nan_aware`` is true, ``mask`` is supplied, and ``df``
            still contains non-finite cells. The combination is
            ambiguous: the user's mask might mark a non-finite cell
            observable, in which case silently rewriting it to a
            sentinel would bias :math:`N_j` and the moment sum. Drop
            the explicit mask (let finiteness-inference run), scrub
            NaN / inf in ``df`` before calling, or pass
            ``nan_aware=False`` to opt back into passthrough.

            Also raised when some column of the resolved mask has no
            supported observation at all (``N_j = 0``): a dead moment
            makes every downstream statistic that touches it undefined,
            so it is rejected here with the dead indices named.

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
            # Infer mask from cell finiteness: 1 where finite, 0 where
            # NaN or +/-inf. Both are "missing" -- an inf cell left
            # mask-ON would be silently rewritten by safe_x_for_psi
            # below, fabricating an observation.
            mask_arr = jnp.isfinite(x_arr).astype(jnp.float32)
            m = int(x_arr.shape[1])
        else:
            m = int(x_arr.shape[1])
            mask_arr = labels_mod.normalise_mask(None, n, m)

        # Clean non-finite cells in x so downstream JAX arithmetic /
        # vmap of psi is safe even where the user's psi happens to read
        # masked-out cells. The mask still controls aggregation; this
        # is purely a defensive substitution at the I/O boundary.
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
        # was inferred from finiteness. If the user supplied an
        # explicit mask alongside non-finite x, silently rewriting the
        # non-finite cells would turn an unobserved value into a "real"
        # observation at any (i, j) the user marked observable, biasing
        # N_j and the moment sum. The conflict is almost always user
        # error, so raise loudly instead of guessing. The check is on
        # isfinite, not isnan: an inf cell is exactly as biasing, since
        # the hot path's safe_x_for_psi pass now rewrites it too.
        if nan_aware and mask is None:
            x_arr = safe_x_for_psi(x_arr)
        elif nan_aware and mask is not None and not bool(jnp.all(jnp.isfinite(x_arr))):
            raise ValueError(
                "EmpiricalMeasure.from_pandas: an explicit mask was supplied "
                "alongside NaN or infinite values in the data. Silently "
                "rewriting non-finite cells to a sentinel would bias the "
                "per-coordinate sums at cells the mask marks observable. "
                "Either (a) drop the mask argument so nan_aware can infer it "
                "from the finite cells, or (b) scrub NaN / inf in the data "
                "before calling from_pandas (e.g. df.fillna(0) or "
                "df.dropna()), or (c) pass nan_aware=False to keep the legacy "
                "all-ones-mask / passthrough behaviour."
            )

        _assert_mask_columns_supported(mask_arr, source="from_pandas")
        return cls(x=x_arr, mask=mask_arr, weights=w_arr)

    @classmethod
    def from_nan_aware(
        cls,
        x: Any,
        weights: Any | None = None,
        *,
        M: int | None = None,
    ) -> "EmpiricalMeasure":
        """Construct an :class:`EmpiricalMeasure` from an array with missing cells.

        Convenience wrapper for the NaN-as-missing semantics described
        in the module docstring; ``+/-inf`` cells are treated exactly
        like NaN. Non-finite cells in the stored ``x`` are replaced
        with the per-column observed mean so the hot path is finite,
        and a per-coordinate mask is inferred from
        ``jnp.isfinite(x)``. Accepts any 2-D array-like that
        :func:`jax.numpy.asarray` can coerce; for
        :class:`pandas.DataFrame` inputs use :meth:`from_pandas`
        instead (which preserves column-label semantics).

        The user's :math:`\\psi(x_i, \\theta)` returns an :math:`M`-vector
        of moments per observation. The number of moments :math:`M`
        is in general *not* the number of data columns :math:`D`. The
        canonical example is a Hansen--Singleton-style Euler equation
        where ``x`` carries ``(c_t, c_{t+1}, r_{t+1})`` (so :math:`D=3`)
        but :math:`\\psi` is a scalar (so :math:`M=1`). The internally
        stored ``mask`` array must have shape :math:`(N, M)`, matching
        the per-observation residual layout; mismatched broadcast at
        ``expectation`` time surfaces as an opaque JAX shape error from
        deep in the hot path. The ``M=`` keyword exists to make the
        moment count explicit at construction time.

        Parameters
        ----------
        x : array-like, shape (N, D)
            Observations with NaN (or ``+/-inf``) as the missing-cell
            sentinel.
        weights : array-like of length N, optional
            Per-observation weights. Defaults to all-ones.
        M : int, keyword-only, optional
            Number of moments returned by the user's :math:`\\psi`. When
            supplied, the constructed mask has shape ``(N, M)``: row
            :math:`i` contributes to every moment iff *every* observed
            column of ``x[i, :]`` is finite (i.e. the row is fully
            observed; any non-finite cell in ``x[i, :]`` knocks the
            entire row out of every moment's sum). This matches the
            typical structural model where each scalar moment is a
            function of the full row, so a single missing component
            makes the row unusable for any moment.

            When ``None`` (the default), the mask has shape ``(N, D)``,
            i.e. the legacy column-wise behaviour: row :math:`i`
            contributes to moment :math:`j` iff ``x[i, j]`` is finite.
            That default is correct only when :math:`M = D` *and* the
            mapping from column :math:`d` to moment :math:`d` is the
            identity, which is rarely the case for non-trivial
            structural models. Prefer to pass ``M=`` explicitly.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`

        Raises
        ------
        ValueError
            If ``x`` is not 2-D, or if ``M`` is supplied as a
            non-positive integer. Also raised when some column of the
            inferred mask has no supported observation at all
            (``N_j = 0``): a dead moment makes every downstream
            statistic that touches it undefined, so it is rejected
            here with the dead indices named.

        Examples
        --------
        Hansen--Singleton Euler equation with a single moment built from
        three data columns:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from emu_gmm.measures.empirical import EmpiricalMeasure
        >>> # 100 observations of (c_t, c_{t+1}, r_{t+1}); D = 3.
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(100, 3))
        >>> # psi returns one moment, so M = 1.
        >>> meas = EmpiricalMeasure.from_nan_aware(x, M=1)
        >>> meas.mask.shape
        (100, 1)
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

        if M is None:
            # Legacy behaviour: per-column finiteness mask, shape
            # (N, D). Uses isfinite -- not isnan -- so +/-inf cells are
            # masked out exactly like the M= branch below; an inf left
            # mask-ON would be silently rewritten by safe_x_for_psi,
            # fabricating an observation. This branch is correct only
            # when the user's psi returns M == D moments and the
            # column-to-moment mapping is the identity. The ``mask``
            # lookup is silently broadcast against psi at expectation
            # time, so a (N, D) mask with M != D produces an opaque
            # shape error from inside the hot path -- pass M= explicitly
            # to avoid that trap.
            mask_arr = jnp.isfinite(x_arr).astype(jnp.float32)
        else:
            if not isinstance(M, int) or M <= 0:
                raise ValueError(
                    f"EmpiricalMeasure.from_nan_aware: M must be a positive "
                    f"integer, got {M!r}"
                )
            # Row-wise missingness: a row contributes to every moment iff
            # every column of x at that row is finite. A single NaN in
            # the row knocks the entire row out of all M moments, which
            # is the right semantics whenever psi reads multiple columns
            # to produce one moment (e.g. Hansen-Singleton's
            # u'(c_{t+1}) / u'(c_t) * r_{t+1} reads three columns to
            # build one scalar).
            row_finite = jnp.all(jnp.isfinite(x_arr), axis=1)  # (N,)
            mask_arr = jnp.broadcast_to(row_finite[:, None], (n, M)).astype(jnp.float32)

        # Substitute the per-column observed-mean sentinel at
        # non-finite cells rather than ``0.0`` so that partial
        # residuals like ``log(x[0])`` or ``1.0 / x[1]`` cannot produce
        # a NaN cotangent at masked-out cells under reverse-mode AD
        # (see :func:`emu_gmm._internal.nan_safety.safe_x_for_psi`).
        x_clean = safe_x_for_psi(x_arr)
        _assert_mask_columns_supported(mask_arr, source="from_nan_aware")
        return cls(x=x_clean, mask=mask_arr, weights=w_arr)

    @classmethod
    def from_arrays(
        cls,
        x: Any,
        *,
        M: int | None = None,
        mask: Any = None,
        weights: Any = None,
    ) -> "EmpiricalMeasure":
        """Build a complete-data measure from plain arrays, defaulting the mask.

        The zero-boilerplate constructor for the common fully-observed case:
        pass just ``x`` and the mask defaults to all-ones of shape ``(N, M)``
        and the weights to all-ones, so the caller need not hand-build
        ``jnp.ones((N, M))`` / ``jnp.ones(N)``.

        Unlike :meth:`from_pandas` / :meth:`from_nan_aware`, this does **not**
        infer missingness from NaN --- it asserts complete data. Use the
        NaN-aware constructors when cells are missing.

        Parameters
        ----------
        x : array-like, shape (N, D)
            Observations. Coerced to ``float64``.
        M : int, keyword-only, optional
            Number of moments returned by ``psi``. The default mask has shape
            ``(N, M)``; ``M`` defaults to the data width ``D`` (``x.shape[1]``).
            Pass ``M=`` when ``M != D`` --- e.g. a Hansen--Singleton Euler
            equation where ``x`` is ``(N, 3)`` but ``psi`` is scalar
            (``M=1``). Ignored when ``mask`` is given explicitly.
        mask : array-like, shape (N, M), keyword-only, optional
            Override the all-ones default (e.g. a hand-built observability
            pattern with no NaN in ``x``).
        weights : array-like of length N, keyword-only, optional
            Override the all-ones default.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`

        Raises
        ------
        ValueError
            If ``weights`` is not a 1-D length-``N`` vector of finite
            values; if an explicit ``mask`` is not 2-D with ``N`` rows;
            if ``x`` holds non-finite cells (NaN or ``+/-inf``) at
            mask-ON positions (this constructor *asserts* complete
            data --- use the NaN-aware constructors for missingness);
            or if some mask column has no supported observation at all
            (``N_j = 0``, a dead moment).

        Notes
        -----
        Shape checks are static and always enforced (they work under
        ``jit`` / ``vmap`` tracing). The value-dependent checks
        (finite weights, complete data, dead mask columns) are eager
        and are skipped when the corresponding input is a JAX tracer,
        so traced construction still traces.
        """
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        n = x_arr.shape[0]
        m = M if M is not None else x_arr.shape[1]
        if mask is None:
            mask_arr = jnp.ones((n, m), dtype=jnp.float64)
        else:
            mask_arr = jnp.asarray(mask, dtype=jnp.float64)
            if mask_arr.ndim != 2 or mask_arr.shape[0] != n:
                raise ValueError(
                    f"EmpiricalMeasure.from_arrays: mask must be 2-D with "
                    f"shape (N={n}, M); got shape {tuple(mask_arr.shape)}"
                )
        if weights is None:
            w_arr = jnp.ones((n,), dtype=jnp.float64)
        else:
            w_arr = jnp.asarray(weights, dtype=jnp.float64)
            if w_arr.ndim != 1 or w_arr.shape[0] != n:
                raise ValueError(
                    f"EmpiricalMeasure.from_arrays: weights must be 1-D with "
                    f"length N={n}; got shape {tuple(w_arr.shape)}"
                )
        _assert_finite_weights(w_arr, source="from_arrays")
        _assert_complete_data_under_mask(x_arr, mask_arr, source="from_arrays")
        _assert_mask_columns_supported(mask_arr, source="from_arrays")
        return cls(x=x_arr, mask=mask_arr, weights=w_arr)


__all__ = ["EmpiricalMeasure"]
