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
            is stripped internally.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        m : (M,) jax array
            ``(1 / N_j) * sum_i d_ij * w_i * psi_j(x_i, theta)`` for
            each coordinate ``j``; coordinates with ``N_j = 0`` map to
            zero rather than NaN.

        Notes
        -----
        NaN-mask semantics: residual entries where ``mask[i, j] = 0`` are
        zeroed out before the weighted sum. This means a moment function
        that returns NaN on a row whose mask entry is zero (e.g. because
        an input column is NaN under the standard pandas missing-data
        convention) does not poison the integral. The framework treats
        the mask as the authoritative source of observability; NaN in
        ``psi`` at masked positions is silently dropped. NaN at an
        observed (``mask = 1``) position still propagates --- that case
        is a genuine residual evaluation error, not a missingness signal.
        """

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(self.x)  # (N, M)
        # Pairwise per-coordinate mass: d_ij * w_i, broadcast across moments.
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        # NaN-mask semantics: drop NaN at masked positions before summing.
        # We replace NaN with 0 only where mask=0; observed-position NaNs
        # are preserved and propagate (legitimately surfacing model bugs).
        psi_safe = jnp.where(self.mask == 0.0, 0.0, psi_batch)
        numer = jnp.sum(weight_mask * psi_safe, axis=0)  # (M,)
        denom = jnp.sum(weight_mask, axis=0)  # (M,)
        return _safe_divide(numer, denom)

    def jacobian(self, psi: StructuralModel, theta: ParamsLike) -> Float[Array, "M K"]:
        """Per-coordinate weighted mean of :math:`\\nabla_\\theta \\psi`.

        Uses :func:`jax.jacfwd` on the flattened ``theta`` (see
        :mod:`emu_gmm._internal.params`) at the per-observation level,
        then applies the same mask / weight aggregation as
        :meth:`expectation`. Returns a plain ``(M, K)`` JAX array.

        NaN-mask semantics: matches :meth:`expectation`. Gradient entries
        ``grad[i, j, k]`` with ``mask[i, j] = 0`` are zeroed before the
        weighted sum so that NaN at masked positions does not poison the
        Jacobian (or the autodiff tape into which it feeds via the
        inference engine's :math:`G' \\Lambda G` step).
        """
        flat_theta, treedef = flatten_params(theta)

        def psi_flat(x: Float[Array, " D"], flat: Float[Array, " K"]):
            params = unflatten_params(flat, treedef)
            return _to_plain(psi(x, params))

        def grad_at(x: Float[Array, " D"]) -> Float[Array, "M K"]:
            return jax.jacfwd(lambda flat: psi_flat(x, flat))(flat_theta)

        grad_batch = jax.vmap(grad_at)(self.x)  # (N, M, K)
        weight_mask = self.mask * self.weights[:, None]  # (N, M)
        # NaN-mask semantics: zero out gradient entries at masked positions.
        # The mask is (N, M); broadcast over the (N, M, K) gradient.
        grad_safe = jnp.where(self.mask[:, :, None] == 0.0, 0.0, grad_batch)
        numer = jnp.sum(weight_mask[:, :, None] * grad_safe, axis=0)  # (M, K)
        denom = jnp.sum(weight_mask, axis=0)  # (M,)
        return _safe_divide(numer, denom[:, None])

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        weights: Any | None = None,
        mask: Any | None = None,
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
            NaN entries are taken as the pandas missing-data sentinel:
            when no explicit ``mask`` is supplied the mask is derived
            from the non-NaN positions of ``df`` (any row with any NaN
            is treated as unobserved for *all* moments). The NaN entries
            of ``x`` are then replaced by ``0`` so they do not poison
            the autodiff tape downstream --- the mask is the
            authoritative source of observability.
        weights : :class:`pandas.Series` or array-like, optional
            Per-observation weights. Defaults to all-ones.
        mask : :class:`pandas.DataFrame` or array-like of shape ``(N, M)``,
            optional. Per-coordinate observability. When omitted, derived
            from ``~df.isna().any(axis=1)`` broadcast across all moments
            (a single row-level observability indicator replicated to
            ``M = D`` columns). The number of moments ``M`` is inferred
            from the mask's column count when one is supplied; otherwise
            the caller is responsible for selecting a compatible model.

        Returns
        -------
        measure : :class:`EmpiricalMeasure`

        Notes
        -----
        The "row-level mask broadcast across moments" default is the
        simplest pandas convention. Per-coordinate masks (where some
        moments are observable on rows that other moments are not) must
        be passed explicitly via the ``mask=`` kwarg; design.org Section
        2 calls this the pairwise-overlap missingness pattern.
        """
        x_arr, _cols, _obs_name = labels_mod.normalise_x(df)
        n = int(x_arr.shape[0])
        w_arr = labels_mod.normalise_weights(weights, n)
        if mask is None:
            # No mask supplied: derive from NaN positions when present.
            # Row-level observability indicator (1 if no NaN in the row),
            # broadcast across all D moments. Users who need a fine-
            # grained per-coordinate mask must pass mask= explicitly.
            m = int(x_arr.shape[1])
            row_observed = (~jnp.any(jnp.isnan(x_arr), axis=1)).astype(jnp.float32)
            mask_arr = jnp.broadcast_to(row_observed[:, None], (n, m))
            # Replace NaN in x with 0 so masked entries don't poison
            # downstream tape (e.g. via jacfwd through psi(x, theta)).
            x_arr = jnp.where(jnp.isnan(x_arr), jnp.zeros_like(x_arr), x_arr)
            return cls(x=x_arr, mask=mask_arr, weights=w_arr)
        # Explicit mask path.
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
        # With an explicit mask the user has declared per-coordinate
        # observability; replace NaN in x with 0 so that AD does not
        # propagate NaN through psi for rows the user marked as missing.
        # The expectation/jacobian methods further guard against NaN at
        # masked positions in psi's *output*. NaN at observed positions
        # is a genuine residual evaluation error and will surface.
        x_arr = jnp.where(jnp.isnan(x_arr), jnp.zeros_like(x_arr), x_arr)
        return cls(x=x_arr, mask=mask_arr, weights=w_arr)


__all__ = ["EmpiricalMeasure"]
