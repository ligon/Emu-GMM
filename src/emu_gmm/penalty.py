"""In-objective parameter penalty hook.

The framework's GMM objective is

.. math::
   Q_\\mu(\\theta) = \\| L_\\mu(\\theta)^{-1}\\, m_\\mu(\\theta) \\|^2.

For some workflows --- notably K-Aggregators's ``LAMBDA_C`` path, where a
Tikhonov ridge is added on the consumption coefficient vector to stabilise
the recovery on noisy aggregate data --- the user wants an explicit
parameter-space penalty added directly to the criterion:

.. math::
   Q_{\\mu,\\mathrm{pen}}(\\theta)
     = Q_\\mu(\\theta) + p(\\theta).

This module defines the public protocol :class:`PenaltyStrategy`, the
default :class:`TikhonovPenalty`, and :class:`ResidualPenalty` for custom
least-squares penalties. The estimator appends ``residuals(theta)`` when
available, preserving their Gauss--Newton curvature. Scalar-only penalties
retain the single :math:`\\sqrt{p(\\theta)+10^{-30}}` residual fallback.

This is architecturally distinct from
:class:`emu_gmm.regularization.DiagonalTikhonov`, which lives on the
covariance side (PD-restoration of :math:`V`). The penalty here acts on
:math:`\\theta` directly and enters the criterion, gradient and optimization
curvature. The reported data-only GN matrix remains separate.

See ``docs/design.org`` and GitHub issue #7 for context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float

from emu_gmm._internal.params import flatten_params_for_ad

ParamsLike = Any


def _residual_vector(value: Any) -> Float[Array, " R"]:
    residuals = jnp.asarray(value)
    real_numeric = jnp.issubdtype(residuals.dtype, jnp.integer) or jnp.issubdtype(
        residuals.dtype, jnp.floating
    )
    if residuals.ndim != 1 or not real_numeric:
        raise ValueError(
            "penalty.residuals(theta) must return a one-dimensional real array"
        )
    return residuals


def _check_coefficient_metadata(shape: tuple[int, ...], dtype: Any) -> None:
    real_numeric = jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(
        dtype, jnp.floating
    )
    if shape != () or not real_numeric:
        raise ValueError("TikhonovPenalty.c must be a real numeric scalar")


def _check_coefficient_value(c: Any) -> None:
    value = np.asarray(c)
    _check_coefficient_metadata(value.shape, value.dtype)
    if not np.isfinite(value) or value < 0:
        raise ValueError("TikhonovPenalty.c must be finite and non-negative")


def _validated_coefficient(c: Any) -> Any:
    # Inspect the original object: asarray of a captured Python scalar inside
    # jit would create a tracer and unnecessarily add a callback (ledger 25).
    if isinstance(c, jax.core.Tracer):
        _check_coefficient_metadata(c.shape, c.dtype)
        jax.debug.callback(_check_coefficient_value, c, ordered=True)
    else:
        _check_coefficient_value(c)
    return jnp.asarray(c)


@runtime_checkable
class PenaltyStrategy(Protocol):
    """In-objective penalty on :math:`\\theta`.

    Implementations expose:

    - :meth:`penalty` --- the scalar :math:`p(\\theta) \\geq 0` added to
      :math:`Q_\\mu(\\theta)`.
    - :meth:`gradient` --- :math:`\\nabla_\\theta p(\\theta)` as a pytree
      with the same structure as ``theta``. Provided for callers that
      want the gradient without re-tracing through ``jax.grad``; the
      estimator itself relies on AD through the penalty residuals.

    An implementation may additionally expose ``residuals(theta) -> (R,)``
    with ``sum(residuals(theta)**2) == penalty(theta)``. The estimator uses
    these rows directly instead of the scalar square-root fallback. Their
    length must be fixed across parameter values. They affect optimization
    only, never data moment counts, covariance or the J statistic.
    """

    def penalty(self, theta: ParamsLike) -> Float[Array, ""]: ...

    def gradient(self, theta: ParamsLike) -> ParamsLike: ...


@jdc.pytree_dataclass
class ResidualPenalty:
    """Parameter penalty defined by a real residual vector (issue #268).

    ``residual_fn(theta)`` returns a one-dimensional array of fixed length;
    its squared norm is the penalty. For a ridge on a coefficient block,
    use ``ResidualPenalty(lambda theta: theta.coefficients.array)`` or
    include the square root of a positive ridge weight in that function.
    Keeping the rows separate preserves the block's Gauss--Newton curvature,
    including at a zero penalty. The callable is static for JIT purposes.
    """

    residual_fn: Callable[[ParamsLike], Float[Array, " R"]] = jdc.static_field()  # type: ignore[attr-defined]

    def residuals(self, theta: ParamsLike) -> Float[Array, " R"]:
        """Return the optimization-only penalty rows."""
        return _residual_vector(self.residual_fn(theta))

    def penalty(self, theta: ParamsLike) -> Float[Array, ""]:
        """Return the squared norm, without a square root at the origin."""
        residuals = self.residuals(theta)
        return jnp.sum(residuals * residuals)

    def gradient(self, theta: ParamsLike) -> ParamsLike:
        """Return the scalar penalty gradient in the parameter PyTree."""
        return jax.grad(self.penalty)(theta)


@jdc.pytree_dataclass
class TikhonovPenalty:
    """Quadratic ridge penalty on the flattened parameter vector.

    Computes :math:`p(\\theta) = c \\cdot \\lVert \\theta_{\\mathrm{flat}}
    \\rVert^2` where ``c`` is a non-negative scalar coefficient and
    :math:`\\theta_{\\mathrm{flat}}` is the leaf-stacked 1-D
    representation of ``theta`` produced by the manifold-aware
    :func:`emu_gmm._internal.params.flatten_params_for_ad`.

    The gradient is :math:`\\nabla_\\theta p(\\theta) = 2 c
    \\theta_{\\mathrm{flat}}`, returned in the original pytree shape so
    the user can compose it with their own theta-shaped values.

    Manifold parameters
    -------------------
    For an all-scalar (v1) tree the flatten is the v1 leaf-stack, so the
    value is **bitwise unchanged** from the original scalar-only
    implementation. For a tree with a non-scalar / manifold leaf the ridge
    acts on the **ambient** flatten (issue #150). For a
    :class:`~emu_gmm.manifolds.psd_fixed_rank.PSDFixedRank` factor ``A``
    this is

    .. math::
       c \\, \\lVert A \\rVert_F^2 = c \\, \\operatorname{tr}(A A^\\top)
         = c \\, \\operatorname{tr}(\\Gamma),

    which is **gauge-invariant** --- :math:`\\lVert A Q \\rVert_F =
    \\lVert A \\rVert_F` for orthogonal ``Q`` --- so the penalty is
    well-defined on the quotient (it shrinks the scale of
    :math:`\\Gamma = A A^\\top`, not a gauge-arbitrary coordinate). A
    consumer wanting to penalise only a *specific* leaf (e.g. a coefficient
    sub-block) can use :class:`ResidualPenalty` reading that leaf directly;
    this class ridges the whole ambient vector uniformly and retains its
    scalar square-root representation for compatibility.

    Parameters
    ----------
    c : float or jax scalar (traced)
        Finite, non-negative real scalar coefficient. Traced (not static) so
        users can sweep over ``c`` without retriggering compilation.
        Validation runs when :meth:`penalty` is evaluated: invalid concrete
        values raise ``ValueError``; genuinely traced values use a scalar
        debug callback and raise JAX's runtime error wrapper under JIT.
        Captured concrete values add no callback. Construction itself remains
        unrestricted so JAX can reconstruct abstract carriers and cotangents,
        whose coefficient-shaped leaf need not be a valid primal coefficient.
    """

    c: Float[Array, ""]

    def penalty(self, theta: ParamsLike) -> Float[Array, ""]:
        """Return :math:`p(\\theta) = c \\cdot \\lVert \\theta_{\\mathrm{flat}} \\rVert^2`.

        ``theta`` may carry manifold / non-scalar leaves; the ambient
        flatten is used (see the class docstring for the gauge-invariance
        of the resulting ridge on a ``PSDFixedRank`` factor).
        """
        c = _validated_coefficient(self.c)
        flat, _, _ = flatten_params_for_ad(theta)
        return c * jnp.sum(flat * flat)

    def gradient(self, theta: ParamsLike) -> ParamsLike:
        """Return :math:`\\nabla_\\theta p(\\theta) = 2 c \\theta` shaped like ``theta``.

        Uses :func:`jax.grad` on :meth:`penalty` so the result is in the
        same pytree structure as ``theta``. Cheaper than re-deriving by
        hand and guarantees consistency with what JAX AD would compute
        downstream.
        """
        return jax.grad(self.penalty)(theta)


__all__ = ["PenaltyStrategy", "ResidualPenalty", "TikhonovPenalty"]
