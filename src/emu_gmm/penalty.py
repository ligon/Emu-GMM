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

This module defines the public protocol :class:`PenaltyStrategy` plus the
default :class:`TikhonovPenalty` implementation. The estimator wires the
penalty into the residual vector by appending :math:`\\sqrt{p(\\theta)}` as
an extra residual row so the existing NLLS surface (and its LM Jacobian)
still applies; JAX AD handles the chain rule through the square-root.

This is architecturally distinct from
:class:`emu_gmm.regularization.DiagonalTikhonov`, which lives on the
covariance side (PD-restoration of :math:`V`). The penalty here acts on
:math:`\\theta` directly and enters the criterion, the gradient, and the
information matrix.

See ``docs/design.org`` and GitHub issue #7 for context.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.params import flatten_params_for_ad

ParamsLike = Any


@runtime_checkable
class PenaltyStrategy(Protocol):
    """In-objective penalty on :math:`\\theta`.

    Implementations expose:

    - :meth:`penalty` --- the scalar :math:`p(\\theta) \\geq 0` added to
      :math:`Q_\\mu(\\theta)`.
    - :meth:`gradient` --- :math:`\\nabla_\\theta p(\\theta)` as a pytree
      with the same structure as ``theta``. Provided for callers that
      want the gradient without re-tracing through ``jax.grad``; the
      estimator itself relies on AD through :meth:`penalty`.
    """

    def penalty(self, theta: ParamsLike) -> Float[Array, ""]: ...

    def gradient(self, theta: ParamsLike) -> ParamsLike: ...


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
    sub-block) should supply its own :class:`PenaltyStrategy` reading that
    leaf directly; this class ridges the whole ambient vector uniformly.

    Parameters
    ----------
    c : float or jax scalar (traced)
        Non-negative penalty coefficient. Traced (not a static field) so
        users can sweep over ``c`` without retriggering compilation.
    """

    c: Float[Array, ""]

    def penalty(self, theta: ParamsLike) -> Float[Array, ""]:
        """Return :math:`p(\\theta) = c \\cdot \\lVert \\theta_{\\mathrm{flat}} \\rVert^2`.

        ``theta`` may carry manifold / non-scalar leaves; the ambient
        flatten is used (see the class docstring for the gauge-invariance
        of the resulting ridge on a ``PSDFixedRank`` factor).
        """
        flat, _, _ = flatten_params_for_ad(theta)
        return jnp.asarray(self.c) * jnp.sum(flat * flat)

    def gradient(self, theta: ParamsLike) -> ParamsLike:
        """Return :math:`\\nabla_\\theta p(\\theta) = 2 c \\theta` shaped like ``theta``.

        Uses :func:`jax.grad` on :meth:`penalty` so the result is in the
        same pytree structure as ``theta``. Cheaper than re-deriving by
        hand and guarantees consistency with what JAX AD would compute
        downstream.
        """
        return jax.grad(self.penalty)(theta)


__all__ = ["PenaltyStrategy", "TikhonovPenalty"]
