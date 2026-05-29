"""Weighting / whitening strategies for the GMM objective.

The framework's objective is
:math:`Q_\\mu(\\theta) = \\| L_\\mu(\\theta)^{-1} m_\\mu(\\theta) \\|^2`,
where :math:`L L^\\top = V` is the Cholesky factor of the
moment-estimator variance and :math:`m` is the empirical moment.
A ``WeightingStrategy`` chooses how :math:`L` (and therefore the
weighting matrix :math:`\\Lambda = V^{-1}`) is constructed.

Three concrete strategies exist in v1:

- :class:`Identity` --- :math:`\\Lambda \\equiv I`; ``whitening_residual``
  returns ``m`` unchanged.
- :class:`Fixed` --- :math:`L = L_0` precomputed from an anchor
  :math:`V_0`; the optimisation surface is quadratic in ``m`` alone.
- :class:`ContinuouslyUpdated` --- :math:`L(\\theta)` is recomputed at
  every call; JAX AD threads through the Cholesky and the triangular
  solve so the residual's gradient picks up the dependence of
  :math:`L` on :math:`\\theta` when :math:`V = V(\\theta)`.

See ``docs/design.org`` Section 5 ("Architectural Core Highlights")
for the architectural commitment that the CU gradient must not drop
the :math:`\\nabla_\\theta V` term; that property is delivered here
by computing the Cholesky inside ``whitening_residual`` rather than
caching :math:`L`.
"""

from __future__ import annotations

import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.cholesky import cholesky, forward_solve
from emu_gmm.types import ParamsLike


@jdc.pytree_dataclass
class Identity:
    """Identity weighting: ``y = m``.

    Equivalent to setting :math:`\\Lambda = I_M` in the GMM objective.
    The supplied ``V`` is ignored. Useful as a sanity-check weighting
    or when the user wants an unweighted sum of squares.
    """

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return ``m`` unchanged.

        Parameters
        ----------
        m : (M,) array
            Empirical moment vector.
        V : (M, M) array
            Ignored.
        theta : ParamsLike
            Ignored.
        """
        del V, theta  # accepted for protocol conformance; not used here
        return m


@jdc.pytree_dataclass
class Fixed:
    """Pre-cholesky weighting at a frozen anchor :math:`V_0`.

    Stores the lower-triangular Cholesky factor :math:`L_0` of the
    anchor variance; the optimiser sees a quadratic-in-``m`` surface.
    The ``V`` argument to :meth:`whitening_residual` is accepted (so
    the protocol signature matches) but ignored.

    Parameters
    ----------
    L0 : (M, M) lower-triangular array
        Cholesky factor of the anchor variance. Construct via
        :meth:`from_V0` if you have :math:`V_0` and want the factor
        computed for you.
    """

    L0: Float[Array, "M M"]

    @classmethod
    def from_V0(cls, V0: Float[Array, "M M"]) -> Fixed:
        """Construct from an anchor variance ``V0`` (pre-cholesky)."""
        return cls(L0=cholesky(V0))

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return :math:`y = L_0^{-1} m`.

        The ``V`` and ``theta`` arguments are accepted for protocol
        conformance but ignored: the weighting is frozen at the anchor.
        """
        del V, theta
        return forward_solve(self.L0, m)


@jdc.pytree_dataclass
class ContinuouslyUpdated:
    """Continuously-updated (CU) weighting: :math:`L(\\theta)` per call.

    The Cholesky factor is recomputed at every evaluation, so JAX AD
    traces through the dependence of :math:`L` on :math:`\\theta` via
    :math:`V(\\theta)`. This is the default v1 weighting strategy.

    Has no traced or static state.
    """

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return :math:`y = L(\\theta)^{-1} m` with :math:`L L^\\top = V`.

        Both ``cholesky`` and ``forward_solve`` are differentiable, so
        the gradient of any downstream scalar of ``y`` picks up the
        dependence of :math:`L` on :math:`\\theta` through :math:`V`.

        Parameters
        ----------
        m : (M,) array
            Empirical moment vector at ``theta``.
        V : (M, M) array
            Variance of the moment estimator at ``theta``.
        theta : ParamsLike
            Accepted for protocol conformance; not used directly --- the
            ``theta``-dependence enters via ``V``.
        """
        del theta
        L = cholesky(V)
        return forward_solve(L, m)


__all__ = ["Identity", "Fixed", "ContinuouslyUpdated"]
