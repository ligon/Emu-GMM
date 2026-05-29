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

from typing import Any

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


@jdc.pytree_dataclass(init=False)
class Fixed:
    """Pre-cholesky weighting at a frozen anchor :math:`V_0`.

    Stores the lower-triangular Cholesky factor :math:`L_0` of the
    anchor variance; the optimiser sees a quadratic-in-``m`` surface.
    The ``V`` argument to :meth:`whitening_residual` is accepted (so
    the protocol signature matches) but ignored.

    Construction is keyword-only and requires exactly one of ``L0`` or
    ``V0``:

    - ``Fixed(L0=L0)`` --- supply the Cholesky factor directly.
    - ``Fixed(V0=V0)`` --- supply the anchor variance; the framework
      computes :math:`L_0 = \\mathrm{chol}(V_0)` for you.

    Equivalent classmethod constructors :meth:`from_L0` and
    :meth:`from_V0` are also provided.

    Notes
    -----
    Positional construction is intentionally disallowed. In other GMM
    libraries (notably ManifoldGMM) the analogous one-arg constructor
    accepts the weighting matrix ``W`` directly; in :mod:`emu_gmm` the
    stored object is the Cholesky factor :math:`L_0` of the *variance*
    :math:`V_0 = W^{-1}`. Silently storing ``W`` as ``L0`` would yield
    a wrong-but-runnable estimator, so we require an explicit kwarg.

    Parameters
    ----------
    L0 : (M, M) lower-triangular array, keyword-only
        Cholesky factor of the anchor variance.
    V0 : (M, M) symmetric positive-definite array, keyword-only
        Anchor variance; the Cholesky factor is computed internally.
    """

    L0: Float[Array, "M M"]

    def __init__(
        self,
        *args: Any,
        L0: Float[Array, "M M"] | None = None,
        V0: Float[Array, "M M"] | None = None,
    ) -> None:
        if args:
            raise TypeError(
                "Fixed(...) does not accept positional arguments. "
                "Pass either Fixed(L0=L0) (Cholesky factor of the anchor "
                "variance) or Fixed(V0=V0) (anchor variance; Cholesky "
                "computed internally). Note: the stored object is the "
                "Cholesky factor of the variance V_0, not the weighting "
                "matrix W = V_0^{-1}. If you are porting code that wrote "
                "Fixed(W), use Fixed.from_V0(jnp.linalg.inv(W)) instead."
            )
        if L0 is None and V0 is None:
            raise TypeError(
                "Fixed(...) requires exactly one of L0= or V0=; neither "
                "was supplied. Use Fixed(V0=V0) if you have the anchor "
                "variance, or Fixed(L0=L0) if you already have its "
                "Cholesky factor."
            )
        if L0 is not None and V0 is not None:
            raise TypeError(
                "Fixed(...) requires exactly one of L0= or V0=; both "
                "were supplied. The two are redundant (L0 is the "
                "Cholesky factor of V0); pick one."
            )
        if V0 is not None:
            L0 = cholesky(V0)
        # object.__setattr__ because @jdc.pytree_dataclass is frozen.
        object.__setattr__(self, "L0", L0)

    @classmethod
    def from_V0(cls, V0: Float[Array, "M M"]) -> Fixed:
        """Construct from an anchor variance ``V0`` (pre-cholesky).

        Equivalent to ``Fixed(V0=V0)``.
        """
        return cls(V0=V0)

    @classmethod
    def from_L0(cls, L0: Float[Array, "M M"]) -> Fixed:
        """Construct directly from a Cholesky factor ``L0``.

        Equivalent to ``Fixed(L0=L0)``. Useful if you already hold the
        lower-triangular factor and want to make the intent explicit at
        the call site.
        """
        return cls(L0=L0)

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

    Also exported as the alias :data:`CUE` (continuously-updated
    estimator), the more common name in the econometrics literature
    following Hansen, Heaton & Yaron (1996, "Finite-Sample Properties
    of Some Alternative GMM Estimators", JBES 14(3), 262--280).

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


#: Econometrics-literature alias for :class:`ContinuouslyUpdated`. See
#: Hansen, Heaton & Yaron (1996), "Finite-Sample Properties of Some
#: Alternative GMM Estimators", JBES 14(3), 262--280, where the
#: continuously-updated estimator (CUE) is introduced as an alternative
#: to the two-step and iterated GMM weighting schemes.
CUE = ContinuouslyUpdated


__all__ = ["Identity", "Fixed", "ContinuouslyUpdated", "CUE"]
