"""Closed-form analytical covariance for identification round-trip checks.

``AnalyticalCovariance`` exposes the framework's ``CovarianceStrategy``
protocol over a user-supplied closed-form :math:`V_\\mu(\\theta)`,
the variance of the moment estimator under the population measure. It
is the natural pair for :class:`emu_gmm.measures.analytical.AnalyticalMeasure`
in identification round-trip tests: with no sampling, the variance is
whatever the user supplies on paper.

The user is responsible for the closed-form expression; v1 does not
include a quadrature engine. For identification checks where the
J-statistic distribution is not the object of interest, supplying a
constant (e.g. the identity) is acceptable --- the optimiser still
converges to :math:`\\theta_0` exactly when the moment vector reaches
zero. See ``docs/api-sketch.org`` Section 3 for the architectural
context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import haliax as ha
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
class AnalyticalCovariance:
    """User-supplied closed-form :math:`V_\\mu(\\theta)`.

    Parameters
    ----------
    covariance_fn : callable (static)
        ``covariance_fn(model, theta) -> (M, M) array``. Returns the
        population variance of the moment estimator at ``theta`` in
        closed form. The first argument is the user's
        :data:`StructuralModel`; the implementation is free to ignore
        it (the closed form is typically reproduced inside
        ``covariance_fn`` directly).
    """

    covariance_fn: Callable[[StructuralModel, ParamsLike], Float[Array, "M M"]] = (
        jdc.static_field()  # type: ignore[attr-defined]
    )

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
    ) -> Float[Array, "M M"]:
        """Return the closed-form :math:`V_\\mu(\\theta)`.

        Delegates to ``covariance_fn(psi, theta)`` and strips a returned
        :class:`haliax.NamedArray` to its underlying array. The
        ``measure`` argument is accepted for protocol conformance but
        not used: the closed-form variance is intrinsic to the
        ``covariance_fn`` the user supplied.
        """
        del measure
        return _to_plain(self.covariance_fn(psi, theta))


__all__ = ["AnalyticalCovariance"]
