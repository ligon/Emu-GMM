"""Closed-form analytical measure for identification round-trip checks.

``AnalyticalMeasure`` exposes the framework's ``Measure`` protocol over a
user-supplied closed-form population expectation. Unlike
:class:`emu_gmm.measures.synthetic.SyntheticMeasure`, which Monte Carlo
integrates against a frozen sample, an ``AnalyticalMeasure`` returns the
population value of :math:`\\mathbb{E}_\\mu[\\psi(\\cdot, \\theta)]` with
no sampling noise --- the user supplies the closed-form integral.

The Jacobian is computed via :func:`jax.jacfwd` of the user-supplied
``expectation_fn`` by default; users may also supply an analytical
``jacobian_fn`` and skip AD. This is the natural path for an
identification round-trip acceptance test: at the true parameter the
moment vector is exactly zero (up to floating point), so the optimiser
should recover :math:`\\theta_0` to floating-point precision.

The framework does not include a quadrature engine in v1; users with
non-closed-form populations should reach for
:class:`SyntheticMeasure` instead. See ``docs/design.org`` Section 2
and ``docs/api-sketch.org`` Section 3 for the architectural context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.params import flatten_params_for_ad, unflatten_params
from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a haliax NamedArray wrapper, returning the underlying array.

    Plain arrays / scalars pass through unchanged.
    """
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


@jdc.pytree_dataclass
class AnalyticalMeasure:
    """Closed-form population measure with optional analytical Jacobian.

    Parameters
    ----------
    expectation_fn : callable (static)
        ``expectation_fn(model, theta) -> (M,) array``. Returns the
        population value of :math:`\\mathbb{E}_\\mu[\\psi(\\cdot, \\theta)]`
        in closed form. The first argument is the user's
        :data:`StructuralModel`; the implementation is free to ignore it
        (the closed-form is typically derived from the model on paper
        and reproduced inside ``expectation_fn`` directly).
    jacobian_fn : callable or None (static, optional)
        ``jacobian_fn(model, theta) -> (M, K) array``. If supplied,
        :meth:`jacobian` calls it directly; otherwise the Jacobian is
        computed via :func:`jax.jacfwd` through a flattened-``theta``
        closure of :meth:`expectation`, matching the
        :class:`SyntheticMeasure` AD pattern.
    """

    expectation_fn: Callable[[StructuralModel, ParamsLike], Float[Array, " M"]] = (
        jdc.static_field()  # type: ignore[attr-defined]
    )
    jacobian_fn: Callable[[StructuralModel, ParamsLike], Float[Array, "M K"]] | None = (
        jdc.static_field(default=None)  # type: ignore[attr-defined]
    )

    def expectation(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, " M"]:
        """Closed-form expectation :math:`\\mathbb{E}_\\mu[\\psi(\\cdot, \\theta)]`.

        Delegates to ``expectation_fn(psi, theta)`` and strips a returned
        :class:`haliax.NamedArray` to its underlying array.
        """
        return _to_plain(self.expectation_fn(psi, theta))

    def jacobian(self, psi: StructuralModel, theta: ParamsLike) -> Float[Array, "M K"]:
        """Jacobian of :meth:`expectation` with respect to ``theta``.

        If a ``jacobian_fn`` was supplied at construction, it is called
        directly; otherwise the Jacobian is computed via
        :func:`jax.jacfwd` of a flattened-``theta`` closure of
        :meth:`expectation`, routing ``theta`` through
        :func:`~emu_gmm._internal.params.flatten_params_for_ad` and
        mirroring the :class:`SyntheticMeasure` pattern. The result has
        the canonical ``(M, K)`` shape, with ``K`` equal to the number
        of leaves for all-scalar trees and the total ambient dimension
        for manifold trees (#41).
        """
        if self.jacobian_fn is not None:
            return _to_plain(self.jacobian_fn(psi, theta))

        flat_theta, treedef, mspec = flatten_params_for_ad(theta)

        def fn(flat):
            params = unflatten_params(flat, treedef, manifold_spec=mspec)
            return self.expectation(psi, params)

        return jax.jacfwd(fn)(flat_theta)


__all__ = ["AnalyticalMeasure"]
