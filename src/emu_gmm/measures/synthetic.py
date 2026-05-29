"""Generator-backed measure for simulation-based moment estimation.

``SyntheticMeasure`` exposes the framework's ``Measure`` protocol over a
user-supplied sampler that produces synthetic observations from a frozen
``jax.random.PRNGKey``. Common Random Numbers (CRN) --- reusing the same
key at every parameter --- converts a stochastic Monte Carlo objective
into a deterministic surface in ``theta``, enabling gradient-based
optimisation. Reparameterisation is the user's responsibility: write a
sampler that is smooth in ``theta`` (when it depends on ``theta`` at all)
so that ``jax.jacfwd`` produces meaningful gradients.

See ``docs/design.org`` Section 2 and ``docs/api-sketch.org`` Section 3
for the architectural context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.params import flatten_params, unflatten_params
from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a haliax NamedArray wrapper, returning the underlying array.

    Plain arrays / scalars pass through unchanged.
    """
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


@jdc.pytree_dataclass
class SyntheticMeasure:
    """Generator-backed measure with Common Random Numbers.

    Parameters
    ----------
    key : jax.Array
        Frozen :class:`jax.random.PRNGKey`. The same key is reused at
        every ``theta``, producing a deterministic objective surface.
    n_sim : int (static)
        Number of synthetic draws produced by one ``sampler`` call.
    sampler : callable (static)
        ``sampler(key, theta) -> (n_sim, D) array``. May depend on
        ``theta`` (true SMM) or ignore it (the data-generating process
        is exogenous, e.g. the Euler equation case).
    """

    key: jax.Array
    n_sim: int = jdc.static_field()  # type: ignore[attr-defined]
    sampler: Callable[[jax.Array, ParamsLike], Float[Array, "n_sim D"]] = (
        jdc.static_field()  # type: ignore[attr-defined]
    )

    def _draws(self, theta: ParamsLike) -> Float[Array, "n_sim D"]:
        """Run the sampler once and return the (n_sim, D) draws."""
        return self.sampler(self.key, theta)

    def expectation(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, " M"]:
        """Monte Carlo expectation of ``psi`` under the synthetic measure.

        Returns
        -------
        m : (M,) jax array
            ``(1/n_sim) sum_s psi(x_s, theta)``, with ``x_s`` from the
            sampler.
        """
        x_batch = self._draws(theta)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_batch)
        return jnp.mean(psi_batch, axis=0)

    def jacobian(self, psi: StructuralModel, theta: ParamsLike) -> Float[Array, "M K"]:
        """Jacobian of ``expectation`` with respect to ``theta``.

        Computed by routing ``theta`` through ``flatten_params`` and
        applying ``jax.jacfwd`` to a closure of the flattened argument.
        The result has the canonical ``(M, K)`` shape, with ``K`` equal
        to the number of leaves in ``theta``.
        """
        flat_theta, treedef = flatten_params(theta)

        def fn(flat):
            params = unflatten_params(flat, treedef)
            return self.expectation(psi, params)

        return jax.jacfwd(fn)(flat_theta)


__all__ = ["SyntheticMeasure"]
