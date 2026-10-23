"""Bounded-tangent Jacobian construction for opt-in estimation paths."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def columnwise_jacobian(
    fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
    x: Float[Array, " K"],
) -> Float[Array, "M K"]:
    """Differentiate one tangent column at a time inside a JAX loop.

    ``linearize`` retains one primal tape; ``lax.map`` scans the standard
    basis instead of broadcasting it across the full residual evaluation.
    The final ``(M, K)`` Jacobian is necessarily materialized for the
    information matrix, but no intermediate differentiation lane has a
    leading ``K`` axis.
    """
    _, pushforward = jax.linearize(fn, x)
    basis = jnp.eye(x.shape[0], dtype=x.dtype)
    return jnp.swapaxes(jax.lax.map(pushforward, basis), 0, 1)


__all__ = ["columnwise_jacobian"]
