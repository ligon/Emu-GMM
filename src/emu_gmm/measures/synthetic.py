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

    def moments_and_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> tuple[Float[Array, " M"], Float[Array, "n_sim M"]]:
        """Shared primitive: SMM expectation + per-draw psi in one vmap.

        Returns the per-coordinate sample mean ``m`` together with the
        ``(n_sim, M)`` per-draw residual matrix that
        :class:`~emu_gmm.covariance.synthetic.SyntheticCovariance`
        rebuilds independently when called separately. This is the
        single source of truth for the SMM hot path: the synthetic
        covariance strategy accepts ``psi_batch`` as
        ``cached_intermediates`` and skips its own
        ``_draws`` + ``vmap(psi)`` pass, eliminating the ~2x sampler /
        residual overhead per ``residual_fn`` call
        (``docs/reviews/v1x-performance-review.org`` finding #5).

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.

        Returns
        -------
        m : (M,) jax array
            ``(1 / n_sim) sum_s psi(x_s, theta)``.
        psi_batch : (n_sim, M) jax array
            The vmapped residual matrix from the same sampler draws
            that produced ``m``. CRN preserves identity across the two
            outputs: ``m == jnp.mean(psi_batch, axis=0)`` exactly.
        """
        x_batch = self._draws(theta)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        psi_batch = jax.vmap(psi_at)(x_batch)
        m = jnp.mean(psi_batch, axis=0)
        return m, psi_batch

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
        m, _psi_batch = self.moments_and_contributions(psi, theta)
        return m

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

    def moment_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, "n_sim M"]:
        """Per-draw moment contributions ``g_s(theta) = psi(x_s, theta)``.

        Returns the ``(n_sim, M)`` matrix of per-draw evaluations of
        ``psi`` against the sampler's CRN-frozen draws. This is the
        building block downstream resampling and identification-robust
        inference (e.g. the Kleibergen K-statistic) consume. The
        analogue of :meth:`EmpiricalMeasure.moment_contributions`, but
        backed by the synthetic sampler rather than the observed data.
        """
        x_batch = self._draws(theta)

        def psi_at(x):
            return _to_plain(psi(x, theta))

        return jax.vmap(psi_at)(x_batch)

    def jacobian_contributions(
        self, psi: StructuralModel, theta: ParamsLike
    ) -> Float[Array, "n_sim M K"]:
        """Per-draw Jacobian contributions ``D_s(theta) = grad_theta psi(x_s, theta)``.

        Returns the ``(n_sim, M, K)`` tensor of per-draw Jacobians,
        suitable for estimating :math:`\\Sigma_{G_j, m}` (Kleibergen 2005
        eq. 8) in the synthetic-measure setting. Mirrors
        :meth:`EmpiricalMeasure.jacobian_contributions` but with CRN
        draws standing in for observed rows.

        The sampler draws are produced once via :meth:`_draws` and held
        fixed across the per-draw Jacobian computation, so
        ``jax.jacfwd`` differentiates ``psi`` (and not the sampler) with
        respect to ``theta`` --- which is correct for a CRN
        construction. If the sampler depends on ``theta``, the resulting
        Jacobian contribution captures only the residual's parameter
        dependence at the held draws; pair it with a user-supplied
        ``score_cov_fn`` to inference if the sampler-dependence matters.
        """
        flat_theta, treedef = flatten_params(theta)
        # Materialise draws at the current theta and freeze them; the
        # per-draw Jacobian then differentiates psi alone, mirroring the
        # empirical case where x is observation data and not a function
        # of theta.
        x_batch = self._draws(theta)

        def psi_flat(x: Float[Array, " D"], flat: Float[Array, " K"]):
            params = unflatten_params(flat, treedef)
            return _to_plain(psi(x, params))

        def grad_at(x: Float[Array, " D"]) -> Float[Array, "M K"]:
            return jax.jacfwd(lambda flat: psi_flat(x, flat))(flat_theta)

        return jax.vmap(grad_at)(x_batch)


__all__ = ["SyntheticMeasure"]
