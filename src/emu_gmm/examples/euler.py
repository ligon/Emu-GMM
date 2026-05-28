"""Multi-asset consumption Euler equation (Hansen-Singleton 1982 style).

The classical consumption-based asset-pricing setup: with power utility
and ``J`` risky assets, the Euler equation for each asset is

    E[ beta * (c_{t+1}/c_t)^{-gamma} * (1 + r_j) ] = 1.

With J > K = 2 (two structural parameters beta, gamma), the system is
over-identified and admits a proper J-test.

This module provides:

- :data:`EulerParams`: the parameter dataclass.
- :func:`euler_residual`: the per-observation residual function.
- :func:`euler_sampler_factory`: a SyntheticMeasure-compatible sampler
  for a given ``n_sim``.
- :func:`euler_analytical_expectation`: closed-form E[psi] under the DGP,
  for use as AnalyticalMeasure.expectation_fn.
- :func:`euler_data`: pre-generated fixed observations for use with
  EmpiricalMeasure.

The DGP is engineered so all Euler conditions vanish exactly at the
true parameters (BETA_TRUE, GAMMA_TRUE). The sampler / data inherit
random shocks but the population-level moment conditions hold by
construction.

Derivation
==========

Single shock ``z ~ N(0, 1)``; consumption growth and asset returns:

    log(c_{t+1}/c_t) = MU_C + SIGMA_C * z
    r_j              = MU_R[j] + NU[j] * z + eta * eps_j,   eps_j ~ N(0,1)

The eta term is idiosyncratic noise per asset, included so that
Var(psi(theta_0)) is full rank (without it, all moments are deterministic
functions of z and V is rank-1).

Using E[exp(a z)] = exp(a^2/2) and E[exp(a z) z] = a * exp(a^2/2),

    E[(c'/c)^{-gamma} (1 + r_j)]
        = exp(-gamma*MU_C + gamma^2 * SIGMA_C^2 / 2)
        * [(1 + MU_R[j]) - gamma * SIGMA_C * NU[j]].

(The eta term drops out because E[eps_j] = 0 and eps_j is independent of
z.) Setting BETA_TRUE times this equal to 1 and solving for MU_R[j]:

    MU_R[j] = K - 1 + GAMMA_TRUE * SIGMA_C * NU[j],
    K       = (1 / BETA_TRUE) * exp(GAMMA_TRUE * MU_C
                                    - GAMMA_TRUE^2 * SIGMA_C^2 / 2).

Choosing NU = (0.5, 1.0, 1.5) gives three linearly-independent moment
conditions: the slope of (MU_R[j], NU[j]) in (mu, nu)-space is
GAMMA_TRUE * SIGMA_C, so the wrong gamma produces non-zero moments
even when beta is free to adjust.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

# ---- Ground truth ----
BETA_TRUE = 0.96
GAMMA_TRUE = 2.0
MU_C = 0.02
SIGMA_C = 0.05
ETA = 0.05  # idiosyncratic asset-return shock SD

# ---- Asset structure ----
NU = jnp.array([0.5, 1.0, 1.5])
N_ASSETS = int(NU.shape[0])

# Solve for MU_R[j] so that all Euler conditions vanish at the truth.
_K = (1.0 / BETA_TRUE) * jnp.exp(GAMMA_TRUE * MU_C - GAMMA_TRUE**2 * SIGMA_C**2 / 2)
MU_R = _K - 1.0 + GAMMA_TRUE * SIGMA_C * NU

# Observation layout: (c_t, c_{t+1}, r_1, ..., r_J).
D = 2 + N_ASSETS


@jdc.pytree_dataclass
class EulerParams:
    """Discount factor ``beta`` and risk-aversion ``gamma``."""

    beta: float
    gamma: float


def euler_residual(x: Float[Array, " D"], theta: EulerParams) -> Float[Array, " M"]:
    """Per-observation Euler residual, one per asset.

    ``psi_j(x, theta) = beta * (c'/c)^{-gamma} * (1 + r_j) - 1``
    """
    c_t = x[0]
    c_next = x[1]
    rs = x[2:]  # (N_ASSETS,)
    sdf = theta.beta * (c_next / c_t) ** (-theta.gamma)
    return sdf * (1.0 + rs) - 1.0


def euler_sampler_factory(n_sim: int):
    """Return a :class:`SyntheticMeasure`-compatible sampler.

    The sampler closure draws ``n_sim`` observations from the DGP. It
    ignores ``theta`` --- the data-generating process is fixed at the
    truth; ``theta`` is what we're trying to estimate.

    Parameters
    ----------
    n_sim
        Number of synthetic observations per call.

    Returns
    -------
    callable
        ``sampler(key, theta) -> (n_sim, D) array``.
    """

    def _sampler(key: jax.Array, theta: EulerParams) -> Float[Array, "n_sim D"]:
        del theta  # DGP doesn't depend on the structural parameters
        k_z, k_eps = jax.random.split(key)
        z = jax.random.normal(k_z, (n_sim,))
        eps = jax.random.normal(k_eps, (n_sim, N_ASSETS))
        c_t = jnp.ones(n_sim)
        c_next = jnp.exp(MU_C + SIGMA_C * z)
        rs = MU_R[None, :] + NU[None, :] * z[:, None] + ETA * eps
        return jnp.concatenate([c_t[:, None], c_next[:, None], rs], axis=1)

    return _sampler


def euler_analytical_expectation(model, theta: EulerParams) -> Float[Array, " M"]:
    """Closed-form ``E[psi(theta)]`` under the DGP.

    Suitable as ``AnalyticalMeasure.expectation_fn``. Evaluates to zero
    exactly at ``(BETA_TRUE, GAMMA_TRUE)``.

    The argument ``model`` is unused (the residual structure is baked
    into the closed form); it is present to match the protocol signature.
    """
    del model
    of_gamma = jnp.exp(-theta.gamma * MU_C + theta.gamma**2 * SIGMA_C**2 / 2)
    factor = (1.0 + MU_R) - theta.gamma * SIGMA_C * NU
    return theta.beta * of_gamma * factor - 1.0


def euler_data(seed: int, n: int) -> Float[Array, "N D"]:
    """Pre-generate ``n`` observations from the DGP.

    Useful as fixed input to :class:`EmpiricalMeasure`. Equivalent to
    calling the sampler once with a freshly-keyed RNG.
    """
    sampler = euler_sampler_factory(n)
    key = jax.random.PRNGKey(seed)
    return sampler(key, EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE))


__all__ = [
    "BETA_TRUE",
    "GAMMA_TRUE",
    "MU_C",
    "SIGMA_C",
    "ETA",
    "NU",
    "MU_R",
    "N_ASSETS",
    "D",
    "EulerParams",
    "euler_residual",
    "euler_sampler_factory",
    "euler_analytical_expectation",
    "euler_data",
]
