"""Fair-coin Bernoulli: the smallest interesting GMM problem.

DGP
---
``X_i ~ Bernoulli(p)`` with ``p = 0.5``, drawn i.i.d.

Population moment
-----------------
A single scalar moment identifies the single scalar parameter::

    m(theta) = E_X[ psi(X, theta) ] = E[X] - theta.p

At the truth ``theta.p = p_true = 0.5`` this equals zero. The sample
analogue is the difference between the sample mean and ``theta.p`` --
classical Manski-style analog estimation.

What's being recovered
----------------------
A single scalar ``p_hat ~= 0.5``, with an asymptotic standard error
``sqrt(p (1 - p) / N) ~= 0.0158`` at ``N = 1000``. With ``M = K = 1``
the problem is just-identified, so the J-statistic is identically
zero (``J_dof = 0``).

Emu-GMM surface this showcases
------------------------------
- ``EmpiricalMeasure`` constructed directly from a JAX array of
  pre-generated Bernoulli draws (no pandas import needed).
- ``IIDCovariance`` for the per-observation covariance estimator.
- ``Identity`` weighting -- when ``M = K = 1`` the weighting choice is
  irrelevant (any positive scalar gives the same root), so we use the
  smallest API surface.
- The ``estimate(...)`` entry point with the default optimiser
  (``optimistix_lm``).
- ``EstimationResult.coef_table`` for a pandas-formatted summary
  (estimate, std_error, t_stat, p_value).

This is the pedagogical "what is the minimum amount of code?" example.
Ported from ``DGP_Protocol/examples/fair_coin.py``; the data generation
is reimplemented here on JAX primitives so there is no runtime
dependency on ``dgp_protocol``.

Run from the repo root with::

    poetry run python examples/fair_coin.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm import (
    EmpiricalMeasure,
    EstimationResult,
    IIDCovariance,
    Identity,
    estimate,
)

# ---- Ground truth and defaults ----
P_TRUE: float = 0.5
N_DATA: int = 1000
DATA_SEED: int = 2026


@jdc.pytree_dataclass
class CoinParams:
    """Single scalar parameter: the Bernoulli success probability."""

    p: float


def coin_residual(x: Float[Array, " D"], theta: CoinParams) -> Float[Array, " M"]:
    """Per-observation moment residual: ``psi(x, theta) = x - theta.p``.

    The observation ``x`` is a length-1 vector holding the single
    Bernoulli draw; the moment is the scalar ``x - p``. At the truth,
    ``E[X] = p_true = theta.p``, so ``E[psi] = 0``.
    """
    return jnp.atleast_1d(x[0] - theta.p)


def make_coin_data(n: int = N_DATA, seed: int = DATA_SEED) -> Float[Array, "n 1"]:
    """Generate ``n`` Bernoulli(P_TRUE) draws as an ``(n, 1)`` JAX array.

    Mirrors the ``rng.binomial(n=1, p=0.5, ...)`` named-parameter
    spelling of the DGP_Protocol source: the intent (a Bernoulli draw)
    is visible in the code.
    """
    key = jax.random.PRNGKey(seed)
    draws = jax.random.bernoulli(key, p=P_TRUE, shape=(n,)).astype(jnp.float64)
    return draws[:, None]  # tabular (N, p=1) layout


def run_fair_coin(
    n: int = N_DATA,
    seed: int = DATA_SEED,
    p_init: float = 0.3,
) -> EstimationResult:
    """Estimate ``p`` for a fair coin via Emu-GMM. Returns the full result.

    Parameters
    ----------
    n
        Sample size.
    seed
        PRNG seed for the Bernoulli draws.
    p_init
        Starting value for the optimiser. Far enough from ``P_TRUE``
        that recovery is a non-trivial check.
    """
    x = make_coin_data(n=n, seed=seed)
    measure = EmpiricalMeasure(
        x=x,
        mask=jnp.ones((n, 1)),
        weights=jnp.ones(n),
    )
    return estimate(
        model=coin_residual,
        measure=measure,
        covariance=IIDCovariance(),
        weighting=Identity(),
        theta_init=CoinParams(p=p_init),
    )


def main() -> None:
    print(
        f"Fair-coin Bernoulli demo: N = {N_DATA} draws, "
        f"true p = {P_TRUE}, seed = {DATA_SEED}."
    )
    result = run_fair_coin()
    p_hat = float(result.theta_hat.p)
    print()
    print("Coefficient table:")
    print(result.coef_table.to_string())
    print()
    print(
        f"  p_hat  = {p_hat:.6f}   "
        f"(truth {P_TRUE:.2f}, |err| = {abs(p_hat - P_TRUE):.2e})"
    )
    print(f"  J-stat = {float(result.J_stat):.4e}   (dof = {result.J_dof})")
    print(f"  converged = {result.converged}   " f"({result.iterations} iterations)")


if __name__ == "__main__":
    main()
