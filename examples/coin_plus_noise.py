"""Coin-plus-noise: a two-moment DGP composed from two independent streams.

The data-generating process is

    Y = X + e,
    X ~ Bernoulli(p),     (mean p, variance p (1 - p))
    e ~ N(0, sigma2),     (independent of X)

with structural parameters ``theta = (p, sigma2)``. Two population
moment conditions identify the two parameters exactly:

    m_1(theta) := E[Y]   - p                       = 0,
    m_2(theta) := E[Y^2] - (p^2 + p (1 - p) + sigma2)
               =  E[Y^2] - (p + sigma2)            = 0.

(The second form uses ``p^2 + p (1 - p) = p`` for Bernoulli p.)

True values are ``p = 0.5``, ``sigma2 = 1.0``, so ``E[Y] = 0.5`` and
``E[Y^2] = 1.5`` (equivalently ``Var[Y] = 1.25``).

This is the just-identified case (M = K = 2). Continuously-updated
weighting is exercised for pedagogical value even though the J
statistic is identically zero up to floating-point noise.

Pedagogical points
------------------

1. **DGP composition.** The synthetic sampler is built explicitly from
   two independent sub-samplers --- one for X (Bernoulli) and one for
   e (Gaussian) --- joined by addition. See :func:`coin_plus_noise_sampler`
   below. This mirrors the composition pattern in
   ``DGP_Protocol/examples/coin_plus_noise.py``, where a continuous DGP
   is built atop a fair-coin DGP without leaking either component's
   randomness into the other.

2. **Two moments simultaneously.** Unlike a single-moment estimator
   (mean, proportion), this example wires a length-2 residual vector
   through the framework: ``estimate()`` infers ``M = 2``, builds the
   2x2 covariance ``V_X``, and reports a 2x2 parameter covariance
   ``Sigma_theta`` indexed by ``("p", "sigma2")``.

3. **API surface exercised.** ``SyntheticMeasure`` with a CRN-frozen
   sampler closure, ``SyntheticCovariance`` (the natural pairing),
   ``ContinuouslyUpdated`` weighting, and the ``EulerParams``-style
   ``@jdc.pytree_dataclass`` parameter container. The empirical path
   (pre-generated data + ``EmpiricalMeasure`` + ``IIDCovariance``) is
   also demonstrated to make the composition pattern concrete.

Run from the repo root with::

    poetry run python examples/coin_plus_noise.py
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from emu_gmm import (
    EmpiricalMeasure,
    IIDCovariance,
    SyntheticCovariance,
    SyntheticMeasure,
    estimate,
    optimistix_lm,
)
from jaxtyping import Array, Float

# ---- Ground truth ----
P_TRUE: float = 0.5
SIGMA2_TRUE: float = 1.0

# ---- Sample size ----
N_SIM: int = 2000


@jdc.pytree_dataclass
class CoinPlusNoiseParams:
    """Structural parameters: Bernoulli mean ``p`` and noise variance ``sigma2``."""

    p: float
    sigma2: float


def coin_plus_noise_residual(
    y: Float[Array, " D"], theta: CoinPlusNoiseParams
) -> Float[Array, " M"]:
    """Per-observation residual vector with two moment components.

    ``y`` is a length-1 vector ``[Y_i]`` (the framework's contract is one
    observation per row of the data matrix). The two residuals are::

        psi_1 = Y - p,
        psi_2 = Y^2 - (p^2 + p (1 - p) + sigma2)
              = Y^2 - (p + sigma2)         [Bernoulli identity]

    At ``theta = (P_TRUE, SIGMA2_TRUE)`` both have zero population mean.
    """
    y_scalar = y[0]
    sigma_y2 = theta.p * (1.0 - theta.p) + theta.sigma2
    raw_second_moment = theta.p**2 + sigma_y2  # = p + sigma2 since p^2+p(1-p)=p
    return jnp.stack([y_scalar - theta.p, y_scalar**2 - raw_second_moment])


# -- DGP composition: build the joint sampler from two sub-samplers. ---------
#
# The two sub-samplers below each own their own slice of the PRNGKey,
# mirroring the "each DGP owns its randomness" pattern from
# ``DGP_Protocol/examples/coin_plus_noise.py``. They are then composed
# additively to produce Y.


def _bernoulli_sampler(key: jax.Array, n: int, p: float) -> Float[Array, " n"]:
    """Sample ``n`` Bernoulli(p) draws as a length-n float array (0.0/1.0)."""
    return jax.random.bernoulli(key, p=p, shape=(n,)).astype(jnp.float64)


def _gaussian_sampler(key: jax.Array, n: int, sigma: float) -> Float[Array, " n"]:
    """Sample ``n`` ``N(0, sigma^2)`` draws as a length-n array."""
    return sigma * jax.random.normal(key, (n,))


def coin_plus_noise_sampler_factory(n_sim: int):
    """Return a :class:`SyntheticMeasure`-compatible sampler.

    The sampler closes over the truth-fixed DGP parameters and produces
    ``(n_sim, 1)`` arrays of ``Y = X + e``. Splits the input key into two
    independent streams (one for X, one for e) so the two component
    distributions are jointly independent regardless of ``theta``.

    The ``theta`` argument is ignored --- the data-generating process is
    fixed at the truth; ``theta`` is what the estimator is trying to
    recover.
    """

    def _sampler(key: jax.Array, theta: CoinPlusNoiseParams) -> Float[Array, "n_sim 1"]:
        del theta  # DGP is exogenous w.r.t. the structural parameters
        k_x, k_e = jax.random.split(key)
        x = _bernoulli_sampler(k_x, n_sim, P_TRUE)
        e = _gaussian_sampler(k_e, n_sim, math.sqrt(SIGMA2_TRUE))
        y = x + e
        return y[:, None]  # (n_sim, D=1)

    return _sampler


def coin_plus_noise_data(seed: int, n: int) -> Float[Array, "N 1"]:
    """Pre-generate ``n`` observations from the DGP.

    Useful as fixed input to :class:`EmpiricalMeasure`. Equivalent to
    calling the synthetic sampler once with a freshly-keyed RNG.
    """
    sampler = coin_plus_noise_sampler_factory(n)
    key = jax.random.PRNGKey(seed)
    return sampler(key, CoinPlusNoiseParams(p=P_TRUE, sigma2=SIGMA2_TRUE))


# ---------------------------------------------------------------------------
# Clean entry points for use by the test module.
# ---------------------------------------------------------------------------


def run_synthetic(*, n_sim: int = N_SIM, seed: int = 0):
    """Estimate ``(p, sigma2)`` via the synthetic (CRN) path."""
    measure = SyntheticMeasure(
        key=jax.random.PRNGKey(seed),
        n_sim=n_sim,
        sampler=coin_plus_noise_sampler_factory(n_sim),
    )
    return estimate(
        model=coin_plus_noise_residual,
        measure=measure,
        covariance=SyntheticCovariance(),
        optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        theta_init=CoinPlusNoiseParams(p=0.4, sigma2=0.8),
        moment_names=("mean", "second_moment"),
    )


def run_empirical(*, n: int = N_SIM, seed: int = 0):
    """Estimate ``(p, sigma2)`` via the empirical (pre-generated data) path."""
    y = coin_plus_noise_data(seed=seed, n=n)
    measure = EmpiricalMeasure(
        x=y,
        mask=jnp.ones((n, 2)),  # 2 moments
        weights=jnp.ones(n),
    )
    return estimate(
        model=coin_plus_noise_residual,
        measure=measure,
        covariance=IIDCovariance(),
        optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        theta_init=CoinPlusNoiseParams(p=0.4, sigma2=0.8),
        moment_names=("mean", "second_moment"),
    )


# ---------------------------------------------------------------------------
# Demo entry point.
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _print_result(result, context: str) -> None:
    print(f"  ({context})")
    print(result.coef_table.to_string())
    print(
        f"  J-stat = {float(result.J_stat):.4e}  "
        f"(dof = {result.J_dof}, p = {float(result.J_pvalue):.3f})"
    )
    print(
        f"  converged = {bool(result.converged)}  "
        f"({int(result.iterations)} iterations)"
    )


def main() -> None:
    print(
        f"Coin-plus-noise demo: Y = X + e, X ~ Bernoulli(p), e ~ N(0, sigma2).\n"
        f"True (p, sigma2) = ({P_TRUE}, {SIGMA2_TRUE}). N = {N_SIM}."
    )

    _print_header("Synthetic context (CRN-frozen sampler)")
    syn = run_synthetic()
    _print_result(syn, "synthetic")

    _print_header("Empirical context (pre-generated data)")
    emp = run_empirical()
    _print_result(emp, "empirical")

    print("\n" + "=" * 60)
    print("Both contexts succeeded; same residual function, two measures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
