"""Runnable end-to-end demo of the multi-asset Euler equation.

Runs the same StructuralModel through three Measure / CovarianceStrategy
pairings (synthetic, analytical, empirical) and prints recovery and the
J-statistic for each. Demonstrates the framework's central architectural
claim: one operator interface, three implementations, one estimate() call.

Run from the repo root with::

    poetry run python examples/run_euler.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    EmpiricalMeasure,
    IIDCovariance,
    SyntheticCovariance,
    SyntheticMeasure,
    estimate,
    optimistix_lm,
    scipy_lm,
)
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    N_ASSETS,
    EulerParams,
    euler_analytical_expectation,
    euler_data,
    euler_residual,
    euler_sampler_factory,
)

N_SIM = 5000
THETA_INIT = EulerParams(beta=0.9, gamma=1.0)


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _print_result(result, *, recover_atol_beta: float, recover_atol_gamma: float) -> None:
    beta = float(result.theta_hat.beta)
    gamma = float(result.theta_hat.gamma)
    print(f"  beta  = {beta:.6f}   (truth {BETA_TRUE:.2f}, |err| = {abs(beta - BETA_TRUE):.2e})")
    print(f"  gamma = {gamma:.6f}   (truth {GAMMA_TRUE:.2f}, |err| = {abs(gamma - GAMMA_TRUE):.2e})")
    print(
        f"  J-stat = {result.J_stat:.4e}   "
        f"(dof = {result.J_dof}, p = {result.J_pvalue:.3f})"
    )
    print(
        f"  converged = {result.converged}   "
        f"({result.iterations} iterations, "
        f"final_grad_norm = {result.diagnostics.final_gradient_norm:.2e})"
    )
    if result.diagnostics.binding_ridge:
        print(
            f"  WARNING: regularisation tau = "
            f"{result.diagnostics.tau_realised:.3e} was binding"
        )

    # Sanity asserts so the script blows up if recovery regresses.
    assert abs(beta - BETA_TRUE) < recover_atol_beta, f"beta off by {abs(beta - BETA_TRUE)}"
    assert abs(gamma - GAMMA_TRUE) < recover_atol_gamma, f"gamma off by {abs(gamma - GAMMA_TRUE)}"


def run_synthetic() -> None:
    _print_header("Synthetic context (CRN-frozen sampler)")
    measure = SyntheticMeasure(
        key=jax.random.PRNGKey(0),
        n_sim=N_SIM,
        sampler=euler_sampler_factory(N_SIM),
    )
    result = estimate(
        model=euler_residual,
        measure=measure,
        covariance=SyntheticCovariance(),
        optimizer=optimistix_lm(rtol=1e-8),
        theta_init=THETA_INIT,
    )
    _print_result(result, recover_atol_beta=0.05, recover_atol_gamma=0.5)


def run_analytical() -> None:
    _print_header("Analytical context (closed-form expectation)")

    def identity_covariance(model, theta):
        return jnp.eye(N_ASSETS)

    result = estimate(
        model=euler_residual,
        measure=AnalyticalMeasure(expectation_fn=euler_analytical_expectation),
        covariance=AnalyticalCovariance(covariance_fn=identity_covariance),
        optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
        theta_init=EulerParams(beta=0.8, gamma=1.0),
    )
    _print_result(result, recover_atol_beta=1e-4, recover_atol_gamma=1e-4)


def run_empirical() -> None:
    _print_header("Empirical context (pre-generated data)")
    n = N_SIM
    x = euler_data(seed=0, n=n)
    measure = EmpiricalMeasure(
        x=x,
        mask=jnp.ones((n, N_ASSETS)),
        weights=jnp.ones(n),
    )
    result = estimate(
        model=euler_residual,
        measure=measure,
        covariance=IIDCovariance(),
        optimizer=scipy_lm(),
        theta_init=THETA_INIT,
    )
    _print_result(result, recover_atol_beta=0.05, recover_atol_gamma=0.5)
    print("\n  Sigma_theta (labelled, as DataFrame):")
    print(result.to_pandas()["Sigma_theta"].to_string())


def main() -> None:
    print(
        f"Multi-asset Euler demo: {N_ASSETS} assets, "
        f"true (beta, gamma) = ({BETA_TRUE}, {GAMMA_TRUE})."
    )
    print(f"Estimating from {N_SIM} draws (where applicable) via three contexts.")

    run_synthetic()
    run_analytical()
    run_empirical()

    print("\n" + "=" * 60)
    print("All three contexts succeeded.")
    print("Same StructuralModel; only Measure and CovarianceStrategy changed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
