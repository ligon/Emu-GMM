"""Empirical-bootstrap bridge: wild + cluster bootstraps side-by-side.

A re-cast of DGP_Protocol's ``EmpiricalDGP`` / ``IIDSampling`` /
``ClusteredSampling`` showcase using Emu-GMM's two cluster-aware
bootstrap helpers. The pedagogical point: starting from a single
:class:`~emu_gmm.EmpiricalMeasure` paired with a
:class:`~emu_gmm.ClusteredCovariance`, you can run *both*

* :func:`emu_gmm.moment_wild_bootstrap` (refit-free, cluster-wild
  Rademacher) -- gives a bootstrap distribution of the J-statistic and a
  bootstrap p-value calibrated to the analytic cluster-robust variance.
* :func:`emu_gmm.cluster_bootstrap` (refit-based, pairs / non-parametric
  cluster bootstrap) -- gives a bootstrap distribution of ``theta_hat``
  and a corresponding bootstrap standard-error estimate.

The two are not interchangeable: the refit-free version answers the
"is the analytic asymptotic distribution well-calibrated?" question and
is cheap (no re-optimisation); the refit-based version answers the
"what does the sampling distribution of theta_hat look like *including*
the GMM map's nonlinearity?" question and is more expensive. Both
should yield SE estimates that agree up to Monte-Carlo error when the
sample size is moderate and identification is strong.

DGP
---

We synthesise ``N = n_clusters * obs_per_cluster`` observations with
intra-cluster correlation:

.. math::
   x_i \\;=\\; \\mu_0 + u_{c(i)} + e_i,
   \\qquad u_c \\sim N(0, \\sigma_u^2),\\;
   e_i \\sim N(0, \\sigma_e^2),\\; \\text{indep.}

so the within-cluster correlation is
:math:`\\rho = \\sigma_u^2 / (\\sigma_u^2 + \\sigma_e^2)`. The cluster
bootstrap preserves this correlation; an iid bootstrap would destroy
it. ``ClusteredCovariance`` is the analytic counterpart that captures
the same effect on the GMM variance.

Population moments and recovery
-------------------------------

With :math:`\\theta = (\\mu, \\sigma^2)`, total variance
:math:`\\sigma^2_\\text{tot} = \\sigma_u^2 + \\sigma_e^2`, we use three
moments,

.. math::
   \\psi(x, \\theta) =
   \\begin{pmatrix}
     x - \\mu \\\\
     (x - \\mu)^2 - \\sigma^2 \\\\
     (x - \\mu)^3
   \\end{pmatrix},

with :math:`E[\\psi] = 0` at :math:`(\\mu_0, \\sigma^2_\\text{tot})`
because the underlying distribution is symmetric about :math:`\\mu_0`.
Three moments on two parameters over-identifies the model so the J-test
is informative.

API surface
-----------

This example exercises:

* :class:`emu_gmm.EmpiricalMeasure` -- the raw-data measure.
* :class:`emu_gmm.ClusteredCovariance` -- cluster-totals variance.
* :func:`emu_gmm.estimate` -- the GMM entry point.
* :func:`emu_gmm.moment_wild_bootstrap` -- refit-free wild bootstrap.
* :func:`emu_gmm.cluster_bootstrap` -- refit-based pairs bootstrap.
* :attr:`emu_gmm.EstimationResult.standard_errors` --- analytic SE
  the bootstrap SEs are benchmarked against.
* :attr:`emu_gmm.EstimationResult.coef_table` -- the headline tabular
  output.

Run directly::

    poetry run python examples/empirical_bootstrap.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas as pd

from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    cluster_bootstrap,
    estimate,
    moment_wild_bootstrap,
    optimistix_lm,
)

# ---------------------------------------------------------------------------
# Ground truth + DGP knobs.
# ---------------------------------------------------------------------------
MU_TRUE: float = 1.5
SIGMA_U: float = 0.6  # cluster effect SD
SIGMA_E: float = 0.6  # within-cluster noise SD
SIGMA2_TRUE: float = SIGMA_U**2 + SIGMA_E**2  # total variance

N_CLUSTERS: int = 60
OBS_PER_CLUSTER: int = 12
N: int = N_CLUSTERS * OBS_PER_CLUSTER  # 720 rows

DATA_SEED: int = 2029
BOOT_SEED: int = 7
# Number of bootstrap replicates for each helper. The refit-based
# cluster bootstrap dominates the runtime cost: each replicate re-solves
# the GMM problem and JAX re-traces the residual function on each call
# inside the resample loop. 150 is comfortably above the +/-10% MC-noise
# floor for the bootstrap-SE estimator (the chi-squared SE of an SD from
# B draws is roughly 1 / sqrt(2(B - 1))) and stays inside the
# LLVM JIT memory budget on a typical workstation. Increase if you want
# tighter MC bounds and have headroom for several hundred fresh
# compilations; setting JAX_COMPILATION_CACHE_DIR makes that affordable.
N_BOOT: int = 150


@jdc.pytree_dataclass
class MeanVarParams:
    """Location ``mu`` and variance ``sigma2``."""

    mu: float
    sigma2: float


def mean_var_residual(x, theta):
    """Per-observation residual: (x - mu, (x - mu)^2 - sigma2, (x - mu)^3).

    ``x`` is a length-1 vector here -- :class:`EmpiricalMeasure` stores
    observations row-wise as shape ``(N, D)`` and the per-observation
    residual receives a single row at a time under ``jax.vmap``.
    """
    centred = x[0] - theta.mu
    m1 = centred
    m2 = centred**2 - theta.sigma2
    m3 = centred**3
    return jnp.stack([m1, m2, m3])


def make_dataset(
    *,
    seed: int = DATA_SEED,
    n_clusters: int = N_CLUSTERS,
    obs_per_cluster: int = OBS_PER_CLUSTER,
    mu: float = MU_TRUE,
    sigma_u: float = SIGMA_U,
    sigma_e: float = SIGMA_E,
):
    """Draw the synthetic clustered dataset.

    Returns
    -------
    measure : :class:`EmpiricalMeasure`
        The (N, 1) observations packaged as a measure.
    covariance : :class:`ClusteredCovariance`
        The cluster-aware variance strategy matching the layout.
    cluster_ids : :class:`numpy.ndarray`
        The per-observation cluster labels (for diagnostic display).
    """
    n = n_clusters * obs_per_cluster
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=sigma_u, size=n_clusters)
    e = rng.normal(scale=sigma_e, size=(n_clusters, obs_per_cluster))
    x_2d = mu + u[:, None] + e  # (n_clusters, obs_per_cluster)
    x = jnp.asarray(x_2d.reshape(n, 1))
    # mask is (N, M) where M is the number of moments (3), not D (1).
    measure = EmpiricalMeasure(
        x=x,
        mask=jnp.ones((n, 3)),
        weights=jnp.ones(n),
    )
    cluster_ids = jnp.repeat(jnp.arange(n_clusters, dtype=jnp.float64), obs_per_cluster)
    covariance = ClusteredCovariance(
        cluster_ids=cluster_ids,
        n_clusters=n_clusters,
    )
    return measure, covariance, np.asarray(cluster_ids).astype(int)


def fit(measure, covariance):
    """Run :func:`emu_gmm.estimate` from a generic starting point."""
    return estimate(
        model=mean_var_residual,
        measure=measure,
        covariance=covariance,
        optimizer=optimistix_lm(rtol=1e-8),
        theta_init=MeanVarParams(mu=0.0, sigma2=1.0),
    )


def run(
    *,
    data_seed: int = DATA_SEED,
    boot_seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
):
    """Estimate, run both bootstraps, return the comparison.

    Returns
    -------
    result : :class:`emu_gmm.EstimationResult`
        The point estimate + analytic-variance container.
    wild : :class:`emu_gmm.WildBootstrapResult`
        Refit-free cluster-wild bootstrap.
    cluster : :class:`emu_gmm.ClusterBootstrapResult`
        Refit-based pairs cluster bootstrap.
    se_table : :class:`pandas.DataFrame`
        Side-by-side analytic vs cluster-bootstrap SE table.
    """
    measure, covariance, _ = make_dataset(seed=data_seed)
    result = fit(measure, covariance)

    boot_key = jax.random.PRNGKey(boot_seed)
    wild_key, cluster_key = jax.random.split(boot_key)

    # Refit-free: cluster-wild Rademacher on the moments.
    wild = moment_wild_bootstrap(
        mean_var_residual,
        result.theta_hat,
        measure,
        covariance,
        n_boot=n_boot,
        key=wild_key,
        sign="rademacher",
        V=result.V_X,
    )

    # Refit-based: resample whole clusters, re-solve.
    cluster = cluster_bootstrap(
        model=mean_var_residual,
        theta_init=MeanVarParams(mu=0.0, sigma2=1.0),
        measure=measure,
        covariance=covariance,
        n_boot=n_boot,
        key=cluster_key,
    )

    # Bootstrap SE from the refit distribution. theta_boot is a haliax
    # NamedArray with axes (bootstrap, parameters); reduce along the
    # bootstrap axis on the underlying jnp array.
    theta_boot_arr = np.asarray(cluster.theta_boot.array)
    converged = np.asarray(cluster.convergence)
    if converged.any():
        boot_se = np.nanstd(theta_boot_arr[converged], axis=0, ddof=1)
    else:
        boot_se = np.full(theta_boot_arr.shape[1], np.nan)

    analytic_se = np.asarray(result.standard_errors.array)
    point = np.asarray(result.coef_table["estimate"].to_numpy())  # estimate column

    se_table = pd.DataFrame(
        {
            "estimate": point,
            "analytic_SE": analytic_se,
            "cluster_boot_SE": boot_se,
            "ratio_boot_over_analytic": boot_se / analytic_se,
        },
        index=list(cluster.param_names),
    )
    return result, wild, cluster, se_table


def main() -> None:
    print(
        "Empirical bootstrap bridge: comparing the cluster-wild bootstrap "
        "of the J-stat\nagainst the refit-based pairs cluster bootstrap of "
        "theta_hat."
    )
    print(
        f"DGP: N = {N} ({N_CLUSTERS} clusters x {OBS_PER_CLUSTER} rows), "
        f"mu_true = {MU_TRUE}, sigma2_true = {SIGMA2_TRUE:.4f}."
    )
    print(
        f"Bootstrap replicates per helper: {N_BOOT}. "
        f"Random seeds: data = {DATA_SEED}, bootstrap = {BOOT_SEED}.\n"
    )

    result, wild, cluster, se_table = run()

    print("=" * 60)
    print("Point estimates (coef_table)")
    print("=" * 60)
    print(result.coef_table.to_string())
    print()

    print("=" * 60)
    print("Analytic J-test")
    print("=" * 60)
    print(
        f"J_stat = {float(result.J_stat):.4f}   "
        f"(dof = {result.J_dof}, nominal p = {float(result.J_pvalue):.3f}, "
        f"adjusted p = {float(result.J_pvalue_adjusted):.3f})"
    )
    print()

    print("=" * 60)
    print("Refit-free cluster-wild bootstrap of the J-stat")
    print("=" * 60)
    j_boot = np.asarray(wild.J_boot)
    print(
        f"J_observed = {float(wild.J_observed):.4f}   "
        f"bootstrap p = {float(wild.p_value):.3f}   "
        f"(sign = {wild.sign}, n_boot = {wild.n_boot})"
    )
    print(
        f"J_boot summary: mean = {j_boot.mean():.3f}, "
        f"median = {float(np.median(j_boot)):.3f}, "
        f"95th pct = {float(np.quantile(j_boot, 0.95)):.3f}"
    )
    print()

    print("=" * 60)
    print("Refit-based pairs cluster bootstrap of theta_hat")
    print("=" * 60)
    n_converged = int(np.asarray(cluster.convergence).sum())
    print(f"Replicates converged: {n_converged} / {cluster.theta_boot.array.shape[0]}")
    print()
    print("Side-by-side SE comparison:")
    print(se_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(
        "Interpretation: a ratio near 1.0 means the analytic CLT-based SE\n"
        "and the resampling SE agree. They are different estimators of the\n"
        "same sampling-distribution width; modest disagreement (within ~20%)\n"
        "is expected from MC noise and from the refit-based estimator's\n"
        "ability to pick up nonlinearity that the asymptotic linearisation\n"
        "misses."
    )


if __name__ == "__main__":
    main()
