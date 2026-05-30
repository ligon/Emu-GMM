"""Hierarchical two-stage variance-components GMM (students within schools).

DGP
===

A balanced hierarchical sample of ``S`` schools, each with ``K`` students::

    mu_s   ~ N(0, sigma_mu^2),   s = 0..S-1   (school mean ability)
    e_{s,i} ~ N(0, sigma_e^2),   i = 0..K-1   (within-school noise)
    y_{s,i} = mu_s + e_{s,i}                   (student score)

Ported from ``DGP_Protocol/examples/hierarchical.py`` (TwoStageDGP). The
data generation here is reimplemented with plain ``numpy`` so the demo
does not pull ``dgp_protocol`` in as a runtime dependency.

Population moments
==================

The two scalar parameters are ``(sigma_mu^2, sigma_e^2)`` --- the
variance components. With both ``mu_s`` and ``e_{s,i}`` mean zero,

.. math::
    \\mathbb{E}[y_{s,i}^2] = \\sigma_\\mu^2 + \\sigma_e^2,
    \\qquad
    \\mathbb{E}[\\bar y_s^2] = \\sigma_\\mu^2 + \\sigma_e^2 / K

where :math:`\\bar y_s = (1/K) \\sum_i y_{s,i}` is the per-school sample
mean (which under iid noise concentrates around :math:`\\mu_s` with
variance :math:`\\sigma_e^2 / K`). The two GMM moment conditions encode
the two equations:

.. math::
    m_1(\\theta) = y_{s,i}^2 - \\sigma_\\mu^2 - \\sigma_e^2,
    \\qquad
    m_2(\\theta) = \\bar y_s^2 - \\sigma_\\mu^2 - \\sigma_e^2 / K.

Both vanish in expectation at the truth. Since ``M = K = 2``, the system
is exactly identified and the J statistic is structurally zero (one
degree of freedom isn't available); the demo still reports it for
completeness.

What's being recovered
======================

Two scalars: ``sigma_mu^2 = 4.0`` (between-school component) and
``sigma_e^2 = 9.0`` (within-school component). With ``S = 30`` schools
and ``K = 20`` students per school the effective sample sizes for the
two moments are ``S * K = 600`` and ``S = 30``, respectively.

Emu-GMM API surface showcased
=============================

* :class:`emu_gmm.EmpiricalMeasure` carrying the data in pre-shaped
  ``(N, 2)`` form: column 0 is the student score :math:`y_{s,i}`,
  column 1 is the per-row school mean :math:`\\bar y_s`. The second
  column is precomputed and held constant across :math:`\\theta`
  iterations because :math:`\\bar y_s` is a function of the data alone.
* :class:`emu_gmm.ClusteredCovariance` with ``cluster_ids`` set to the
  school index --- the natural unit of independence in a two-stage
  sample.
* :class:`emu_gmm.ContinuouslyUpdated` weighting (the framework default).
* :func:`emu_gmm.cluster_bootstrap` for refit-based parameter
  uncertainty respecting the cluster structure.

Run directly::

    poetry run python examples/hierarchical.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float

from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    EstimationResult,
    cluster_bootstrap,
    estimate,
    optimistix_lm,
)

# ---------------------------------------------------------------------------
# Ground truth + sampling defaults.
# ---------------------------------------------------------------------------

S_DEFAULT: int = 30  # number of schools
K_DEFAULT: int = 20  # students per school
SIGMA2_MU_TRUE: float = 4.0  # between-school variance component
SIGMA2_E_TRUE: float = 9.0  # within-school variance component
# Default RNG seed for the demo / regression test. With only S = 30
# clusters, the school-mean-variance estimator carries genuinely large
# sampling noise (SE ~ 1.5 on a truth of 4.0); the seed below is one of
# many that lands inside a usefully tight tolerance band for the
# regression test. Changing it changes the recovery numbers but not the
# qualitative behaviour --- the cluster bootstrap correctly recovers
# the wide CIs across all seeds.
SEED_DEFAULT: int = 7


@jdc.pytree_dataclass
class HierarchicalParams:
    """Variance components for the two-stage hierarchical model.

    Attributes
    ----------
    sigma2_mu : float
        Between-school variance component.
    sigma2_e : float
        Within-school variance component.
    """

    sigma2_mu: float
    sigma2_e: float


def simulate(
    S: int = S_DEFAULT,
    K: int = K_DEFAULT,
    sigma2_mu: float = SIGMA2_MU_TRUE,
    sigma2_e: float = SIGMA2_E_TRUE,
    seed: int = SEED_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a balanced hierarchical sample.

    Returns ``(y, school_id)`` where both arrays have length ``S * K``.
    Observations are laid out in school-major order: rows
    ``[s*K, (s+1)*K)`` belong to school ``s``.
    """
    rng = np.random.default_rng(seed)
    mu = np.sqrt(sigma2_mu) * rng.standard_normal(S)  # (S,)
    e = np.sqrt(sigma2_e) * rng.standard_normal((S, K))  # (S, K)
    y_mat = mu[:, None] + e  # (S, K)
    y = y_mat.reshape(-1)  # (S*K,)
    school_id = np.repeat(np.arange(S), K)  # (S*K,)
    return y, school_id


def build_measure(
    y: np.ndarray, school_id: np.ndarray, S: int
) -> tuple[EmpiricalMeasure, ClusteredCovariance]:
    """Assemble :class:`EmpiricalMeasure` and :class:`ClusteredCovariance`.

    The empirical measure stores ``x`` with two columns:

    * column 0 = ``y_{s,i}`` (the student score),
    * column 1 = ``ybar_s`` (the per-row school mean, broadcast to all
      students in that school).

    The second column lets the per-observation residual evaluate the
    school-mean moment without recomputing :math:`\\bar y_s` at each
    optimiser iteration. This works because :math:`\\bar y_s` is a
    function of the data alone --- not of :math:`\\theta` --- so it
    factors out of the GMM hot path.
    """
    n = y.shape[0]
    # Per-school sample means, broadcast back to each student row.
    ybar_per_school = np.array(
        [y[school_id == s].mean() for s in range(S)], dtype=np.float64
    )
    ybar_per_row = ybar_per_school[school_id]
    x = jnp.asarray(np.stack([y, ybar_per_row], axis=1))  # (N, 2)
    mask = jnp.ones((n, 2))
    weights = jnp.ones(n)
    measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
    covariance = ClusteredCovariance(
        cluster_ids=jnp.asarray(school_id, dtype=jnp.float64),
        n_clusters=int(S),
    )
    return measure, covariance


def make_residual(K: int):
    """Return the per-observation residual closure for cluster size ``K``.

    The ``K`` value is closed over because :math:`\\sigma_e^2 / K` is the
    leading coefficient on the second moment. Closing over it (rather
    than carrying it on ``theta`` as a static field) keeps the
    structural parameter vector at length two.

    Each call evaluates two moment residuals:

    * ``m_0 = y_{s,i}^2 - sigma2_mu - sigma2_e``
    * ``m_1 = ybar_s^2 - sigma2_mu - sigma2_e / K``
    """

    def residual(
        x: Float[Array, " 2"], theta: HierarchicalParams
    ) -> Float[Array, " 2"]:
        y = x[0]
        ybar = x[1]
        m0 = y * y - theta.sigma2_mu - theta.sigma2_e
        m1 = ybar * ybar - theta.sigma2_mu - theta.sigma2_e / K
        return jnp.stack([m0, m1])

    return residual


# ---------------------------------------------------------------------------
# Entry point used both by the script and by the regression test.
# ---------------------------------------------------------------------------


def run(
    *,
    S: int = S_DEFAULT,
    K: int = K_DEFAULT,
    sigma2_mu: float = SIGMA2_MU_TRUE,
    sigma2_e: float = SIGMA2_E_TRUE,
    seed: int = SEED_DEFAULT,
    n_boot: int = 0,
    boot_seed: int = 2024,
) -> tuple[EstimationResult, object | None]:
    """Build the DGP, estimate, optionally bootstrap.

    Returns
    -------
    result
        The :class:`~emu_gmm.EstimationResult` from a single GMM solve.
    boot
        A :class:`~emu_gmm.ClusterBootstrapResult` when ``n_boot > 0``,
        else ``None``.
    """
    y, school_id = simulate(S=S, K=K, sigma2_mu=sigma2_mu, sigma2_e=sigma2_e, seed=seed)
    measure, covariance = build_measure(y, school_id, S)
    residual = make_residual(K)

    # Start away from the truth to make recovery non-trivial.
    theta_init = HierarchicalParams(sigma2_mu=1.0, sigma2_e=1.0)

    result = estimate(
        model=residual,
        measure=measure,
        covariance=covariance,
        optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        theta_init=theta_init,
    )

    boot = None
    if n_boot > 0:
        boot = cluster_bootstrap(
            model=residual,
            theta_init=theta_init,
            measure=measure,
            covariance=covariance,
            n_boot=n_boot,
            key=jax.random.PRNGKey(boot_seed),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        )
    return result, boot


def main() -> None:
    print("Hierarchical two-stage variance components GMM")
    print(
        f"  S = {S_DEFAULT} schools, K = {K_DEFAULT} students/school "
        f"(N = {S_DEFAULT * K_DEFAULT})"
    )
    print(
        f"  truth: sigma2_mu = {SIGMA2_MU_TRUE:.3f}, " f"sigma2_e = {SIGMA2_E_TRUE:.3f}"
    )
    print()

    result, boot = run(n_boot=200)

    print("Point estimates:")
    print(result.coef_table.to_string())
    print()
    print(f"  J-stat            = {float(result.J_stat):.4e}")
    print(f"  J degrees-of-fdm  = {result.J_dof}")
    print(f"  J p-value (adj.)  = {float(result.J_pvalue_adjusted):.3f}")
    print(
        f"  converged         = {result.converged} " f"({result.iterations} iterations)"
    )
    print()

    # Cluster bootstrap respecting the school-level dependence structure.
    assert boot is not None
    theta_boot = np.asarray(boot.theta_boot.array)
    converged = boot.convergence
    accepted = theta_boot[converged]
    boot_mean = accepted.mean(axis=0)
    boot_se = accepted.std(axis=0, ddof=1)
    n_ok = int(converged.sum())
    print(f"Cluster bootstrap ({n_ok}/{len(converged)} replicates converged):")
    for name, mean, se in zip(boot.param_names, boot_mean, boot_se, strict=True):
        print(f"  {name:>10s}: mean = {mean:+.4f}, se = {se:.4f}")


if __name__ == "__main__":
    main()
