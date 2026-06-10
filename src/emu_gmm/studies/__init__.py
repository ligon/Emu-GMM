"""``emu_gmm.studies`` --- the Monte Carlo / repeated-sampling driver (#114).

Industrializes the canonical batching gesture the api-sketch names ---
``tree_map(jnp.stack, *[result.record() for ...])`` --- in three strictly
separated layers (the #114 critical-read shape):

1. **Replication engine** --- :func:`replicate`: an eager Python loop
   ``key -> dgp(fold_in(key, r)) -> run(...) -> result.record()``,
   stacked into :class:`MCRecords`. The #124 traced-measure kernel makes
   each rep a zero-retrace cache hit, so the loop is not the bottleneck;
   a batched ``lax.map`` execution model is a follow-up on #114.
2. **Summarizers** --- :func:`bias_sd`, :func:`coverage`,
   :func:`size_power`, :func:`tau_binding`, :func:`j_calibration`: pure
   numpy reductions over the stacked records, each excluding-but-counting
   non-converged replicates.
3. **Sugar** --- :func:`monte_carlo_study` / :class:`StudyResult`:
   composition by delegation only (the ``Context`` precedent).

Typical use::

    from emu_gmm.studies import monte_carlo_study

    run = build_estimator(psi, measure=template, covariance=..., parameters=p0)
    study = monte_carlo_study(
        run, dgp, n_reps=500, key=jax.random.PRNGKey(0),
        theta_init=p0, theta0=truth,
    )
    study.coverage.coverage      # per-coordinate Wald coverage
    study.records.to_pandas()    # one row per replicate

Deliberately **not** re-exported at the ``emu_gmm`` top level yet: the
studies API gets a release of real use (#130) before its surface is
frozen into the package namespace.
"""

from emu_gmm.studies.driver import MCRecords, replicate
from emu_gmm.studies.study import StudyResult, monte_carlo_study
from emu_gmm.studies.summaries import (
    BiasSD,
    Coverage,
    JCalibration,
    SizePower,
    TauBinding,
    bias_sd,
    coverage,
    j_calibration,
    size_power,
    tau_binding,
)

__all__ = [
    "MCRecords",
    "replicate",
    "BiasSD",
    "Coverage",
    "SizePower",
    "TauBinding",
    "JCalibration",
    "bias_sd",
    "coverage",
    "size_power",
    "tau_binding",
    "j_calibration",
    "StudyResult",
    "monte_carlo_study",
]
