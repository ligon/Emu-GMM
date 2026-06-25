"""Layer 1 of the Monte Carlo driver (#114): :func:`replicate`.

The only genuinely new machinery in the #114 cluster: the replication
engine ``key -> dataset -> estimate -> record``. Per-rep statistics are
owned by :class:`~emu_gmm.types.EstimationResult` (via ``.record()``,
#125) and aggregation by the layer-2 summarizers in
:mod:`emu_gmm.studies.summaries`; this module only runs the loop and
stacks the records.

Execution model (v1, deliberate)
--------------------------------
An **eager Python loop** over replicates. The #124 traced-measure kernel
makes each rep a cache-hit call (~5 ms/rep, zero retraces for fresh
same-structure measures), so the Python loop is not the bottleneck. The
batched ``lax.map``-over-stacked-datasets path is a follow-up tracked on
#114: it needs a slim *in-trace* kernel record (today ``FitRecord`` is
assembled host-side from ``EstimationResult``, which is deliberately a
host-side leaf), and on CPU at study scale it measured comparable to the
traced-arg loop anyway (#124 spike).

Common random numbers (CRN)
---------------------------
Replicate ``r`` always sees ``jax.random.fold_in(key, r)``. Two studies
(arms) run with the same master ``key`` and the same ``dgp`` therefore
estimate on **identical draws** rep-for-rep --- the study-level extension
of the CRN principle ``SyntheticMeasure`` is built on, and what power
curves and covariance-strategy comparisons require. Any single replicate
is reproducible from ``(key, r)`` alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas as pd

from emu_gmm.types import EstimationResult, FitRecord, Measure


@jdc.pytree_dataclass
class MCRecords:
    """The stacked output of :func:`replicate`.

    A thin pytree wrapper around the ``tree_map(jnp.stack)``-stacked
    :class:`~emu_gmm.types.FitRecord` --- every array field of
    ``records`` carries a leading replication axis of length
    ``n_reps`` --- plus the study metadata needed for reproducibility
    and exclusion accounting.

    Chosen over a bare ``(FitRecord, tuple)`` return because the
    summarizers, the ``to_pandas`` materializer, and the
    exclude-but-count bookkeeping all want one self-describing object;
    registered as a pytree so the stack remains jit/vmap-traversable
    (the api-sketch "results are pytrees" invariant).

    Non-convergence policy: **all** replicates are recorded
    (``records.converged`` is the 0/1 flag); nothing is dropped here.
    Summarizers exclude-but-count (the adaptive-bootstrap precedent,
    #91).
    """

    #: Stacked per-rep records; leading axis ``n_reps`` on every array.
    records: FitRecord
    #: The master PRNG key; rep ``r`` used ``fold_in(key, r)``.
    key: jax.Array
    #: Number of replicates run (static; the leading-axis length).
    n_reps: int = jdc.static_field()  # type: ignore[attr-defined]
    #: Optional CRN-coupling token (static). Two arms are CRN-coupled iff
    #: they were drawn from the *same* DGP/fold-in stream; master-key
    #: equality is necessary but NOT sufficient to witness that (the key
    #: does not see the DGP's internal ``split`` scheme). Stamp a value
    #: here -- any hashable identifying the (DGP, master-key) stream --
    #: and :func:`emu_gmm.studies.crn_pair` verifies it before zipping two
    #: arms. ``None`` (the default) means "coupling unverifiable"; pairing
    #: then requires an explicit ``assert_coupled=True``.
    coupling_id: Any = jdc.static_field(default=None)  # type: ignore[attr-defined]

    @property
    def converged_mask(self) -> np.ndarray:
        """Boolean ``(n_reps,)`` host array: which reps converged."""
        return np.asarray(self.records.converged) > 0.5

    @property
    def n_converged(self) -> int:
        """Number of converged replicates."""
        return int(self.converged_mask.sum())

    @property
    def n_excluded(self) -> int:
        """Replicates the summarizers will exclude (non-converged)."""
        return self.n_reps - self.n_converged

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter labels on the ambient flat axis (static)."""
        return tuple(self.records.param_names)

    def to_pandas(self) -> pd.DataFrame:
        """One row per replicate; host-side materialization.

        Columns: ``theta_<name>`` / ``se_<name>`` per parameter, the J
        triple, ``converged``, ``tau_realised``, ``binding_ridge``,
        ``sigma_meat_indefinite`` (the #138 NaN-SE event; #143).
        Pandas stays outside the compiled boundary --- this is the only
        pandas touchpoint in the studies module.
        """
        rec = self.records
        data: dict[str, np.ndarray] = {}
        theta = np.asarray(rec.theta_flat)
        se = np.asarray(rec.se)
        for d, name in enumerate(self.param_names):
            data[f"theta_{name}"] = theta[:, d]
            data[f"se_{name}"] = se[:, d]
        data["J_stat"] = np.asarray(rec.J_stat)
        data["J_pvalue"] = np.asarray(rec.J_pvalue)
        data["J_pvalue_adjusted"] = np.asarray(rec.J_pvalue_adjusted)
        data["converged"] = np.asarray(rec.converged)
        data["tau_realised"] = np.asarray(rec.tau_realised)
        data["binding_ridge"] = np.asarray(rec.binding_ridge)
        data["sigma_meat_indefinite"] = np.asarray(rec.sigma_meat_indefinite)
        df = pd.DataFrame(data)
        df.index.name = "rep"
        return df


def replicate(
    run: Callable[[Any, Measure], EstimationResult],
    dgp: Callable[[jax.Array], Measure],
    *,
    n_reps: int,
    key: jax.Array,
    theta_init: Any,
    anchor_per_rep: bool = False,
    coupling_id: Any = None,
) -> MCRecords:
    """Run ``n_reps`` independent draw-and-estimate replicates.

    Parameters
    ----------
    run
        A fitted-estimator callable ``run(theta_init, measure) ->
        EstimationResult`` --- i.e. the return value of
        :func:`emu_gmm.build_estimator`. Taking the callable (rather
        than rebuilding internally from ``(model, covariance, ...)``)
        keeps the driver from duplicating ``build_estimator``'s kwarg
        surface, preserves measure x covariance orthogonality at the
        call site, and makes CRN arm comparisons natural: build one
        ``run`` per arm, call :func:`replicate` with the **same**
        ``key`` and ``dgp``, and the arms see identical draws.
    dgp
        The data-generating hook ``dgp(rep_key) -> Measure``. Returning
        a full :class:`~emu_gmm.types.Measure` (not a bare array) is
        what lets masked / weighted / clustered designs be first-class
        study designs --- an ``(n, D)`` array hook could only express
        the balanced ``mask=ones`` case that commitment 9 warns is
        blind to the per-coordinate ``N_j`` scaling bugs. Each
        replicate's measure must share the template structure ``run``
        was built with (same class, same shapes) to ride the
        zero-retrace kernel path (#124); a new shape retraces once and
        is then cached.
    n_reps
        Number of replicates.
    key
        Master PRNG key. Replicate ``r`` receives
        ``jax.random.fold_in(key, r)``; the scheme is part of the
        public contract (see the module docstring on CRN).
    theta_init
        Starting point passed to ``run`` for every replicate.
    anchor_per_rep
        Default ``False``: the fast factory path. ``run`` is invoked
        as-is for every replicate, so the regularisation anchor
        ``tau_anchor`` that :func:`~emu_gmm.build_estimator` froze on
        its TEMPLATE measure (the anchor-once-then-freeze policy;
        CLAUDE.md commitment 3) is shared by every replicate ---
        ``FitRecord.binding_ridge`` is then constant across reps, an
        unlucky template draw can poison a whole arm, and a lucky one
        can mask per-rep pathology (#142).

        ``True``: each replicate runs the bare
        :func:`~emu_gmm.estimator.estimate` path, so the
        anchor-once-then-freeze policy applies **per dataset** (per-fit
        == per-dataset semantics). Use it whenever ``tau_anchor > 0``
        matters --- in particular whenever the tau-binding column is a
        study deliverable (the #130 harness's
        ``run_arm_per_rep_anchor`` is the prior art and can migrate to
        this flag). Requires ``run`` to be a
        :func:`~emu_gmm.build_estimator` factory: the driver reads the
        construction kwargs off ``run._emu_gmm_factory_spec`` (attached
        by the factory) rather than duplicating ``build_estimator``'s
        kwarg surface; a hand-rolled ``run`` callable raises
        :class:`ValueError`. Cost: the slow path --- every replicate
        pays the full closure-build + retrace, ~seconds/rep instead of
        ~ms/rep. The driver calls ``jax.clear_caches()`` every 25
        replicates on this path: bare ``estimate()`` builds fresh
        closures per call, so JAX's global caches accumulate write-only
        traces (~14 MB/call measured; the unmitigated leak OOM-killed a
        300-rep study at 9.4 GB --- the #139 merge-verification
        thread) --- but clearing also drops XLA's re-hit kernel
        compilations, so per-replicate clearing costs ~3-5x wall-clock;
        every-25 bounds the swing at ~350 MB with a few percent
        recompilation overhead. The CRN contract is unchanged:
        replicate ``r`` draws
        with ``fold_in(key, r)`` on both paths, so the two modes see
        identical datasets and differ only in anchoring.
    coupling_id
        Optional CRN-coupling token stamped onto the returned
        :class:`MCRecords` (default ``None``). Give two arms drawn from
        the *same* ``(dgp, key)`` stream the *same* value, and
        :func:`emu_gmm.studies.crn_pair` will verify it before zipping
        them. Master-key equality alone is necessary but not sufficient
        evidence of coupling (it does not witness the DGP's internal
        ``split`` scheme), so unmatched / ``None`` ids force an explicit
        ``assert_coupled=True`` at pairing time.

    Returns
    -------
    MCRecords
        Stacked :class:`~emu_gmm.types.FitRecord` plus study metadata.
        Non-converged replicates are recorded, not dropped; their
        ``converged`` field is 0.
    """
    if n_reps < 1:
        raise ValueError(f"replicate(): n_reps must be >= 1, got {n_reps}")
    fit_per_rep: Callable[[Measure], EstimationResult] | None = None
    if anchor_per_rep:
        spec = getattr(run, "_emu_gmm_factory_spec", None)
        if spec is None:
            raise ValueError(
                "replicate(anchor_per_rep=True) requires `run` to be a "
                "factory built by emu_gmm.build_estimator: the per-rep "
                "anchoring path rebuilds a bare estimate() per replicate "
                "from the construction kwargs the factory attaches to its "
                "returned callable (run._emu_gmm_factory_spec), and the "
                "supplied callable does not carry them. Build `run` with "
                "build_estimator(...), or drop anchor_per_rep."
            )
        # Local import: emu_gmm/__init__ imports this module while wiring
        # the public API, so a module-level import of the estimator would
        # be load-order sensitive (the _resolve_parameters precedent).
        from emu_gmm.estimator import estimate

        def _fit_fresh_anchor(measure: Measure) -> EstimationResult:
            return estimate(
                spec["model"],
                measure,
                covariance=spec["covariance"],
                weighting=spec["weighting"],
                regularization=spec["regularization"],
                optimizer=spec["optimizer"],
                parameters=theta_init,
                moment_names=spec["moment_names"],
                penalty=spec["penalty"],
            )

        fit_per_rep = _fit_fresh_anchor

    records: list[FitRecord] = []
    for r in range(n_reps):
        rep_key = jax.random.fold_in(key, r)
        if fit_per_rep is not None:
            result = fit_per_rep(dgp(rep_key))
            records.append(result.record())
            # Bare estimate() builds fresh closures per call, so JAX's
            # global caches accumulate write-only traces (~14 MB/call
            # measured; the unmitigated leak OOM-killed a 300-rep study
            # at 9.4 GB -- the #139 merge-verification thread). But
            # clearing ALSO drops XLA's re-hit kernel compilations:
            # per-rep clearing measured ~3-5x wall-clock (#139 thread,
            # the #130 re-run). Clear every 25 reps: ~350 MB swing,
            # recompilation amortized to a few percent.
            if (r + 1) % 25 == 0:
                jax.clear_caches()
        else:
            result = run(theta_init, dgp(rep_key))
            records.append(result.record())
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)
    return MCRecords(
        records=stacked, key=jnp.asarray(key), n_reps=n_reps, coupling_id=coupling_id
    )


__all__ = ["MCRecords", "replicate"]
