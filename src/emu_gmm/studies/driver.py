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
        triple, ``converged``, ``tau_realised``, ``binding_ridge``.
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

    Returns
    -------
    MCRecords
        Stacked :class:`~emu_gmm.types.FitRecord` plus study metadata.
        Non-converged replicates are recorded, not dropped; their
        ``converged`` field is 0.
    """
    if n_reps < 1:
        raise ValueError(f"replicate(): n_reps must be >= 1, got {n_reps}")
    records: list[FitRecord] = []
    for r in range(n_reps):
        rep_key = jax.random.fold_in(key, r)
        result = run(theta_init, dgp(rep_key))
        records.append(result.record())
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)
    return MCRecords(records=stacked, key=jnp.asarray(key), n_reps=n_reps)


__all__ = ["MCRecords", "replicate"]
