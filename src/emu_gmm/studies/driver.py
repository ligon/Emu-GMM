"""Layer 1 of the Monte Carlo driver (#114): :func:`replicate`.

The only genuinely new machinery in the #114 cluster: the replication
engine ``key -> dataset -> estimate -> record``. The default per-fit
projection is :func:`fit_record` (#125); custom procedures can supply
their own record extractor. Aggregation belongs to the layer-2 summarizers
in :mod:`emu_gmm.studies.summaries`; the driver only runs the loop and
stacks the records.

Execution model (v1, deliberate)
--------------------------------
An **eager Python loop** over replicates. The #124 traced-measure kernel
makes each rep a cache-hit call (~5 ms/rep, zero retraces for fresh
same-structure measures), so the Python loop is not the bottleneck. The
batched ``lax.map``-over-stacked-datasets path is a follow-up tracked on
#114: it needs a slim *in-trace* kernel record (today ``FitRecord`` is
assembled host-side from ``OptimizationResult``, which is deliberately a
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

import dataclasses
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas as pd

from emu_gmm.types import (
    FitRecord,
    Measure,
    OptimizationResult,
    _flat_theta,
    _result_param_names,
)


def _rep_range(n_reps: int | None, reps: range | None) -> range:
    """Resolve a draw selection without uint32 fold-in collisions (ledger 9.3)."""
    if (n_reps is None) == (reps is None):
        raise ValueError("Supply exactly one of n_reps or reps.")
    if reps is None:
        if isinstance(n_reps, bool) or not isinstance(n_reps, int | np.integer):
            raise ValueError("n_reps must be a positive integer.")
        if n_reps < 1:
            raise ValueError("n_reps must be >= 1.")
        reps = range(int(n_reps))
    if not isinstance(reps, range):
        raise ValueError("reps must be a range with step=1.")
    if reps.step != 1 or reps.stop <= reps.start:
        raise ValueError("reps must be a nonempty increasing range with step=1.")
    if reps.start < 0 or reps[-1] > np.iinfo(np.uint32).max:
        raise ValueError("rep indices must lie in the uint32 fold_in domain.")
    return reps


def _keys_equal(a: jax.Array | None, b: jax.Array | None) -> bool:
    """Compare typed and legacy PRNG keys; an absent key proves nothing."""
    if a is None or b is None:
        return False
    return bool(
        str(jax.random.key_impl(a)) == str(jax.random.key_impl(b))
        and np.array_equal(
            np.asarray(jax.random.key_data(a)), np.asarray(jax.random.key_data(b))
        )
    )


def fit_record(
    result: OptimizationResult,
    *,
    f: Callable[[tuple[Any, ...]], Any] | None = None,
    names: tuple[str, ...] | None = None,
) -> FitRecord:
    """The slim, stackable per-fit summary pytree for one fit (#125).

    The empirical-law bridge: it extracts exactly the fields
    repeated-estimation consumers need --- ``theta_flat`` (manifold-aware
    ambient flatten, the ``Sigma_theta`` axis), ``se``, the J triple,
    ``converged``, ``tau_realised``, ``binding_ridge``,
    ``sigma_meat_indefinite`` (the #138 NaN-SE event; #143),
    ``tau_saturated`` (the #205 ridge-saturation event) --- as a
    :class:`~emu_gmm.types.FitRecord` ready for ``tree_map(jnp.stack,
    *records)``.

    This lives in the studies layer, not on the
    :class:`~emu_gmm.types.OptimizationResult`, because it is inherently
    *statistical*: the per-rep ``se`` and the J p-values are inference
    quantities, so they are computed through the fit's
    :class:`~emu_gmm.law.AsymptoticLaw` (the same shared #133 sandwich the
    live law asserts), not off the bare optimization result
    (docs/optimization-result-law-split.org).

    With ``f``, record ``f(result.components())`` and its full-covariance
    delta-method SE from the same law. ``f`` must be JAX-differentiable,
    gauge-invariant, and return a scalar or a 1-D array. ``names`` labels
    that codomain (default ``f_0``, ``f_1``, ...); without ``f``, labels
    default to the fit's ambient parameter names. Supplied names must be
    distinct and match the recorded dimension.

    The original fit's J triple, degrees of freedom, convergence, and named
    events are preserved. Pass these records to ``bias_sd`` / ``coverage``
    with truth on the functional axis. This summarizes the supplied Wald
    procedure; it does not establish valid inference at rank boundaries or
    repeated eigenvalues. Functional SEs require the live joint covariance
    and cannot be recovered from historical marginal-SE records.
    """
    import warnings

    from emu_gmm.law import AsymptoticLaw

    law = AsymptoticLaw(result)
    # Reuse the joint-covariance delta method, never marginal SEs (ledger 8.9).
    if f is None:
        theta_arr = _flat_theta(result.theta_hat, result.manifold_spec)
        param_names = _result_param_names(result)
        se_arr = jnp.asarray(law.se())
    else:
        theta_arr = jnp.atleast_1d(jnp.asarray(f(result.components())))
        if theta_arr.ndim != 1:
            raise ValueError(
                "fit_record(): f must return a scalar or a 1-D array; "
                f"got shape {tuple(theta_arr.shape)}."
            )
        param_names = tuple(f"f_{i}" for i in range(theta_arr.shape[0]))
        se_arr = jnp.asarray(law.functional_se(f)[0])
    if names is not None:
        param_names = tuple(names)
        if not all(isinstance(name, str) for name in param_names):
            raise TypeError("fit_record(): names must contain only strings.")
        if len(param_names) != int(theta_arr.shape[0]):
            raise ValueError(
                "fit_record(): names length "
                f"{len(param_names)} does not match recorded dimension "
                f"{int(theta_arr.shape[0])}."
            )
        if len(set(param_names)) != len(param_names):
            raise ValueError("fit_record(): names must be distinct; got duplicates.")
    if int(se_arr.shape[0]) != int(theta_arr.shape[0]):
        raise ValueError(
            "fit_record(): recorded estimate length "
            f"{int(theta_arr.shape[0])} does not match the SE axis "
            f"{int(se_arr.shape[0])}; this indicates a manifold-spec routing "
            "bug (mirrors the coef_table guard)."
        )
    diag = result.diagnostics
    assert diag is not None
    # Under a non-efficient weighting the law reports the J p-values as NaN
    # (the #188 guard) --- the correct, honest value to RECORD (a J-calibration
    # study then surfaces NaN rather than a falsely-precise miscalibrated curve).
    # The accompanying interactive warning belongs on a live-law read, not this
    # per-rep batch builder, so silence just that one message here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*non-efficient weighting.*", category=UserWarning
        )
        j_pvalue = jnp.asarray(law.J_pvalue)
        j_pvalue_adjusted = jnp.asarray(law.J_pvalue_adjusted)
    return FitRecord(
        theta_flat=theta_arr,
        se=se_arr,
        J_stat=jnp.asarray(result.objective_value),
        J_pvalue=j_pvalue,
        J_pvalue_adjusted=j_pvalue_adjusted,
        converged=jnp.asarray(result.converged, dtype=jnp.float64),
        tau_realised=jnp.asarray(diag.tau_realised),
        binding_ridge=jnp.asarray(diag.binding_ridge, dtype=jnp.float64),
        sigma_meat_indefinite=jnp.asarray(
            diag.sigma_meat_indefinite, dtype=jnp.float64
        ),
        # A pre-#205 pickled result's Diagnostics resolves this through the
        # class-level dataclass default (False), so old results record 0.0.
        tau_saturated=jnp.asarray(diag.tau_saturated, dtype=jnp.float64),
        J_dof=int(result.n_overid),
        param_names=param_names,
    )


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
    #: The master PRNG key; rep ``r`` used ``fold_in(key, r)`` when present.
    #: A manually supplied keyless carrier can be summarized and persisted,
    #: but cannot prove CRN coupling or concatenate with another shard.
    key: jax.Array | None
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
    #: Optional per-draw custom statistics collected by ``replicate(statistics=)``
    #: (#179). A ``{name: (n_reps, *shape)-array}`` dict, leading rep axis,
    #: with a fixed per-draw shape for each statistic, or ``None``
    #: when no extractors were supplied. This is the general
    #: "let the harness collect a statistic the FitRecord schema doesn't carry"
    #: channel --- e.g. an identification-robust K p-value at a null, an
    #: identification-strength scalar --- each computed by package code from the
    #: harness's own ``(OptimizationResult, Measure)`` draw. A traced pytree child
    #: (not static): it rides jit/vmap and stacks like ``records``.
    extra: dict[str, jax.Array] | None = None
    #: Global fold-in indices, in row order. None means legacy range(n_reps).
    rep_indices: jax.Array | None = None

    @property
    def effective_rep_indices(self) -> jax.Array:
        """Global draw indices, including the legacy zero-based default."""
        if self.rep_indices is None:
            return jnp.arange(self.n_reps, dtype=jnp.uint32)
        return self.rep_indices

    def _validated_rep_indices(self) -> np.ndarray:
        """Validate the host-side alignment witness before assembly/pairing."""
        if (
            isinstance(self.n_reps, bool)
            or not isinstance(self.n_reps, int | np.integer)
            or self.n_reps < 1
        ):
            raise ValueError("n_reps must be >= 1.")
        indices = np.asarray(self.effective_rep_indices)
        if indices.shape != (self.n_reps,) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("rep_indices must be an integer vector of length n_reps.")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.uint32).max):
            raise ValueError("rep_indices must lie in the uint32 fold_in domain.")
        if np.unique(indices).size != self.n_reps:
            raise ValueError("rep_indices contain duplicate global indices.")
        return indices

    @classmethod
    def concat(
        cls, shards: Sequence[MCRecords], *, reps: range | None = None
    ) -> MCRecords:
        """Assemble compatible shards in global draw order (#222; ledger 9.3).

        ``reps`` specifies exact expected coverage, including terminal shards.
        Without it, only the inferred contiguous interval is checked: a missing
        leading or terminal shard cannot be discovered from the shards alone.
        Multiple shards must carry the same non-None ``coupling_id``; job-array
        tasks should pass one explicit study token to ``replicate_coupled``.
        """
        if not shards:
            raise ValueError("MCRecords.concat(): no shards supplied.")
        expected = None if reps is None else _rep_range(None, reps)
        first = shards[0]
        first_leaves = jax.tree_util.tree_leaves(first.records)
        extra_names = tuple(first.extra or {})
        indices = []
        for shard in shards:
            indices.append(shard._validated_rep_indices())
            if not _keys_equal(first.key, shard.key):
                raise ValueError("MCRecords.concat(): master PRNG keys differ.")
            if shard.coupling_id != first.coupling_id or (
                len(shards) > 1 and shard.coupling_id is None
            ):
                raise ValueError(
                    "MCRecords.concat(): shared non-None coupling_id required."
                )
            if shard.param_names != first.param_names:
                raise ValueError("MCRecords.concat(): param_names differ.")
            if shard.records.J_dof != first.records.J_dof:
                raise ValueError("MCRecords.concat(): J_dof differs.")
            leaves = jax.tree_util.tree_leaves(shard.records)
            if len(leaves) != len(first_leaves):
                raise ValueError("MCRecords.concat(): record structures differ.")
            for leaf, reference in zip(leaves, first_leaves, strict=True):
                arr, ref = np.asarray(leaf), np.asarray(reference)
                if arr.shape[:1] != (shard.n_reps,) or arr.shape[1:] != ref.shape[1:]:
                    raise ValueError(
                        "MCRecords.concat(): record leaf shapes do not match."
                    )
                if arr.dtype != ref.dtype:
                    raise ValueError("MCRecords.concat(): record leaf dtypes differ.")
            for field in dataclasses.fields(shard.records):
                if field.name in ("J_dof", "param_names"):
                    continue
                shape: tuple[int, ...] = (shard.n_reps,)
                if field.name in ("theta_flat", "se"):
                    shape += (len(shard.param_names),)
                if np.shape(getattr(shard.records, field.name)) != shape:
                    raise ValueError("MCRecords.concat(): invalid record leaf shape.")
            if set(shard.extra or {}) != set(extra_names):
                raise ValueError("MCRecords.concat(): extra statistic names differ.")
            for name, col in (shard.extra or {}).items():
                ref = np.asarray((first.extra or {})[name])
                arr = np.asarray(col)
                if arr.shape[:1] != (shard.n_reps,) or arr.shape[1:] != ref.shape[1:]:
                    raise ValueError(
                        "MCRecords.concat(): extra statistic shapes differ."
                    )
                if arr.dtype != ref.dtype:
                    raise ValueError(
                        "MCRecords.concat(): extra statistic dtypes differ."
                    )
        combined = np.concatenate(indices)
        order = np.argsort(combined)
        ordered = combined[order]
        if np.unique(ordered).size != ordered.size:
            raise ValueError("MCRecords.concat(): overlapping rep indices.")
        if np.any(np.diff(ordered.astype(np.int64)) != 1):
            raise ValueError("MCRecords.concat(): gaps in rep coverage.")
        if expected is not None and not np.array_equal(ordered, np.asarray(expected)):
            raise ValueError(
                "MCRecords.concat(): rep coverage differs from expected reps."
            )
        records = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0)[order],
            *(shard.records for shard in shards),
        )
        extra = (
            {
                name: jnp.concatenate([shard.extra[name] for shard in shards])[order]  # type: ignore[index]
                for name in extra_names
            }
            if extra_names
            else None
        )
        return cls(
            records=records,
            key=first.key,
            n_reps=int(ordered.size),
            coupling_id=first.coupling_id,
            extra=extra,
            rep_indices=jnp.asarray(ordered, dtype=jnp.uint32),
        )

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
        ``sigma_meat_indefinite`` (the #138 NaN-SE event; #143),
        ``tau_saturated`` (the #205 ridge-saturation event), and one
        column per scalar custom statistic in :attr:`extra` (#179).
        Vector/tensor extras expand to indexed columns such as ``gbar[0]``
        or ``matrix[0,1]``, in row-major order. Duplicate output column names
        raise ``ValueError`` instead of overwriting a column.
        Pandas stays outside the compiled boundary --- this is the only
        pandas touchpoint in the studies module.
        """
        rec = self.records
        data: dict[str, np.ndarray] = {}

        def add_column(name: str, values: np.ndarray) -> None:
            if name in data:
                raise ValueError(
                    "MCRecords.to_pandas(): duplicate output column "
                    f"{name!r}; choose distinct parameter/statistic names."
                )
            data[name] = values

        theta = np.asarray(rec.theta_flat)
        se = np.asarray(rec.se)
        for d, name in enumerate(self.param_names):
            add_column(f"theta_{name}", theta[:, d])
            add_column(f"se_{name}", se[:, d])
        add_column("J_stat", np.asarray(rec.J_stat))
        add_column("J_pvalue", np.asarray(rec.J_pvalue))
        add_column("J_pvalue_adjusted", np.asarray(rec.J_pvalue_adjusted))
        add_column("converged", np.asarray(rec.converged))
        add_column("tau_realised", np.asarray(rec.tau_realised))
        add_column("binding_ridge", np.asarray(rec.binding_ridge))
        add_column("sigma_meat_indefinite", np.asarray(rec.sigma_meat_indefinite))
        add_column("tau_saturated", np.asarray(rec.tau_saturated))
        # Only materialization changes: storage/persistence keep tensor axes
        # intact (ledger 8.9).
        for name, col in (self.extra or {}).items():
            array = np.asarray(col)
            if array.ndim < 1 or array.shape[0] != self.n_reps:
                raise ValueError(
                    f"MCRecords.to_pandas(): statistic {name!r} has shape "
                    f"{array.shape}; expected leading rep axis {self.n_reps}."
                )
            for index in np.ndindex(array.shape[1:]):
                column = f"{name}[{','.join(str(i) for i in index)}]" if index else name
                add_column(column, array[(slice(None),) + index])
        df = pd.DataFrame(data, index=np.asarray(self.effective_rep_indices))
        df.index.name = "rep"
        return df


def replicate(
    run: Callable[[Any, Any], Any],
    dgp: Callable[[jax.Array], Any],
    *,
    n_reps: int | None = None,
    reps: range | None = None,
    key: jax.Array,
    theta_init: Any,
    anchor_per_rep: bool = False,
    coupling_id: Any = None,
    record: Callable[[Any, Any], FitRecord] | None = None,
    statistics: Mapping[str, Callable[[Any, Any], Any]] | None = None,
) -> MCRecords:
    """Run ``n_reps`` independent draw-and-estimate replicates.

    Parameters
    ----------
    run
        A fitted-estimator callable ``run(theta_init, dataset) -> result``.
        The default recorder expects an :class:`OptimizationResult`, as
        returned by :func:`emu_gmm.build_estimator`; a multi-call procedure
        can return another type with a custom ``record`` extractor.
        Taking the callable (rather
        than rebuilding internally from ``(model, covariance, ...)``)
        keeps the driver from duplicating ``build_estimator``'s kwarg
        surface, preserves measure x covariance orthogonality at the
        call site, and makes CRN arm comparisons natural: build one
        ``run`` per arm, call :func:`replicate` with the **same**
        ``key`` and ``dgp``, and the arms see identical draws.
    dgp
        The data-generating hook ``dgp(rep_key) -> dataset``; the dataset
        can be any type accepted by ``run`` and the extractors. The driver
        passes the same object to all three and calls no Measure methods.
        For a :func:`~emu_gmm.build_estimator` callable, return a full
        :class:`~emu_gmm.types.Measure` so masked / weighted / clustered
        designs remain first-class, including unequal per-coordinate
        ``N_j`` (commitment 9). Same-structure measures ride the zero-retrace
        kernel path (#124); a new shape retraces once and is then cached.
    n_reps, reps
        Exactly one: a positive count (global indices ``range(n_reps)``), or
        a nonempty increasing unit-step range of uint32-safe global indices.
        Shards reproduce the corresponding part of the unsharded stream.
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
        :class:`ValueError`. This mode requires Measure datasets; a custom
        procedure owns its own per-dataset anchoring policy.
        Cost: the slow path --- every replicate
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
    record
        Optional ``record(result, dataset) -> FitRecord`` extractor, called
        once per fit. ``None`` uses :func:`fit_record` on the result, retaining
        the existing optimization/inference split. A custom extractor must
        report the procedure's own estimates, SEs, J quantities, and named
        events in a :class:`~emu_gmm.types.FitRecord`; it does not change the
        fold-in schedule, stacking, or exclusion accounting. All records
        must have the same shapes and static metadata. To record a functional
        with its joint-covariance SE, use
        ``record=lambda result, dataset: fit_record(result, f=f, names=names)``.
    statistics
        Optional ``{name: extractor}`` map of **custom per-draw
        statistics** the harness should collect alongside the
        :class:`~emu_gmm.types.FitRecord` (#179). Each
        ``extractor(result, dataset) -> array`` is called once per
        replicate with that rep's result and the
        dataset the harness drew (so there is no dataset
        re-generation and no hand loop), and its outputs are stacked into
        :attr:`MCRecords.extra` under ``name`` with the same leading
        ``n_reps`` axis. This is the general way to record a statistic the
        ``FitRecord`` schema does not carry --- e.g. an identification-robust
        ``k_statistic(..., interest=[...]).p_K`` at a null, or an
        :func:`~emu_gmm.inference.identification.identification_strength`
        scalar --- computed by package code rather than reinvented in a
        caller's loop. Scalars, vectors, and tensors are accepted, with a
        fixed shape per statistic across replicates. Values are converted
        with ``jnp.asarray``; NaNs propagate unchanged (the #140 convention).
        ``None`` (default)
        leaves :attr:`MCRecords.extra` ``None``.

    Returns
    -------
    MCRecords
        Stacked :class:`~emu_gmm.types.FitRecord` plus study metadata.
        Non-converged replicates are recorded, not dropped; their
        ``converged`` field is 0. When ``statistics`` is given, the
        collected per-draw values are on :attr:`MCRecords.extra`.
    """
    selected = _rep_range(n_reps, reps)
    fit_per_rep: Callable[[Measure], OptimizationResult] | None = None
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

        def _fit_fresh_anchor(measure: Measure) -> OptimizationResult:
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
                jacobian_memory=spec.get("jacobian_memory", "default"),
            )

        fit_per_rep = _fit_fresh_anchor

    records: list[FitRecord] = []
    extra_cols: dict[str, list[Any]] = {name: [] for name in (statistics or {})}
    for local_r, r in enumerate(selected):
        rep_key = jax.random.fold_in(key, r)
        dataset = dgp(rep_key)
        if fit_per_rep is not None:
            result = fit_per_rep(dataset)
        else:
            result = run(theta_init, dataset)
        row = fit_record(result) if record is None else record(result, dataset)
        if not isinstance(row, FitRecord):
            raise TypeError(
                "replicate(): record extractor must return a FitRecord; "
                f"got {type(row).__name__} at replicate {r}."
            )
        records.append(row)
        _collect_statistics(statistics, extra_cols, result, dataset)
        if fit_per_rep is not None:
            # Bare estimate() builds fresh closures per call, so JAX's
            # global caches accumulate write-only traces (~14 MB/call
            # measured; the unmitigated leak OOM-killed a 300-rep study
            # at 9.4 GB -- the #139 merge-verification thread). But
            # clearing ALSO drops XLA's re-hit kernel compilations:
            # per-rep clearing measured ~3-5x wall-clock (#139 thread,
            # the #130 re-run). Clear every 25 reps: ~350 MB swing,
            # recompilation amortized to a few percent.
            if (local_r + 1) % 25 == 0:
                jax.clear_caches()
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)
    extra = (
        {name: jnp.stack(vals) for name, vals in extra_cols.items()}
        if statistics
        else None
    )
    return MCRecords(
        records=stacked,
        key=jnp.asarray(key),
        n_reps=len(selected),
        coupling_id=coupling_id,
        extra=extra,
        rep_indices=jnp.asarray(selected, dtype=jnp.uint32),
    )


def _collect_statistics(
    statistics: Mapping[str, Callable[[Any, Any], Any]] | None,
    extra_cols: dict[str, list[Any]],
    result: Any,
    dataset: Any,
) -> None:
    """Collect fixed-shape per-draw arrays, preserving NaNs (ledger 8.9)."""
    if not statistics:
        return
    for name, fn in statistics.items():
        value = jnp.asarray(fn(result, dataset))
        values = extra_cols[name]
        if values and value.shape != values[0].shape:
            raise ValueError(
                f"replicate(): statistic {name!r} changed shape at replicate "
                f"{len(values)}: expected {values[0].shape}, got {value.shape}; "
                "per-draw outputs must have a fixed shape."
            )
        values.append(value)


def replicate_coupled(
    runs: Mapping[str, Callable[[Any, Any], Any]],
    dgp: Callable[[jax.Array], Any],
    *,
    n_reps: int | None = None,
    reps: range | None = None,
    key: jax.Array,
    theta_init: Any,
    anchor_per_rep: bool = False,
    coupling_id: Any = None,
    record: Callable[[Any, Any], FitRecord] | None = None,
    statistics: Mapping[str, Callable[[Any, Any], Any]] | None = None,
) -> dict[str, MCRecords]:
    """Replicate several estimator arms over the SAME draws, as a CRN set.

    Each arm in ``runs`` (name -> an estimator callable) is
    replicated over the *identical* ``(dgp, key, n_reps)`` stream --- so rep
    ``r`` of every arm sees ``dgp(fold_in(key, r))`` (the documented CRN
    contract) --- and every returned :class:`MCRecords` is stamped with **one
    shared** ``coupling_id``. Because this factory *enforces* the shared
    ``(dgp, key)``, that token is a sound witness of coupling:
    :func:`emu_gmm.studies.crn_pair` will verify any two of the returned arms
    with no explicit token and no ``assert_coupled``.

    This is the ergonomic, sound alternative to a per-arm hand-rolled token ---
    and to auto-deriving one from the ``dgp`` callable, which is **unsound**: a
    fresh lambda per arm defeats object identity (false-refuse), while name /
    bytecode collide across *different* DGPs that share a lambda body
    (false-accept). A function's semantic identity is not computable; the
    coupling witness must come from *construction*, which is what this factory
    provides (issue #171; prior-art ledger ``.coder/ledger.org`` Section 6).

    Parameters
    ----------
    runs
        Mapping of arm name to a fitted-estimator callable. Arms typically
        differ only in covariance / weighting strategy --- the same data, scored
        differently --- which is exactly the CRN-paired-contrast use case.
    dgp, n_reps, reps, key, theta_init, anchor_per_rep, record, statistics
        Passed through to :func:`replicate`, identical for every arm.
    coupling_id
        The shared token stamped on every returned arm. Defaults to a fresh,
        unique value **per call**, so two *separate* coupled sets never compare
        equal (no cross-set false-accept even at the same ``key``). Pass an
        explicit value for a stable / meaningful token.

    Returns
    -------
    dict[str, MCRecords]
        One :class:`MCRecords` per arm (insertion order preserved), all sharing
        ``coupling_id``.
    """
    if not runs:
        raise ValueError("replicate_coupled(): `runs` is empty; nothing to couple.")
    selected = _rep_range(n_reps, reps)
    token = f"coupled-{uuid.uuid4().hex}" if coupling_id is None else coupling_id
    return {
        name: replicate(
            run,
            dgp,
            reps=selected,
            key=key,
            theta_init=theta_init,
            anchor_per_rep=anchor_per_rep,
            coupling_id=token,
            record=record,
            statistics=statistics,
        )
        for name, run in runs.items()
    }


__all__ = ["MCRecords", "replicate", "replicate_coupled"]
