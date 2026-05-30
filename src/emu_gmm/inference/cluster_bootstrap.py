"""Refit-based cluster bootstrap for GMM with clustered dependence.

This module provides :func:`cluster_bootstrap`, a refit-based
cluster bootstrap that resamples *whole clusters* with replacement
(the unit of independence under
:class:`~emu_gmm.covariance.clustered.ClusteredCovariance`),
constructs a resampled :class:`~emu_gmm.measures.empirical.EmpiricalMeasure`
on each draw, and re-runs :func:`~emu_gmm.estimator.estimate` from the
user-supplied ``theta_init``. The returned object collects the
per-replicate estimates ``theta_boot``, the per-replicate
:math:`J` statistic ``J_boot``, a per-replicate convergence flag, and
the consumed PRNG key.

Refit-based vs refit-free
=========================

This routine is the *refit-based* companion to the moment-wild
bootstrap that does **not** re-solve the GMM problem at each replicate
(issue #6, ``MomentWildBootstrap``). The two answer subtly different
inference questions and are not interchangeable:

* **Refit-based** (this routine). Resamples observations, re-solves
  the GMM problem, records :math:`\\hat\\theta^{(b)}`. The induced
  distribution of :math:`\\hat\\theta^{(b)} - \\hat\\theta` is a
  finite-sample approximation to the sampling distribution of
  :math:`\\hat\\theta - \\theta_0`, *including* the non-linearity of
  the GMM map. Suitable when the asymptotic-normal approximation
  underlying ``Sigma_theta`` from :func:`~emu_gmm.estimator.estimate`
  is suspect (small ``n_clusters``, weak identification, near-flat
  Jacobian).
* **Refit-free** (``MomentWildBootstrap``, issue #6). Perturbs the
  empirical moments via a wild weighting and uses the *first-order
  delta-method* link :math:`\\theta \\approx \\hat\\theta + A \\Delta m`
  to translate a moment-level bootstrap distribution into a
  parameter-level one *without* re-solving. Cheaper by a factor of
  one optimisation per replicate; but inherits the linearisation, so
  it cannot capture the non-linearity that refit-based bootstrap
  picks up.

Both are cluster-aware when the user passes a
:class:`ClusteredCovariance`; the present routine resamples *clusters*
(not individual observations) because the cluster is the unit of
independence in that model.

Algorithm
=========

Given an empirical measure with ``N`` observations partitioned into
``n_clusters`` groups via ``covariance.cluster_ids``, for replicate
:math:`b = 1, \\ldots, B`:

1. Draw ``n_clusters`` cluster indices with replacement from
   :math:`\\{0, \\ldots, n_\\text{clusters} - 1\\}` (uniform).
2. Assemble the resampled measure by concatenating, for each drawn
   cluster, the rows of ``measure.x`` / ``measure.mask`` /
   ``measure.weights`` whose ``cluster_ids`` equal that cluster's
   index. The new ``cluster_ids`` array renumbers the drawn clusters
   ``0, 1, ..., n_clusters - 1`` so the bootstrap covariance remains
   well-defined (a cluster drawn twice contributes two distinct
   clusters in the bootstrap world).
3. Build a fresh ``ClusteredCovariance`` matched to the resampled
   layout and call :func:`~emu_gmm.estimator.estimate` from
   ``theta_init`` with the user's ``weighting``, ``regularization``,
   and ``optimizer``.
4. Record :math:`\\hat\\theta^{(b)}`, :math:`J^{(b)}`, and a Python
   boolean convergence flag.

Implementation note
===================

To keep the bootstrap loop straightforward and to share the
labelling / diagnostics machinery, we call :func:`emu_gmm.estimate`
once per replicate from the host. The hot work --- the LM solve and
the gradient evaluations --- still runs through JAX inside
:func:`estimate`. The bootstrap *can* be vmapped in principle, but
the cluster-resampling layout has a data-dependent number of rows
per replicate (since clusters can vary in size and a cluster may be
drawn more than once), which is awkward to express under ``vmap``.
The per-replicate host loop trades a small Python overhead for
implementation clarity and is consistent with how ``estimate`` is
already exposed.

This routine is also a v1-compatible primitive: it requires only the
existing :func:`estimate` entry point and the array surface of
:class:`EmpiricalMeasure` / :class:`ClusteredCovariance`.

References
==========

The cluster-bootstrap construction here corresponds to the "pairs"
or "non-parametric" cluster bootstrap of Cameron, Gelbach, and
Miller (2008): resample *whole clusters* with replacement, re-solve
the estimator on each replicate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal import params as params_mod
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.estimator import estimate
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    Emu_GMM_DimensionError,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import ContinuouslyUpdated

# A bare ``Optimizer``-protocol callable is hard to name precisely here
# because the framework's protocol is annotated for a single solve. The
# bootstrap simply re-invokes whatever the user supplied (defaulting to
# the standard :func:`optimistix_lm`) once per replicate.
_OptimizerLike = Callable


@jdc.pytree_dataclass
class ClusterBootstrapResult:
    """Output of :func:`cluster_bootstrap`.

    A :func:`jax_dataclasses.pytree_dataclass` so the record flows
    through ``jit`` and ``vmap`` boundaries unchanged --- matching its
    inference-result siblings :class:`JTestResult`,
    :class:`KStatisticResult`, and :class:`WildBootstrapResult`. The
    array fields are traced leaves; ``param_names`` is a
    :func:`jax_dataclasses.static_field` (hashable, used only for
    re-tracing on configuration change).

    Parameters
    ----------
    theta_boot : :class:`haliax.NamedArray`
        Per-replicate parameter estimates, with axes
        ``(bootstrap, parameters)`` and shape ``(n_boot, K)``. The
        ``parameters`` axis carries the names from the user's
        parameter dataclass via :attr:`param_names` and
        :attr:`coords` (haliax's :class:`Axis` itself only stores a
        single name + size, not per-coordinate labels, so the
        coordinate strings live on this result object).
    J_boot : :class:`jax.Array`
        Per-replicate J statistic, shape ``(n_boot,)``. ``NaN`` for
        replicates whose solver diverged.
    convergence : :class:`jax.Array` of bool
        Per-replicate boolean convergence flag, shape ``(n_boot,)``.
        Carried as a JAX bool array (rather than NumPy) so the
        whole result is a single pytree --- the underlying optimiser
        status is resolved on the host but lifted into a JAX array
        when packaged into the result, so that ``jax.vmap`` over
        seed batches stacks the convergence flags along with the
        rest of the fields.
    key : :class:`jax.Array`
        The PRNG key as consumed by the bootstrap. Returning the
        already-consumed key prevents accidental reuse upstream;
        callers who want to continue the random stream should derive
        a fresh key from their original seed.
    param_names : tuple[str, ...] (static)
        Parameter names matching ``theta_boot``'s ``parameters`` axis,
        in PyTree-flatten order. Lifted from the user's parameter
        dataclass via :func:`emu_gmm._internal.params.param_names`
        and carried through so downstream tabular gestures
        (``pd.Series(boot_se, index=result.param_names)``) work
        without the caller re-reading the dataclass. Marked as a
        :func:`jax_dataclasses.static_field` because the names ride
        on the pytree treedef rather than as a traced leaf (strings
        are not JAX values; they trigger re-tracing on configuration
        change rather than re-tracing per-replicate).
    """

    theta_boot: ha.NamedArray
    J_boot: jnp.ndarray
    convergence: jnp.ndarray
    key: jax.Array
    param_names: tuple[str, ...] = jdc.static_field()  # type: ignore[attr-defined]

    @property
    def coords(self) -> Mapping[str, tuple[str | int, ...]]:
        """Per-axis coordinate labels for :attr:`theta_boot`.

        Returns a read-only mapping with two keys:

        * ``"parameters"`` --- the parameter names from
          :attr:`param_names`, aligned with the ``parameters`` axis
          of ``theta_boot``.
        * ``"bootstrap"`` --- positional replicate indices
          ``(0, 1, ..., n_boot - 1)``.

        haliax's :class:`Axis` only carries a name + size, not
        per-coordinate strings, so this mapping is the framework's
        analogue of :attr:`xarray.DataArray.coords` for the cluster
        bootstrap output. Construction is cheap (no array copies).
        """
        n_boot = int(self.theta_boot.array.shape[0])
        return MappingProxyType(
            {
                axes_mod.PARAMS_NAME: tuple(self.param_names),
                "bootstrap": tuple(range(n_boot)),
            }
        )


def _cluster_row_indices(cluster_ids: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Return the row indices belonging to each cluster.

    For cluster ``c`` in ``[0, n_clusters)`` returns
    ``np.where(cluster_ids == c)[0]``. Used to build a quick lookup
    for the resampling step.
    """
    rows_by_cluster: list[np.ndarray] = []
    for c in range(n_clusters):
        rows_by_cluster.append(np.where(cluster_ids == c)[0])
    return rows_by_cluster


def _resample_one(
    measure: EmpiricalMeasure,
    rows_by_cluster: list[np.ndarray],
    drawn: np.ndarray,
) -> tuple[EmpiricalMeasure, ClusteredCovariance]:
    """Build the resampled measure + covariance for one bootstrap draw.

    Parameters
    ----------
    measure : :class:`EmpiricalMeasure`
        The original sample.
    rows_by_cluster : list of np.ndarray
        Output of :func:`_cluster_row_indices` for the original
        cluster layout.
    drawn : np.ndarray of int
        The cluster indices selected for this bootstrap replicate;
        length equals the number of clusters in the resampled world
        (``n_clusters`` for the standard with-replacement draw).

    Returns
    -------
    boot_measure : :class:`EmpiricalMeasure`
    boot_cov : :class:`ClusteredCovariance`
        The resampled covariance, with cluster IDs renumbered so each
        drawn cluster maps to a distinct bootstrap-cluster index. The
        number of bootstrap clusters equals ``len(drawn)``.
    """
    # Use NumPy for the assembly: the data-dependent shape (varying
    # cluster sizes, repeated draws) means jax.numpy concatenation
    # would have to materialise the rows on host anyway.
    x_np = np.asarray(measure.x)
    mask_np = np.asarray(measure.mask)
    weights_np = np.asarray(measure.weights)

    boot_x_rows: list[np.ndarray] = []
    boot_mask_rows: list[np.ndarray] = []
    boot_weights_rows: list[np.ndarray] = []
    boot_cluster_ids: list[np.ndarray] = []
    for new_cluster_idx, original_cluster_idx in enumerate(drawn):
        rows = rows_by_cluster[int(original_cluster_idx)]
        if rows.size == 0:
            continue
        boot_x_rows.append(x_np[rows])
        boot_mask_rows.append(mask_np[rows])
        boot_weights_rows.append(weights_np[rows])
        boot_cluster_ids.append(
            np.full(rows.shape[0], new_cluster_idx, dtype=np.float64)
        )

    if not boot_x_rows:
        # Pathological: every drawn cluster was empty (cannot happen
        # for a non-empty input but guard against it anyway). Fall
        # back to the original sample so estimate() doesn't choke on
        # a zero-row measure.
        x_b = jnp.asarray(x_np)
        mask_b = jnp.asarray(mask_np)
        weights_b = jnp.asarray(weights_np)
        cluster_ids_b = jnp.zeros(x_b.shape[0])
        n_clusters_b = 1
    else:
        x_b = jnp.asarray(np.concatenate(boot_x_rows, axis=0))
        mask_b = jnp.asarray(np.concatenate(boot_mask_rows, axis=0))
        weights_b = jnp.asarray(np.concatenate(boot_weights_rows, axis=0))
        cluster_ids_b = jnp.asarray(np.concatenate(boot_cluster_ids, axis=0))
        n_clusters_b = int(len(boot_x_rows))

    boot_measure = EmpiricalMeasure(x=x_b, mask=mask_b, weights=weights_b)
    boot_cov = ClusteredCovariance(
        cluster_ids=cluster_ids_b,
        n_clusters=n_clusters_b,
    )
    return boot_measure, boot_cov


def cluster_bootstrap(
    model: StructuralModel,
    theta_init: ParamsLike,
    measure: EmpiricalMeasure,
    covariance: ClusteredCovariance,
    *,
    n_boot: int,
    key: jax.Array,
    weighting: WeightingStrategy | None = None,
    regularization: RegularizationStrategy | None = None,
    optimizer: _OptimizerLike | None = None,
) -> ClusterBootstrapResult:
    """Refit-based cluster bootstrap for a GMM estimator.

    Parameters
    ----------
    model
        The structural model passed to :func:`emu_gmm.estimate`.
    theta_init
        Starting parameters; the same value is used to seed every
        bootstrap solve.
    measure : :class:`EmpiricalMeasure`
        The original sample. Each replicate constructs a resampled
        ``EmpiricalMeasure`` from this object.
    covariance : :class:`ClusteredCovariance`
        Cluster-level covariance for the original sample. Its
        ``cluster_ids`` array defines the unit of independence; the
        bootstrap resamples clusters with replacement at that level.
    n_boot : int, keyword-only
        Number of bootstrap replicates.
    key : :class:`jax.Array`, keyword-only
        JAX PRNG key. Consumed and returned on
        :class:`ClusterBootstrapResult` so the caller can verify the
        random stream was used as intended.
    weighting, regularization, optimizer
        Forwarded to :func:`emu_gmm.estimate` on each replicate.
        Defaults match the framework defaults:
        :class:`~emu_gmm.weighting.ContinuouslyUpdated`,
        :class:`~emu_gmm.regularization.DiagonalTikhonov`,
        :func:`~emu_gmm.optimizer.optimistix_lm`.

    Returns
    -------
    :class:`ClusterBootstrapResult`

    Notes
    -----
    See the module docstring for the distinction from the refit-free
    moment-wild bootstrap (issue #6).
    """
    if n_boot <= 0:
        raise ValueError(f"cluster_bootstrap: n_boot must be positive, got {n_boot}")

    # Resolve defaults to the same values used by :func:`estimate`.
    if weighting is None:
        weighting = ContinuouslyUpdated()
    if regularization is None:
        regularization = DiagonalTikhonov()
    if optimizer is None:
        optimizer = optimistix_lm()

    # Pre-compute the cluster -> row-indices lookup once. Pulled to
    # NumPy here so the per-replicate assembly stays cheap.
    cluster_ids_np = np.asarray(covariance.cluster_ids).astype(np.int64)
    n_clusters = int(covariance.n_clusters)
    rows_by_cluster = _cluster_row_indices(cluster_ids_np, n_clusters)

    # Probe K (number of parameters) so we can allocate theta_boot.
    theta_init_flat, _treedef = params_mod.flatten_params(theta_init)
    K = int(theta_init_flat.shape[0])
    param_names = tuple(params_mod.param_names(theta_init))

    # Draw the cluster indices for all replicates from a single key
    # via split: B splits give B independent (n_clusters,) draws.
    keys = jax.random.split(key, n_boot)
    drawn_all = np.empty((n_boot, n_clusters), dtype=np.int64)
    for b in range(n_boot):
        drawn_all[b] = np.asarray(
            jax.random.randint(
                keys[b], shape=(n_clusters,), minval=0, maxval=n_clusters
            )
        )

    theta_boot = np.empty((n_boot, K), dtype=np.float64)
    J_boot = np.empty(n_boot, dtype=np.float64)
    convergence = np.empty(n_boot, dtype=bool)

    for b in range(n_boot):
        boot_measure, boot_cov = _resample_one(measure, rows_by_cluster, drawn_all[b])
        # NB: do *not* wrap this call in a broad ``except Exception``.
        # The framework's optimisers (``optimistix_lm`` with
        # ``throw=False``, ``scipy_lm``) report non-convergence via
        # ``result.converged`` / ``info.status`` rather than raising.
        # Only the *known*, *intentional* divergence pathways below
        # are caught and surfaced as a convergence flag; everything
        # else (TypeErrors from a buggy ``psi``, ValueErrors from
        # measure construction, etc.) propagates so the user can see
        # the real bug instead of a silent NaN row.
        try:
            result = estimate(
                model=model,
                measure=boot_measure,
                covariance=boot_cov,
                weighting=weighting,
                regularization=regularization,
                optimizer=optimizer,
                theta_init=theta_init,
            )
        except (
            np.linalg.LinAlgError,
            FloatingPointError,
            Emu_GMM_DimensionError,
        ):
            # Documented bootstrap-divergence pathways:
            #   * LinAlgError: a Cholesky / inv inside the regulariser
            #     hit a non-PD matrix that the adaptive ridge could
            #     not rescue (e.g. a draw of identical clusters).
            #   * FloatingPointError: only raised when the host has
            #     opted into strict NaN checks via
            #     ``jax.config.update('jax_debug_nans', True)``; in
            #     that mode an LM iterate that produces NaNs is a
            #     legitimate "this bootstrap world is degenerate"
            #     signal, not a programming error.
            #   * Emu_GMM_DimensionError: a draw of all-empty clusters
            #     could in principle collapse the working sample to
            #     zero rows (``_resample_one`` guards against this by
            #     falling back to the original sample, but if a
            #     future refactor surfaces a degenerate world the
            #     dimension check should still be treated as
            #     non-convergence rather than a hard failure).
            theta_boot[b] = np.nan
            J_boot[b] = np.nan
            convergence[b] = False
            continue
        theta_hat_flat, _ = params_mod.flatten_params(result.theta_hat)
        theta_boot[b] = np.asarray(theta_hat_flat)
        J_boot[b] = float(result.J_stat)
        convergence[b] = bool(result.converged)

    # Label the (n_boot, K) array along the canonical parameters axis;
    # the bootstrap-replicate axis carries no semantic labels (just a
    # replicate index) so we leave it as a plain axis named "bootstrap".
    # The parameter-name tuple is carried on the result via
    # ``param_names`` (and accessible as ``result.coords['parameters']``)
    # because haliax's :class:`Axis` only stores a single axis name +
    # size, not per-coordinate strings -- the framework's standing
    # convention for parameter-named labels (see
    # :class:`emu_gmm._internal.labels.LabelContext`).
    boot_axis = ha.Axis(name="bootstrap", size=n_boot)
    params_axis = axes_mod.params_axis(K)
    theta_boot_named = labels_mod.label_matrix(
        jnp.asarray(theta_boot), boot_axis, params_axis
    )

    return ClusterBootstrapResult(
        theta_boot=theta_boot_named,
        J_boot=jnp.asarray(J_boot),
        convergence=jnp.asarray(convergence),
        key=key,
        param_names=param_names,
    )


__all__ = ["cluster_bootstrap", "ClusterBootstrapResult"]
