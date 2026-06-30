"""Identification-robust confidence sets by K-statistic inversion (#41).

Inverts the Kleibergen (2005) :math:`K`-statistic (or the :math:`J` /
:math:`S` statistics, for Anderson--Rubin-style sets) over a caller-
supplied one-dimensional grid of nulls:

.. math::
    C_\\alpha \\;=\\; \\{\\, g \\in \\mathrm{grid} :
        p_{K}(\\theta_0(g)) > \\alpha \\,\\}.

Because :math:`K` is asymptotically pivotal *regardless of
identification strength* (Kleibergen 2005, Proposition 2; see
:mod:`emu_gmm.inference.k_statistic`), the inverted set has correct
asymptotic coverage even when the parameter is weakly identified — the
regime where a Wald interval ``theta_hat +/- z * SE`` is meaningless.
Weak identification shows up *honestly* in the set's geometry instead:
wide, half-unbounded, disconnected, or empty sets are findings, not
failures, and the result classifies them explicitly rather than
flattening everything into two interval endpoints.

Design notes (upstreamed from the Seasonality consumer's prototype,
``scripts/euler_robust_set.py``, per the issue-#41 thread):

- **The caller owns the domain.** The grid is a 1-D array of scalar
  values and ``theta_builder`` lifts each value to a full parameter
  PyTree. Parameter-space constraints are respected by construction —
  e.g. a strictly positive grid for a ``Positive``-manifold scalar (use
  geometric spacing to match the manifold's natural scale).
- **Topology is explicit.** Connected runs of accepted grid points are
  reported as interpolated ``[lo, hi]`` components plus a topology
  label; runs touching a grid edge are flagged open (the set may extend
  beyond the scanned window) rather than silently truncated.
- **Boundaries are interpolated.** Interior component edges are refined
  by linear interpolation of :math:`p - \\alpha` between the adjacent
  grid points, so the resolution is better than the raw grid spacing.
- **NaN is an event, not a value** (the #140 convention; see design.org): a grid point
  whose p-value is NaN (e.g. a Cholesky failure at an extreme null) is
  *excluded* from the set, counted in ``n_invalid``, and surfaced in
  ``invalid_indices`` and the summary — it never silently becomes
  either acceptance or rejection. If invalid points border a component,
  treat that boundary as unreliable.

The grid evaluation reuses one jit-compiled kernel across grid points
(``theta_builder`` must therefore return PyTrees of identical structure
for every grid value — the natural behaviour of any builder that just
injects the scalar into a fixed dataclass).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import numpy as np
from jaxtyping import Array, Float

from emu_gmm.inference.k_statistic import (
    KStatisticResult,
    _kappa_chi2_sf,
    k_statistic,
)
from emu_gmm.types import (
    CovarianceStrategy,
    Measure,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
)

_STATISTICS = ("K", "S", "J")


@dataclass(frozen=True)
class KConfidenceSet:
    """Result of :func:`k_confidence_set`: an inverted robust set.

    Attributes
    ----------
    grid : (G,) numpy array
        The scanned scalar nulls, strictly increasing.
    p_grid : (G,) numpy array
        The inverted statistic's p-value at each grid point.
    in_set : (G,) numpy bool array
        ``p_grid > alpha`` (NaN p-values are ``False`` — see
        ``invalid_indices``).
    intervals : list of (lo, hi) float pairs
        Connected components of the set, with interior edges linearly
        interpolated in :math:`p - \\alpha`. Component edges that
        coincide with a grid edge are NOT interpolated (the set may
        continue beyond the window; see ``open_left`` / ``open_right``).
    topology : str
        One of ``'empty'``, ``'interval'``, ``'disconnected'``,
        ``'unbounded-left'``, ``'unbounded-right'``, ``'full-grid'``.
        Edge-open *disconnected* sets keep the ``'disconnected'`` label;
        consult ``open_left`` / ``open_right`` for the edge flags.
    open_left, open_right : bool
        Whether the first / last grid point is in the set — i.e. the
        set may extend beyond the scanned window on that side.
    alpha : float
        The level inverted at; coverage of the set is ``1 - alpha``
        asymptotically (pointwise in the null).
    statistic : str
        Which p-value was inverted: ``'K'`` (default), ``'S'``, or
        ``'J'`` (Anderson--Rubin-style full-vector test).
    n_invalid : int
        Number of grid points with NaN p-values (excluded from the set
        and listed in ``invalid_indices``).
    invalid_indices : tuple of int
        Grid indices whose p-value was NaN.
    results : tuple of KStatisticResult, or None
        The full per-grid-point decompositions, retained only when
        ``keep_results=True`` was passed.
    """

    grid: np.ndarray
    p_grid: np.ndarray
    in_set: np.ndarray
    intervals: list[tuple[float, float]]
    topology: str
    open_left: bool
    open_right: bool
    alpha: float
    statistic: str
    n_invalid: int = 0
    invalid_indices: tuple[int, ...] = ()
    results: tuple[KStatisticResult, ...] | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Human-readable one-stop description of the set."""
        if self.topology == "empty":
            setstr = f"EMPTY (every grid point rejected at alpha={self.alpha:g})"
        else:
            parts = []
            for i, (lo, hi) in enumerate(self.intervals):
                lo_s = f"{lo:.4g}"
                hi_s = f"{hi:.4g}"
                if i == 0 and self.open_left:
                    lo_s = f"<{lo_s}"
                if i == len(self.intervals) - 1 and self.open_right:
                    hi_s = f"{hi_s}+"
                parts.append(f"[{lo_s}, {hi_s}]")
            setstr = " U ".join(parts)
        lines = [
            f"C_{self.alpha:g} ({self.statistic}-statistic inversion) = {setstr}",
            f"  topology = {self.topology}"
            + (" (open at left grid edge)" if self.open_left else "")
            + (" (open at right grid edge)" if self.open_right else ""),
        ]
        if self.n_invalid:
            lines.append(
                f"  WARNING: {self.n_invalid} grid point(s) had NaN "
                f"p-values (indices {list(self.invalid_indices)}); they "
                f"were excluded from the set — boundaries adjacent to "
                f"them are unreliable."
            )
        return "\n".join(lines)


def _validate_grid_args(
    grid: Sequence[float] | np.ndarray, statistic: str, alpha: float
) -> np.ndarray:
    """Validate ``grid`` / ``statistic`` / ``alpha`` (shared by both paths)."""
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.ndim != 1 or grid_arr.shape[0] < 2:
        raise ValueError(
            f"k_confidence_set: grid must be 1-D with at least 2 points; "
            f"got shape {grid_arr.shape}."
        )
    if not np.all(np.diff(grid_arr) > 0):
        raise ValueError("k_confidence_set: grid must be strictly increasing.")
    if statistic not in _STATISTICS:
        raise ValueError(
            f"k_confidence_set: statistic must be one of {_STATISTICS}; "
            f"got {statistic!r}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"k_confidence_set: alpha must be in (0, 1); got {alpha!r}.")
    return grid_arr


def _connected_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index runs ``(i0, i1)`` where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _interp_edge(g0: float, g1: float, p0: float, p1: float, alpha: float) -> float:
    """Grid value where ``p`` crosses ``alpha``, linear between two points.

    Falls back without interpolating when either p-value is NaN (an
    invalid neighbour gives nothing to interpolate against — return the
    in-set / higher-p side, treating NaN as ``-inf``) or the segment is
    flat (return ``g0``; the edge is then at grid resolution).
    """
    if not (np.isfinite(p0) and np.isfinite(p1)):
        p0f = p0 if np.isfinite(p0) else -np.inf
        p1f = p1 if np.isfinite(p1) else -np.inf
        return float(g1) if p1f > p0f else float(g0)
    if (p1 - p0) == 0.0:
        return float(g0)
    t = (alpha - p0) / (p1 - p0)
    t = min(max(t, 0.0), 1.0)
    return float(g0 + t * (g1 - g0))


def _classify(runs: list[tuple[int, int]], n_grid: int) -> tuple[str, bool, bool]:
    """Topology label + (open_left, open_right) edge flags."""
    if not runs:
        return "empty", False, False
    open_left = runs[0][0] == 0
    open_right = runs[-1][1] == n_grid - 1
    if len(runs) > 1:
        return "disconnected", open_left, open_right
    if open_left and open_right:
        return "full-grid", True, True
    if open_left:
        return "unbounded-left", True, False
    if open_right:
        return "unbounded-right", False, True
    return "interval", False, False


def k_confidence_set(
    theta_builder: Callable[[float], ParamsLike],
    grid: Sequence[float] | np.ndarray,
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    *,
    alpha: float = 0.05,
    statistic: str = "K",
    regularization: RegularizationStrategy | None = None,
    score_cov_fn: Callable[..., Float[Array, "p M M"]] | None = None,
    V: Float[Array, "M M"] | None = None,
    L: Float[Array, "M M"] | None = None,
    gauge_nullspace_dim: int | None = None,
    strong_id_fallback: bool = False,
    keep_results: bool = False,
    profile: Sequence[str] | None = None,
    nuisance_optimizer: Any = None,
    nuisance_weighting: Any = None,
) -> KConfidenceSet:
    """Invert the K-statistic over a 1-D grid of nulls (#41 PR (b)).

    Profiling hook (#176)
    ---------------------
    By default this evaluates the K-statistic along the **fixed curve**
    ``theta_builder(g)`` — the nuisance coordinates are whatever the builder
    set, *not* re-optimised. Pass ``profile=[<field names>]`` to instead get a
    *profiled* (concentrated) set: at each grid value the named fields are held
    fixed at ``theta_builder(g)`` and the remaining parameter leaves — a
    manifold nuisance such as a ``PSDFixedRank`` ``Gamma`` factor included — are
    **re-optimised** before the K/S/J p-value is evaluated. This delegates to
    :func:`profiled_k_confidence_set`; see it for the full contract (the inner
    re-optimisation reuses :func:`emu_gmm.estimate`, so the manifold gauge is
    handled by the existing quotient-aware path). ``nuisance_optimizer`` /
    ``nuisance_weighting`` configure the inner solve. ``profile`` must be
    ``None`` for the default fixed-curve behaviour.

    Evaluates :func:`emu_gmm.inference.k_statistic` at
    ``theta_builder(g)`` for every ``g`` in ``grid`` and returns the
    level-``alpha`` robust confidence set
    ``{ g : p(theta_0(g)) > alpha }`` with explicit topology
    (empty / interval / disconnected / open-at-an-edge) and
    interpolated component boundaries. See the module docstring for the
    design rationale and the weak-identification reading of each
    topology.

    Parameters
    ----------
    theta_builder : callable ``float -> ParamsLike``
        Lifts a scalar grid value to the full parameter PyTree to test.
        Must return the SAME PyTree structure for every value (one jit
        trace is reused across the grid). Constraints live here: for a
        ``sigma > 0`` parameter pass a positive grid and build the
        ``Positive``-annotated dataclass as usual.
    grid : 1-D array-like of float
        Strictly increasing scan values, length >= 2. Geometry hint:
        space the grid by the parameter's natural scale (geometric for
        scale parameters); the component edges are interpolated, so the
        grid bounds resolution but not validity.
    measure, covariance, model
        Exactly as for :func:`k_statistic`.
    alpha : float, default 0.05
        Test level; the set has asymptotic coverage ``1 - alpha``.
    statistic : {'K', 'S', 'J'}, default 'K'
        Which p-value to invert. ``'K'`` is the identification-robust
        score test (the default and the point of this routine); ``'J'``
        gives the Anderson--Rubin-style full-vector set (conservative
        but immune to the K-statistic's known power dips at criterion
        saddle points — comparing the two sets is a useful diagnostic);
        ``'S'`` inverts the overidentification residual alone.
    regularization, score_cov_fn, V, L, gauge_nullspace_dim, strong_id_fallback
        Forwarded to :func:`k_statistic` unchanged (see its docstring;
        the #41 dispatch rules — loud strong-ID fallback, refused design
        sandwiches — apply per grid point at trace time).
    keep_results : bool, default False
        Retain the full :class:`KStatisticResult` per grid point on the
        returned object (memory ~ grid length; off by default).

    Returns
    -------
    :class:`KConfidenceSet`

    Raises
    ------
    ValueError
        If ``grid`` is not 1-D / strictly increasing / length >= 2, or
        ``statistic`` is not one of ``'K'``, ``'S'``, ``'J'`` — plus
        anything :func:`k_statistic` itself raises (it is evaluated
        once eagerly at the first grid point before the compiled sweep,
        so its dispatch errors surface with their own messages).

    Notes
    -----
    With ``profile=None`` (the default) the K-statistic tests the FULL
    parameter vector at each null and this routine scans a one-dimensional
    family of nulls: ``theta_builder`` traces a curve through the parameter
    space (e.g. varying one coordinate with the others held at calibrated
    values), so the resulting set is a *slice* of the joint robust set along
    that curve, NOT a marginal subvector set for one coordinate.

    For a marginal subvector set, pass ``profile=[...]`` (see
    :func:`profiled_k_confidence_set`), which **concentrates** the nuisance
    (re-optimises it at each grid value) rather than holding it fixed — and
    note that *concentration / profiling* (plug in the restricted nuisance
    estimate) is a different procedure from *projection* (sup the statistic
    over the nuisance); they have different validity conditions. See
    :func:`profiled_k_confidence_set` for the subvector reference
    distribution and its strong-nuisance-identification precondition.
    """
    # #176 profiling hook: delegate to the re-optimising path when the caller
    # names fixed (interest) fields. ``profile=None`` keeps the fixed-curve
    # default bitwise unchanged.
    if profile is not None:
        # The delegation never requests return_profiled_points, so the result
        # is always a KConfidenceSet (not the tuple variant).
        return cast(
            KConfidenceSet,
            profiled_k_confidence_set(
                theta_builder,
                grid,
                measure,
                covariance,
                model,
                profile=profile,
                alpha=alpha,
                statistic=statistic,
                regularization=regularization,
                score_cov_fn=score_cov_fn,
                V=V,
                L=L,
                gauge_nullspace_dim=gauge_nullspace_dim,
                strong_id_fallback=strong_id_fallback,
                keep_results=keep_results,
                nuisance_optimizer=nuisance_optimizer,
                nuisance_weighting=nuisance_weighting,
            ),
        )
    if nuisance_optimizer is not None or nuisance_weighting is not None:
        raise ValueError(
            "k_confidence_set: nuisance_optimizer / nuisance_weighting only "
            "apply to the profiled path; pass profile=[...] to use them."
        )

    grid_arr = _validate_grid_args(grid, statistic, alpha)

    def _eval(theta_0: ParamsLike) -> KStatisticResult:
        return k_statistic(
            theta_0,
            measure,
            covariance,
            model,
            regularization=regularization,
            score_cov_fn=score_cov_fn,
            V=V,
            L=L,
            gauge_nullspace_dim=gauge_nullspace_dim,
            strong_id_fallback=strong_id_fallback,
        )

    # First grid point eagerly: k_statistic's static dispatch errors
    # (under-identification, refused covariance strategies, missing
    # robust inputs) surface here with their own messages rather than
    # wrapped in a jit trace.
    first = _eval(theta_builder(float(grid_arr[0])))

    # Compiled sweep for the rest: one trace, reused across the grid
    # (theta_builder returns an identical PyTree structure per value).
    eval_jit = jax.jit(_eval)
    results: list[KStatisticResult] = [first]
    for g in grid_arr[1:]:
        results.append(eval_jit(theta_builder(float(g))))

    return _assemble_set(grid_arr, results, statistic, alpha, keep_results)


def _assemble_set(
    grid_arr: np.ndarray,
    results: Sequence[KStatisticResult],
    statistic: str,
    alpha: float,
    keep_results: bool,
) -> KConfidenceSet:
    """Classify a grid of :class:`KStatisticResult` into a :class:`KConfidenceSet`.

    The shared tail of :func:`k_confidence_set` and
    :func:`profiled_k_confidence_set`: extract the chosen p-value per grid
    point, mark invalid (NaN-p) points as an event (#140), find the
    accepted runs, interpolate component edges, classify the topology. Both
    callers produce a ``results`` list of one :class:`KStatisticResult` per
    grid point — whether by evaluating the K-statistic along a fixed curve
    (``k_confidence_set``) or at a re-optimised profiled point
    (``profiled_k_confidence_set``) — so the classification is identical.
    """
    attr = f"p_{statistic}"
    p_grid = np.array(
        [float(np.asarray(getattr(r, attr))) for r in results], dtype=float
    )

    invalid = ~np.isfinite(p_grid)
    # NaN is an event, not a value (#140): an invalid p-value is never
    # silently 'in' or 'out' — it is excluded AND surfaced.
    in_set = np.where(invalid, False, p_grid > alpha)

    runs = _connected_runs(in_set)
    intervals: list[tuple[float, float]] = []
    n = grid_arr.shape[0]
    for i0, i1 in runs:
        if i0 == 0:
            lo = float(grid_arr[0])  # open edge: no outer point to interpolate
        else:
            lo = _interp_edge(
                grid_arr[i0 - 1], grid_arr[i0], p_grid[i0 - 1], p_grid[i0], alpha
            )
        if i1 == n - 1:
            hi = float(grid_arr[-1])
        else:
            hi = _interp_edge(
                grid_arr[i1], grid_arr[i1 + 1], p_grid[i1], p_grid[i1 + 1], alpha
            )
        intervals.append((lo, hi))

    topology, open_left, open_right = _classify(runs, n)

    return KConfidenceSet(
        grid=grid_arr,
        p_grid=p_grid,
        in_set=in_set,
        intervals=intervals,
        topology=topology,
        open_left=open_left,
        open_right=open_right,
        alpha=float(alpha),
        statistic=statistic,
        n_invalid=int(invalid.sum()),
        invalid_indices=tuple(int(i) for i in np.nonzero(invalid)[0]),
        results=tuple(results) if keep_results else None,
    )


class _ProfileReducer:
    """Freeze the ``profile`` fields of a flat parameter dataclass (#176).

    The named ``profile`` fields are the **interest** coordinates held fixed at
    each grid value; every other field is **nuisance**, re-optimised. The
    reducer materialises a slim ``@jdc.pytree_dataclass`` of just the nuisance
    fields (built once, reused across grid points) so the inner
    :func:`emu_gmm.estimate` optimises the *reduced* parameter while the model
    still receives the full ``theta``: :meth:`free_init` projects a full
    ``theta`` to the reduced start, and :meth:`recombine` lifts a reduced
    ``theta`` back, splicing the fixed fields in.

    Whole-field granularity (#176 scope): a fixed field freezes its entire
    leaf. A manifold field (e.g. a ``PSDFixedRank`` ``Gamma`` factor) is
    therefore either wholly fixed or wholly free — never split — so the inner
    ``estimate`` re-optimises it through the existing quotient-aware manifold
    path and the gauge is handled there, never hand-rolled. The framework's
    flat-dataclass convention (each field is one leaf — a scalar or a
    ``ManifoldLeaf``) makes field-level == leaf-level. Declare each scalar
    coordinate you may want to profile as its own field.
    """

    def __init__(self, template: ParamsLike, profile: Sequence[str]) -> None:
        import dataclasses

        import jax_dataclasses as jdc

        if not dataclasses.is_dataclass(template):
            raise ValueError(
                "profiled_k_confidence_set: profiling by field name requires a "
                "@jdc.pytree_dataclass parameter; theta_builder returned a "
                f"{type(template).__name__}."
            )
        self._cls: Any = type(template)
        all_names = [f.name for f in dataclasses.fields(template)]
        fixed = list(profile)
        if not fixed:
            raise ValueError("profiled_k_confidence_set: profile must name >= 1 field.")
        unknown = [nm for nm in fixed if nm not in all_names]
        if unknown:
            raise ValueError(
                f"profiled_k_confidence_set: profile names {unknown} are not "
                f"parameter fields; available fields are {all_names}."
            )
        self._fixed = fixed
        self._free = [nm for nm in all_names if nm not in set(fixed)]
        if not self._free:
            raise ValueError(
                "profiled_k_confidence_set: every parameter field is fixed by "
                "profile, leaving nothing to optimise. Leave at least one "
                "nuisance field out of profile."
            )
        # Slim nuisance dataclass, registered as a pytree once (reused across
        # the grid; field order matches the original so leaf-walk order is
        # preserved for the manifold spec / gauge handling). Build a plain
        # annotated class and let jdc.pytree_dataclass generate the frozen
        # dataclass + __init__ (make_dataclass' own __init__ uses setattr,
        # which the frozen pytree blocks).
        reduced_cls = type(
            "_ReducedProfileParams",
            (),
            {"__annotations__": {nm: Any for nm in self._free}},
        )
        self._reduced_cls: Any = jdc.pytree_dataclass(reduced_cls)

    def free_init(self, theta_full: ParamsLike) -> Any:
        """Project a full ``theta`` to the reduced (nuisance-only) start."""
        return self._reduced_cls(**{nm: getattr(theta_full, nm) for nm in self._free})

    def recombine(self, theta_full: ParamsLike, reduced: Any) -> ParamsLike:
        """Lift a reduced ``theta`` back, splicing in the fixed fields of ``theta_full``."""
        kwargs = {nm: getattr(theta_full, nm) for nm in self._fixed}
        kwargs.update({nm: getattr(reduced, nm) for nm in self._free})
        return self._cls(**kwargs)


def profiled_k_confidence_set(
    theta_builder: Callable[[float], ParamsLike],
    grid: Sequence[float] | np.ndarray,
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    *,
    profile: Sequence[str],
    alpha: float = 0.05,
    statistic: str = "K",
    nuisance_optimizer: Any = None,
    nuisance_weighting: Any = None,
    regularization: RegularizationStrategy | None = None,
    score_cov_fn: Callable[..., Float[Array, "p M M"]] | None = None,
    V: Float[Array, "M M"] | None = None,
    L: Float[Array, "M M"] | None = None,
    gauge_nullspace_dim: int | None = None,
    strong_id_fallback: bool = False,
    keep_results: bool = False,
    return_profiled_points: bool = False,
) -> KConfidenceSet | tuple[KConfidenceSet, tuple[ParamsLike, ...]]:
    r"""Profiled identification-robust set: concentrate out the nuisance (#176).

    The profiling sibling of :func:`k_confidence_set`. At each grid value the
    fields named in ``profile`` are held fixed at ``theta_builder(g)`` while the
    remaining parameter leaves — **including a manifold nuisance** such as a
    ``PSDFixedRank`` ``Gamma`` factor — are re-optimised by an inner
    :func:`emu_gmm.estimate`; the K/S/J p-value is then evaluated at the
    resulting profiled point. The returned set has the *same* topology
    classification (empty / interval / disconnected / open-edge) the
    full-vector :func:`k_confidence_set` produces, via the shared
    :func:`_assemble_set` core.

    This is the in-framework version of the hand-rolled re-optimising
    ``theta_builder`` workaround: the inner solve reuses the production
    estimator, so the manifold gauge (the ``k(k-1)/2`` quotient directions of a
    ``PSDFixedRank`` factor) is handled by the existing quotient-aware path
    (horizontal ``G_riem``, ``pinv_eigvalrule`` bread) rather than re-derived at
    the call site. The final p-value's ``gauge_nullspace_dim`` is auto-detected
    from the full profiled ``theta`` (override via ``gauge_nullspace_dim``).

    Parameters
    ----------
    theta_builder : callable ``float -> ParamsLike``
        Lifts a scalar grid value to the FULL parameter PyTree. The fields in
        ``profile`` are read at their built values (the constraint); the other
        leaves' built values are the **warm start** for the inner re-optimise.
        Must return the same PyTree structure for every grid value.
    grid, measure, covariance, model, alpha, statistic
        As for :func:`k_confidence_set`.
    profile : sequence of str
        The interest field names held fixed at each grid value (whole-leaf
        granularity; see :func:`_freeze_fixed_leaves`). Everything else is the
        re-optimised nuisance.
    nuisance_optimizer : optional
        Optimiser for the inner re-optimise. ``None`` (default) lets
        :func:`emu_gmm.estimate` auto-dispatch — ``optimistix_lm`` for a scalar
        nuisance, ``riemannian_lm`` when the nuisance carries a manifold leaf.
        Pass e.g. ``riemannian_tr()`` for a demonstrably non-convex manifold
        criterion.
    nuisance_weighting : optional
        Weighting for the inner re-optimise; defaults to
        :class:`~emu_gmm.weighting.ContinuouslyUpdated` (the estimator default).
    regularization, score_cov_fn, V, L, gauge_nullspace_dim, strong_id_fallback
        Forwarded to the inner :func:`emu_gmm.estimate` (``regularization``
        only) and to the per-point :func:`emu_gmm.inference.k_statistic` (all),
        exactly as :func:`k_confidence_set` forwards them.
    keep_results : bool, default False
        Retain the per-grid :class:`KStatisticResult` on the returned set.
    return_profiled_points : bool, default False
        Also return the tuple of profiled full-``theta`` PyTrees (one per grid
        value) alongside the set, for inspecting the concentrated nuisance path.

    Returns
    -------
    :class:`KConfidenceSet`, or ``(KConfidenceSet, tuple_of_theta)`` when
    ``return_profiled_points`` is True.

    Notes
    -----
    **Subvector reference distribution (#179).** Because the nuisance is
    *concentrated out* (re-optimised at each grid value) rather than held fixed,
    the per-point statistics are referenced to their **subvector** dofs, not the
    full-vector dofs :func:`~emu_gmm.inference.k_statistic` returns:

    - :math:`K \sim \chi^2_{d_I}` where ``d_I = dim(interest)`` is the identified
      dimension of the profiled fields (ambient size minus gauge dim per leaf).
      At the concentrated point the nuisance score is zero by the inner FOC, so
      the full-vector :math:`K` collapses onto the interest block of dimension
      :math:`d_I` (Kleibergen--Mavroeidis 2009).
    - the restricted-model :math:`J \sim \chi^2_{M - d_N}`, with ``d_N`` the
      identified nuisance dimension;
    - :math:`S = J - K \sim \chi^2_{M - p_{id}}` is **unchanged** from the
      full-vector value (``df_K + df_S = df_J`` continues to hold).

    Referencing the profiled :math:`K` to the full-vector :math:`\chi^2_{p_{id}}`
    (as the pre-#179 code did) over-states the dof and makes the set
    systematically **conservative** — badly so when the nuisance is
    high-dimensional (e.g. a ``PSDFixedRank`` factor).

    **Precondition: the nuisance must be strongly identified.** The plug-in
    concentration above is the subvector statistic *evaluated at the restricted
    CUE*, which is identification-robust **in the interest direction** but
    assumes the nuisance is well-identified at each null (the inner solve
    returns a well-separated minimiser and the concentrated score genuinely
    vanishes). If the nuisance is itself weak, this plug-in is not robust and
    the fully-robust alternative is *projection* — sup the statistic over the
    nuisance rather than concentrating it (not implemented here; see the
    :func:`k_confidence_set` Notes for the concentration-vs-projection
    distinction). The interest direction may be weak (an unbounded / open-edge
    set is the honest report); it is the **nuisance** that must be strong.
    Check the nuisance block with
    :func:`~emu_gmm.inference.identification.identification_strength` (pass the
    profiled-out fields as a block) before trusting a profiled set: a weak
    nuisance block invalidates the subvector reference distribution.

    Cost: one full inner estimation per grid point (the issue's acknowledged
    expense). ``jax.clear_caches()`` is called per grid value to avoid the
    per-call JAX cache growth documented in CLAUDE.md (each grid point traces a
    fresh reduced-model closure). Eager-only.
    """
    from emu_gmm.estimator import estimate
    from emu_gmm.weighting import ContinuouslyUpdated

    grid_arr = _validate_grid_args(grid, statistic, alpha)
    if nuisance_weighting is None:
        nuisance_weighting = ContinuouslyUpdated()

    # Build the reduced (nuisance-only) parameter dataclass once from the first
    # grid point; its structure is identical across the grid.
    theta0 = theta_builder(float(grid_arr[0]))
    reducer = _ProfileReducer(theta0, profile)
    interest_dim = _interest_identified_dim(theta0, profile)

    results: list[KStatisticResult] = []
    profiled_points: list[ParamsLike] = []
    for g in grid_arr:
        theta_full_init = theta_builder(float(g))
        free_init = reducer.free_init(theta_full_init)

        def _reduced_model(x: Any, free: Any, _tf: Any = theta_full_init) -> Any:
            return model(x, reducer.recombine(_tf, free))

        inner = estimate(
            _reduced_model,
            measure,
            covariance=covariance,
            weighting=nuisance_weighting,
            regularization=regularization,
            optimizer=nuisance_optimizer,
            theta_init=free_init,
        )
        theta_prof = reducer.recombine(theta_full_init, inner.theta_hat)
        ks = k_statistic(
            theta_prof,
            measure,
            covariance,
            model,
            regularization=regularization,
            score_cov_fn=score_cov_fn,
            V=V,
            L=L,
            gauge_nullspace_dim=gauge_nullspace_dim,
            strong_id_fallback=strong_id_fallback,
        )
        # Re-reference to the SUBVECTOR null distribution (#179): the nuisance
        # is concentrated, so K -> chi^2_{dim(interest)} and the restricted-model
        # J -> chi^2_{M - dim(nuisance)}, NOT the full-vector dofs k_statistic
        # returns. Using the full-vector dofs makes the set systematically
        # conservative (badly so with a high-dimensional nuisance).
        results.append(_rescore_subvector(ks, interest_dim))
        profiled_points.append(theta_prof)
        # Per-iter cache clear: each grid point builds a fresh reduced-model
        # closure (the fixed value is captured), so estimate retraces and the
        # global JAX caches would otherwise grow ~per call (CLAUDE.md).
        jax.clear_caches()

    cs = _assemble_set(grid_arr, results, statistic, alpha, keep_results)
    if return_profiled_points:
        return cs, tuple(profiled_points)
    return cs


def _interest_identified_dim(theta_full: ParamsLike, profile: Sequence[str]) -> int:
    r"""Identified dimension of the profiled (interest) fields --- the subvector dof.

    The subvector test references the profiled K to :math:`\chi^2` on the
    *interest* dimension, which for a gauge-bearing interest leaf is its ambient
    size minus its gauge dimension (the same drop-by-count rule as everywhere
    else). For the usual scalar-field interest this is just the number of fixed
    coordinates.
    """
    from emu_gmm._internal.params import manifold_spec_from_params

    fixed = set(profile)
    spec = manifold_spec_from_params(theta_full)
    total = 0
    for ls in spec.leaf_specs:
        if ls.field_name in fixed:
            size = int(np.prod(ls.ambient_shape)) if ls.ambient_shape != () else 1
            total += size - int(ls.manifold.gauge_dim)
    return total


def _rescore_subvector(ks: KStatisticResult, interest_dim: int) -> KStatisticResult:
    r"""Re-reference a concentrated :class:`KStatisticResult` to the subvector dof (#179).

    At a nuisance-concentrated point the Kleibergen--Mavroeidis subvector
    statistics are :math:`K \sim \chi^2_{d_I}` (``d_I = dim(interest)``, recovered
    because the concentrated nuisance score is zero) and the restricted-model
    :math:`J \sim \chi^2_{M - d_N}` (``d_N`` the concentrated nuisance
    dimension), while :math:`S = J - K \sim \chi^2_{M - p_{id}}` is **unchanged**
    from the full-vector value. The statistic *values* are the same as
    :func:`k_statistic` returns; only the reference dofs (and hence the p-values)
    change. Valid when the **nuisance is strongly identified** (see
    :func:`profiled_k_confidence_set`).
    """
    M = int(ks.df_J)  # k_statistic sets df_J == M
    p_id = int(ks.df_K)  # k_statistic sets df_K == full identified dim
    d_i = int(interest_dim)
    df_K = d_i
    df_S = int(ks.df_S)  # == M - p_id, unchanged under concentration
    df_J = M - p_id + d_i  # restricted-model over-id; == df_K + df_S
    p_K = _kappa_chi2_sf(ks.K, df_K)
    p_J = _kappa_chi2_sf(ks.J, df_J)
    return KStatisticResult(
        K=ks.K,
        S=ks.S,
        J=ks.J,
        p_K=p_K,
        p_S=ks.p_S,
        p_J=p_J,
        df_K=df_K,
        df_S=df_S,
        df_J=df_J,
    )


__all__ = ["KConfidenceSet", "k_confidence_set", "profiled_k_confidence_set"]
