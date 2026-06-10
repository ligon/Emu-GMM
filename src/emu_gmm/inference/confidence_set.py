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
- **NaN is an event, not a value** (the #140 convention): a grid point
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

import jax
import numpy as np
from jaxtyping import Array, Float

from emu_gmm.inference.k_statistic import KStatisticResult, k_statistic
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
) -> KConfidenceSet:
    """Invert the K-statistic over a 1-D grid of nulls (#41 PR (b)).

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
    The K-statistic tests the FULL parameter vector at each null; this
    routine scans a one-dimensional family of nulls. For a
    multi-parameter model, ``theta_builder`` therefore traces a curve
    through the parameter space (e.g. varying one coordinate with the
    others held at calibrated values) — the resulting set is a slice of
    the joint robust set along that curve, NOT a projection-based
    marginal set for one coordinate. Projection (profiling the K over
    nuisance directions) is deliberately out of scope here.
    """
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


__all__ = ["KConfidenceSet", "k_confidence_set"]
