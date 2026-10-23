"""Finite-grid empirical law of repeated inverted confidence sets (#180).

This is a sibling of the point-valued :class:`EstimatorLaw`: a scanned grid
does not certify membership between its points or emptiness of the full
parameter space. See the existing definitions in ledger Section 12.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from emu_gmm.inference.confidence_set import (
    KConfidenceSet,
    _classify,
    _connected_runs,
    _interp_edge,
)

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SetQueryReport:
    """A finite-grid capacity or coverage estimate with its denominator.

    ``n_total`` counts the original law before conditioning, ``n_used`` the
    selected draws, and ``n_decidable`` those with a valid event verdict.
    ``n_invalid`` and ``n_nonconverged`` count undecidable draws with the
    corresponding queried-point event; one draw can contribute to both.
    """

    points: tuple[float, ...]
    n_total: int
    n_used: int
    n_decidable: int
    n_hit: int
    n_invalid: int
    n_nonconverged: int
    estimate: float

    @property
    def n_unknown(self) -> int:
        return self.n_used - self.n_decidable

    @property
    def n_miss(self) -> int:
        return self.n_decidable - self.n_hit


@dataclass(frozen=True)
class SetEventShare:
    """Share of the current selection satisfying a scanned-grid event."""

    event: str
    negate: bool
    n_selected: int
    n_total: int
    fraction_all: float


def _readonly_copy(value: Any, *, dtype: Any) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _copy_set(draw: KConfidenceSet) -> KConfidenceSet:
    return replace(
        draw,
        grid=_readonly_copy(draw.grid, dtype=float),
        p_grid=_readonly_copy(draw.p_grid, dtype=float),
        in_set=_readonly_copy(draw.in_set, dtype=bool),
        intervals=[(float(lo), float(hi)) for lo, hi in draw.intervals],
        results=None,
    )


def _validate_set(draw: KConfidenceSet) -> None:
    grid = np.asarray(draw.grid)
    p = np.asarray(draw.p_grid)
    inside = np.asarray(draw.in_set)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("KConfidenceSet grid must have at least two points")
    if not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
        raise ValueError("KConfidenceSet grid must be finite and strictly increasing")
    if p.shape != grid.shape or inside.shape != grid.shape or inside.dtype != bool:
        raise ValueError("KConfidenceSet p_grid/in_set must match its grid")
    if not np.isfinite(draw.alpha) or not 0 < draw.alpha < 1:
        raise ValueError("KConfidenceSet alpha must lie strictly between 0 and 1")
    invalid = tuple(int(i) for i in np.flatnonzero(~np.isfinite(p)))
    noncon = tuple(int(i) for i in draw.nonconverged_indices)
    if (
        invalid != tuple(draw.invalid_indices)
        or draw.n_invalid != len(invalid)
        or draw.n_nonconverged != len(noncon)
        or tuple(sorted(set(noncon))) != noncon
        or any(i < 0 or i >= grid.size for i in noncon)
    ):
        raise ValueError("KConfidenceSet invalid/nonconverged metadata disagree")
    expected = np.isfinite(p) & (p > draw.alpha)
    expected[list(noncon)] = False
    if not np.array_equal(inside, expected):
        raise ValueError("KConfidenceSet in_set disagrees with p and event flags")
    runs = _connected_runs(inside)
    topology, open_left, open_right = _classify(runs, grid.size)
    if (
        draw.topology != topology
        or draw.open_left != open_left
        or draw.open_right != open_right
        or len(draw.intervals) != len(runs)
    ):
        raise ValueError("KConfidenceSet topology disagrees with in_set")
    p_edge = np.array(p, copy=True)
    p_edge[list(noncon)] = np.nan
    for (i0, i1), (lo, hi) in zip(runs, draw.intervals, strict=True):
        expected_lo = (
            float(grid[0])
            if i0 == 0
            else _interp_edge(
                grid[i0 - 1], grid[i0], p_edge[i0 - 1], p_edge[i0], draw.alpha
            )
        )
        expected_hi = (
            float(grid[-1])
            if i1 == grid.size - 1
            else _interp_edge(
                grid[i1], grid[i1 + 1], p_edge[i1], p_edge[i1 + 1], draw.alpha
            )
        )
        if lo != expected_lo or hi != expected_hi:
            raise ValueError("KConfidenceSet intervals disagree with scanned p-values")


@dataclass(frozen=True)
class RandomSetLaw:
    """Empirical law over repeated same-grid :class:`KConfidenceSet` draws.

    All membership queries are exact scanned-grid queries. A point excluded
    due to an invalid p-value or nonconverged nuisance fit has an *unknown*
    verdict, never a valid rejection. Intervals and open edge flags are
    descriptive and do not authorize membership away from scanned points.
    """

    _draws: tuple[KConfidenceSet, ...]
    grid: np.ndarray
    alpha: float
    statistic: str
    origin_indices: tuple[int, ...]
    source_n_total: int

    def __post_init__(self) -> None:
        if (
            self.grid.ndim != 1
            or self.grid.size < 2
            or not np.isfinite(self.grid).all()
            or not np.all(np.diff(self.grid) > 0)
            or not np.isfinite(self.alpha)
            or not 0 < self.alpha < 1
            or self.statistic not in {"K", "S", "J"}
            or len(self.origin_indices) != len(self._draws)
            or not isinstance(self.source_n_total, int)
            or self.source_n_total < len(self._draws)
            or any(
                not isinstance(i, int) or i < 0 or i >= self.source_n_total
                for i in self.origin_indices
            )
            or tuple(sorted(set(self.origin_indices))) != self.origin_indices
        ):
            raise ValueError("RandomSetLaw grid or origin metadata are inconsistent")
        for draw in self._draws:
            _validate_set(draw)
            if (
                not np.array_equal(draw.grid, self.grid)
                or draw.alpha != self.alpha
                or draw.statistic != self.statistic
            ):
                raise ValueError(
                    "all KConfidenceSet draws need one grid/alpha/statistic"
                )
        object.__setattr__(self, "grid", _readonly_copy(self.grid, dtype=float))
        object.__setattr__(self, "_draws", tuple(_copy_set(d) for d in self._draws))

    @classmethod
    def from_k_confidence_sets(cls, sets: Sequence[KConfidenceSet]) -> RandomSetLaw:
        """Build from one or more independent set realizations on one grid."""
        draws = tuple(sets)
        if not draws:
            raise ValueError("at least one KConfidenceSet is needed to define a grid")
        for draw in draws:
            _validate_set(draw)
        first = draws[0]
        for draw in draws[1:]:
            if (
                not np.array_equal(draw.grid, first.grid)
                or draw.alpha != first.alpha
                or draw.statistic != first.statistic
            ):
                raise ValueError(
                    "all KConfidenceSet draws need one grid/alpha/statistic"
                )
        return cls(
            draws,
            first.grid,
            float(first.alpha),
            first.statistic,
            tuple(range(len(draws))),
            len(draws),
        )

    @property
    def draws(self) -> tuple[KConfidenceSet, ...]:
        """Copies of selected set draws, without live per-grid estimator results."""
        return tuple(_copy_set(draw) for draw in self._draws)

    @property
    def n_used(self) -> int:
        return len(self._draws)

    def _point_index(self, point: float) -> int:
        p = float(point)
        if not np.isfinite(p):
            raise ValueError("query point must be finite")
        found = np.flatnonzero(self.grid == p)
        if found.size != 1:
            raise ValueError("query point must equal one scanned grid point exactly")
        return int(found[0])

    def point_verdicts(self, point: float) -> tuple[bool | None, ...]:
        """Per-draw exact-grid membership; ``None`` means invalid/unknown."""
        j = self._point_index(point)
        out: list[bool | None] = []
        for draw in self._draws:
            if j in draw.invalid_indices or j in draw.nonconverged_indices:
                out.append(None)
            else:
                out.append(bool(draw.in_set[j]))
        return tuple(out)

    def coverage(self, point: float) -> SetQueryReport:
        """Estimate ``P(point in S)`` over draws with a valid point verdict."""
        return self.capacity((point,))

    def capacity(self, points: Sequence[float]) -> SetQueryReport:
        """Estimate ``P(S intersects points)`` for a nonempty grid subset.

        A draw is a hit if any queried point is valid and accepted. Otherwise
        it is unknown if any queried point is invalid/nonconverged, and a miss
        only if every point is valid and rejected (three-valued event logic).
        """
        query = tuple(float(p) for p in points)
        if not query:
            raise ValueError("capacity requires at least one grid point")
        indices = tuple(dict.fromkeys(self._point_index(p) for p in query))
        n_hit = n_decidable = n_invalid = n_nonconverged = 0
        for draw in self._draws:
            invalid = any(j in draw.invalid_indices for j in indices)
            noncon = any(j in draw.nonconverged_indices for j in indices)
            hit = any(
                draw.in_set[j]
                for j in indices
                if j not in draw.invalid_indices and j not in draw.nonconverged_indices
            )
            if hit:
                n_hit += 1
                n_decidable += 1
            elif invalid or noncon:
                n_invalid += int(invalid)
                n_nonconverged += int(noncon)
            else:
                n_decidable += 1
        return SetQueryReport(
            points=tuple(float(self.grid[j]) for j in indices),
            n_total=self.source_n_total,
            n_used=self.n_used,
            n_decidable=n_decidable,
            n_hit=n_hit,
            n_invalid=n_invalid,
            n_nonconverged=n_nonconverged,
            estimate=n_hit / n_decidable if n_decidable else float("nan"),
        )

    def prob(self, point: float) -> SetQueryReport:
        """Structured point capacity; see :meth:`coverage` for its denominator."""
        return self.coverage(point)

    def _event_mask(self, event: str, negate: bool) -> tuple[bool, ...]:
        if event != "grid_empty":
            raise ValueError("supported set event: 'grid_empty'")
        return tuple((not any(draw.in_set)) != negate for draw in self._draws)

    def event_share(self, event: str, *, negate: bool = False) -> SetEventShare:
        """Count a scanned-grid event within the current selection."""
        selected = sum(self._event_mask(event, negate))
        return SetEventShare(
            event=f"~{event}" if negate else event,
            negate=negate,
            n_selected=selected,
            n_total=self.n_used,
            fraction_all=selected / self.n_used if self.n_used else float("nan"),
        )

    def given(self, event: str, *, negate: bool = False) -> RandomSetLaw:
        """Select draws by a scanned-grid event, retaining source indices."""
        mask = self._event_mask(event, negate)
        return replace(
            self,
            _draws=tuple(
                draw for draw, keep in zip(self._draws, mask, strict=True) if keep
            ),
            origin_indices=tuple(
                i for i, keep in zip(self.origin_indices, mask, strict=True) if keep
            ),
        )

    def sample(self, key: jax.Array, n: int) -> tuple[KConfidenceSet, ...]:
        """Resample set realizations; an empty conditional law cannot be sampled."""
        if n < 0 or (n > 0 and not self._draws):
            raise ValueError("sample requires n >= 0 and a nonempty law for n > 0")
        if n == 0:
            return ()
        indices = np.asarray(jax.random.randint(key, (n,), 0, self.n_used))
        return tuple(_copy_set(self._draws[int(i)]) for i in indices)

    @staticmethod
    def _point_summary_error() -> None:
        raise TypeError("RandomSetLaw has no point-valued mean/cov/se/quantile")

    def mean(self, *args: Any, **kwargs: Any) -> None:
        self._point_summary_error()

    def cov(self, *args: Any, **kwargs: Any) -> None:
        self._point_summary_error()

    def se(self, *args: Any, **kwargs: Any) -> None:
        self._point_summary_error()

    def quantile(self, *args: Any, **kwargs: Any) -> None:
        self._point_summary_error()

    def save(self, path: str | Path) -> None:
        """Write a versioned inert NPZ artifact, with no pickle or model code."""
        manifest = {
            "kind": "random_set_law",
            "schema_version": _SCHEMA_VERSION,
            "alpha": self.alpha,
            "statistic": self.statistic,
            "source_n_total": self.source_n_total,
            "origin_indices": list(self.origin_indices),
            "topology": [d.topology for d in self._draws],
            "open_left": [d.open_left for d in self._draws],
            "open_right": [d.open_right for d in self._draws],
            "intervals": [d.intervals for d in self._draws],
        }
        shape = (self.n_used, self.grid.size)
        with Path(path).open("wb") as target:
            np.savez_compressed(
                target,
                manifest=np.asarray(json.dumps(manifest, allow_nan=False)),
                grid=self.grid,
                p_grid=(
                    np.stack([d.p_grid for d in self._draws])
                    if self.n_used
                    else np.empty(shape)
                ),
                in_set=(
                    np.stack([d.in_set for d in self._draws])
                    if self.n_used
                    else np.empty(shape, dtype=bool)
                ),
                invalid=(
                    np.stack([~np.isfinite(d.p_grid) for d in self._draws])
                    if self.n_used
                    else np.empty(shape, dtype=bool)
                ),
                nonconverged=(
                    np.stack(
                        [
                            np.isin(np.arange(self.grid.size), d.nonconverged_indices)
                            for d in self._draws
                        ]
                    )
                    if self.n_used
                    else np.empty(shape, dtype=bool)
                ),
            )

    @classmethod
    def load(cls, path: str | Path) -> RandomSetLaw:
        """Rebuild a queryable set law from its inert NPZ artifact."""
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {
                "manifest",
                "grid",
                "p_grid",
                "in_set",
                "invalid",
                "nonconverged",
            }:
                raise ValueError("random-set artifact has missing or unexpected arrays")
            manifest = json.loads(str(data["manifest"].item()))
            if (
                manifest.get("kind") != "random_set_law"
                or manifest.get("schema_version") != _SCHEMA_VERSION
            ):
                raise ValueError("unsupported random-set artifact kind/schema")
            grid = np.array(data["grid"], dtype=float, copy=True)
            p_grid = np.array(data["p_grid"], dtype=float, copy=True)
            in_set = np.array(data["in_set"], copy=True)
            invalid = np.array(data["invalid"], copy=True)
            noncon = np.array(data["nonconverged"], copy=True)
        n = len(manifest["origin_indices"])
        if (
            grid.ndim != 1
            or grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
            or any(a.shape != (n, grid.size) for a in (p_grid, in_set, invalid, noncon))
            or any(a.dtype != bool for a in (in_set, invalid, noncon))
            or any(
                len(manifest[k]) != n
                for k in ("topology", "open_left", "open_right", "intervals")
            )
            or not np.array_equal(invalid, ~np.isfinite(p_grid))
        ):
            raise ValueError("random-set artifact has inconsistent dimensions or flags")
        draws = []
        for i in range(n):
            invalid_i = tuple(int(j) for j in np.flatnonzero(invalid[i]))
            noncon_i = tuple(int(j) for j in np.flatnonzero(noncon[i]))
            draw = KConfidenceSet(
                grid=grid,
                p_grid=p_grid[i],
                in_set=in_set[i],
                intervals=[
                    (float(lo), float(hi)) for lo, hi in manifest["intervals"][i]
                ],
                topology=manifest["topology"][i],
                open_left=manifest["open_left"][i],
                open_right=manifest["open_right"][i],
                alpha=float(manifest["alpha"]),
                statistic=manifest["statistic"],
                n_invalid=len(invalid_i),
                invalid_indices=invalid_i,
                n_nonconverged=len(noncon_i),
                nonconverged_indices=noncon_i,
            )
            _validate_set(draw)
            draws.append(draw)
        source_n_total = manifest["source_n_total"]
        origins = tuple(manifest["origin_indices"])
        if (
            not isinstance(source_n_total, int)
            or source_n_total < n
            or any(
                not isinstance(i, int) or i < 0 or i >= source_n_total for i in origins
            )
            or tuple(sorted(origins)) != origins
            or len(set(origins)) != len(origins)
        ):
            raise ValueError("random-set artifact has invalid origin indices")
        return cls(
            tuple(draws),
            _readonly_copy(grid, dtype=float),
            float(manifest["alpha"]),
            manifest["statistic"],
            origins,
            source_n_total,
        )
