"""Tests for emu_gmm.inference.confidence_set (#41 PR (b)).

Two layers:

(1) Pure topology/classification unit tests on the helpers
    (``_connected_runs`` / ``_interp_edge`` / ``_classify``) — these
    cover every topology label exhaustively, including the shapes that
    are awkward to realise from data (disconnected, unbounded-left).

(2) End-to-end inversion on a linear regression fixture (scalar theta,
    M = 3, p = 1, truth theta = 1): the K-based set is an interval
    around the truth; edge-clipped grids surface the open-edge
    topologies; the NaN policy and the input validation are loud.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import IIDCovariance
from emu_gmm.inference import KConfidenceSet, KStatisticResult, k_confidence_set
from emu_gmm.inference.confidence_set import (
    _classify,
    _connected_runs,
    _interp_edge,
)
from emu_gmm.manifolds import Positive
from emu_gmm.measures import EmpiricalMeasure

# ---------------------------------------------------------------------------
# (1) Topology helpers: exhaustive over the label vocabulary.
# ---------------------------------------------------------------------------


class TestConnectedRuns:
    def test_empty(self):
        assert _connected_runs(np.array([False, False, False])) == []

    def test_single_run_interior(self):
        assert _connected_runs(np.array([False, True, True, False])) == [(1, 2)]

    def test_multiple_runs(self):
        mask = np.array([True, False, True, True, False, True])
        assert _connected_runs(mask) == [(0, 0), (2, 3), (5, 5)]

    def test_all_true(self):
        assert _connected_runs(np.array([True, True])) == [(0, 1)]


class TestInterpEdge:
    def test_linear_crossing(self):
        # p goes 0.01 -> 0.09 across [1.0, 2.0]; alpha = 0.05 crosses
        # exactly halfway.
        assert _interp_edge(1.0, 2.0, 0.01, 0.09, 0.05) == pytest.approx(1.5)

    def test_clipped_to_segment(self):
        # alpha outside [p0, p1] clips to the nearer endpoint.
        assert _interp_edge(1.0, 2.0, 0.2, 0.3, 0.05) == pytest.approx(1.0)

    def test_flat_segment_falls_back(self):
        out = _interp_edge(1.0, 2.0, 0.05, 0.05, 0.05)
        assert out in (1.0, 2.0)

    def test_nan_neighbour_falls_back_inward(self):
        # An invalid neighbour gives nothing to interpolate against;
        # the inner (valid, in-set) point is returned.
        assert _interp_edge(1.0, 2.0, float("nan"), 0.5, 0.05) == 2.0


class TestClassify:
    @pytest.mark.parametrize(
        "runs, n, expected",
        [
            ([], 5, ("empty", False, False)),
            ([(1, 3)], 5, ("interval", False, False)),
            ([(0, 2)], 5, ("unbounded-left", True, False)),
            ([(2, 4)], 5, ("unbounded-right", False, True)),
            ([(0, 4)], 5, ("full-grid", True, True)),
            ([(0, 0), (2, 3)], 5, ("disconnected", True, False)),
            ([(1, 1), (3, 4)], 5, ("disconnected", False, True)),
            ([(1, 1), (3, 3)], 5, ("disconnected", False, False)),
        ],
    )
    def test_labels(self, runs, n, expected):
        assert _classify(runs, n) == expected


# ---------------------------------------------------------------------------
# (2) End-to-end on a linear regression fixture.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _OneParam:
    theta: float


def _linear_psi(x, params):
    """r = y - x*theta; moments (r*x, r*x^2, r). M=3, p=1."""
    y, xval, _one = x[0], x[1], x[2]
    resid = y - xval * params.theta
    return jnp.array([resid * xval, resid * xval * xval, resid])


THETA_TRUE = 1.0


def _measure(seed: int = 0, n: int = 300) -> EmpiricalMeasure:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = x * THETA_TRUE + rng.standard_normal(n)
    data = jnp.asarray(np.column_stack([y, x, np.ones(n)]))
    return EmpiricalMeasure(x=data, mask=jnp.ones((n, 3)), weights=jnp.ones(n))


def _builder(g: float) -> _OneParam:
    return _OneParam(theta=float(g))


@pytest.fixture(scope="module")
def measure():
    return _measure(seed=0)


class TestIntervalAroundTruth:
    def test_interval_topology_and_coverage(self, measure):
        cs = k_confidence_set(
            _builder,
            np.arange(0.2, 1.9, 0.1),
            measure,
            IIDCovariance(),
            _linear_psi,
        )
        assert isinstance(cs, KConfidenceSet)
        assert cs.topology == "interval"
        assert len(cs.intervals) == 1
        lo, hi = cs.intervals[0]
        assert lo < THETA_TRUE < hi
        # Far nulls rejected on both flanks.
        assert not cs.in_set[0] and not cs.in_set[-1]
        assert cs.n_invalid == 0

    def test_alpha_monotone(self, measure):
        grid = np.arange(0.2, 1.9, 0.1)
        loose = k_confidence_set(
            _builder, grid, measure, IIDCovariance(), _linear_psi, alpha=0.05
        )
        tight = k_confidence_set(
            _builder, grid, measure, IIDCovariance(), _linear_psi, alpha=0.5
        )
        # Higher alpha -> fewer accepted grid points (subset).
        assert tight.in_set.sum() <= loose.in_set.sum()
        assert bool(np.all(loose.in_set | ~tight.in_set))

    def test_summary_mentions_interval(self, measure):
        cs = k_confidence_set(
            _builder,
            np.arange(0.2, 1.9, 0.1),
            measure,
            IIDCovariance(),
            _linear_psi,
        )
        s = cs.summary()
        assert "interval" in s
        assert "U" not in s  # single component


class TestOpenEdges:
    def test_unbounded_right(self, measure):
        # Grid ends at the truth: the last run touches the right edge.
        cs = k_confidence_set(
            _builder,
            np.arange(0.2, 1.01, 0.1),
            measure,
            IIDCovariance(),
            _linear_psi,
        )
        assert cs.topology == "unbounded-right"
        assert cs.open_right and not cs.open_left
        # The open edge is reported at the grid bound, un-interpolated.
        assert cs.intervals[-1][1] == pytest.approx(float(cs.grid[-1]))

    def test_full_grid(self, measure):
        cs = k_confidence_set(
            _builder,
            np.array([0.95, 1.0, 1.05]),
            measure,
            IIDCovariance(),
            _linear_psi,
        )
        assert cs.topology == "full-grid"
        assert cs.open_left and cs.open_right

    def test_empty(self, measure):
        cs = k_confidence_set(
            _builder,
            np.array([3.0, 3.5, 4.0]),
            measure,
            IIDCovariance(),
            _linear_psi,
        )
        assert cs.topology == "empty"
        assert cs.intervals == []
        assert "EMPTY" in cs.summary()


class TestStatisticChoice:
    def test_J_inversion_runs(self, measure):
        cs = k_confidence_set(
            _builder,
            np.arange(0.5, 1.6, 0.25),
            measure,
            IIDCovariance(),
            _linear_psi,
            statistic="J",
        )
        assert cs.statistic == "J"
        # AR-style set also contains the truth here.
        idx_truth = int(np.argmin(np.abs(cs.grid - THETA_TRUE)))
        assert cs.in_set[idx_truth]

    def test_keep_results(self, measure):
        grid = np.array([0.9, 1.0, 1.1])
        cs = k_confidence_set(
            _builder,
            grid,
            measure,
            IIDCovariance(),
            _linear_psi,
            keep_results=True,
        )
        assert cs.results is not None and len(cs.results) == len(grid)
        assert all(isinstance(r, KStatisticResult) for r in cs.results)
        # p_grid is exactly the per-result p_K.
        for p, r in zip(cs.p_grid, cs.results, strict=False):
            assert p == pytest.approx(float(np.asarray(r.p_K)), abs=0.0)


class TestNaNPolicy:
    def test_nan_pvalues_are_loud_events(self, measure):
        """NaN p never silently joins the set (#140 convention)."""
        cs = k_confidence_set(
            _builder,
            np.array([0.9, 1.0, 1.1]),
            measure,
            IIDCovariance(),
            _linear_psi,
            # Poison the orthogonalisation: NaN Sigma -> NaN K -> NaN p.
            score_cov_fn=lambda model, theta: jnp.full((1, 3, 3), jnp.nan),
        )
        assert cs.n_invalid == 3
        assert cs.invalid_indices == (0, 1, 2)
        assert not cs.in_set.any()
        assert cs.topology == "empty"
        assert "WARNING" in cs.summary()


class TestValidation:
    def test_grid_must_increase(self, measure):
        with pytest.raises(ValueError, match="strictly increasing"):
            k_confidence_set(
                _builder,
                np.array([1.0, 0.5, 2.0]),
                measure,
                IIDCovariance(),
                _linear_psi,
            )

    def test_grid_too_short(self, measure):
        with pytest.raises(ValueError, match="at least 2"):
            k_confidence_set(
                _builder, np.array([1.0]), measure, IIDCovariance(), _linear_psi
            )

    def test_bad_statistic(self, measure):
        with pytest.raises(ValueError, match="statistic"):
            k_confidence_set(
                _builder,
                np.array([0.9, 1.1]),
                measure,
                IIDCovariance(),
                _linear_psi,
                statistic="W",
            )

    def test_bad_alpha(self, measure):
        with pytest.raises(ValueError, match="alpha"):
            k_confidence_set(
                _builder,
                np.array([0.9, 1.1]),
                measure,
                IIDCovariance(),
                _linear_psi,
                alpha=1.5,
            )

    def test_kstat_dispatch_errors_surface(self, measure):
        """k_statistic's loud dispatch fires through the inverter."""
        from emu_gmm.measures import AnalyticalMeasure

        analytical = AnalyticalMeasure(
            expectation_fn=lambda model, theta: jnp.array([theta.theta - 1.0, 0.0, 0.0])
        )
        from emu_gmm.covariance import AnalyticalCovariance

        cov = AnalyticalCovariance(covariance_fn=lambda model, theta: jnp.eye(3))
        with pytest.raises(ValueError, match="strong_id_fallback"):
            k_confidence_set(
                _builder,
                np.array([0.9, 1.1]),
                analytical,
                cov,
                _linear_psi,
            )


class TestManifoldBuilder:
    """The builder may return manifold-annotated trees (sigma > 0)."""

    def test_positive_annotated_param(self, measure):
        @jdc.pytree_dataclass
        class _PositiveParam:
            theta: jnp.ndarray

            __emu_manifolds__ = {"theta": Positive()}

        def psi(x, params):
            y, xval, _one = x[0], x[1], x[2]
            resid = y - xval * params.theta
            return jnp.array([resid * xval, resid * xval * xval, resid])

        cs = k_confidence_set(
            lambda g: _PositiveParam(theta=jnp.asarray(float(g))),
            np.geomspace(0.25, 4.0, 13),  # positive, geometric spacing
            measure,
            IIDCovariance(),
            psi,
        )
        assert cs.topology in ("interval", "unbounded-left", "unbounded-right")
        idx_truth = int(np.argmin(np.abs(cs.grid - THETA_TRUE)))
        assert cs.in_set[idx_truth]
