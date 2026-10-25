"""Numerical FOC certification guards; no fits or statistical cutoffs (#180)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.inference.confidence_set import _foc_decrement


def _decrement(H, g, gauge_dim):
    return _foc_decrement(
        SimpleNamespace(gradient=jnp.asarray(g), gn_hessian=jnp.asarray(H)),
        gauge_dim,
    )


def _one_gauge_gram():
    """Known rank-four Gram factor with null (1,1,0,-1,0).

    The last column is orthogonal to the first four, so a score of two in
    that coordinate has decrement four. Forming the Gram introduces realistic
    roundoff in the exact gauge without introducing another physical null.
    """
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(5, 5)))
    R = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    B = np.column_stack((Q[:, :3] @ R, Q[:, 3]))
    return B.T @ B, np.array([0.0, 0.0, 0.0, 0.0, 2.0])


@pytest.mark.parametrize(
    "units",
    [
        np.ones(5),
        np.array([1.0, 1.0, 1.0, 1.0, 1e-12]),
        np.array([1e90, 1e-90, 1e40, 1e-40, 1e-12]),
    ],
    ids=["ordinary", "tiny-physical-direction", "mixed-extreme-units"],
)
def test_known_gauge_quadratic_form_is_invariant_to_coordinate_units(units):
    H, g = _one_gauge_gram()
    H_units = (units[:, None] * H) * units[None, :]
    g_units = units * g
    actual = _decrement(H_units, g_units, 1)
    assert actual == pytest.approx(4.0, rel=1e-10, abs=1e-10)
    # The tiny direction is identified, not another gauge to discard. Losing
    # it would falsely satisfy the existing 1e-3 profiler FOC tolerance.
    assert actual > 1.0


@pytest.mark.parametrize(
    "units",
    [np.ones(2), np.array([1e150, 1e-150]), np.array([1e-150, 1e150])],
    ids=["ordinary", "large-small", "small-large"],
)
def test_no_gauge_quadratic_form_is_invariant_to_coordinate_units(units):
    H = np.array([[2.0, 0.2], [0.2, 3.0]])
    g = np.array([0.25, -0.5])
    expected = float(g @ np.linalg.solve(H, g))
    actual = _decrement((units[:, None] * H) * units[None, :], units * g, 0)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1e-150, 1e150])
def test_exact_axis_aligned_known_gauge_may_have_zero_diagonal(scale):
    H = np.diag([0.0, 2.0 * scale**2, 4.0])
    g = np.array([0.0, scale, 2.0])
    assert _decrement(H, g, 1) == pytest.approx(1.5, rel=1e-12)


def _assert_uncertifiable(H, g, gauge_dim):
    actual = _decrement(H, g, gauge_dim)
    # NaN or None both prevent the profiler's finite-and-<=tol fallback.
    # A finite zero/negative value would silently certify these bad inputs.
    assert actual is None or not np.isfinite(actual)


@pytest.mark.parametrize(
    "H,gauge_dim",
    [
        (np.diag([0.0, 1.0]), 0),
        (np.diag([0.0, 0.0, 1.0]), 1),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), 0),
        (np.array([[1.0, 1.0 - 1e-16], [1.0 - 1e-16, 1.0]]), 0),
        (
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0 - 1e-16], [0.0, 1.0 - 1e-16, 1.0]]
            ),
            1,
        ),
    ],
    ids=[
        "unexplained-zero-axis",
        "extra-zero-beyond-known-gauge",
        "exact-collinearity",
        "unresolved-positive-collinearity",
        "unresolved-collinearity-beyond-known-gauge",
    ],
)
def test_extra_unresolved_direction_cannot_certify_zero_score(H, gauge_dim):
    _assert_uncertifiable(H, np.zeros(len(H)), gauge_dim)


@pytest.mark.parametrize(
    "H,g",
    [
        (np.diag([-1.0, 2.0]), np.zeros(2)),
        (np.diag([-1.0, 2.0]), np.array([1.0, 0.0])),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), np.zeros(2)),
        (np.diag([np.inf, 1.0]), np.zeros(2)),
        (np.diag([np.nan, 1.0]), np.zeros(2)),
        (np.eye(2), np.array([np.inf, 0.0])),
        (np.eye(2), np.array([np.nan, 0.0])),
        (np.diag([1e-300, 1.0]), np.array([1e300, 0.0])),
    ],
    ids=[
        "negative-diagonal-zero-score",
        "negative-quadratic-form",
        "indefinite-positive-diagonal",
        "nonfinite-information-inf",
        "nonfinite-information-nan",
        "nonfinite-score-inf",
        "nonfinite-score-nan",
        "scaled-score-overflow",
    ],
)
def test_invalid_information_or_score_cannot_certify(H, g):
    _assert_uncertifiable(H, g, 0)


def test_material_negative_curvature_cannot_be_discarded_as_known_gauge():
    H = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _assert_uncertifiable(H, np.zeros(3), 1)
