"""Dimension-guard regression tests for ``estimate()``.

The estimator previously failed in opaque ways on three degenerate
shapes:

- ``M = 0``: ``jnp.linalg.cond(V)`` raised "input array must not be
  empty" deep inside ``DiagonalTikhonov.apply``;
- ``K = 0``: ``flatten_params`` called ``jnp.stack([])`` and raised
  "Need at least one array to stack";
- ``K > M``: silent ``inf`` / ``nan`` ``Sigma_theta`` from inverting a
  rank-deficient information matrix.

The fix validates ``M >= 1``, ``K >= 1``, and ``M >= K`` at the top of
``estimate()`` and raises a typed
:class:`emu_gmm.Emu_GMM_DimensionError` with an actionable message.

Reference: docs/reviews/v1x-test-gaps.org HIGH findings for M=0, K=0,
and K>M.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm import (
    AnalyticalCovariance,
    AnalyticalMeasure,
    ContinuouslyUpdated,
    DiagonalTikhonov,
    Emu_GMM_DimensionError,
    Identity,
    estimate,
    optimistix_lm,
)

# ---------------------------------------------------------------------------
# Fixture dataclasses
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _P0:
    """Empty parameter dataclass: K = 0."""

    pass


@jdc.pytree_dataclass
class _P1:
    """Single-parameter dataclass."""

    a: float


@jdc.pytree_dataclass
class _P3:
    """Three-parameter dataclass: drives K > M when M = 1."""

    a: float
    b: float
    c: float


def _dummy_model(x, theta):
    """Placeholder ``StructuralModel``; AnalyticalMeasure ignores it."""
    del x, theta
    return jnp.zeros((1,))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDimensionGuards:
    def test_M_zero_raises_typed_error(self):
        """A measure returning an empty (M=0) moment vector triggers
        ``Emu_GMM_DimensionError`` at the top of ``estimate``."""

        def empty_moments(model, theta):
            del model, theta
            return jnp.zeros((0,))

        measure = AnalyticalMeasure(expectation_fn=empty_moments)

        def cov_fn(model, theta):
            del model, theta
            return jnp.zeros((0, 0))

        covariance = AnalyticalCovariance(covariance_fn=cov_fn)

        with pytest.raises(Emu_GMM_DimensionError, match="M >= 1"):
            estimate(
                model=_dummy_model,
                measure=measure,
                covariance=covariance,
                weighting=Identity(),
                regularization=DiagonalTikhonov(),
                optimizer=optimistix_lm(),
                theta_init=_P1(a=0.5),
            )

    def test_K_zero_raises_typed_error(self):
        """An empty parameter dataclass triggers
        ``Emu_GMM_DimensionError`` before ``flatten_params`` is hit."""

        def two_moments(model, theta):
            del model, theta
            return jnp.array([0.1, -0.2])

        measure = AnalyticalMeasure(expectation_fn=two_moments)

        def cov_fn(model, theta):
            del model, theta
            return jnp.eye(2)

        covariance = AnalyticalCovariance(covariance_fn=cov_fn)

        with pytest.raises(Emu_GMM_DimensionError, match="K >= 1"):
            estimate(
                model=_dummy_model,
                measure=measure,
                covariance=covariance,
                weighting=Identity(),
                regularization=DiagonalTikhonov(),
                optimizer=optimistix_lm(),
                theta_init=_P0(),
            )

    def test_K_greater_than_M_raises_typed_error(self):
        """Under-identified problem (K > M) triggers
        ``Emu_GMM_DimensionError`` before silent inf/nan Sigma_theta."""

        def one_moment(model, theta):
            del model
            return jnp.array([theta.a + theta.b + theta.c])

        measure = AnalyticalMeasure(expectation_fn=one_moment)

        def cov_fn(model, theta):
            del model, theta
            return jnp.eye(1)

        covariance = AnalyticalCovariance(covariance_fn=cov_fn)

        with pytest.raises(Emu_GMM_DimensionError, match="M >= K"):
            estimate(
                model=_dummy_model,
                measure=measure,
                covariance=covariance,
                weighting=ContinuouslyUpdated(),
                regularization=DiagonalTikhonov(),
                optimizer=optimistix_lm(),
                theta_init=_P3(a=0.0, b=0.0, c=0.0),
            )

    def test_well_dimensioned_problem_runs(self):
        """The just-identified case (M = K = 2) is allowed by the
        guards (no error raised). Sanity check that the guards don't
        over-reject."""

        def two_moments(model, theta):
            del model
            return jnp.array([theta.a, theta.a + 1.0])

        measure = AnalyticalMeasure(expectation_fn=two_moments)

        def cov_fn(model, theta):
            del model, theta
            return jnp.eye(2)

        covariance = AnalyticalCovariance(covariance_fn=cov_fn)

        # Should not raise.
        estimate(
            model=_dummy_model,
            measure=measure,
            covariance=covariance,
            weighting=Identity(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(),
            theta_init=_P1(a=0.5),
        )
