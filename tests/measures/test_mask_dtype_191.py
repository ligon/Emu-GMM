"""Convenience masks share float64 without altering the explicit constructor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from emu_gmm import EmpiricalMeasure
from emu_gmm._internal.labels import normalise_mask


def _measures():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    frame = pd.DataFrame(x)
    mask = np.ones((3, 2), dtype=np.float32)
    return [
        EmpiricalMeasure.from_arrays(x),
        EmpiricalMeasure.from_arrays(x, mask=mask),
        EmpiricalMeasure.from_pandas(frame),
        EmpiricalMeasure.from_pandas(frame, mask=mask),
        EmpiricalMeasure.from_pandas(frame, nan_aware=True),
        EmpiricalMeasure.from_nan_aware(x),
        EmpiricalMeasure.from_nan_aware(x, M=2),
    ]


@pytest.mark.parametrize("mask", [None, np.ones((3, 2), dtype=np.float32)])
def test_mask_normalizer_uses_float64(mask):
    normalized = normalise_mask(mask, 3, 2)
    assert normalized.dtype == jnp.float64
    np.testing.assert_array_equal(normalized, np.ones((3, 2)))


@pytest.mark.parametrize("labelled", [False, True])
def test_convenience_constructors_share_mask_dtype_and_trace_with_same_labels(labelled):
    traces = []

    @jax.jit
    def mean(measure):
        traces.append(True)
        return measure.expectation(lambda x, theta: x - theta, jnp.zeros(2))

    # Labelled and plain inputs have distinct static metadata (#132).
    # Mask normalization must not retrace within either metadata group.
    measures = [m for m in _measures() if (m.labels is not None) == labelled]
    assert measures
    for measure in measures:
        assert measure.mask.dtype == jnp.float64
        np.testing.assert_array_equal(measure.mask, np.ones((3, 2)))
        np.testing.assert_array_equal(mean(measure), [3.0, 4.0])
    assert len(traces) == 1


def test_inferred_missingness_values_are_unchanged():
    x = np.array([[1.0, np.nan], [3.0, 4.0], [np.inf, 6.0]])
    expected = np.isfinite(x).astype(float)
    for measure in [
        EmpiricalMeasure.from_pandas(pd.DataFrame(x), nan_aware=True),
        EmpiricalMeasure.from_nan_aware(x),
    ]:
        assert measure.mask.dtype == jnp.float64
        np.testing.assert_array_equal(measure.mask, expected)
    row_mask = EmpiricalMeasure.from_nan_aware(x, M=1)
    assert row_mask.mask.dtype == jnp.float64
    np.testing.assert_array_equal(row_mask.mask, [[0.0], [1.0], [0.0]])


def test_low_level_constructor_retains_explicit_float32_mask():
    measure = EmpiricalMeasure(
        x=jnp.ones((3, 2)),
        mask=jnp.ones((3, 2), dtype=jnp.float32),
        weights=jnp.ones(3),
    )
    assert measure.mask.dtype == jnp.float32
