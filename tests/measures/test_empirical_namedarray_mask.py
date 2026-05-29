"""Mask-dispatch regression tests for :meth:`EmpiricalMeasure.from_pandas`.

``from_pandas`` accepts the mask in any of three forms --- a pandas
DataFrame, a haliax NamedArray, or a plain (jax / numpy) array. A prior
revision of the mask-shape probe iterated ``int(s) for s in mask.shape``
unconditionally, which silently broke for :class:`haliax.NamedArray`:
that type returns ``shape`` as a ``{name: size}`` dict, so the probe
tried to coerce axis names to ``int`` and raised ``ValueError``. The
fix dispatches on ``isinstance(mask, ha.NamedArray)`` before the generic
``hasattr(mask, "shape")`` branch. This module pins all three dispatch
paths.
"""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import numpy as np
import pandas as pd
from emu_gmm.measures.empirical import EmpiricalMeasure


def _df() -> pd.DataFrame:
    """Reference ``(N=4, D=2)`` frame, no NaN (Fix 2 conflict avoided)."""
    return pd.DataFrame(
        {
            "r0": [1.0, 2.0, 3.0, 4.0],
            "r1": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _expected_mask_np() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )


class TestFromPandasMaskDispatch:
    """All three mask-input wrappers route to the same ``(N, M)`` output."""

    def test_dataframe_mask(self):
        df = _df()
        mask_df = pd.DataFrame(_expected_mask_np(), columns=["m0", "m1"])
        meas = EmpiricalMeasure.from_pandas(df, mask=mask_df)
        assert meas.mask.shape == (4, 2)
        np.testing.assert_allclose(np.asarray(meas.mask), _expected_mask_np())

    def test_named_array_mask(self):
        """Regression: ``ha.NamedArray.shape`` is a dict, not a tuple."""
        df = _df()
        n_axis = ha.Axis("obs", 4)
        m_axis = ha.Axis("moment", 2)
        mask_named = ha.named(jnp.asarray(_expected_mask_np()), (n_axis, m_axis))
        # Prior to the fix this raised
        # ``ValueError: invalid literal for int() with base 10: 'obs'``.
        meas = EmpiricalMeasure.from_pandas(df, mask=mask_named)
        assert meas.mask.shape == (4, 2)
        np.testing.assert_allclose(np.asarray(meas.mask), _expected_mask_np())

    def test_plain_array_mask(self):
        df = _df()
        mask_plain = jnp.asarray(_expected_mask_np())
        meas = EmpiricalMeasure.from_pandas(df, mask=mask_plain)
        assert meas.mask.shape == (4, 2)
        np.testing.assert_allclose(np.asarray(meas.mask), _expected_mask_np())

    def test_plain_numpy_mask(self):
        """Plain numpy arrays go through the same branch as jax arrays."""
        df = _df()
        meas = EmpiricalMeasure.from_pandas(df, mask=_expected_mask_np())
        assert meas.mask.shape == (4, 2)
        np.testing.assert_allclose(np.asarray(meas.mask), _expected_mask_np())
