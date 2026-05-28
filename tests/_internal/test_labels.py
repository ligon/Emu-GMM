"""Tests for emu_gmm._internal.labels."""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import labels


class TestLabelContext:
    def test_default_empty(self):
        lc = labels.LabelContext()
        assert lc.param_names == ()
        assert lc.moment_names == ()
        assert lc.variable_names == ()
        assert lc.obs_name is None

    def test_hashable(self):
        # frozen dataclass with tuple fields must be hashable.
        lc = labels.LabelContext(param_names=("beta", "gamma"))
        d = {lc: 1}
        assert d[lc] == 1

    def test_with_moment_names(self):
        lc = labels.LabelContext(param_names=("beta",))
        lc2 = lc.with_moment_names(("euler", "excess_return"))
        assert lc.moment_names == ()  # original unchanged
        assert lc2.param_names == ("beta",)
        assert lc2.moment_names == ("euler", "excess_return")


class TestNormaliseX:
    def test_pandas_dataframe(self):
        df = pd.DataFrame(
            {"c_t": [1.0, 1.1], "c_tp1": [1.05, 1.08], "r": [0.03, 0.04]}
        )
        df.index.name = "hh_id"
        arr, cols, idx_name = labels.normalise_x(df)
        assert arr.shape == (2, 3)
        assert cols == ("c_t", "c_tp1", "r")
        assert idx_name == "hh_id"

    def test_pandas_dataframe_unnamed_index(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        arr, cols, idx_name = labels.normalise_x(df)
        assert cols == ("a", "b")
        assert idx_name is None

    def test_plain_array(self):
        arr_in = jnp.arange(6).reshape(3, 2).astype(jnp.float32)
        arr, cols, idx_name = labels.normalise_x(arr_in)
        assert arr.shape == (3, 2)
        assert cols == ("v_0", "v_1")
        assert idx_name is None

    def test_numpy_array(self):
        arr_in = np.arange(6).reshape(2, 3).astype(np.float32)
        arr, cols, _ = labels.normalise_x(arr_in)
        assert arr.shape == (2, 3)
        assert cols == ("v_0", "v_1", "v_2")

    def test_rejects_1d_plain(self):
        with pytest.raises(ValueError, match="2-D array"):
            labels.normalise_x(jnp.arange(5))

    def test_haliax_named(self):
        Obs = ha.Axis("obs", 3)
        Vars = ha.Axis("vars", 2)
        named = ha.named(
            jnp.arange(6).reshape(3, 2).astype(jnp.float32), (Obs, Vars)
        )
        arr, cols, idx_name = labels.normalise_x(named)
        assert arr.shape == (3, 2)
        assert idx_name == "obs"
        # Haliax axes don't carry per-coordinate names, so we get positional.
        assert cols == ("v_0", "v_1")


class TestNormaliseWeights:
    def test_default_ones(self):
        w = labels.normalise_weights(None, n=5)
        assert jnp.allclose(w, jnp.ones(5))

    def test_pandas_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        w = labels.normalise_weights(s, n=3)
        assert jnp.allclose(w, jnp.array([1.0, 2.0, 3.0]))

    def test_plain_array(self):
        w = labels.normalise_weights(jnp.array([0.5, 1.5, 2.5]), n=3)
        assert jnp.allclose(w, jnp.array([0.5, 1.5, 2.5]))

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="expected length 5"):
            labels.normalise_weights(jnp.ones(3), n=5)


class TestNormaliseMask:
    def test_default_ones(self):
        mask = labels.normalise_mask(None, n=4, m=2)
        assert mask.shape == (4, 2)
        assert jnp.allclose(mask, jnp.ones((4, 2)))

    def test_plain_array(self):
        m_in = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mask = labels.normalise_mask(m_in, n=3, m=2)
        assert jnp.allclose(mask, m_in)

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 0]})
        mask = labels.normalise_mask(df, n=3, m=2)
        assert mask.shape == (3, 2)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"expected shape \(3, 2\)"):
            labels.normalise_mask(jnp.ones((3, 3)), n=3, m=2)


class TestResolveMomentNames:
    def test_positional_fallback(self):
        names = labels.resolve_moment_names(
            model_return=None, kwarg_names=None, m=3
        )
        assert names == ("m_0", "m_1", "m_2")

    def test_kwarg_wins_over_positional(self):
        names = labels.resolve_moment_names(
            model_return=None, kwarg_names=("euler", "excess"), m=2
        )
        assert names == ("euler", "excess")

    def test_kwarg_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length 1, expected 2"):
            labels.resolve_moment_names(
                model_return=None, kwarg_names=("euler",), m=2
            )

    def test_model_return_wins_over_kwarg(self):
        Moments = axes_mod.moments_axis(2)
        named_return = ha.named(jnp.zeros(2), (Moments,))
        names = labels.resolve_moment_names(
            model_return=named_return,
            kwarg_names=("ignored",),
            m=2,
        )
        # Model-return path produces positional names (haliax axes don't
        # carry per-coord names), but it takes precedence over the kwarg.
        assert names == ("m_0", "m_1")

    def test_model_return_size_mismatch_raises(self):
        Moments = axes_mod.moments_axis(3)
        named_return = ha.named(jnp.zeros(3), (Moments,))
        with pytest.raises(ValueError, match="size 3, expected 2"):
            labels.resolve_moment_names(
                model_return=named_return, kwarg_names=None, m=2
            )

    def test_model_return_without_moments_axis_falls_through(self):
        # If model_return is a NamedArray but without a "moments" axis,
        # the kwarg/positional logic should still apply.
        Other = ha.Axis("other", 2)
        named_return = ha.named(jnp.zeros(2), (Other,))
        names = labels.resolve_moment_names(
            model_return=named_return,
            kwarg_names=("a", "b"),
            m=2,
        )
        assert names == ("a", "b")


class TestLabelMatrix:
    def test_round_trip(self):
        Params = axes_mod.params_axis(2)
        ParamsDual = axes_mod.params_dual_axis(2)
        arr = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        named = labels.label_matrix(arr, Params, ParamsDual)
        assert isinstance(named, ha.NamedArray)
        assert named.axes == (Params, ParamsDual)
        assert jnp.allclose(named.array, arr)

    def test_rejects_shape_mismatch(self):
        Params = axes_mod.params_axis(2)
        ParamsDual = axes_mod.params_dual_axis(3)  # wrong size
        arr = jnp.zeros((2, 3))
        # arr.shape == (2, 3); axes sizes are (2, 3); should succeed.
        labels.label_matrix(arr, Params, ParamsDual)

        # Now mismatch:
        arr_bad = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="does not match axes"):
            labels.label_matrix(arr_bad, Params, ParamsDual)

    def test_rejects_1d_input(self):
        Params = axes_mod.params_axis(3)
        ParamsDual = axes_mod.params_dual_axis(3)
        with pytest.raises(ValueError, match="2-D array"):
            labels.label_matrix(jnp.zeros(3), Params, ParamsDual)


class TestLabelVector:
    def test_round_trip(self):
        Moments = axes_mod.moments_axis(3)
        arr = jnp.array([1.0, 2.0, 3.0])
        named = labels.label_vector(arr, Moments)
        assert isinstance(named, ha.NamedArray)
        assert named.axes == (Moments,)
        assert jnp.allclose(named.array, arr)

    def test_rejects_2d_input(self):
        Moments = axes_mod.moments_axis(3)
        with pytest.raises(ValueError, match="1-D array"):
            labels.label_vector(jnp.zeros((3, 3)), Moments)

    def test_rejects_length_mismatch(self):
        Moments = axes_mod.moments_axis(3)
        with pytest.raises(ValueError, match="does not match axis"):
            labels.label_vector(jnp.zeros(5), Moments)
