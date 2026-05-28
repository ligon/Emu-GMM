"""Tests for emu_gmm._internal.axes."""

from __future__ import annotations

import haliax as ha
import pytest

from emu_gmm._internal import axes


class TestFactories:
    def test_params_axis(self):
        a = axes.params_axis(5)
        assert isinstance(a, ha.Axis)
        assert a.name == "parameters"
        assert a.size == 5

    def test_params_dual_axis(self):
        a = axes.params_dual_axis(5)
        assert a.name == "parameters_dual"
        assert a.size == 5

    def test_moments_axis(self):
        a = axes.moments_axis(3)
        assert a.name == "moments"
        assert a.size == 3

    def test_moments_dual_axis(self):
        a = axes.moments_dual_axis(3)
        assert a.name == "moments_dual"
        assert a.size == 3

    def test_obs_axis(self):
        a = axes.obs_axis(100)
        assert a.name == "observations"
        assert a.size == 100

    def test_name_constants_match_factories(self):
        # Factories should produce axes whose names match the exported constants.
        assert axes.params_axis(1).name == axes.PARAMS_NAME
        assert axes.params_dual_axis(1).name == axes.PARAMS_DUAL_NAME
        assert axes.moments_axis(1).name == axes.MOMENTS_NAME
        assert axes.moments_dual_axis(1).name == axes.MOMENTS_DUAL_NAME
        assert axes.obs_axis(1).name == axes.OBS_NAME


class TestDual:
    def test_dual_params(self):
        primary = axes.params_axis(7)
        result = axes.dual(primary)
        assert result.name == "parameters_dual"
        assert result.size == 7

    def test_dual_moments(self):
        primary = axes.moments_axis(4)
        result = axes.dual(primary)
        assert result.name == "moments_dual"
        assert result.size == 4

    def test_dual_obs(self):
        # Even axes that don't have a "natural" dual still work --- dual()
        # is a pure name+size transform.
        primary = axes.obs_axis(50)
        result = axes.dual(primary)
        assert result.name == "observations_dual"
        assert result.size == 50

    def test_dual_raises_on_already_dual(self):
        already = axes.params_dual_axis(5)
        with pytest.raises(ValueError, match="already a dual axis"):
            axes.dual(already)

    def test_dual_matches_factory(self):
        # dual(params_axis(K)) should equal params_dual_axis(K).
        K = 3
        assert axes.dual(axes.params_axis(K)) == axes.params_dual_axis(K)
        assert axes.dual(axes.moments_axis(K)) == axes.moments_dual_axis(K)
