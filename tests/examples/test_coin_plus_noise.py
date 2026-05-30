"""Recovery smoke test for the coin-plus-noise demo.

Loads ``examples/coin_plus_noise.py`` as a module, runs its clean
synthetic and empirical entry points, and asserts the estimated
parameters lie within Monte-Carlo tolerance of the truth. Guards against
bit-rot in the demo without re-running the whole printing/main block.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Locate ``examples/coin_plus_noise.py`` two levels above this file
# (tests/examples/ -> repo root -> examples/).
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_COIN_PATH = _EXAMPLES_DIR / "coin_plus_noise.py"


def _load_coin_plus_noise_module():
    """Import ``examples/coin_plus_noise.py`` as a standalone module.

    Registers the module in :data:`sys.modules` *before* executing it so
    that :func:`jax_dataclasses.pytree_dataclass` can resolve the
    module's namespace when the dataclass decorator runs.
    """
    module_name = "coin_plus_noise_demo"
    spec = importlib.util.spec_from_file_location(module_name, str(_COIN_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coin_module():
    return _load_coin_plus_noise_module()


@pytest.mark.slow
class TestCoinPlusNoiseSynthetic:
    """Recovery via SyntheticMeasure + SyntheticCovariance."""

    def test_recovers_p(self, coin_module):
        result = coin_module.run_synthetic(n_sim=2000, seed=0)
        assert result.converged
        p_hat = float(result.theta_hat.p)
        assert 0.45 <= p_hat <= 0.55, f"p_hat = {p_hat} out of [0.45, 0.55]"

    def test_recovers_sigma2(self, coin_module):
        result = coin_module.run_synthetic(n_sim=2000, seed=0)
        assert result.converged
        sigma2_hat = float(result.theta_hat.sigma2)
        assert 0.9 <= sigma2_hat <= 1.1, f"sigma2_hat = {sigma2_hat} out of [0.9, 1.1]"

    def test_just_identified_J_dof_zero(self, coin_module):
        result = coin_module.run_synthetic(n_sim=2000, seed=0)
        # Just-identified: M = K = 2.
        assert result.J_dof == 0


@pytest.mark.slow
class TestCoinPlusNoiseEmpirical:
    """Recovery via EmpiricalMeasure + IIDCovariance."""

    def test_recovers_p(self, coin_module):
        result = coin_module.run_empirical(n=2000, seed=0)
        assert result.converged
        p_hat = float(result.theta_hat.p)
        assert 0.45 <= p_hat <= 0.55, f"p_hat = {p_hat} out of [0.45, 0.55]"

    def test_recovers_sigma2(self, coin_module):
        result = coin_module.run_empirical(n=2000, seed=0)
        assert result.converged
        sigma2_hat = float(result.theta_hat.sigma2)
        assert 0.9 <= sigma2_hat <= 1.1, f"sigma2_hat = {sigma2_hat} out of [0.9, 1.1]"

    def test_moment_labels_passed_through(self, coin_module):
        result = coin_module.run_empirical(n=2000, seed=0)
        # The example passes moment_names=("mean", "second_moment").
        assert result.labels.moment_names == ("mean", "second_moment")

    def test_param_labels(self, coin_module):
        result = coin_module.run_empirical(n=2000, seed=0)
        assert result.labels.param_names == ("p", "sigma2")
