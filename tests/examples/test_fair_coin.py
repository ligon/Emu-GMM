"""Smoke test for the fair-coin Bernoulli example.

Verifies that ``examples/fair_coin.py`` continues to recover
``p_true = 0.5`` within Monte Carlo tolerance. The example is the
smallest interesting demo of the Emu-GMM surface; a recovery
regression here means a regression in the public ``estimate(...)``
entry point on a problem so simple it should never fail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Make the ``examples/`` directory importable so the test calls into
# the same module that ``poetry run python examples/fair_coin.py``
# executes -- guarding the example itself against bit-rot rather than
# duplicating its setup here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import fair_coin  # noqa: E402  (sys.path manipulation above is intentional)


@pytest.fixture(scope="module")
def result():
    """Run the example once per module and share the EstimationResult."""
    return fair_coin.run_fair_coin()


def test_recovers_p_within_three_se(result):
    """``p_hat`` lands within 3 asymptotic SEs of ``p_true = 0.5``."""
    p_hat = float(result.theta_hat.p)
    # Asymptotic SE for the sample mean of Bernoulli(p) is
    # sqrt(p (1 - p) / N). At p = 0.5, N = 1000 this is ~0.01581.
    se = math.sqrt(fair_coin.P_TRUE * (1.0 - fair_coin.P_TRUE) / fair_coin.N_DATA)
    assert abs(p_hat - fair_coin.P_TRUE) < 3.0 * se


def test_converged(result):
    """The optimiser certifies convergence on this trivial problem."""
    assert result.converged


def test_just_identified_J_dof_is_zero(result):
    """``M = K = 1`` makes the J-test degenerate; ``J_dof`` must be 0."""
    assert result.J_dof == 0
    assert jnp.isfinite(result.J_stat)


def test_coef_table_has_expected_shape(result):
    """The coefficient table is a 1-row DataFrame indexed by ``"p"``."""
    table = result.coef_table
    assert list(table.index) == ["p"]
    assert set(table.columns) == {"estimate", "std_error", "t_stat", "p_value"}
    assert math.isfinite(float(table.loc["p", "std_error"]))
