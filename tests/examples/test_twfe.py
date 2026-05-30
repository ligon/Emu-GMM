"""Smoke test for ``examples/twfe.py``.

A small Monte-Carlo style check that the bundled two-way fixed effects
example recovers the structural slope ``c`` within a generous
cluster-robust tolerance and that the J-test of the over-identifying
moment is finite. The test imports the example's ``run_twfe`` entry
point rather than executing the runnable demo, so it does not depend on
the ``__main__`` block.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# The bundled examples live under ``<repo>/examples`` rather than a
# package; make sure that directory is on ``sys.path`` so the test can
# import ``twfe`` without an ad-hoc ``conftest``. Same pattern as a
# user would use to run ``poetry run python examples/twfe.py`` directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = str(_REPO_ROOT / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

twfe = importlib.import_module("twfe")


@pytest.mark.slow
def test_twfe_recovers_c_within_2_cluster_robust_se():
    """``c_hat`` should land within 2 cluster-robust SE of the truth.

    Asymptotically this holds with probability ~95% under correct
    specification; with a single seed and modest N=20, T=15 the test is
    a smoke check rather than a coverage probability calibration.
    """
    result = twfe.run_twfe(seed=0)
    c_hat = float(result.theta_hat.c)
    se = float(result.coef_table["std_error"].iloc[0])
    assert se > 0.0 and jnp.isfinite(jnp.asarray(se))
    assert abs(c_hat - twfe.C_TRUE) < 2.0 * se, (
        f"c_hat={c_hat:.4f} is {abs(c_hat - twfe.C_TRUE):.4f} from truth "
        f"({twfe.C_TRUE}); exceeds 2 * cluster-robust SE = {2 * se:.4f}"
    )


@pytest.mark.slow
def test_twfe_J_stat_finite_and_modest():
    """The over-identifying J-stat should be finite and chi^2_1-ish."""
    result = twfe.run_twfe(seed=0)
    assert result.J_dof == 1  # M=2, K=1
    assert jnp.isfinite(result.J_stat)
    # Correctly specified DGP; J ~ chi^2_1 in the limit. Allow generous
    # finite-sample slack so the test is not flaky on a single seed.
    assert float(result.J_stat) < 30.0


@pytest.mark.slow
def test_twfe_converged():
    """The estimator should report a converged outer-loop status.

    A non-converged status would surface in
    :attr:`EstimationResult.converged` as ``False`` and would mean the
    iterated weighting loop hit its iteration budget without stabilising
    ``c``.
    """
    result = twfe.run_twfe(seed=0)
    assert bool(result.converged)


@pytest.mark.slow
def test_twfe_provenance_echoed():
    """The result should echo the configured measure / covariance / weighting."""
    from emu_gmm import (
        ClusteredCovariance,
        EmpiricalMeasure,
        IteratedWeighting,
    )

    result = twfe.run_twfe(seed=0)
    assert isinstance(result.measure, EmpiricalMeasure)
    assert isinstance(result.covariance, ClusteredCovariance)
    assert isinstance(result.weighting, IteratedWeighting)
