"""Regression test for ``examples/hierarchical.py``.

Drives the example's clean entry-point function (``examples.hierarchical.run``)
so we exercise the same DGP + ``estimate(...)`` + ``cluster_bootstrap(...)``
pipeline that the demo script uses, without re-running the demo's print-to-
stdout main.

The test is intentionally not on the example's module-level main: a demo
should be free to print, time itself, or otherwise add noise; the regression
target is recovery within a Monte Carlo-sized tolerance band.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import jax.numpy as jnp
import pytest


def _load_example():
    """Load ``examples/hierarchical.py`` by path.

    ``examples/`` is a top-level directory in the repo but is not a
    Python package (no ``__init__.py``), so ``from examples.hierarchical
    import ...`` cannot resolve it. Loading by spec keeps the test free
    of any top-level package-ification of ``examples/`` and matches the
    pattern used by the runnable demo (the demo also injects the repo
    root onto ``sys.path`` at import time).

    The module is registered in :data:`sys.modules` before exec because
    ``@jdc.pytree_dataclass`` resolves the defining class's
    ``__module__`` via ``sys.modules.get(cls.__module__)``; without the
    registration the lookup returns ``None`` and dataclass construction
    raises ``AttributeError`` on the missing ``__dict__``.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    module_path = os.path.abspath(
        os.path.join(here, "..", "..", "examples", "hierarchical.py")
    )
    name = "_examples_hierarchical"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_example()
SIGMA2_MU_TRUE = _mod.SIGMA2_MU_TRUE
SIGMA2_E_TRUE = _mod.SIGMA2_E_TRUE
HierarchicalParams = _mod.HierarchicalParams
run = _mod.run


class TestHierarchicalRecovery:
    """Recovery of the two variance components on the canonical seed.

    ``S = 30`` clusters is small enough that the between-school variance
    estimator has SE ~ 1.5 (i.e. roughly 35 % of the truth), so a strict
    10 % tolerance on a single seed is not always achievable. The default
    seed in ``examples.hierarchical`` is one such that recovery lands
    inside a usefully tight band; the tolerances below correspond to
    roughly one bootstrap standard error.

    The point-estimate tests are fast (one LM solve); the bootstrap
    sanity-check carries the ``slow`` marker because each replicate
    re-runs ``estimate``.
    """

    def test_recovers_sigma2_mu(self):
        result, _ = run(n_boot=0)
        sigma2_mu_hat = float(result.theta_hat.sigma2_mu)
        # Roughly within 1 bootstrap SE of truth on the default seed.
        assert sigma2_mu_hat == pytest.approx(SIGMA2_MU_TRUE, abs=1.5)

    def test_recovers_sigma2_e(self):
        result, _ = run(n_boot=0)
        sigma2_e_hat = float(result.theta_hat.sigma2_e)
        # Within-school variance is much better identified (SE ~ 0.5);
        # 1.5 absolute is roughly 3 SE.
        assert sigma2_e_hat == pytest.approx(SIGMA2_E_TRUE, abs=1.5)

    def test_params_axis_carries_two_components(self):
        result, _ = run(n_boot=0)
        assert result.labels.param_names == ("sigma2_mu", "sigma2_e")

    def test_theta_hat_is_hierarchical_params(self):
        result, _ = run(n_boot=0)
        assert isinstance(result.theta_hat, HierarchicalParams)

    def test_converged(self):
        result, _ = run(n_boot=0)
        assert result.converged

    @pytest.mark.slow
    def test_run_returns_result_and_bootstrap(self):
        result, boot = run(n_boot=20)
        assert result.converged
        assert boot is not None
        # 20 replicates is plenty for a smoke check; all should clear LM.
        assert int(boot.convergence.sum()) >= 19

    @pytest.mark.slow
    def test_bootstrap_se_recovers_in_band(self):
        """Cluster bootstrap SE on sigma2_mu reflects the genuine wide CI."""
        _, boot = run(n_boot=80)
        assert boot is not None
        accepted = jnp.asarray(boot.theta_boot.array)[boot.convergence]
        se = accepted.std(axis=0, ddof=1)
        # sigma2_mu SE should be O(1); a value <0.3 would suggest a bug.
        # An upper guard catches a degenerate run too.
        assert 0.3 < float(se[0]) < 5.0
        # sigma2_e SE is much smaller (SK observations on the y^2 moment).
        assert 0.05 < float(se[1]) < 2.0
