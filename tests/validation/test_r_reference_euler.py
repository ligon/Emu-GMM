r"""Cross-validation of the bundled Euler example against R's ``gmm``.

Tier 2 of ``docs/validation/r-reference-crosschecks.org``: a Hansen-Singleton
style multi-asset consumption-Euler / SDF estimation. The point is that this
drives the *shipped* model --- it imports
:func:`emu_gmm.examples.euler.euler_residual` verbatim (the same residual three
acceptance tests use) --- and compares its fit to R's ``gmm`` on the *identical*
frozen data (``tests/data/gmm_euler.csv``, sha256-guarded). CI needs no R;
regenerate with ``gen_euler_data.py`` then ``reference_euler.R``.

Moment: :math:`\psi_j = \beta (c'/c)^{-\gamma}(1+r_j) - 1` for the three assets,
so ``M=3``, ``K=2`` (``beta, gamma``), one over-identifying restriction. On this
nonlinear fit emu-gmm and gmm agree to ~5--6 significant figures --- tighter than
the linear-IV case, because at the over-identified optimum the moment mean is
~0, so the centered-vs-uncentered covariance-estimator gap nearly vanishes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    IIDCovariance,
    IteratedWeighting,
    estimate,
    optimistix_lm,
)
from emu_gmm.examples.euler import EulerParams, euler_residual

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CSV = _DATA_DIR / "gmm_euler.csv"
_REF = _DATA_DIR / "gmm_euler_reference.json"


@pytest.fixture(scope="module")
def data_and_ref():
    ref = json.loads(_REF.read_text())
    X = np.loadtxt(_CSV, delimiter=",", skiprows=1)
    return X, ref


def _fit(X, weighting):
    meas = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    r = estimate(
        euler_residual,
        meas,
        covariance=IIDCovariance(),
        weighting=weighting,
        optimizer=optimistix_lm(),
        theta_init=EulerParams(beta=0.95, gamma=1.5),
    )
    return r, r.asymptotic()


def test_frozen_data_matches_reference_provenance(data_and_ref):
    """The committed Euler CSV is exactly the data the R reference used."""
    _, ref = data_and_ref
    digest = hashlib.sha256(_CSV.read_bytes()).hexdigest()
    assert digest == ref["provenance"]["data_sha256"], (
        "tests/data/gmm_euler.csv drifted from the R reference; regenerate BOTH: "
        "python scripts/gmm_reference/gen_euler_data.py && "
        "Rscript scripts/gmm_reference/reference_euler.R"
    )


def test_J_is_on_the_chi2_scale_no_explicit_N(data_and_ref):
    """Commitment 9 on a NONLINEAR criterion: J ~ chi^2_1, matching gmm."""
    X, ref = data_and_ref
    r, law = _fit(X, IteratedWeighting())
    g = ref["gmm_iterative"]
    assert r.n_overid == g["J_df"] == 1
    # A factor-of-n bug (n=2000) would be ~2000x off; J is O(1).
    assert 0.2 < float(r.objective_value) < 5.0
    np.testing.assert_allclose(float(r.objective_value), g["J"], rtol=0.02)
    np.testing.assert_allclose(float(law.J_pvalue), g["J_pvalue"], atol=0.02)


@pytest.mark.parametrize(
    "weighting,key",
    [
        (ContinuouslyUpdated(), "gmm_cue"),
        (IteratedWeighting(), "gmm_iterative"),
    ],
)
def test_matches_gmm_to_few_significant_figures(data_and_ref, weighting, key):
    """emu-gmm's bundled Euler fit reproduces R's gmm to ~5-6 sig figs."""
    X, ref = data_and_ref
    r, law = _fit(X, weighting)
    se = np.asarray(law.se())
    g = ref[key]
    np.testing.assert_allclose(float(r.theta_hat.beta), g["coef"]["beta"], rtol=2e-3)
    np.testing.assert_allclose(float(r.theta_hat.gamma), g["coef"]["gamma"], rtol=1e-2)
    np.testing.assert_allclose(se[0], g["se"]["beta"], rtol=2e-2)
    np.testing.assert_allclose(se[1], g["se"]["gamma"], rtol=2e-2)
