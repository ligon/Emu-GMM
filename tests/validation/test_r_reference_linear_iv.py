r"""Cross-validation of emu-gmm against R's ``gmm`` / ``AER`` on a linear IV fit.

The external-reference acceptance test (``docs/validation/r-reference-crosschecks.org``):
a frozen over-identified IV dataset (``tests/data/gmm_linear_iv.csv``) is fit in
R (``scripts/gmm_reference/reference.R`` -> ``gmm_linear_iv_reference.json``) and
here in emu-gmm, and the two are compared. Like ``test_estimator_realdata.py``
this consumes *frozen* reference numbers, so CI needs no R toolchain; regenerate
with ``gen_data.py`` then ``reference.R`` (R installs from the Ubuntu ``r-cran-*``
debs -- CRAN is blocked by the agent egress policy; see the doc).

Three claims, in decreasing tightness:

1. **Machine-precision correctness** (``test_matched_weight_reproduces_2sls``):
   with the 2SLS weight :math:`(Z'Z/n)^{-1}` held identical, emu-gmm's linear
   GMM solver reproduces ``AER::ivreg`` to ~1e-6. This isolates the estimator
   math from any covariance-estimator convention.
2. **Commitment 9 -- the J scale** (``test_J_is_on_the_chi2_scale``): emu folds
   the per-coordinate :math:`N_j` into :math:`V_X`, so its ``objective_value``
   is on the SAME scale as gmm's :math:`n\,\bar g' S^{-1}\bar g` -- an O(1)-few
   value referenced to :math:`\chi^2_2`, NOT off by a factor of ``n``.
3. **The documented weighting band** (``test_estimates_match_within_band``):
   the emu CUE / iterated estimates agree with gmm's to ~1% (coef) / ~5% (SE) ---
   a small, systematic difference in how each package builds the CUE weight
   :math:`V(\theta)` (NOT moment-covariance centering: an ``IIDCovariance(
   centered=True)`` toggle leaves the emu CUE point byte-identical; see
   ``docs/validation/r-reference-crosschecks.org``). A tighter match needs a
   matched weight (claim 1); a WIDER gap would flag a real regression.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    Fixed,
    IIDCovariance,
    IteratedWeighting,
    estimate,
    optimistix_lm,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CSV = _DATA_DIR / "gmm_linear_iv.csv"
_REF = _DATA_DIR / "gmm_linear_iv_reference.json"

# Acceptance-level external cross-check (mirrors test_estimator_realdata.py): run
# in the full-suite validation leg, not the fast per-push quick-check gate.
pytestmark = pytest.mark.slow


@jdc.pytree_dataclass
class _P:
    b0: float
    b1: float


def _iv_resid(row, th):
    # Instruments (1, z1, z2, z3) times the structural residual -> M=4 moments.
    Zr = jnp.array([1.0, row[2], row[3], row[4]])
    return Zr * (row[0] - th.b0 - th.b1 * row[1])


@pytest.fixture(scope="module")
def data_and_ref():
    ref = json.loads(_REF.read_text())
    X = np.loadtxt(_CSV, delimiter=",", skiprows=1)
    return X, ref


def _fit(X, weighting):
    meas = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=4)
    r = estimate(
        _iv_resid,
        meas,
        covariance=IIDCovariance(),
        weighting=weighting,
        optimizer=optimistix_lm(),
        theta_init=_P(b0=0.0, b1=0.0),
    )
    return r, r.asymptotic()


def test_frozen_data_matches_reference_provenance(data_and_ref):
    """The committed CSV is exactly the data the R reference was computed on."""
    _, ref = data_and_ref
    digest = hashlib.sha256(_CSV.read_bytes()).hexdigest()
    assert digest == ref["provenance"]["data_sha256"], (
        "tests/data/gmm_linear_iv.csv has drifted from the R reference JSON; "
        "regenerate BOTH: python scripts/gmm_reference/gen_data.py && "
        "Rscript scripts/gmm_reference/reference.R"
    )


def test_matched_weight_reproduces_2sls_to_machine_precision(data_and_ref):
    """Claim 1: matched (2SLS) weight -> emu reproduces AER::ivreg exactly."""
    X, ref = data_and_ref
    n = X.shape[0]
    Z = np.column_stack([np.ones(n), X[:, 2], X[:, 3], X[:, 4]])
    V0 = (Z.T @ Z) / n  # 2SLS weight is (Z'Z/n)^{-1} == Fixed(V0)^{-1}
    r, _ = _fit(X, Fixed(V0=jnp.asarray(V0)))
    tsls = ref["twoStageLeastSquares"]["coef"]
    np.testing.assert_allclose(float(r.theta_hat.b0), tsls["b0"], rtol=1e-5)
    np.testing.assert_allclose(float(r.theta_hat.b1), tsls["b1"], rtol=1e-5)


def test_J_is_on_the_chi2_scale_no_explicit_N(data_and_ref):
    """Claim 2 (commitment 9): J is chi^2_2-scaled, not off by a factor of n."""
    X, ref = data_and_ref
    r, law = _fit(X, IteratedWeighting())
    g = ref["gmm_iterative"]
    assert r.n_overid == g["J_df"] == 2
    # A factor-of-n bug would put J near 0.007 or near 1900; it is neither.
    assert 1.0 < float(r.objective_value) < 8.0
    np.testing.assert_allclose(float(r.objective_value), g["J"], rtol=0.06)
    np.testing.assert_allclose(float(law.J_pvalue), g["J_pvalue"], atol=0.02)


@pytest.mark.parametrize(
    "weighting,key",
    [
        (ContinuouslyUpdated(), "gmm_cue"),
        (IteratedWeighting(), "gmm_iterative"),
    ],
)
def test_estimates_match_within_convention_band(data_and_ref, weighting, key):
    """Claim 3: emu vs gmm agree to the documented ~1% coef / ~5% SE band."""
    X, ref = data_and_ref
    r, law = _fit(X, weighting)
    se = np.asarray(law.se())
    g = ref[key]
    np.testing.assert_allclose(float(r.theta_hat.b0), g["coef"]["b0"], rtol=0.015)
    np.testing.assert_allclose(float(r.theta_hat.b1), g["coef"]["b1"], rtol=0.015)
    np.testing.assert_allclose(se[0], g["se"]["b0"], rtol=0.06)
    np.testing.assert_allclose(se[1], g["se"]["b1"], rtol=0.06)
