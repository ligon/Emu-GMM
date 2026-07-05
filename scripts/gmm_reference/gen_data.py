#!/usr/bin/env python
"""Generate the frozen linear-IV dataset for the R cross-check (docs/validation).

Deterministic (``numpy`` PCG64, fixed seed) so the committed CSV is the single
source of truth read *identically* by the R reference script
(``reference.R``) and the acceptance test
(``tests/validation/test_r_reference_linear_iv.py``). Regenerate with::

    python scripts/gmm_reference/gen_data.py

then re-run ``reference.R`` to refresh the frozen reference numbers. The DGP is
a textbook over-identified IV:

    xe = Z @ pi + v                     (one endogenous regressor)
    u  = rho * v + noise                (=> Cov(xe, u) != 0, endogeneity)
    y  = b0 + b1 * xe + u,   b0=1, b1=2

with three excluded instruments ``z1, z2, z3`` (plus the constant), so the
moment vector ``g_i = (1, z1, z2, z3)_i * (y_i - b0 - b1 xe_i)`` has M=4 and
K=2 -> two over-identifying restrictions (J ~ chi^2_2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SEED = 20260704
N = 500
B0_TRUE, B1_TRUE = 1.0, 2.0
PI = np.array([0.9, 0.7, 0.5])  # first-stage instrument strength
RHO = 1.3  # endogeneity: loading of v in u

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "gmm_linear_iv.csv"


def generate() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(N, 3))
    v = rng.normal(size=N)
    xe = Z @ PI + v
    u = RHO * v + rng.normal(size=N) * 0.6
    y = B0_TRUE + B1_TRUE * xe + u
    # Columns consumed by BOTH sides: [y, xe, z1, z2, z3].
    return np.column_stack([y, xe, Z[:, 0], Z[:, 1], Z[:, 2]])


def main() -> None:
    X = generate()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    # High precision so the CSV round-trip is lossless to float64.
    header = "y,xe,z1,z2,z3"
    np.savetxt(DATA, X, delimiter=",", header=header, comments="", fmt="%.17g")
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print(f"wrote {DATA}  ({X.shape[0]} rows x {X.shape[1]} cols)")
    print(f"sha256 = {digest}")


if __name__ == "__main__":
    main()
