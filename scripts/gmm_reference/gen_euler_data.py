#!/usr/bin/env python
"""Freeze a dataset from the bundled multi-asset Euler DGP for the R cross-check.

Tier 2 of ``docs/validation/r-reference-crosschecks.org``: a Hansen-Singleton
style consumption-Euler / SDF estimation, using the *exact* DGP and moment of
``src/emu_gmm/examples/euler.py`` (the bundled example that three acceptance
tests already drive) so the cross-check validates the shipped nonlinear model,
not a re-implementation.

Deterministic (jax PRNG, fixed key) so the committed CSV is read identically by
the R reference (``reference_euler.R``) and the acceptance test
(``tests/validation/test_r_reference_euler.py``). Columns ``c_t, c_next, r0, r1,
r2``; the per-observation moment is
``psi_j = beta * (c_next/c_t)^(-gamma) * (1 + r_j) - 1`` for the three assets,
so M=3, K=2 (beta, gamma), one over-identifying restriction. Regenerate::

    python scripts/gmm_reference/gen_euler_data.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import numpy as np

from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_sampler_factory,
)

SEED = 20260704
N = 2000

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "gmm_euler.csv"


def main() -> None:
    sampler = euler_sampler_factory(N)
    X = np.asarray(
        sampler(jax.random.PRNGKey(SEED), EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE))
    )
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        DATA, X, delimiter=",", header="c_t,c_next,r0,r1,r2", comments="", fmt="%.17g"
    )
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print(f"wrote {DATA}  ({X.shape[0]} rows x {X.shape[1]} cols)")
    print(f"truth: beta={BETA_TRUE}, gamma={GAMMA_TRUE}")
    print(f"sha256 = {digest}")


if __name__ == "__main__":
    main()
