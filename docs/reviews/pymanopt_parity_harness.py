#!/usr/bin/env python3
"""Pymanopt-parity verification harness for v2 PSDFixedRank.

This is a standalone verifier (NOT a pytest test) that exercises
``emu_gmm.manifolds.PSDFixedRank`` (when it exists) against the reference
``pymanopt.manifolds.PSDFixedRank`` implementation across many random
inputs and a grid of (m, k) shapes.

Usage
-----

    poetry run python docs/reviews/pymanopt_parity_harness.py

Behaviour
---------

* If ``emu_gmm.manifolds`` is not importable (current state on ``main``),
  only the pymanopt side runs, and emu-gmm operations are reported as
  ``skipped``.  This lets us stage the harness early and verify the
  pymanopt-side plumbing is correct before Chunk A lands.
* Once Chunk A (issue #3 / PR #28) is merged, the deferred import
  succeeds and the harness compares the two implementations.

Operations checked
------------------

For each ``(seed, shape)`` cell:

* ``projection``                — ambient -> tangent projection at X.
* ``retraction``                — retraction of a tangent vector at X.
* ``riemannian_gradient``       — pymanopt's
  ``euclidean_to_riemannian_gradient(X, egrad)``; emu_gmm is expected to
  expose this as ``riemannian_gradient(X, egrad)``.
* ``distance``                  — geodesic / chord distance between two
  random points on the manifold.

Comparison tolerance: ``rtol=1e-9`` in float64.  This is strict; the two
implementations should agree to numerical roundoff (both are pure
linear-algebra formulae over the same X, V, egrad inputs).
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Reference implementation (pymanopt) ---------------------------------
from pymanopt.manifolds import PSDFixedRank as PymanoptPSDFixedRank

# --- Implementation under test (emu_gmm) ---------------------------------
#
# Deferred so that on ``main`` (pre-Chunk-A) the harness still runs and
# exercises the pymanopt side.  When Chunk A lands, the import succeeds
# and the parity comparisons activate automatically.

EMU_AVAILABLE: bool
EMU_IMPORT_ERROR: str | None
try:
    from emu_gmm.manifolds import PSDFixedRank as EmuPSDFixedRank

    EMU_AVAILABLE = True
    EMU_IMPORT_ERROR = None
except Exception as exc:
    EmuPSDFixedRank = None  # type: ignore[assignment,misc]
    EMU_AVAILABLE = False
    EMU_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

N_SEEDS = 100
SHAPES: list[tuple[int, int]] = [
    (3, 1),
    (3, 2),
    (5, 1),
    (5, 2),
    (5, 3),
    (8, 2),
    (8, 3),
    (10, 3),
]
RTOL = 1e-9
ATOL = 0.0
OPERATIONS = ("projection", "retraction", "riemannian_gradient", "distance")


# -------------------------------------------------------------------------
# Tally
# -------------------------------------------------------------------------


@dataclass
class Tally:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errored: int = 0

    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errored


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _as_numpy(x: Any) -> np.ndarray:
    """Convert JAX / pymanopt / numpy outputs to a plain float64 ndarray."""
    arr = np.asarray(x)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr


def _diff_report(
    seed: int,
    shape: tuple[int, int],
    op: str,
    ref: Any,
    cand: Any,
) -> str:
    ref_a = _as_numpy(ref)
    cand_a = _as_numpy(cand)
    if ref_a.shape != cand_a.shape:
        return (
            f"  shape mismatch: pymanopt={ref_a.shape} vs emu_gmm={cand_a.shape}\n"
            f"  pymanopt={ref_a}\n  emu_gmm={cand_a}"
        )
    abs_diff = np.abs(ref_a - cand_a)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(ref_a), 1.0)
    rel = float(np.max(abs_diff / denom))
    return (
        f"  seed={seed} shape={shape} op={op}\n"
        f"  max_abs_diff = {max_abs:.3e}\n"
        f"  max_rel_diff = {rel:.3e}\n"
        f"  pymanopt (first 8 entries) = {ref_a.ravel()[:8]}\n"
        f"  emu_gmm  (first 8 entries) = {cand_a.ravel()[:8]}"
    )


def _allclose(ref: Any, cand: Any) -> bool:
    return np.allclose(_as_numpy(ref), _as_numpy(cand), rtol=RTOL, atol=ATOL)


# -------------------------------------------------------------------------
# Per-operation comparisons
# -------------------------------------------------------------------------


def _call_emu(
    emu_manifold: Any,
    op: str,
    X: np.ndarray,
    ambient: np.ndarray,
    V: np.ndarray,
    Y: np.ndarray,
) -> Any:
    """Invoke the emu_gmm operation, accepting plausible method names."""
    fn: Callable | None
    if op == "projection":
        fn = emu_manifold.projection
        return fn(X, ambient)
    if op == "retraction":
        fn = emu_manifold.retraction
        return fn(X, V)
    if op == "riemannian_gradient":
        # Prefer the emu_gmm-style ``riemannian_gradient``; fall back to
        # the pymanopt name if the v2 implementation chose to mirror it.
        fn = getattr(
            emu_manifold,
            "riemannian_gradient",
            getattr(emu_manifold, "euclidean_to_riemannian_gradient", None),
        )
        if fn is None:
            raise AttributeError("no riemannian_gradient method on emu_gmm manifold")
        return fn(X, ambient)
    if op == "distance":
        fn = getattr(emu_manifold, "distance", getattr(emu_manifold, "dist", None))
        if fn is None:
            raise AttributeError("no distance/dist method on emu_gmm manifold")
        return fn(X, Y)
    raise ValueError(f"unknown op {op!r}")


def _call_pymanopt(
    pym_manifold: Any,
    op: str,
    X: np.ndarray,
    ambient: np.ndarray,
    V: np.ndarray,
    Y: np.ndarray,
) -> Any:
    if op == "projection":
        return pym_manifold.projection(X, ambient)
    if op == "retraction":
        return pym_manifold.retraction(X, V)
    if op == "riemannian_gradient":
        return pym_manifold.euclidean_to_riemannian_gradient(X, ambient)
    if op == "distance":
        return pym_manifold.dist(X, Y)
    raise ValueError(f"unknown op {op!r}")


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------


def run() -> int:
    print("=" * 72)
    print("Pymanopt-parity verification harness for v2 PSDFixedRank")
    print("=" * 72)
    print(f"N_SEEDS  = {N_SEEDS}")
    print(f"SHAPES   = {SHAPES}")
    print(f"RTOL     = {RTOL}")
    print(f"emu_gmm.manifolds importable: {EMU_AVAILABLE}")
    if not EMU_AVAILABLE:
        print(f"  (deferred-import error: {EMU_IMPORT_ERROR})")
        print(
            "  emu_gmm comparisons will be reported as 'skipped'.\n"
            "  This is the expected state pre-Chunk-A."
        )
    print()

    tallies: dict[str, Tally] = defaultdict(Tally)
    first_failures: list[str] = []

    for shape in SHAPES:
        m, k = shape
        pym = PymanoptPSDFixedRank(m, k)
        emu = EmuPSDFixedRank(m, k) if EMU_AVAILABLE else None

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 9973 + m * 131 + k)
            # Seed pymanopt's global numpy RNG (random_point uses it).
            np.random.seed(int(rng.integers(0, 2**31 - 1)))

            X = np.asarray(pym.random_point(), dtype=np.float64)
            Y = np.asarray(pym.random_point(), dtype=np.float64)
            V = np.asarray(pym.random_tangent_vector(X), dtype=np.float64)
            ambient = rng.standard_normal(size=X.shape).astype(np.float64)

            for op in OPERATIONS:
                # 1. pymanopt reference (always computed; this validates
                #    our pymanopt-side plumbing).
                try:
                    ref = _call_pymanopt(pym, op, X, ambient, V, Y)
                except Exception as exc:
                    tallies[op].errored += 1
                    if len(first_failures) < 10:
                        first_failures.append(
                            f"[pymanopt-ref ERROR] op={op} shape={shape} seed={seed}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue

                # 2. emu_gmm candidate (skipped if not available).
                if not EMU_AVAILABLE:
                    tallies[op].skipped += 1
                    continue
                try:
                    cand = _call_emu(emu, op, X, ambient, V, Y)
                except AttributeError:
                    # emu_gmm doesn't expose this op yet — record as
                    # skipped (partial coverage).
                    tallies[op].skipped += 1
                    continue
                except Exception as exc:
                    tallies[op].errored += 1
                    if len(first_failures) < 10:
                        first_failures.append(
                            f"[emu_gmm ERROR] op={op} shape={shape} seed={seed}: "
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}"
                        )
                    continue

                if _allclose(ref, cand):
                    tallies[op].passed += 1
                else:
                    tallies[op].failed += 1
                    if len(first_failures) < 10:
                        first_failures.append(
                            "[MISMATCH]\n" + _diff_report(seed, shape, op, ref, cand)
                        )

    # -- Summary ----------------------------------------------------------
    print("Summary")
    print("-" * 72)
    header = f"{'operation':<24} {'passed':>8} {'failed':>8} {'skipped':>8} {'errored':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    overall = Tally()
    for op in OPERATIONS:
        t = tallies[op]
        overall.passed += t.passed
        overall.failed += t.failed
        overall.skipped += t.skipped
        overall.errored += t.errored
        print(
            f"{op:<24} {t.passed:>8d} {t.failed:>8d} {t.skipped:>8d} "
            f"{t.errored:>8d} {t.total():>8d}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<24} {overall.passed:>8d} {overall.failed:>8d} "
        f"{overall.skipped:>8d} {overall.errored:>8d} {overall.total():>8d}"
    )
    print()

    if first_failures:
        print("First failures / errors (up to 10):")
        print("-" * 72)
        for msg in first_failures:
            print(msg)
            print()

    # Exit code policy:
    #   0 -> all comparisons passed (or only emu-side skips when emu absent)
    #   1 -> at least one mismatch or error on the pymanopt or emu side
    if overall.failed or overall.errored:
        return 1
    if not EMU_AVAILABLE:
        # Pymanopt-only run: report success of the staging.
        print(
            "Pymanopt-only run complete.  emu_gmm comparisons skipped; the "
            "harness is ready to activate when Chunk A lands."
        )
    return 0


if __name__ == "__main__":
    sys.exit(run())
