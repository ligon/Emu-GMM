"""Shared pytest fixtures for the emu-gmm suite.

The module fixture bounds JAX's process-global compilation cache. The optional
trace observer separates lazy advisory compilation from solve/inference reuse.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture
def lm_advisory_trace_counts(monkeypatch):
    """Observe model traces excluding the separately cached curvature probe.

    The first stationary fit may occur after initial factory warm-up. Its
    advisory is compiled then, rather than paying Hessian compilation at every
    nonstationary cost stop. Tests still bound this probe to one trace and
    require solve/inference model traces to remain frozen.
    """
    import importlib

    module = importlib.import_module("emu_gmm.manifolds.riemannian_lm")
    original = module._horizontal_curvature_eigenvalues

    def observe(counter):
        totals = [0] * len(counter())
        traces = 0

        def curvature(*args, **kwargs):
            nonlocal traces
            before = counter()
            result = original(*args, **kwargs)
            after = counter()
            for i in range(len(totals)):
                totals[i] += after[i] - before[i]
            traces += 1
            return result

        monkeypatch.setattr(module, "_horizontal_curvature_eigenvalues", curvature)

        def snapshot():
            return (
                tuple(
                    value - total
                    for value, total in zip(counter(), totals, strict=True)
                ),
                traces,
            )

        return snapshot

    return observe


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches_between_modules():
    """Clear JAX's global compilation cache at each test-module boundary.

    Every test compiles fresh jitted functions, and JAX's process-global
    caches grow monotonically (~14 MB per fresh-closure ``estimate`` call;
    see the "64-core JIT-mmap hazard" note in ``CLAUDE.md``). On a many-core
    host a long single-process ``pytest`` run accumulates enough cached
    executables / mmap'd executable pages to ``SIGABRT`` / ``SIGSEGV``
    partway through with no summary line -- the suite never completes, which
    is how a real, deterministic failure (the design-spec real-data J, #151)
    sat hidden for days.

    Clearing the caches once per *module* (on teardown, after that module's
    tests have run) bounds the growth so the whole suite completes in a single
    process. The clear is between modules, not between tests, so it adds only
    one cold-compile per module rather than per test.

    Deliberately module-scoped, not function-scoped:

    * the observed crash is cross-module accumulation -- every individual test
      module passes in isolation -- so a per-module reset is sufficient;
    * the within-test cache assertions (the #124 / #139 retrace-count tests,
      which check that a second ``estimate`` call does *not* retrace) exercise
      caching *inside* a single test; this teardown runs strictly after a
      module's tests, so it cannot perturb them.
    """
    yield
    jax.clear_caches()
