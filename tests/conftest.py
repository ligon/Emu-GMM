"""Shared pytest fixtures for the emu-gmm suite.

The single fixture here bounds JAX's process-global compilation cache so the
full suite can run in one process. See :func:`_clear_jax_caches_between_modules`.
"""

from __future__ import annotations

import jax
import pytest


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
