"""Host-configuration helpers for the many-core CPU JIT-mmap hazard (#115).

Running many ``estimate`` compilations on a many-core CPU host can fail
with ``LLVM compilation error: Cannot allocate memory`` even with tens of
GB free: XLA's JIT thread pool sizes itself to the affinity-visible CPU
count and ``mmap``\\ s executable pages per worker, so a many-core host --
or several uncapped JAX processes at once -- oversubscribes the address
space and the kernel's overcommit heuristic rejects the allocation. This
bit a Seasonality Euler Monte Carlo (a 200-rep x 4-tau run OOM'd
mid-flight before the cause was understood).

The remedy is to bound JAX to a single host CPU *device* (and, optionally,
cap the BLAS thread pools) **before the XLA backend initialises**:

    JAX_NUM_CPU_DEVICES=1 XLA_FLAGS=--xla_force_host_platform_device_count=1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

Timing matters and is unforgiving. ``XLA_FLAGS`` / ``JAX_NUM_CPU_DEVICES``
are read once when the backend initialises, and the BLAS caps are read by
their libraries even earlier (at import). **Importing ``emu_gmm`` already
initialises the backend** (a transitive ``haliax`` import touches a device),
so the only fully reliable point is to set these *before* launching Python
(or before the first ``import emu_gmm``). This module therefore exposes the
remedy two ways:

* :func:`recommended_env` -- the remedy as a plain ``dict`` to export
  (the reliable path);
* :func:`configure` -- an in-process setter that **raises** rather than
  silently no-op when the backend is already initialised.

and emits a one-time :func:`warning <maybe_warn_cpu_oversubscription>` at
the first ``estimate`` on an at-risk host, pointing at the fix.

For *concurrent* JAX processes, prefer disjoint CPU affinity
(``taskset -c 0-31`` / ``32-63``): each process then sizes its pools to its
own cores at full speed, which a single shared one-device cap cannot.
"""

from __future__ import annotations

import os
import warnings

# The XLA flag token that caps the number of host CPU devices.
_XLA_DEVICE_FLAG = "xla_force_host_platform_device_count"
# BLAS / OpenMP thread-pool caps.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# Warn at first ``estimate`` only above this affinity-visible core count: the
# OOM is a many-core hazard (the reproduction is 64-core), and a low threshold
# would be noise on laptops / small CI runners. Tunable for tests.
_WARN_CORE_THRESHOLD = 16
# Set this (to anything truthy) to silence the one-time warning.
SILENCE_ENV_VAR = "EMU_GMM_NO_CPU_WARNING"

# Module-level "warn at most once per process" latch.
_warned = False


def recommended_env(
    host_devices: int = 1, *, thread_caps: bool = True
) -> dict[str, str]:
    """Return the environment variables that avoid the JIT-mmap OOM (#115).

    A plain ``dict[str, str]`` for the caller to export *before launching
    Python* (the only fully reliable point -- see the module docstring):

    >>> import emu_gmm
    >>> emu_gmm.recommended_env()                      # doctest: +SKIP
    {'JAX_NUM_CPU_DEVICES': '1',
     'XLA_FLAGS': '--xla_force_host_platform_device_count=1',
     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}

    Parameters
    ----------
    host_devices
        Number of host CPU devices XLA should expose (``>= 1``). ``1`` is
        the statistics-workload default; the framework is not data-parallel,
        so extra CPU devices only multiply the per-worker mmap pressure.
    thread_caps
        When ``True`` (default) also cap the BLAS / OpenMP thread pools to a
        single thread. Set ``False`` to leave them alone (e.g. a workload
        that also does multi-threaded host-side NumPy / SciPy).
    """
    if host_devices < 1:
        raise ValueError(
            f"recommended_env: host_devices must be >= 1, got {host_devices}"
        )
    env = {
        "JAX_NUM_CPU_DEVICES": str(host_devices),
        "XLA_FLAGS": f"--{_XLA_DEVICE_FLAG}={host_devices}",
    }
    if thread_caps:
        env.update(dict.fromkeys(_THREAD_VARS, "1"))
    return env


def _backend_initialized() -> bool:
    """Best-effort: has the JAX backend already initialised? (Never triggers it.)"""
    try:
        from jax._src import xla_bridge

        return bool(xla_bridge.backends_are_initialized())
    except Exception:
        # Unknown JAX internal -> assume not initialised (configure() proceeds
        # rather than spuriously raising on a version it can't introspect).
        return False


def _merge_xla_flags(host_devices: int) -> str:
    """``XLA_FLAGS`` with the device-count flag set, preserving other flags.

    Any pre-existing ``--xla_force_host_platform_device_count=...`` token is
    replaced; every other flag is kept verbatim.
    """
    existing = os.environ.get("XLA_FLAGS", "")
    kept = [tok for tok in existing.split() if _XLA_DEVICE_FLAG not in tok]
    return " ".join([*kept, f"--{_XLA_DEVICE_FLAG}={host_devices}"]).strip()


def configure(
    host_devices: int = 1,
    *,
    thread_caps: bool = True,
    force: bool = False,
) -> dict[str, str]:
    """Apply :func:`recommended_env` into ``os.environ`` in-process (#115).

    Merges the XLA device-count flag into any existing ``XLA_FLAGS`` (other
    flags preserved) and sets the device / thread-cap variables. Returns the
    applied mapping.

    **Must run before the JAX backend initialises.** The device-count flag is
    read once at backend init, so a later call cannot take effect; rather than
    silently no-op, :func:`configure` raises :class:`RuntimeError` when the
    backend is already initialised (pass ``force=True`` to set the variables
    anyway, e.g. so a *child* process you launch inherits them).

    Because ``import emu_gmm`` itself initialises the backend, the reliable
    path for the current process is to export :func:`recommended_env` *before*
    launching Python; :func:`configure` is for entry points that set JAX up
    lazily, and for stamping the environment of subprocesses.
    """
    if not force and _backend_initialized():
        raise RuntimeError(
            "emu_gmm.configure() was called after the JAX backend "
            "initialised, so the host-device cap can no longer take effect "
            "(XLA reads it once at backend init). Importing emu_gmm "
            "initialises the backend, so set the variables from "
            "emu_gmm.recommended_env() BEFORE launching Python (or before the "
            "first `import emu_gmm`). Pass force=True to set them anyway "
            "(e.g. for child processes)."
        )
    env = recommended_env(host_devices, thread_caps=thread_caps)
    env["XLA_FLAGS"] = _merge_xla_flags(host_devices)
    os.environ.update(env)
    return env


def _device_cap_in_effect() -> bool:
    """True if a single-host-device cap is already configured (env)."""
    if _XLA_DEVICE_FLAG in os.environ.get("XLA_FLAGS", ""):
        return True
    if os.environ.get("JAX_NUM_CPU_DEVICES"):
        return True
    return False


def _affinity_core_count() -> int:
    """Affinity-visible CPU count -- what XLA sizes its pools to. Tunable for tests."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def maybe_warn_cpu_oversubscription() -> None:
    """Emit a one-time warning on an at-risk many-core host (#115).

    Fires at most once per process, and only when *all* of: not silenced via
    :data:`SILENCE_ENV_VAR`; no device cap already in effect
    (:func:`recommended_env` / ``taskset`` not applied); and the
    affinity-visible core count exceeds :data:`_WARN_CORE_THRESHOLD`. Called
    from ``estimate`` / ``build_estimator`` at first use.
    """
    global _warned
    if _warned:
        return
    _warned = True  # evaluate the condition at most once per process
    if os.environ.get(SILENCE_ENV_VAR):
        return
    if _device_cap_in_effect():
        return
    n_cores = _affinity_core_count()
    if n_cores <= _WARN_CORE_THRESHOLD:
        return
    warnings.warn(
        f"emu_gmm: running on a {n_cores}-core host with no JAX device cap. "
        "XLA's JIT thread pool mmaps executable pages per core, so repeated "
        "estimate() compilations -- or several uncapped JAX processes at once "
        "-- can hit 'LLVM compilation error: Cannot allocate memory' even with "
        "free RAM (issue #115). Fix: export emu_gmm.recommended_env() before "
        "launching Python, or pin CPU affinity (e.g. taskset -c 0-31). "
        f"Silence with {SILENCE_ENV_VAR}=1.",
        UserWarning,
        stacklevel=3,
    )


__all__ = ["recommended_env", "configure", "maybe_warn_cpu_oversubscription"]
