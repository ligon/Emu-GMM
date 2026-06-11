"""Function-object-keyed memoisation for derived wrappers (#139).

The optimiser layer memoises derived callables -- an optimistix
``fn(y, args)`` adapter, a ``jax.jit`` wrapper, a traced solve core --
per user residual function, because JAX / optimistix trace caches key on
the *identity* of the callable they are handed: rebuilding the wrapper
on every call would miss the trace cache every time even when the
underlying residual is unchanged.

The v1 implementation keyed module-level dicts on ``id(fn)`` and
registered a ``weakref.finalize`` eviction. Audit L1 (issue #139) found
the eviction was dead code: every cached value strongly references the
keyed function (closure cell / jit retention), so the function could
never be collected while the module-rooted dict held the value -- the
caches were append-only, one immortal entry per keyed function.

A ``weakref.WeakKeyDictionary`` does NOT fix this. It holds *values*
strongly, and the value -> key strong reference keeps the key reachable
from the (module-rooted) dict forever: the same immortality with extra
steps. Verified empirically before this redesign.

Design chosen instead: attach the cache to the keyed function object
itself, as ``fn.<attr> = (fn, table)``. The resulting reference cycle
(``fn.__dict__ -> table -> wrapper -> closure cell -> fn``) is
unreachable once the caller drops ``fn``, so the cycle GC collects the
whole group -- the wrapper's lifetime is exactly the keyed function's
lifetime. Object identity does the keying, so the id()-recycling hazard
of the old design (a stale hit when a dead function's ``id`` is reused
by a new object) is structurally impossible: there is no global table
in which a recycled id could alias. The stored ``(fn, table)`` owner
tuple guards the one aliasing route attributes have: a bound method
delegates attribute *reads* to ``__func__``, so a wrapper cached on a
plain function would otherwise be visible through any bound method built
from it; the ``owner is fn`` check turns that into a miss.

Callables that refuse ``setattr`` (bound methods, slotted / frozen
callables) fall back to a small bounded module-level LRU keyed on
``id(fn)`` that stores the keyed object STRONGLY next to its table:
while an entry lives, the strong reference pins the object's ``id`` so
it cannot be recycled, and on eviction the entry is removed entirely, so
a recycled id can only miss, never alias. The bound (default 64 keyed
objects) is the documented cost of the fallback; the attribute path --
which every callable the framework itself constructs takes (plain
functions, closures, lambdas, ``functools.partial``) -- is unbounded in
entry count but object-lifetime scoped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_FALLBACK_MAXSIZE = 64


class FunctionKeyedCache:
    """Memoise values derived from a callable on the callable itself.

    ``get_or_build(fn, build, key=...)`` returns the value memoised for
    ``(fn, key)``, building it via ``build()`` on a miss. ``key`` is an
    optional *hashable* secondary key for callers that derive several
    values from one function (e.g. one traced solve per (solver, spec)
    pair); the default single-value use passes ``key=None``.

    Storage is an attribute on ``fn`` (see the module docstring for the
    lifetime / id-recycling rationale), with a bounded strong-reference
    LRU fallback for callables that refuse ``setattr``.
    """

    def __init__(self, attr: str, fallback_maxsize: int = _FALLBACK_MAXSIZE) -> None:
        self._attr = attr
        self._fallback_maxsize = fallback_maxsize
        # id(fn) -> (fn, table). The strong reference to ``fn`` pins its
        # id while the entry lives (no recycling); eviction drops the
        # whole entry (no stale aliasing).
        self._fallback: dict[int, tuple[Any, dict[Any, Any]]] = {}

    def get_or_build(self, fn: Any, build: Callable[[], Any], key: Any = None) -> Any:
        table = self._table(fn)
        if key in table:
            return table[key]
        value = build()
        table[key] = value
        return value

    # -- storage ------------------------------------------------------

    def _table(self, fn: Any) -> dict[Any, Any]:
        entry = getattr(fn, self._attr, None)
        # ``owner is fn`` forecloses bound-method attribute delegation:
        # ``m.__getattr__`` falls through to ``m.__func__``, so an entry
        # cached on the plain function is readable through every bound
        # method built from it -- a wrong-closure hit if trusted.
        if isinstance(entry, tuple) and len(entry) == 2 and entry[0] is fn:
            return entry[1]
        table: dict[Any, Any] = {}
        try:
            setattr(fn, self._attr, (fn, table))
        except (AttributeError, TypeError):
            return self._fallback_table(fn)
        return table

    def _fallback_table(self, fn: Any) -> dict[Any, Any]:
        fid = id(fn)
        entry = self._fallback.get(fid)
        if entry is not None and entry[0] is fn:
            # Belt and braces: with ``fn`` held strongly its id is
            # pinned, so a present entry with matching id *must* be the
            # same object; the identity check documents and enforces it.
            self._fallback.pop(fid)  # LRU: refresh recency
            self._fallback[fid] = entry
            return entry[1]
        table: dict[Any, Any] = {}
        self._fallback.pop(fid, None)
        self._fallback[fid] = (fn, table)
        while len(self._fallback) > self._fallback_maxsize:
            # Evict least recently used (dicts preserve insertion order;
            # hits re-insert). Eviction only costs a rebuild/retrace.
            self._fallback.pop(next(iter(self._fallback)))
        return table
