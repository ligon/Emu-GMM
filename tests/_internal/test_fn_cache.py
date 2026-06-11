"""Tests for emu_gmm._internal.fn_cache (#139).

The contract under test: a value derived from a function and memoised
per function object (a) is returned identically on a second lookup while
the function lives, (b) dies WITH the function -- the old id()-keyed
module dicts were append-only because the cached value strongly
referenced the keyed function, so the registered finalizer could never
fire -- and (c) can never stale-hit through id() recycling or
bound-method attribute delegation.
"""

from __future__ import annotations

import gc
import weakref

from emu_gmm._internal.fn_cache import FunctionKeyedCache


def _make_closure():
    payload = object()

    def fn(x):
        return (x, payload)

    return fn


def _wrap(fn):
    """A derived wrapper that strongly references ``fn`` -- the shape that
    made the old design append-only (closure cell / jit retention)."""

    def wrapper(x):
        return fn(x)

    return wrapper


class TestMemoisation:
    def test_second_lookup_returns_same_object(self):
        cache = FunctionKeyedCache("_emu_test_memo")
        fn = _make_closure()
        v1 = cache.get_or_build(fn, lambda: _wrap(fn))
        v2 = cache.get_or_build(fn, lambda: _wrap(fn))
        assert v1 is v2

    def test_secondary_keys_are_independent(self):
        cache = FunctionKeyedCache("_emu_test_keys")
        fn = _make_closure()
        va = cache.get_or_build(fn, lambda: "a", key=("solver-1",))
        vb = cache.get_or_build(fn, lambda: "b", key=("solver-2",))
        assert va == "a" and vb == "b"
        assert cache.get_or_build(fn, lambda: "MISS", key=("solver-1",)) == "a"

    def test_distinct_functions_do_not_share(self):
        cache = FunctionKeyedCache("_emu_test_distinct")
        f1, f2 = _make_closure(), _make_closure()
        v1 = cache.get_or_build(f1, lambda: _wrap(f1))
        v2 = cache.get_or_build(f2, lambda: _wrap(f2))
        assert v1 is not v2


class TestEviction:
    def test_entry_dies_with_function(self):
        """The #139 regression: under the old id()-keyed design the cached
        wrapper kept the function immortal (finalize never fired)."""
        cache = FunctionKeyedCache("_emu_test_evict")
        fn = _make_closure()
        wrapper = cache.get_or_build(fn, lambda: _wrap(fn))
        ref = weakref.ref(fn)
        del fn, wrapper
        gc.collect()
        assert ref() is None, "keyed function immortal: cache entry not evicted"

    def test_fallback_is_bounded(self):
        """Non-setattr-able callables (bound methods) go to the bounded LRU."""

        class Obj:
            def psi(self, x):
                return x

        cache = FunctionKeyedCache("_emu_test_bound", fallback_maxsize=4)
        for _ in range(10):
            m = Obj().psi  # bound method: setattr raises AttributeError
            cache.get_or_build(m, lambda m=m: _wrap(m))
        assert len(cache._fallback) <= 4

    def test_fallback_lru_refresh_on_hit(self):
        class Obj:
            def psi(self, x):
                return x

        cache = FunctionKeyedCache("_emu_test_lru", fallback_maxsize=2)
        m1, m2 = Obj().psi, Obj().psi
        v1 = cache.get_or_build(m1, lambda: _wrap(m1))
        cache.get_or_build(m2, lambda: _wrap(m2))
        # Hit m1 (refreshes recency), then insert a third: m2 evicted.
        assert cache.get_or_build(m1, lambda: "MISS") is v1
        m3 = Obj().psi
        cache.get_or_build(m3, lambda: _wrap(m3))
        assert cache.get_or_build(m1, lambda: "MISS") is v1  # survived
        assert cache.get_or_build(m2, lambda: "rebuilt") == "rebuilt"  # evicted


class TestAliasingForeclosed:
    def test_bound_method_does_not_inherit_plain_function_entry(self):
        """Attribute READS on a bound method delegate to ``__func__``; a
        wrapper cached on the plain function must not leak to the method
        (it would call the wrong closure -- no ``self``)."""

        def f(self, x):
            return x

        cache = FunctionKeyedCache("_emu_test_alias")
        v_plain = cache.get_or_build(f, lambda: "plain-wrapper")
        assert v_plain == "plain-wrapper"

        class A:
            pass

        A.g = f
        m = A().g
        assert m.__func__ is f
        # The delegation is real: the plain function's entry is visible...
        assert m._emu_test_alias[0] is f
        # ...but the owner guard turns it into a miss, not a stale hit.
        v_method = cache.get_or_build(m, lambda: "method-wrapper")
        assert v_method == "method-wrapper"

    def test_fallback_id_pinned_while_entry_lives(self):
        """While a fallback entry lives, the keyed object is held strongly,
        so its id() cannot be recycled by a new object -- the recycling
        stale-hit needs a dead key under a live entry, which cannot occur."""

        class Obj:
            def psi(self, x):
                return x

        cache = FunctionKeyedCache("_emu_test_pin", fallback_maxsize=4)
        m = Obj().psi
        cache.get_or_build(m, lambda: _wrap(m))
        (entry,) = cache._fallback.values()
        assert entry[0] is m  # strong reference pins id(m)
