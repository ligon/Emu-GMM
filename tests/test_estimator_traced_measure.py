"""PR A acceptance gates for the #124 traced-measure kernel path.

Four contracts:

(1) Parity: the traced-argument kernel and the legacy closure path are
    the same math (one shared ``_residual_core``); results agree to
    float-identical tolerance (NOT bitwise -- the jit boundaries differ,
    per the spike's measurement of ~4e-16 disagreement).
(2) No-retrace: fresh same-structure measures ride ONE trace. Pinned by
    counting model invocations -- psi is executed by Python only while
    tracing, so the counter freezes once the kernel is compiled.
(3) Traced ``done``: the optimistix backend now reports the REAL
    convergence flag (#78 extension), not the status-string fallback.
(4) Fallback: a third-party optimiser written to the v1 two-argument
    protocol routes down the legacy closure path and still works.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import IIDCovariance, build_estimator, estimate
from emu_gmm.examples.euler import EulerParams, euler_data, euler_residual
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm, scipy_lm
from emu_gmm.types import OptimizerInfo

N = 800


def _measure(seed: int) -> EmpiricalMeasure:
    x = euler_data(seed=seed, n=N)
    return EmpiricalMeasure(x=x, mask=jnp.ones((N, 3)), weights=jnp.ones(N))


def _theta0() -> EulerParams:
    return EulerParams(beta=0.9, gamma=1.0)


class _CountingModel:
    """psi wrapper counting Python-level executions (== trace events)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, theta):
        self.calls += 1
        return euler_residual(x, theta)


class _TwoArgOptimizer:
    """A third-party optimiser on the v1 protocol: no ``args`` channel."""

    def __init__(self):
        self.inner = optimistix_lm()
        self.invocations = 0

    def __call__(self, residual_fn, theta_init):
        self.invocations += 1
        return self.inner(residual_fn, theta_init)


class TestParity:
    def test_traced_path_matches_legacy_closure_path(self):
        m = _measure(seed=3)
        # Traced path: default optimistix (args-capable).
        run_traced = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        res_traced = run_traced(_theta0(), m)

        # Legacy path: force it with a 2-arg third-party optimiser
        # wrapping the SAME solver.
        run_legacy = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            optimizer=_TwoArgOptimizer(),
            parameters=_theta0(),
        )
        res_legacy = run_legacy(_theta0(), m)

        assert bool(res_traced.converged) and bool(res_legacy.converged)
        np.testing.assert_allclose(
            float(res_traced.theta_hat.beta),
            float(res_legacy.theta_hat.beta),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(res_traced.theta_hat.gamma),
            float(res_legacy.theta_hat.gamma),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(res_traced.J_stat), float(res_legacy.J_stat), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(res_traced.Sigma_theta.array),
            np.asarray(res_legacy.Sigma_theta.array),
            rtol=1e-9,
        )

    def test_traced_path_matches_bare_estimate(self):
        m = _measure(seed=4)
        res_bare = estimate(
            euler_residual,
            m,
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        run = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        res_traced = run(_theta0(), m)
        np.testing.assert_allclose(
            float(res_traced.theta_hat.beta),
            float(res_bare.theta_hat.beta),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(res_traced.J_stat), float(res_bare.J_stat), rtol=1e-10
        )


class TestNoRetrace:
    def test_fresh_same_structure_measures_share_one_trace(self):
        """The #124 headline: psi is re-executed (traced) for the first
        measure only; reps 2..5 with fresh same-structure measures run
        the compiled kernel without touching Python."""
        counting = _CountingModel()
        run = build_estimator(
            counting,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        res = run(_theta0(), _measure(seed=1))
        assert bool(res.converged)
        calls_after_first = counting.calls
        assert calls_after_first > 0  # tracing happened

        for seed in (2, 3, 4, 5):
            res = run(_theta0(), _measure(seed=seed))
            assert bool(res.converged)
        assert counting.calls == calls_after_first, (
            f"psi re-traced on fresh same-structure measures: "
            f"{counting.calls} != {calls_after_first}"
        )

    def test_different_shape_retraces_once_then_caches(self):
        counting = _CountingModel()
        run = build_estimator(
            counting,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        run(_theta0(), _measure(seed=1))
        calls_n = counting.calls

        def small_measure(seed):
            n2 = 400
            x = euler_data(seed=seed, n=n2)
            return EmpiricalMeasure(x=x, mask=jnp.ones((n2, 3)), weights=jnp.ones(n2))

        run(_theta0(), small_measure(7))
        calls_after_new_shape = counting.calls
        assert calls_after_new_shape > calls_n  # one retrace for the new shape
        run(_theta0(), small_measure(8))
        assert counting.calls == calls_after_new_shape  # cached thereafter


class TestTracedDone:
    def test_optimistix_info_carries_done_flag(self):
        res = estimate(
            euler_residual,
            _measure(seed=9),
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        done = res.diagnostics.optimizer_info.done
        assert done is not None
        assert bool(done) is True
        assert bool(res.converged)

    def test_done_false_when_max_steps_exhausted(self):
        res = estimate(
            euler_residual,
            _measure(seed=10),
            covariance=IIDCovariance(),
            optimizer=optimistix_lm(max_steps=1),
            parameters=EulerParams(beta=0.5, gamma=5.0),
        )
        assert bool(res.diagnostics.optimizer_info.done) is False
        assert not bool(res.converged)


class TestFallbacks:
    def test_two_arg_optimizer_routes_legacy_and_works(self):
        opt = _TwoArgOptimizer()
        run = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            optimizer=opt,
            parameters=_theta0(),
        )
        res = run(_theta0(), _measure(seed=2))
        assert opt.invocations == 1
        assert bool(res.converged)
        assert 0.5 < float(res.theta_hat.beta) < 1.5

    def test_scipy_backend_on_traced_gate(self):
        """scipy_lm exposes the args channel (partial application
        inside); the gate admits it and results match optimistix."""
        m = _measure(seed=11)
        res_sp = estimate(
            euler_residual,
            m,
            covariance=IIDCovariance(),
            optimizer=scipy_lm(),
            parameters=_theta0(),
        )
        res_ox = estimate(
            euler_residual,
            m,
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        assert bool(res_sp.converged)
        np.testing.assert_allclose(
            float(res_sp.theta_hat.beta),
            float(res_ox.theta_hat.beta),
            rtol=1e-4,  # different solver internals; loose by design
        )


class TestKernelPathSemantics:
    def test_template_measure_itself_rides_traced_path(self):
        """The identity case is served by the same kernel (one code
        path) -- pinned PATH-SENSITIVELY via an args-channel spy
        (audit L5: the previous determinism-only assertion held on
        either path)."""

        class _Spy:
            def __init__(self):
                self.inner = optimistix_lm()
                self.args_calls = 0

            def __call__(self, residual_fn, theta_init, *, args=None):
                if args is not None:
                    self.args_calls += 1
                    return self.inner(residual_fn, theta_init, args=args)
                return self.inner(residual_fn, theta_init)

        m = _measure(seed=0)
        spy = _Spy()
        run = build_estimator(
            euler_residual,
            measure=m,
            covariance=IIDCovariance(),
            optimizer=spy,
            parameters=_theta0(),
        )
        r1 = run(_theta0(), m)
        r2 = run(_theta0(), m)
        assert spy.args_calls == 2  # identity case rides the kernel
        # Deterministic: same measure, same kernel, same trace.
        assert float(r1.theta_hat.beta) == float(r2.theta_hat.beta)
        assert float(r1.J_stat) == float(r2.J_stat)

    def test_different_measure_class_routes_legacy(self):
        """A measure of a DIFFERENT CLASS than the template must not ride
        the kernel (factory-time dispatch -- cache attr name, label
        probing, tau anchor -- assumed the template's type); it routes
        down the legacy closure path and still estimates correctly.

        Audit M3: the original version of this test passed
        ``template.with_key(...)`` -- the SAME class -- and asserted only
        ``theta_hat is not None``, so it could not fail. This version
        constructs a genuinely different class (an EmpiricalMeasure
        subclass) and asserts PATH-SENSITIVELY via an optimizer spy that
        records whether the args channel was used."""

        class _SpyOptimizer:
            """args-capable optimizer recording which channel each call used."""

            def __init__(self):
                self.inner = optimistix_lm()
                self.args_calls = 0
                self.legacy_calls = 0

            def __call__(self, residual_fn, theta_init, *, args=None):
                if args is None:
                    self.legacy_calls += 1
                    return self.inner(residual_fn, theta_init)
                self.args_calls += 1
                return self.inner(residual_fn, theta_init, args=args)

        class _SubclassMeasure(EmpiricalMeasure):
            """Same surface, different class: must NOT ride the kernel."""

        # jdc pytree registration is per-class; register the subclass
        # so it round-trips like its parent.
        import jax_dataclasses as _jdc

        _SubclassMeasure = _jdc.pytree_dataclass(_SubclassMeasure)

        spy = _SpyOptimizer()
        run = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            optimizer=spy,
            parameters=_theta0(),
        )
        # Same class: kernel path (args channel used).
        res_same = run(_theta0(), _measure(seed=1))
        assert spy.args_calls == 1 and spy.legacy_calls == 0
        assert bool(res_same.converged)

        # Different class: legacy path (no args), and the estimate is
        # still correct (same data => same theta_hat as the kernel path
        # would produce on the parent class, to float-identical tol).
        x = euler_data(seed=1, n=N)
        sub = _SubclassMeasure(x=x, mask=jnp.ones((N, 3)), weights=jnp.ones(N))
        res_sub = run(_theta0(), sub)
        assert spy.legacy_calls == 1, "different-class measure rode the kernel"
        assert bool(res_sub.converged)
        np.testing.assert_allclose(
            float(res_sub.theta_hat.beta),
            float(res_same.theta_hat.beta),
            rtol=1e-10,
        )


def test_optimizer_info_pytree_with_done():
    """OptimizerInfo with a concrete done flag still round-trips as a
    pytree (the #78 field is a traced child)."""
    info = OptimizerInfo(
        steps=3,
        final_objective=0.5,
        status="converged",
        backend="optimistix",
        done=jnp.asarray(True),
    )
    leaves, treedef = jax.tree_util.tree_flatten(info)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bool(rebuilt.done) is True
    assert rebuilt.status == "converged"


@pytest.mark.parametrize("seed", [0, 1])
def test_recovery_quality_on_traced_path(seed):
    """End-to-end sanity: the traced path still recovers the Euler truth."""
    res = estimate(
        euler_residual,
        _measure(seed=seed),
        covariance=IIDCovariance(),
        parameters=_theta0(),
    )
    assert bool(res.converged)
    assert abs(float(res.theta_hat.beta) - 0.96) < 0.05
    # N=800 sampling noise on gamma is wide; this is a sanity bound,
    # not a calibration claim (the MC studies own calibration).
    assert abs(float(res.theta_hat.gamma) - 2.0) < 1.2


class TestSupportsArgsProbe:
    """Audit L3: the probe must check Parameter.kind, not just the name."""

    def test_var_positional_args_is_not_args_capable(self):
        from emu_gmm.optimizer import _supports_args

        class _StarArgs:
            def __call__(self, residual_fn, theta_init, *args):
                return optimistix_lm()(residual_fn, theta_init)

        assert not _supports_args(_StarArgs())

    def test_keyword_forms_are_args_capable(self):
        from emu_gmm.optimizer import _supports_args

        class _Kw:
            def __call__(self, residual_fn, theta_init, *, args=None): ...

        class _PosKw:
            def __call__(self, residual_fn, theta_init, args=None): ...

        assert _supports_args(_Kw())
        assert _supports_args(_PosKw())

    def test_star_args_optimizer_estimates_via_legacy_path(self):
        """End-to-end: a v1-valid *args optimizer is served by the
        closure path and produces a correct estimate (pre-fix this
        crashed with a deep TypeError on the kernel path)."""

        class _StarArgs:
            def __init__(self):
                self.inner = optimistix_lm()

            def __call__(self, residual_fn, theta_init, *args):
                assert not args  # legacy path passes exactly two
                return self.inner(residual_fn, theta_init)

        res = estimate(
            euler_residual,
            _measure(seed=6),
            covariance=IIDCovariance(),
            optimizer=_StarArgs(),
            parameters=_theta0(),
        )
        assert bool(res.converged)
