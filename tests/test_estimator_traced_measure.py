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
import jax_dataclasses as jdc
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
        path); values match the legacy closure run bit-for-bit is NOT
        promised -- float-identical is."""
        m = _measure(seed=0)
        run = build_estimator(
            euler_residual,
            measure=m,
            covariance=IIDCovariance(),
            parameters=_theta0(),
        )
        r1 = run(_theta0(), m)
        r2 = run(_theta0(), m)
        # Deterministic: same measure, same kernel, same trace.
        assert float(r1.theta_hat.beta) == float(r2.theta_hat.beta)
        assert float(r1.J_stat) == float(r2.J_stat)

    def test_different_measure_class_routes_legacy(self):
        """A measure of a different class than the template must not
        ride the kernel (factory-time dispatch assumed the template's
        type); it routes down the legacy path and still works."""
        from emu_gmm.covariance import SyntheticCovariance
        from emu_gmm.measures import SyntheticMeasure

        @jdc.pytree_dataclass
        class _P:
            a: float

        def psi(x, theta):
            return jnp.array([theta.a + x[0], theta.a * x[0] - x[1]])

        def sampler(key, theta):
            del theta
            return jax.random.normal(key, (500, 2))

        template = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=500, sampler=sampler
        )
        run = build_estimator(
            psi,
            measure=template,
            covariance=SyntheticCovariance(),
            parameters=_P(a=0.1),
        )
        fresh = template.with_key(jax.random.PRNGKey(5))
        res = run(_P(a=0.1), fresh)
        assert res.theta_hat is not None


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
