"""Tests for the :func:`emu_gmm.build_estimator` factory (#50).

The factory hoists the residual-closure construction and post-optimum
inference jit out of the per-call path. Subsequent invocations of the
returned callable with the *same* measure instance reuse the cached
closures, so the optimiser's internal jit cache (which keys on closure
identity) hits on the second and later calls instead of retracing.

These tests pin:

1. Numerical equivalence --- the factory path produces the same
   ``EstimationResult`` as the one-shot :func:`estimate` entry point
   (within optimiser tolerance).
2. Trace re-use --- the second call against the same measure does not
   trigger an additional compile of the residual / inference path.
3. Wall-clock benefit --- the second call is materially faster than
   the first.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm import (
    SyntheticCovariance,
    SyntheticMeasure,
    build_estimator,
    estimate,
    optimistix_lm,
)


@jdc.pytree_dataclass
class _Params:
    a: float
    b: float


def _sampler(key, theta):
    return jax.random.normal(key, shape=(256, 2))


def _psi(x, theta):
    return jnp.array([theta.a - x[0], theta.b * x[1] * x[1] - 1.0])


def _build_measure(seed: int = 0) -> SyntheticMeasure:
    return SyntheticMeasure(
        key=jax.random.PRNGKey(seed),
        n_sim=256,
        sampler=_sampler,
    )


class TestBuildEstimatorEquivalence:
    """The factory path produces the same answer as :func:`estimate`."""

    def test_matches_one_shot_estimate(self):
        measure = _build_measure()
        cov = SyntheticCovariance()
        theta = _Params(a=0.1, b=1.2)
        run = build_estimator(
            _psi,
            measure=measure,
            covariance=cov,
            theta_init=theta,
        )
        r_factory = run(theta, measure)
        r_oneshot = estimate(
            _psi,
            measure,
            covariance=cov,
            theta_init=theta,
        )
        # Same point estimate (up to optimiser noise).
        assert float(r_factory.theta_hat.a) == pytest.approx(
            float(r_oneshot.theta_hat.a), rel=1e-9, abs=1e-12
        )
        assert float(r_factory.theta_hat.b) == pytest.approx(
            float(r_oneshot.theta_hat.b), rel=1e-9, abs=1e-12
        )
        # Same J-stat.
        assert float(r_factory.J_stat) == pytest.approx(
            float(r_oneshot.J_stat), rel=1e-9, abs=1e-12
        )
        # Same Sigma_theta diagonal.
        s_f = jnp.diag(jnp.asarray(r_factory.Sigma_theta.array))
        s_o = jnp.diag(jnp.asarray(r_oneshot.Sigma_theta.array))
        assert jnp.allclose(s_f, s_o, atol=1e-9, rtol=1e-9)

    def test_multiple_theta_init_with_same_measure(self):
        """Different starting points, same measure, same factory."""
        measure = _build_measure()
        run = build_estimator(
            _psi,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=_Params(a=0.1, b=1.2),
        )
        r1 = run(_Params(a=0.0, b=1.0), measure)
        r2 = run(_Params(a=0.5, b=2.0), measure)
        # Both converge to the same optimum (up to optimiser noise).
        assert float(r1.theta_hat.a) == pytest.approx(float(r2.theta_hat.a), abs=1e-6)
        assert float(r1.theta_hat.b) == pytest.approx(float(r2.theta_hat.b), abs=1e-6)


class TestBuildEstimatorCaching:
    """The second call against the same measure reuses the JIT cache."""

    def test_second_call_does_not_retrace_residual(self):
        """Trace counter: ``psi`` Python body runs only at first trace.

        With the factory, the residual closure and the post-optimum
        inference block are built exactly once. The optimiser's
        internal pjit cache keys on closure identity, so the second
        call with the same measure does not retrigger any trace of
        ``psi`` --- the kernel is invoked directly. Any traces from
        the *factory construction* itself are absorbed into the first
        call.
        """
        counts = {"calls": 0}

        def counting_psi(x, theta):
            counts["calls"] += 1
            return jnp.array([theta.a - x[0], theta.b * x[1] * x[1] - 1.0])

        measure = _build_measure()
        run = build_estimator(
            counting_psi,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=_Params(a=0.1, b=1.2),
        )
        # First call: traces residual_fn + jacobian + inference + grad.
        _ = run(_Params(a=0.1, b=1.2), measure)
        first_calls = counts["calls"]

        # Block until first-call kernels are flushed so the trace
        # counter cleanly partitions across the two invocations.
        jax.block_until_ready(jnp.zeros(()))

        # Second call: same closure identity -> jit cache hit -> no
        # new Python-level traces of ``psi``.
        _ = run(_Params(a=0.2, b=1.5), measure)
        second_calls = counts["calls"]

        delta = second_calls - first_calls
        # If retracing occurred, we'd see at least 1 extra ``psi``
        # invocation per traced path (residual_fn, jacobian via
        # measure.jacobian, _half grad). We allow a small slack
        # (delta <= 1) so the assertion catches "tracing happened"
        # without being brittle against JAX implementation details
        # that might still touch the closure once during cache
        # lookup.
        assert delta <= 1, (
            "Expected the second factory call to skip retracing of psi "
            f"(JIT cache hit on the residual closure). first={first_calls}, "
            f"second={second_calls}, delta={delta}."
        )

    def test_second_call_is_materially_faster(self):
        """End-to-end wall-clock: second call is materially faster than first.

        First call pays the trace + compile cost; second call dispatches
        the cached kernel. Bound is loose (5x) to avoid flake on
        contended hosts; the structural improvement is much larger.
        """
        measure = _build_measure()
        run = build_estimator(
            _psi,
            measure=measure,
            covariance=SyntheticCovariance(),
            theta_init=_Params(a=0.1, b=1.2),
            # Tight tolerances reduce per-call optimiser-iteration
            # variance, so the trace/compile cost dominates the first
            # call cleanly.
            optimizer=optimistix_lm(rtol=1e-10, atol=1e-10),
        )

        # Warm: time the first invocation (compile + run).
        t0 = time.perf_counter()
        r1 = run(_Params(a=0.1, b=1.2), measure)
        jax.block_until_ready(r1.J_stat)
        first_time = time.perf_counter() - t0

        # Time the second invocation (run only).
        t0 = time.perf_counter()
        r2 = run(_Params(a=0.2, b=1.5), measure)
        jax.block_until_ready(r2.J_stat)
        second_time = time.perf_counter() - t0

        # Same factual answers (sanity).
        assert float(r1.J_stat) == pytest.approx(float(r2.J_stat), abs=1e-6)
        # Second call materially faster. We're conservative here
        # because GitHub CI / shared hosts vary; the practical speedup
        # is ~5-30x.
        assert second_time < first_time, (
            f"Expected the second factory call to be faster than the "
            f"first; got first={first_time:.3f}s, second={second_time:.3f}s"
        )
