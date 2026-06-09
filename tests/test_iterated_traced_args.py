"""PR B (B1) acceptance gates: IteratedWeighting rides the #124 args channel.

The legacy outer loop built a fresh ``Fixed``-weight closure per outer
step (``make_residual_fn(fixed_k)``), so the optimiser re-traced EVERY
outer step and every repeated fit. PR B threads ``args=(measure, L0_k)``
through one factory-stable kernel instead. Contracts pinned here:

(1) Parity: the args path and the legacy closure path (forced with a
    two-argument third-party optimiser wrapping the SAME solver) agree
    to float-identical tolerance -- both are bindings of the shared
    ``_residual_core`` / ``chol(ridge(V))`` math.
(2) No-retrace ACROSS OUTER STEPS: the psi trace count of a fit is
    independent of how many outer steps the driver runs (steps 2..k
    reuse the step-1 traces).
(3) No-retrace ACROSS FITS: repeated fits with fresh same-structure
    measures freeze the psi-call counter after fit 1.
(4) Non-convergence semantics unchanged on the args path: the
    ``max_iterations`` warning and the ``inner_non_convergence``
    override behave exactly as on the legacy path.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from emu_gmm import IIDCovariance, IteratedWeighting, build_estimator
from emu_gmm.examples.euler import EulerParams, euler_data, euler_residual
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm

N = 800


def _measure(seed: int) -> EmpiricalMeasure:
    x = euler_data(seed=seed, n=N)
    return EmpiricalMeasure(x=x, mask=jnp.ones((N, 3)), weights=jnp.ones(N))


def _theta0() -> EulerParams:
    return EulerParams(beta=0.9, gamma=1.0)


def _iterated(k: int = 10, tol: float = 1e-8) -> IteratedWeighting:
    return IteratedWeighting(weighting_iterations=k, weighting_tol=tol)


class _CountingModel:
    """psi wrapper counting Python-level executions (== trace events)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, theta):
        self.calls += 1
        return euler_residual(x, theta)


class _TwoArgOptimizer:
    """A third-party optimiser on the v1 protocol: no ``args`` channel.

    Wraps the SAME optimistix solver, so forcing the legacy pathway with
    it isolates the args-vs-closure difference to the jit boundaries.
    """

    def __init__(self):
        self.inner = optimistix_lm()

    def __call__(self, residual_fn, theta_init):
        return self.inner(residual_fn, theta_init)


class TestParity:
    def test_args_path_matches_legacy_closure_path(self):
        m = _measure(seed=3)

        run_args = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(),
            parameters=_theta0(),
        )
        res_args = run_args(_theta0(), m)

        run_legacy = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(),
            optimizer=_TwoArgOptimizer(),
            parameters=_theta0(),
        )
        res_legacy = run_legacy(_theta0(), m)

        assert bool(res_args.converged) and bool(res_legacy.converged)
        # Tolerances are looser than PR A's single-solve parity (1e-12):
        # the OUTER loop iterates the V-refresh map, so the last-ulp
        # jit-boundary differences in each L0_k anchor compound across
        # outer steps and ~65 inner LM steps (measured ~1e-10 on beta,
        # ~1e-9 on gamma). Both runs land on the same optimum within the
        # inner solver's own rtol=1e-8; this is float-identical-compounded
        # territory, NOT a math difference.
        np.testing.assert_allclose(
            float(res_args.theta_hat.beta),
            float(res_legacy.theta_hat.beta),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            float(res_args.theta_hat.gamma),
            float(res_legacy.theta_hat.gamma),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            float(res_args.J_stat), float(res_legacy.J_stat), rtol=1e-8
        )
        np.testing.assert_allclose(
            float(res_args.diagnostics.optimizer_info.final_objective),
            float(res_legacy.diagnostics.optimizer_info.final_objective),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(res_args.Sigma_theta.array),
            np.asarray(res_legacy.Sigma_theta.array),
            rtol=1e-6,
        )


class TestNoRetrace:
    def test_trace_count_independent_of_outer_step_count(self):
        """Steps 2..k reuse step 1's traces: the psi trace count of a
        one-outer-step fit equals that of a multi-outer-step fit."""
        counting_1 = _CountingModel()
        run_1 = build_estimator(
            counting_1,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(k=1, tol=1e-14),
            parameters=_theta0(),
        )
        with warnings.catch_warnings():
            # One outer step at a tiny tol cannot certify the V-refresh
            # fixed point -> the documented max_iterations warning.
            warnings.simplefilter("ignore", UserWarning)
            res_1 = run_1(_theta0(), _measure(seed=1))
        calls_one_step = counting_1.calls
        assert calls_one_step > 0  # tracing happened

        counting_k = _CountingModel()
        run_k = build_estimator(
            counting_k,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(k=8, tol=1e-10),
            parameters=_theta0(),
        )
        res_k = run_k(_theta0(), _measure(seed=1))
        assert bool(res_k.converged)
        # The multi-step run really ran more inner work than one outer
        # step (result.iterations totals the inner LM steps).
        assert int(res_k.iterations) > int(res_1.iterations)
        assert counting_k.calls == calls_one_step, (
            f"psi re-traced across outer steps: {counting_k.calls} "
            f"(k outer steps) != {calls_one_step} (1 outer step)"
        )

    def test_repeated_fits_with_fresh_measures_share_one_trace(self):
        """The #123/#124 headline on the outer-loop path: fresh
        same-structure measures are new leaf values on existing traces."""
        counting = _CountingModel()
        run = build_estimator(
            counting,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(),
            parameters=_theta0(),
        )
        res = run(_theta0(), _measure(seed=1))
        assert bool(res.converged)
        calls_after_first = counting.calls
        assert calls_after_first > 0

        for seed in (2, 3, 4):
            res = run(_theta0(), _measure(seed=seed))
            assert bool(res.converged)
        assert counting.calls == calls_after_first, (
            f"psi re-traced on fresh same-structure measures: "
            f"{counting.calls} != {calls_after_first}"
        )


class TestNonConvergenceSemantics:
    def test_max_iterations_warns_and_flags_on_args_path(self):
        run = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(k=1, tol=1e-14),
            parameters=_theta0(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run(_theta0(), _measure(seed=5))
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("exhausted" in m and "outer iterations" in m for m in msgs)
        assert not bool(res.converged)
        assert res.diagnostics.optimizer_info.status == "max_iterations"

    def test_inner_non_convergence_warns_and_flags_on_args_path(self):
        # optimistix_lm is args-capable, so max_steps=1 starves the
        # INNER Fixed-weight solve on the args path itself.
        run = build_estimator(
            euler_residual,
            measure=_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=_iterated(k=3, tol=1e-10),
            optimizer=optimistix_lm(max_steps=1),
            parameters=EulerParams(beta=0.5, gamma=5.0),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run(EulerParams(beta=0.5, gamma=5.0), _measure(seed=6))
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("did not certify convergence" in m for m in msgs)
        assert not bool(res.converged)
        assert res.diagnostics.optimizer_info.status == "inner_non_convergence"
