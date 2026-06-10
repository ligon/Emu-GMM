"""#133 gates: Sigma_theta is the weighting-aware sandwich.

The pre-#133 formula ``pinv(G'(V*)^{-1}G)`` (inverse-information under
the efficient weighting) mis-stated the variance whenever the weighting
actually used was not ``(V*)^{-1}`` at ``theta_hat``, and ignored the
ridge. Gates:

(1) CU / tau=0 parity: sandwich collapses to the classical
    ``(G'V^{-1}G)^{-1}`` (float-identical, not bitwise).
(2) Identity weighting: equals the hand-computed robust sandwich
    ``(G'G)^{-1} G'VG (G'G)^{-1}`` and is strictly wider than the
    efficient bound the old formula reported.
(3) Fixed(V0) weighting: hand-computed sandwich with Lambda_0.
(4) Monte Carlo: Wald coverage under Identity weighting is nominal with
    the sandwich (the old formula undercovers -- pinned via the
    SE-ratio check, since re-running the old formula needs only G, V).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import (
    Fixed,
    Identity,
    IIDCovariance,
    build_estimator,
    estimate,
)
from emu_gmm.examples.euler import EulerParams, euler_data, euler_residual
from emu_gmm.measures import EmpiricalMeasure

N = 600


def _measure(seed: int, n: int = N) -> EmpiricalMeasure:
    x = euler_data(seed=seed, n=n)
    return EmpiricalMeasure(x=x, mask=jnp.ones((n, 3)), weights=jnp.ones(n))


def _theta0() -> EulerParams:
    return EulerParams(beta=0.9, gamma=1.0)


def _G_and_V(res, measure):
    """Hand-compute G and the UNREGULARIZED V at theta_hat."""
    G = np.asarray(measure.jacobian(euler_residual, res.theta_hat))
    V = np.asarray(IIDCovariance().covariance(euler_residual, res.theta_hat, measure))
    return G, V


class TestParityAndSandwich:
    def test_cu_tau_zero_collapses_to_classical(self):
        m = _measure(seed=0)
        res = estimate(
            euler_residual, m, covariance=IIDCovariance(), parameters=_theta0()
        )
        assert float(res.diagnostics.tau_realised) == 0.0
        G, V = _G_and_V(res, m)
        classical = np.linalg.inv(G.T @ np.linalg.inv(V) @ G)
        np.testing.assert_allclose(
            np.asarray(res.Sigma_theta.array), classical, rtol=1e-8
        )

    def test_identity_weighting_is_the_robust_sandwich(self):
        m = _measure(seed=1)
        res = estimate(
            euler_residual,
            m,
            covariance=IIDCovariance(),
            weighting=Identity(),
            parameters=_theta0(),
        )
        G, V = _G_and_V(res, m)
        bread_inv = np.linalg.inv(G.T @ G)
        sandwich = bread_inv @ (G.T @ V @ G) @ bread_inv
        np.testing.assert_allclose(
            np.asarray(res.Sigma_theta.array), sandwich, rtol=1e-8
        )
        # (On the Euler DGP Identity weighting happens to be ~99.996%
        # efficient -- the common-SDF moments make V nearly proportional
        # on col(G) -- so the DISCRIMINATION lives in the
        # heteroskedastic fixture below, not here.)

    def test_fixed_weighting_sandwich(self):
        m = _measure(seed=2)
        # A deliberately non-optimal anchor: V0 = V at theta_init,
        # scaled and tilted so Lambda_0 != V(theta_hat)^{-1}.
        V0 = np.asarray(IIDCovariance().covariance(euler_residual, _theta0(), m))
        V0 = 2.0 * V0 + 0.1 * np.diag(np.diag(V0))
        res = estimate(
            euler_residual,
            m,
            covariance=IIDCovariance(),
            weighting=Fixed(V0=jnp.asarray(V0)),
            parameters=_theta0(),
        )
        G, V = _G_and_V(res, m)
        Lam0 = np.linalg.inv(V0)
        bread_inv = np.linalg.inv(G.T @ Lam0 @ G)
        sandwich = bread_inv @ (G.T @ Lam0 @ V @ Lam0 @ G) @ bread_inv
        np.testing.assert_allclose(
            np.asarray(res.Sigma_theta.array), sandwich, rtol=1e-8
        )


class TestCoverage:
    """Heteroskedastic linear fixture where Identity weighting is BADLY
    inefficient (unlike the Euler DGP): C = [[1,0],[1,1],[0,1]], moment
    noise sds (0.05, 2.0, 0.5) with corr(m2, m3) = 0.6. GLS reads
    parameter `a` off the precise moment 1; OLS mixes in the noisy
    moment 2, inflating Var(a) by ~100x. The old efficient-bound formula
    reported the GLS variance for the OLS estimator -- SEs understated
    ~10x on `a` -- so its Wald coverage collapses while the sandwich is
    nominal.
    """

    C = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    THETA_TRUE = np.array([1.0, -0.5])
    SDS = np.array([0.05, 2.0, 0.5])

    @classmethod
    def _sigma_chol(cls):
        corr = np.eye(3)
        corr[1, 2] = corr[2, 1] = 0.6
        Sigma = np.outer(cls.SDS, cls.SDS) * corr
        return np.linalg.cholesky(Sigma)

    @classmethod
    def _dgp(cls, seed: int, n: int) -> EmpiricalMeasure:
        rng = np.random.default_rng(seed)
        x = cls.THETA_TRUE @ cls.C.T + rng.standard_normal((n, 3)) @ cls._sigma_chol().T
        return EmpiricalMeasure(
            x=jnp.asarray(x), mask=jnp.ones((n, 3)), weights=jnp.ones(n)
        )

    def test_identity_weighting_wald_coverage_is_nominal(self):
        """The failing-before gate: with the old efficient-bound SEs the
        CIs on `a` are ~10x too short (coverage ~ a coin flip at best);
        the sandwich restores nominal. 3-sigma binomial band at 250 reps.
        """
        import jax_dataclasses as jdc

        @jdc.pytree_dataclass
        class _P:
            a: float
            b: float

        C = jnp.asarray(self.C)

        def psi(x, theta):
            mean = C @ jnp.array([theta.a, theta.b])
            return mean - x

        n_reps, n = 250, 300
        run = build_estimator(
            psi,
            measure=self._dgp(0, n),
            covariance=IIDCovariance(),
            weighting=Identity(),
            parameters=_P(a=0.5, b=0.0),
        )
        hits_sandwich = np.zeros(2)
        hits_old = np.zeros(2)
        used = 0
        ratios = []
        for r in range(n_reps):
            m = self._dgp(1000 + r, n)
            res = run(_P(a=0.5, b=0.0), m)
            if not bool(res.converged):
                continue
            used += 1
            rec = res.record()
            th = np.asarray(rec.theta_flat)
            se = np.asarray(rec.se)
            hits_sandwich += (np.abs(th - self.THETA_TRUE) <= 1.96 * se).astype(float)
            # The OLD formula on the same fit (efficient bound).
            G = np.asarray(m.jacobian(psi, res.theta_hat))
            V = np.asarray(IIDCovariance().covariance(psi, res.theta_hat, m))
            se_old = np.sqrt(np.diag(np.linalg.inv(G.T @ np.linalg.inv(V) @ G)))
            hits_old += (np.abs(th - self.THETA_TRUE) <= 1.96 * se_old).astype(float)
            if r < 10:
                ratios.append(se_old / se)
        assert used >= 0.95 * n_reps
        cov_sandwich = hits_sandwich / used
        cov_old = hits_old / used
        # Sandwich: nominal (3-sigma band around 0.95 at ~250 reps).
        assert np.all(cov_sandwich >= 0.90), cov_sandwich
        assert np.all(cov_sandwich <= 0.995), cov_sandwich
        # Old formula: catastrophic undercoverage on `a` (the
        # failing-before pin -- this is what #133 fixed).
        assert cov_old[0] < 0.75, cov_old
        # And the understatement is large, not a rounding story.
        assert np.mean(np.stack(ratios)[:, 0]) < 0.5


@pytest.mark.parametrize("seed", [0, 1])
def test_fitrecord_se_carries_the_sandwich(seed):
    """FitRecord.se / standard_errors / coef_table all flow from
    Sigma_theta, so the fix propagates everywhere (spot-check one)."""
    m = _measure(seed=seed)
    res = estimate(
        euler_residual,
        m,
        covariance=IIDCovariance(),
        weighting=Identity(),
        parameters=_theta0(),
    )
    np.testing.assert_array_equal(
        np.asarray(res.record().se),
        np.sqrt(np.diag(np.asarray(res.Sigma_theta.array))),
    )


class TestIndefiniteMeatDiagnosis:
    """#138 (diagnose-loudly policy) + the binding-regime test owed from
    #133's spec: when the raw V(theta_hat) is indefinite, the sandwich
    meat can be indefinite, diag(Sigma) goes negative, SEs are NaN BY
    DESIGN, and the event is surfaced loudly rather than silently.

    Fixture: M=2, K=1 with G aligned to V's NEGATIVE eigenvector.
    V = [[1, 1.2], [1.2, 1]] has eigenpairs (2.2, (1,1)) and
    (-0.2, (1,-1)); the analytical moment m(a) = (a-1, 1-a) gives
    G = (1, -1)' exactly along the negative direction, so
    meat = G' Lambda* V Lambda* G < 0 while the RIDGED V* used for the
    solve is PD (the regulariser does its #111 job; the meat honestly
    reports that V itself is not a covariance matrix).
    """

    @staticmethod
    def _setup():
        import jax_dataclasses as jdc
        from emu_gmm import AnalyticalCovariance, AnalyticalMeasure

        @jdc.pytree_dataclass
        class _P:
            a: float

        V_indef = jnp.array([[1.0, 1.2], [1.2, 1.0]])  # eigs 2.2, -0.2

        measure = AnalyticalMeasure(
            expectation_fn=lambda model, th: jnp.array([th.a - 1.0, 1.0 - th.a])
        )
        cov = AnalyticalCovariance(covariance_fn=lambda model, th: V_indef)
        return _P, measure, cov, V_indef

    def test_flag_fires_and_ses_are_nan(self):
        import warnings

        _P, measure, cov, V_indef = self._setup()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = estimate(
                lambda x, th: jnp.zeros(2),  # psi unused by analytical paths
                measure,
                covariance=cov,
                parameters=_P(a=0.3),
            )
        # The ridge bound (V was indefinite, tau > 0 to restore PD).
        assert float(res.diagnostics.tau_realised) > 0.0
        # The diagnose-loudly contract: flag + NaN SE + UserWarning.
        assert bool(res.diagnostics.sigma_meat_indefinite)
        assert np.isnan(np.asarray(res.standard_errors.array)).any()
        assert any(
            "indefinite" in str(w.message) for w in caught
        ), "no loud warning emitted"
        # And the FitRecord carries the NaN SE for the summarizers'
        # n_valid_se accounting (#140) -- the chain is consistent.
        assert np.isnan(np.asarray(res.record().se)).any()

    def test_flag_silent_on_healthy_problems(self):
        import warnings

        m = _measure(seed=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = estimate(
                euler_residual,
                m,
                covariance=IIDCovariance(),
                parameters=_theta0(),
            )
        assert not bool(res.diagnostics.sigma_meat_indefinite)
        assert not np.isnan(np.asarray(res.standard_errors.array)).any()
        assert not any("indefinite" in str(w.message) for w in caught)
