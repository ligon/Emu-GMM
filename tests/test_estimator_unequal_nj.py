r"""Regression guard: unequal per-moment effective sample sizes ``N_j``.

This file pins the one convention that distinguishes ``emu-gmm`` from the
textbook GMM scaling found in every graduate econometrics text, and that
*no other test exercises* --- every other empirical test runs with
``mask = jnp.ones(...)``, i.e. a common ``N_j == N`` for all moments, the
exact case in which the distinction is invisible.

The convention (``docs/design.org`` Section 2, ``docs/mcar-asymptotics.org``
Sections 4--6): sample moments are per-coordinate means

    m_j = (1 / N_j) * sum_i d_ij w_i psi_j,     N_j = sum_i d_ij w_i

and the variance carries the *per-coordinate* normalisation

    [V_X]_jk = (1 / (N_j N_k)) * sum_i d_ij d_ik w_i^2 psi_j psi_k,

so that the criterion m' V_X^{-1} m -> chi^2_{M-K}. This is algebraically
identical to scaling moment j by sqrt(N_j) and weighting by an O(1)
covariance W_jk = sqrt(N_j N_k) [V_X]_jk --- the quadratic form is
invariant under m -> D m, V -> D V D. The textbook shortcut (means with a
*common* N in the weighting matrix) coincides only when all N_j are equal;
under genuine missingness (unequal N_j) it mis-weights theta_hat and breaks
the chi^2 calibration of J.

``TestPerCoordinateNormalisationGuard`` is the sharp, deterministic guard:
it fails immediately if ``IIDCovariance`` is ever "corrected" to a common-N
form. ``TestUnequalNjEndToEnd`` covers the integration path, and the
``slow`` MC confirms J is calibrated to chi^2_{M-K} under unequal N_j.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.covariance import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_data,
    euler_residual,
)
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated


def _identity_model(x, theta):  # noqa: ARG001 (theta deliberately unused)
    """Trivial model: psi_j(x_i) == x_ij, so V_X is hand-computable."""
    return x


class TestPerCoordinateNormalisationGuard:
    """Deterministic unit guard on the ``1/(N_j N_k)`` normalisation.

    A tiny hand-built case with deliberately unequal column masks. The
    closed form is exact, so we can assert both that ``IIDCovariance``
    matches the per-coordinate form *and* that it does NOT match the
    common-N (textbook) form. The second assertion is the regression
    teeth: switching the implementation to ``numer / N_total**2`` (or
    ``/ (N-1)``, etc.) flips this test red.
    """

    @staticmethod
    def _case():
        # M = 2 moments, N = 6 observations. psi values == x.
        # Column 0 observable in all 6 rows; column 1 in only 3 rows.
        # => N_0 = 6, N_1 = 3: a 2x difference in effective sample size.
        x = jnp.array(
            [
                [1.0, 2.0],
                [-1.0, 0.5],
                [2.0, -1.0],
                [0.5, 3.0],
                [-2.0, 1.5],
                [1.5, -0.5],
            ]
        )
        mask = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        )
        weights = jnp.ones(6)
        return x, mask, weights

    def _closed_form_numer_and_Nj(self):
        x, mask, weights = self._case()
        xn, mn, wn = np.asarray(x), np.asarray(mask), np.asarray(weights)
        N_j = (mn * wn[:, None]).sum(axis=0)  # (M,)
        w2 = wn * wn
        wpsi = mn * xn  # (N, M); masked-out cells zeroed
        numer = np.einsum("i,ij,ik->jk", w2, wpsi, wpsi)  # (M, M)
        return numer, N_j

    def test_matches_per_coordinate_closed_form(self):
        x, mask, weights = self._case()
        measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        V = np.asarray(
            IIDCovariance().covariance(_identity_model, jnp.array(0.0), measure)
        )
        numer, N_j = self._closed_form_numer_and_Nj()
        expected = numer / np.outer(N_j, N_j)  # the emu-gmm convention
        np.testing.assert_allclose(V, expected, rtol=1e-12, atol=1e-12)

    def test_differs_from_common_N_textbook_form(self):
        x, mask, weights = self._case()
        measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        V = np.asarray(
            IIDCovariance().covariance(_identity_model, jnp.array(0.0), measure)
        )
        numer, N_j = self._closed_form_numer_and_Nj()
        N_total = 6.0
        textbook = numer / (N_total**2)  # common-N: every moment treated as N
        # The two forms must NOT coincide here, precisely because N_1 != N.
        assert not np.allclose(V, textbook, rtol=1e-3, atol=1e-12)

    def test_scaling_factor_is_per_moment(self):
        """The (1,1) entry uses N_1^2, not N_total^2 --- the crux."""
        x, mask, weights = self._case()
        measure = EmpiricalMeasure(x=x, mask=mask, weights=weights)
        V = np.asarray(
            IIDCovariance().covariance(_identity_model, jnp.array(0.0), measure)
        )
        numer, N_j = self._closed_form_numer_and_Nj()
        N_total = 6.0
        # V[1,1] under the correct form is numer[1,1] / N_1^2. Relative to
        # the common-N form numer[1,1] / N_total^2 it is inflated by
        # (N_total / N_1)^2 = (6/3)^2 = 4. That factor is the entire point.
        ratio = V[1, 1] / (numer[1, 1] / (N_total**2))
        assert ratio == pytest.approx((N_total / N_j[1]) ** 2, rel=1e-10)
        assert ratio == pytest.approx(4.0, rel=1e-10)


# ---------------------------------------------------------------------------
# End-to-end: an MCAR mask with unequal N_j across the three Euler moments.
# ---------------------------------------------------------------------------

_N_E2E = 8000
# Per-moment inclusion probabilities; deliberately very unequal so the
# distinction from common-N scaling is exercised, not just present.
_INCLUSION_P = jnp.array([1.0, 0.6, 0.35])


def _unequal_nj_measure(seed: int) -> EmpiricalMeasure:
    """EmpiricalMeasure whose mask is MCAR but with unequal column rates.

    The mask is drawn independently of the data ``x`` (MCAR), so the
    estimator stays consistent (mcar-asymptotics.org Thm 4/5) while the
    per-moment effective sample sizes N_j differ substantially.
    """
    x = euler_data(seed=seed, n=_N_E2E)
    key = jax.random.PRNGKey(10_000 + seed)
    u = jax.random.uniform(key, (_N_E2E, 3))
    mask = (u < _INCLUSION_P[None, :]).astype(jnp.float64)
    weights = jnp.ones(_N_E2E)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


class TestUnequalNjEndToEnd:
    """Recovery and J behave correctly when N_j differ across moments."""

    def _run(self, seed: int = 0):
        return estimate(
            model=euler_residual,
            measure=_unequal_nj_measure(seed),
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
            theta_init=EulerParams(beta=0.9, gamma=1.0),
        )

    def test_N_j_are_unequal(self):
        r = self._run()
        n_j = np.asarray(r.diagnostics.N_j.array)
        # Sanity: the three moments really do have different effective N.
        assert n_j[0] > n_j[1] > n_j[2]
        # And close to the design inclusion rates * N (binomial, generous).
        assert n_j[0] == pytest.approx(_N_E2E, rel=0.02)
        assert n_j[1] == pytest.approx(0.6 * _N_E2E, rel=0.06)
        assert n_j[2] == pytest.approx(0.35 * _N_E2E, rel=0.08)

    def test_recovers_beta(self):
        r = self._run()
        assert float(r.theta_hat.beta) == pytest.approx(BETA_TRUE, abs=0.05)

    def test_recovers_gamma(self):
        r = self._run()
        # Masking inflates variance vs the balanced case; allow more slack.
        assert float(r.theta_hat.gamma) == pytest.approx(GAMMA_TRUE, abs=0.6)

    def test_converged(self):
        assert self._run().converged

    def test_J_dof_and_modest(self):
        r = self._run()
        assert r.J_dof == 1  # M=3, K=2, unchanged by masking
        assert jnp.isfinite(r.J_stat)
        # Correctly specified + correctly weighted => J ~ chi^2_1; a single
        # draw should be modest. A common-N misweighting would systematically
        # distort this once N_j are as unequal as here.
        assert r.J_stat < 30.0


class TestUnequalNjJCalibration:
    """Slow distributional check: J ~ chi^2_{M-K} under unequal N_j.

    This is the statistical complement to the deterministic unit guard: it
    confirms that the *calibration* of the over-identification test survives
    genuine missingness. A common-N weighting matrix would bias the mean of
    J away from M-K = 1 once the N_j are very unequal.
    """

    @pytest.mark.slow
    def test_mean_J_near_chi2_dof(self):
        n_reps = 40
        js = []
        for seed in range(n_reps):
            r = estimate(
                model=euler_residual,
                measure=_unequal_nj_measure(seed),
                covariance=IIDCovariance(),
                weighting=ContinuouslyUpdated(),
                regularization=DiagonalTikhonov(),
                optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
                theta_init=EulerParams(beta=0.9, gamma=1.0),
            )
            if bool(r.converged):
                js.append(float(r.J_stat))
        js_arr = np.asarray(js)
        assert js_arr.size >= n_reps - 2  # near-universal convergence
        # E[chi^2_1] = 1, Var = 2; mean over ~40 reps has SD ~ sqrt(2/40)
        # ~ 0.22. A wide band that still rejects a grossly miscalibrated J.
        assert js_arr.mean() == pytest.approx(1.0, abs=0.6)
        # Median of chi^2_1 is 0.455; loose band.
        assert 0.15 < float(np.median(js_arr)) < 1.2
