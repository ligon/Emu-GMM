r"""Golden-fidelity + constructibility gate for the OptimizationResult/law split.

The estimation/inference split (docs/optimization-result-law-split.org) moved
the statistical surface (``se`` / ``cov`` / ``coef_table`` / the J p-values /
gauge-invariant functional SEs) off :class:`~emu_gmm.types.OptimizationResult`
and onto :class:`~emu_gmm.law.AsymptoticLaw`, which now *assembles* the #133
sandwich from the result's optimization ingredients rather than reading a stored
``Sigma_theta``. This module is the contract that the assembly reproduces the
pre-refactor numbers **bit-for-bit** (rtol 1e-9 on Sigma / se; 1e-9 on the J
p-values), against literals frozen from the pre-refactor outputs (the golden
capture), so the refactor is provably drift-free.

It also unit-tests the *constructibility* win (R9): a hand-built artificial
``OptimizationResult`` with a chosen ``(G, Lambda, V)`` feeds ``AsymptoticLaw``
and reproduces the hand-computed sandwich ``B^+ M B^+`` on a tiny known example
--- the inference assembly is testable in isolation, with no fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import (
    AsymptoticLaw,
    ContinuouslyUpdated,
    EmpiricalMeasure,
    IIDCovariance,
    OptimizationResult,
    estimate,
)
from emu_gmm._internal.asymptotic import asymptotic_covariance
from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule
from emu_gmm.optimizer import optimistix_lm

# Manifold Phase-4 fixture lives under tests/manifolds.
sys.path.insert(0, str(Path(__file__).parent / "manifolds"))

import jax_dataclasses as jdc  # noqa: E402

# ---------------------------------------------------------------------------
# Golden literals (frozen from the pre-refactor statistical outputs).
# ---------------------------------------------------------------------------
SCALAR_SIGMA = np.array(
    [
        [0.00037884768172636, -0.00039028360578872],
        [-0.00039028360578872, 0.00041676740305366],
    ]
)
SCALAR_SE = np.array([0.01946400990870991, 0.02041488190153594])
SCALAR_COEF_EST = np.array([1.5185252699449683, -0.7123898245369763])
SCALAR_COEF_SE = np.array([0.01946400990870991, 0.02041488190153594])
SCALAR_COEF_T = np.array([78.01708266010726, -34.89561330665247])
SCALAR_JSTAT = 1.1212542543190924
SCALAR_JPVAL = 0.2896485407864833
SCALAR_JPVAL_ADJ = 0.2896485407864833

MANIF_EIG_SE = np.array([0.00887448665275327, 0.00843274112143029])
MANIF_GAMMA_SE = np.array(
    [
        0.00652541729676450,
        0.00430597672837929,
        0.00086770890211056,
        0.00455982837797327,
        0.00399337835494725,
        0.00580227596393087,
        0.00591368067000553,
        0.00640781917825809,
        0.00648735450505379,
        0.00631020804426361,
        0.00476407343506985,
        0.00356488649214919,
        0.00409831128808384,
        0.00657169819242249,
        0.00619599405898224,
    ]
)
MANIF_JSTAT = 5.486010034635165
MANIF_JPVAL = 0.48314919269944817
MANIF_SIGMA_TRACE = 0.00040719259103943204
MANIF_SIGMA_FRO = 0.00016962850709101044
MANIF_GAMMA_COV_TRACE = 0.0004239957231510945

# Binding-ridge regime: the adjusted p-value is REASSEMBLED by the law from
# ``V`` (moment_covariance), ``V* = inv(weighting_matrix)`` and ``G``
# (moment_jacobian) --- a distinct #133/#137 codepath from the nominal one, so
# it needs its own drift lock. Frozen from the deterministic ill-conditioned
# fixture (kappa_target=10, tau_threshold=1e-6), where the adjusted value
# departs measurably from the nominal.
RIDGE_JSTAT = 1.5697455486495403e-05
RIDGE_JPVAL_NOMINAL = 0.99683878848399776
RIDGE_JPVAL_ADJUSTED = 0.89971500695696771


@jdc.pytree_dataclass
class _P2:
    a: float
    b: float


def _scalar_model(x, th):
    y = x[0]
    u = x[1]
    v = x[2]
    z = x[3:6]
    return z * (y - th.a * u - th.b * v)


def _fit_scalar() -> OptimizationResult:
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(size=(n, 3))
    u = Z @ np.array([1.2, 1.0, 0.9]) + rng.normal(size=n) * 0.4
    v = Z @ np.array([0.9, 1.1, 1.0]) + rng.normal(size=n) * 0.4
    y = 1.5 * u - 0.7 * v + rng.normal(size=n) * 0.3
    X = np.column_stack([y, u, v, Z])
    meas = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    return estimate(
        _scalar_model,
        meas,
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        theta_init=_P2(a=1.5, b=-0.7),
    )


class TestScalarFidelity:
    def test_law_cov_and_se_match_golden(self):
        law = _fit_scalar().asymptotic()
        np.testing.assert_allclose(
            np.asarray(law.cov()), SCALAR_SIGMA, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(np.asarray(law.se()), SCALAR_SE, rtol=1e-9)
        np.testing.assert_allclose(
            np.asarray(law.standard_errors), SCALAR_SE, rtol=1e-9
        )

    def test_law_j_test_matches_golden(self):
        r = _fit_scalar()
        law = r.asymptotic()
        np.testing.assert_allclose(float(r.objective_value), SCALAR_JSTAT, rtol=1e-9)
        np.testing.assert_allclose(float(law.J_pvalue), SCALAR_JPVAL, rtol=1e-9)
        np.testing.assert_allclose(
            float(law.J_pvalue_adjusted), SCALAR_JPVAL_ADJ, rtol=1e-9
        )
        jt = law.j_test()
        assert jt.n_overid == 1
        np.testing.assert_allclose(jt.J_stat, SCALAR_JSTAT, rtol=1e-9)
        np.testing.assert_allclose(jt.J_pvalue, SCALAR_JPVAL, rtol=1e-9)

    def test_law_coef_table_matches_golden(self):
        ct = _fit_scalar().asymptotic().coef_table
        np.testing.assert_allclose(
            ct["estimate"].to_numpy(), SCALAR_COEF_EST, rtol=1e-9
        )
        np.testing.assert_allclose(
            ct["std_error"].to_numpy(), SCALAR_COEF_SE, rtol=1e-9
        )
        np.testing.assert_allclose(ct["t_stat"].to_numpy(), SCALAR_COEF_T, rtol=1e-9)


def _fit_binding_ridge() -> OptimizationResult:
    """A deterministic fit whose DiagonalTikhonov ridge binds (adjusted != nominal)."""
    from emu_gmm import (
        AnalyticalCovariance,
        AnalyticalMeasure,
        DiagonalTikhonov,
    )

    def _model(x, th):
        del x, th
        return jnp.zeros((3,))

    def _moments(model, th):
        del model
        return jnp.array(
            [
                th.a + 0.5 * th.b - 0.1,
                -0.3 * th.a + th.b - 0.05,
                0.7 * th.a + 0.4 * th.b + 0.02,
            ]
        )

    def _ill_cov(model, th):
        del model, th
        rng = np.random.default_rng(seed=11)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        V = jnp.asarray(Q) @ jnp.diag(jnp.array([1.0, 1e-3, 1e-6])) @ jnp.asarray(Q).T
        return 0.5 * (V + V.T)

    return estimate(
        _model,
        AnalyticalMeasure(expectation_fn=_moments),
        covariance=AnalyticalCovariance(covariance_fn=_ill_cov),
        weighting=ContinuouslyUpdated(),
        regularization=DiagonalTikhonov(kappa_target=10.0, tau_threshold=1e-6),
        optimizer=optimistix_lm(rtol=1e-8, atol=1e-8),
        theta_init=_P2(a=0.0, b=0.0),
    )


class TestBindingRidgeFidelity:
    r"""The reassembled adjusted J p-value (binding ridge) is drift-locked (#133/#137)."""

    def test_binding_ridge_adjusted_pvalue_matches_golden(self):
        r = _fit_binding_ridge()
        assert bool(r.diagnostics.binding_ridge) is True
        law = r.asymptotic()
        np.testing.assert_allclose(
            float(r.objective_value), RIDGE_JSTAT, rtol=1e-7, atol=1e-12
        )
        np.testing.assert_allclose(float(law.J_pvalue), RIDGE_JPVAL_NOMINAL, rtol=1e-9)
        # The adjusted value is the reassembled generalised-chi^2 survival
        # function --- a genuinely different number, not the nominal fallback.
        np.testing.assert_allclose(
            float(law.J_pvalue_adjusted), RIDGE_JPVAL_ADJUSTED, rtol=1e-8
        )
        assert abs(float(law.J_pvalue) - float(law.J_pvalue_adjusted)) > 1e-4


class TestManifoldFidelity:
    def _law(self):
        import test_estimator_inference_phase4 as ph4

        res, _spec, _M, _ = ph4._run_estimate(2, seed=300)
        return AsymptoticLaw(res)

    def test_law_cov_matches_golden(self):
        sigma = np.asarray(self._law().cov())
        np.testing.assert_allclose(np.trace(sigma), MANIF_SIGMA_TRACE, rtol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(sigma), MANIF_SIGMA_FRO, rtol=1e-9)

    def test_leaf_eigenvalue_and_gamma_se_match_golden(self):
        law = self._law()
        psd = [nm for nm, m in law.leaves if type(m).__name__ == "PSDFixedRank"][0]
        np.testing.assert_allclose(
            np.asarray(law.leaf(psd).se("eigenvalues")), MANIF_EIG_SE, rtol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(law.leaf(psd).se("gamma")), MANIF_GAMMA_SE, rtol=1e-8
        )
        np.testing.assert_allclose(
            np.trace(np.asarray(law.leaf(psd).cov("gamma"))),
            MANIF_GAMMA_COV_TRACE,
            rtol=1e-8,
        )

    def test_deprecated_gamma_shims_still_work(self):
        law = self._law()
        with pytest.warns(DeprecationWarning):
            np.testing.assert_allclose(
                np.asarray(law.eigenvalue_se()), MANIF_EIG_SE, rtol=1e-8
            )
        with pytest.warns(DeprecationWarning):
            np.testing.assert_allclose(
                np.asarray(law.gamma_se()), MANIF_GAMMA_SE, rtol=1e-8
            )

    def test_law_j_matches_golden(self):
        law = self._law()
        np.testing.assert_allclose(law.j_test().J_stat, MANIF_JSTAT, rtol=1e-9)
        np.testing.assert_allclose(float(law.J_pvalue), MANIF_JPVAL, rtol=1e-9)


class TestConstructibility:
    r"""An artificial OptimizationResult -> AsymptoticLaw gives B^+ M B^+ (R9)."""

    def test_hand_built_result_reproduces_sandwich(self):
        # A tiny known linear-algebra example: M=3 moments, D=2 params, an
        # inefficient (non-V^{-1}) weighting so the sandwich is a genuine
        # B^+ M B^+, not B^{-1}.
        G = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Lam = jnp.asarray(
            [[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 1.5]]
        )  # SPD weighting
        V = jnp.asarray([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.2]])  # raw meat

        result = OptimizationResult(
            theta_hat=_P2(a=0.3, b=-0.5),
            moment_jacobian=G,
            weighting_matrix=Lam,
            moment_covariance=V,
        )
        law = AsymptoticLaw(result)

        # Hand-computed B^+ M B^+.
        bread = G.T @ Lam @ G
        meat = G.T @ Lam @ V @ Lam @ G
        meat = 0.5 * (meat + meat.T)
        bpinv = pinv_eigvalrule(bread, drop_smallest=0)
        sigma_expected = bpinv @ meat @ bpinv
        sigma_expected = 0.5 * (sigma_expected + sigma_expected.T)

        np.testing.assert_allclose(
            np.asarray(law.cov()), np.asarray(sigma_expected), rtol=1e-12, atol=1e-15
        )
        # And the shared routine agrees.
        np.testing.assert_allclose(
            np.asarray(law.cov()),
            np.asarray(asymptotic_covariance(G, Lam, V, gauge_dim=0)),
            rtol=1e-12,
            atol=1e-15,
        )
        # se = sqrt(diag) of that.
        np.testing.assert_allclose(
            np.asarray(law.se()),
            np.sqrt(np.diag(np.asarray(sigma_expected))),
            rtol=1e-12,
        )
