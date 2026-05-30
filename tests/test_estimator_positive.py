"""Positive(1,1) manifold acceptance test --- the v2 lite slice.

A scalar Gaussian-scale GMM with a single positive scale parameter
``sigma > 0`` and two moments (variance + Gaussian 4th-moment), so
``M = 2``, ``K = 1`` and ``J_dof = 1``. Estimated through the full
``estimate`` pipeline, which auto-dispatches to ``RiemannianLM`` because
``sigma`` is annotated as a :class:`Positive` leaf.

Truth: ``sigma_true = 1.5``. Recovery from a sub-true start (0.5) exercises
the exponential retraction staying positive; a separate near-zero start
(0.05) demonstrates the positivity guarantee versus a plain Euclidean LM.
"""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.manifolds import Positive
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import EstimationResult
from emu_gmm.weighting import ContinuouslyUpdated

SIGMA_TRUE = 1.5
N_DATA = 5000


@jdc.pytree_dataclass
class ScaleParams:
    """A single positive scale parameter ``sigma``.

    The ``__emu_manifolds__`` annotation tags ``sigma`` as living on the
    :class:`Positive` manifold so ``manifold_spec_from_params`` routes it
    to ``RiemannianLM`` (rather than the v1 Euclidean default).
    """

    sigma: jnp.ndarray

    __emu_manifolds__ = {"sigma": Positive()}


def scale_residual(x, theta):
    """Per-observation moment errors identifying ``sigma``.

    m_0 = x^2 - sigma^2          (variance moment)
    m_1 = x^4 - 3 sigma^4        (Gaussian 4th-moment / kurtosis)
    """
    xi = x[0]
    s = theta.sigma
    return jnp.stack([xi**2 - s**2, xi**4 - 3.0 * s**4])


def _make_measure(seed: int = 0) -> EmpiricalMeasure:
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, SIGMA_TRUE, size=N_DATA)
    x = jnp.asarray(draws[:, None])  # (N, 1)
    mask = jnp.ones((N_DATA, 2))  # (N, M=2)
    weights = jnp.ones(N_DATA)
    return EmpiricalMeasure(x=x, mask=mask, weights=weights)


class TestPositiveAcceptance:
    """sigma>0 recovery, mirroring the empirical-path acceptance test."""

    def _run(self) -> EstimationResult:
        return estimate(
            model=scale_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=None,  # auto-dispatch -> RiemannianLM
            theta_init=ScaleParams(sigma=jnp.asarray(0.5)),
        )

    def test_recovers_sigma(self):
        r = self._run()
        assert float(r.theta_hat.sigma) == pytest.approx(SIGMA_TRUE, abs=0.1)

    def test_stays_positive(self):
        r = self._run()
        assert float(r.theta_hat.sigma) > 0.0

    def test_converged(self):
        r = self._run()
        assert r.converged

    def test_J_dof_is_one(self):
        r = self._run()
        assert r.J_dof == 1  # M=2, dim_info = total_dimension - gauge = 1

    def test_J_stat_finite_and_modest(self):
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        assert float(r.J_stat) < 30.0

    def test_Sigma_theta_finite_and_rank_one(self):
        r = self._run()
        arr = r.Sigma_theta.array
        assert arr.shape == (1, 1)
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert float(arr[0, 0]) > 0.0

    def test_Sigma_theta_is_ambient_natural_scale(self):
        """Sigma_theta is the ambient natural-scale variance Var(sigma_hat),
        matching ../ManifoldGMM (Convention B) -- NOT a metric-rescaled /
        log-scale variance.

        Every first-order retraction has unit differential at v=0, so the
        Jacobian in the retraction chart equals the ambient Jacobian and
        Sigma_theta = (G' Lambda G)^{-1} = Sigma_eucl. The Positive
        manifold therefore reports the *same* asymptotic variance a plain
        Euclidean leaf would (first-order asymptotics are
        parameterisation-invariant at an interior optimum); its value-add
        is the positivity-preserving retraction during optimisation, not a
        different covariance. A wrong x-scaling (log-scale, x^{-2}) or x^2
        (inverse-metric, x^{-4}) would make these two disagree.
        """
        r = self._run()
        sigma_hat = float(r.theta_hat.sigma)
        sigma_riem = float(r.Sigma_theta.array[0, 0])

        # Euclidean twin: identical residual, sigma as a plain Euclidean
        # leaf, evaluated at the same sigma_hat. Its Sigma is the ambient
        # delta-method variance.
        @jdc.pytree_dataclass
        class EucScaleParams:
            sigma: jnp.ndarray

        def euc_residual(x, theta):
            xi = x[0]
            s = theta.sigma
            return jnp.stack([xi**2 - s**2, xi**4 - 3.0 * s**4])

        r_euc = estimate(
            model=euc_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            theta_init=EucScaleParams(sigma=jnp.asarray(sigma_hat)),
        )
        sigma_eucl = float(r_euc.Sigma_theta.array[0, 0])

        # The manifold path reports the ambient variance: the two agree.
        assert sigma_riem == pytest.approx(sigma_eucl, rel=1e-6)

    def test_labels(self):
        r = self._run()
        # The per-parameter name comes from the Positive leaf's
        # tangent_basis_names("sigma") == ["sigma"], so the label context
        # names the single tangent coordinate "sigma" (unchanged in shape
        # from the v1 Euclidean scalar leaf).
        assert r.labels.param_names == ("sigma",)
        assert isinstance(r.Sigma_theta, ha.NamedArray)
        # Sigma_theta is a 1x1 matrix on the generic Params/ParamsDual
        # axes (v1 labelling); the readable per-coordinate name is in
        # labels.param_names.
        assert r.Sigma_theta.array.shape == (1, 1)


class TestPositivePositivityGuarantee:
    """A near-zero start converges without sigma ever crossing 0."""

    def test_tiny_start_converges_positive(self):
        r = estimate(
            model=scale_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            optimizer=riemannian_lm(),
            theta_init=ScaleParams(sigma=jnp.asarray(0.05)),
        )
        assert float(r.theta_hat.sigma) > 0.0
        assert float(r.theta_hat.sigma) == pytest.approx(SIGMA_TRUE, abs=0.15)


class TestPositiveDefaults:
    def test_minimal_call_dispatches_riemannian(self):
        r = estimate(
            model=scale_residual,
            measure=_make_measure(seed=0),
            covariance=IIDCovariance(),
            theta_init=ScaleParams(sigma=jnp.asarray(0.8)),
        )
        assert isinstance(r, EstimationResult)
        assert float(r.theta_hat.sigma) > 0.0
