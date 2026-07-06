r"""The ``estimate(linear=True)`` affine one-step fast path.

``linear=True`` asserts the whitened moment is affine in :math:`\theta`, so the
least-squares minimiser is reached in a single exact Gauss--Newton step and the
iterative optimiser is skipped (the resolved optimiser is wrapped in
``linear_solver(verify=False)``). These tests pin:

1. **Correctness** --- on a linear over-identified IV fit under a
   :math:`\theta`-independent weight, ``linear=True`` reproduces the iterative
   optimiser's point *and* standard errors to machine precision, and the
   optimiser info reports the one-step ``backend == "linear"`` (``steps == 1``).
2. **The guards** --- ``linear=True`` refuses the two detectable ways its
   affine assertion cannot hold: a manifold parameter (``TypeError``) and
   ``ContinuouslyUpdated`` weighting (``ValueError``).
3. **No double-wrap** --- ``linear=True`` alongside an explicit
   ``optimizer=linear_solver(...)`` uses that solver as-is.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    Fixed,
    Identity,
    IIDCovariance,
    IteratedWeighting,
    ManifoldLeaf,
    Positive,
    estimate,
    linear_solver,
    optimistix_lm,
)


@jdc.pytree_dataclass
class _P:
    b0: float
    b1: float


def _iv_resid(row, th):
    # Instruments (1, z1, z2, z3) times the structural residual -> M=4 moments,
    # affine in (b0, b1).
    Zr = jnp.array([1.0, row[2], row[3], row[4]])
    return Zr * (row[0] - th.b0 - th.b1 * row[1])


@pytest.fixture(scope="module")
def iv_data():
    rng = np.random.default_rng(20260705)
    n = 400
    z = rng.normal(size=(n, 3))
    xe = z @ np.array([1.0, 0.5, -0.5]) + rng.normal(size=n)
    y = 1.0 + 2.0 * xe + rng.normal(size=n)
    X = np.column_stack([y, xe, z])  # cols: y, xe, z1, z2, z3
    n_ = X.shape[0]
    Z = np.column_stack([np.ones(n_), z])
    V0 = jnp.asarray(Z.T @ Z / n_)  # the 2SLS fixed weight (Z'Z/n)
    return jnp.asarray(X), V0


def _fit(X, *, weighting, linear=False, optimizer=None):
    return estimate(
        _iv_resid,
        EmpiricalMeasure.from_arrays(X, M=4),
        covariance=IIDCovariance(),
        weighting=weighting,
        optimizer=optimizer,
        linear=linear,
        theta_init=_P(b0=0.0, b1=0.0),
    )


class TestCorrectness:
    def test_matches_iterative_point_and_se_to_machine_precision(self, iv_data):
        """linear=True == the iterative optimiser (same fixed weight), ~1e-10."""
        X, V0 = iv_data
        r_lin = _fit(X, weighting=Fixed(V0=V0), linear=True)
        r_it = _fit(X, weighting=Fixed(V0=V0), optimizer=optimistix_lm())
        np.testing.assert_allclose(
            float(r_lin.theta_hat.b0), float(r_it.theta_hat.b0), rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            float(r_lin.theta_hat.b1), float(r_it.theta_hat.b1), rtol=0, atol=1e-9
        )
        # Inference is untouched by the solver swap: SEs coincide.
        se_lin = np.asarray(r_lin.asymptotic().se())
        se_it = np.asarray(r_it.asymptotic().se())
        np.testing.assert_allclose(se_lin, se_it, rtol=1e-8, atol=1e-10)

    def test_reports_one_step_linear_backend(self, iv_data):
        """The fast path actually fired: backend 'linear', a single step."""
        X, V0 = iv_data
        info = _fit(X, weighting=Fixed(V0=V0), linear=True).diagnostics.optimizer_info
        assert info.backend == "linear"
        assert int(info.steps) == 1

    def test_identity_weight_also_one_step(self, iv_data):
        """An Identity (unweighted) over-identified fit is affine too -> one step."""
        X, _ = iv_data
        r_lin = _fit(X, weighting=Identity(), linear=True)
        r_it = _fit(X, weighting=Identity(), optimizer=optimistix_lm())
        assert r_lin.diagnostics.optimizer_info.backend == "linear"
        np.testing.assert_allclose(
            float(r_lin.theta_hat.b1), float(r_it.theta_hat.b1), rtol=0, atol=1e-9
        )

    def test_iterated_weighting_inner_solves_are_linear(self, iv_data):
        """IteratedWeighting holds W fixed within each inner solve -> affine there."""
        X, _ = iv_data
        r_lin = _fit(X, weighting=IteratedWeighting(), linear=True)
        r_it = _fit(X, weighting=IteratedWeighting(), optimizer=optimistix_lm())
        np.testing.assert_allclose(
            float(r_lin.theta_hat.b1), float(r_it.theta_hat.b1), rtol=0, atol=1e-8
        )


class TestGuards:
    def test_continuously_updated_weighting_refused(self, iv_data):
        """CU makes even a linear moment a nonlinear program -> ValueError."""
        X, _ = iv_data
        with pytest.raises(ValueError, match="ContinuouslyUpdated"):
            _fit(X, weighting=ContinuouslyUpdated(), linear=True)

    def test_default_weighting_is_cu_and_refused(self, iv_data):
        """The default weighting is CU, so bare linear=True refuses (steers to Fixed)."""
        X, _ = iv_data
        with pytest.raises(ValueError, match="theta-independent weight"):
            estimate(
                _iv_resid,
                EmpiricalMeasure.from_arrays(X, M=4),
                covariance=IIDCovariance(),
                linear=True,
                theta_init=_P(b0=0.0, b1=0.0),
            )

    def test_manifold_parameter_refused(self, iv_data):
        """A curved / gauge leaf has no one-step lstsq solution -> TypeError."""
        X, V0 = iv_data

        @jdc.pytree_dataclass
        class _PM:
            b0: float
            s: ManifoldLeaf

        def _resid(row, th):
            return jnp.array([row[0] - th.b0, th.s.array * row[2], row[3], row[4]])

        with pytest.raises(TypeError, match="Euclidean"):
            estimate(
                _resid,
                EmpiricalMeasure.from_arrays(X, M=4),
                covariance=IIDCovariance(),
                weighting=Fixed(V0=V0),
                linear=True,
                theta_init=_PM(b0=0.0, s=ManifoldLeaf(jnp.array(1.0), Positive())),
            )


class TestNoDoubleWrap:
    def test_explicit_linear_solver_used_as_is(self, iv_data):
        """linear=True with an explicit linear_solver(...) doesn't re-wrap it."""
        X, V0 = iv_data
        # verify=True solver: certifies then would delegate; on this affine fit
        # the certificate holds, so it still lands on the one-step solution.
        r = _fit(
            X,
            weighting=Fixed(V0=V0),
            linear=True,
            optimizer=linear_solver(fallback=optimistix_lm(), verify=True),
        )
        r_it = _fit(X, weighting=Fixed(V0=V0), optimizer=optimistix_lm())
        np.testing.assert_allclose(
            float(r.theta_hat.b1), float(r_it.theta_hat.b1), rtol=0, atol=1e-9
        )
