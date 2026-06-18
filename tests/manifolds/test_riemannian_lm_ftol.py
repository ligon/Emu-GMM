"""ftol (cost-stagnation) convergence for the Riemannian LM (#156).

A continuously-updated + clustered manifold solve reaches the cost basin
(the estimate is correct) but the Gauss--Newton step cannot drive the
horizontal gradient to zero, so the gradient / step criteria never trip
and the LM previously ran to ``max_steps`` reporting ``converged=False``
even though the estimate was good. The diagnosis (issue #156): the LM had
no ftol (MINPACK / scipy cost-stagnation) termination. This module guards
the fix.

The fixture mirrors the #150 manifold bootstrap fixture: a
``PSDFixedRank(4, 2)`` factor ``A`` + ``Euclidean(1)`` ``phi`` with
gauge-invariant moments ``triu(A A') ++ phi``, estimated under a
``ClusteredCovariance`` (whose theta-dependence is what stalls the GN
step) with the default continuously-updated weighting.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import ClusteredCovariance, IIDCovariance
from emu_gmm.estimator import estimate
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, Positive, PSDFixedRank
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.weighting import ContinuouslyUpdated

_N, _K = 4, 2
_TRIU = jnp.array(np.triu_indices(_N)).T
_M = _N * (_N + 1) // 2 + 1  # 11
_MAX_STEPS = 200  # riemannian_lm default


@jdc.pytree_dataclass
class _P:
    A: ManifoldLeaf
    phi: ManifoldLeaf


def _mk(A, phi) -> _P:
    return _P(
        A=ManifoldLeaf(jnp.asarray(A), PSDFixedRank(_N, _K)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _model(x, th):
    A = th.A.array
    phi = th.phi.array[0]
    g = (A @ A.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _setup(seed=3, nc=6, opc=40, noise=0.1):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.normal(size=(_N, _K)))
    phi = 0.7
    tgt = jnp.concatenate(
        [(A @ A.T)[_TRIU[:, 0], _TRIU[:, 1]], jnp.reshape(jnp.asarray(phi), (1,))]
    )
    n = nc * opc
    x = np.asarray(tgt)[None, :] + noise * rng.standard_normal((n, _M))
    meas = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.ones((n, _M)), weights=jnp.ones(n)
    )
    cid = jnp.repeat(jnp.arange(nc, dtype=jnp.float64), opc)
    cov = ClusteredCovariance(cluster_ids=cid, n_clusters=nc)
    return _mk(A, phi), meas, cov, A


@pytest.mark.slow
class TestFtolCertifiesStalledManifoldSolve:
    """The CU + clustered manifold solve now certifies via cost stagnation."""

    def test_cu_clustered_now_certifies_with_correct_estimate(self):
        th, meas, cov, A_true = _setup()
        r = estimate(model=_model, measure=meas, covariance=cov, theta_init=th)
        # Pre-#156: steps == max_steps, converged=False. Now certified early.
        assert bool(r.converged) is True
        assert int(r.diagnostics.optimizer_info.steps) < _MAX_STEPS
        assert str(r.diagnostics.optimizer_info.status) == "converged"
        # The estimate is correct: gauge-invariant Gamma matches truth.
        A_hat = np.asarray(r.theta_hat.A.array)
        assert np.all(np.isfinite(A_hat))
        ev_hat = np.sort(np.linalg.eigvalsh(A_hat @ A_hat.T))
        ev_true = np.sort(np.linalg.eigvalsh(np.asarray(A_true) @ np.asarray(A_true).T))
        np.testing.assert_allclose(ev_hat, ev_true, atol=0.3)  # finite-sample

    def test_disabling_ftol_restores_pre_156_non_certification(self):
        # With ftol_patience above max_steps the criterion cannot fire, so the
        # stall reverts to the pre-#156 converged=False / max_iterations --
        # proving ftol is what flips the flag and that grad/step alone do not.
        th, meas, cov, _ = _setup()
        r = estimate(
            model=_model,
            measure=meas,
            covariance=cov,
            theta_init=th,
            optimizer=riemannian_lm(ftol_patience=10**9),
        )
        assert bool(r.converged) is False
        assert int(r.diagnostics.optimizer_info.steps) == _MAX_STEPS
        assert str(r.diagnostics.optimizer_info.status) == "max_iterations"


@pytest.mark.slow
class TestFtolDoesNotAlterHealthySolve:
    """A healthy solve (CU + IID) certifies via the gradient test long before
    the cost-stagnation counter could accumulate, so enabling ftol changes
    neither its iterate nor its step count.
    """

    def test_cu_iid_iterate_bitwise_unchanged_by_ftol(self):
        th, meas, _, _ = _setup()
        iid = IIDCovariance()
        r_default = estimate(model=_model, measure=meas, covariance=iid, theta_init=th)
        r_noftol = estimate(
            model=_model,
            measure=meas,
            covariance=iid,
            theta_init=th,
            optimizer=riemannian_lm(ftol_patience=10**9),
        )
        assert bool(r_default.converged) is True
        # Same step count and a bitwise-identical iterate: ftol never fired.
        assert int(r_default.diagnostics.optimizer_info.steps) == int(
            r_noftol.diagnostics.optimizer_info.steps
        )
        np.testing.assert_array_equal(
            np.asarray(r_default.theta_hat.A.array),
            np.asarray(r_noftol.theta_hat.A.array),
        )


@jdc.pytree_dataclass
class _ScaleParams:
    """A single positive scale parameter (Positive(1,1) leaf)."""

    sigma: jnp.ndarray
    __emu_manifolds__ = {"sigma": Positive()}


def _scale_residual(x, theta):
    s = theta.sigma
    return jnp.array([x[0] ** 2 - s**2, x[0] ** 4 - 3.0 * s**4])


class TestFtolGatedOffForGaugeFreeLeaf:
    """ftol is gated on ``total_gauge_dim > 0``. A ``Positive`` (gauge-free)
    solve must therefore be untouched by ftol -- which protects two things the
    #156 gate flagged: (1) the transient *rejected*-step stuck start of a
    ``Positive`` leaf climbing off a sub-true value, and (2) the ill-
    conditioned ``sigma -> 0`` boundary collapse whose pinned J / p-values are
    sensitive to the exact stopping point.
    """

    def test_positive_recovers_with_ftol_that_would_otherwise_fire(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 1.5, size=4000)  # sigma_true = 1.5
        meas = EmpiricalMeasure(
            x=jnp.asarray(draws[:, None]),
            mask=jnp.ones((4000, 2)),
            weights=jnp.ones(4000),
        )
        # A deliberately tiny ftol_patience: were ftol active on this gauge-
        # free tree, it would certify the stuck 0.5 start within 2 steps (the
        # initial transient is rejected steps at constant cost). Gauge-gating
        # keeps ftol OFF for total_gauge_dim==0, so the solve still climbs to
        # the truth and certifies via the gradient test.
        r = estimate(
            model=_scale_residual,
            measure=meas,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            optimizer=riemannian_lm(ftol_patience=2),
            theta_init=_ScaleParams(sigma=jnp.asarray(0.5)),
        )
        assert bool(r.converged) is True
        assert float(r.theta_hat.sigma) == pytest.approx(1.5, abs=0.1)
        assert float(r.diagnostics.final_gradient_norm) < 1e-4

    def test_positive_iterate_bitwise_identical_across_ftol_patience(self):
        # Gauge-gating => ftol is inert for a Positive leaf, so the iterate is
        # independent of ftol_patience (default vs a value that would fire).
        rng = np.random.default_rng(1)
        draws = rng.normal(0.0, 1.5, size=3000)
        meas = EmpiricalMeasure(
            x=jnp.asarray(draws[:, None]),
            mask=jnp.ones((3000, 2)),
            weights=jnp.ones(3000),
        )
        kw = dict(
            model=_scale_residual,
            measure=meas,
            covariance=IIDCovariance(),
            weighting=ContinuouslyUpdated(),
            regularization=DiagonalTikhonov(),
            theta_init=_ScaleParams(sigma=jnp.asarray(0.5)),
        )
        r_default = estimate(optimizer=riemannian_lm(), **kw)
        r_tiny = estimate(optimizer=riemannian_lm(ftol_patience=2), **kw)
        assert int(r_default.diagnostics.optimizer_info.steps) == int(
            r_tiny.diagnostics.optimizer_info.steps
        )
        np.testing.assert_array_equal(
            np.asarray(r_default.theta_hat.sigma),
            np.asarray(r_tiny.theta_hat.sigma),
        )
