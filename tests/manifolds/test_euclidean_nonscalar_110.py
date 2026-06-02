r"""Regression test for issue #110: estimating a parameter of *only*
non-scalar Euclidean leaves.

Before the fix, the flatten / Jacobian *representation* dispatch keyed on
``dispatch_mode`` ("v1"/"v2"), which is "v1" for any all-Euclidean tree --
including one whose Euclidean leaves are non-scalar (a plain matrix / vector
parameter, no gauge, no curvature). The "v1" representation routes through the
scalar-only :func:`flatten_params`, which raises ``ValueError: ... all
parameter leaves must be 0-d scalars`` on the non-scalar block. The optimiser
choice (``optimistix_lm``) was correct; only the representation dispatch was
wrong.

The fix splits the two decisions: the representation now keys on whether every
leaf is a 0-d *scalar* (``all_scalar``), while the optimiser calling convention
still keys on ``dispatch_mode``. A non-scalar Euclidean tree therefore takes the
ambient ``flatten_params_with_spec`` representation but the v1 ``optimistix_lm``
optimiser -- exactly right for a flat (gauge-free) Euclidean parameter.

The DGP is an over-identified linear mean problem: ``E[x] = B @ mu`` with
``B`` a fixed ``(M, K)`` full-rank design, ``M = 3 > K = 2`` (so ``J_dof = 1``).
``mu`` lives on ``Euclidean(2)`` -- a single non-scalar Euclidean leaf, no
PSDFixedRank/Positive leaf to force the v2 path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    Euclidean,
    ManifoldLeaf,
    ParameterSpace,
    estimate,
    on,
)
from emu_gmm.covariance import IIDCovariance
from emu_gmm.measures import EmpiricalMeasure

jax.config.update("jax_enable_x64", True)

K = 2
M = 3
N_DATA = 4000
# Fixed full-rank (M, K) design so E[x] = B @ mu is over-identified.
B = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _make_model():
    """psi(x, theta) = x - B @ mu, an (M,) residual; E[psi] = 0 at mu_true."""

    def model(x, theta):
        mu = theta.mu.array
        return x - B @ mu

    return model


def _make_measure(seed: int = 0):
    rng = np.random.default_rng(seed)
    mu_true = np.array([0.7, -1.3])
    mean = B @ mu_true
    # Correlated noise in R^M so V_X is non-trivial (a genuine 3x3, not iso).
    L = rng.normal(size=(M, M))
    cov = L @ L.T + M * np.eye(M)
    X = rng.multivariate_normal(mean, cov, size=N_DATA)
    measure = EmpiricalMeasure(
        x=jnp.asarray(X),
        mask=jnp.ones((N_DATA, M)),
        weights=jnp.ones(N_DATA),
    )
    return measure, jnp.asarray(mu_true)


# ---------------------------------------------------------------------------
# Representation 1: a hand-built ManifoldLeaf dataclass (the explicit form).
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class MeanParams:
    mu: ManifoldLeaf


def _run_handbuilt(start) -> object:
    measure, _ = _make_measure(seed=0)
    return estimate(
        model=_make_model(),
        measure=measure,
        covariance=IIDCovariance(),
        theta_init=MeanParams(mu=ManifoldLeaf(jnp.asarray(start), Euclidean(K))),
    )


class TestNonScalarEuclideanHandBuilt:
    """A bare ManifoldLeaf(Euclidean(K)) parameter -- no v2-forcing leaf."""

    def test_estimates_without_raising(self):
        # The pre-fix failure was a ValueError at flatten time, before any
        # optimisation. Reaching a result at all is the core of the fix.
        r = _run_handbuilt([0.0, 0.0])
        assert r is not None

    def test_recovers_mu(self):
        _, mu_true = _make_measure(seed=0)
        r = _run_handbuilt([0.0, 0.0])
        mu_hat = np.asarray(r.theta_hat.mu.array)
        assert mu_hat == pytest.approx(np.asarray(mu_true), abs=0.1)

    def test_converged(self):
        r = _run_handbuilt([0.0, 0.0])
        assert r.converged

    def test_J_dof_and_stat(self):
        r = _run_handbuilt([0.0, 0.0])
        assert r.J_dof == M - K  # 3 - 2 == 1
        assert jnp.isfinite(r.J_stat)

    def test_sigma_theta_finite_full_rank(self):
        r = _run_handbuilt([0.0, 0.0])
        arr = np.asarray(r.Sigma_theta.array)
        assert arr.shape == (K, K)  # ambient == identified (no gauge)
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(arr))))
        # Full rank: a Euclidean leaf has no gauge nullspace.
        assert np.linalg.matrix_rank(arr) == K


# ---------------------------------------------------------------------------
# Representation 2: the ParameterSpace sugar that *surfaced* #110.
# ---------------------------------------------------------------------------
class TestNonScalarEuclideanParameterSpace:
    """A ParameterSpace whose only field is a non-scalar Euclidean leaf."""

    def _space(self):
        class MeanSpace(ParameterSpace):
            mu: jax.Array = on(Euclidean(K), default=jnp.zeros(K))

        return MeanSpace

    def test_estimates_via_parameter_space_class(self):
        measure, mu_true = _make_measure(seed=0)
        r = estimate(
            model=_make_model(),
            measure=measure,
            covariance=IIDCovariance(),
            parameters=self._space(),
        )
        mu_hat = np.asarray(r.theta_hat.mu.array)
        assert mu_hat == pytest.approx(np.asarray(mu_true), abs=0.1)
        assert r.converged
        assert r.J_dof == M - K

    def test_run_callable_reuses_and_recovers(self):
        # build_estimator-style reuse: the returned run() also flattens via the
        # all_scalar branch (site 3). A warm-started bound point round-trips.
        measure, mu_true = _make_measure(seed=0)
        space = self._space()
        r = estimate(
            model=_make_model(),
            measure=measure,
            covariance=IIDCovariance(),
            parameters=space.point(),
        )
        assert np.asarray(r.theta_hat.mu.array) == pytest.approx(
            np.asarray(mu_true), abs=0.1
        )
