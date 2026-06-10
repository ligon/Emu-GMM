"""Quotient (gauge-aware) K-statistic tests — issue #41.

Before #41 the K-statistic had no manifold support at all: a
``ManifoldLeaf`` parameter tree died in the v1 scalar-only
``flatten_params`` inside ``measure.jacobian``. The #41 probe
(2026-06-10, recorded on the issue) established what a naive wiring
would have done: for a gauge-invariant model the whitened
:math:`\\widetilde D` carries exactly ``gauge_dim`` roundoff-scale
singular values, and a blind thin-QR manufactures junk directions whose
projection is empirically ~chi^2_gauge — leaving ``p_K`` *accidentally*
calibrated at the ambient df but the realised statistic
roundoff-determined (K differing by up to ~6 chi-squared units between
algebraically equivalent routes), ``S`` referred to the wrong df, and
``gauge_dim`` degrees of freedom of pure noise diluting power. The
repair must be the PAIR (top-``p_id`` SVD projection together with
``df_K = p_id``): the probe measured the pair calibrated (rej@5%
0.033–0.053) while swapping the df alone *creates* miscalibration.

Fixture adapted from ``tests/manifolds/test_manifold_acceptance_phase6.py``:
a ``PSDFixedRank(5, 2)`` leaf ``Y`` plus a ``Euclidean(1)`` leaf ``phi``,
moments ``triu(Y Y') ++ phi`` — gauge-invariant by construction.
``M = 16``, ambient ``p = 11``, ``gauge_dim = 1``, identified
``p_id = 10``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import IIDCovariance, SyntheticCovariance
from emu_gmm.inference import KStatisticResult, k_statistic
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.measures import EmpiricalMeasure, SyntheticMeasure

N_SIDE = 5
K_RANK = 2
GAUGE_DIM = K_RANK * (K_RANK - 1) // 2  # 1
AMBIENT_P = N_SIDE * K_RANK + 1  # 11 (vec Y + phi)
IDENTIFIED_P = AMBIENT_P - GAUGE_DIM  # 10
M_FULL = N_SIDE * (N_SIDE + 1) // 2 + 1  # 16

_TRIU = jnp.array(np.triu_indices(N_SIDE)).T  # (15, 2)


@jdc.pytree_dataclass
class ProductParams:
    """``PSDFixedRank(5, 2)`` ``Y`` leaf + ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N_SIDE, K_RANK)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _model(x, theta):
    """psi = (triu(Y Y') ++ phi) - x: gauge-invariant in Y by construction."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _model_sliced(m_keep: int):
    """First ``m_keep`` coordinates of the full model (for M < p tests)."""

    def model(x, theta):
        return _model_full_padded(x, theta)[:m_keep]

    def _model_full_padded(x, theta):
        Y = theta.Y.array
        phi = theta.phi.array[0]
        g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
        full = jnp.concatenate([g, jnp.reshape(phi, (1,))])
        return full - x_pad(x, full.shape[0])

    def x_pad(x, m_full):
        # x already has m_keep entries; pad with zeros for the dropped
        # coordinates so the subtraction broadcasts, then re-slice.
        return jnp.concatenate([x, jnp.zeros(m_full - x.shape[0], dtype=x.dtype)])

    return model


def _orthogonal(seed: int, k: int, reflect: bool) -> jnp.ndarray:
    """Random orthogonal Q with the requested O(K) component (det sign)."""
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(k, k)))
    q, r = jnp.linalg.qr(g)
    q = q @ jnp.diag(jnp.sign(jnp.diag(r)))
    want = -1.0 if reflect else 1.0
    if float(jnp.linalg.det(q)) * want < 0:
        q = q.at[:, 0].set(-q[:, 0])
    return q


def _truth(seed: int = 0):
    rng = np.random.default_rng(seed)
    A_true = jnp.asarray(rng.normal(size=(N_SIDE, K_RANK)))
    phi_true = 0.7
    g_true = (A_true @ A_true.T)[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    return A_true, phi_true, target


def _empirical_measure(target, n_obs: int = 400, noise: float = 0.1, seed: int = 1):
    rng = np.random.default_rng(seed)
    m = int(target.shape[0])
    x = np.asarray(target)[None, :] + noise * rng.standard_normal((n_obs, m))
    return EmpiricalMeasure(
        x=jnp.asarray(x),
        mask=jnp.ones((n_obs, m)),
        weights=jnp.ones(n_obs),
    )


@pytest.fixture(scope="module")
def fixture():
    A_true, phi_true, target = _truth(seed=0)
    theta_true = _make_params(A_true, phi_true)
    measure = _empirical_measure(target)
    result = k_statistic(theta_true, measure, IIDCovariance(), model=_model)
    return {
        "A_true": A_true,
        "phi_true": phi_true,
        "target": target,
        "theta_true": theta_true,
        "measure": measure,
        "result": result,
    }


class TestQuotientDecomposition:
    """The K-statistic runs on manifold trees with quotient semantics."""

    def test_returns_result_on_manifold_tree(self, fixture):
        # Pre-#41 this died in flatten_params ("all parameter leaves
        # must be 0-d scalars in v1") before any statistic was computed.
        assert isinstance(fixture["result"], KStatisticResult)

    def test_df_quotient(self, fixture):
        r = fixture["result"]
        assert r.df_K == IDENTIFIED_P  # 10, not ambient 11
        assert r.df_S == M_FULL - IDENTIFIED_P  # 6
        assert r.df_J == M_FULL  # 16

    def test_J_equals_K_plus_S(self, fixture):
        r = fixture["result"]
        assert float(r.J) == pytest.approx(float(r.K) + float(r.S), abs=1e-10)

    def test_pvalues_finite(self, fixture):
        r = fixture["result"]
        for pv in (r.p_K, r.p_S, r.p_J):
            assert jnp.isfinite(jnp.asarray(pv))

    def test_runs_on_synthetic_measure(self, fixture):
        """The SyntheticMeasure AD path takes the same ambient flatten."""
        target = fixture["target"]
        noise_key = jax.random.PRNGKey(3)

        def sampler(key, theta):
            del key
            draws = 0.1 * jax.random.normal(noise_key, (300, M_FULL))
            return target[None, :] + draws

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=300, sampler=sampler
        )
        r = k_statistic(
            fixture["theta_true"], measure, SyntheticCovariance(), model=_model
        )
        assert r.df_K == IDENTIFIED_P
        assert jnp.isfinite(jnp.asarray(r.K))


class TestGaugeInvariance:
    """The realised statistic is invariant along the gauge orbit.

    ``Y -> Y O`` (O orthogonal) leaves ``Gamma = Y Y'`` — hence m, V and
    the identified column space of D-tilde — unchanged. The top-p_id SVD
    projection therefore returns the same K bit-stably; the pre-#41
    blind QR could not deliver this (its junk directions were roundoff-
    determined). Both components of O(K) are exercised (rotation and
    reflection).
    """

    @pytest.mark.parametrize("reflect", [False, True])
    def test_K_S_invariant_under_gauge_action(self, fixture, reflect):
        q_gauge = _orthogonal(seed=42, k=K_RANK, reflect=reflect)
        theta_rot = _make_params(fixture["A_true"] @ q_gauge, fixture["phi_true"])
        r0 = fixture["result"]
        r1 = k_statistic(theta_rot, fixture["measure"], IIDCovariance(), model=_model)
        assert float(r1.K) == pytest.approx(float(r0.K), rel=1e-8, abs=1e-10)
        assert float(r1.S) == pytest.approx(float(r0.S), rel=1e-8, abs=1e-10)
        assert float(r1.J) == pytest.approx(float(r0.J), rel=1e-8, abs=1e-10)


class TestGaugeOverride:
    """gauge_nullspace_dim: explicit override vs auto-detection."""

    def test_explicit_override_matches_autodetect(self, fixture):
        r_auto = fixture["result"]
        r_explicit = k_statistic(
            fixture["theta_true"],
            fixture["measure"],
            IIDCovariance(),
            model=_model,
            gauge_nullspace_dim=GAUGE_DIM,
        )
        assert r_explicit.df_K == r_auto.df_K
        assert float(r_explicit.K) == pytest.approx(float(r_auto.K), abs=0.0)

    def test_zero_override_restores_ambient_df(self, fixture):
        """Forcing gauge=0 documents the ambient (pre-#41-style) choice."""
        r = k_statistic(
            fixture["theta_true"],
            fixture["measure"],
            IIDCovariance(),
            model=_model,
            gauge_nullspace_dim=0,
        )
        assert r.df_K == AMBIENT_P
        assert r.df_S == M_FULL - AMBIENT_P

    def test_invalid_override_raises(self, fixture):
        with pytest.raises(ValueError, match="outside"):
            k_statistic(
                fixture["theta_true"],
                fixture["measure"],
                IIDCovariance(),
                model=_model,
                gauge_nullspace_dim=AMBIENT_P + 1,
            )


class TestQuotientUnderIdentificationGuard:
    """The M-vs-p guard compares against p_id, not ambient p (#41).

    With M = 10 = p_id < p = 11 the problem is just-identified ON THE
    QUOTIENT: well-posed (df_S = 0), but the pre-#41 ambient guard
    (M < p) wrongly refused it. With M = 9 < p_id it must refuse.
    """

    def _sliced_measure_and_model(self, fixture, m_keep: int):
        target = fixture["target"][:m_keep]
        measure = _empirical_measure(target, seed=5)
        return measure, _model_sliced(m_keep)

    def test_just_identified_on_quotient_runs(self, fixture):
        measure, model = self._sliced_measure_and_model(fixture, IDENTIFIED_P)
        r = k_statistic(fixture["theta_true"], measure, IIDCovariance(), model=model)
        assert r.df_K == IDENTIFIED_P
        assert r.df_S == 0
        assert jnp.isnan(jnp.asarray(r.p_S))  # df_S == 0 sentinel

    def test_underidentified_on_quotient_refused(self, fixture):
        measure, model = self._sliced_measure_and_model(fixture, IDENTIFIED_P - 1)
        with pytest.raises(ValueError, match="p_id"):
            k_statistic(fixture["theta_true"], measure, IIDCovariance(), model=model)
