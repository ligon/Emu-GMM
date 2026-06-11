r"""Gauge-aware ``cond_info['exclude_gauge']`` — issue #20.

Before #20, ``diagnostics.cond_info['exclude_gauge']`` was a raw alias of
``'raw'``. For a gauge-bearing manifold parameter (``PSDFixedRank(n, K)``:
``gauge_dim = K(K-1)/2``) the information matrix has exactly ``gauge_dim``
spectrally-zero directions BY CONSTRUCTION, so the full-spectrum raw
condition number is meaningless for ``K >= 2`` — it reports the structural
gauge zeros, not identification. The K-Aggregators consumer's
identification analysis needs the pair:

- ``'raw'``: full spectrum, kept for continuity (huge/inf on a gauge
  fixture — that is expected, not a defect);
- ``'exclude_gauge'``: the *quotient* condition number — cond over the
  spectrum EXCLUDING the ``gauge_dim`` smallest eigenvalues BY COUNT, the
  same drop-by-count rule as ``pinv_eigvalrule`` and the #137/#41
  projectors. Any ADDITIONAL near-zero eigenvalues beyond the dropped
  count blow ``exclude_gauge`` up too: genuine structural
  rank-deficiency stays visible (the consumer contract).

Three layers here:

1. Deterministic unit tests on :func:`compute_cond_info` (diagonal info
   matrices where the quotient cond is known exactly, plus the
   static-int / bounds guards mirroring ``pinv_eigvalrule``).
2. End-to-end ``estimate`` on the ``Product(PSDFixedRank(5, 2),
   Euclidean(1))`` gauge fixture (the
   ``tests/manifolds/test_manifold_acceptance_phase6.py`` /
   ``tests/inference/test_k_statistic_gauge.py`` DGP): raw is
   gauge-contaminated (huge), exclude_gauge is finite and small.
3. End-to-end structural-deficiency signal: add a DEAD ``Euclidean(1)``
   leaf the model ignores — one extra exact-zero direction beyond the
   gauge count — and ``exclude_gauge`` must blow up too.

The all-Euclidean path asserts the bitwise ``exclude_gauge == raw`` alias
(v1 non-regression).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.diagnostics import compute_cond_info
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N_SIDE = 5
K_RANK = 2
GAUGE_DIM = K_RANK * (K_RANK - 1) // 2  # 1
M_FULL = N_SIDE * (N_SIDE + 1) // 2 + 1  # 16

_TRIU = jnp.array(np.triu_indices(N_SIDE)).T  # (15, 2)


# ---------------------------------------------------------------------------
# Unit level: compute_cond_info on matrices with a known quotient cond.
# ---------------------------------------------------------------------------
class TestComputeCondInfoQuotient:
    """Deterministic drop-by-count semantics of ``exclude_gauge``."""

    @staticmethod
    def _diag_G(*col_scales: float) -> jnp.ndarray:
        """G with orthogonal columns of the given scales (M = 6 rows).

        With V = I, ``info = G'G = diag(col_scales**2)`` exactly, so the
        full and quotient condition numbers are known in closed form.
        """
        K = len(col_scales)
        G = jnp.zeros((6, K))
        for j, s in enumerate(col_scales):
            G = G.at[j, j].set(s)
        return G

    def test_gauge_zero_dropped_by_count(self):
        """One exact-zero column = the gauge direction: raw is inf,
        exclude_gauge is the cond of the kept block (16/1) exactly."""
        G = self._diag_G(1.0, 2.0, 4.0, 0.0)  # info = diag(1, 4, 16, 0)
        info = compute_cond_info(G, jnp.eye(6), gauge_nullspace_dim=GAUGE_DIM)
        assert not jnp.isfinite(jnp.asarray(info["raw"]))  # gauge-contaminated
        assert float(info["exclude_gauge"]) == pytest.approx(16.0, rel=1e-10)

    def test_extra_zero_beyond_count_blows_up(self):
        """TWO zero directions but gauge_dim = 1: the additional exact
        zero survives the drop and the quotient cond is +inf — the
        structural-rank-deficiency signal the consumer tests for."""
        G = self._diag_G(1.0, 2.0, 0.0, 0.0)  # info = diag(1, 4, 0, 0)
        info = compute_cond_info(G, jnp.eye(6), gauge_nullspace_dim=1)
        assert not jnp.isfinite(jnp.asarray(info["exclude_gauge"]))

    def test_gauge_dim_zero_is_bitwise_alias(self):
        """gauge_nullspace_dim == 0 (the default): exclude_gauge IS raw,
        the same object — the v1 / all-Euclidean path is untouched."""
        G = jnp.array([[1.0, 0.0], [0.5, 1.0], [0.2, 0.3]])
        V = jnp.diag(jnp.array([1.0, 2.0, 0.5]))
        info_default = compute_cond_info(G, V)
        info_explicit = compute_cond_info(G, V, gauge_nullspace_dim=0)
        assert info_default["exclude_gauge"] is info_default["raw"]
        assert info_explicit["exclude_gauge"] is info_explicit["raw"]

    def test_non_int_gauge_dim_raises(self):
        G = jnp.eye(3, 2)
        with pytest.raises(TypeError, match="static Python int"):
            compute_cond_info(G, jnp.eye(3), gauge_nullspace_dim=jnp.asarray(1))

    def test_negative_gauge_dim_raises(self):
        G = jnp.eye(3, 2)
        with pytest.raises(ValueError, match=">= 0"):
            compute_cond_info(G, jnp.eye(3), gauge_nullspace_dim=-1)

    def test_gauge_dim_geq_K_raises(self):
        """Dropping the whole spectrum leaves no identified subspace."""
        G = jnp.eye(3, 2)
        with pytest.raises(ValueError, match="must be < K"):
            compute_cond_info(G, jnp.eye(3), gauge_nullspace_dim=2)


# ---------------------------------------------------------------------------
# Shared gauge-fixture DGP (phase-6 / #41 pattern, K = 2).
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class ProductParams:
    """``PSDFixedRank(5, 2)`` ``Y`` leaf + ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


@jdc.pytree_dataclass
class DeadLeafParams:
    """ProductParams plus a DEAD ``Euclidean(1)`` leaf the model ignores."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf
    dead: ManifoldLeaf


def _gauge_invariant_model(x, theta):
    """psi = (triu(Y Y') ++ phi) - x: gauge-invariant in Y; ignores any
    ``dead`` leaf (works for both param trees above)."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _make_measure(noise: float, n_sim: int, data_seed: int):
    """Synthetic measure: truth + frozen i.i.d. noise (phase-6 pattern)."""
    rng = np.random.default_rng(data_seed)
    A_true = jnp.asarray(rng.normal(size=(N_SIDE, K_RANK)))
    phi_true = 0.7
    g_true = (A_true @ A_true.T)[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])

    noise_key = jax.random.PRNGKey(data_seed)

    def sampler(key, theta):
        del key
        draws = noise * jax.random.normal(noise_key, (n_sim, M_FULL))
        return target[None, :] + draws

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N_SIDE, K_RANK)))
    return measure, A_true, Y0


def _estimate(model, measure, theta_init, *, max_steps: int = 400):
    return estimate(
        model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=max_steps),
        theta_init=theta_init,
    )


@pytest.fixture(scope="module")
def gauge_result():
    measure, _A_true, Y0 = _make_measure(noise=0.01, n_sim=200, data_seed=2)
    theta_init = ProductParams(
        Y=ManifoldLeaf(Y0, PSDFixedRank(N_SIDE, K_RANK)),
        phi=ManifoldLeaf(jnp.asarray([0.65]), Euclidean(1)),
    )
    return _estimate(_gauge_invariant_model, measure, theta_init)


# ---------------------------------------------------------------------------
# End-to-end: the gauge fixture through estimate().
# ---------------------------------------------------------------------------
class TestGaugeFixtureEndToEnd:
    """``Product(PSDFixedRank(5, 2), Euclidean(1))``: gauge_dim = 1.

    The information matrix is (11, 11) with EXACTLY one structural zero
    (the O(2) fibre), so ``raw`` is huge/inf by construction while the
    quotient is genuinely well identified at this DGP (well-separated
    Gamma spectrum, noise = 0.01, n_sim = 200).
    """

    # The gauge zero sits at AD-roundoff scale relative to the largest
    # eigenvalue, so raw is ~1e16 (observed 1.11e16 at this DGP) while
    # the identified (quotient) spread is small (observed 33.4). The
    # thresholds leave orders-of-magnitude headroom on each side of
    # that >14-decade gap, so they fail only on a real regression
    # (raw suddenly clean, or the quotient picking up a near-zero).
    RAW_MIN = 1e8
    QUOTIENT_MAX = 1e6
    SEPARATION_MIN = 1e4  # observed separation ~3e14

    def test_raw_is_gauge_contaminated(self, gauge_result):
        raw = float(jnp.asarray(gauge_result.diagnostics.cond_info["raw"]))
        # inf > RAW_MIN is True; NaN would (rightly) fail.
        assert raw > self.RAW_MIN

    def test_exclude_gauge_finite_and_small(self, gauge_result):
        eg = float(jnp.asarray(gauge_result.diagnostics.cond_info["exclude_gauge"]))
        assert jnp.isfinite(eg)
        assert eg > 1.0  # a condition number
        assert eg < self.QUOTIENT_MAX

    def test_quotient_separates_from_raw(self, gauge_result):
        info = gauge_result.diagnostics.cond_info
        raw = float(jnp.asarray(info["raw"]))
        eg = float(jnp.asarray(info["exclude_gauge"]))
        assert raw / eg > self.SEPARATION_MIN

    def test_gauge_dim_consistent_with_diagnostics(self, gauge_result):
        assert int(gauge_result.diagnostics.gauge_nullspace_dim) == GAUGE_DIM


class TestStructuralDeficiencyEndToEnd:
    """A DEAD Euclidean(1) leaf = one extra exact-zero direction.

    gauge_dim stays 1 (the dead leaf is Euclidean), so the drop-by-count
    rule removes only the O(2) zero; the dead direction's zero SURVIVES
    into the kept spectrum and ``exclude_gauge`` blows up — the
    structural-rank-deficiency signal must NOT be absorbed by the
    gauge drop.
    """

    def test_dead_leaf_blows_up_exclude_gauge(self):
        measure, _A_true, Y0 = _make_measure(noise=0.01, n_sim=200, data_seed=3)
        theta_init = DeadLeafParams(
            Y=ManifoldLeaf(Y0, PSDFixedRank(N_SIDE, K_RANK)),
            phi=ManifoldLeaf(jnp.asarray([0.65]), Euclidean(1)),
            dead=ManifoldLeaf(jnp.asarray([0.0]), Euclidean(1)),
        )
        result = _estimate(_gauge_invariant_model, measure, theta_init)
        info = result.diagnostics.cond_info
        # Still only the one O(2) gauge zero is dropped...
        assert int(result.diagnostics.gauge_nullspace_dim) == GAUGE_DIM
        # ...so the dead direction's exact zero keeps the quotient cond
        # huge/inf (inf > 1e8 is True; NaN would fail, by design).
        eg = float(jnp.asarray(info["exclude_gauge"]))
        assert eg > 1e8


# ---------------------------------------------------------------------------
# End-to-end: the all-Euclidean (v1) path keeps the bitwise alias.
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class ScalarParams:
    a: jax.Array
    b: jax.Array


def _scalar_model(x, theta):
    m = jnp.stack([theta.a, theta.b, theta.a + theta.b])
    return m - x


class TestAllEuclideanAliasEndToEnd:
    """gauge_dim == 0 through estimate(): exclude_gauge == raw exactly."""

    def test_exclude_gauge_equals_raw(self):
        target = jnp.array([0.5, 1.0, 1.5])
        noise_key = jax.random.PRNGKey(7)

        def sampler(key, theta):
            del key
            return target[None, :] + 0.05 * jax.random.normal(noise_key, (200, 3))

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=200, sampler=sampler
        )
        result = estimate(
            _scalar_model,
            measure,
            covariance=SyntheticCovariance(),
            theta_init=ScalarParams(a=jnp.asarray(0.4), b=jnp.asarray(0.9)),
        )
        info = result.diagnostics.cond_info
        assert int(result.diagnostics.gauge_nullspace_dim) == 0
        # Bitwise alias, exactly as pre-#20 (v1 non-regression).
        assert float(info["exclude_gauge"]) == float(info["raw"])
        assert jnp.isfinite(jnp.asarray(info["raw"]))
