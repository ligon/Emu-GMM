r"""Tests for :func:`emu_gmm.inference.identification_strength` (#177).

Coverage:

(a) Schur-complement math + the block-inverse identity. The per-block
    partial information :math:`\mathcal I_{b\cdot c}` matches a direct numpy
    Schur complement, and its eigenvalues equal the reciprocals of the
    eigenvalues of the block-:math:`b` sub-block of :math:`\mathcal I^{-1}`
    (the identity that ties the concentration parameter to the marginal Wald
    variance).

(b) Flags the weak block on a designed weak-instrument DGP: the
    weakly-identified coordinate has the smallest concentration parameter and
    is reported as ``.weakest``.

(c) Agrees with the K-statistic's robust-vs-Wald divergence: the flagged
    block's concentration is in the weak-identification regime
    (``min_eigenvalue < 1``) while the strong block is far above it, and the
    Wald-SE inflation is localised to the same block — the regime where the
    identification-robust K-statistic is needed and the Wald interval is
    unreliable. The block-inverse identity makes the SE/curvature tie exact.

(d) Composes with a manifold ``PSDFixedRank`` block: the gauge directions are
    dropped by count (``dim == ambient - gauge_dim``), the reported spectrum
    is finite, and a block per leaf is returned.

(e) Validation: out-of-range / overlapping / empty custom blocks, and a
    custom block that splits a gauge-bearing leaf, all raise.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    ContinuouslyUpdated,
    EmpiricalMeasure,
    IIDCovariance,
    estimate,
    identification_strength,
    k_statistic,
)
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.inference.identification import (
    BlockStrength,
    IdentificationStrength,
    _block_strength,
    _leaf_index_ranges,
)
from emu_gmm.optimizer import optimistix_lm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# (a) Schur math + block-inverse identity (unit, deterministic).
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _FourScalar:
    a: float
    b: float
    c: float
    d: float


def _euclidean_leaf_ranges(n: int):
    """``_leaf_index_ranges`` for an ``n``-scalar (all-Euclidean) tree."""
    params = _FourScalar(a=0.0, b=0.0, c=0.0, d=0.0)
    assert n == 4
    return _leaf_index_ranges(manifold_spec_from_params(params))


class TestSchurMath:
    def _spd(self, seed: int, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, n))
        return A @ A.T + n * np.eye(n)  # well-conditioned SPD

    def test_partial_information_matches_numpy_schur(self):
        info = jnp.asarray(self._spd(0, 4))
        leaf_ranges = _euclidean_leaf_ranges(4)
        # Block {0,1}, complement {2,3}.
        blk = _block_strength("blk", (0, 1), info, leaf_ranges, total_dim=4)
        Imat = np.asarray(info)
        b, c = [0, 1], [2, 3]
        schur = (
            Imat[np.ix_(b, b)]
            - Imat[np.ix_(b, c)]
            @ np.linalg.inv(Imat[np.ix_(c, c)])
            @ Imat[np.ix_(c, b)]
        )
        np.testing.assert_allclose(
            np.asarray(blk.partial_information), schur, rtol=1e-10, atol=1e-10
        )

    def test_block_inverse_identity(self):
        # eig(I_{b|c}) == 1 / eig( (I^{-1})_{bb} ): the concentration
        # parameters are the reciprocals of the marginal-covariance spectrum
        # (the exact tie between min_eigenvalue and the Wald block variance).
        info = jnp.asarray(self._spd(3, 4))
        leaf_ranges = _euclidean_leaf_ranges(4)
        blk = _block_strength("blk", (0, 1), info, leaf_ranges, total_dim=4)
        I_inv = np.linalg.inv(np.asarray(info))
        sub = I_inv[np.ix_([0, 1], [0, 1])]
        recip = np.sort(1.0 / np.linalg.eigvalsh(sub))  # ascending
        got = np.sort(np.asarray(blk.eigenvalues))
        np.testing.assert_allclose(got, recip, rtol=1e-9, atol=1e-9)

    def test_whole_parameter_block_is_full_information(self):
        # No complement -> partial information IS the full info matrix.
        info = jnp.asarray(self._spd(7, 3))

        @jdc.pytree_dataclass
        class _Three:
            a: float
            b: float
            c: float

        lr = _leaf_index_ranges(manifold_spec_from_params(_Three(0.0, 0.0, 0.0)))
        blk = _block_strength("all", (0, 1, 2), info, lr, total_dim=3)
        np.testing.assert_allclose(
            np.asarray(blk.partial_information), np.asarray(info), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(blk.eigenvalues)),
            np.sort(np.linalg.eigvalsh(np.asarray(info))),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Weak-instrument DGP fixture: theta_s strong, theta_w weak.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class IVParams:
    theta_s: float
    theta_w: float


def _iv_model(x, theta):
    y = x[0]
    a = x[1]
    b = x[2]
    z = x[3:6]
    return z * (y - theta.theta_s * a - theta.theta_w * b)


def _fit_weak_iv(seed: int = 1, n: int = 4000):
    """Strong regressor ``a`` (instruments correlated), weak regressor ``b``."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    pi_a = np.array([1.4, 1.2, 1.1])  # strong first stage
    pi_b = np.array([0.015, 0.012, 0.010])  # genuinely weak first stage
    a = Z @ pi_a + rng.normal(size=n) * 0.4
    b = Z @ pi_b + rng.normal(size=n) * 0.4
    ts0, tw0 = 1.5, -0.7
    y = ts0 * a + tw0 * b + rng.normal(size=n) * 0.3
    X = np.column_stack([y, a, b, Z])
    measure = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    result = estimate(
        _iv_model,
        measure,
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        theta_init=IVParams(theta_s=ts0, theta_w=tw0),  # warm start
    )
    return result


# ---------------------------------------------------------------------------
# (b) Flags the weak block.
# ---------------------------------------------------------------------------


class TestFlagsWeakBlock:
    def test_weakest_is_the_weak_coordinate(self):
        result = _fit_weak_iv()
        assert bool(result.converged)
        ident = identification_strength(result, _iv_model)
        assert isinstance(ident, IdentificationStrength)
        assert ident.weakest == "theta_w"

    def test_concentration_ordering_and_one_block_per_leaf(self):
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model)
        assert set(ident.blocks) == {"theta_s", "theta_w"}
        strong = float(ident["theta_s"].min_eigenvalue)
        weak = float(ident["theta_w"].min_eigenvalue)
        # The strong coordinate's concentration dwarfs the weak one's.
        assert strong > 100.0 * weak
        # Each scalar leaf is a 1-D, gauge-free block.
        for name in ("theta_s", "theta_w"):
            assert ident[name].dim == 1
            assert ident[name].gauge_dim == 0
            assert ident[name].eigenvalues.shape == (1,)


# ---------------------------------------------------------------------------
# (c) Agrees with the K-statistic's robust-vs-Wald divergence on the weak
#     block: the flagged block is in the weak-ID regime (concentration < 1),
#     the Wald SE inflation is localised there, and the concentration is
#     exactly the reciprocal of the marginal Wald variance (block-inverse).
# ---------------------------------------------------------------------------


class TestRobustVsWaldDivergence:
    def test_weak_block_in_weak_regime_and_se_inflation_localised(self):
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model)
        se = np.asarray(result.standard_errors.array)  # [theta_s, theta_w]

        weak_conc = float(ident["theta_w"].min_eigenvalue)
        strong_conc = float(ident["theta_s"].min_eigenvalue)
        # Weak block sits below the conventional weak-ID threshold; strong far
        # above. This is precisely the regime where the identification-robust
        # K-statistic is required and the Wald interval is unreliable.
        assert weak_conc < 1.0
        assert strong_conc > 100.0
        # Wald-SE inflation is localised to the flagged (weak) block.
        se_strong, se_weak = float(se[0]), float(se[1])
        assert se_weak > 10.0 * se_strong

    def test_block_inverse_identity_against_sigma_theta(self):
        # For the efficient CU fit (no binding ridge) Sigma_theta == B^{-1},
        # so each 1-D block's concentration is exactly 1 / diag(Sigma_theta) --
        # the exact curvature/Wald-variance tie behind the divergence claim.
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model)
        Sigma = np.asarray(result.Sigma_theta.array)
        for i, name in enumerate(("theta_s", "theta_w")):
            np.testing.assert_allclose(
                float(ident[name].min_eigenvalue), 1.0 / Sigma[i, i], rtol=1e-6
            )

    def test_k_statistic_available_as_the_robust_alternative(self):
        # Sanity: the robust statistic this diagnostic points the user toward
        # is finite/valid at theta_hat (m ~ 0 there -> K ~ 0).
        result = _fit_weak_iv()
        ks = k_statistic(result, result.measure, IIDCovariance(), _iv_model)
        assert bool(jnp.isfinite(ks.K))
        assert bool(jnp.isfinite(ks.p_K))


# ---------------------------------------------------------------------------
# (d) Composes with a PSDFixedRank manifold block (gauge-aware).
# ---------------------------------------------------------------------------


def _load_phase4_fixture():
    """Reuse the Phase-4 Product(PSDFixedRank(5,k), Euclidean(1)) estimate."""
    manifolds_dir = Path(__file__).resolve().parents[1] / "manifolds"
    if str(manifolds_dir) not in sys.path:
        sys.path.insert(0, str(manifolds_dir))
    import test_estimator_inference_phase4 as ph4

    return ph4


@pytest.mark.parametrize("k", [2, 3])
class TestManifoldComposition:
    def test_gauge_dropped_by_count_and_spectrum_finite(self, k):
        ph4 = _load_phase4_fixture()
        result, spec, _M, _ = ph4._run_estimate(k, seed=200 + k)
        ident = identification_strength(result, ph4._model)
        gauge = k * (k - 1) // 2
        ambient_psd = ph4.N * k

        # A block per leaf: the PSDFixedRank factor 'Y' and the scalar 'phi'.
        assert set(ident.blocks) == {"Y", "phi"}
        Yblk = ident["Y"]
        # The Y block drops EXACTLY its gauge_dim = k(k-1)/2 directions by
        # count; the identified dimension is ambient - gauge.
        assert Yblk.gauge_dim == gauge
        assert Yblk.dim == ambient_psd - gauge
        assert Yblk.eigenvalues.shape == (ambient_psd - gauge,)
        # No inf/nan: the gauge zeros were dropped, not inverted.
        assert bool(jnp.all(jnp.isfinite(Yblk.eigenvalues)))
        # phi is a strongly-identified gauge-free scalar.
        assert ident["phi"].gauge_dim == 0
        assert ident["phi"].dim == 1
        assert bool(jnp.all(jnp.isfinite(ident["phi"].eigenvalues)))

    def test_split_gauge_leaf_is_refused(self, k):
        ph4 = _load_phase4_fixture()
        result, spec, _M, _ = ph4._run_estimate(k, seed=210 + k)
        # A custom block that takes only the first column of the PSD leaf
        # splits its gauge fibre -> no well-defined per-block gauge count.
        with pytest.raises(ValueError, match="split"):
            identification_strength(result, ph4._model, blocks={"half": [0]})


# ---------------------------------------------------------------------------
# (e) Validation + API surface.
# ---------------------------------------------------------------------------


class TestValidationAndApi:
    def test_custom_blocks_partition(self):
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model, blocks={"both": [0, 1]})
        assert set(ident.blocks) == {"both"}
        assert ident["both"].dim == 2
        assert ident["both"].eigenvalues.shape == (2,)

    def test_index_out_of_range(self):
        result = _fit_weak_iv()
        with pytest.raises(ValueError, match="outside"):
            identification_strength(result, _iv_model, blocks={"x": [0, 5]})

    def test_overlapping_blocks(self):
        result = _fit_weak_iv()
        with pytest.raises(ValueError, match="disjoint"):
            identification_strength(result, _iv_model, blocks={"x": [0], "y": [0, 1]})

    def test_empty_block(self):
        result = _fit_weak_iv()
        with pytest.raises(ValueError, match="empty"):
            identification_strength(result, _iv_model, blocks={"x": []})

    def test_to_pandas_and_repr(self):
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model)
        df = ident.to_pandas()
        assert list(df.index) == ["theta_s", "theta_w"]
        assert {"dim", "gauge_dim", "min_eigenvalue", "max_eigenvalue"} <= set(
            df.columns
        )
        # min <= max within each block.
        assert bool((df["min_eigenvalue"] <= df["max_eigenvalue"]).all())
        # The metric note documents the efficient (V*)^{-1} weighting.
        assert "V*" in ident.metric

    def test_block_strength_is_frozen(self):
        result = _fit_weak_iv()
        ident = identification_strength(result, _iv_model)
        blk = ident["theta_s"]
        assert isinstance(blk, BlockStrength)
        with pytest.raises(dataclasses.FrozenInstanceError):
            blk.name = "nope"
