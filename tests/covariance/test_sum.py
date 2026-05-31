"""Tests for ``SumCovariance`` and two-way clustering (#81).

The two-way (Cameron--Gelbach--Miller) estimator ``V = V_a + V_b - V_{a∩b}``
is cross-checked against an INDEPENDENT numpy reimplementation (the class is
never used to check itself), plus reduction sanity checks (identical
dimensions / a singleton dimension collapse to one-way clustering) and the
general signed-sum / contract / cached-parity properties.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    SumCovariance,
)
from emu_gmm.types import CovarianceStrategy

TOL = 1e-10


class _MeasureStub:
    def __init__(self, x, mask, weights):
        self.x = jnp.asarray(x, dtype=jnp.float64)
        self.mask = jnp.asarray(mask, dtype=jnp.float64)
        self.weights = jnp.asarray(weights, dtype=jnp.float64)


def identity_psi(xi, theta):
    return xi


# ---------------------------------------------------------------------------
# Independent numpy references
# ---------------------------------------------------------------------------
def _numpy_clustered(psi_vals, mask, weights, ids, n):
    psi_vals = np.asarray(psi_vals, float)
    mask = np.asarray(mask, float)
    weights = np.asarray(weights, float)
    cl = np.rint(np.asarray(ids)).astype(int)
    M = psi_vals.shape[1]
    contrib = mask * weights[:, None] * np.where(mask > 0, psi_vals, 0.0)
    tot = np.zeros((n, M))
    for i in range(len(cl)):
        tot[cl[i]] += contrib[i]
    numer = tot.T @ tot
    Nj = (mask * weights[:, None]).sum(0)
    denom = np.outer(Nj, Nj)
    V = np.zeros((M, M))
    nz = denom != 0.0
    V[nz] = numer[nz] / denom[nz]
    return V


def _numpy_two_way(psi_vals, mask, weights, a, na, b, nb):
    Va = _numpy_clustered(psi_vals, mask, weights, a, na)
    Vb = _numpy_clustered(psi_vals, mask, weights, b, nb)
    ab = np.rint(a).astype(int) * nb + np.rint(b).astype(int)
    Vab = _numpy_clustered(psi_vals, mask, weights, ab, na * nb)
    V = Va + Vb - Vab
    return 0.5 * (V + V.T)


# ---------------------------------------------------------------------------
# Fixture: cross-cutting dimensions a (4) x b (3), with per-dimension common
# components so the two-way V genuinely differs from either one-way V.
# ---------------------------------------------------------------------------
def build_two_way(seed=20):
    rng = np.random.default_rng(seed)
    na, nb, per_cell, M = 4, 3, 2, 2
    a_comp = rng.normal(size=(na, M))
    b_comp = rng.normal(size=(nb, M))
    rows_x, a_ids, b_ids = [], [], []
    for ai in range(na):
        for bi in range(nb):
            for _ in range(per_cell):
                rows_x.append(a_comp[ai] + b_comp[bi] + rng.normal(size=M))
                a_ids.append(ai)
                b_ids.append(bi)
    x = np.array(rows_x)
    n = len(rows_x)
    return dict(
        x=x,
        mask=np.ones((n, M)),
        weights=np.ones(n),
        a=np.array(a_ids, float),
        na=na,
        b=np.array(b_ids, float),
        nb=nb,
        n=n,
    )


def _two_way(fx):
    return SumCovariance.two_way_cluster(
        jnp.asarray(fx["a"]), fx["na"], jnp.asarray(fx["b"]), fx["nb"]
    )


def _clustered(ids, n):
    return ClusteredCovariance(cluster_ids=jnp.asarray(ids), n_clusters=n)


# ===========================================================================


def test_two_way_matches_numpy_cgm_and_is_symmetric():
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    V = np.asarray(_two_way(fx).covariance(identity_psi, None, meas))
    V_ref = _numpy_two_way(
        fx["x"], fx["mask"], fx["weights"], fx["a"], fx["na"], fx["b"], fx["nb"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL
    assert np.max(np.abs(V - V.T)) < 1e-12


def test_two_way_differs_from_one_way():
    """Two-way captures both dimensions' correlation -> differs from either
    one-way V (else the composition would be pointless).
    """
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    V_two = np.asarray(_two_way(fx).covariance(identity_psi, None, meas))
    V_a = np.asarray(_clustered(fx["a"], fx["na"]).covariance(identity_psi, None, meas))
    V_b = np.asarray(_clustered(fx["b"], fx["nb"]).covariance(identity_psi, None, meas))
    assert np.max(np.abs(V_two - V_a)) > 1e-6
    assert np.max(np.abs(V_two - V_b)) > 1e-6


def test_identical_dimensions_reduce_to_one_way():
    """a == b: V_a + V_a - V_{a∩a} = V_a."""
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    cov = SumCovariance.two_way_cluster(
        jnp.asarray(fx["a"]), fx["na"], jnp.asarray(fx["a"]), fx["na"]
    )
    V = np.asarray(cov.covariance(identity_psi, None, meas))
    V_a = np.asarray(_clustered(fx["a"], fx["na"]).covariance(identity_psi, None, meas))
    assert np.max(np.abs(V - V_a)) < TOL


def test_singleton_second_dimension_reduces_to_one_way_a():
    """b = each obs its own cluster: V_b and V_{a∩b} are both the IID form,
    so V = V_a + V_iid - V_iid = V_a.
    """
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    b_singleton = np.arange(fx["n"], dtype=float)
    cov = SumCovariance.two_way_cluster(
        jnp.asarray(fx["a"]), fx["na"], jnp.asarray(b_singleton), fx["n"]
    )
    V = np.asarray(cov.covariance(identity_psi, None, meas))
    V_a = np.asarray(_clustered(fx["a"], fx["na"]).covariance(identity_psi, None, meas))
    assert np.max(np.abs(V - V_a)) < TOL


def test_general_signed_sum():
    """Raw SumCovariance(terms, signs): V = V_a - V_b, symmetrised."""
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    cov_a = _clustered(fx["a"], fx["na"])
    cov_b = _clustered(fx["b"], fx["nb"])
    s = SumCovariance(terms=(cov_a, cov_b), signs=(1.0, -1.0))
    V = np.asarray(s.covariance(identity_psi, None, meas))
    Va = np.asarray(cov_a.covariance(identity_psi, None, meas))
    Vb = np.asarray(cov_b.covariance(identity_psi, None, meas))
    expected = Va - Vb
    expected = 0.5 * (expected + expected.T)
    assert np.max(np.abs(V - expected)) < TOL


def test_cached_self_parity():
    fx = build_two_way()
    meas = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    cov = _two_way(fx)
    cached = meas.expectation_and_contributions(identity_psi, None)
    V_self = np.asarray(cov.covariance(identity_psi, None, meas))
    V_cached = np.asarray(
        cov.covariance(identity_psi, None, meas, cached_intermediates=cached)
    )
    assert np.array_equal(V_self, V_cached)


def test_protocol_pytree_and_jit():
    fx = build_two_way()
    cov = _two_way(fx)
    assert isinstance(cov, CovarianceStrategy)
    # signs are static -> NOT pytree leaves; the child cluster_ids are leaves.
    leaves = jax.tree_util.tree_leaves(cov)
    assert all(np.asarray(leaf).dtype != bool for leaf in leaves)
    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    fn = jax.jit(lambda m: cov.covariance(identity_psi, None, m))
    V = np.asarray(fn(measure))
    assert np.isfinite(V).all()


def test_length_mismatch_raises():
    """A terms/signs length mismatch fails loudly (strict zip), not silently
    dropping a term.
    """
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    bad = SumCovariance(
        terms=(_clustered(fx["a"], fx["na"]), _clustered(fx["b"], fx["nb"])),
        signs=(1.0,),  # one sign for two terms
    )
    with pytest.raises(ValueError):
        bad.covariance(identity_psi, None, meas)


def test_signed_difference_can_be_indefinite():
    """A signed sum can be indefinite (e.g. the difference V_a - V_b of two
    PSD cluster matrices, generically indefinite). The routine deliberately
    does NOT repair it -- DiagonalTikhonov does. Lock the 'no internal PD
    repair' contract.
    """
    min_eig = np.inf
    for seed in range(8):
        fx = build_two_way(seed=seed)
        meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
        s = SumCovariance(
            terms=(_clustered(fx["a"], fx["na"]), _clustered(fx["b"], fx["nb"])),
            signs=(1.0, -1.0),
        )
        V = np.asarray(s.covariance(identity_psi, None, meas))
        assert np.isfinite(V).all()
        min_eig = min(min_eig, float(np.linalg.eigvalsh(V).min()))
    assert min_eig < -1e-9


def test_dof_correction_forwarded_to_all_terms():
    fx = build_two_way()
    meas = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    args = (jnp.asarray(fx["a"]), fx["na"], jnp.asarray(fx["b"]), fx["nb"])
    cov0 = SumCovariance.two_way_cluster(*args, dof_correction=False)
    cov1 = SumCovariance.two_way_cluster(*args, dof_correction=True)
    assert all(t.dof_correction for t in cov1.terms)  # forwarded to every term
    assert not any(t.dof_correction for t in cov0.terms)
    V0 = np.asarray(cov0.covariance(identity_psi, None, meas))
    V1 = np.asarray(cov1.covariance(identity_psi, None, meas))
    assert np.max(np.abs(V1 - V0)) > 1e-6  # correction actually changes V
