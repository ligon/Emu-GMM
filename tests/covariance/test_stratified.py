"""Tests for ``StratifiedCovariance`` (#79) and ``DesignAwareCovariance`` (#80).

The numerical core is cross-checked against an INDEPENDENT plain-numpy
reimplementation of the per-pair between-PSU Neyman covariance
(``_numpy_vtt`` below): the JAX class is never used to check itself.

Fixtures (the union of the external ``validate_vtt`` set and the gaps the
design red-team / Seasonality review surfaced):

- F1 Neyman recovery (complete data): diagonal == S^2_arm / H_arm.
- F2 shared control: off-diagonal Cov(tau_C, tau_M) == S^2_0 / H_0 > 0.
- F3 per-coordinate masking: degenerate pair -> exactly 0; support drop.
- F4 co-observed (NEW): a proper, non-degenerate pair-overlap subset, with
  a discriminating check that per-pair centering != per-coordinate centering.
- F5 p-weighted (NEW): heterogeneous w_i AND unequal N_j -- the
  weight-once-not-w^2 guard, invisible to every w==1 fixture.
- F6 FPC indicator (NEW): a single-arm coordinate with fpc=True.

Plus: CovarianceStrategy conformance, cached-vs-self-compute parity on a
D != M real measure with NaN feeding a non-masked moment, gradient
AD-safety with a singular psi, and the #80 DesignAwareCovariance assembly
(all-design bit-for-bit reduction; mixed V_TT + V_SS + estimated V_TS).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    DesignAwareCovariance,
    EmpiricalMeasure,
    StratifiedCovariance,
)
from emu_gmm.types import CovarianceStrategy

TOL = 1e-10


# ---------------------------------------------------------------------------
# Minimal measure stub matching the CovarianceStrategy contract.
# ---------------------------------------------------------------------------
class _MeasureStub:
    def __init__(self, x, mask, weights):
        self.x = jnp.asarray(x, dtype=jnp.float64)
        self.mask = jnp.asarray(mask, dtype=jnp.float64)
        self.weights = jnp.asarray(weights, dtype=jnp.float64)


def identity_psi(xi, theta):
    """psi_j(x_i) = x_{ij}: the residual is just the data coordinate."""
    return xi


# ---------------------------------------------------------------------------
# Independent numpy reference: per-pair between-PSU Neyman covariance.
# From-scratch; NOT a call into the class.
# ---------------------------------------------------------------------------
def _numpy_vtt(
    psi_vals,
    mask,
    weights,
    psu_ids,
    cell_ids,
    n_psu,
    *,
    fpc=False,
    stratum_of_psu=None,
):
    psi_vals = np.asarray(psi_vals, float)
    mask = np.asarray(mask, float)
    weights = np.asarray(weights, float)
    psu_ids = np.rint(np.asarray(psu_ids)).astype(int)
    cell_ids = np.rint(np.asarray(cell_ids)).astype(int)
    N, M = psi_vals.shape

    Nj = (mask * weights[:, None]).sum(0)
    psi_safe = np.where(mask > 0, psi_vals, 0.0)
    contrib = mask * weights[:, None] * psi_safe
    sup_unit = (mask > 0).astype(float)

    t = np.zeros((n_psu, M))
    sup = np.zeros((n_psu, M))
    cell_of = np.full(n_psu, -1, int)
    for i in range(N):
        g = psu_ids[i]
        t[g] += contrib[i]
        sup[g] += sup_unit[i]
        cell_of[g] = cell_ids[i]
    s = (sup > 0).astype(float)

    if fpc:
        strat_of_psu = np.rint(np.asarray(stratum_of_psu)).astype(int)
        pop = np.where(cell_of >= 0)[0]  # populated PSUs
        H_s_by_strat = {}
        for g in pop:
            H_s_by_strat[strat_of_psu[g]] = H_s_by_strat.get(strat_of_psu[g], 0) + 1

    numer = np.zeros((M, M))
    for c in np.unique(cell_of[cell_of >= 0]):
        gs = np.where(cell_of == c)[0]
        # Convention (ii): a single coordinate-INDEPENDENT FPC scalar per cell,
        # 1 - H_{sD,c}/H_s, with H_{sD,c} = populated PSUs in the cell.
        fpc_c = 1.0
        if fpc:
            strat_c = strat_of_psu[gs[0]]
            fpc_c = 1.0 - len(gs) / H_s_by_strat[strat_c]
        for j in range(M):
            for k in range(M):
                u = s[gs, j] * s[gs, k]
                H = u.sum()
                if H < 2:
                    continue
                A = (u * t[gs, j]).sum()
                B = (u * t[gs, k]).sum()
                P = (u * t[gs, j] * t[gs, k]).sum()
                term = (H / (H - 1.0)) * (P - A * B / H)
                numer[j, k] += term * fpc_c

    V = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            d = Nj[j] * Nj[k]
            V[j, k] = numer[j, k] / d if d != 0.0 else 0.0
    return 0.5 * (V + V.T)


def _run_impl(fx, fpc=False):
    strat = StratifiedCovariance(
        psu_ids=jnp.asarray(fx["psu_ids"], dtype=jnp.float64),
        cell_ids=jnp.asarray(fx["cell_ids"], dtype=jnp.float64),
        stratum_ids=jnp.asarray(fx["stratum_ids"], dtype=jnp.float64),
        n_psu=int(fx["n_psu"]),
        n_cells=int(fx["n_cells"]),
        n_strata=int(fx["n_strata"]),
        fpc=fpc,
    )
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    return np.asarray(strat.covariance(identity_psi, None, measure))


def _neyman_arm_var(psu_totals_by_stratum):
    """sum_s H/(H-1) sum_g (t_g - tbar_s)^2 over PSU totals; caller / N^2."""
    total = 0.0
    for arr in psu_totals_by_stratum:
        t = np.asarray(arr, float)
        H = len(t)
        if H < 2:
            continue
        total += (H / (H - 1.0)) * ((t - t.mean()) ** 2).sum()
    return total


# ===========================================================================
# Fixture builders
# ===========================================================================
def build_full(seed=1):
    """F1: 3 strata x {C,M,0}, 2 PSUs/arm/stratum, identity psi, arm masks."""
    rng = np.random.default_rng(seed)
    S, arms, Hpa, upp = 3, ["C", "M", "0"], 2, 4
    arm_idx = {"C": 0, "M": 1, "0": 2}
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    psu_tot = {a: [[] for _ in range(S)] for a in arms}
    g = c = 0
    for s in range(S):
        for a in arms:
            cell = c
            c += 1
            for _ in range(Hpa):
                tot = 0.0
                for v in rng.normal(arm_idx[a], 1.5, upp):
                    xr = np.zeros(3)
                    mr = np.zeros(3)
                    xr[arm_idx[a]] = v
                    mr[arm_idx[a]] = 1.0
                    rows_x.append(xr)
                    rows_mask.append(mr)
                    weights.append(1.0)
                    cell_ids.append(cell)
                    psu_ids.append(g)
                    tot += v
                psu_tot[a][s].append(tot)
                g += 1
    cell_ids = np.array(cell_ids)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=cell_ids,
        psu_ids=np.array(psu_ids),
        stratum_ids=cell_ids // 3,
        n_psu=g,
        n_cells=c,
        n_strata=S,
        psu_tot=psu_tot,
        arm_idx=arm_idx,
    )


def build_shared(seed=2):
    """F2: coords [C, M, CTRL]; control units feed the shared CTRL coord."""
    rng = np.random.default_rng(seed)
    S, Hpa, upp = 3, 2, 4
    arm_idx = {"C": 0, "M": 1}
    ctrl = 2
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    ctrl_tot = [[] for _ in range(S)]
    g = c = 0
    for s in range(S):
        for a in ["C", "M", "0"]:
            cell = c
            c += 1
            for _ in range(Hpa):
                tot = 0.0
                for v in rng.normal(0.0, 1.5, upp):
                    xr = np.zeros(3)
                    mr = np.zeros(3)
                    col = arm_idx.get(a, ctrl)
                    xr[col] = v
                    mr[col] = 1.0
                    rows_x.append(xr)
                    rows_mask.append(mr)
                    weights.append(1.0)
                    cell_ids.append(cell)
                    psu_ids.append(g)
                    tot += v
                if a == "0":
                    ctrl_tot[s].append(tot)
                g += 1
    cell_ids = np.array(cell_ids)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=cell_ids,
        psu_ids=np.array(psu_ids),
        stratum_ids=cell_ids // 3,
        n_psu=g,
        n_cells=c,
        n_strata=S,
        arm_idx=arm_idx,
        ctrl=ctrl,
        ctrl_tot=ctrl_tot,
    )


def build_masked(seed=3):
    """F3: coords 0,1 full; coord 2 support 3->2 in cell 0; coord 3 degenerate."""
    rng = np.random.default_rng(seed)
    M, ncell, ppc, upp = 4, 2, 3, 3
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    g = 0
    for cell in range(ncell):
        for h in range(ppc):
            for _ in range(upp):
                mr = np.ones(M)
                if (cell, h) == (0, 0):
                    mr[2] = 0.0  # coord 2: one PSU fully masked in cell 0
                if (cell, h) != (0, 0):
                    mr[3] = 0.0  # coord 3: supported by exactly one PSU
                rows_x.append(rng.normal(size=M))
                rows_mask.append(mr)
                weights.append(1.0)
                cell_ids.append(cell)
                psu_ids.append(g)
            g += 1
    cell_ids = np.array(cell_ids)
    fx = dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=cell_ids,
        psu_ids=np.array(psu_ids),
        stratum_ids=cell_ids.copy(),
        n_psu=g,
        n_cells=ncell,
        n_strata=ncell,
    )
    fx_un = {**fx, "mask": np.ones_like(fx["mask"])}
    return fx, fx_un


def build_co_observed(seed=4):
    """F4: one cell, 4 PSUs; coord 0 in {0,1,2}, coord 1 in {1,2,3}.

    Overlap {1,2} has H=2 (non-degenerate, proper subset). The (0,1) entry
    must center coord 0 over {1,2} ONLY -- not over its own support {0,1,2}.
    """
    rng = np.random.default_rng(seed)
    upp = 3
    supp = {0: {0, 1, 2}, 1: {1, 2, 3}}
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    for g in range(4):
        for _ in range(upp):
            mr = np.array([1.0 if g in supp[0] else 0.0, 1.0 if g in supp[1] else 0.0])
            rows_x.append(rng.normal(size=2))
            rows_mask.append(mr)
            weights.append(1.0)
            cell_ids.append(0)
            psu_ids.append(g)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=np.array(cell_ids),
        psu_ids=np.array(psu_ids),
        stratum_ids=np.zeros(len(psu_ids)),
        n_psu=4,
        n_cells=1,
        n_strata=1,
    )


def build_pweighted(seed=5):
    """F5: heterogeneous weights AND unequal N_j (partial mask)."""
    rng = np.random.default_rng(seed)
    M, ncell, ppc, upp = 2, 2, 3, 3
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    g = 0
    for cell in range(ncell):
        for h in range(ppc):
            for u in range(upp):
                mr = np.ones(M)
                if (h + u) % 3 == 0:
                    mr[1] = 0.0  # knock coord 1 out unevenly -> unequal N_j
                rows_x.append(rng.normal(size=M))
                rows_mask.append(mr)
                weights.append(float(1 + (g % 3)))  # w in {1,2,3}
                cell_ids.append(cell)
                psu_ids.append(g)
            g += 1
    cell_ids = np.array(cell_ids)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights, float),
        cell_ids=cell_ids,
        psu_ids=np.array(psu_ids),
        stratum_ids=cell_ids.copy(),
        n_psu=g,
        n_cells=ncell,
        n_strata=ncell,
    )


def build_fpc_indicator(seed=6):
    """F6: 2 strata, 3 arms, plus a coordinate observed in a single arm only."""
    rng = np.random.default_rng(seed)
    S, Hpa, upp = 2, 2, 3
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    g = c = 0
    for _s in range(S):
        for a in range(3):  # 3 arms
            cell = c
            c += 1
            for _ in range(Hpa):
                for v in rng.normal(size=upp):
                    mr = np.zeros(3)
                    xr = np.zeros(3)
                    mr[a] = 1.0  # coord a = this arm's outcome
                    xr[a] = v
                    # coord 2 also acts as a single-arm indicator (arm 2 only)
                    rows_x.append(xr)
                    rows_mask.append(mr)
                    weights.append(1.0)
                    cell_ids.append(cell)
                    psu_ids.append(g)
                g += 1
    cell_ids = np.array(cell_ids)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=cell_ids,
        psu_ids=np.array(psu_ids),
        stratum_ids=cell_ids // 3,
        n_psu=g,
        n_cells=c,
        n_strata=S,
    )


# ===========================================================================
# F1 -- Neyman recovery
# ===========================================================================
def test_f1_matches_numpy_reference():
    fx = build_full()
    V = _run_impl(fx)
    V_ref = _numpy_vtt(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["cell_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL


def test_f1_diagonal_is_neyman_variance():
    fx = build_full()
    V = _run_impl(fx)
    for arm, j in fx["arm_idx"].items():
        numer = _neyman_arm_var(fx["psu_tot"][arm])
        N_arm = (fx["mask"][:, j] * fx["weights"]).sum()
        assert abs(V[j, j] - numer / N_arm**2) < TOL


def test_f1_symmetric_and_psd():
    V = _run_impl(build_full())
    assert np.max(np.abs(V - V.T)) < 1e-12
    assert np.linalg.eigvalsh(V).min() >= -1e-10


# ===========================================================================
# F2 -- shared-control off-diagonal
# ===========================================================================
def test_f2_shared_control_covariance_positive():
    fx = build_shared()
    V = _run_impl(fx)
    V_ref = _numpy_vtt(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["cell_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL

    aC, aM, c4 = fx["arm_idx"]["C"], fx["arm_idx"]["M"], fx["ctrl"]
    cov = V[aC, aM] - V[aC, c4] - V[c4, aM] + V[c4, c4]
    Nc4 = (fx["mask"][:, c4] * fx["weights"]).sum()
    cov_ref = _neyman_arm_var(fx["ctrl_tot"]) / Nc4**2
    assert abs(cov - cov_ref) < TOL
    assert cov > 0.0


# ===========================================================================
# F3 -- per-coordinate masking
# ===========================================================================
def test_f3_matches_numpy_reference():
    fx, _ = build_masked()
    V = _run_impl(fx)
    V_ref = _numpy_vtt(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["cell_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL


def test_f3_degenerate_pair_is_exactly_zero_and_finite():
    fx, _ = build_masked()
    V = _run_impl(fx)
    assert np.isfinite(V).all()
    assert V[0, 3] == 0.0  # coord 3 supported by a single PSU -> H<2 everywhere
    assert V[3, 3] == 0.0


def test_f3_fully_observed_subblock_unchanged_by_masking_elsewhere():
    fx, fx_un = build_masked()
    V = _run_impl(fx)
    V_un = _run_impl(fx_un)
    fo = np.ix_([0, 1], [0, 1])
    assert np.max(np.abs(V[fo] - V_un[fo])) < TOL


# ===========================================================================
# F4 -- co-observed, non-degenerate proper pair-overlap (per-pair != per-coord)
# ===========================================================================
def test_f4_matches_numpy_reference():
    fx = build_co_observed()
    V = _run_impl(fx)
    V_ref = _numpy_vtt(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["cell_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL


def test_f4_per_pair_centering_differs_from_per_coordinate():
    """The fixture is discriminating: a per-COORDINATE centering of the
    (0,1) entry (centering coord 0 over its own support {0,1,2} rather than
    over the pair-overlap {1,2}) gives a materially different value. Guards
    against silently dropping the per-pair H_{c,jk} back to per-coordinate.
    """
    fx = build_co_observed()
    V = _run_impl(fx)

    # Reconstruct PSU totals/supports to build the "wrong" per-coordinate off-diag.
    mask, w, x = fx["mask"], fx["weights"], fx["x"]
    psu = np.rint(fx["psu_ids"]).astype(int)
    contrib = mask * w[:, None] * np.where(mask > 0, x, 0.0)
    t = np.zeros((fx["n_psu"], 2))
    s = np.zeros((fx["n_psu"], 2))
    for i in range(len(psu)):
        t[psu[i]] += contrib[i]
        s[psu[i]] += (mask[i] > 0).astype(float)
    s = (s > 0).astype(float)
    Nj = (mask * w[:, None]).sum(0)

    # WRONG: center each coord over its OWN support, then cross only the overlap.
    gj = np.where(s[:, 0] > 0)[0]
    gk = np.where(s[:, 1] > 0)[0]
    overlap = np.array(sorted(set(gj) & set(gk)))
    mj = t[gj, 0].mean()
    mk = t[gk, 1].mean()
    H = len(overlap)
    wrong = (H / (H - 1.0)) * ((t[overlap, 0] - mj) * (t[overlap, 1] - mk)).sum()
    wrong /= Nj[0] * Nj[1]

    assert abs(V[0, 1] - wrong) > 1e-6  # the fixture actually exercises the distinction


# ===========================================================================
# F5 -- p-weighted (weight once, not w^2) with unequal N_j
# ===========================================================================
def test_f5_pweighted_matches_numpy_reference():
    fx = build_pweighted()
    assert len(set(fx["weights"].tolist())) > 1  # weights genuinely vary
    Nj = (fx["mask"] * fx["weights"][:, None]).sum(0)
    assert abs(Nj[0] - Nj[1]) > 1e-9  # N_j genuinely unequal
    V = _run_impl(fx)
    V_ref = _numpy_vtt(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["cell_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(V - V_ref)) < TOL


# ===========================================================================
# F6 -- FPC
# ===========================================================================
def test_f6_fpc_matches_numpy_reference():
    fx = build_fpc_indicator()
    # map PSU -> stratum for the numpy reference
    psu = np.rint(fx["psu_ids"]).astype(int)
    strat_of_psu = np.zeros(fx["n_psu"], int)
    for i in range(len(psu)):
        strat_of_psu[psu[i]] = int(fx["stratum_ids"][i])
    V = _run_impl(fx, fpc=True)
    V_ref = _numpy_vtt(
        fx["x"],
        fx["mask"],
        fx["weights"],
        fx["psu_ids"],
        fx["cell_ids"],
        fx["n_psu"],
        fpc=True,
        stratum_of_psu=strat_of_psu,
    )
    assert np.max(np.abs(V - V_ref)) < TOL


def test_fpc_shrinks_variance_relative_to_no_fpc():
    # build_fpc_indicator: each cell has H_sD = 2 populated PSUs and a stratum
    # total H_s = 6 (3 arms x 2), so the coordinate-independent convention-(ii)
    # factor is exactly 1 - 2/6 = 2/3 on every contributing cell.
    fx = build_fpc_indicator()
    V0 = _run_impl(fx, fpc=False)
    V1 = _run_impl(fx, fpc=True)
    d0, d1 = np.diag(V0), np.diag(V1)
    pos = d0 > 1e-12  # coordinates that carry a between-PSU variance
    assert np.all(d1 <= d0 + 1e-12)  # never larger than no-FPC
    assert np.allclose(d1[pos], (2.0 / 3.0) * d0[pos], rtol=1e-9)


# ===========================================================================
# H1 -- empty / non-contiguous PSU robustness
# ===========================================================================
def _inject_empty_psu(fx, gap_at):
    """Copy ``fx`` with an UNPOPULATED PSU slot inserted at index ``gap_at``:
    every PSU id >= gap_at is shifted up by one and ``n_psu`` grows by one, so
    slot ``gap_at`` exists in ``[0, n_psu)`` but holds no observation. The
    partition into populated PSUs is unchanged (a relabelling), so V must be
    bit-unchanged up to floating tolerance.
    """
    psu = np.rint(np.asarray(fx["psu_ids"])).astype(int).copy()
    psu[psu >= gap_at] += 1
    return {**fx, "psu_ids": psu.astype(float), "n_psu": int(fx["n_psu"]) + 1}


def test_empty_psu_slot_is_inert():
    """An empty PSU slot must contribute exactly nothing (H1). Guards the
    segment_max empty-segment fill on the non-FPC cell mapping AND the FPC
    H_s count / cell->stratum gather, neither of which may be corrupted by the
    INT32_MIN fill of an unpopulated slot.
    """
    fx = build_fpc_indicator()
    fx_gap = _inject_empty_psu(fx, gap_at=2)
    for fpc in (False, True):
        V = _run_impl(fx, fpc=fpc)
        V_gap = _run_impl(fx_gap, fpc=fpc)
        assert np.isfinite(V_gap).all()
        assert np.max(np.abs(V - V_gap)) < TOL


# ===========================================================================
# FPC convention (ii): coordinate independence + indefiniteness lock
# ===========================================================================
def build_coord_indep_fpc(seed=9):
    """One stratum, two cells, UNEQUAL per-pair overlap within cell 0.

    Cell 0 (PSUs 0-3) carries coords 0 and 1: coord 0 observed in {0,1,2}
    (support H_00 = 3), coord 1 in {1,2} (H_11 = 2), overlap {1,2} (H_01 = 2).
    Cell 1 (PSUs 4,5) carries coord 2, so the stratum total H_s = 6 exceeds
    cell 0's assigned count H_sD = 4 and the FPC factor 1 - 4/6 is non-trivial.
    """
    rng = np.random.default_rng(seed)
    M, upp = 3, 3
    supp0 = {0: {0, 1, 2}, 1: {1, 2}}
    rows_x, rows_mask, weights, cell_ids, psu_ids = [], [], [], [], []
    for g in range(4):  # cell 0
        for _ in range(upp):
            mr = np.zeros(M)
            if g in supp0[0]:
                mr[0] = 1.0
            if g in supp0[1]:
                mr[1] = 1.0
            rows_x.append(rng.normal(size=M))
            rows_mask.append(mr)
            weights.append(1.0)
            cell_ids.append(0)
            psu_ids.append(g)
    for g in (4, 5):  # cell 1
        for _ in range(upp):
            mr = np.zeros(M)
            mr[2] = 1.0
            rows_x.append(rng.normal(size=M))
            rows_mask.append(mr)
            weights.append(1.0)
            cell_ids.append(1)
            psu_ids.append(g)
    return dict(
        x=np.array(rows_x),
        mask=np.array(rows_mask),
        weights=np.array(weights),
        cell_ids=np.array(cell_ids),
        psu_ids=np.array(psu_ids),
        stratum_ids=np.zeros(len(psu_ids)),
        n_psu=6,
        n_cells=2,
        n_strata=1,
    )


def test_fpc_is_coordinate_independent_under_unequal_overlap():
    """Convention (ii) lock. With unequal per-pair overlap in a cell
    (H_00=3, H_11=2, H_01=2), the FPC factor applied to the (0,0), (1,1), and
    (0,1) entries must be IDENTICAL -- a single coordinate-independent per-cell
    scalar 1 - H_sD/H_s = 1 - 4/6. The rejected per-pair numerator (i) would
    give 1/2, 2/3, 2/3 respectively, failing this test.
    """
    fx = build_coord_indep_fpc()
    V0 = _run_impl(fx, fpc=False)
    V1 = _run_impl(fx, fpc=True)

    # impl matches the convention-(ii) numpy reference
    V1_ref = _numpy_vtt(
        fx["x"],
        fx["mask"],
        fx["weights"],
        fx["psu_ids"],
        fx["cell_ids"],
        fx["n_psu"],
        fpc=True,
        stratum_of_psu=np.zeros(fx["n_psu"], int),
    )
    assert np.max(np.abs(V1 - V1_ref)) < TOL

    entries = [(0, 0), (1, 1), (0, 1)]
    assert all(abs(V0[e]) > 1e-9 for e in entries)  # fixture is non-degenerate
    ratios = [V1[e] / V0[e] for e in entries]
    assert max(ratios) - min(ratios) < 1e-9  # coordinate-INDEPENDENT
    assert abs(ratios[0] - (1.0 - 4.0 / 6.0)) < 1e-9  # the per-cell factor


def test_v_can_be_indefinite_under_ragged_missingness():
    """Lock the 'PSD only with complete data' caveat: under ragged per-pair
    support, the available-pairs V_TT can be genuinely indefinite (a real
    negative eigenvalue). Repairing that is the regularization layer's job
    (DiagonalTikhonov), not this routine's -- so a negative eigenvalue here is
    EXPECTED, not a bug. Guards against a future 'helpful' internal PD repair.
    """
    min_eig = np.inf
    for seed in range(24):
        fx, _ = build_masked(seed=seed)
        V = _run_impl(fx)
        assert np.isfinite(V).all()
        min_eig = min(min_eig, float(np.linalg.eigvalsh(V).min()))
    assert min_eig < -1e-9


# ===========================================================================
# Contract conformance
# ===========================================================================
def test_satisfies_covariance_strategy_protocol():
    strat = StratifiedCovariance(
        psu_ids=jnp.zeros(4),
        cell_ids=jnp.zeros(4),
        stratum_ids=jnp.zeros(4),
        n_psu=4,
        n_cells=1,
        n_strata=1,
    )
    assert isinstance(strat, CovarianceStrategy)


def test_jit_compiles():
    fx = build_full()
    strat = StratifiedCovariance(
        psu_ids=jnp.asarray(fx["psu_ids"], jnp.float64),
        cell_ids=jnp.asarray(fx["cell_ids"], jnp.float64),
        stratum_ids=jnp.asarray(fx["stratum_ids"], jnp.float64),
        n_psu=fx["n_psu"],
        n_cells=fx["n_cells"],
        n_strata=fx["n_strata"],
    )
    # A real EmpiricalMeasure is a pytree, so it can cross the jit boundary
    # (the estimator always passes one); the stub above is not a pytree.
    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    fn = jax.jit(lambda m: strat.covariance(identity_psi, None, m))
    V = np.asarray(fn(measure))
    assert np.isfinite(V).all()


# ===========================================================================
# Cached-vs-self-compute parity (D != M, NaN feeding a NON-masked moment)
# ===========================================================================
@jdc.pytree_dataclass
class _AB:
    a: float
    b: float


def _psi_DneqM(xi, theta):
    # D = 3, M = 2. Moment 0 READS column 1 (which carries NaN for some rows);
    # both moments are observable for those rows -> the D != M case Seasonality
    # flagged. safe_x_for_psi must substitute column-mean identically on both paths.
    return jnp.array([xi[0] + xi[1] - theta.a, xi[2] - theta.b])


def test_cached_and_self_compute_agree_DneqM():
    rng = np.random.default_rng(7)
    N = 24
    x = rng.normal(size=(N, 3))
    x[::5, 1] = np.nan  # NaN in column 1, feeding non-masked moment 0
    mask = np.ones((N, 2))
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.ones(N)
    )
    psu = jnp.asarray(np.repeat(np.arange(6), 4), jnp.float64)
    cell = jnp.asarray(np.repeat(np.arange(3), 8), jnp.float64)
    strat = StratifiedCovariance(
        psu_ids=psu,
        cell_ids=cell,
        stratum_ids=cell,
        n_psu=6,
        n_cells=3,
        n_strata=3,
    )
    theta = _AB(a=0.1, b=-0.2)

    V_self = strat.covariance(_psi_DneqM, theta, measure)
    cached = measure.expectation_and_contributions(_psi_DneqM, theta)
    V_cached = strat.covariance(_psi_DneqM, theta, measure, cached_intermediates=cached)

    assert np.isfinite(np.asarray(V_self)).all()
    assert np.array_equal(np.asarray(V_self), np.asarray(V_cached))


# ===========================================================================
# Gradient AD-safety with a singular psi + NaN-marked missing data
# ===========================================================================
@jdc.pytree_dataclass
class _A:
    a: float


def _psi_singular(xi, theta):
    # log is singular at 0; column carries NaN at missing rows. safe_x_for_psi
    # substitutes the (positive) observed mean so the gradient stays finite.
    return jnp.array([jnp.log(xi[0]) - theta.a])


def test_gradient_is_finite_with_singular_psi():
    rng = np.random.default_rng(8)
    N = 18
    x = np.abs(rng.normal(2.0, 0.5, size=(N, 1))) + 0.5  # strictly positive
    x[::6, 0] = np.nan  # missing -> NaN-marked
    mask = (~np.isnan(x)).astype(float)
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.ones(N)
    )
    psu = jnp.asarray(np.repeat(np.arange(6), 3), jnp.float64)
    cell = jnp.asarray(np.repeat(np.arange(3), 6), jnp.float64)
    strat = StratifiedCovariance(
        psu_ids=psu,
        cell_ids=cell,
        stratum_ids=cell,
        n_psu=6,
        n_cells=3,
        n_strata=3,
    )

    def V_of_theta(a):
        return strat.covariance(_psi_singular, _A(a=a), measure)[0, 0]

    g_fwd = jax.jacfwd(V_of_theta)(0.3)
    g_rev = jax.grad(V_of_theta)(0.3)
    assert np.isfinite(float(g_fwd))
    assert np.isfinite(float(g_rev))


# ===========================================================================
# #80 DesignAwareCovariance scaffold
# ===========================================================================
def _design_and_sampling(fx):
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(fx["psu_ids"], jnp.float64),
        cell_ids=jnp.asarray(fx["cell_ids"], jnp.float64),
        stratum_ids=jnp.asarray(fx["stratum_ids"], jnp.float64),
        n_psu=fx["n_psu"],
        n_cells=fx["n_cells"],
        n_strata=fx["n_strata"],
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(fx["psu_ids"], jnp.float64), n_clusters=fx["n_psu"]
    )
    return design, sampling


def test_design_aware_all_design_reduces_bit_for_bit():
    fx = build_full()
    design, sampling = _design_and_sampling(fx)
    M = fx["mask"].shape[1]
    dac = DesignAwareCovariance.from_design_mask(design, sampling, jnp.ones(M))
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])

    # The engine is held by SHARED reference, not copied (the "not copied"
    # half of the #80 invariant): only an identity check distinguishes a
    # shared reference from a deep-copy/replace fork that would also pass the
    # output-equality assertion below.
    assert dac.design is design

    V_design = np.asarray(design.covariance(identity_psi, None, measure))
    V_dac = np.asarray(dac.covariance(identity_psi, None, measure))
    assert np.array_equal(V_design, V_dac)  # bit-for-bit


def test_design_aware_is_pytree_and_jits_all_design():
    """#80 static/traced invariant: ``all_design`` must stay a *static* bool
    (it drives a Python branch) while ``design_moment_mask`` is a *traced*
    leaf. If a refactor demoted ``all_design`` to a traced leaf, this jit
    would raise TracerBoolConversionError at ``if self.all_design`` -- a
    failure the eager-only tests above cannot catch.
    """
    fx = build_full()
    design, sampling = _design_and_sampling(fx)
    M = fx["mask"].shape[1]
    dac = DesignAwareCovariance.from_design_mask(design, sampling, jnp.ones(M))
    assert isinstance(dac, CovarianceStrategy)

    leaves = jax.tree_util.tree_leaves(dac)
    # design_moment_mask is a traced leaf; all_design (a bool) is static and
    # must NOT appear among the pytree leaves.
    assert any(getattr(leaf, "shape", None) == (M,) for leaf in leaves)
    assert not any(np.asarray(leaf).dtype == bool for leaf in leaves)

    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    fn = jax.jit(lambda m: dac.covariance(identity_psi, None, m))
    V = np.asarray(fn(measure))
    assert np.isfinite(V).all()


# ---------------------------------------------------------------------------
# #80 mixed design/sampling assembly (V_TT + V_SS + estimated V_TS)
# ---------------------------------------------------------------------------
def build_mixed(seed=11):
    """Mixed fixture: full mask, M=4 (coords 0,1 design; 2,3 sampling), with a
    per-PSU common component so the cluster totals are cross-correlated across
    coordinates -> a genuinely non-zero V_TS cross block.
    """
    rng = np.random.default_rng(seed)
    S, arms, Hpa, upp, M = 2, 2, 3, 4, 4
    rows_x, weights, cell_ids, psu_ids, stratum_ids = [], [], [], [], []
    g = c = 0
    for s in range(S):
        for _a in range(arms):
            cell = c
            c += 1
            for _ in range(Hpa):
                psu_common = rng.normal(size=M)  # shared across coords in a PSU
                for _ in range(upp):
                    rows_x.append(psu_common + rng.normal(size=M))
                    weights.append(1.0)
                    cell_ids.append(cell)
                    psu_ids.append(g)
                    stratum_ids.append(s)
                g += 1
    n = len(rows_x)
    return dict(
        x=np.array(rows_x),
        mask=np.ones((n, M)),
        weights=np.array(weights),
        cell_ids=np.array(cell_ids),
        psu_ids=np.array(psu_ids),
        stratum_ids=np.array(stratum_ids),
        n_psu=g,
        n_cells=c,
        n_strata=S,
        M=M,
    )


def _numpy_cluster_cross(x, mask, weights, psu_ids, n_psu):
    """Independent numpy cluster-total outer product / (N_j N_k)."""
    contrib = mask * weights[:, None] * np.where(mask > 0, x, 0.0)
    M = x.shape[1]
    tot = np.zeros((n_psu, M))
    psu = np.rint(psu_ids).astype(int)
    for i in range(len(psu)):
        tot[psu[i]] += contrib[i]
    numer = tot.T @ tot
    Nj = (mask * weights[:, None]).sum(0)
    V = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            d = Nj[j] * Nj[k]
            V[j, k] = numer[j, k] / d if d != 0.0 else 0.0
    return V


_DESIGN_MASK = jnp.array([1.0, 1.0, 0.0, 0.0])  # coords 0,1 design; 2,3 sampling
_T_IDX, _S_IDX = [0, 1], [2, 3]


def _mixed_dac_and_measure(fx):
    design, sampling = _design_and_sampling(fx)
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    return dac, design, sampling, measure


def test_design_aware_mixed_runs_finite_and_symmetric():
    fx = build_mixed()
    dac, *_, measure = _mixed_dac_and_measure(fx)
    V = np.asarray(dac.covariance(identity_psi, None, measure))
    assert np.isfinite(V).all()
    assert np.max(np.abs(V - V.T)) < 1e-12  # symmetric


def test_design_aware_VTS_estimated_not_zero_matches_numpy():
    """The cardinal #80 property: the cross block is ESTIMATED, not zeroed,
    and equals the independent cluster-total cross reference.
    """
    fx = build_mixed()
    dac, *_, measure = _mixed_dac_and_measure(fx)
    V = np.asarray(dac.covariance(identity_psi, None, measure))
    cross = V[np.ix_(_T_IDX, _S_IDX)]
    assert np.any(np.abs(cross) > 1e-9)  # NOT zeroed
    Vc = _numpy_cluster_cross(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["n_psu"]
    )
    assert np.max(np.abs(cross - Vc[np.ix_(_T_IDX, _S_IDX)])) < TOL


def test_design_aware_block_structure():
    """V_TT is the design-exact (centered) block; V_SS is the uncentered
    cluster block; they differ on the T coords (design != sampling form).
    """
    fx = build_mixed()
    dac, design, sampling, measure = _mixed_dac_and_measure(fx)
    V = np.asarray(dac.covariance(identity_psi, None, measure))
    V_design = np.asarray(design.covariance(identity_psi, None, measure))
    V_samp = np.asarray(sampling.covariance(identity_psi, None, measure))
    assert (
        np.max(np.abs(V[np.ix_(_T_IDX, _T_IDX)] - V_design[np.ix_(_T_IDX, _T_IDX)]))
        < TOL
    )
    assert (
        np.max(np.abs(V[np.ix_(_S_IDX, _S_IDX)] - V_samp[np.ix_(_S_IDX, _S_IDX)])) < TOL
    )
    # design-exact TT is the centered Neyman block, NOT the uncentered cluster form
    assert (
        np.max(np.abs(V[np.ix_(_T_IDX, _T_IDX)] - V_samp[np.ix_(_T_IDX, _T_IDX)]))
        > 1e-9
    )


def test_design_aware_mixed_cached_self_parity():
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    cached = measure.expectation_and_contributions(identity_psi, None)
    V_self = np.asarray(dac.covariance(identity_psi, None, measure))
    V_cached = np.asarray(
        dac.covariance(identity_psi, None, measure, cached_intermediates=cached)
    )
    assert np.array_equal(V_self, V_cached)


def test_design_aware_VTS_uses_sampling_cluster_unit():
    """The cross block clusters at self.sampling's unit (caller-controlled),
    NOT the design PSU. Build sampling at the COARSER stratum unit: V_TS must
    equal the stratum-level cross and DIFFER from the PSU-level cross (so a
    regression hard-coding design.psu_ids would be caught).
    """
    fx = build_mixed()
    design = StratifiedCovariance(
        psu_ids=jnp.asarray(fx["psu_ids"], jnp.float64),
        cell_ids=jnp.asarray(fx["cell_ids"], jnp.float64),
        stratum_ids=jnp.asarray(fx["stratum_ids"], jnp.float64),
        n_psu=fx["n_psu"],
        n_cells=fx["n_cells"],
        n_strata=fx["n_strata"],
    )
    sampling = ClusteredCovariance(
        cluster_ids=jnp.asarray(fx["stratum_ids"], jnp.float64),
        n_clusters=fx["n_strata"],
    )
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    cross = np.asarray(dac.covariance(identity_psi, None, measure))[
        np.ix_(_T_IDX, _S_IDX)
    ]

    cross_strat = _numpy_cluster_cross(
        fx["x"], fx["mask"], fx["weights"], fx["stratum_ids"], fx["n_strata"]
    )[np.ix_(_T_IDX, _S_IDX)]
    cross_psu = _numpy_cluster_cross(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["n_psu"]
    )[np.ix_(_T_IDX, _S_IDX)]
    assert np.max(np.abs(cross - cross_strat)) < TOL  # uses sampling's unit
    assert np.max(np.abs(cross_strat - cross_psu)) > 1e-9  # genuinely coarser


def test_design_aware_fpc_enters_VTT_only():
    """The design FPC scales V_TT only; V_SS and V_TS carry no FPC factor."""
    fx = build_mixed()
    psu = jnp.asarray(fx["psu_ids"], jnp.float64)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])

    def _V(fpc):
        design = StratifiedCovariance(
            psu_ids=psu,
            cell_ids=jnp.asarray(fx["cell_ids"], jnp.float64),
            stratum_ids=jnp.asarray(fx["stratum_ids"], jnp.float64),
            n_psu=fx["n_psu"],
            n_cells=fx["n_cells"],
            n_strata=fx["n_strata"],
            fpc=fpc,
        )
        sampling = ClusteredCovariance(cluster_ids=psu, n_clusters=fx["n_psu"])
        dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
        return np.asarray(dac.covariance(identity_psi, None, measure))

    V0, V1 = _V(False), _V(True)
    assert (
        np.max(np.abs(V1[np.ix_(_T_IDX, _T_IDX)] - V0[np.ix_(_T_IDX, _T_IDX)])) > 1e-9
    )
    assert np.max(np.abs(V1[np.ix_(_S_IDX, _S_IDX)] - V0[np.ix_(_S_IDX, _S_IDX)])) < TOL
    assert np.max(np.abs(V1[np.ix_(_T_IDX, _S_IDX)] - V0[np.ix_(_T_IDX, _S_IDX)])) < TOL


# ---------------------------------------------------------------------------
# #109: native cross-block (V_TS) ablation / accessor.
# ---------------------------------------------------------------------------
def test_design_aware_couple_false_zeroes_cross_corners():
    """couple=False gives the block-diagonal V_TT (+) V_SS: cross corners are
    exactly zero, while the coupled default leaves them estimated."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    dac_coupled = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    dac_block = DesignAwareCovariance.from_design_mask(
        design, sampling, _DESIGN_MASK, couple=False
    )
    Vc = np.asarray(dac_coupled.covariance(identity_psi, None, measure))
    Vb = np.asarray(dac_block.covariance(identity_psi, None, measure))
    # Block-diagonal: cross corners are exactly zero.
    assert np.all(Vb[np.ix_(_T_IDX, _S_IDX)] == 0.0)
    assert np.all(Vb[np.ix_(_S_IDX, _T_IDX)] == 0.0)
    # Coupled: cross corners are NOT zero (regression vs #80's estimated cross).
    assert np.any(np.abs(Vc[np.ix_(_T_IDX, _S_IDX)]) > 1e-9)


def test_design_aware_couple_false_keeps_diagonal_blocks_bit_identical():
    """Dropping the cross must not perturb the TT or SS diagonal blocks."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    Vc = np.asarray(
        DesignAwareCovariance.from_design_mask(
            design, sampling, _DESIGN_MASK
        ).covariance(identity_psi, None, measure)
    )
    Vb = np.asarray(
        DesignAwareCovariance.from_design_mask(
            design, sampling, _DESIGN_MASK, couple=False
        ).covariance(identity_psi, None, measure)
    )
    assert np.array_equal(Vc[np.ix_(_T_IDX, _T_IDX)], Vb[np.ix_(_T_IDX, _T_IDX)])
    assert np.array_equal(Vc[np.ix_(_S_IDX, _S_IDX)], Vb[np.ix_(_S_IDX, _S_IDX)])


def test_design_aware_cross_block_decomposition_identity():
    """The cardinal #109 identity: coupled == block-diagonal + cross_block."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    dac_block = DesignAwareCovariance.from_design_mask(
        design, sampling, _DESIGN_MASK, couple=False
    )
    Vc = np.asarray(dac.covariance(identity_psi, None, measure))
    Vb = np.asarray(dac_block.covariance(identity_psi, None, measure))
    cross = np.asarray(dac.cross_block(identity_psi, None, measure))
    assert np.max(np.abs(Vc - (Vb + cross))) < TOL
    # cross_block carries ONLY the off-diagonal corners (TT/SS are zero).
    assert np.all(cross[np.ix_(_T_IDX, _T_IDX)] == 0.0)
    assert np.all(cross[np.ix_(_S_IDX, _S_IDX)] == 0.0)
    # ...and its corners equal the coupled cross corners.
    assert (
        np.max(np.abs(cross[np.ix_(_T_IDX, _S_IDX)] - Vc[np.ix_(_T_IDX, _S_IDX)])) < TOL
    )


def test_design_aware_cross_block_matches_numpy_reference():
    """cross_block's estimated corner equals the independent cluster-cross ref."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    cross = np.asarray(dac.cross_block(identity_psi, None, measure))
    Vc = _numpy_cluster_cross(
        fx["x"], fx["mask"], fx["weights"], fx["psu_ids"], fx["n_psu"]
    )
    assert (
        np.max(np.abs(cross[np.ix_(_T_IDX, _S_IDX)] - Vc[np.ix_(_T_IDX, _S_IDX)])) < TOL
    )


def test_design_aware_couple_default_is_true_bitwise_unchanged():
    """Non-regression: the default construction is couple=True and bit-for-bit
    equal to the explicit couple=True (the pre-#109 behaviour)."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    V_default = np.asarray(
        DesignAwareCovariance.from_design_mask(
            design, sampling, _DESIGN_MASK
        ).covariance(identity_psi, None, measure)
    )
    V_explicit = np.asarray(
        DesignAwareCovariance.from_design_mask(
            design, sampling, _DESIGN_MASK, couple=True
        ).covariance(identity_psi, None, measure)
    )
    assert np.array_equal(V_default, V_explicit)


def test_design_aware_all_design_cross_block_is_zero():
    """all_design => no sampled coords => cross_block is exactly zero, and
    couple has no effect on covariance()."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    measure = _MeasureStub(fx["x"], fx["mask"], fx["weights"])
    M = fx["M"]
    dac = DesignAwareCovariance.from_design_mask(design, sampling, jnp.ones(M))
    cross = np.asarray(dac.cross_block(identity_psi, None, measure))
    assert np.all(cross == 0.0)
    V_coupled = np.asarray(dac.covariance(identity_psi, None, measure))
    V_block = np.asarray(
        DesignAwareCovariance.from_design_mask(
            design, sampling, jnp.ones(M), couple=False
        ).covariance(identity_psi, None, measure)
    )
    assert np.array_equal(V_coupled, V_block)


def test_design_aware_cross_block_cached_self_parity():
    """cross_block agrees between the cached and self-compute paths."""
    fx = build_mixed()
    design, sampling = _design_and_sampling(fx)
    dac = DesignAwareCovariance.from_design_mask(design, sampling, _DESIGN_MASK)
    measure = EmpiricalMeasure(
        x=jnp.asarray(fx["x"]),
        mask=jnp.asarray(fx["mask"]),
        weights=jnp.asarray(fx["weights"]),
    )
    cached = measure.expectation_and_contributions(identity_psi, None)
    c_self = np.asarray(dac.cross_block(identity_psi, None, measure))
    c_cached = np.asarray(
        dac.cross_block(identity_psi, None, measure, cached_intermediates=cached)
    )
    assert np.array_equal(c_self, c_cached)
