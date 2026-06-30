"""Tests for the #167 conditional/coupled empirical-law queries.

Covers the red-team's required guards (mask threshold + partition, the
paired-mask decoupling, the empty/warning contract, the coupling soundness
via ``coupling_id``) and the byte-equality of the two ladder readouts the
helpers retrofit.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm.studies import (
    MCRecords,
    SelectionConditionalWarning,
    coverage,
    crn_pair,
    event_share,
    given,
)
from emu_gmm.studies.conditioning import CoupledRecords
from emu_gmm.types import FitRecord


def _rec(
    *,
    theta,
    se,
    converged,
    binding,
    J_pvalue=None,
    J_pvalue_adjusted=None,
    param_names=("b0", "b1"),
) -> FitRecord:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    n = theta.shape[0]
    o = jnp.ones(n, dtype=jnp.float64)

    def v(x, d):
        return jnp.asarray(x if x is not None else d, dtype=jnp.float64)

    p = v(J_pvalue, 0.5 * o)
    return FitRecord(
        theta_flat=theta,
        se=jnp.asarray(se, dtype=jnp.float64),
        J_stat=o,
        J_pvalue=p,
        J_pvalue_adjusted=v(J_pvalue_adjusted, p),
        converged=v(converged, o),
        tau_realised=0.0 * o,
        binding_ridge=v(binding, 0.0 * o),
        sigma_meat_indefinite=0.0 * o,
        J_dof=1,
        param_names=param_names,
    )


def _mc(rec, *, seed=0, coupling_id="X") -> MCRecords:
    return MCRecords(
        records=rec,
        key=jax.random.PRNGKey(seed),
        n_reps=int(np.asarray(rec.converged).shape[0]),
        coupling_id=coupling_id,
    )


# --------------------------------------------------------------------------
# given() / event_share()
# --------------------------------------------------------------------------


class TestGiven:
    def test_masks_event_and_preserves_static_fields(self):
        rec = _rec(
            theta=[[1.0, 2.0], [1.0, 3.0], [1.0, 9.0]],
            se=[[0.5, 0.5]] * 3,
            converged=[1, 1, 0],
            binding=[1, 0, 1],
        )
        g = given(rec, "binding_ridge", acknowledge_conditional=True)
        assert np.asarray(g.theta_flat).shape == (2, 2)  # 2 binding reps
        assert g.J_dof == 1 and g.param_names == ("b0", "b1")  # statics survive
        np.testing.assert_array_equal(np.asarray(g.binding_ridge), [1.0, 1.0])

    def test_partition_identity_including_converged(self):
        rec = _rec(
            theta=[[0.0, 0.0]] * 5,
            se=[[1.0, 1.0]] * 5,
            converged=[1, 1, 0, 1, 0],
            binding=[1, 0, 1, 1, 0],
        )
        a = event_share(rec, "binding_ridge")
        b = event_share(rec, "binding_ridge", negate=True)
        assert a.n_selected + b.n_selected == a.n_total
        assert a.n_selected_converged + b.n_selected_converged == a.n_total_converged

    def test_event_share_both_denominators(self):
        rec = _rec(
            theta=[[0.0, 0.0]] * 4,
            se=[[1.0, 1.0]] * 4,
            converged=[1, 1, 0, 1],
            binding=[1, 0, 1, 1],
        )
        es = event_share(rec, "binding_ridge")
        assert es.n_selected == 3 and es.n_total == 4
        assert es.n_selected_converged == 2 and es.n_total_converged == 3
        assert es.fraction_all == pytest.approx(0.75)
        assert es.fraction_converged == pytest.approx(2 / 3)

    def test_composes_with_coverage_equals_manual(self):
        rec = _rec(
            theta=[[1.0, 3.0], [1.0, 3.0], [1.0, 99.0]],
            se=[[1.0, 1.0]] * 3,
            converged=[1, 1, 1],
            binding=[1, 1, 0],
        )
        cov_helper = coverage(
            given(rec, "binding_ridge", acknowledge_conditional=True), theta0=[1.0, 3.0]
        )
        # Manual: same two binding reps.
        manual = coverage(
            _rec(
                theta=[[1.0, 3.0], [1.0, 3.0]],
                se=[[1.0, 1.0]] * 2,
                converged=[1, 1],
                binding=[1, 1],
            ),
            theta0=[1.0, 3.0],
        )
        np.testing.assert_array_equal(cov_helper.coverage, manual.coverage)

    def test_conditional_differs_from_marginal(self):
        # binding reps miscover b1 by construction; marginal covers.
        rec = _rec(
            theta=[[1.0, 3.0], [1.0, 3.0], [1.0, 50.0], [1.0, 50.0]],
            se=[[1.0, 1.0]] * 4,
            converged=[1, 1, 1, 1],
            binding=[0, 0, 1, 1],  # the far-off reps are the binding ones
        )
        marginal = coverage(rec, theta0=[1.0, 3.0]).coverage[1]
        conditional = coverage(
            given(rec, "binding_ridge", acknowledge_conditional=True), theta0=[1.0, 3.0]
        ).coverage[1]
        assert conditional < marginal  # selection on binding shows miscoverage

    def test_empty_subset_is_nan_not_crash(self):
        rec = _rec(
            theta=[[1.0, 2.0]],
            se=[[1.0, 1.0]],
            converged=[1],
            binding=[0],  # no binding reps
        )
        g = given(rec, "binding_ridge", acknowledge_conditional=True)
        assert np.asarray(g.theta_flat).shape == (0, 2)
        cov = coverage(g, theta0=[1.0, 2.0])
        assert np.all(np.isnan(cov.coverage)) and cov.n_used == 0

    def test_negate(self):
        rec = _rec(
            theta=[[0.0, 0.0]] * 3,
            se=[[1.0, 1.0]] * 3,
            converged=[1, 1, 1],
            binding=[1, 0, 0],
        )
        assert (
            np.asarray(
                given(rec, "binding_ridge", acknowledge_conditional=True).binding_ridge
            ).size
            == 1
        )
        assert (
            np.asarray(
                given(
                    rec, "binding_ridge", negate=True, acknowledge_conditional=True
                ).binding_ridge
            ).size
            == 2
        )

    def test_predicate_form(self):
        rec = _rec(
            theta=[[1.0, 2.0], [1.0, 8.0], [1.0, 3.0]],
            se=[[1.0, 1.0]] * 3,
            converged=[1, 1, 1],
            binding=[0, 0, 0],
        )
        g = given(rec, lambda r: np.asarray(r.theta_flat)[:, 1] > 5.0)
        assert np.asarray(g.theta_flat).shape == (1, 2)

    def test_bad_flag_name_raises(self):
        rec = _rec(theta=[[1.0, 2.0]], se=[[1.0, 1.0]], converged=[1], binding=[0])
        with pytest.raises(ValueError, match="unknown flag"):
            given(rec, "not_a_flag")

    def test_non_bool_predicate_raises(self):
        rec = _rec(theta=[[1.0, 2.0]], se=[[1.0, 1.0]], converged=[1], binding=[0])
        with pytest.raises(TypeError, match="boolean array"):
            given(rec, lambda r: np.asarray(r.theta_flat)[:, 1])  # float, not bool

    def test_flags_are_exact_binary(self):
        # Guard: if a future producer emits a fractional flag, >0.5 vs >0
        # would diverge and the byte-equal retrofit would silently drift.
        rec = _rec(
            theta=[[0.0, 0.0]] * 3,
            se=[[1.0, 1.0]] * 3,
            converged=[1, 0, 1],
            binding=[1, 1, 0],
        )
        for field in ("converged", "binding_ridge", "sigma_meat_indefinite"):
            vals = np.asarray(getattr(rec, field))
            assert np.all(np.isin(vals, (0.0, 1.0))), f"{field} not exact-binary"


# --------------------------------------------------------------------------
# crn_pair()
# --------------------------------------------------------------------------


def _pair_fixture(*, seed_a=0, seed_b=0, id_a="X", id_b="X"):
    ra = _rec(
        theta=[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        se=[[0.5, 0.5]] * 3,
        converged=[1, 1, 1],
        binding=[0, 0, 0],
    )
    rb = _rec(
        theta=[[1.0, 2.2], [1.0, 2.9], [1.0, 4.1]],
        se=[[0.7, 0.7]] * 3,
        converged=[1, 1, 1],
        binding=[0, 0, 0],
    )
    return _mc(ra, seed=seed_a, coupling_id=id_a), _mc(
        rb, seed=seed_b, coupling_id=id_b
    )


class TestCrnPair:
    def test_matching_coupling_id_accepts(self):
        a, b = _pair_fixture(id_a="study1", id_b="study1")
        cp = crn_pair(a, b)
        assert isinstance(cp, CoupledRecords) and cp.n_reps == 3

    def test_coupling_id_mismatch_refuses(self):
        a, b = _pair_fixture(id_a="study1", id_b="study2")
        with pytest.raises(ValueError, match="coupling_id mismatch"):
            crn_pair(a, b)

    def test_missing_id_refuses_without_assert(self):
        a, b = _pair_fixture(id_a=None, id_b=None)
        with pytest.raises(ValueError, match="cannot verify CRN coupling"):
            crn_pair(a, b)

    def test_missing_id_accepts_with_assert(self):
        a, b = _pair_fixture(id_a=None, id_b=None)
        cp = crn_pair(a, b, assert_coupled=True)
        assert cp.n_reps == 3

    def test_key_mismatch_refuses_even_with_matching_id(self):
        a, b = _pair_fixture(seed_a=0, seed_b=1, id_a="s", id_b="s")
        with pytest.raises(ValueError, match="master PRNG keys differ"):
            crn_pair(a, b)

    def test_n_reps_mismatch_refuses(self):
        a, _ = _pair_fixture()
        short = _rec(theta=[[1.0, 2.0]], se=[[1.0, 1.0]], converged=[1], binding=[0])
        b = _mc(short, coupling_id="X")
        with pytest.raises(ValueError, match="different n_reps"):
            crn_pair(a, b)

    def test_param_names_mismatch_refuses(self):
        a, _ = _pair_fixture()
        other = _rec(
            theta=[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
            se=[[1.0, 1.0]] * 3,
            converged=[1, 1, 1],
            binding=[0, 0, 0],
            param_names=("x", "y"),
        )
        b = _mc(other, coupling_id="X")
        with pytest.raises(ValueError, match="different param_names"):
            crn_pair(a, b)

    def test_rejects_non_mcrecords(self):
        a, _ = _pair_fixture()
        with pytest.raises(TypeError, match="whole MCRecords"):
            crn_pair(a, a.records)  # a bare FitRecord
        with pytest.raises(TypeError, match="whole MCRecords"):
            crn_pair(
                a, given(a, "binding_ridge", acknowledge_conditional=True)
            )  # a conditioned record

    def test_self_pair_positive_control(self):
        a, _ = _pair_fixture()
        cp = crn_pair(a, a)
        ind = np.array([1.0, 0.0, 1.0])
        fl = cp.flips(ind, ind)
        assert fl.gain == 0 and fl.lose == 0
        x = np.asarray(a.records.J_stat)
        assert cp.mean_paired_diff(x, x, where=cp.both_finite(x, x)) == 0.0


class TestPairedContrasts:
    def test_flips_directional(self):
        a, b = _pair_fixture(id_a="s", id_b="s")
        cp = crn_pair(a, b)
        ca = np.array([1.0, 0.0, np.nan])
        cb = np.array([1.0, 1.0, 1.0])
        fl = cp.flips(ca, cb)
        assert fl.gain == 1 and fl.lose == 0 and fl.n_both == 2  # NaN rep excluded

    def test_paired_diff_requires_where(self):
        a, b = _pair_fixture(id_a="s", id_b="s")
        cp = crn_pair(a, b)
        with pytest.raises(TypeError):
            cp.paired_diff(np.array([1.0]), np.array([2.0]))  # no where=

    def test_flips_n_both_matches_paired_diff_len(self):
        a, b = _pair_fixture(id_a="s", id_b="s")
        cp = crn_pair(a, b)
        ca = np.array([1.0, np.nan, 0.0])
        cb = np.array([0.0, 1.0, 1.0])
        both = cp.both_finite(ca, cb)
        fl = cp.flips(ca, cb, where=both)
        diff = cp.paired_diff(ca, cb, where=both)
        assert fl.n_both == diff.shape[0]

    def test_mean_paired_diff_empty_is_nan_no_warning(self):
        a, b = _pair_fixture(id_a="s", id_b="s")
        cp = crn_pair(a, b)
        empty = np.zeros(3, dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning becomes an error
            out = cp.mean_paired_diff(
                np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), where=empty
            )
        assert np.isnan(out)


# --------------------------------------------------------------------------
# Byte-equality of the retrofitted ladder readouts (the #167 acceptance bar)
# --------------------------------------------------------------------------


class TestRetrofitByteEquality:
    def _arm(self, *, J_pvalue, J_pvalue_adjusted, converged, binding, seed, cid):
        n = len(converged)
        rec = _rec(
            theta=[[1.0, 2.0]] * n,
            se=[[1.0, 1.0]] * n,
            converged=converged,
            binding=binding,
            J_pvalue=J_pvalue,
            J_pvalue_adjusted=J_pvalue_adjusted,
        )
        return _mc(rec, seed=seed, coupling_id=cid)

    def test_jadj_binding_subset_byte_equal(self):
        rng = np.random.default_rng(0)
        n = 60
        conv = (rng.random(n) > 0.1).astype(float)
        binding = (rng.random(n) > 0.4).astype(float)
        pn = rng.random(n)
        pa = rng.random(n)
        mc = self._arm(
            J_pvalue=pn,
            J_pvalue_adjusted=pa,
            converged=conv,
            binding=binding,
            seed=0,
            cid="X",
        )
        rec = mc.records
        # OLD inline logic
        used = np.asarray(rec.converged) > 0
        binding_old = (np.asarray(rec.binding_ridge) > 0) & used
        nb_old = int(binding_old.sum())
        pnb_old = np.asarray(rec.J_pvalue)[binding_old]
        pab_old = np.asarray(rec.J_pvalue_adjusted)[binding_old]
        # NEW helper logic
        cond = given(mc, "binding_ridge", acknowledge_conditional=True)
        share = event_share(mc, "binding_ridge")
        bconv = np.asarray(cond.converged) > 0.5
        pnb_new = np.asarray(cond.J_pvalue)[bconv]
        pab_new = np.asarray(cond.J_pvalue_adjusted)[bconv]
        assert share.n_selected_converged == nb_old
        assert share.n_total_converged == int(used.sum())
        np.testing.assert_array_equal(pnb_new, pnb_old)
        np.testing.assert_array_equal(pab_new, pab_old)

    def test_paired_dof_byte_equal(self):
        rng = np.random.default_rng(1)
        n = 40
        ca = np.where(rng.random(n) > 0.2, (rng.random(n) > 0.5).astype(float), np.nan)
        cb = np.where(rng.random(n) > 0.2, (rng.random(n) > 0.5).astype(float), np.nan)
        Ja = rng.random(n)
        Jb = rng.random(n)
        a = self._arm(
            J_pvalue=rng.random(n),
            J_pvalue_adjusted=rng.random(n),
            converged=np.ones(n),
            binding=np.zeros(n),
            seed=7,
            cid="P",
        )
        b = self._arm(
            J_pvalue=rng.random(n),
            J_pvalue_adjusted=rng.random(n),
            converged=np.ones(n),
            binding=np.zeros(n),
            seed=7,
            cid="P",
        )
        # OLD inline logic
        both = np.isfinite(ca) & np.isfinite(cb)
        gain_old = int(np.sum((cb[both] == 1) & (ca[both] == 0)))
        lose_old = int(np.sum((cb[both] == 0) & (ca[both] == 1)))
        jd_old = float(np.nanmean(Jb[both] - Ja[both]))
        n_old = int(both.sum())
        # NEW helper logic
        cp = crn_pair(a, b)
        both_new = cp.both_finite(ca, cb)
        fl = cp.flips(ca, cb, where=both_new)
        jd_new = cp.mean_paired_diff(Ja, Jb, where=both_new)
        assert (fl.gain, fl.lose, fl.n_both) == (gain_old, lose_old, n_old)
        assert jd_new == pytest.approx(jd_old, abs=0, rel=0)


class TestSelectionConditionalGate:
    """#167 §6 Q1: given() warns when conditioning on an estimator-internal
    event (selection-conditional, not nominal), unless acknowledged."""

    def _rec3(self):
        return _rec(
            theta=[[0.0, 0.0]] * 3,
            se=[[1.0, 1.0]] * 3,
            converged=[1, 1, 1],
            binding=[1, 0, 1],
        )

    @pytest.mark.parametrize("flag", ["binding_ridge", "sigma_meat_indefinite"])
    def test_warns_on_internal_hazard_flags(self, flag):
        with pytest.warns(SelectionConditionalWarning, match="SELECTION-CONDITIONAL"):
            given(self._rec3(), flag)

    def test_warning_silenced_by_acknowledge(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            given(self._rec3(), "binding_ridge", acknowledge_conditional=True)

    def test_complement_also_warns(self):
        # coverage among NON-binding reps is equally selection-conditional.
        with pytest.warns(SelectionConditionalWarning):
            given(self._rec3(), "binding_ridge", negate=True)

    def test_converged_not_gated(self):
        # the standard exclude-but-count exclusion is not a hazard.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            given(self._rec3(), "converged")

    def test_predicate_not_gated(self):
        # a predicate event is the caller's explicit responsibility.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            given(self._rec3(), lambda r: np.asarray(r.binding_ridge) > 0.5)

    def test_event_share_not_gated(self):
        # event_share only reports a count; it produces no subset for coverage.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            event_share(self._rec3(), "binding_ridge")
