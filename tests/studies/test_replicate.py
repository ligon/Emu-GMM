"""Layer-1 driver contracts (#114): shapes, CRN, reproducibility,
non-convergence accounting, the inherited no-retrace property, and the
per-replicate ridge-anchoring mode (#142)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import IIDCovariance, build_estimator
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.studies import (
    MCRecords,
    crn_pair,
    fit_record,
    replicate,
    replicate_coupled,
)
from emu_gmm.types import FitRecord

N = 400
THETA_TRUE = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)


def _theta0() -> EulerParams:
    return EulerParams(beta=0.9, gamma=1.0)


def _make_dgp(n: int = N):
    sampler = euler_sampler_factory(n)

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        x = sampler(key, THETA_TRUE)
        return EmpiricalMeasure(x=x, mask=jnp.ones((n, 3)), weights=jnp.ones(n))

    return dgp


def _make_run(model=euler_residual, n: int = N, **kwargs):
    return build_estimator(
        model,
        measure=_make_dgp(n)(jax.random.PRNGKey(99)),
        covariance=IIDCovariance(),
        parameters=_theta0(),
        **kwargs,
    )


class TestShapesAndMetadata:
    def test_records_carry_leading_rep_axis(self):
        recs = replicate(
            _make_run(),
            _make_dgp(),
            n_reps=3,
            key=jax.random.PRNGKey(0),
            theta_init=_theta0(),
        )
        assert isinstance(recs, MCRecords)
        assert recs.n_reps == 3
        assert recs.records.theta_flat.shape == (3, 2)
        assert recs.records.se.shape == (3, 2)
        assert recs.records.J_stat.shape == (3,)
        assert recs.records.J_pvalue.shape == (3,)
        assert recs.records.converged.shape == (3,)
        assert recs.records.tau_realised.shape == (3,)
        assert recs.records.binding_ridge.shape == (3,)
        assert recs.records.sigma_meat_indefinite.shape == (3,)
        assert recs.records.tau_saturated.shape == (3,)
        assert recs.records.J_dof == 1  # M=3, K=2; static, unstacked
        assert recs.param_names == ("beta", "gamma")

    def test_records_is_a_pytree(self):
        recs = replicate(
            _make_run(),
            _make_dgp(),
            n_reps=2,
            key=jax.random.PRNGKey(0),
            theta_init=_theta0(),
        )
        leaves, treedef = jax.tree_util.tree_flatten(recs)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.n_reps == 2
        np.testing.assert_array_equal(
            np.asarray(rebuilt.records.theta_flat),
            np.asarray(recs.records.theta_flat),
        )

    def test_to_pandas_one_row_per_rep(self):
        recs = replicate(
            _make_run(),
            _make_dgp(),
            n_reps=3,
            key=jax.random.PRNGKey(0),
            theta_init=_theta0(),
        )
        df = recs.to_pandas()
        assert len(df) == 3
        for col in (
            "theta_beta",
            "se_beta",
            "theta_gamma",
            "se_gamma",
            "J_stat",
            "J_pvalue",
            "J_pvalue_adjusted",
            "converged",
            "tau_realised",
            "binding_ridge",
            "sigma_meat_indefinite",
            "tau_saturated",
        ):
            assert col in df.columns

    def test_to_pandas_carries_sigma_meat_indefinite_values(self):
        """The #138 NaN-SE event flag survives record -> stack ->
        to_pandas (#143): hand-build a stacked FitRecord with the flag
        set on one rep and read it back out of the DataFrame."""
        from emu_gmm.types import FitRecord

        n = 3
        flag = jnp.asarray([0.0, 1.0, 0.0])
        rec = FitRecord(
            theta_flat=jnp.zeros((n, 2)),
            se=jnp.ones((n, 2)),
            J_stat=jnp.zeros(n),
            J_pvalue=0.5 * jnp.ones(n),
            J_pvalue_adjusted=0.5 * jnp.ones(n),
            converged=jnp.ones(n),
            tau_realised=jnp.zeros(n),
            binding_ridge=jnp.zeros(n),
            sigma_meat_indefinite=flag,
            tau_saturated=jnp.zeros(n),
            J_dof=1,
            param_names=("beta", "gamma"),
        )
        recs = MCRecords(records=rec, key=jax.random.PRNGKey(0), n_reps=n)
        df = recs.to_pandas()
        np.testing.assert_array_equal(
            df["sigma_meat_indefinite"].to_numpy(), np.array([0.0, 1.0, 0.0])
        )

    def test_to_pandas_carries_tau_saturated_values(self):
        """The #205 ridge-saturation flag survives record -> stack ->
        to_pandas (the sigma_meat_indefinite pattern): hand-build a
        stacked FitRecord with the flag set on one rep and read it back
        out of the DataFrame."""
        from emu_gmm.types import FitRecord

        n = 3
        flag = jnp.asarray([0.0, 1.0, 0.0])
        rec = FitRecord(
            theta_flat=jnp.zeros((n, 2)),
            se=jnp.ones((n, 2)),
            J_stat=jnp.zeros(n),
            J_pvalue=0.5 * jnp.ones(n),
            J_pvalue_adjusted=0.5 * jnp.ones(n),
            converged=jnp.ones(n),
            tau_realised=jnp.zeros(n),
            binding_ridge=jnp.zeros(n),
            sigma_meat_indefinite=jnp.zeros(n),
            tau_saturated=flag,
            J_dof=1,
            param_names=("beta", "gamma"),
        )
        recs = MCRecords(records=rec, key=jax.random.PRNGKey(0), n_reps=n)
        df = recs.to_pandas()
        np.testing.assert_array_equal(
            df["tau_saturated"].to_numpy(), np.array([0.0, 1.0, 0.0])
        )

    def test_n_reps_must_be_positive(self):
        import pytest

        with pytest.raises(ValueError, match="n_reps"):
            replicate(
                _make_run(),
                _make_dgp(),
                n_reps=0,
                key=jax.random.PRNGKey(0),
                theta_init=_theta0(),
            )


class TestRNG:
    def test_reproducible_from_same_key(self):
        run = _make_run()
        dgp = _make_dgp()
        a = replicate(
            run, dgp, n_reps=3, key=jax.random.PRNGKey(7), theta_init=_theta0()
        )
        b = replicate(
            run, dgp, n_reps=3, key=jax.random.PRNGKey(7), theta_init=_theta0()
        )
        np.testing.assert_array_equal(
            np.asarray(a.records.theta_flat), np.asarray(b.records.theta_flat)
        )
        np.testing.assert_array_equal(
            np.asarray(a.records.J_stat), np.asarray(b.records.J_stat)
        )

    def test_reps_use_distinct_draws(self):
        recs = replicate(
            _make_run(),
            _make_dgp(),
            n_reps=3,
            key=jax.random.PRNGKey(7),
            theta_init=_theta0(),
        )
        theta = np.asarray(recs.records.theta_flat)
        assert not np.allclose(theta[0], theta[1])
        assert not np.allclose(theta[1], theta[2])

    def test_crn_same_per_rep_keys_across_arms(self):
        """Two arms run with the same master key see identical draws
        rep-for-rep when the dgp is deterministic in its key --- the CRN
        contract (fold_in(key, r), documented in the driver)."""

        def recording_dgp():
            seen: list[np.ndarray] = []
            base = _make_dgp()

            def dgp(key: jax.Array) -> EmpiricalMeasure:
                m = base(key)
                seen.append(np.asarray(m.x))
                return m

            return dgp, seen

        key = jax.random.PRNGKey(11)
        dgp_a, seen_a = recording_dgp()
        dgp_b, seen_b = recording_dgp()
        # Two arms: same solver family but separately built estimators
        # (in a real study these would differ in covariance/weighting).
        replicate(_make_run(), dgp_a, n_reps=3, key=key, theta_init=_theta0())
        replicate(_make_run(), dgp_b, n_reps=3, key=key, theta_init=_theta0())
        assert len(seen_a) == len(seen_b) == 3
        for xa, xb in zip(seen_a, seen_b, strict=False):
            np.testing.assert_array_equal(xa, xb)
        # Audit L4: the two-arm comparison above is satisfied by ANY
        # deterministic key schedule (it compares an arm to itself).
        # Pin the DOCUMENTED schedule itself: rep r sees exactly
        # dgp(fold_in(key, r)) -- a refactor to e.g. split() must fail
        # here, because (key, r) reproducibility is the advertised
        # public contract.
        base = _make_dgp()
        for r in range(3):
            expected = np.asarray(base(jax.random.fold_in(key, r)).x)
            np.testing.assert_array_equal(seen_a[r], expected)


class TestNonConvergence:
    def test_non_converged_counted_not_dropped(self):
        """A starved optimiser (max_steps=1) cannot certify convergence;
        every rep is still recorded and the flag does the accounting."""
        n_reps = 3
        run = _make_run(optimizer=optimistix_lm(max_steps=1))
        recs = replicate(
            run,
            _make_dgp(),
            n_reps=n_reps,
            key=jax.random.PRNGKey(0),
            theta_init=EulerParams(beta=0.5, gamma=5.0),
        )
        conv = np.asarray(recs.records.converged)
        assert conv.shape == (n_reps,)  # nothing dropped
        assert conv.mean() < 1.0  # at least one non-converged
        assert recs.n_converged + recs.n_excluded == n_reps
        assert recs.n_excluded >= 1


class TestNoRetrace:
    def test_study_inherits_kernel_no_retrace(self):
        """psi trace count is independent of n_reps: a 5-rep study costs
        exactly the traces of a 2-rep study (the #124 kernel property,
        inherited by the driver)."""

        class CountingModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, x, theta):
                self.calls += 1
                return euler_residual(x, theta)

        def study_psi_calls(n_reps: int) -> int:
            counting = CountingModel()
            run = _make_run(model=counting)
            replicate(
                run,
                _make_dgp(),
                n_reps=n_reps,
                key=jax.random.PRNGKey(0),
                theta_init=_theta0(),
            )
            return counting.calls

        assert study_psi_calls(5) == study_psi_calls(2)


# ---------------------------------------------------------------------------
# #142 fixtures: a rigged covariance whose definiteness is controlled by a
# poison flag planted in the data (column 2), so the ridge-anchor regime is a
# property of each DATASET, not of the model or strategy configuration.
# ---------------------------------------------------------------------------

N_LOC = 64


@jdc.pytree_dataclass
class _Loc:
    mu: jax.Array


def _loc_model(x, theta):
    """Two location moments for one parameter (M=2, K=1, J_dof=1)."""
    return jnp.array([x[0] - theta.mu, x[1] - theta.mu])


@jdc.pytree_dataclass
class _PoisonableCovariance:
    """V keyed on the dataset's poison column ``x[:, 2]``.

    flag 0 -> identity (clean: DiagonalTikhonov takes tau = 0);
    flag 1 -> [[1, 2], [2, 1]], eigenvalues {-1, 3}: indefinite, so the
    regulariser must bisect to tau ~ 1 >> tau_threshold (= 0.01) and the
    anchor BINDS. Definiteness is the dataset's property -- exactly the
    #142 field regime (an indefinite rep-0 V poisoning a whole arm).
    """

    def covariance(self, psi, theta, measure):
        off = 2.0 * measure.x[0, 2]
        return jnp.array([[1.0, off], [off, 1.0]])


def _poison_measure(key: jax.Array, flag) -> EmpiricalMeasure:
    x01 = 1.0 + 0.1 * jax.random.normal(key, (N_LOC, 2))
    col = jnp.full((N_LOC, 1), flag)
    x = jnp.concatenate([x01, col], axis=1)
    return EmpiricalMeasure(x=x, mask=jnp.ones((N_LOC, 2)), weights=jnp.ones(N_LOC))


def _random_poison_dgp():
    """dgp whose poison flag is a fair coin in the rep key."""

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        kx, kf = jax.random.split(key)
        flag = (jax.random.uniform(kf) < 0.5).astype(jnp.float64)
        return _poison_measure(kx, flag)

    return dgp


def _clean_dgp():
    """dgp drawing the same x as :func:`_random_poison_dgp` but never poisoned."""

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        kx, _ = jax.random.split(key)
        return _poison_measure(kx, jnp.asarray(0.0))

    return dgp


def _expected_flags(key: jax.Array, n_reps: int) -> list[bool]:
    """Replay the documented CRN schedule: rep r's flag from fold_in(key, r)."""
    out = []
    for r in range(n_reps):
        _, kf = jax.random.split(jax.random.fold_in(key, r))
        out.append(bool(jax.random.uniform(kf) < 0.5))
    return out


def _make_loc_run(template_flag: float):
    return build_estimator(
        _loc_model,
        measure=_poison_measure(jax.random.PRNGKey(123), template_flag),
        covariance=_PoisonableCovariance(),
        parameters=_Loc(mu=jnp.asarray(1.0)),
    )


def _loc_theta0() -> _Loc:
    return _Loc(mu=jnp.asarray(1.0))


class TestAnchorPerRep:
    """#142: ``replicate(anchor_per_rep=True)`` -- per-dataset anchoring."""

    def test_binding_ridge_varies_per_rep_only_under_flag(self):
        """The headline #142 contract. On the factory path every rep
        inherits the TEMPLATE's frozen anchor (constant binding column:
        an unlucky template poisons the arm, a lucky one masks per-rep
        pathology); under ``anchor_per_rep=True`` binding tracks each
        replicate's OWN V -- the column varies and equals the replayed
        per-rep poison schedule exactly."""
        n_reps = 4
        key = jax.random.PRNGKey(6)  # chosen: flags [F, T, T, F] -- mixed
        flags = _expected_flags(key, n_reps)
        assert 0 < sum(flags) < n_reps  # fixture sanity: both regimes occur
        dgp = _random_poison_dgp()
        theta0 = _loc_theta0()

        # Unlucky template (poisoned, indefinite V): the anchor binds.
        run_poisoned = _make_loc_run(1.0)
        factory = replicate(
            run_poisoned, dgp, n_reps=n_reps, key=key, theta_init=theta0
        )
        per_rep = replicate(
            run_poisoned,
            dgp,
            n_reps=n_reps,
            key=key,
            theta_init=theta0,
            anchor_per_rep=True,
        )

        # Factory path: rep-0-anchor inheritance -> constant 1 (and one
        # shared tau for the whole arm).
        np.testing.assert_array_equal(
            np.asarray(factory.records.binding_ridge), np.ones(n_reps)
        )
        assert float(np.ptp(np.asarray(factory.records.tau_realised))) == 0.0
        # Per-rep path: binding is each dataset's own regime.
        np.testing.assert_array_equal(
            np.asarray(per_rep.records.binding_ridge),
            np.asarray(flags, dtype=float),
        )
        b = np.asarray(per_rep.records.binding_ridge)
        assert 0.0 < b.mean() < 1.0  # VARIES across reps -- the point of #142
        assert float(np.ptp(np.asarray(per_rep.records.tau_realised))) > 0.0

        # Lucky template (clean V): the factory path records binding
        # 0.000 across the SAME poisoned draws -- pathology masked.
        run_clean = _make_loc_run(0.0)
        factory_clean = replicate(
            run_clean, dgp, n_reps=n_reps, key=key, theta_init=theta0
        )
        np.testing.assert_array_equal(
            np.asarray(factory_clean.records.binding_ridge), np.zeros(n_reps)
        )

    def test_records_same_structure_and_shapes(self):
        key = jax.random.PRNGKey(0)
        dgp = _clean_dgp()
        run = _make_loc_run(0.0)
        theta0 = _loc_theta0()
        a = replicate(run, dgp, n_reps=2, key=key, theta_init=theta0)
        b = replicate(
            run, dgp, n_reps=2, key=key, theta_init=theta0, anchor_per_rep=True
        )
        assert isinstance(b, MCRecords)
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
        for la, lb in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
        ):
            assert jnp.shape(la) == jnp.shape(lb)
        assert b.records.theta_flat.shape == (2, 1)
        assert b.records.J_dof == a.records.J_dof == 1
        assert b.param_names == a.param_names == ("mu",)

    def test_crn_schedule_and_theta_equal_when_never_binding(self):
        """The CRN contract is mode-invariant: rep r draws
        dgp(fold_in(key, r)) on the per-rep path too, and with a clean
        template and clean reps (tau = 0 everywhere) the two modes
        differ in NOTHING -- per-rep theta is identical, so any
        divergence between the modes can only come from anchoring."""
        key = jax.random.PRNGKey(5)
        theta0 = _loc_theta0()
        base = _clean_dgp()
        seen: list[np.ndarray] = []

        def recording_dgp(k: jax.Array) -> EmpiricalMeasure:
            m = base(k)
            seen.append(np.asarray(m.x))
            return m

        run = _make_loc_run(0.0)
        a = replicate(run, base, n_reps=3, key=key, theta_init=theta0)
        b = replicate(
            run,
            recording_dgp,
            n_reps=3,
            key=key,
            theta_init=theta0,
            anchor_per_rep=True,
        )
        # Pin the documented schedule on the per-rep path.
        assert len(seen) == 3
        for r in range(3):
            np.testing.assert_array_equal(
                seen[r], np.asarray(base(jax.random.fold_in(key, r)).x)
            )
        # The ridge never binds on either path ...
        np.testing.assert_array_equal(np.asarray(a.records.binding_ridge), np.zeros(3))
        np.testing.assert_array_equal(np.asarray(b.records.binding_ridge), np.zeros(3))
        # ... so theta agrees rep-for-rep across the modes.
        np.testing.assert_allclose(
            np.asarray(a.records.theta_flat),
            np.asarray(b.records.theta_flat),
            rtol=0.0,
            atol=0.0,
        )

    def test_specless_callable_raises_before_any_work(self):
        """anchor_per_rep=True needs build_estimator's attached factory
        spec; a hand-rolled run callable fails LOUDLY, up front (neither
        the dgp nor the callable is ever invoked)."""

        def handrolled(theta_init, measure):
            raise AssertionError("run must not be invoked")

        def dgp(key):
            raise AssertionError("dgp must not be drawn")

        with pytest.raises(ValueError, match="build_estimator"):
            replicate(
                handrolled,
                dgp,
                n_reps=2,
                key=jax.random.PRNGKey(0),
                theta_init=_loc_theta0(),
                anchor_per_rep=True,
            )

    def test_monte_carlo_study_threads_the_flag(self):
        """monte_carlo_study(anchor_per_rep=True) reaches replicate: the
        stacked binding column matches the per-rep poison schedule (it
        would be constant 1.0 on the factory path)."""
        from emu_gmm.studies import monte_carlo_study

        n_reps = 4
        key = jax.random.PRNGKey(6)
        flags = _expected_flags(key, n_reps)
        study = monte_carlo_study(
            _make_loc_run(1.0),
            _random_poison_dgp(),
            n_reps=n_reps,
            key=key,
            theta_init=_loc_theta0(),
            theta0=_loc_theta0(),
            anchor_per_rep=True,
        )
        np.testing.assert_array_equal(
            np.asarray(study.records.records.binding_ridge),
            np.asarray(flags, dtype=float),
        )


class TestReplicateCoupled:
    """#171: the coupled-arms factory stamps a sound shared coupling_id."""

    def test_shared_token_and_crn_pair_accepts(self):
        out = replicate_coupled(
            {"a": _make_run(), "b": _make_run()},
            _make_dgp(),
            n_reps=3,
            key=jax.random.PRNGKey(11),
            theta_init=_theta0(),
        )
        assert set(out) == {"a", "b"}
        # one shared, non-None token across the set
        assert out["a"].coupling_id is not None
        assert out["a"].coupling_id == out["b"].coupling_id
        # crn_pair verifies with NO explicit token and NO assert_coupled
        cp = crn_pair(out["a"], out["b"])
        assert cp.n_reps == 3

    def test_arms_see_identical_crn_draws(self):
        """Every arm sees dgp(fold_in(key, r)) rep-for-rep (the CRN contract)."""
        base = _make_dgp()
        seen: list[np.ndarray] = []

        def recording_dgp(key):
            m = base(key)
            seen.append(np.asarray(m.x))
            return m

        key = jax.random.PRNGKey(5)
        replicate_coupled(
            {"a": _make_run(), "b": _make_run()},
            recording_dgp,
            n_reps=3,
            key=key,
            theta_init=_theta0(),
        )
        assert len(seen) == 6  # 2 arms x 3 reps
        for r in range(3):
            np.testing.assert_array_equal(seen[r], seen[3 + r])  # arm a == arm b
            expected = np.asarray(base(jax.random.fold_in(key, r)).x)
            np.testing.assert_array_equal(seen[r], expected)  # documented schedule

    def test_distinct_calls_get_distinct_tokens(self):
        """Two SEPARATE coupled sets never share a token, even at the same key
        --- so crn_pair cannot false-accept a cross-set pairing (the reason the
        token is per-call-unique, not derived from (key, n_reps))."""
        k = jax.random.PRNGKey(0)
        o1 = replicate_coupled(
            {"a": _make_run()}, _make_dgp(), n_reps=2, key=k, theta_init=_theta0()
        )
        o2 = replicate_coupled(
            {"a": _make_run()}, _make_dgp(), n_reps=2, key=k, theta_init=_theta0()
        )
        assert o1["a"].coupling_id != o2["a"].coupling_id

    def test_explicit_coupling_id_override(self):
        out = replicate_coupled(
            {"a": _make_run(), "b": _make_run()},
            _make_dgp(),
            n_reps=2,
            key=jax.random.PRNGKey(1),
            theta_init=_theta0(),
            coupling_id="study-X",
        )
        assert out["a"].coupling_id == out["b"].coupling_id == "study-X"

    def test_empty_runs_raises(self):
        with pytest.raises(ValueError, match="empty"):
            replicate_coupled(
                {},
                _make_dgp(),
                n_reps=2,
                key=jax.random.PRNGKey(0),
                theta_init=_theta0(),
            )


def _procedure_dgp(key):
    return {"draw": jax.random.normal(key, ())}


def _procedure_run(theta_init, dataset):
    # A procedure's reported quantities need not be an OptimizationResult.
    point = dataset["draw"] + theta_init
    return {"point": point, "reported_J": point**2 + 11.0}


def _procedure_record(result, dataset):
    return FitRecord(
        theta_flat=jnp.atleast_1d(result["point"]),
        se=jnp.array([2.0]),
        J_stat=result["reported_J"],
        J_pvalue=jnp.asarray(0.2),
        J_pvalue_adjusted=jnp.asarray(0.3),
        converged=jnp.asarray(dataset["draw"] > 0, dtype=jnp.float64),
        tau_realised=jnp.asarray(0.0),
        binding_ridge=jnp.asarray(0.0),
        sigma_meat_indefinite=jnp.asarray(0.0),
        tau_saturated=jnp.asarray(0.0),
        J_dof=2,
        param_names=("reported",),
    )


class TestCustomRecords:
    def test_exact_dataset_and_reported_inference_are_preserved(self):
        seen = []

        def run(theta_init, dataset):
            result = _procedure_run(theta_init, dataset)
            seen.append((result, dataset))
            return result

        def record(result, dataset):
            assert (result is seen[-1][0]) and (dataset is seen[-1][1])
            return _procedure_record(result, dataset)

        def statistic(result, dataset):
            assert (result is seen[-1][0]) and (dataset is seen[-1][1])
            return result["point"] - dataset["draw"]

        key = jax.random.PRNGKey(17)
        out = replicate(
            run,
            _procedure_dgp,
            n_reps=4,
            key=key,
            theta_init=1.5,
            record=record,
            statistics={"initial": statistic},
        )
        draws = np.array(
            [float(jax.random.normal(jax.random.fold_in(key, r), ())) for r in range(4)]
        )
        np.testing.assert_array_equal(out.records.theta_flat[:, 0], draws + 1.5)
        np.testing.assert_array_equal(out.records.se, np.full((4, 1), 2.0))
        np.testing.assert_array_equal(out.records.J_stat, (draws + 1.5) ** 2 + 11.0)
        np.testing.assert_array_equal(out.records.J_pvalue, np.full(4, 0.2))
        np.testing.assert_array_equal(out.records.J_pvalue_adjusted, np.full(4, 0.3))
        np.testing.assert_allclose(out.extra["initial"], 1.5, rtol=0, atol=1e-15)
        assert len(seen) == 4
        assert out.records.J_dof == 2
        assert out.param_names == ("reported",)
        assert out.n_converged == int((draws > 0).sum())
        assert out.n_excluded == int((draws <= 0).sum())

    def test_bad_record_extractor_raises_at_the_rep(self):
        with pytest.raises(TypeError, match="FitRecord.*dict.*replicate 0"):
            replicate(
                _procedure_run,
                _procedure_dgp,
                n_reps=2,
                key=jax.random.PRNGKey(0),
                theta_init=0.0,
                record=lambda result, dataset: result,
            )

    def test_custom_record_does_not_bypass_anchor_factory_guard(self):
        def dgp(key):
            pytest.fail("factory guard must run before drawing data")

        with pytest.raises(ValueError, match="build_estimator"):
            replicate(
                _procedure_run,
                dgp,
                n_reps=1,
                key=jax.random.PRNGKey(0),
                theta_init=0.0,
                record=_procedure_record,
                anchor_per_rep=True,
            )

    def test_coupled_forwards_both_extractors_per_arm(self):
        key = jax.random.PRNGKey(9)
        seen = []

        def dgp(rep_key):
            dataset = _procedure_dgp(rep_key)
            seen.append(dataset)
            return dataset

        def record(result, dataset):
            assert dataset is seen[-1]
            return _procedure_record(result, dataset)

        out = replicate_coupled(
            {
                "a": _procedure_run,
                "b": lambda theta, dataset: _procedure_run(theta + 3.0, dataset),
            },
            dgp,
            n_reps=3,
            key=key,
            theta_init=1.0,
            record=record,
            statistics={
                "draw": lambda result, dataset: dataset["draw"],
                "reported_J": lambda result, dataset: result["reported_J"],
            },
        )
        assert len(seen) == 6
        np.testing.assert_array_equal(out["a"].extra["draw"], out["b"].extra["draw"])
        np.testing.assert_allclose(
            out["b"].records.theta_flat - out["a"].records.theta_flat, 3.0
        )
        for arm in out.values():
            np.testing.assert_array_equal(arm.extra["reported_J"], arm.records.J_stat)
        assert crn_pair(out["a"], out["b"]).n_reps == 3

    def test_anchor_path_uses_custom_record_and_statistics(self):
        def record(result, dataset):
            return dataclasses.replace(fit_record(result), J_stat=jnp.asarray(123.0))

        out = replicate(
            _make_loc_run(template_flag=0.0),
            _clean_dgp(),
            n_reps=1,
            key=jax.random.PRNGKey(0),
            theta_init=_loc_theta0(),
            anchor_per_rep=True,
            record=record,
            statistics={"n_obs": lambda result, dataset: dataset.x.shape[0]},
        )
        assert float(out.records.J_stat[0]) == 123.0
        assert int(out.extra["n_obs"][0]) == N_LOC


class TestArrayStatistics:
    def _replicate(self, statistics):
        return replicate(
            _procedure_run,
            _procedure_dgp,
            n_reps=2,
            key=jax.random.PRNGKey(5),
            theta_init=0.0,
            record=_procedure_record,
            statistics=statistics,
        )

    def test_shapes_indexed_export_nan_and_pytree(self):
        out = self._replicate(
            {
                "scalar": lambda result, dataset: dataset["draw"],
                "gbar": lambda result, dataset: jnp.array([dataset["draw"], jnp.nan]),
                "matrix": lambda result, dataset: jnp.array([[1, 2], [3, 4]]),
            }
        )
        assert out.extra["gbar"].shape == (2, 2)
        assert out.extra["matrix"].shape == (2, 2, 2)
        leaves, treedef = jax.tree_util.tree_flatten(out)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(rebuilt.extra["gbar"], out.extra["gbar"])
        frame = out.to_pandas()
        assert frame.index.name == "rep"
        assert list(frame.columns[-7:]) == [
            "scalar",
            "gbar[0]",
            "gbar[1]",
            "matrix[0,0]",
            "matrix[0,1]",
            "matrix[1,0]",
            "matrix[1,1]",
        ]
        np.testing.assert_array_equal(frame["gbar[0]"], out.extra["gbar"][:, 0])
        assert frame["gbar[1]"].isna().all()
        for i, j in np.ndindex(2, 2):
            np.testing.assert_array_equal(
                frame[f"matrix[{i},{j}]"], out.extra["matrix"][:, i, j]
            )

    def test_changing_shape_names_statistic_and_replicate(self):
        calls = []

        def changing(result, dataset):
            calls.append(None)
            return jnp.zeros(len(calls))

        with pytest.raises(
            ValueError,
            match="statistic 'gbar' changed shape at replicate 1.*fixed shape",
        ):
            self._replicate({"gbar": changing})

    @pytest.mark.parametrize(
        "statistics",
        [
            {"J_stat": lambda result, dataset: 0.0},
            {"theta_reported": lambda result, dataset: 0.0},
            {
                "x": lambda result, dataset: jnp.ones(1),
                "x[0]": lambda result, dataset: 0.0,
            },
            {
                "x[0]": lambda result, dataset: 0.0,
                "x": lambda result, dataset: jnp.ones(1),
            },
        ],
    )
    def test_extra_column_collisions_raise(self, statistics):
        out = self._replicate(statistics)
        with pytest.raises(ValueError, match="duplicate output column"):
            out.to_pandas()

    def test_duplicate_parameter_columns_raise(self):
        out = self._replicate(None)
        duplicated = dataclasses.replace(
            out.records,
            theta_flat=jnp.ones((2, 2)),
            se=jnp.ones((2, 2)),
            param_names=("x", "x"),
        )
        with pytest.raises(ValueError, match="duplicate output column 'theta_x'"):
            dataclasses.replace(out, records=duplicated).to_pandas()

    def test_extra_requires_leading_rep_axis(self):
        out = self._replicate(None)
        malformed = dataclasses.replace(out, extra={"gbar": jnp.zeros((3, 2))})
        with pytest.raises(ValueError, match="statistic 'gbar'.*leading rep axis 2"):
            malformed.to_pandas()


def _indexed_shard(indices, *, key=None, coupling_id="sharded-study", extra=True):
    """Distinct row markers make accidental leaf/extra reordering visible."""
    from emu_gmm.types import FitRecord

    r = jnp.asarray(indices, dtype=jnp.float64)
    n = len(indices)
    flags = (r % 2).astype(jnp.float64)
    rec = FitRecord(
        theta_flat=jnp.stack([r, -r], axis=1),
        se=jnp.stack([r + 1, jnp.where(flags > 0, jnp.nan, r + 2)], axis=1),
        J_stat=r + 3,
        J_pvalue=r / 100,
        J_pvalue_adjusted=r / 200,
        converged=1 - flags,
        tau_realised=r / 1000,
        binding_ridge=flags,
        sigma_meat_indefinite=flags,
        tau_saturated=flags,
        J_dof=1,
        param_names=("a", "b"),
    )
    return MCRecords(
        records=rec,
        key=jax.random.PRNGKey(4) if key is None else key,
        n_reps=n,
        coupling_id=coupling_id,
        extra={"marker": r * 10} if extra else None,
        rep_indices=jnp.asarray(indices, dtype=jnp.uint32),
    )


class TestShardedReplication:
    def test_shards_match_unsharded_every_leaf_and_extra(self):
        run, dgp = _make_run(n=40), _make_dgp(n=40)
        kwargs = dict(
            key=jax.random.PRNGKey(11),
            theta_init=_theta0(),
            coupling_id="study",
            statistics={"data_mean": lambda result, measure: jnp.mean(measure.x)},
        )
        whole = replicate(run, dgp, n_reps=4, **kwargs)
        joined = MCRecords.concat(
            [
                replicate(run, dgp, reps=range(2, 4), **kwargs),
                replicate(run, dgp, reps=range(0, 2), **kwargs),
            ],
            reps=range(4),
        )
        for got, expected in zip(
            jax.tree_util.tree_leaves(joined.records),
            jax.tree_util.tree_leaves(whole.records),
            strict=True,
        ):
            np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(
            joined.extra["data_mean"], whole.extra["data_mean"]
        )
        np.testing.assert_array_equal(joined.effective_rep_indices, np.arange(4))

    def test_coupled_global_key_schedule(self):
        base = _make_dgp(n=40)
        seen = []
        key = jax.random.PRNGKey(8)

        def dgp(k):
            seen.append(np.asarray(jax.random.key_data(k)))
            return base(k)

        arms = replicate_coupled(
            {"a": _make_run(n=40), "b": _make_run(n=40)},
            dgp,
            reps=range(10, 12),
            key=key,
            theta_init=_theta0(),
            coupling_id="distributed",
        )
        expected = [np.asarray(jax.random.fold_in(key, r)) for r in range(10, 12)]
        np.testing.assert_array_equal(seen, expected + expected)
        np.testing.assert_array_equal(
            crn_pair(arms["a"], arms["b"]).rep_indices, [10, 11]
        )
        np.testing.assert_array_equal(arms["a"].to_pandas().index, [10, 11])

    @pytest.mark.parametrize(
        "selection",
        [
            {},
            {"n_reps": 2, "reps": range(2)},
            {"n_reps": 0},
            {"n_reps": True},
            {"n_reps": 1.5},
            {"reps": [0, 1]},
            {"reps": range(0)},
            {"reps": range(0, 4, 2)},
            {"reps": range(2, 0, -1)},
            {"reps": range(-1, 1)},
            {"reps": range(2**32, 2**32 + 1)},
        ],
    )
    def test_invalid_selection_fails_before_dgp(self, selection):
        def unexpected(*args):
            raise AssertionError("invalid selection reached drawing/fitting")

        with pytest.raises(ValueError):
            replicate(
                unexpected,
                unexpected,
                key=jax.random.PRNGKey(0),
                theta_init=None,
                **selection,
            )
        with pytest.raises(ValueError):
            replicate_coupled(
                {"arm": unexpected},
                unexpected,
                key=jax.random.PRNGKey(0),
                theta_init=None,
                **selection,
            )

    def test_uint32_upper_boundary_is_accepted(self):
        from emu_gmm.studies.driver import _rep_range

        assert _rep_range(None, range(2**32 - 1, 2**32))[-1] == 2**32 - 1


class TestConcat:
    def test_sorted_rows_keep_flags_nan_and_extras(self):
        joined = MCRecords.concat([_indexed_shard([4, 5]), _indexed_shard([2, 3])])
        expected = _indexed_shard([2, 3, 4, 5])
        for got, want in zip(
            jax.tree_util.tree_leaves(joined.records),
            jax.tree_util.tree_leaves(expected.records),
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(joined.extra["marker"], expected.extra["marker"])
        np.testing.assert_array_equal(joined.to_pandas().index, [2, 3, 4, 5])
        assert joined.n_reps == 4 and joined.n_excluded == 2

    @pytest.mark.parametrize(
        "indices, expected, message",
        [
            ([1, 2], None, "overlapping"),
            ([3, 4], None, "gaps"),
            ([2, 3], range(5), "coverage"),
            ([2, 3], range(1, 4), "coverage"),
        ],
    )
    def test_invalid_coverage(self, indices, expected, message):
        with pytest.raises(ValueError, match=message):
            MCRecords.concat(
                [_indexed_shard([0, 1]), _indexed_shard(indices)], reps=expected
            )

    def test_empty_shards_refused(self):
        with pytest.raises(ValueError, match="no shards"):
            MCRecords.concat([])

    def test_legacy_zero_based_single_shard(self):
        from dataclasses import replace

        legacy = replace(
            _indexed_shard([0, 1], coupling_id=None, extra=False), rep_indices=None
        )
        joined = MCRecords.concat([legacy], reps=range(2))
        np.testing.assert_array_equal(joined.effective_rep_indices, [0, 1])
        assert joined.extra is None

    @pytest.mark.parametrize(
        "mutation, message",
        [
            (lambda s: {"key": jax.random.PRNGKey(9)}, "keys"),
            (lambda s: {"coupling_id": "different"}, "coupling_id"),
            (lambda s: {"coupling_id": None}, "coupling_id"),
            (lambda s: {"n_reps": 3}, "rep_indices"),
            (lambda s: {"rep_indices": jnp.asarray([2.0, 3.0])}, "integer"),
            (lambda s: {"rep_indices": jnp.asarray([2, 2])}, "duplicate"),
            (lambda s: {"rep_indices": jnp.asarray([-1, 2])}, "uint32"),
            (lambda s: {"rep_indices": jnp.asarray([2, 2**32])}, "uint32"),
            (lambda s: {"extra": {"other": jnp.ones(2)}}, "names"),
            (lambda s: {"extra": {"marker": jnp.ones(3)}}, "shapes"),
            (lambda s: {"extra": {"marker": jnp.ones(2, dtype=jnp.float32)}}, "dtypes"),
        ],
    )
    def test_metadata_mismatch(self, mutation, message):
        from dataclasses import replace

        second = _indexed_shard([2, 3])
        with pytest.raises(ValueError, match=message):
            MCRecords.concat(
                [_indexed_shard([0, 1]), replace(second, **mutation(second))]
            )

    @pytest.mark.parametrize(
        "updates, message",
        [
            ({"param_names": ("x", "y")}, "param_names"),
            ({"J_dof": 2}, "J_dof"),
            ({"J_stat": jnp.ones(3)}, "shapes"),
            ({"J_stat": jnp.ones(2, dtype=jnp.float32)}, "dtypes"),
            ({"theta_flat": jnp.ones((2, 3))}, "shapes"),
        ],
    )
    def test_record_mismatch(self, updates, message):
        from dataclasses import replace

        second = _indexed_shard([2, 3])
        bad = replace(second, records=replace(second.records, **updates))
        with pytest.raises(ValueError, match=message):
            MCRecords.concat([_indexed_shard([0, 1]), bad])

    def test_single_malformed_record_shape_refused(self):
        from dataclasses import replace

        shard = _indexed_shard([0, 1])
        bad = replace(shard, records=replace(shard.records, J_stat=jnp.ones((2, 1))))
        with pytest.raises(ValueError, match="shape"):
            MCRecords.concat([bad])

    def test_multiple_unknown_tokens_refused(self):
        with pytest.raises(ValueError, match="coupling_id"):
            MCRecords.concat(
                [
                    _indexed_shard([0], coupling_id=None),
                    _indexed_shard([1], coupling_id=None),
                ]
            )

    def test_typed_and_legacy_keys_compare_by_data(self):
        joined = MCRecords.concat(
            [
                _indexed_shard([0], key=jax.random.key(4)),
                _indexed_shard([1], key=jax.random.PRNGKey(4)),
            ]
        )
        np.testing.assert_array_equal(joined.effective_rep_indices, [0, 1])

    def test_different_prng_implementation_refused(self):
        first = jax.random.key(4, impl="rbg")
        second = jax.random.wrap_key_data(jax.random.key_data(first), impl="unsafe_rbg")
        with pytest.raises(ValueError, match="keys"):
            MCRecords.concat(
                [_indexed_shard([0], key=first), _indexed_shard([1], key=second)]
            )


def test_sharded_anchor_cache_clear_uses_local_counter(monkeypatch):
    import emu_gmm.estimator as estimator_module
    import emu_gmm.studies.driver as driver_module

    record = jax.tree_util.tree_map(lambda x: x[0], _indexed_shard([0]).records)
    draws = []
    cleared_at = []

    def run(*args):
        raise AssertionError("per-rep anchor must use estimate")

    run._emu_gmm_factory_spec = dict.fromkeys(
        [
            "model",
            "covariance",
            "weighting",
            "regularization",
            "optimizer",
            "moment_names",
            "penalty",
        ]
    )

    def dgp(key):
        draws.append(key)
        return None

    monkeypatch.setattr(estimator_module, "estimate", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver_module, "fit_record", lambda result: record)
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared_at.append(len(draws)))
    result = replicate(
        run,
        dgp,
        reps=range(10, 36),
        key=jax.random.PRNGKey(0),
        theta_init=None,
        anchor_per_rep=True,
    )
    assert cleared_at == [25]
    assert result.n_reps == 26


def test_concat_preserves_first_shard_extra_column_order():
    from dataclasses import replace

    first, second = _indexed_shard([0]), _indexed_shard([1])
    first = replace(
        first, extra={"z_last": jnp.asarray([0.0]), "a_first": jnp.asarray([1.0])}
    )
    second = replace(
        second, extra={"a_first": jnp.asarray([2.0]), "z_last": jnp.asarray([3.0])}
    )
    result = MCRecords.concat([first, second])
    assert list(result.extra) == ["z_last", "a_first"]
    np.testing.assert_array_equal(result.extra["z_last"], [0.0, 3.0])
    np.testing.assert_array_equal(result.extra["a_first"], [1.0, 2.0])
