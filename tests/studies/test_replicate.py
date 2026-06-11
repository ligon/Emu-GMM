"""Layer-1 driver contracts (#114): shapes, CRN, reproducibility,
non-convergence accounting, the inherited no-retrace property, and the
per-replicate ridge-anchoring mode (#142)."""

from __future__ import annotations

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
from emu_gmm.studies import MCRecords, replicate

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
            J_dof=1,
            param_names=("beta", "gamma"),
        )
        recs = MCRecords(records=rec, key=jax.random.PRNGKey(0), n_reps=n)
        df = recs.to_pandas()
        np.testing.assert_array_equal(
            df["sigma_meat_indefinite"].to_numpy(), np.array([0.0, 1.0, 0.0])
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
