"""Layer-1 driver contracts (#114): shapes, CRN, reproducibility,
non-convergence accounting, and the inherited no-retrace property."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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
