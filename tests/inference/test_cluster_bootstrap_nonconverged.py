"""Non-converged refits are NaN rows in the cluster bootstrap (#209).

A cluster-resampled draw can extinguish a rare moment's support
entirely. The refit's ridge then saturates, its criterion is NaN by
design (#138), and the optimiser exits at step 1 without moving -- so
``estimate()`` returns ``converged=False`` with ``theta_hat`` equal to
``theta_init``, *finite*. ``cluster_bootstrap`` used to store that row
unconditionally: a degenerate atom at the initialisation point that the
NaN-based validity accounting of ``adaptive_bootstrap`` (``n_invalid``)
could not see. This module guards the fix: every ``converged=False``
refit is a NaN row, named by ``convergence`` and explained by the
per-replicate ``zero_support`` / ``tau_saturated`` event flags.

Fixture: a location/scale model whose third moment duplicates the first
but is *supported in one cluster only*, so a draw that omits that
cluster (probability ``(1 - 1/C)^C``, about 35% at ``C = 12``) leaves
``N_3 = 0`` in the resampled world.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm.covariance import ClusteredCovariance
from emu_gmm.inference import (
    BootstrapSE,
    ClusterBootstrapResult,
    adaptive_bootstrap,
    cluster_bootstrap,
)
from emu_gmm.measures import EmpiricalMeasure

_N, _C = 60, 12
_MU0, _S20 = 0.5, 2.0  # theta_init -- deliberately away from the truth


@jdc.pytree_dataclass
class _LocScale:
    mu: jnp.ndarray
    s2: jnp.ndarray


def _model(x, th):
    # Moments 0 and 2 are the same location condition; moment 2 is the
    # rare one (its support is controlled by the mask).
    return jnp.array([x[0] - th.mu, (x[0] - th.mu) ** 2 - th.s2, x[0] - th.mu])


def _init() -> _LocScale:
    return _LocScale(mu=jnp.asarray(_MU0), s2=jnp.asarray(_S20))


def _setup(rare: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 1.5, size=(_N, 1))
    cid = np.repeat(np.arange(_C), _N // _C).astype(np.float64)
    mask = np.ones((_N, 3))
    if rare:
        # Moment 2 supported on three rows of cluster 0 only.
        mask[:, 2] = 0.0
        mask[[0, 1, 2], 2] = 1.0
    measure = EmpiricalMeasure(
        x=jnp.asarray(x), mask=jnp.asarray(mask), weights=jnp.ones(_N)
    )
    cov = ClusteredCovariance(cluster_ids=jnp.asarray(cid), n_clusters=_C)
    return measure, cov


def _boot(rare: bool, n_boot: int = 12, seed: int = 1) -> ClusterBootstrapResult:
    measure, cov = _setup(rare)
    with warnings.catch_warnings():
        # The #138 per-refit warning and the #209 summary warning are both
        # expected on the rare fixture; they are asserted on separately.
        warnings.simplefilter("ignore", UserWarning)
        return cluster_bootstrap(
            model=_model,
            theta_init=_init(),
            measure=measure,
            covariance=cov,
            n_boot=n_boot,
            key=jax.random.PRNGKey(seed),
        )


# ---------------------------------------------------------------------------
# The regression: zero-support draws are NaN rows, not theta_init rows.
# ---------------------------------------------------------------------------


class TestZeroSupportDrawsAreNaNRows:
    def test_fixture_exercises_both_regimes(self):
        # The fixture must produce some zero-support draws AND some healthy
        # ones under the pinned key, or the tests below prove nothing.
        r = _boot(rare=True)
        assert 0 < r.n_zero_support < r.theta_boot.array.shape[0]
        assert r.n_valid > 0

    def test_nonconverged_rows_are_nan_and_converged_rows_finite(self):
        r = _boot(rare=True)
        tb = np.asarray(r.theta_boot.array)
        J = np.asarray(r.J_boot)
        conv = np.asarray(r.convergence)
        row_nan = np.all(np.isnan(tb), axis=1)
        row_finite = np.all(np.isfinite(tb), axis=1)
        # NaN row  <=>  convergence == False; no half-NaN rows.
        np.testing.assert_array_equal(row_nan, ~conv)
        np.testing.assert_array_equal(row_finite, conv)
        np.testing.assert_array_equal(np.isnan(J), ~conv)

    def test_no_row_sits_at_theta_init(self):
        # The #209 atom: before the fix every zero-support draw stored the
        # finite, unmoved theta_init. Now no row equals it.
        r = _boot(rare=True)
        tb = np.asarray(r.theta_boot.array)
        at_init = np.all(np.isclose(tb, [_MU0, _S20]), axis=1)
        assert not np.any(at_init)
        # The converged rows are genuine refits: finite and moved off the
        # start in the location coordinate (theta_init sits well below the
        # sample mean on this fixture).
        conv = np.asarray(r.convergence)
        assert np.all(np.isfinite(tb[conv]))
        assert np.all(tb[conv, 0] != _MU0)

    def test_event_flags_explain_the_failures(self):
        r = _boot(rare=True)
        conv = np.asarray(r.convergence)
        zs = np.asarray(r.zero_support)
        ts = np.asarray(r.tau_saturated)
        assert zs.dtype == bool and ts.dtype == bool
        assert zs.shape == conv.shape == ts.shape
        # A zero-support world never converges ...
        assert not np.any(zs & conv)
        # ... and on this fixture it is the only failure mode.
        np.testing.assert_array_equal(zs, ~conv)
        # A zero diagonal entry is invariant under the ridge, so every
        # zero-support refit also saturates (#205's 139/139).
        assert np.all(ts[zs])

    def test_counts_are_exclude_and_count(self):
        r = _boot(rare=True)
        n_boot = int(r.theta_boot.array.shape[0])
        conv = np.asarray(r.convergence)
        assert r.n_nonconverged == int((~conv).sum())
        assert r.n_valid == int(conv.sum())
        assert r.n_valid + r.n_nonconverged == n_boot
        assert r.n_zero_support == int(np.asarray(r.zero_support).sum())
        assert r.n_tau_saturated == int(np.asarray(r.tau_saturated).sum())
        assert r.n_tau_saturated >= r.n_zero_support

    def test_summary_warning_reports_counts(self):
        measure, cov = _setup(rare=True)
        with pytest.warns(
            UserWarning, match=r"cluster_bootstrap: \d+ of 12 .*did not converge"
        ) as rec:
            r = cluster_bootstrap(
                model=_model,
                theta_init=_init(),
                measure=measure,
                covariance=cov,
                n_boot=12,
                key=jax.random.PRNGKey(1),
            )
        msgs = [str(w.message) for w in rec if "cluster_bootstrap:" in str(w.message)]
        assert len(msgs) == 1  # one summary warning per call, not per draw
        assert f"{r.n_nonconverged} of 12" in msgs[0]
        assert f"{r.n_zero_support} with a zero-support moment" in msgs[0]
        assert "#209" in msgs[0]


# ---------------------------------------------------------------------------
# The documented adaptive feed now sees the invalid draws.
# ---------------------------------------------------------------------------


class TestAdaptiveFeedSeesInvalidDraws:
    def test_raw_theta_column_is_excluded_and_counted(self):
        measure, cov = _setup(rare=True)
        seen: list[int] = []

        def draw(key, size):
            # The documented feed (adaptive.py module docstring): raw
            # theta_boot columns, no caller-side filtering.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res = cluster_bootstrap(
                    model=_model,
                    theta_init=_init(),
                    measure=measure,
                    covariance=cov,
                    n_boot=int(size),
                    key=key,
                )
            seen.append(res.n_nonconverged)
            return np.asarray(res.theta_boot.array[:, 0])

        r = adaptive_bootstrap(
            draw,
            BootstrapSE(),
            key=jax.random.PRNGKey(3),
            batch_size=12,
            b_min=4,
            b_max=12,
            atol=1e3,  # converge on the first batch
        )
        assert r.n_boot == 12
        assert r.n_valid + r.n_invalid == r.n_boot
        # Every non-converged refit is an invalid (NaN) draw to the reducer
        # -- before #209 these were finite theta_init rows and n_invalid
        # read 0.
        assert r.n_invalid == sum(seen)
        assert r.n_invalid > 0
        assert np.isfinite(r.value) and r.value > 0.0


# ---------------------------------------------------------------------------
# A healthy fixture is untouched: all rows finite, no flags, no warning.
# ---------------------------------------------------------------------------


class TestHealthyFixtureUnchanged:
    def test_all_converged_no_flags_no_warning(self):
        measure, cov = _setup(rare=False)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = cluster_bootstrap(
                model=_model,
                theta_init=_init(),
                measure=measure,
                covariance=cov,
                n_boot=6,
                key=jax.random.PRNGKey(1),
            )
        assert np.all(np.asarray(r.convergence))
        assert np.all(np.isfinite(np.asarray(r.theta_boot.array)))
        assert np.all(np.isfinite(np.asarray(r.J_boot)))
        assert not np.any(np.asarray(r.zero_support))
        assert r.n_nonconverged == 0 and r.n_zero_support == 0
        assert r.n_valid == 6
        assert not [w for w in rec if "cluster_bootstrap:" in str(w.message)]


# ---------------------------------------------------------------------------
# Pytree contract with the additive flag fields.
# ---------------------------------------------------------------------------


class TestPytreeWithEventFlags:
    def test_round_trip_and_stacking(self):
        r1 = _boot(rare=True, n_boot=4, seed=1)
        r2 = _boot(rare=True, n_boot=4, seed=2)
        leaves, treedef = jax.tree_util.tree_flatten(r1)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(
            np.asarray(rebuilt.zero_support), np.asarray(r1.zero_support)
        )
        np.testing.assert_array_equal(
            np.asarray(rebuilt.tau_saturated), np.asarray(r1.tau_saturated)
        )
        # The canonical batching gesture (api-sketch.org) still works: the
        # counts are properties, not static fields, so two results with
        # different failure counts share a treedef.
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), r1, r2)
        assert isinstance(stacked, ClusterBootstrapResult)
        assert stacked.zero_support.shape == (2, 4)
        assert stacked.tau_saturated.shape == (2, 4)
        assert stacked.theta_boot.array.shape == (2, 4, 2)

    def test_fields_default_to_none_for_pre_209_records(self):
        # Additive fields: an older record constructed without them still
        # builds, and the counts degrade to zero rather than raising.
        r = _boot(rare=False, n_boot=3)
        old = ClusterBootstrapResult(
            theta_boot=r.theta_boot,
            J_boot=r.J_boot,
            convergence=r.convergence,
            key=r.key,
            param_names=r.param_names,
        )
        assert old.zero_support is None and old.tau_saturated is None
        assert old.n_zero_support == 0 and old.n_tau_saturated == 0
        assert old.n_nonconverged == 0 and old.n_valid == 3
