"""Tests for ``emu_gmm.law`` --- the ``EstimatorLaw`` carrier (#144).

The charter §2 criteria, each as a decidable test:

* §2.1  exports: ``EstimatorLaw`` / ``EmpiricalLaw`` / ``AsymptoticLaw`` in
  ``emu_gmm.__all__``.
* §2.2  retrofit: the summarizers + the ``Bootstrap*`` functionals route
  through the carrier with NUMERIC-IDENTICAL results (regression-pinned).
* §2.3  ``given(event)`` returns an ``EstimatorLaw`` at the empirical grade and
  REFUSES at the asymptotic grade (no silent approximation).
* §2.4  the coupling constructor verifies key/provenance before zipping; a
  mismatch raises.
* §2.5  ``PSDFixedRank`` Gamma functionals (eigenvalues / gamma) queryable
  through the law, manifold/gauge-aware.
* §2.6a a K-Aggregators-style het result (point + analytic SE + cluster-wild J
  + MC sweep) routed through ONE interface, with ``given`` on a named event
  (``sigma_meat_indefinite`` / ``binding_ridge``) that fires PARTIALLY.
* §2.6b the #130-harness summarizers + data exercise the carrier.

The event-firing fixtures are REAL Emu MC sweeps of a het/degenerate DGP (no
declared flags): a location model with a poison covariance whose indefinite
direction aligns with the moment gradient, so ``sigma_meat_indefinite`` (and
``binding_ridge``) fire partially under the #138 diagnose-loudly policy --- the
exact event that NaN's the K-Agg het lambda_2 SE.
"""

from __future__ import annotations

import warnings

import emu_gmm
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AsymptoticLaw,
    EmpiricalLaw,
    EstimatorLaw,
    build_estimator,
    couple,
    estimate,
)
from emu_gmm.inference.adaptive import (
    BootstrapMean,
    BootstrapPValue,
    BootstrapQuantile,
    BootstrapSE,
)
from emu_gmm.law import eigenvalue_functional, gamma_functional
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.measures import EmpiricalMeasure
from emu_gmm.studies import (
    SelectionConditionalWarning,
    bias_sd,
    coverage,
    crn_pair,
    given,
    j_calibration,
    replicate,
    replicate_coupled,
    size_power,
    tau_binding,
)

# ---------------------------------------------------------------------------
# Scalar location model with a poison covariance (the het/degenerate DGP).
# ---------------------------------------------------------------------------
N_LOC = 64


@jdc.pytree_dataclass
class _Loc:
    mu: jax.Array


def _loc_model(x, theta):
    """M=2, K=1 location moments; gradient G is along (1, 1)."""
    return jnp.array([x[0] - theta.mu, x[1] - theta.mu])


@jdc.pytree_dataclass
class _MeatPoisonCov:
    """V keyed on the dataset's poison column ``x[:, 2]``.

    flag 0 -> identity (clean); flag 1 -> ``[[1, -1.2], [-1.2, 1]]`` with
    eigenpairs ``(2.2, (1, -1))`` and ``(-0.2, (1, 1))``. The moment gradient
    ``G`` is along ``(1, 1)`` --- the NEGATIVE direction --- so the sandwich
    meat is indefinite (``sigma_meat_indefinite`` fires, SE NaN'd by the #138
    policy) AND the ridge binds to restore PD (``binding_ridge`` fires).
    """

    def covariance(self, psi, theta, measure):
        off = -1.2 * measure.x[0, 2]
        return jnp.array([[1.0, off], [off, 1.0]])


@jdc.pytree_dataclass
class _CleanCov:
    def covariance(self, psi, theta, measure):
        del psi, theta, measure
        return jnp.eye(2)


def _poison_measure(key: jax.Array, flag) -> EmpiricalMeasure:
    x01 = 1.0 + 0.1 * jax.random.normal(key, (N_LOC, 2))
    col = jnp.full((N_LOC, 1), flag)
    x = jnp.concatenate([x01, col], axis=1)
    return EmpiricalMeasure(x=x, mask=jnp.ones((N_LOC, 2)), weights=jnp.ones(N_LOC))


def _meat_poison_dgp():
    """A fair-coin poison flag per replicate key."""

    def dgp(key: jax.Array) -> EmpiricalMeasure:
        kx, kf = jax.random.split(key)
        flag = (jax.random.uniform(kf) < 0.5).astype(jnp.float64)
        return _poison_measure(kx, flag)

    return dgp


def _clean_dgp():
    def dgp(key: jax.Array) -> EmpiricalMeasure:
        kx, _ = jax.random.split(key)
        return _poison_measure(kx, jnp.asarray(0.0))

    return dgp


def _make_poison_run(template_flag: float):
    return build_estimator(
        _loc_model,
        measure=_poison_measure(jax.random.PRNGKey(123), template_flag),
        covariance=_MeatPoisonCov(),
        parameters=_Loc(mu=jnp.asarray(1.0)),
    )


def _make_clean_run():
    return build_estimator(
        _loc_model,
        measure=_poison_measure(jax.random.PRNGKey(123), 0.0),
        covariance=_CleanCov(),
        parameters=_Loc(mu=jnp.asarray(1.0)),
    )


def _theta0() -> _Loc:
    return _Loc(mu=jnp.asarray(1.0))


@pytest.fixture(scope="module")
def poison_sweep():
    """A 12-rep sweep firing ``sigma_meat_indefinite`` / ``binding_ridge`` ~50%.

    ``anchor_per_rep=True`` makes the binding regime a property of each
    DATASET (the #142 path), so the events vary across reps; key=PRNGKey(6)
    yields a mixed (partial-firing) schedule.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = replicate(
            _make_poison_run(1.0),
            _meat_poison_dgp(),
            n_reps=12,
            key=jax.random.PRNGKey(6),
            theta_init=_theta0(),
            anchor_per_rep=True,
        )
    return mc


@pytest.fixture(scope="module")
def clean_scalar_result():
    """A clean scalar (M=2, K=1) estimate for the asymptotic-grade API tests."""
    measure = _poison_measure(jax.random.PRNGKey(1), 0.0)
    return estimate(
        _loc_model,
        measure,
        covariance=_CleanCov(),
        parameters=_theta0(),
    )


# ---------------------------------------------------------------------------
# A wild-bootstrap law of Q (the cluster-wild J grade; carrier #4).
# ---------------------------------------------------------------------------
@jdc.pytree_dataclass
class _ScalarP:
    a: float


def _overid_psi(x, theta):
    """M=2 over-identified residual for one mean parameter (K=1)."""
    return jnp.array([x[0] - theta.a, x[1] - theta.a])


@pytest.fixture(scope="module")
def wild_J_boot():
    """A real cluster-wild bootstrap J-statistic stack."""
    from emu_gmm.covariance.clustered import ClusteredCovariance
    from emu_gmm.inference import moment_wild_bootstrap

    n, n_clusters = 200, 20
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (n, 2))
    measure = EmpiricalMeasure(x=x, mask=jnp.ones((n, 2)), weights=jnp.ones(n))
    cluster_ids = jnp.repeat(jnp.arange(n_clusters, dtype=jnp.float64), n // n_clusters)
    cov = ClusteredCovariance(cluster_ids=cluster_ids, n_clusters=n_clusters)
    res = moment_wild_bootstrap(
        _overid_psi,
        _ScalarP(a=0.0),
        measure,
        cov,
        n_boot=400,
        key=jax.random.PRNGKey(7),
    )
    return np.asarray(res.J_boot), float(res.J_observed)


# ---------------------------------------------------------------------------
# §2.1 --- exports.
# ---------------------------------------------------------------------------
class TestExports:
    def test_classes_in_top_level_all(self):
        for name in ("EstimatorLaw", "EmpiricalLaw", "AsymptoticLaw"):
            assert name in emu_gmm.__all__
            assert getattr(emu_gmm, name) is not None

    def test_grade_instances_are_estimator_laws(self):
        assert issubclass(EmpiricalLaw, EstimatorLaw)
        assert issubclass(AsymptoticLaw, EstimatorLaw)
        assert EmpiricalLaw.grade == "empirical"
        assert AsymptoticLaw.grade == "asymptotic"


# ---------------------------------------------------------------------------
# §2.2 --- the Bootstrap* functionals route through the carrier identically.
# ---------------------------------------------------------------------------
class TestBootstrapRetrofit:
    def _law_and_col(self):
        rng = np.random.default_rng(0)
        vals = np.concatenate([rng.normal(5.0, 1.3, 300), [np.nan]])  # one invalid
        law = EmpiricalLaw.from_draws(vals, names=("J",))
        finite = vals[np.isfinite(vals)]
        return law, finite

    def test_se_routes_through_bootstrap_se(self):
        law, finite = self._law_and_col()
        assert law.se()[0] == BootstrapSE().evaluate(finite)[0]

    def test_mean_routes_through_bootstrap_mean(self):
        law, finite = self._law_and_col()
        assert law.mean()[0] == BootstrapMean().evaluate(finite)[0]

    @pytest.mark.parametrize("q", [0.025, 0.5, 0.975])
    def test_quantile_routes_through_bootstrap_quantile(self, q):
        law, finite = self._law_and_col()
        assert law.quantile(q)[0] == BootstrapQuantile(q).evaluate(finite)[0]

    def test_pvalue_routes_through_bootstrap_pvalue(self):
        law, finite = self._law_and_col()
        assert law.pvalue(6.0) == BootstrapPValue(6.0, "greater").evaluate(finite)[0]

    def test_invalid_rows_excluded_but_counted(self):
        law, finite = self._law_and_col()
        assert law.n_used == finite.size
        assert law.n_draws == finite.size + 1  # the NaN row is kept in the denom


# ---------------------------------------------------------------------------
# §2.2 --- the studies summarizers route through the carrier identically.
# ---------------------------------------------------------------------------
class TestSummariesRetrofit:
    def test_bias_sd_identical(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        a = law.bias_sd(_theta0())
        b = bias_sd(poison_sweep, _theta0())
        np.testing.assert_array_equal(a.bias, b.bias)
        np.testing.assert_array_equal(a.mc_sd, b.mc_sd)
        np.testing.assert_array_equal(a.mean_se, b.mean_se)
        np.testing.assert_array_equal(a.n_valid_se, b.n_valid_se)

    def test_coverage_identical(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        a = law.coverage(_theta0())
        b = coverage(poison_sweep, _theta0())
        np.testing.assert_array_equal(a.coverage, b.coverage)
        np.testing.assert_array_equal(a.n_valid_se, b.n_valid_se)

    def test_size_power_identical(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        a = law.size_power()
        b = size_power(poison_sweep)
        np.testing.assert_array_equal(a.reject_nominal, b.reject_nominal)
        np.testing.assert_array_equal(a.reject_adjusted, b.reject_adjusted)

    def test_tau_binding_identical(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        a = law.tau_binding()
        b = tau_binding(poison_sweep)
        assert a.binding_frequency == b.binding_frequency
        np.testing.assert_array_equal(a.tau_quantiles, b.tau_quantiles)

    def test_j_calibration_identical(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        a = law.j_calibration()
        b = j_calibration(poison_sweep)
        np.testing.assert_array_equal(a.deviation, b.deviation)

    def test_summaries_refuse_on_raw_draws_law(self):
        law = EmpiricalLaw.from_draws(np.arange(10.0), names=("J",))
        with pytest.raises(TypeError, match="records-backed"):
            law.bias_sd(_theta0())


# ---------------------------------------------------------------------------
# §2.3 --- given() returns an EstimatorLaw (empirical); asymptotic REFUSES.
# ---------------------------------------------------------------------------
class TestGivenConditional:
    def test_given_returns_estimator_law(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        cond = law.given("binding_ridge", acknowledge_conditional=True)
        assert isinstance(cond, EmpiricalLaw)
        assert isinstance(cond, EstimatorLaw)
        assert cond.conditioned is True

    def test_given_selects_event_subpopulation(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        es = law.event_share("binding_ridge")
        cond = law.given("binding_ridge", acknowledge_conditional=True)
        assert cond.n_draws == es.n_selected

    def test_event_fires_partially(self, poison_sweep):
        """The fixture's named events fire in SOME-but-not-all reps (a
        non-vacuous conditional; charter §6 verify-first)."""
        law = EmpiricalLaw.from_records(poison_sweep)
        for flag in ("binding_ridge", "sigma_meat_indefinite"):
            frac = law.event_share(flag).fraction_all
            assert 0.0 < frac < 1.0, f"{flag} did not fire partially ({frac})"

    def test_conditional_coverage_identical_to_free_function(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = law.given("binding_ridge", acknowledge_conditional=True)
            ref = coverage(
                given(poison_sweep, "binding_ridge", acknowledge_conditional=True),
                _theta0(),
            )
        np.testing.assert_array_equal(cond.coverage(_theta0()).coverage, ref.coverage)

    def test_given_preserves_selection_conditional_warning(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        with pytest.warns(SelectionConditionalWarning, match="SELECTION-CONDITIONAL"):
            law.given("sigma_meat_indefinite")

    def test_given_acknowledge_silences_warning(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            law.given("binding_ridge", acknowledge_conditional=True)

    def test_given_partition(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = law.given("binding_ridge").n_draws
            b = law.given("binding_ridge", negate=True).n_draws
        assert a + b == law.n_draws

    def test_given_on_raw_draws_refuses(self):
        law = EmpiricalLaw.from_draws(np.arange(10.0), names=("J",))
        with pytest.raises(TypeError, match="event flags"):
            law.given("binding_ridge")

    def test_asymptotic_given_refuses(self, clean_scalar_result):
        """§2.3: the conditional has no closed form at the Gaussian grade ---
        the implementation refuses rather than approximates."""
        law = AsymptoticLaw(clean_scalar_result)
        with pytest.raises(NotImplementedError, match="refuses"):
            law.given("binding_ridge")


# ---------------------------------------------------------------------------
# §2.4 --- couple() verifies key/provenance before zipping; mismatch raises.
# ---------------------------------------------------------------------------
class TestCoupleProvenance:
    def _coupled_arms(self, *, coupling_id=None):
        return replicate_coupled(
            {"a": _make_clean_run(), "b": _make_clean_run()},
            _clean_dgp(),
            n_reps=6,
            key=jax.random.PRNGKey(3),
            theta_init=_theta0(),
            coupling_id=coupling_id,
        )

    def test_couple_accepts_coupled_arms(self):
        arms = self._coupled_arms(coupling_id="study1")
        cp = couple(
            EmpiricalLaw.from_records(arms["a"]),
            EmpiricalLaw.from_records(arms["b"]),
        )
        assert cp.n_reps == 6

    def test_couple_method_equals_free_function(self):
        arms = self._coupled_arms(coupling_id="study1")
        la, lb = (
            EmpiricalLaw.from_records(arms["a"]),
            EmpiricalLaw.from_records(arms["b"]),
        )
        assert couple(la, lb).n_reps == la.couple(lb).n_reps

    def test_couple_mismatched_id_raises(self):
        a = replicate(
            _make_clean_run(),
            _clean_dgp(),
            n_reps=6,
            key=jax.random.PRNGKey(3),
            theta_init=_theta0(),
            coupling_id="s1",
        )
        b = replicate(
            _make_clean_run(),
            _clean_dgp(),
            n_reps=6,
            key=jax.random.PRNGKey(3),
            theta_init=_theta0(),
            coupling_id="s2",
        )
        with pytest.raises(ValueError, match="coupling_id mismatch"):
            couple(EmpiricalLaw.from_records(a), EmpiricalLaw.from_records(b))

    def test_couple_refuses_conditioned_law(self):
        arms = self._coupled_arms(coupling_id="s")
        la = EmpiricalLaw.from_records(arms["a"])
        lb = EmpiricalLaw.from_records(arms["b"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = la.given("binding_ridge")
        with pytest.raises(TypeError, match="conditioned"):
            couple(cond, lb)

    def test_couple_refuses_raw_draws_law(self):
        arms = self._coupled_arms(coupling_id="s")
        raw = EmpiricalLaw.from_draws(np.arange(6.0), names=("J",))
        with pytest.raises(TypeError, match="records-backed"):
            couple(raw, EmpiricalLaw.from_records(arms["a"]))

    def test_couple_returns_crn_pair_object(self):
        arms = self._coupled_arms(coupling_id="s")
        la = EmpiricalLaw.from_records(arms["a"])
        lb = EmpiricalLaw.from_records(arms["b"])
        ref = crn_pair(arms["a"], arms["b"])
        cp = la.couple(lb)
        assert type(cp) is type(ref)
        assert cp.coupling_id == ref.coupling_id


# ---------------------------------------------------------------------------
# Asymptotic-grade API (the Gaussian closed forms).
# ---------------------------------------------------------------------------
class TestAsymptoticGrade:
    def test_mean_cov_se_identity(self, clean_scalar_result):
        law = AsymptoticLaw(clean_scalar_result)
        np.testing.assert_allclose(
            law.cov(), np.asarray(clean_scalar_result.Sigma_theta.array)
        )
        np.testing.assert_allclose(
            law.se(), np.asarray(clean_scalar_result.standard_errors.array)
        )

    def test_quantile_is_gaussian_marginal(self, clean_scalar_result):
        law = AsymptoticLaw(clean_scalar_result)
        import scipy.stats

        z = float(scipy.stats.norm.ppf(0.975))
        np.testing.assert_allclose(law.quantile(0.975), law.mean() + z * law.se())

    def test_prob_refuses(self, clean_scalar_result):
        law = AsymptoticLaw(clean_scalar_result)
        with pytest.raises(NotImplementedError, match="refuses"):
            law.prob(lambda v: v[0] > 0.0)

    def test_sample_shape(self, clean_scalar_result):
        law = AsymptoticLaw(clean_scalar_result)
        draws = law.sample(jax.random.PRNGKey(0), 50)
        assert draws.shape == (50, len(law.param_names))

    def test_wraps_only_estimation_result(self):
        with pytest.raises(TypeError, match="EstimationResult"):
            AsymptoticLaw(object())


# ---------------------------------------------------------------------------
# The law of Q (carrier #4): cluster-wild J through the EmpiricalLaw interface.
# ---------------------------------------------------------------------------
class TestLawOfQ:
    def test_pvalue_matches_bootstrap_pvalue(self, wild_J_boot):
        J_boot, J_obs = wild_J_boot
        law = EmpiricalLaw.from_draws(J_boot, names=("J",))
        finite = J_boot[np.isfinite(J_boot)]
        assert (
            law.pvalue(J_obs) == BootstrapPValue(J_obs, "greater").evaluate(finite)[0]
        )

    def test_prob_tail_is_a_valid_probability(self, wild_J_boot):
        J_boot, J_obs = wild_J_boot
        law = EmpiricalLaw.from_draws(J_boot, names=("J",))
        p = law.prob(lambda v: v[0] >= J_obs)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# §2.5 / §2.6a (gauge-aware codomain + the multi-grade het result).
# A PSDFixedRank(5, K) het estimate; module-scoped (one solve), marked slow.
# ---------------------------------------------------------------------------
_N = 5
_K = 2
_TRIU = jnp.array(np.triu_indices(_N)).T


@jdc.pytree_dataclass
class _ProductParams:
    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _psd_components():
    from emu_gmm.manifolds import Euclidean, PSDFixedRank
    from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf

    def mk(Y, phi):
        return _ProductParams(
            Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(_N, _K)),
            phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
        )

    return mk


def _gauge_invariant_model(x, theta):
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


@pytest.fixture(scope="module")
def psd_result():
    from emu_gmm import SyntheticCovariance
    from emu_gmm.manifolds.riemannian_lm import riemannian_lm
    from emu_gmm.measures import SyntheticMeasure
    from emu_gmm.weighting import ContinuouslyUpdated

    mk = _psd_components()
    rng = np.random.default_rng(302)
    A_true = jnp.asarray(rng.normal(size=(_N, _K)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    M = _N * (_N + 1) // 2 + 1
    noise_key = jax.random.PRNGKey(302)

    def sampler(key, theta):
        del key, theta
        return target[None, :] + 0.01 * jax.random.normal(noise_key, (200, M))

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=200, sampler=sampler)
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(_N, _K)))
    result = estimate(
        _gauge_invariant_model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=400),
        theta_init=mk(Y0, 0.65),
    )
    assert bool(result.converged)
    return result


@pytest.mark.slow
class TestGaugeAwareCodomain:
    def test_asymptotic_eigenvalue_se_reuses_result(self, psd_result):
        law = AsymptoticLaw(psd_result)
        ev = law.eigenvalue_se()
        assert ev.shape == (_K,)
        assert np.all(np.isfinite(ev)) and np.all(ev > 0.0)
        np.testing.assert_allclose(ev, np.asarray(psd_result.eigenvalue_se()))

    def test_asymptotic_gamma_se_shape(self, psd_result):
        law = AsymptoticLaw(psd_result)
        q = _N * (_N + 1) // 2
        assert law.gamma_se().shape == (q,)
        assert np.all(np.isfinite(law.gamma_se()))

    def test_eigenvalue_functional_equals_convenience(self, psd_result):
        law = AsymptoticLaw(psd_result)
        np.testing.assert_allclose(
            law.se(eigenvalue_functional(_K)), law.eigenvalue_se()
        )

    def test_empirical_eigenvalue_per_draw_glue(self, psd_result):
        """The empirical grade applies the gauge-invariant eigenvalue
        functional PER DRAW (landmine 3): sampling the asymptotic Gaussian and
        reducing each draw to ``eigvalsh(A A^T)`` recovers the delta-method
        eigenvalue SE (cross-grade coherence)."""
        from emu_gmm.inference.functional_se import _component_shapes

        law = AsymptoticLaw(psd_result)
        shapes = tuple(_component_shapes(psd_result.components()))
        draws = law.sample(jax.random.PRNGKey(1), 8000)
        emp = EmpiricalLaw.from_draws(
            draws, names=law.param_names, component_shapes=shapes
        )
        ev_emp = emp.eigenvalue_se(_K)
        ev_asy = law.eigenvalue_se()
        np.testing.assert_allclose(ev_emp, ev_asy, rtol=0.1)

    def test_empirical_gamma_functional_runs(self, psd_result):
        from emu_gmm.inference.functional_se import _component_shapes

        law = AsymptoticLaw(psd_result)
        shapes = tuple(_component_shapes(psd_result.components()))
        draws = law.sample(jax.random.PRNGKey(2), 2000)
        emp = EmpiricalLaw.from_draws(
            draws, names=law.param_names, component_shapes=shapes
        )
        q = _N * (_N + 1) // 2
        assert emp.gamma_se().shape == (q,)
        # the explicit functional agrees with the named convenience.
        np.testing.assert_array_equal(emp.se(gamma_functional()), emp.gamma_se())

    def test_component_functional_without_shapes_raises(self, poison_sweep):
        law = EmpiricalLaw.from_records(poison_sweep)  # no component_shapes
        with pytest.raises(ValueError, match="component_shapes"):
            law.eigenvalue_se(1)


@pytest.mark.slow
class TestAcceptance2_6a:
    """§2.6a: ONE interface routes a K-Agg-style het result across grades.

    point + analytic SE + gauge-aware eigenvalue SE (asymptotic) | cluster-wild
    J (empirical law of Q) | MC sweep + given on a named event that fires
    partially (empirical theta-law). All three are EstimatorLaw instances and
    answer the same query algebra uniformly (charter §6.3: one interface used
    across grades, NOT one object).
    """

    def test_three_grades_one_interface(self, psd_result, wild_J_boot, poison_sweep):
        # Grade 1 -- asymptotic: the het point + analytic SE + eigenvalue SE.
        asy = AsymptoticLaw(psd_result)
        assert isinstance(asy, EstimatorLaw)
        assert asy.mean().shape[0] == len(asy.param_names)
        assert asy.cov().shape == (len(asy.param_names), len(asy.param_names))
        ev_se = asy.eigenvalue_se()
        assert ev_se.shape == (_K,) and np.all(ev_se > 0.0)

        # Grade 2 -- empirical law of Q: the cluster-wild J.
        J_boot, J_obs = wild_J_boot
        qlaw = EmpiricalLaw.from_draws(J_boot, names=("J",))
        assert isinstance(qlaw, EstimatorLaw)
        p = qlaw.pvalue(J_obs)
        assert 0.0 <= p <= 1.0
        assert qlaw.mean().shape == (1,)  # same algebra (mean/se/quantile)

        # Grade 3 -- empirical theta-law: the MC sweep + given on a named event.
        mlaw = EmpiricalLaw.from_records(poison_sweep)
        assert isinstance(mlaw, EstimatorLaw)
        # the preferred named event (the K-Agg het lambda_2 NaN-SE event) fires
        # PARTIALLY -> a non-vacuous conditional.
        share = mlaw.event_share("sigma_meat_indefinite")
        assert 0.0 < share.fraction_all < 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = mlaw.given("sigma_meat_indefinite")
        assert isinstance(cond, EstimatorLaw)
        assert cond.n_draws == share.n_selected
        # the conditional answers the SAME query algebra (coverage summary).
        cov_cond = cond.coverage(_theta0())
        assert cov_cond.n_used <= cond.n_draws

        # The three grades are the ONE interface: every law answers mean/cov/se.
        for law in (asy, qlaw, mlaw):
            assert isinstance(law.mean(), np.ndarray)
            assert isinstance(law.cov(), np.ndarray)
            assert isinstance(law.se(), np.ndarray)


# ---------------------------------------------------------------------------
# §2.6b --- the #130-harness summarizers + data exercise the carrier.
# Reproduces the ladder_mc readouts (jadj_readout via given+event_share;
# paired_dof_readout via crn_pair) through the law, byte-identical.
# ---------------------------------------------------------------------------
class TestHarness130ExercisesCarrier:
    def test_jadj_readout_through_carrier(self, poison_sweep):
        """jadj_readout (#130 fix 1iii): the binding subpopulation via given()
        + event_share() yields the same nominal/adjusted J p-values as the
        inline harness logic."""
        rec = poison_sweep.records
        # OLD inline harness logic.
        used = np.asarray(rec.converged) > 0
        binding_old = (np.asarray(rec.binding_ridge) > 0) & used
        pnb_old = np.asarray(rec.J_pvalue)[binding_old]
        pab_old = np.asarray(rec.J_pvalue_adjusted)[binding_old]
        # NEW carrier path.
        law = EmpiricalLaw.from_records(poison_sweep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = law.given("binding_ridge")
        share = law.event_share("binding_ridge")
        bconv = np.asarray(cond._records.converged) > 0.5
        pnb_new = np.asarray(cond._records.J_pvalue)[bconv]
        pab_new = np.asarray(cond._records.J_pvalue_adjusted)[bconv]
        assert share.n_selected_converged == int(binding_old.sum())
        np.testing.assert_array_equal(pnb_new, pnb_old)
        np.testing.assert_array_equal(pab_new, pab_old)

    def test_paired_dof_readout_through_carrier(self):
        """paired_dof_readout (#130 fix 2): the CRN-paired contrast via the
        carrier's couple() matches the inline crn_pair logic."""
        arms = replicate_coupled(
            {"a": _make_clean_run(), "b": _make_clean_run()},
            _clean_dgp(),
            n_reps=8,
            key=jax.random.PRNGKey(11),
            theta_init=_theta0(),
            coupling_id="ladder",
        )
        la = EmpiricalLaw.from_records(arms["a"])
        lb = EmpiricalLaw.from_records(arms["b"])
        cp_ref = crn_pair(arms["a"], arms["b"])
        cp_law = la.couple(lb)
        Ja = np.asarray(arms["a"].records.J_stat)
        Jb = np.asarray(arms["b"].records.J_stat)
        both = cp_ref.both_finite(Ja, Jb)
        assert cp_law.mean_paired_diff(Ja, Jb, where=both) == cp_ref.mean_paired_diff(
            Ja, Jb, where=both
        )

    def test_summary_battery_over_carrier(self, poison_sweep):
        """The full #130 summary battery runs over the carrier with results
        identical to the free summarizers (the carrier is a faithful router)."""
        law = EmpiricalLaw.from_records(poison_sweep)
        assert law.bias_sd(_theta0()).n_used == bias_sd(poison_sweep, _theta0()).n_used
        assert (
            law.tau_binding().binding_frequency
            == tau_binding(poison_sweep).binding_frequency
        )
        assert (
            law.j_calibration().max_abs_deviation
            == j_calibration(poison_sweep).max_abs_deviation
        )
