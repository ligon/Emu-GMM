r"""Tests for typed, versioned law persistence (#181).

Persisting a fitted law to an inert, versioned ``.npz`` artifact and reloading
it into a queryable law --- with NO live :class:`EstimationResult` on the
reload path.

Asymptotic grade:
(a) Round-trip fidelity on a ``Product(PSDFixedRank(5,2), Euclidean(1))`` fit:
    ``se`` / ``eigenvalue_se`` / ``gamma_se`` / a functional query all match the
    live law bit-for-bit, and the reloaded law has no live ``.result``.
(b) Round-trip on a v1 / all-scalar (Euclidean) fit; a file-like (BytesIO) target.
(c) ``from_moments`` reconstructs the asymptotic grade directly from arrays.
(d) The manifold tag codec round-trips (Euclidean / PSDFixedRank / Positive /
    Interval).

Empirical grade:
(e) Records-backed round-trip: ``se`` / ``size_power`` (J) / ``given`` / ``couple``
    all work on reload through the existing ``emu_gmm.studies`` reuse.
(f) Draws-backed round-trip with ``events``: ``se`` / ``pvalue`` / ``given``.

Guardrails:
(g) Schema-version mismatch / bare ``.npz`` / conditioned-law save /
    non-JSON-able coupling_id / reloaded-asymptotic ``given``-``prob`` /
    no-PSD ``eigenvalue_se`` all refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import (
    AsymptoticLaw,
    ContinuouslyUpdated,
    EmpiricalLaw,
    EmpiricalMeasure,
    Euclidean,
    IIDCovariance,
    Interval,
    LawState,
    Positive,
    PSDFixedRank,
    estimate,
    load_law,
    save_law,
)
from emu_gmm._internal.law_state import (
    SCHEMA_VERSION,
    manifold_to_tag,
    tag_to_manifold,
)
from emu_gmm.optimizer import optimistix_lm


def _load_phase4_fixture():
    manifolds_dir = Path(__file__).resolve().parent / "manifolds"
    if str(manifolds_dir) not in sys.path:
        sys.path.insert(0, str(manifolds_dir))
    import test_estimator_inference_phase4 as ph4

    return ph4


# ---------------------------------------------------------------------------
# (a) Manifold round-trip fidelity.
# ---------------------------------------------------------------------------


class TestManifoldRoundTrip:
    def _law(self):
        ph4 = _load_phase4_fixture()
        result, _spec, _M, _ = ph4._run_estimate(2, seed=300)
        return AsymptoticLaw(result)

    def test_queries_match_live_after_reload(self, tmp_path):
        law = self._law()
        live_se = np.asarray(law.se())
        live_eig = np.asarray(law.eigenvalue_se())
        live_gamma = np.asarray(law.gamma_se())

        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)

        assert isinstance(reloaded, AsymptoticLaw)
        assert reloaded.grade == "asymptotic"
        assert reloaded.param_names == law.param_names
        np.testing.assert_array_equal(np.asarray(reloaded.se()), live_se)
        np.testing.assert_array_equal(np.asarray(reloaded.eigenvalue_se()), live_eig)
        np.testing.assert_array_equal(np.asarray(reloaded.gamma_se()), live_gamma)

    def test_functional_query_matches(self, tmp_path):
        from emu_gmm.law import eigenvalue_functional, gamma_functional

        law = self._law()
        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)
        for f in (gamma_functional(), eigenvalue_functional(2)):
            np.testing.assert_array_equal(reloaded.cov(f), law.cov(f))
            np.testing.assert_array_equal(reloaded.se(f), law.se(f))

    def test_reloaded_has_no_live_result(self, tmp_path):
        law = self._law()
        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)
        with pytest.raises(AttributeError, match="reconstructed from moments"):
            _ = reloaded.result

    def test_artifact_is_inert_and_versioned(self, tmp_path):
        law = self._law()
        p = tmp_path / "law.npz"
        law.save(p)  # the EstimatorLaw.save() method
        # allow_pickle=False load proves no Python objects are entombed.
        with np.load(p, allow_pickle=False) as data:
            manifest = json.loads(str(data["__manifest__"]))
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["grade"] == "asymptotic"
        # The PSD factor persists as a typed tag, not a live ManifoldLeaf.
        assert {"type": "PSDFixedRank", "n": 5, "k": 2} in manifest["leaf_tags"]
        assert manifest["psd_rank"] == 2

    def test_save_load_accepts_a_file_like_buffer(self):
        # An object-store / fsspec target is an open binary buffer, not a path.
        import io

        law = self._law()
        buf = io.BytesIO()
        save_law(law, buf)  # write to the buffer directly (no temp file)
        buf.seek(0)
        reloaded = load_law(buf)  # read back from the buffer
        np.testing.assert_array_equal(
            np.asarray(reloaded.eigenvalue_se()), np.asarray(law.eigenvalue_se())
        )

    def test_resave_of_reloaded_law_is_idempotent(self, tmp_path):
        law = self._law()
        p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
        save_law(law, p1)
        reloaded = load_law(p1)
        save_law(reloaded, p2)  # re-save a moments-backed law
        again = load_law(p2)
        np.testing.assert_array_equal(
            np.asarray(again.eigenvalue_se()), np.asarray(law.eigenvalue_se())
        )


# ---------------------------------------------------------------------------
# v1 / all-scalar (Euclidean) fixture.
# ---------------------------------------------------------------------------


@jdc.pytree_dataclass
class _TwoScalar:
    a: float
    b: float


def _scalar_model(x, theta):
    y = x[0]
    u = x[1]
    v = x[2]
    z = x[3:6]
    return z * (y - theta.a * u - theta.b * v)


def _fit_scalar(seed: int = 0, n: int = 2000):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    u = Z @ np.array([1.2, 1.0, 0.9]) + rng.normal(size=n) * 0.4
    v = Z @ np.array([0.9, 1.1, 1.0]) + rng.normal(size=n) * 0.4
    y = 1.5 * u - 0.7 * v + rng.normal(size=n) * 0.3
    X = np.column_stack([y, u, v, Z])
    measure = EmpiricalMeasure.from_arrays(jnp.asarray(X), M=3)
    return estimate(
        _scalar_model,
        measure,
        covariance=IIDCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=optimistix_lm(),
        theta_init=_TwoScalar(a=1.5, b=-0.7),
    )


class TestScalarRoundTrip:
    def test_all_euclidean_round_trip(self, tmp_path):
        law = AsymptoticLaw(_fit_scalar())
        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)
        assert reloaded.param_names == law.param_names
        np.testing.assert_array_equal(np.asarray(reloaded.se()), np.asarray(law.se()))
        np.testing.assert_array_equal(
            np.asarray(reloaded.mean()), np.asarray(law.mean())
        )

    def test_no_psd_law_refuses_eigenvalue_se(self, tmp_path):
        law = AsymptoticLaw(_fit_scalar())
        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)
        with pytest.raises(TypeError, match="no\n?.*PSDFixedRank|PSDFixedRank"):
            reloaded.eigenvalue_se()


# ---------------------------------------------------------------------------
# (c) from_moments + (d) tag codec + (e) guardrails.
# ---------------------------------------------------------------------------


class TestFromMomentsAndCodec:
    def test_from_moments_matches_live(self):
        ph4 = _load_phase4_fixture()
        result, _spec, _M, _ = ph4._run_estimate(2, seed=301)
        live = AsymptoticLaw(result)
        comps = result.components()
        sigma = np.asarray(result.Sigma_theta.array)
        leaf_specs = tuple(ls.manifold for ls in result.manifold_spec.leaf_specs)
        moments = AsymptoticLaw.from_moments(
            comps, sigma, leaf_specs=leaf_specs, names=live.param_names
        )
        np.testing.assert_array_equal(
            np.asarray(moments.eigenvalue_se()), np.asarray(live.eigenvalue_se())
        )
        np.testing.assert_array_equal(
            np.asarray(moments.gamma_se()), np.asarray(live.gamma_se())
        )

    def test_from_moments_shape_guard(self):
        with pytest.raises(ValueError, match="ambient dimension"):
            AsymptoticLaw.from_moments(
                (np.zeros((5, 2)), np.zeros(())), np.eye(3)  # D=11, sigma 3x3
            )

    @pytest.mark.parametrize(
        "manifold",
        [
            Euclidean(),
            Euclidean(3),
            PSDFixedRank(5, 2),
            Positive(),
            Interval(0.0, 1.0),
        ],
    )
    def test_manifold_tag_round_trips(self, manifold):
        rebuilt = tag_to_manifold(manifold_to_tag(manifold))
        assert type(rebuilt) is type(manifold)
        assert tuple(rebuilt.ambient_shape) == tuple(manifold.ambient_shape)
        assert int(rebuilt.gauge_dim) == int(manifold.gauge_dim)


# ---------------------------------------------------------------------------
# Empirical-grade round-trip (records-backed + draws-backed).
# ---------------------------------------------------------------------------


def _fake_mcrecords(seed: int = 0, n: int = 80, d: int = 2):
    import jax
    from emu_gmm.studies.driver import MCRecords
    from emu_gmm.types import FitRecord

    rng = np.random.default_rng(seed)
    rec = FitRecord(
        theta_flat=jnp.asarray(rng.normal(size=(n, d))),
        se=jnp.asarray(np.abs(rng.normal(size=(n, d)))),
        J_stat=jnp.asarray(rng.chisquare(3, size=n)),
        J_pvalue=jnp.asarray(rng.uniform(size=n)),
        J_pvalue_adjusted=jnp.asarray(rng.uniform(size=n)),
        converged=jnp.asarray((rng.uniform(size=n) < 0.95).astype(float)),
        tau_realised=jnp.asarray(np.abs(rng.normal(size=n)) * 0.01),
        binding_ridge=jnp.asarray((rng.uniform(size=n) < 0.25).astype(float)),
        sigma_meat_indefinite=jnp.asarray((rng.uniform(size=n) < 0.1).astype(float)),
        J_dof=1,
        param_names=("a", "b"),
    )
    return MCRecords(
        records=rec, key=jax.random.PRNGKey(seed), n_reps=n, coupling_id=seed
    )


class TestEmpiricalRoundTrip:
    def test_records_backed_round_trip(self, tmp_path):
        from emu_gmm import EmpiricalLaw

        law = EmpiricalLaw.from_records(_fake_mcrecords())
        p = tmp_path / "emp.npz"
        save_law(law, p)
        r = load_law(p)

        assert r.grade == "empirical"
        assert r.n_draws == law.n_draws and r.n_used == law.n_used
        np.testing.assert_array_equal(np.asarray(r.se()), np.asarray(law.se()))
        # J-test rejection rates round-trip (size_power reuses the records).
        np.testing.assert_array_equal(
            r.size_power().reject_nominal, law.size_power().reject_nominal
        )

    def test_records_given_works_on_reload(self, tmp_path):
        import warnings

        from emu_gmm import EmpiricalLaw

        law = EmpiricalLaw.from_records(_fake_mcrecords(seed=2))
        p = tmp_path / "emp.npz"
        save_law(law, p)
        r = load_law(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            live = law.given("binding_ridge", acknowledge_conditional=True)
            reloaded = r.given("binding_ridge", acknowledge_conditional=True)
        assert reloaded.n_used == live.n_used
        np.testing.assert_array_equal(
            np.asarray(reloaded.mean()), np.asarray(live.mean())
        )

    def test_records_couple_works_on_reload(self, tmp_path):
        # Two CRN-coupled arms (same key + coupling_id) round-trip and still couple.
        from emu_gmm import EmpiricalLaw

        a = EmpiricalLaw.from_records(_fake_mcrecords(seed=5))
        b = EmpiricalLaw.from_records(_fake_mcrecords(seed=5))  # same key/coupling
        save_law(a, tmp_path / "a.npz")
        save_law(b, tmp_path / "b.npz")
        ra = load_law(tmp_path / "a.npz")
        rb = load_law(tmp_path / "b.npz")
        coupled = ra.couple(rb)  # raises if the CRN provenance was lost
        assert coupled is not None

    def test_draws_backed_round_trip_with_events(self, tmp_path):
        import warnings

        from emu_gmm import EmpiricalLaw

        rng = np.random.default_rng(3)
        draws = rng.normal(size=(120, 2))
        events = {"binding_ridge": (rng.uniform(size=120) < 0.3).astype(float)}
        law = EmpiricalLaw.from_draws(draws, names=("x", "y"), events=events)
        p = tmp_path / "draws.npz"
        save_law(law, p)
        r = load_law(p)

        assert r.grade == "empirical"
        assert r.param_names == ("x", "y")
        assert r.event_names == ("binding_ridge",)
        np.testing.assert_array_equal(np.asarray(r.se()), np.asarray(law.se()))
        assert np.isclose(r.pvalue(0.0), law.pvalue(0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert r.given("binding_ridge").n_used == law.given("binding_ridge").n_used

    def test_non_jsonable_coupling_id_refused(self, tmp_path):
        from emu_gmm import EmpiricalLaw
        from emu_gmm.studies.driver import MCRecords

        mc = _fake_mcrecords(seed=8)
        # A non-JSON-able coupling token (a numpy array) can't ride the manifest.
        mc_bad = MCRecords(
            records=mc.records, key=mc.key, n_reps=mc.n_reps, coupling_id=np.arange(3)
        )
        law = EmpiricalLaw.from_records(mc_bad)
        with pytest.raises(TypeError, match="coupling_id"):
            save_law(law, tmp_path / "x.npz")


class TestGuardrails:
    def test_schema_version_mismatch_refuses(self, tmp_path):
        law = AsymptoticLaw(_fit_scalar())
        p = tmp_path / "law.npz"
        save_law(law, p)
        # Rewrite the manifest with a bumped schema_version.
        with np.load(p, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        manifest = json.loads(str(arrays["__manifest__"]))
        manifest["schema_version"] = SCHEMA_VERSION + 99
        arrays["__manifest__"] = np.asarray(json.dumps(manifest))
        np.savez(p, **arrays)
        with pytest.raises(ValueError, match="schema_version"):
            load_law(p)

    def test_bare_npz_refuses(self, tmp_path):
        p = tmp_path / "bare.npz"
        np.savez(p, foo=np.arange(3))
        with pytest.raises(ValueError, match="no __manifest__"):
            load_law(p)

    def test_conditioned_empirical_law_save_refused(self, tmp_path):
        import warnings

        draws = np.random.default_rng(0).normal(size=(60, 2))
        ev = {"binding_ridge": (np.random.default_rng(1).uniform(size=60) < 0.4)}
        law = EmpiricalLaw.from_draws(draws, names=("a", "b"), events=ev)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conditioned = law.given("binding_ridge")
        with pytest.raises(NotImplementedError, match="conditioned"):
            save_law(conditioned, tmp_path / "x.npz")

    def test_reloaded_given_and_prob_refuse(self, tmp_path):
        law = AsymptoticLaw(_fit_scalar())
        p = tmp_path / "law.npz"
        save_law(law, p)
        reloaded = load_law(p)
        with pytest.raises(NotImplementedError, match="given refuses"):
            reloaded.given("binding_ridge")
        with pytest.raises(NotImplementedError, match="prob refuses"):
            reloaded.prob(lambda v: v > 0)

    def test_law_state_is_frozen_typed_record(self):
        # The schema is a typed dataclass, not a dict (validation/migration).
        assert hasattr(LawState, "__dataclass_fields__")
        assert "schema_version" in LawState.__dataclass_fields__
