r"""Tests for typed, versioned law persistence (#181).

Covers the first slice: persisting an :class:`AsymptoticLaw` to an inert,
versioned ``.npz`` artifact and reloading it into a queryable, moments-backed
law --- with NO live :class:`EstimationResult` on the reload path.

(a) Round-trip fidelity on a ``Product(PSDFixedRank(5,2), Euclidean(1))`` fit:
    ``se`` / ``eigenvalue_se`` / ``gamma_se`` / a functional query all match the
    live law bit-for-bit, and the reloaded law has no live ``.result``.
(b) Round-trip on a v1 / all-scalar (Euclidean) fit.
(c) ``from_moments`` reconstructs the asymptotic grade directly from arrays.
(d) The manifold tag codec round-trips (Euclidean / PSDFixedRank / Positive /
    Interval).
(e) Guardrails: schema-version mismatch refuses; a bare ``.npz`` (no manifest)
    refuses; saving an ``EmpiricalLaw`` refuses (next slice); ``given`` / ``prob``
    still refuse on a reloaded law; a no-PSD reloaded law refuses
    ``eigenvalue_se``.
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

    def test_empirical_law_save_refused(self, tmp_path):
        law = EmpiricalLaw.from_draws(np.random.default_rng(0).normal(size=(50, 2)))
        with pytest.raises(NotImplementedError, match="empirical grade|EmpiricalLaw"):
            save_law(law, tmp_path / "x.npz")

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
