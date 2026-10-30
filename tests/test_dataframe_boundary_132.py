"""Label fidelity and optional eager dataframe boundaries (issue #132)."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas as pd
import pytest
from emu_gmm import EmpiricalMeasure, Fixed, IIDCovariance, build_estimator, estimate
from emu_gmm._internal import frames


@jdc.pytree_dataclass
class Theta:
    mean: float


def model(row, theta):
    return jnp.array([row[0] - theta.mean, row[1] - 2 * theta.mean])


def data():
    return pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "quantity": [2.1, 3.7, 6.2, 8.4, 9.8, 12.5],
            "instrument": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        },
        index=pd.Index(["h3", "h1", "h2", "h6", "h4", "h5"], name="household"),
    )


def fit(frame):
    measure = EmpiricalMeasure.from_frame(frame, mask=np.ones((len(frame), 2)))
    return estimate(
        model,
        measure,
        parameters=Theta(0.0),
        covariance=IIDCovariance(),
        weighting=Fixed(L0=jnp.eye(2)),
        linear=True,
        moment_names=("level", "slope"),
    )


def convert(frame, backend):
    if backend == "pandas":
        return frame
    if backend == "datamat":
        dm = pytest.importorskip("datamat")
        return dm.DataMat(frame)
    pl = pytest.importorskip("polars")
    return pl.DataFrame({c: frame[c].to_numpy() for c in frame.columns})


@pytest.mark.parametrize("backend", ["pandas", "polars", "datamat"])
def test_backend_labels_survive_estimation_without_becoming_moment_names(backend):
    frame = convert(data(), backend)
    result = fit(frame)
    assert result.labels.variable_names == tuple(frame.columns)
    assert result.labels.moment_names == ("level", "slope")
    assert result.labels.param_names == ("mean",)
    assert result.moment_covariance.shape == (2, 2)  # D=3, M=2, K=1.
    assert result.asymptotic().cov().shape == (1, 1)
    assert result.labels.obs_name == (None if backend == "polars" else "household")
    reference = fit(data())
    np.testing.assert_array_equal(result.moment_covariance, reference.moment_covariance)
    assert float(result.theta_hat.mean) == float(reference.theta_hat.mean)
    assert float(result.objective_value) == float(reference.objective_value)


def test_pandas_constructor_keeps_nonstring_and_multiindex_labels_verbatim():
    frame = data()
    frame.columns = pd.MultiIndex.from_tuples(
        [("goods", 1), ("goods", 2), ("iv", 3)], names=["kind", "asset"]
    )
    frame.index = pd.MultiIndex.from_tuples(
        [(i, "wave1") for i in frame.index], names=["household", "wave"]
    )
    measure = EmpiricalMeasure.from_pandas(frame, mask=np.ones((len(frame), 2)))
    assert measure.labels.variable_names == tuple(frame.columns)
    assert measure.labels.variable_axis_names == ("kind", "asset")
    assert measure.labels.obs_name == ("household", "wave")
    assert hash(measure.labels)
    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(measure))
    np.testing.assert_allclose(
        jax.jit(lambda mu: mu.expectation(model, Theta(1.0)))(measure),
        measure.expectation(model, Theta(1.0)),
    )


def test_haliax_axis_names_survive_without_inventing_coordinate_labels():
    named = ha.named(
        jnp.ones((3, 2)), (ha.Axis("households", 3), ha.Axis("features", 2))
    )
    measure = EmpiricalMeasure.from_frame(named)
    assert measure.labels.obs_name == "households"
    assert measure.labels.variable_axis_names == ("features",)
    assert measure.labels.variable_names == ("v_0", "v_1")


def test_indexed_weights_and_masks_align_rows_without_reordering_moment_columns():
    frame = data()
    weights = pd.Series(np.arange(1.0, 7.0), index=frame.index).iloc[::-1]
    mask = pd.DataFrame(
        [[1, 1], [0, 1], [1, 0], [1, 1], [1, 0], [1, 1]],
        index=frame.index,
        columns=["slope", "level"],
    )
    measure = EmpiricalMeasure.from_frame(frame, weights, mask.iloc[::-1])
    np.testing.assert_array_equal(measure.weights, weights.loc[frame.index])
    np.testing.assert_array_equal(measure.mask, mask.to_numpy())


def test_legacy_from_pandas_keeps_positional_auxiliary_semantics():
    frame = data()
    weights = pd.Series(np.arange(1.0, 7.0))
    measure = EmpiricalMeasure.from_pandas(frame, weights, np.ones((6, 2)))
    np.testing.assert_array_equal(measure.weights, weights)


def test_datamat_single_level_index_aligns_with_flat_pandas_keys():
    dm = pytest.importorskip("datamat")
    frame = dm.DataMat(data())
    weights = pd.Series(np.arange(1.0, 7.0), index=data().index).iloc[::-1]
    measure = EmpiricalMeasure.from_frame(frame, weights, np.ones((6, 2)))
    np.testing.assert_array_equal(measure.weights, np.arange(1.0, 7.0))


def test_datamat_multiindex_ingress_retains_axes_and_aligns_auxiliaries():
    dm = pytest.importorskip("datamat")
    frame = data()
    frame.index = pd.MultiIndex.from_tuples(
        [(key, 1) for key in frame.index], names=["household", "wave"]
    )
    frame.columns = pd.MultiIndex.from_tuples(
        [("goods", "price"), ("goods", "quantity"), ("iv", "one")],
        names=["block", "variable"],
    )
    weights = dm.DataVec(pd.Series(np.arange(1.0, 7.0), index=frame.index)).iloc[::-1]
    mask = dm.DataMat(pd.DataFrame(np.ones((6, 2)), index=frame.index)).iloc[::-1]
    measure = EmpiricalMeasure.from_frame(dm.DataMat(frame), weights, mask)
    assert measure.labels.variable_names == tuple(frame.columns)
    assert measure.labels.variable_axis_names == ("block", "variable")
    assert measure.labels.obs_name == ("household", "wave")
    np.testing.assert_array_equal(measure.weights, np.arange(1.0, 7.0))
    np.testing.assert_array_equal(measure.x, frame.to_numpy())


def test_bootstrap_world_keeps_only_static_label_metadata():
    from emu_gmm.inference.cluster_bootstrap import _resample_one

    measure = EmpiricalMeasure.from_frame(data(), mask=np.ones((6, 2)))
    rows = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    world, _ = _resample_one(measure, rows, np.array([2, 0, 2]))
    assert world.labels is measure.labels
    np.testing.assert_array_equal(world.x, np.asarray(measure.x)[[4, 5, 0, 1, 4, 5], :])
    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(world))


@pytest.mark.parametrize("auxiliary", ["weights", "mask"])
def test_mismatched_row_keys_are_rejected(auxiliary):
    frame = data()
    bad = pd.Series(np.ones(6), index=["unknown"] + list(frame.index[1:]))
    kwargs = {auxiliary: bad if auxiliary == "weights" else bad.to_frame()}
    with pytest.raises(ValueError, match="observation labels do not match"):
        EmpiricalMeasure.from_frame(frame, **kwargs)


def test_duplicate_keys_are_allowed_in_order_but_not_ambiguously_realigned():
    frame = data()
    frame.index = ["a", "a", "b", "c", "d", "e"]
    weights = pd.Series(np.arange(1.0, 7.0), index=frame.index)
    measure = EmpiricalMeasure.from_frame(frame, weights, np.ones((6, 2)))
    np.testing.assert_array_equal(measure.weights, weights)
    with pytest.raises(ValueError, match="duplicate observation labels"):
        EmpiricalMeasure.from_frame(frame, weights.iloc[::-1], np.ones((6, 2)))


@pytest.mark.parametrize("backend", ["polars", "datamat"])
def test_vectors_are_supported_for_weights_and_single_moment_masks(backend):
    frame = data().iloc[:, :1]
    if backend == "polars":
        pl = pytest.importorskip("polars")
        x = pl.Series("price", frame.iloc[:, 0].to_numpy())
        weights = pl.Series("weights", np.arange(1.0, 7.0))
        mask = pl.Series("observed", [1, 0, 1, 1, 1, 1])
    else:
        dm = pytest.importorskip("datamat")
        x = dm.DataVec(frame.iloc[:, 0])
        weights = dm.DataVec(pd.Series(np.arange(1.0, 7.0), index=frame.index))
        mask = dm.DataVec(pd.Series([1, 0, 1, 1, 1, 1], index=frame.index))
    measure = EmpiricalMeasure.from_frame(x, weights, mask)
    assert measure.x.shape == measure.mask.shape == (6, 1)
    np.testing.assert_array_equal(measure.weights, np.arange(1.0, 7.0))
    np.testing.assert_array_equal(measure.mask[:, 0], [1, 0, 1, 1, 1, 1])


def test_nullable_numeric_pandas_data_uses_existing_missingness_policy():
    frame = pd.DataFrame({"a": pd.Series([1, None, 3], dtype="Int64")})
    measure = EmpiricalMeasure.from_frame(frame)
    np.testing.assert_array_equal(measure.x[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(measure.mask[:, 0], [1.0, 0.0, 1.0])


def test_plain_array_legacy_constructor_stays_unlabelled_and_strings_are_not_parsed():
    measure = EmpiricalMeasure.from_pandas(np.ones((3, 2)))
    assert measure.labels is None
    with pytest.raises(ValueError, match="numeric columns"):
        EmpiricalMeasure.from_frame(pd.DataFrame({"id": ["1", "2"]}))


def test_unknown_observability_in_a_nullable_frame_is_rejected():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    mask = pd.DataFrame({"m": pd.Series([True, None, True], dtype="boolean")})
    with pytest.raises(ValueError, match="observability values must be finite"):
        EmpiricalMeasure.from_frame(frame, mask=mask)


def test_polars_nullable_boolean_data_and_unknown_mask_are_distinct():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"flag": [True, None, False]})
    measure = EmpiricalMeasure.from_frame(frame)
    np.testing.assert_array_equal(measure.mask[:, 0], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(measure.x[:, 0], [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="observability values must be finite"):
        EmpiricalMeasure.from_frame(pl.DataFrame({"flag": [1.0, 2.0, 3.0]}), mask=frame)


@pytest.mark.parametrize("labelled", [True, False])
def test_factory_returns_labels_of_current_measure(labelled):
    original = data()
    template = EmpiricalMeasure.from_frame(original, mask=np.ones((6, 2)))
    run = build_estimator(
        model,
        measure=template,
        parameters=Theta(0.0),
        covariance=IIDCovariance(),
        weighting=Fixed(L0=jnp.eye(2)),
        linear=True,
        moment_names=("level", "slope"),
    )
    changed = original.rename(columns={"price": "new_price"})
    current = EmpiricalMeasure.from_frame(
        changed if labelled else changed.to_numpy(), mask=np.ones((6, 2))
    )
    result = run(Theta(0.0), current)
    assert result.labels.variable_names == (
        tuple(changed.columns) if labelled else ("v_0", "v_1", "v_2")
    )
    assert result.labels.obs_name == ("household" if labelled else None)
    assert template.labels.variable_names == tuple(original.columns)


@pytest.mark.parametrize("backend", ["pandas", "polars", "datamat"])
def test_optimization_and_inference_egress_use_existing_readouts(backend):
    if backend != "pandas":
        pytest.importorskip(backend)
    result = fit(data())
    law = result.asymptotic()
    tables = result.to_frame(backend=backend)
    coef = law.coef_frame(backend=backend)
    cov = law.cov_frame(backend=backend)
    if backend == "pandas":
        for name, expected in result.to_pandas().items():
            pd.testing.assert_series_equal(tables[name], expected)
        pd.testing.assert_frame_equal(coef, law.coef_table)
        np.testing.assert_array_equal(cov.to_numpy(), law.cov())
    elif backend == "datamat":
        import datamat as dm

        assert isinstance(coef, dm.DataMat) and isinstance(cov, dm.DataMat)
        assert cov.shape == (1, 1)  # Never squeeze a one-parameter matrix.
        assert list(coef.index.get_level_values(0)) == list(law.param_names)
        np.testing.assert_array_equal(coef.to_numpy(), law.coef_table.to_numpy())
        np.testing.assert_array_equal(cov.to_numpy(), law.cov())
        assert all(isinstance(tables[k], dm.DataVec) for k in result.to_pandas())
        assert isinstance(tables["moment_covariance"], dm.DataMat)
        np.testing.assert_array_equal(
            tables["moment_covariance"].to_numpy(), result.moment_covariance
        )
        assert list(tables["N_j"].index.get_level_values(0)) == ["level", "slope"]
    else:
        assert coef["index"].to_list() == list(law.param_names)
        np.testing.assert_array_equal(
            coef["estimate"].to_numpy(), law.coef_table["estimate"]
        )
        np.testing.assert_array_equal(cov["mean"].to_numpy(), law.cov()[:, 0])
        assert tables["N_j"]["index"].to_list() == ["level", "slope"]
        summary = dict(
            zip(
                tables["summary"]["index"].to_list(),
                tables["summary"]["value"].to_list(),
                strict=False,
            )
        )
        assert isinstance(summary["converged"], bool)

    if backend == "pandas":
        meat = tables["moment_covariance"]
        assert list(meat.index) == list(meat.columns) == ["level", "slope"]
        np.testing.assert_array_equal(meat.to_numpy(), result.moment_covariance)
    elif backend == "polars":
        assert tables["moment_covariance"]["moment"].to_list() == ["level", "slope"]
        np.testing.assert_array_equal(
            tables["moment_covariance"].select("level", "slope").to_numpy(),
            result.moment_covariance,
        )


def test_matrix_readouts_use_ambient_tangent_parameter_labels():
    from emu_gmm import Euclidean, ManifoldLeaf, riemannian_lm

    @jdc.pytree_dataclass
    class VectorTheta:
        coefficients: ManifoldLeaf

    def psi(row, theta):
        a, b = theta.coefficients.array
        return jnp.array([row[0] - a, row[1] - b, row[2] - a - b])

    result = estimate(
        psi,
        EmpiricalMeasure.from_frame(data()),
        parameters=VectorTheta(ManifoldLeaf(jnp.zeros(2), Euclidean(2))),
        covariance=IIDCovariance(),
        weighting=Fixed(L0=jnp.eye(3)),
        optimizer=riemannian_lm(),
        moment_names=("one", "two", "three"),
    )
    tables = result.to_frame()
    names = result.asymptotic().param_names
    assert tuple(tables["moment_jacobian"].columns) == names
    assert (
        tuple(tables["gn_hessian"].index)
        == tuple(tables["gn_hessian"].columns)
        == names
    )
    assert tables["gn_hessian"].shape == (2, 2)


def test_datamat_materialisation_preserves_multiindex_names_and_labels():
    pytest.importorskip("datamat")
    index = pd.MultiIndex.from_product([["a", "b"], [1]], names=["group", "wave"])
    columns = pd.MultiIndex.from_tuples(
        [("v", "x"), ("v", "y")], names=["kind", "asset"]
    )
    frame = pd.DataFrame(np.eye(2), index=index, columns=columns)
    output = frames.materialise(frame, "datamat")
    assert output.index.equals(index) and output.columns.equals(columns)
    assert list(output.index.names) == ["group", "wave"]
    assert list(output.columns.names) == ["kind", "asset"]


def test_polars_materialisation_preserves_multiindex_row_keys():
    pytest.importorskip("polars")
    index = pd.MultiIndex.from_tuples([("a", 2), ("b", 1)], names=["group", "wave"])
    output = frames.materialise(
        pd.Series([1.0, 2.0], index=index, name="value"), "polars"
    )
    assert output.to_dict(as_series=False) == {
        "group": ["a", "b"],
        "wave": [2, 1],
        "value": [1.0, 2.0],
    }


@pytest.mark.parametrize("backend", ["polars", "datamat"])
def test_missing_optional_backend_has_an_actionable_error(monkeypatch, backend):
    original = frames.importlib.import_module

    def missing(name):
        if name == backend:
            raise ImportError(name)
        return original(name)

    monkeypatch.setattr(frames.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=f"Install '{backend}'"):
        frames.materialise(pd.Series([1.0]), backend)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataframe backend"):
        frames.materialise(pd.Series([1.0]), "other")


def test_polars_refuses_unrepresentable_column_labels_and_collisions():
    pytest.importorskip("polars")
    with pytest.raises(ValueError, match="string data-column labels"):
        frames.materialise(pd.DataFrame([[1]], columns=[("a", "b")]), "polars")
    with pytest.raises(ValueError, match="collide"):
        frames.materialise(pd.Series([1.0], name="index"), "polars")
