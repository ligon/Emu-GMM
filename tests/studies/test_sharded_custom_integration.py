"""Cross-feature integration: custom procedures, vector extras, and CRN shards."""

import jax
import jax.numpy as jnp
import numpy as np
from emu_gmm import EmpiricalLaw, load_law, save_law
from emu_gmm.studies import MCRecords, crn_pair, replicate_coupled
from emu_gmm.types import FitRecord


def test_sharded_custom_nested_procedures_round_trip(tmp_path):
    def dgp(key):
        x = jax.random.randint(key, (2,), -20, 20).astype(jnp.float64)
        return {"x": x, "event": jnp.remainder(x[0], 2)}

    def arm(width):
        def run(initial, dataset):
            return {
                "dataset": dataset,
                "theta": dataset["x"][:width] + initial["offset"],
                "objective": dataset["x"][0] + 100 + width,
                "width": width,
            }

        return run

    def record(result, dataset):
        assert result["dataset"] is dataset
        event, width = dataset["event"], result["width"]
        return FitRecord(
            theta_flat=result["theta"],
            se=jnp.full((width,), jnp.nan),
            J_stat=result["objective"],
            J_pvalue=jnp.asarray(0.2),
            J_pvalue_adjusted=jnp.asarray(0.3),
            converged=1.0 - event if width == 2 else jnp.asarray(1.0),
            tau_realised=event / 100,
            binding_ridge=event,
            sigma_meat_indefinite=jnp.asarray(1.0),
            tau_saturated=event,
            J_dof=3 - width,
            param_names=("beta", "gamma")[:width],
        )

    common = dict(
        runs={"narrow": arm(1), "wide": arm(2)},
        dgp=dgp,
        key=jax.random.key(19),
        theta_init={"offset": 2.0},
        coupling_id="custom-sharded-study",
        record=record,
        statistics={
            "vector": lambda result, data: jnp.stack(
                [data["x"][0], result["objective"], jnp.asarray(jnp.nan)]
            )
        },
    )
    expected_reps = range(5, 11)
    reference = replicate_coupled(**common, reps=expected_reps)
    shards = [
        replicate_coupled(**common, reps=range(8, 11)),
        replicate_coupled(**common, reps=range(5, 8)),
    ]
    joined = {}
    restored = {}
    for name, expected in reference.items():
        joined[name] = MCRecords.concat([s[name] for s in shards], reps=expected_reps)
        actual = joined[name]
        for got, want in zip(
            jax.tree_util.tree_leaves(actual.records),
            jax.tree_util.tree_leaves(expected.records),
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(actual.extra["vector"], expected.extra["vector"])
        frame = actual.to_pandas()
        np.testing.assert_array_equal(frame.index, list(expected_reps))
        assert frame.index.name == "rep"
        columns = ["vector[0]", "vector[1]", "vector[2]"]
        np.testing.assert_array_equal(frame[columns].to_numpy(), actual.extra["vector"])
        assert np.isnan(frame["vector[2]"]).all()
        assert np.isnan(actual.records.se).all()
        path = tmp_path / f"{name}.npz"
        save_law(EmpiricalLaw.from_records(actual), path)
        restored[name] = load_law(path)._records
        np.testing.assert_array_equal(
            restored[name].effective_rep_indices, list(expected_reps)
        )
        assert restored[name].coupling_id == common["coupling_id"]
        crn_pair(actual, restored[name])
        for got, want in zip(
            jax.tree_util.tree_leaves(restored[name].records),
            jax.tree_util.tree_leaves(actual.records),
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(
            restored[name].extra["vector"], actual.extra["vector"]
        )

    for arms in (joined, restored):
        paired = crn_pair(arms["narrow"], arms["wide"])
        assert paired.param_names_a == ("beta",)
        assert paired.param_names_b == ("beta", "gamma")
        assert paired.a.J_dof == 2 and paired.b.J_dof == 1
        np.testing.assert_array_equal(
            paired.paired_diff(
                paired.a.J_stat, paired.b.J_stat, where=np.ones(6, bool)
            ),
            np.ones(6),
        )
        flips = paired.flips(paired.a.converged, paired.b.converged)
        assert flips.gain == 0 and flips.lose == int(np.sum(paired.b.binding_ridge))
        assert flips.n_both == 6
