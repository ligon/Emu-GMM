"""Deterministic, estimation-free regressions for ledger Section 21 / #191."""

from __future__ import annotations

import io
import json
import os
import stat

import numpy as np
import pytest
from emu_gmm import EmpiricalLaw, load_law, load_law_state, save_law


def _law():
    return EmpiricalLaw(
        draws=np.array([[1.0, 2.0], [100.0, 200.0], [3.0, 4.0], [np.nan, 6.0]]),
        used=np.array([True, False, True, False]),
        names=("x", "y"),
        component_shapes=((), ()),
        events={"chosen": np.array([True, True, False, False])},
    )


@pytest.mark.parametrize("buffer", [False, True])
def test_explicit_used_mask_and_queries_survive_round_trip(tmp_path, buffer):
    law = _law()
    target = io.BytesIO() if buffer else tmp_path / "law"
    save_law(law, target)
    if buffer:
        target.seek(0)
    loaded = load_law(target)
    np.testing.assert_array_equal(loaded._draws, law._draws)
    np.testing.assert_array_equal(loaded._used, law._used)
    assert loaded.n_used == 2
    assert loaded.n_draws == 4
    assert loaded.param_names == law.param_names
    assert loaded._component_shapes == law._component_shapes
    np.testing.assert_array_equal(loaded.mean(), law.mean())
    np.testing.assert_array_equal(loaded.se(), law.se())
    np.testing.assert_array_equal(loaded.cov(), law.cov())
    np.testing.assert_array_equal(
        loaded.mean(lambda c: c[0] + c[1]), law.mean(lambda c: c[0] + c[1])
    )
    assert loaded.event_share("chosen") == law.event_share("chosen")
    selected = loaded.given("chosen")
    assert selected.n_used == 1
    np.testing.assert_array_equal(selected.mean(), [1.0, 2.0])
    with pytest.raises(NotImplementedError, match="conditioned"):
        save_law(selected, io.BytesIO())
    # Re-saving the reconstructed law must not revive the excluded finite row.
    second = io.BytesIO()
    save_law(loaded, second)
    second.seek(0)
    np.testing.assert_array_equal(load_law(second)._used, law._used)


@pytest.mark.parametrize(
    "used",
    [
        np.array([True]),
        np.array([[True], [False], [True], [False]]),
        np.array([1, 0, 1, 0]),
        np.array([1.0, np.nan, 1.0, 0.0]),
        np.array([True, False, True, True]),
    ],
)
def test_load_refuses_malformed_or_nonfinite_used_mask(tmp_path, used):
    target = tmp_path / "law.npz"
    save_law(_law(), target)
    with np.load(target, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["used"] = used
    np.savez(target, **arrays)
    with pytest.raises(ValueError, match="used"):
        load_law(target)


@pytest.mark.parametrize("bad", ["length", "dtype", "nonfinite"])
def test_save_refuses_bad_mask_before_touching_existing_artifact(tmp_path, bad):
    target = tmp_path / "law.npz"
    save_law(_law(), target)
    original = target.read_bytes()
    invalid = _law()
    # The explicit constructor normalizes dtype/shape. Deliberate corruption
    # here exercises the persistence boundary's own checks, before serialization.
    invalid._used = {
        "length": np.array([True]),
        "dtype": np.array([1, 0, 1, 0]),
        "nonfinite": np.array([True, False, True, True]),
    }[bad]
    with pytest.raises(ValueError, match="used"):
        save_law(invalid, target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_all_finite_draws_may_be_excluded(tmp_path):
    law = EmpiricalLaw(
        draws=np.array([[1.0], [2.0]]),
        used=np.array([False, False]),
        names=("x",),
    )
    target = tmp_path / "empty-law.npz"
    save_law(law, target)
    loaded = load_law(target)
    assert loaded.n_used == 0
    assert loaded.n_draws == 2


def test_loaded_conditioned_marker_is_retained(tmp_path):
    target = tmp_path / "law.npz"
    save_law(_law(), target)
    with np.load(target, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    manifest = json.loads(str(arrays["__manifest__"]))
    manifest["conditioned"] = True
    arrays["__manifest__"] = np.asarray(json.dumps(manifest))
    np.savez(target, **arrays)
    loaded = load_law(target)
    with pytest.raises(NotImplementedError, match="conditioned"):
        save_law(loaded, io.BytesIO())


@pytest.mark.parametrize("failure", ["serialization", "sync", "replacement"])
@pytest.mark.parametrize("existing", [False, True])
def test_path_write_failure_preserves_old_artifact_and_cleans_temp(
    tmp_path, monkeypatch, failure, existing
):
    import emu_gmm.persistence as persistence

    target = tmp_path / "law.npz"
    original = None
    if existing:
        save_law(_law(), target)
        original = target.read_bytes()

    def fail(*args, **kwargs):
        if failure == "serialization":
            args[0].write(b"partial archive")
        raise OSError(f"injected {failure} failure")

    if failure == "serialization":
        monkeypatch.setattr(persistence.np, "savez", fail)
    elif failure == "sync":
        monkeypatch.setattr(persistence.os, "fsync", fail)
    else:
        monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError, match=f"injected {failure}"):
        save_law(EmpiricalLaw.from_draws([10.0, 20.0]), target)
    if existing:
        assert target.read_bytes() == original
        np.testing.assert_array_equal(load_law(target)._used, _law()._used)
        assert list(tmp_path.iterdir()) == [target]
    else:
        assert list(tmp_path.iterdir()) == []


def test_successful_replace_is_sibling_complete_and_preserves_mode(
    tmp_path, monkeypatch
):
    import emu_gmm.persistence as persistence

    target = tmp_path / "law-without-suffix"
    save_law(_law(), target)
    target.chmod(0o640)
    before = target.read_bytes()
    replace = os.replace
    seen = []

    def observe(source, destination):
        assert os.path.dirname(source) == str(tmp_path)
        assert destination == str(target)
        assert target.read_bytes() == before
        # The complete new archive is readable before the namespace changes.
        np.testing.assert_array_equal(load_law(source).mean(), [15.0])
        seen.append(source)
        replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", observe)
    save_law(EmpiricalLaw.from_draws([10.0, 20.0]), target)
    assert len(seen) == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    np.testing.assert_array_equal(load_law(target).mean(), [15.0])
    assert list(tmp_path.iterdir()) == [target]


def test_new_artifact_permissions_respect_umask(tmp_path):
    target = tmp_path / "law.npz"
    previous = os.umask(0o027)
    try:
        save_law(_law(), target)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_file_like_write_stays_direct_and_non_atomic(monkeypatch):
    import emu_gmm.persistence as persistence

    target = io.BytesIO()

    def no_replace(*args):
        pytest.fail("file-like targets must not use filesystem replacement")

    monkeypatch.setattr(persistence.os, "replace", no_replace)
    save_law(_law(), target)
    target.seek(0)
    np.testing.assert_array_equal(load_law_state(target).used, _law()._used)

    def partial_then_fail(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("file-like caller owns rollback")

    monkeypatch.setattr(persistence.np, "savez", partial_then_fail)
    partial = io.BytesIO()
    with pytest.raises(OSError, match="caller owns rollback"):
        save_law(_law(), partial)
    assert partial.getvalue() == b"partial archive"
