"""Optional cross-package boundary oracle; no consumer data or runtime deps."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "scripts/validation/clustered_parity_18.py"
_SPEC = importlib.util.spec_from_file_location("clustered_parity_18", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


@pytest.fixture
def oracle():
    pytest.importorskip("dgp_protocol")
    pytest.importorskip("manifoldgmm")


@pytest.mark.parametrize("centered", [False, True])
def test_actual_dgp_and_manifoldgmm_paths_match_clustered_boundary(oracle, centered):
    rng = np.random.default_rng(18)
    # Unbalanced, interleaved clusters and nonzero means expose centering and
    # layout errors; raw labels differ from dense segment codes.
    ids = np.tile([20, 7, 20, 91, 7, 40, 20, 91, 40], 10)
    moments = rng.normal(size=(len(ids), 3)) + [0.5, -1.0, 2.0]
    report = driver.compare_clustered(moments, moments.copy(), ids, centered=centered)
    assert report["n_clusters"] == 4
    assert report["N"] == len(ids)
    assert report["covariance_max_relative_error"] < 1e-12
    assert report["criterion_relative_error"] < 1e-12
    assert report["refitted"] is False


def test_ported_moment_mismatch_is_not_hidden_by_covariance_comparison(oracle):
    moments = np.arange(36.0).reshape(12, 3)
    changed = moments.copy()
    changed[4, 2] += 0.01
    with pytest.raises(AssertionError):
        driver.compare_clustered(changed, moments, np.arange(12))


def test_missingness_and_compound_ids_are_not_silently_reinterpreted(oracle):
    moments = np.eye(3)
    missing = moments.copy()
    missing[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete finite"):
        driver.compare_clustered(missing, missing, np.arange(3))
    with pytest.raises(ValueError, match="one integer code"):
        driver.compare_clustered(moments, moments, np.ones((3, 2), dtype=int))
