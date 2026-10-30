"""Optional ManifoldGMM/DGP clustered-boundary comparison (issue #18).

Compare complete, equally weighted moment arrays at identical parameters.
No fit is performed. Legacy Omega is on the sqrt(N)-moment scale; Emu V is
on the mean-moment scale. Their quadratic criteria agree after this boundary
conversion. Floating-point reduction order and legacy PSD projection preclude
a general bitwise-equality claim. No raw consumer data belong in this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from emu_gmm import ClusteredCovariance, EmpiricalMeasure
from emu_gmm._internal.cholesky import quadratic_form


def _identity_moments(theta: Any, observation: Any) -> Any:
    del theta
    return jnp.asarray(observation)


def _row_moment(row: Any, theta: Any) -> Any:
    del theta
    return row


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.max(np.abs(expected)))
    difference = float(np.max(np.abs(actual - expected)))
    return difference / scale if scale else difference


def compare_clustered(
    emu_moments: Any,
    legacy_moments: Any,
    cluster_ids: Any,
    *,
    centered: bool = True,
) -> dict[str, Any]:
    """Compare actual package paths without estimating or duplicating formulas.

    Moment arrays must have the same row and column order. IDs must be one
    integer code per row; encode compound (time, market) labels before entry.
    This validation deliberately excludes weights and unequal observability:
    those have different legacy contracts and do not admit a common-N port.
    """
    try:
        from dgp_protocol import ClusteredSampling, EmpiricalDGP
        from manifoldgmm import GMM
    except ImportError as exc:
        raise ImportError(
            "This optional validation requires ManifoldGMM and DGP_Protocol "
            "in a development environment"
        ) from exc

    emu = np.asarray(emu_moments, dtype=np.float64)
    legacy = np.asarray(legacy_moments, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    if emu.ndim != 2 or min(emu.shape) == 0 or emu.shape != legacy.shape:
        raise ValueError("moment arrays must have the same nonempty (N, M) shape")
    if not np.all(np.isfinite(emu)) or not np.all(np.isfinite(legacy)):
        raise ValueError("this parity target requires complete finite moments")
    if ids.shape != (len(emu),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("cluster_ids must contain one integer code per observation")
    if np.any(ids < 0):
        raise ValueError("cluster_ids must be nonnegative")
    _, codes = np.unique(ids, return_inverse=True)
    np.testing.assert_allclose(emu, legacy, rtol=1e-12, atol=1e-12)

    n, m = emu.shape
    groups = int(codes.max()) + 1
    measure = EmpiricalMeasure.from_arrays(emu)
    strategy = ClusteredCovariance(
        cluster_ids=jnp.asarray(codes),
        n_clusters=groups,
        centered=centered,
        dof_correction=False,
    )
    mean = measure.expectation(_row_moment, 0.0)
    V = strategy.covariance(psi=_row_moment, theta=0.0, measure=measure)
    dgp = EmpiricalDGP(
        observation=jnp.asarray(legacy), sampling=ClusteredSampling(cluster_ids=codes)
    )
    oracle = GMM(moment_func=_identity_moments, dgp=dgp, backend="jax")
    omega = oracle.moment_restriction.omega_hat(0.0, centered=centered)
    # This is a comparison between package conventions, not a factor added
    # to Emu's criterion. Both covariance engines own their full calculations.
    expected_mean = np.asarray(oracle.gN(0.0)).ravel() / np.sqrt(n)
    expected_V = np.asarray(omega) / n
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    covariance_error = _relative_error(np.asarray(V), expected_V)
    np.testing.assert_allclose(
        V, expected_V, rtol=1e-10, atol=1e-12 * float(np.max(np.abs(expected_V)))
    )

    # Read the reference criterion through GMM with its own fixed weight;
    # Emu uses the same quadratic-form primitive as its criterion pipeline.
    fixed_oracle = GMM(
        moment_func=_identity_moments,
        dgp=dgp,
        backend="jax",
        weighting=jnp.linalg.inv(omega),
    )
    emu_Q = float(quadratic_form(V, mean))
    legacy_Q = float(fixed_oracle.data_criterion(0.0))
    np.testing.assert_allclose(emu_Q, legacy_Q, rtol=1e-9, atol=1e-10)
    return {
        "N": n,
        "M": m,
        "n_clusters": groups,
        "centered": centered,
        "scope": "complete moments; unit weights; no dof correction; identical point",
        "moment_max_relative_error": _relative_error(emu, legacy),
        "mean_max_relative_error": _relative_error(np.asarray(mean), expected_mean),
        "covariance_max_relative_error": covariance_error,
        "covariance_bitwise_equal_after_scale_conversion": bool(
            np.array_equal(np.asarray(V), expected_V)
        ),
        "emu_criterion": emu_Q,
        "legacy_criterion": legacy_Q,
        "criterion_relative_error": abs(emu_Q - legacy_Q) / max(abs(legacy_Q), 1.0),
        "covariance_condition": float(np.linalg.cond(expected_V)),
        "refitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle", type=Path, help="NPZ: emu_moments, legacy_moments, cluster_ids"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with np.load(arguments.bundle, allow_pickle=False) as bundle:
        results = [
            compare_clustered(
                bundle["emu_moments"],
                bundle["legacy_moments"],
                bundle["cluster_ids"],
                centered=centered,
            )
            for centered in (False, True)
        ]
    report = {
        "bundle_sha256": hashlib.sha256(arguments.bundle.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "installed_distribution_versions": {
            name: importlib.metadata.version(name)
            for name in ("emu-gmm", "manifoldgmm", "DGP_Protocol", "jax", "numpy")
        },
        "executed_sources": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name in (
                "emu_gmm.measures.empirical",
                "emu_gmm.covariance.clustered",
                "emu_gmm.covariance._outer_product",
                "emu_gmm._internal.cholesky",
                "manifoldgmm.econometrics.gmm",
                "manifoldgmm.econometrics.moment_restriction",
                "dgp_protocol.sampling",
            )
            for path in [_module_path(name)]
        },
        "source_identity_note": (
            "PYTHONPATH may override installed distributions; executed source "
            "paths and hashes, rather than distribution stamps, identify the code"
        ),
        "comparisons": results,
    }
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def _module_path(name: str) -> Path:
    source = importlib.import_module(name).__file__
    assert source is not None
    return Path(source)


if __name__ == "__main__":
    main()
