"""Audit native #180 sample identification without refitting (ledger Section 35).

This independent NumPy calculation uses the frozen preprojected instruments and
the exact FE identity in .coder/native180-proof-probe.org. It checks package G
against an analytic Jacobian, then reports sample ranks and partial information.
It does not estimate a population lower bound or change the native set law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/global/scratch/users/ligon/emu-gmm-backlog-20261024-job40438881")
DATA = ROOT / "native180-data-prep/results/job-40446236-aligned-data.npz"
WARM = ROOT / "native180-prereqs/raw/consumer/var/het_robust/het_point_corrected_K2.npz"
POSTFIT = (
    ROOT
    / "native180-rtr-replay-20261025/results/job-40498497/experiment/postfit-arrays.npz"
)
HASHES = {
    "data": "47fb95edc77cb403c5377d71e78939d7472fe5c2459b28559835aa24a8018d8e",
    "warm": "077911639de23bc044a7362fecf07034d894bdf62603edf9234995b1508d5161",
    "postfit": "cc54d5b92889a62aa3bfb146e3b2986b1698fc6dcf1d97be7c493228ac62c7f7",
}
NULL_S = 0.050088084721517094


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symmetric_basis(n: int) -> np.ndarray:
    """Columns are an orthonormal Frobenius basis of symmetric n-by-n arrays."""
    columns = []
    for i in range(n):
        matrix = np.zeros((n, n))
        matrix[i, i] = 1.0
        columns.append(matrix.ravel())
    for i in range(n):
        for j in range(i + 1, n):
            matrix = np.zeros((n, n))
            matrix[i, j] = matrix[j, i] = 1.0 / np.sqrt(2.0)
            columns.append(matrix.ravel())
    return np.column_stack(columns)


def singular_summary(matrix: np.ndarray, rank: int) -> dict[str, float | int]:
    values = np.linalg.svd(matrix, compute_uv=False)
    observed = int(np.count_nonzero(values > 1e-10 * values[0]))
    if observed != rank:
        raise ValueError(f"expected rank {rank}, observed {observed}")
    return {
        "rank": observed,
        "largest": float(values[0]),
        "smallest_nonzero": float(values[rank - 1]),
    }


def residual_norm(column: np.ndarray, basis: np.ndarray, rank: int) -> float:
    left, values, _ = np.linalg.svd(basis, full_matrices=False)
    if np.count_nonzero(values > 1e-10 * values[0]) != rank:
        raise ValueError(f"projection basis does not have rank {rank}")
    residual = column - left[:, :rank] @ (left[:, :rank].T @ column)
    return float(np.linalg.norm(residual))


def audit(data: Path, warm: Path, postfit: Path) -> dict:
    paths = {"data": data, "warm": warm, "postfit": postfit}
    actual_hashes = {name: sha256(path) for name, path in paths.items()}
    if actual_hashes != HASHES:
        raise ValueError(f"frozen input hashes differ: {actual_hashes}")

    with np.load(data, allow_pickle=False) as saved:
        obs = saved["obs_data"]
        product = saved["product_codes"]
        pair = saved["market_codes"]
        schur_pinv = saved["fe_schur_pinv"]
    with np.load(warm, allow_pickle=False) as saved:
        direction = saved["c_dir"]
    with np.load(postfit, allow_pickle=False) as saved:
        A = saved["theta_hat_A"]
        c0 = saved["theta_hat_c0"].item()
        G_package = saved["moment_jacobian"]
        H_package = saved["gn_hessian"]
        W = saved["weighting_matrix"]

    if obs.shape != (236088, 85) or G_package.shape != (43, 21):
        raise ValueError("frozen panel or restricted Jacobian dimensions differ")
    if A.shape != (5, 2) or direction.shape != (5,):
        raise ValueError("factor or fixed direction dimensions differ")
    Z = obs[:, 42:85]
    n = len(obs)
    product_means = np.bincount(product, minlength=45)
    pair_means = np.bincount(pair, minlength=5460)
    incidence = np.zeros((45, 5460))
    np.add.at(incidence, (product, pair), 1.0)

    def project_outcome(outcome: np.ndarray) -> np.ndarray:
        """Apply the frozen two-way FE projector to a transformed outcome."""
        by_product = np.bincount(product, weights=outcome, minlength=45)
        by_pair = np.bincount(pair, weights=outcome, minlength=5460)
        product_effect = schur_pinv @ (by_product - incidence @ (by_pair / pair_means))
        pair_effect = (by_pair - incidence.T @ product_effect) / pair_means
        return outcome - product_effect[product] - pair_effect[pair]

    product_orthogonality = max(
        np.max(np.abs(np.bincount(product, weights=Z[:, j]) / product_means))
        for j in range(43)
    )
    pair_orthogonality = max(
        np.max(np.abs(np.bincount(pair, weights=Z[:, j]) / pair_means))
        for j in range(43)
    )
    if product_orthogonality > 1e-12 or pair_orthogonality > 1e-12:
        raise ValueError("frozen instruments are not FE-orthogonal")

    z = obs[:, 27:32] @ direction
    y_without_amplitude = np.exp(-NULL_S * z) * obs[:, 0]
    delta = float(np.exp(-c0))
    b_unprojected = Z.T @ y_without_amplitude / n
    b = Z.T @ project_outcome(y_without_amplitude) / n
    projection_gap = float(np.linalg.norm(b - b_unprojected))
    B = Z.T @ obs[:, 2:27] / n
    P = Z.T @ obs[:, 32:42] / n
    unrestricted = np.column_stack((b, B @ symmetric_basis(5), P))
    unrestricted_rank = singular_summary(unrestricted, 26)
    unprojected_rank = singular_summary(
        np.column_stack((b_unprojected, B @ symmetric_basis(5), P)), 26
    )
    if unprojected_rank["smallest_nonzero"] <= projection_gap:
        raise ValueError("FE projection could erase the sample rank")
    normalized = unrestricted / np.linalg.norm(unrestricted, axis=0)
    normalized_rank = singular_summary(normalized, 26)

    factor_columns = []
    for i in range(5):
        for j in range(2):
            tangent = np.zeros_like(A)
            tangent[i, j] = 1.0
            d_gamma = tangent @ A.T + A @ tangent.T
            factor_columns.append(-B @ d_gamma.ravel())
    G_analytic = np.column_stack((*factor_columns, -delta * b, -P))
    G_gap = float(np.max(np.abs(G_analytic - G_package)))
    H_gap = float(np.max(np.abs(G_package.T @ W @ G_package - H_package)))
    if G_gap > 1e-11 or H_gap > 1e-9:
        raise ValueError("analytic/native Jacobian or information parity failed")

    whitened = np.linalg.cholesky(W).T @ G_package
    other = np.r_[0:10, 11:21]
    other_rank = singular_summary(whitened[:, other], 19)
    nuisance_rank = singular_summary(whitened, 20)
    c0_partial = residual_norm(whitened[:, 10], whitened[:, other], 19)
    s_jacobian = -delta * (Z.T @ project_outcome(z * y_without_amplitude) / n)
    s_partial = residual_norm(np.linalg.cholesky(W).T @ s_jacobian, whitened, 20)
    full_rank = singular_summary(
        np.column_stack((whitened, np.linalg.cholesky(W).T @ s_jacobian)), 21
    )

    return {
        "kind": "native180-frozen-sample-identification-v1",
        "input_sha256": actual_hashes,
        "null_s": NULL_S,
        "n_rows": n,
        "n_moments": 43,
        "n_pairs": 5460,
        "max_abs_fe_instrument_mean": {
            "product": float(product_orthogonality),
            "pair": float(pair_orthogonality),
        },
        "delta": delta,
        "factor_min_singular": float(np.linalg.svd(A, compute_uv=False)[-1]),
        "projected_amplitude_column_gap": projection_gap,
        "unprojected_linear_design_min_singular": unprojected_rank["smallest_nonzero"],
        "weyl_projected_rank_lower_bound": unprojected_rank["smallest_nonzero"]
        - projection_gap,
        "unrestricted_linear_design": unrestricted_rank,
        "column_normalized_linear_design": normalized_rank,
        "max_abs_package_jacobian_gap": G_gap,
        "max_abs_package_information_gap": H_gap,
        "other_nuisance_weighted": other_rank,
        "all_nuisance_weighted": nuisance_rank,
        "full_parameter_weighted": full_rank,
        "c0_partial_information": c0_partial**2,
        "delta_coordinate_partial_information": (c0_partial / delta) ** 2,
        "s_partial_information": s_partial**2,
        "s_partial_information_div_delta_squared": (s_partial / delta) ** 2,
        "interpretation": (
            "sample rank and original pair-covariance-weighted information only; "
            "no two-way or population lower bound"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--warm", type=Path, default=WARM)
    parser.add_argument("--postfit", type=Path, default=POSTFIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data, args.warm, args.postfit)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
