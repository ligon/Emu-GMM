"""Event-complete current-CU/subvector native panel artifact (ledger Section 28).

This validation driver projects the theta-dependent outcome jointly on product
and market effects. It does not recreate the historical Fixed/full-vector run.
Run small tests first; actual panel diagnostics and fits belong in Slurm jobs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from emu_gmm import (
    ClusteredCovariance,
    EmpiricalMeasure,
    Euclidean,
    ManifoldLeaf,
    PSDFixedRank,
    RandomSetLaw,
    profiled_k_confidence_set,
    riemannian_lm,
)
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.csgraph import connected_components

ARTIFACT_ROOT = Path("/global/scratch/users/ligon/emu-gmm-backlog-20261024-job40438881")
DATA = ARTIFACT_ROOT / "native180-data-prep/results/job-40446236-aligned-data.npz"
REPORT = ARTIFACT_ROOT / "native180-data-prep/results/job-40446236-report.json"
WARM = (
    ARTIFACT_ROOT
    / "native180-prereqs/raw/consumer/var/het_robust/het_point_corrected_K2.npz"
)
HASHES = {
    "data": "47fb95edc77cb403c5377d71e78939d7472fe5c2459b28559835aa24a8018d8e",
    "report": "9939853c1691d1af6ddd05bbf505b3a7b0d8aea8a3c6974c61cdc09b9cec4307",
    "warm": "077911639de23bc044a7362fecf07034d894bdf62603edf9234995b1508d5161",
}
COLUMNS = {
    "Y_idx": 0,
    "logp_idx": 1,
    "zetaZ_start": 2,
    "zetaZ_end": 27,
    "zt_start": 27,
    "zt_end": 32,
    "pi_start": 32,
    "pi_end": 42,
    "inst_start": 42,
}
DIMENSIONS = {
    "N": 236088,
    "M": 43,
    "m_chars": 5,
    "n_pi": 10,
    "n_clusters": 5460,
    "K": 2,
}


@jdc.pytree_dataclass
class NativePoint:
    A: ManifoldLeaf
    c0: ManifoldLeaf
    s: ManifoldLeaf
    pi: ManifoldLeaf


def point(A, c0, s, pi):
    """Exact native shape contract; never truncate/pad or substitute a warm leaf."""
    if (
        np.shape(A) != (5, 2)
        or np.shape(pi) != (10,)
        or np.shape(c0) not in ((), (1,))
        or np.shape(s) not in ((), (1,))
    ):
        raise ValueError("native point requires A(5,2), pi(10), scalar/length-one c0,s")
    return NativePoint(
        ManifoldLeaf(jnp.asarray(A), PSDFixedRank(5, 2)),
        ManifoldLeaf(jnp.asarray(c0).reshape(1), Euclidean(1)),
        ManifoldLeaf(jnp.asarray(s).reshape(1), Euclidean(1)),
        ManifoldLeaf(jnp.asarray(pi), Euclidean(10)),
    )


class ProjectedModel:
    """Unweighted retained-panel FE reduction with all keys/counts from mu.

    The product-side sum-zero basis removes only the known connected-graph
    gauge. Solving Q' S Q gives the same effects as S+ with rank P-1; no
    N-by-N projection, frozen theta-dependent effects, or stopped AD is used.
    """

    def __init__(self, direction, n_products, n_markets, columns=None, inst_end=85):
        self.direction = jnp.asarray(direction)
        self.n_products = int(n_products)
        self.n_markets = int(n_markets)
        self.columns = dict(COLUMNS if columns is None else columns)
        self.inst_end = int(inst_end)
        basis = np.vstack([np.eye(n_products - 1), -np.ones(n_products - 1)])
        self.Q = jnp.asarray(np.linalg.qr(basis, mode="reduced")[0])

    def transformed_outcome(self, x, theta):
        c = self.columns
        z = x[:, c["zt_start"] : c["zt_end"]] @ self.direction
        return jnp.exp(-theta.c0.array[0] - theta.s.array[0] * z) * x[:, c["Y_idx"]]

    def aux(self, measure, theta):
        # The declared population is every retained row, unweighted. main()
        # validates unit weights/full support before fitting the frozen panel.
        x = measure.x
        product = x[:, self.inst_end].astype(jnp.int32)
        market = x[:, self.inst_end + 1].astype(jnp.int32)
        C = jnp.bincount(
            product * self.n_markets + market,
            length=self.n_products * self.n_markets,
        ).reshape(self.n_products, self.n_markets)
        np_, nm = jnp.sum(C, axis=1), jnp.sum(C, axis=0)
        y = self.transformed_outcome(x, theta)
        yp = jnp.bincount(product, weights=y, length=self.n_products)
        ym = jnp.bincount(market, weights=y, length=self.n_markets)
        schur = jnp.diag(np_) - (C / nm[None, :]) @ C.T
        rhs = yp - C @ (ym / nm)
        a = self.Q @ jnp.linalg.solve(self.Q.T @ schur @ self.Q, self.Q.T @ rhs)
        b = (ym - C.T @ a) / nm
        return a, b

    def psi(self, row, theta, aux):
        c = self.columns
        a, b = aux
        product = row[self.inst_end].astype(jnp.int32)
        market = row[self.inst_end + 1].astype(jnp.int32)
        z = row[c["zt_start"] : c["zt_end"]] @ self.direction
        y = jnp.exp(-theta.c0.array[0] - theta.s.array[0] * z) * row[c["Y_idx"]]
        residual = y - a[product] - b[market] - row[c["logp_idx"]]
        residual -= (
            row[c["zetaZ_start"] : c["zetaZ_end"]]
            @ (theta.A.array @ theta.A.array.T).ravel()
        )
        residual -= row[c["pi_start"] : c["pi_end"]] @ theta.pi.array
        # Explicit original endpoint excludes the two appended grouping keys.
        return residual * row[c["inst_start"] : self.inst_end]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    """Strict JSON while retaining explicit nonfinite scalar provenance."""
    if dataclasses.is_dataclass(value):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | bool | int) or value is None:
        return value
    if isinstance(value, np.ndarray | jax.Array | np.generic):
        return json_value(np.asarray(value).tolist())
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    raise TypeError(f"unsupported provenance value {type(value)}")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def protocol(grid, alpha):
    return {
        "kind": "new-projected-native-current-CU-subvector-artifact",
        "grid": grid,
        "alpha": alpha,
        "statistic": "K",
        "interest": ["s"],
        "nuisance": ["A", "c0", "pi"],
        "optimizer": {
            "backend": "riemannian_lm",
            "rtol": 1e-9,
            "atol": 1e-9,
            "max_steps": 1500,
        },
        "weighting": "public profiler default ContinuouslyUpdated",
        "regularization": "package default DiagonalTikhonov",
        "penalty": None,
        "nuisance_validity": "public profiler certified-or-FOC (default foc_tol)",
        "reference_assumption": "the public subvector reference also assumes strong nuisance identification; this artifact is not a validation of that assumption",
        "projection": "unweighted retained-panel joint product/market FE at every theta",
        "FE_gauge": "one connected component; sum-zero product gauge; rank44",
        "historical_limit": "does not reconstruct historical runtime or missing convergence",
        "interpretation_limit": "one event-complete grid artifact, not coverage calibration or continuous-domain emptiness",
    }


def source_environment():
    import emu_gmm

    root = Path(emu_gmm.__file__).resolve().parent
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    packages = {}
    for name in (
        "emu-gmm",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "jax-dataclasses",
        "optimistix",
        "haliax",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    installed = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    return {
        "driver_sha256": sha256(__file__),
        "package_source": str(root),
        "git_HEAD": head,
        "source_files_sha256": files,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "installed_distributions": installed,
        "installed_distributions_sha256": hashlib.sha256(
            json.dumps(installed, sort_keys=True).encode()
        ).hexdigest(),
        "numerical_environment": {
            k: os.environ.get(k)
            for k in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
            )
        },
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "slurm": {
            k: os.environ.get(k)
            for k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")
        },
    }


def load_inputs(data_path, report_path, warm_path):
    paths = {"data": data_path, "report": report_path, "warm": warm_path}
    hashes = {key: sha256(path) for key, path in paths.items()}
    if hashes != HASHES:
        raise ValueError(f"frozen input hash mismatch: {hashes}")
    report = json.loads(Path(report_path).read_text())
    if report["dimensions"] != DIMENSIONS or report["obs_columns"] != COLUMNS:
        raise ValueError("frozen report dimensions/column layout mismatch")
    with np.load(warm_path, allow_pickle=False) as z:
        if any(
            np.asarray(z[k]).shape != () or int(z[k]) != v
            for k, v in DIMENSIONS.items()
        ):
            raise ValueError("warm dimension metadata mismatch")
        warm = {
            k: np.asarray(z[k]).copy() for k in ("A", "c0", "slope_norm", "pi", "c_dir")
        }
    for key, shape in {
        "A": (5, 2),
        "c0": (),
        "slope_norm": (),
        "pi": (10,),
        "c_dir": (5,),
    }.items():
        if warm[key].shape != shape or not np.isfinite(warm[key]).all():
            raise ValueError(f"invalid warm component {key}")
    if np.linalg.matrix_rank(warm["A"]) != 2 or not np.isclose(
        np.linalg.norm(warm["c_dir"]), 1
    ):
        raise ValueError("warm A rank or direction normalization mismatch")
    with np.load(data_path, allow_pickle=False) as z:
        obs = z["obs_data"]
        product, market = z["product_codes"], z["market_codes"]
        oracle = z["fe_projected_raw_Y"]
        oracle_pinv = z["fe_schur_pinv"]
        if (
            obs.shape != (236088, 85)
            or obs.dtype != np.float64
            or not np.isfinite(obs).all()
        ):
            raise ValueError("frozen OBS shape/dtype/finiteness mismatch")
        for key, ids, n in (("product", product, 45), ("market", market, 5460)):
            if (
                ids.shape != (len(obs),)
                or ids.dtype.kind not in "iu"
                or np.any(ids < 0)
                or np.any(ids >= n)
                or np.any(np.bincount(ids, minlength=n) == 0)
            ):
                raise ValueError(f"invalid {key} grouping")
        C = coo_matrix((np.ones(len(obs)), (product, market)), shape=(45, 5460)).tocsr()
        components = connected_components(
            bmat([[None, C], [C.T, None]], format="csr"), directed=False
        )[0]
        if components != 1 or report["bipartite_components"] != 1:
            raise ValueError(
                "the experiment requires the known connected grouping graph"
            )
        counts_p, counts_m = np.bincount(product), np.bincount(market)
        np.testing.assert_array_equal(counts_p, z["product_counts"])
        np.testing.assert_array_equal(counts_m, z["market_counts"])
        dense_C = C.toarray()
        np.testing.assert_array_equal(dense_C, z["fe_C"])
        schur = np.diag(counts_p) - (dense_C / counts_m[None, :]) @ dense_C.T
        eig = np.linalg.eigvalsh((schur + schur.T) / 2)
        tol = 64 * 45 * np.finfo(float).eps * max(1.0, np.max(np.abs(eig)))
        if abs(eig[0]) > tol or np.any(eig[1:] <= tol):
            raise ValueError("Schur rank differs from the graph-identified 44")
    x = np.column_stack([obs, product, market])
    measure = EmpiricalMeasure(
        jnp.asarray(x), jnp.ones((len(obs), 43)), jnp.ones(len(obs))
    )
    model = ProjectedModel(warm["c_dir"], 45, 5460)
    initial = point(warm["A"], warm["c0"], warm["slope_norm"], warm["pi"])
    raw_point = point(warm["A"], 0.0, 0.0, warm["pi"])
    a, b = jax.jit(model.aux)(measure, raw_point)
    projected = obs[:, 0] - np.asarray(a)[product] - np.asarray(b)[market]
    np.testing.assert_allclose(projected, oracle, atol=1e-9, rtol=1e-9)
    # Independent frozen NumPy Schur operator, applied to the CURRENT warm
    # transformed outcome only as an oracle, never used by model.aux().
    transformed = (
        np.exp(-warm["c0"] - warm["slope_norm"] * (obs[:, 27:32] @ warm["c_dir"]))
        * obs[:, 0]
    )
    yp = np.bincount(product, weights=transformed, minlength=45)
    ym = np.bincount(market, weights=transformed, minlength=5460)
    oracle_a = oracle_pinv @ (yp - dense_C @ (ym / counts_m))
    oracle_b = (ym - dense_C.T @ oracle_a) / counts_m
    oracle_warm = transformed - oracle_a[product] - oracle_b[market]
    warm_a, warm_b = jax.jit(model.aux)(measure, initial)
    projected_warm = (
        transformed - np.asarray(warm_a)[product] - np.asarray(warm_b)[market]
    )
    np.testing.assert_allclose(projected_warm, oracle_warm, atol=1e-9, rtol=1e-9)
    evidence = {
        "paths": paths,
        "sha256": hashes,
        "dimensions": DIMENSIONS,
        "columns": {**COLUMNS, "inst_end": 85, "product_key": 85, "market_key": 86},
        "connected_components": components,
        "schur_eigenvalues": eig,
        "schur_rank_tolerance": tol,
        "raw_Y_oracle_max_abs_gap": np.max(np.abs(projected - oracle)),
        "warm_transformed_Y_oracle_max_abs_gap": np.max(
            np.abs(projected_warm - oracle_warm)
        ),
        "unit_weights_full_mask": True,
        "data_preparation_index_fingerprint": report["index_fingerprint_sha256"],
        "warm": warm,
    }
    covariance = ClusteredCovariance(jnp.asarray(market, dtype=jnp.float64), 5460)
    return model, initial, measure, covariance, evidence


class RecordingOptimizer:
    """Transparent optimizer-protocol forwarding; profiler validity is separate."""

    def __init__(self, optimizer, path):
        self.optimizer = optimizer
        self.path = Path(path)
        self.records = []

    def __call__(self, residual_fn, theta_init, manifold_spec, *, args=None):
        start = time.perf_counter()
        print(
            json.dumps({"event": "nuisance-start", "index": len(self.records)}),
            flush=True,
        )
        theta, info = self.optimizer(residual_fn, theta_init, manifold_spec, args=args)
        jax.block_until_ready((theta, info))
        paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(theta)
        checkpoint = self.path.parent / f"point-{len(self.records):03d}-nuisance.npz"
        leaves = {
            f"leaf_{i:03d}": np.asarray(leaf)
            for i, (_path, leaf) in enumerate(paths_and_leaves)
        }
        np.savez_compressed(checkpoint, **leaves)
        record = {
            "index": len(self.records),
            "start_monotonic": start,
            "solver_wall_seconds": time.perf_counter() - start,
            "optimizer_info": info,
            "parameter_checkpoint": checkpoint.name,
            "parameter_checkpoint_sha256": sha256(checkpoint),
            "parameter_treedef_repr": repr(treedef),
            "parameter_leaf_order": [
                {
                    "key": f"leaf_{i:03d}",
                    "path": [str(k) for k in path],
                    "shape": np.shape(leaf),
                    "dtype": str(np.asarray(leaf).dtype),
                }
                for i, (path, leaf) in enumerate(paths_and_leaves)
            ],
            "stage": "raw optimizer only; public profiler K/FOC validity pending",
            "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        self.records.append(record)
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(json_value(record), sort_keys=True, allow_nan=False) + "\n"
            )
        print(json.dumps({"event": "nuisance-end", **json_value(record)}), flush=True)
        return theta, info


def persistence_check(cs, path):
    law = RandomSetLaw.from_k_confidence_sets([cs])
    law.save(path)
    restored = RandomSetLaw.load(path)
    for p in cs.grid:
        assert law.point_verdicts(float(p)) == restored.point_verdicts(float(p))
    for size in range(1, len(cs.grid) + 1):
        for query in itertools.combinations(cs.grid, size):
            assert json_value(law.capacity(query)) == json_value(
                restored.capacity(query)
            )
    for negate in (False, True):
        assert law.event_share("grid_empty", negate=negate) == restored.event_share(
            "grid_empty", negate=negate
        )
    before, after = law.draws[0], restored.draws[0]
    for name in ("grid", "p_grid", "in_set"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    for name in (
        "alpha",
        "statistic",
        "intervals",
        "topology",
        "open_left",
        "open_right",
        "n_invalid",
        "invalid_indices",
        "n_nonconverged",
        "nonconverged_indices",
    ):
        assert getattr(before, name) == getattr(after, name)
    return {
        "parity": True,
        "capacity_queries_checked": 2 ** len(cs.grid) - 1,
        "point_verdicts": [law.point_verdicts(float(p))[0] for p in cs.grid],
        "capacity_full_grid": law.capacity(cs.grid),
        "grid_empty_recorded_event": law.event_share("grid_empty"),
        "known_empty_scanned_set": not any(cs.in_set)
        and cs.n_invalid == 0
        and cs.n_nonconverged == 0,
        "interpretation": "an all-unknown scan is not a known empty set; no global-domain claim",
    }


def diagnostics(model, initial, measure, covariance, output):
    """Pre-fit package means/G/V and complete row-D summaries (can be large)."""
    start = time.perf_counter()
    means, _g, _exposure, N = measure.expectation_and_contributions(model, initial)
    G = measure.jacobian(model, initial)
    V = covariance.covariance(model, initial, measure)
    D = measure.jacobian_contributions(model, initial)
    probe = np.unique(np.linspace(0, len(measure.x) - 1, 8, dtype=int))
    jax.block_until_ready((means, G, V, D))
    np.testing.assert_allclose(
        np.asarray(jnp.sum(D, axis=0) / N[:, None]), G, rtol=1e-8, atol=1e-8
    )
    np.savez_compressed(
        output / "pre-fit-diagnostics.npz",
        means=np.asarray(means),
        G=np.asarray(G),
        V=np.asarray(V),
        D_probe=np.asarray(D[probe]),
        D_probe_rows=probe,
        N_j=np.asarray(N),
    )
    return {
        "wall_seconds": time.perf_counter() - start,
        "D_shape": D.shape,
        "D_finite": bool(jnp.all(jnp.isfinite(D))),
        "D_max_abs": float(jnp.max(jnp.abs(D))),
        "G_finite": bool(jnp.all(jnp.isfinite(G))),
        "V_finite": bool(jnp.all(jnp.isfinite(V))),
        "means": means,
    }


def synthetic_smoke_inputs():
    """Tiny connected API control, explicitly unrelated to scientific data."""
    rng = np.random.default_rng(180)
    product = np.tile([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], 8)
    market = np.tile([0, 1, 1, 1, 2, 3, 0, 2, 3, 3], 8)
    F = np.column_stack([np.eye(3)[product], np.eye(4)[market]])
    obs = rng.normal(size=(len(product), 85))
    # A dense projector is used ONLY for this tiny independent fixture.
    within = np.eye(len(obs)) - F @ np.linalg.pinv(F)
    indices = np.r_[1:27, 32:85]
    obs[:, indices] = within @ obs[:, indices]
    direction = rng.normal(size=5)
    direction /= np.linalg.norm(direction)
    A, pi = rng.normal(scale=0.2, size=(5, 2)), rng.normal(scale=0.1, size=10)
    level, magnitude = 0.01, 0.1
    rhs = obs[:, 1] + obs[:, 2:27] @ (A @ A.T).ravel() + obs[:, 32:42] @ pi
    rhs += F @ rng.normal(size=7) + rng.normal(scale=0.03, size=len(obs))
    obs[:, 0] = np.exp(level + magnitude * (obs[:, 27:32] @ direction)) * rhs
    x = np.column_stack([obs, product, market])
    measure = EmpiricalMeasure(
        jnp.asarray(x), jnp.ones((len(obs), 43)), jnp.ones(len(obs))
    )
    return (
        ProjectedModel(direction, 3, 4),
        point(0.95 * A, level + 0.01, magnitude, 0.95 * pi),
        measure,
        ClusteredCovariance(jnp.asarray(market, dtype=float), 4),
        {
            "kind": "synthetic-api-control-not-scientific",
            "seed": 180,
            "N": len(obs),
            "data_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--warm", type=Path, default=WARM)
    parser.add_argument("--grid", default="0,0.10017617")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--print-protocol", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="tiny public-profiler/codec API test, max_steps=2; never scientific panel output",
    )
    args = parser.parse_args()
    grid = np.array([float(v) for v in args.grid.split(",")])
    if (
        not 2 <= len(grid) <= 12
        or not np.isfinite(grid).all()
        or not np.all(np.diff(grid) > 0)
        or not 0 < args.alpha < 1
    ):
        parser.error("need 2..12 strictly increasing finite grid points and 0<alpha<1")
    manifest = {
        "protocol": protocol(grid, args.alpha),
        "started_UTC": datetime.now(UTC).isoformat(),
        "diagnostics_only": args.diagnostics_only,
    }
    if args.synthetic_smoke:
        manifest["protocol"]["kind"] = "synthetic-api-smoke-not-scientific"
        manifest["protocol"]["optimizer"]["max_steps"] = 2
        manifest["protocol"][
            "FE_gauge"
        ] = "one connected component; sum-zero product gauge; rank2"
    if args.print_protocol:
        print(json.dumps(json_value(manifest), indent=2, sort_keys=True))
        return
    try:
        from emu_gmm.types import ReducingModel
    except ImportError as exc:
        raise RuntimeError(
            "native projected fitting requires the released #113 empirical ReducingModel/profile adapter"
        ) from exc
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest["source_environment"] = source_environment()
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        model, initial, measure, covariance, evidence = (
            synthetic_smoke_inputs()
            if args.synthetic_smoke
            else load_inputs(args.data, args.report, args.warm)
        )
        if not isinstance(model, ReducingModel):
            raise RuntimeError("incompatible ReducingModel protocol")
        manifest["inputs"] = evidence
        write_json(args.output_dir / "manifest.json", manifest)
        if args.diagnostics_only:
            manifest["pre_fit_diagnostics"] = diagnostics(
                model, initial, measure, covariance, args.output_dir
            )
        else:
            optimizer = RecordingOptimizer(
                riemannian_lm(
                    rtol=1e-9, atol=1e-9, max_steps=2 if args.synthetic_smoke else 1500
                ),
                args.output_dir / "optimizer.jsonl",
            )

            def build(s):
                return point(initial.A.array, initial.c0.array, s, initial.pi.array)

            cs, points = profiled_k_confidence_set(
                build,
                grid,
                measure,
                covariance,
                model,
                profile=["s"],
                alpha=args.alpha,
                nuisance_optimizer=optimizer,
                keep_results=True,
                return_profiled_points=True,
            )
            stop = time.perf_counter()
            if len(optimizer.records) != len(grid):
                raise RuntimeError(
                    "expected exactly one optimizer record per profile point"
                )
            for i, record in enumerate(optimizer.records):
                next_start = (
                    stop
                    if i + 1 == len(grid)
                    else optimizer.records[i + 1]["start_monotonic"]
                )
                record["optimizer_start_to_next_optimizer_or_profile_end_seconds"] = (
                    next_start - record["start_monotonic"]
                )
                record["grid_value"] = grid[i]
                record["profiler_nuisance_valid"] = i not in cs.nonconverged_indices
                record["statistic_invalid"] = i in cs.invalid_indices
            manifest["per_point"] = optimizer.records
            manifest["confidence_set"] = {
                f.name: getattr(cs, f.name)
                for f in dataclasses.fields(cs)
                if f.name != "results"
            }
            manifest["statistics"] = [json_value(result) for result in cs.results]
            manifest["persistence"] = persistence_check(
                cs, args.output_dir / "random-set-law.npz"
            )
            arrays = {
                name: np.array([getattr(result, name) for result in cs.results])
                for name in ("K", "S", "J", "p_K", "p_S", "p_J", "df_K", "df_S", "df_J")
            }
            np.savez_compressed(
                args.output_dir / "profile-grid.npz",
                grid=grid,
                p_grid=cs.p_grid,
                in_set=cs.in_set,
                invalid_indices=np.asarray(cs.invalid_indices, dtype=int),
                nonconverged_indices=np.asarray(cs.nonconverged_indices, dtype=int),
                A=np.stack([p.A.array for p in points]),
                c0=np.stack([p.c0.array for p in points]),
                s=np.stack([p.s.array for p in points]),
                pi=np.stack([p.pi.array for p in points]),
                **arrays,
            )
    except Exception as exc:
        manifest["execution_failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest["max_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["finished_UTC"] = datetime.now(UTC).isoformat()
        manifest["outputs_sha256"] = {
            p.name: sha256(p)
            for p in sorted(args.output_dir.iterdir())
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "complete",
                "manifest": str(args.output_dir / "manifest.json"),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "diagnostics_only": args.diagnostics_only,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
