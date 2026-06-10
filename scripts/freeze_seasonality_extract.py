#!/usr/bin/env python
"""Freeze the Seasonality euler_observation() extract into tests/data/.

Provenance tool for the real-data acceptance test (issue #128).  Runs
``src/euler_data.py::euler_observation(stock_indicator=2)`` from a
checkout of the sibling ``Seasonality`` repository and writes

- ``tests/data/seasonality_euler_extract.npz`` -- the derived arrays
  (compressed; no household identifiers: the (i, t) row MultiIndex is
  deliberately dropped), and
- ``tests/data/seasonality_euler_extract.json`` -- machine-readable
  provenance: source commit, generation command, shapes, per-column
  NaN-aware checksums, and the SHA-256 of the committed npz.

The derived columns are transformed m/delta/R/z quantities (owner
approved for in-repo storage 2026-06-10; see issue #128).  Regeneration
requires the private Seasonality data tree, so the committed npz is the
artifact of record; this script documents exactly how it was produced
and lets a holder of the source data verify array-level equality.

Usage (from the Emu-GMM repo root)::

    <seasonality>/.venv/bin/python scripts/freeze_seasonality_extract.py \
        --seasonality ../Seasonality --out tests/data

The interpreter must be Seasonality's own venv (its loader stack reads
the data tree); this script itself needs only numpy + stdlib.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

NPZ_NAME = "seasonality_euler_extract.npz"
JSON_NAME = "seasonality_euler_extract.json"


def _git_describe(repo: Path) -> dict[str, str]:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if _run("status", "--porcelain", "-uno") else "no",
    }


def _column_checksums(observation: np.ndarray, names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for j, name in enumerate(names):
        col = observation[:, j]
        finite = np.isfinite(col)
        out[name] = {
            "n_finite": int(finite.sum()),
            "nan_aware_sum": float(np.nansum(col)),
            "nan_aware_abs_sum": float(np.nansum(np.abs(col))),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seasonality",
        type=Path,
        required=True,
        help="path to the Seasonality repo root",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="output directory (Emu-GMM tests/data)"
    )
    ap.add_argument(
        "--stock-indicator",
        type=int,
        default=2,
        help="forwarded to euler_observation (2 = published table)",
    )
    args = ap.parse_args()

    seasonality = args.seasonality.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # euler_data loads from data/ via relative paths: run from the repo root.
    os.chdir(seasonality)
    sys.path.insert(0, str(seasonality / "src"))
    from euler_data import euler_observation  # noqa: E402

    eo = euler_observation(stock_indicator=args.stock_indicator)

    column_names = [None] * len(eo.column_index)
    for name, j in eo.column_index.items():
        column_names[j] = name
    assert all(n is not None for n in column_names)

    # Cast label arrays to unicode ('<U*') explicitly: pandas-derived
    # string arrays are object dtype, which np.load refuses without
    # allow_pickle (and pickled arrays do not belong in a test fixture).
    npz_path = out_dir / NPZ_NAME
    np.savez_compressed(
        npz_path,
        observation=eo.observation.astype(np.float64),
        column_names=np.asarray(column_names, dtype=str),
        crops=np.asarray(list(eo.crops), dtype=str),
        cluster_ids=eo.cluster_ids.astype(np.int64),
        arm_codes=eo.arm_codes.astype(np.int64),
        psu_ids=eo.psu_ids.astype(np.int64),
        t_codes=eo.t_codes.astype(np.int64),
        cid_labels=np.asarray(eo.cid_labels, dtype=str),
        psu_labels=np.asarray(eo.psu_labels, dtype=str),
    )

    sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    provenance = {
        "artifact": NPZ_NAME,
        "sha256": sha256,
        "generated": _dt.date.today().isoformat(),
        "generator": "scripts/freeze_seasonality_extract.py",
        "source": {
            "repo": "Seasonality (sibling repo; private data tree)",
            "entry_point": (
                f"src/euler_data.py::euler_observation("
                f"stock_indicator={args.stock_indicator})"
            ),
            **_git_describe(seasonality),
        },
        "command": (
            "<seasonality>/.venv/bin/python "
            "scripts/freeze_seasonality_extract.py "
            f"--seasonality <seasonality> --out tests/data "
            f"--stock-indicator {args.stock_indicator}"
        ),
        "notes": [
            "Derived columns only (transformed m/delta/R/z); the (i, t) "
            "row MultiIndex (household identifiers) is deliberately not "
            "included. Owner approved in-repo storage 2026-06-10 "
            "(Emu-GMM issue #128).",
            "Numerical content of m/delta/R byte-matches Seasonality "
            "src/analysis_data.py::analysis_data(stock_indicator=2); see "
            "the euler_data.py module docstring.",
            "The npz byte stream is not reproducible across regenerations "
            "(zip metadata); verify regenerations via the array-level "
            "column checksums below, not the file hash.",
        ],
        "shapes": {
            "observation": list(eo.observation.shape),
            "n_strata": int(len(eo.cid_labels)),
            "n_psu": int(eo.n_psu),
            "n_t": int(eo.t_codes.max()) + 1,
            "crops": list(eo.crops),
        },
        "column_checksums": _column_checksums(
            eo.observation, column_names  # type: ignore[arg-type]
        ),
    }
    json_path = out_dir / JSON_NAME
    json_path.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {npz_path} ({npz_path.stat().st_size} bytes)")
    print(f"wrote {json_path}")
    print(f"sha256: {sha256}")
    print(f"observation shape: {eo.observation.shape}")


if __name__ == "__main__":
    main()
