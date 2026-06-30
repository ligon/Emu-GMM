"""Load the pyblp reference arrays into emu-side ``BLPData`` bundles."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from blp_model import BLPData

REF = Path(__file__).parent / "reference"


def load_estimates() -> dict:
    return json.loads((REF / "estimates.json").read_text())


def load_data(spec: str) -> BLPData:
    a = np.load(REF / f"{spec}_arrays.npz")
    T, J, _ = a["shares"].shape
    K2 = a["X2"].shape[2]

    def get(name, shape):
        if name in a.files:
            return jnp.asarray(a[name], dtype=jnp.float64)
        return jnp.zeros(shape, dtype=jnp.float64)

    I = a["nodes"].shape[1] if "nodes" in a.files else 1
    D = a["demographics"].shape[2] if "demographics" in a.files else 0

    return BLPData(
        shares=jnp.asarray(a["shares"][..., 0], dtype=jnp.float64),  # (T, J)
        prices=jnp.asarray(a["prices"][..., 0], dtype=jnp.float64),  # (T, J)
        X1=jnp.asarray(a["X1"], dtype=jnp.float64),  # (T, J, K1)
        X2=jnp.asarray(a["X2"], dtype=jnp.float64),  # (T, J, K2)
        Z=jnp.asarray(a["ZD_tilde"], dtype=jnp.float64),  # (T, J, M) demeaned
        nodes=get("nodes", (T, I, K2)),  # (T, I, K2)
        demographics=get("demographics", (T, I, D)),  # (T, I, D)
        agent_weights=(
            jnp.asarray(a["agent_weights"][..., 0], dtype=jnp.float64)
            if "agent_weights" in a.files
            else jnp.ones((T, I), dtype=jnp.float64)
        ),
    )


def load_W(spec: str) -> jnp.ndarray:
    return jnp.asarray(np.load(REF / f"{spec}_W.npy"), dtype=jnp.float64)


def load_updated_W(spec: str) -> jnp.ndarray:
    """The weight pyblp recomputes at theta_hat for its reported SEs."""
    p = REF / f"{spec}_updated_W.npy"
    return jnp.asarray(np.load(p), dtype=jnp.float64) if p.exists() else load_W(spec)


def load_delta(spec: str) -> jnp.ndarray:
    """pyblp's converged delta, grouped by market into (T, J)."""
    a = np.load(REF / f"{spec}_arrays.npz")
    T, J, _ = a["shares"].shape
    return jnp.asarray(np.load(REF / f"{spec}_delta.npy").reshape(T, J), dtype=jnp.float64)
