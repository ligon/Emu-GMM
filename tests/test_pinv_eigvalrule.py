r"""Unit tests for the Phase-4 gauge-aware pseudo-inverse helper.

Covers the red-team risks on ``pinv_eigvalrule``:

* R1/R11: drops the SMALLEST eigenvalues (ascending ``eigh``), correct
  ``[drop_smallest:]`` slice;
* R2/R24: ``drop_smallest`` is a static Python int; jit/vmap-safe;
* R3/R11/R12: true Moore--Penrose property; ``eigh`` (symmetric);
* R13: ``drop_smallest == 0`` reduces to ``inv()`` BITWISE;
* R23: symmetry enforced (asymmetric input handled).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule

jax.config.update("jax_enable_x64", True)


def _spd(seed, d, evals=None):
    rng = np.random.default_rng(seed)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(d, d))))
    if evals is None:
        evals = jnp.asarray(rng.uniform(1.0, 5.0, size=d))
    else:
        evals = jnp.asarray(evals)
    return (Q * evals) @ Q.T, Q, evals


def test_drop_zero_is_bitwise_inv():
    """v1 non-regression: drop_smallest=0 -> exact jnp.linalg.inv (R13)."""
    M, _, _ = _spd(0, 5)
    assert bool(jnp.array_equal(pinv_eigvalrule(M, drop_smallest=0), jnp.linalg.inv(M)))


def test_drop_zero_scalar_one_by_one():
    """1x1 Positive-scalar info: drop 0 -> exact reciprocal."""
    M = jnp.asarray([[3.0]])
    assert bool(jnp.array_equal(pinv_eigvalrule(M, drop_smallest=0), jnp.linalg.inv(M)))


def test_drops_smallest_eigenvalues_not_largest():
    """Drop the gauge-zero (smallest), keep the identified (largest) (R1)."""
    evals = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    M, Q, _ = _spd(1, 5, evals=evals)
    pinv = pinv_eigvalrule(M, drop_smallest=1)
    assert bool(jnp.all(jnp.isfinite(pinv)))
    # The reconstructed pinv has eigenvalues {0, 1, 1/2, 1/3, 1/4}:
    # the zero direction stays null, the four identified directions invert.
    pev = jnp.sort(jnp.linalg.eigvalsh(0.5 * (pinv + pinv.T)))
    expected = jnp.sort(jnp.asarray([0.0, 1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4]))
    assert bool(jnp.allclose(pev, expected, atol=1e-10))


def test_moore_penrose_property():
    """pinv @ M @ pinv == pinv on the identified subspace (R3)."""
    evals = jnp.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])  # gauge_dim=2
    M, _, _ = _spd(2, 6, evals=evals)
    pinv = pinv_eigvalrule(M, drop_smallest=2)
    assert bool(jnp.allclose(pinv @ M @ pinv, pinv, atol=1e-9))
    assert bool(jnp.allclose(M @ pinv @ M, M, atol=1e-9))


def test_rank_is_d_minus_drop():
    evals = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    M, _, _ = _spd(3, 5, evals=evals)
    pinv = pinv_eigvalrule(M, drop_smallest=1)
    nz = int(jnp.sum(jnp.abs(jnp.linalg.eigvalsh(pinv)) > 1e-9))
    assert nz == 4


def test_matches_numpy_pinv_for_rank_deficient():
    """Agrees with jnp.linalg.pinv on a known-rank matrix (R12)."""
    evals = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    M, _, _ = _spd(4, 5, evals=evals)
    ours = pinv_eigvalrule(M, drop_smallest=1)
    ref = jnp.linalg.pinv(M)
    assert bool(jnp.allclose(ours, ref, atol=1e-9))


def test_jit_and_vmap_safe_with_static_int():
    """drop_smallest as a Python int survives jit + vmap (R2/R24)."""
    evals = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    M, _, _ = _spd(5, 5, evals=evals)
    jp = jax.jit(lambda m: pinv_eigvalrule(m, drop_smallest=1))
    assert bool(jnp.all(jnp.isfinite(jp(M))))
    vp = jax.vmap(lambda m: pinv_eigvalrule(m, drop_smallest=1))
    out = vp(jnp.stack([M, M]))
    assert out.shape == (2, 5, 5)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_asymmetric_input_symmetrised():
    """A tiny rounding asymmetry is neutralised before eigh (R23)."""
    M, _, _ = _spd(6, 5, evals=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0]))
    M_asym = M.at[0, 1].add(1e-13)
    out = pinv_eigvalrule(M_asym, drop_smallest=1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_rejects_traced_drop_smallest():
    M, _, _ = _spd(7, 5)
    with pytest.raises(TypeError):
        pinv_eigvalrule(M, drop_smallest=jnp.asarray(1))


def test_rejects_negative_drop():
    M, _, _ = _spd(8, 5)
    with pytest.raises(ValueError):
        pinv_eigvalrule(M, drop_smallest=-1)
