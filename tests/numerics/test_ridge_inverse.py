"""Tests for emu_gmm.numerics.ridge_inverse."""

from __future__ import annotations

import haliax as ha
import jax.numpy as jnp
import numpy as np
import pytest
from emu_gmm import ridge_inverse
from emu_gmm.numerics import ridge_inverse as ridge_inverse_via_submodule


def _ill_conditioned_M(target_kappa: float = 1.0e6, dim: int = 3) -> jnp.ndarray:
    """Symmetric PD matrix with condition number ``target_kappa`` and a
    non-axis-aligned eigenbasis (otherwise the diagonal-Tikhonov ridge
    is a no-op on the condition number).

    Seed matches the regularisation test suite so that any drift in the
    underlying bisection shows up in both places at once.
    """
    rng = np.random.default_rng(seed=1)
    eigvals = np.geomspace(1.0, 1.0 / target_kappa, num=dim)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    M = (Q * eigvals) @ Q.T
    M = 0.5 * (M + M.T)
    return jnp.asarray(M)


# ---------------------------------------------------------------------------
# Re-export sanity
# ---------------------------------------------------------------------------


def test_top_level_reexport_matches_submodule():
    assert ridge_inverse is ridge_inverse_via_submodule


# ---------------------------------------------------------------------------
# (a) Well-conditioned input
# ---------------------------------------------------------------------------


class TestWellConditioned:
    def test_tau_is_zero_and_inverse_is_unridged(self):
        """If kappa(M) <= target_condition already, tau == 0 and the
        returned inverse equals M^{-1} exactly (no ridge applied)."""
        # Well-conditioned diagonal: kappa = 4.
        M = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        M_inv_named, info = ridge_inverse(M, target_condition=1.0e6)

        assert info["tau"] == pytest.approx(0.0, abs=1e-12)
        assert info["binding"] is False
        # kappa_before is 4 (diagonal); kappa_after is the same (tau=0).
        assert info["kappa_before"] == pytest.approx(4.0, rel=1e-6)
        assert info["kappa_after"] == pytest.approx(4.0, rel=1e-6)

        expected_inv = jnp.diag(jnp.array([1.0, 0.5, 0.25]))
        assert jnp.allclose(M_inv_named.array, expected_inv, atol=1e-12)

    def test_returns_named_array_with_positional_axes_for_plain_input(self):
        M = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        M_inv_named, _ = ridge_inverse(M, target_condition=1.0e6)
        assert isinstance(M_inv_named, ha.NamedArray)
        names = tuple(a.name for a in M_inv_named.axes)
        sizes = tuple(a.size for a in M_inv_named.axes)
        assert names == ("dim", "dim_dual")
        assert sizes == (3, 3)

    def test_preserves_input_namedarray_axes(self):
        """A NamedArray input round-trips its axes onto the output."""
        M = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        ax_row = ha.Axis(name="moments", size=3)
        ax_col = ha.Axis(name="moments_dual", size=3)
        M_named = ha.named(M, (ax_row, ax_col))
        M_inv_named, info = ridge_inverse(M_named, target_condition=1.0e6)
        assert isinstance(M_inv_named, ha.NamedArray)
        names = tuple(a.name for a in M_inv_named.axes)
        assert names == ("moments", "moments_dual")
        assert info["tau"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (b) Ill-conditioned input
# ---------------------------------------------------------------------------


class TestIllConditioned:
    def test_tau_positive_and_kappa_hits_target(self):
        """An ill-conditioned input forces tau > 0 and the post-ridge
        condition number respects the target (within bisection
        precision)."""
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=3)
        target = 1.0e3
        _, info = ridge_inverse(M, target_condition=target)
        assert info["tau"] > 0.0
        # Bisection's finite resolution permits a small overshoot.
        assert info["kappa_after"] <= target * 1.01
        # kappa_before should reflect the input's actual conditioning.
        assert info["kappa_before"] > target

    def test_binding_flag_set_when_tau_large(self):
        """For an aggressive target the chosen tau is well above the
        binding threshold; the flag should be True."""
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=3)
        _, info = ridge_inverse(M, target_condition=10.0)
        assert info["tau"] > 1.0e-2
        assert info["binding"] is True

    def test_inverse_satisfies_M_star_M_inv_eq_I(self):
        """The returned inverse is the inverse of the ridged matrix,
        not of the original M: (M + tau diag(M)) @ M_inv == I."""
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=3)
        M_inv_named, info = ridge_inverse(M, target_condition=1.0e3)
        M_star = M + info["tau"] * jnp.diag(jnp.diag(M))
        product = M_star @ M_inv_named.array
        # Reasonable tolerance: condition number ~1e3 puts roundoff at
        # ~1e-13 in float64, well within atol=1e-8.
        assert jnp.allclose(product, jnp.eye(3), atol=1e-8)


# ---------------------------------------------------------------------------
# (c) Symmetry preserved
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_inverse_is_symmetric_well_conditioned(self):
        M = _ill_conditioned_M(target_kappa=1.0e4, dim=4)
        M_inv_named, _ = ridge_inverse(M, target_condition=1.0e6)
        inv = M_inv_named.array
        assert jnp.allclose(inv, inv.T, atol=1e-12)

    def test_inverse_is_symmetric_ill_conditioned(self):
        # Moderately ill-conditioned: bisection must be feasible at the
        # chosen target so the ridged matrix actually gets formed.
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=4)
        M_inv_named, _ = ridge_inverse(M, target_condition=1.0e3)
        inv = M_inv_named.array
        assert jnp.allclose(inv, inv.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_target_condition_must_exceed_one(self):
        M = jnp.eye(3)
        with pytest.raises(ValueError, match="target_condition"):
            ridge_inverse(M, target_condition=1.0)
        with pytest.raises(ValueError, match="target_condition"):
            ridge_inverse(M, target_condition=0.5)

    def test_non_square_input_raises(self):
        M = jnp.zeros((3, 4))
        with pytest.raises(ValueError, match="square"):
            ridge_inverse(M, target_condition=1.0e6)

    def test_one_d_input_raises(self):
        with pytest.raises(ValueError, match="square"):
            ridge_inverse(jnp.array([1.0, 2.0, 3.0]), target_condition=1.0e6)


# ---------------------------------------------------------------------------
# Info dict surface
# ---------------------------------------------------------------------------


class TestInfoDict:
    def test_keys_present(self):
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=3)
        _, info = ridge_inverse(M, target_condition=1.0e3)
        assert set(info.keys()) == {"tau", "kappa_before", "kappa_after", "binding"}

    def test_types_are_python_scalars(self):
        """The info dict carries plain Python floats and a bool so it
        can be JSON-serialised by downstream callers without surprises."""
        M = _ill_conditioned_M(target_kappa=1.0e6, dim=3)
        _, info = ridge_inverse(M, target_condition=1.0e3)
        assert isinstance(info["tau"], float)
        assert isinstance(info["kappa_before"], float)
        assert isinstance(info["kappa_after"], float)
        assert isinstance(info["binding"], bool)
