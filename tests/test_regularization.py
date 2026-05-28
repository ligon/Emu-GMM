"""Tests for emu_gmm.regularization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from emu_gmm import types as t
from emu_gmm.regularization import DiagonalTikhonov


def _diag_V(eigvals: jnp.ndarray) -> jnp.ndarray:
    """Build a diagonal V from a vector of eigenvalues."""
    return jnp.diag(jnp.asarray(eigvals))


def _ill_conditioned_V(target_kappa: float = 1.0e8, dim: int = 3) -> jnp.ndarray:
    """Build a symmetric PD matrix with condition number ``target_kappa``.

    The diagonal-Tikhonov regulariser can only improve conditioning when
    the eigenstructure is not axis-aligned (otherwise V + tau diag(V) =
    (1+tau) V, which has the same condition number). We therefore
    construct V = Q diag(eigvals) Q' with a non-trivial random
    orthogonal Q.
    """
    rng = np.random.default_rng(seed=1)
    # Eigenvalues spanning target_kappa: from 1 down to 1/target_kappa.
    eigvals = np.geomspace(1.0, 1.0 / target_kappa, num=dim)
    # Random orthogonal via QR of a Gaussian matrix.
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    V = (Q * eigvals) @ Q.T
    # Symmetrise to wash out any QR floating-point asymmetry.
    V = 0.5 * (V + V.T)
    return jnp.asarray(V)


# ---------------------------------------------------------------------------


class TestDiagonalTikhonov:
    def test_satisfies_protocol(self):
        assert isinstance(DiagonalTikhonov(), t.RegularizationStrategy)

    def test_defaults(self):
        reg = DiagonalTikhonov()
        assert reg.kappa_target == pytest.approx(1.0e6)
        assert reg.tau_threshold == pytest.approx(1.0e-2)

    def test_well_conditioned_input_no_op(self):
        """If kappa(V) is already below target, return ``(V, 0)``."""
        # kappa = 4 on this diagonal: well below 1e6.
        V = _diag_V(jnp.array([1.0, 2.0, 4.0]))
        reg = DiagonalTikhonov(kappa_target=1.0e6)
        V_star, tau = reg.apply(V)
        assert float(tau) == pytest.approx(0.0, abs=1e-10)
        assert jnp.allclose(V_star, V)

    def test_ill_conditioned_input_brings_kappa_under_target(self):
        """A non-diagonal V with kappa ~ 1e6 -> regularised V has
        kappa <= kappa_target.

        Note: a purely diagonal V cannot have its condition number
        reduced by ``V + tau * diag(V)``, since that reduces to
        ``(1 + tau) V``. The regulariser only acts non-trivially when
        the eigenstructure is not axis-aligned, which is the realistic
        case in the pairwise-overlap empirical covariance.
        """
        # Kappa of input ~ 1e6 (within reach of float32 to measure);
        # reduce to kappa_target = 1e3.
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        target = 1.0e3
        reg = DiagonalTikhonov(kappa_target=target)
        V_star, tau = reg.apply(V)
        # The realised condition number should respect the target. The
        # bisection's finite precision means we accept a small overshoot.
        kappa_star = float(jnp.linalg.cond(V_star))
        assert kappa_star <= target * 1.01
        # And tau must be strictly positive (we did need to regularise).
        assert float(tau) > 0.0

    def test_returns_v_star_eq_v_plus_tau_diag(self):
        """V_star = V + tau * diag(V) holds exactly."""
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e3)
        V_star, tau = reg.apply(V)
        expected = V + float(tau) * jnp.diag(jnp.diag(V))
        assert jnp.allclose(V_star, expected, atol=1e-6)

    def test_scale_equivariance_under_per_moment_rescaling(self):
        """Per-moment rescaling commutes with the diagonal-Tikhonov ridge.

        Concretely: if ``V`` is rescaled by a diagonal ``D``
        (``V' = D V D``, which corresponds to scaling moment ``j`` by
        ``D_{jj}``), then ``D V^star D`` equals ``V'`` post-regularisation
        at the same ``tau``. This is the design's "scale-equivariance"
        property: tau lives in units of ``diag(V)``, so the additive
        ridge is in the same units as the data.

        Note that the *chosen* tau may differ between V and V' because
        their condition numbers differ; what is equivariant is the
        identity ``D V^star D == (D V D) + tau' diag(D V D)`` for
        whichever tau' apply selects on ``D V D``. We therefore test
        the algebraic identity directly: at a fixed tau, the
        regulariser commutes with diagonal rescaling.
        """
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e3)
        V_star, tau = reg.apply(V)

        # Per-moment rescaling: multiply moment 1 by alpha.
        alpha = 10.0
        D = jnp.diag(jnp.array([1.0, alpha, 1.0]))
        V_scaled = D @ V @ D

        # At the *same* tau, applying the ridge to V_scaled and
        # rescaling V_star both yield the same matrix.
        manual_V_star_scaled = V_scaled + float(tau) * jnp.diag(jnp.diag(V_scaled))
        rescaled_V_star = D @ V_star @ D
        assert jnp.allclose(manual_V_star_scaled, rescaled_V_star, atol=1e-6)

    def test_scale_equivariance_uniform_rescaling(self):
        """Uniform rescaling ``V -> alpha V`` leaves ``tau`` invariant
        (since ``kappa`` is invariant) and rescales ``V_star`` by the
        same alpha.
        """
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e3)
        V_star, tau = reg.apply(V)

        alpha = 42.0
        V_scaled = alpha * V
        V_star_scaled, tau_scaled = reg.apply(V_scaled)

        # Uniform scaling preserves the condition number, so tau matches.
        assert float(tau_scaled) == pytest.approx(float(tau), rel=1e-4)
        # And V_star_scaled == alpha * V_star.
        assert jnp.allclose(V_star_scaled, alpha * V_star, rtol=1e-4)

    def test_tau_in_units_of_diag_V(self):
        """The diagonal entries pick up exactly a multiplicative factor
        of ``1 + tau``: that is, ``V_star[i,i] = V[i,i] * (1 + tau)``.
        """
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e4)
        V_star, tau = reg.apply(V)
        for i in range(3):
            assert float(V_star[i, i]) == pytest.approx(
                float(V[i, i]) * (1.0 + float(tau)), rel=1e-5
            )

    def test_is_pytree_with_static_fields(self):
        """``kappa_target`` and ``tau_threshold`` are static; the
        instance has no traced leaves.
        """
        reg = DiagonalTikhonov(kappa_target=1.0e5, tau_threshold=0.02)
        leaves, treedef = jax.tree_util.tree_flatten(reg)
        assert leaves == []
        # Two instances with different static fields differ as PyTreeDefs
        # (so jit will trace each separately).
        reg2 = DiagonalTikhonov(kappa_target=1.0e7, tau_threshold=0.02)
        _, treedef2 = jax.tree_util.tree_flatten(reg2)
        assert treedef != treedef2

    def test_jits(self):
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e3)

        @jax.jit
        def compute(r, vv):
            return r.apply(vv)

        V_star_eager, tau_eager = reg.apply(V)
        V_star_jit, tau_jit = compute(reg, V)
        assert jnp.allclose(V_star_eager, V_star_jit, atol=1e-6)
        assert float(tau_eager) == pytest.approx(float(tau_jit), rel=1e-6)
