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


class TestDiagonalVCase:
    """Regression tests for the diagonal-V dispatch.

    Before the fix, ``V_star = V + tau * diag(V)`` reduced to
    ``(1 + tau) V`` on diagonal ``V``, leaving ``kappa(V_star) =
    kappa(V)`` regardless of ``tau``. The bisection would saturate at
    ``_TAU_MAX`` and the regulariser silently returned a useless
    answer. Diagonal ``V`` arises naturally from ``IIDCovariance`` and
    ``ClusteredCovariance`` on uncorrelated moments, and from
    ``SyntheticCovariance`` on independent simulation draws --- so the
    bug is on a realistic code path, not a corner case.

    The fix dispatches to ``V + tau * (tr(V)/M) * I`` (an additive
    eigenvalue shift) when ``V`` is detected as (near-)diagonal. See
    ``docs/reviews/v1x-jax-ad-review.org`` (HIGH finding #2) and
    ``src/emu_gmm/regularization.py`` for the rationale.
    """

    def test_ill_conditioned_diagonal_V_brings_kappa_under_target(self):
        """Diagonal V with kappa = 1e8: regulariser must produce
        ``tau > 0`` AND drive ``kappa(V_star) <= kappa_target``.

        Before the fix this returned ``tau = _TAU_MAX = 1000`` with
        ``kappa(V_star) = 1e8`` --- regularisation completely failed
        with no error raised.
        """
        # Construct a diagonal V spanning kappa = 1e8.
        eigvals = jnp.array([1.0, 1.0e-4, 1.0e-8])
        V = _diag_V(eigvals)
        # Sanity check: input really is ill-conditioned and exactly
        # diagonal.
        assert float(jnp.linalg.cond(V)) == pytest.approx(1.0e8, rel=1e-6)
        assert jnp.allclose(V - jnp.diag(jnp.diag(V)), 0.0)

        kappa_target = 1.0e4
        reg = DiagonalTikhonov(kappa_target=kappa_target)
        V_star, tau = reg.apply(V)

        # tau must be strictly positive (we did regularise) and
        # well below the saturation cap.
        assert float(tau) > 0.0
        assert float(tau) < 999.0

        # And the realised condition number must actually meet the
        # target (accept small bisection-precision overshoot).
        kappa_star = float(jnp.linalg.cond(V_star))
        assert kappa_star <= kappa_target * 1.01

    def test_extremely_ill_conditioned_diagonal_V(self):
        """Diagonal V with kappa = 1e12: still works. (Probe in
        v1x-jax-ad-review reported kappa=1e15 saturating at TAU_MAX
        before the fix.)"""
        eigvals = jnp.array([1.0, 1.0e-6, 1.0e-12])
        V = _diag_V(eigvals)
        reg = DiagonalTikhonov(kappa_target=1.0e6)
        V_star, tau = reg.apply(V)
        assert float(tau) > 0.0
        kappa_star = float(jnp.linalg.cond(V_star))
        assert kappa_star <= 1.0e6 * 1.01

    def test_well_conditioned_diagonal_V_is_no_op(self):
        """Diagonal V already under target: short-circuit to tau=0."""
        V = _diag_V(jnp.array([1.0, 2.0, 4.0]))
        reg = DiagonalTikhonov(kappa_target=1.0e6)
        V_star, tau = reg.apply(V)
        assert float(tau) == pytest.approx(0.0, abs=1e-10)
        assert jnp.allclose(V_star, V)

    def test_diagonal_branch_uses_additive_identity_shift(self):
        """Verify the algebraic form of the diagonal-branch ridge:
        ``V_star = V + tau * (tr(V)/M) * I``.

        This is the formula that actually shifts the eigenvalues
        additively and makes regularisation effective on diagonal V.
        """
        eigvals = jnp.array([1.0, 1.0e-4, 1.0e-8])
        V = _diag_V(eigvals)
        reg = DiagonalTikhonov(kappa_target=1.0e4)
        V_star, tau = reg.apply(V)
        M = V.shape[0]
        scale = float(jnp.trace(V) / M)
        expected = V + float(tau) * scale * jnp.eye(M, dtype=V.dtype)
        assert jnp.allclose(V_star, expected, atol=1e-10)

    def test_apply_fixed_tau_consistent_with_apply_on_diagonal_V(self):
        """Anchor-once-then-freeze path: ``apply_fixed_tau(V, tau)``
        must reproduce the ``V_star`` from ``apply(V)`` on a diagonal
        ``V``. Otherwise the anchored optimisation surface would not
        match the surface the bisection optimised against.
        """
        eigvals = jnp.array([1.0, 1.0e-4, 1.0e-8])
        V = _diag_V(eigvals)
        reg = DiagonalTikhonov(kappa_target=1.0e4)
        V_star, tau = reg.apply(V)
        V_star_fixed = reg.apply_fixed_tau(V, tau)
        assert jnp.allclose(V_star_fixed, V_star, atol=1e-12)

    def test_non_diagonal_path_uses_canonical_diag_formula(self):
        """Regression guard: non-diagonal ``V`` must still go through
        the canonical ``V + tau * diag(V)`` branch (per-moment
        scale-equivariant; CLAUDE.md commitment 3). Verified by checking
        the diagonal-entry identity ``V_star[i,i] = V[i,i] * (1+tau)``,
        which only holds for the canonical formula.
        """
        V = _ill_conditioned_V(target_kappa=1.0e6, dim=3)
        reg = DiagonalTikhonov(kappa_target=1.0e3)
        V_star, tau = reg.apply(V)
        for i in range(3):
            assert float(V_star[i, i]) == pytest.approx(
                float(V[i, i]) * (1.0 + float(tau)), rel=1e-5
            )

    def test_diagonal_dispatch_jits(self):
        """The diagonal-V dispatch must trace under jit."""
        V = _diag_V(jnp.array([1.0, 1.0e-4, 1.0e-8]))
        reg = DiagonalTikhonov(kappa_target=1.0e4)

        @jax.jit
        def compute(r, vv):
            return r.apply(vv)

        V_star_eager, tau_eager = reg.apply(V)
        V_star_jit, tau_jit = compute(reg, V)
        assert jnp.allclose(V_star_eager, V_star_jit, atol=1e-10)
        assert float(tau_eager) == pytest.approx(float(tau_jit), rel=1e-6)

    def test_apply_fixed_tau_jits_on_diagonal_V(self):
        V = _diag_V(jnp.array([1.0, 1.0e-4, 1.0e-8]))
        reg = DiagonalTikhonov(kappa_target=1.0e4)
        tau = jnp.asarray(0.1)

        @jax.jit
        def compute(r, vv, tt):
            return r.apply_fixed_tau(vv, tt)

        eager = reg.apply_fixed_tau(V, tau)
        jitted = compute(reg, V, tau)
        assert jnp.allclose(eager, jitted, atol=1e-12)


def _indefinite_well_conditioned_V(
    neg_eig: float = -2.715e-7,
    pos_eigs=(1.0, 5.0, 30.0, 123.0),
    seed: int = 0,
) -> jnp.ndarray:
    """Symmetric V with ONE tiny negative eigenvalue, SVD-cond < a loose target.

    Mirrors the #111 reproduction (Seasonality design-aware Euler spec at
    sigma=1.2): ``eig[min,max] = [-2.715e-07, 123]`` so ``cond = 123/2.715e-7
    ~ 4.5e8`` is below a loose ``kappa_target`` of 1e10, yet V is indefinite.
    Non-axis-aligned (random Q) so ``V + tau diag(V)`` actually moves the
    spectrum.
    """
    rng = np.random.default_rng(seed)
    w = np.array([neg_eig, *pos_eigs])
    Q, _ = np.linalg.qr(rng.standard_normal((w.size, w.size)))
    V = (Q * w) @ Q.T
    return jnp.asarray(0.5 * (V + V.T))


class TestDefinitenessFloor:
    """Regression for #111: an indefinite-but-well-conditioned V must be
    repaired to PD, not passed through with ~zero ridge (which silently NaNs
    the downstream Cholesky)."""

    def test_raw_input_is_indefinite_but_well_conditioned(self):
        # Sanity: the fixture really is the #111 trap -- indefinite, yet its
        # SVD condition number sits below the loose target.
        V = _indefinite_well_conditioned_V()
        ev = np.linalg.eigvalsh(np.asarray(V))
        assert ev[0] < 0.0  # indefinite
        svd_cond = float(jnp.linalg.cond(V))
        assert svd_cond < 1.0e10  # would have read as "well-conditioned"

    def test_apply_makes_v_star_pd(self):
        V = _indefinite_well_conditioned_V()
        reg = DiagonalTikhonov(kappa_target=1.0e10)
        V_star, tau = reg.apply(V)
        ev = np.linalg.eigvalsh(np.asarray(V_star))
        assert ev[0] > 0.0  # strictly PD now
        assert float(tau) > 0.0  # a repairing ridge was actually applied

    def test_v_star_cholesky_is_finite(self):
        # The concrete downstream symptom: Cholesky of the regularised V must
        # be finite (not NaN), so k_statistic / bootstrap whitening is valid.
        V = _indefinite_well_conditioned_V()
        reg = DiagonalTikhonov(kappa_target=1.0e10)
        V_star, _ = reg.apply(V)
        L = jnp.linalg.cholesky(V_star)
        assert bool(jnp.all(jnp.isfinite(L)))

    def test_v_star_respects_condition_target(self):
        V = _indefinite_well_conditioned_V()
        reg = DiagonalTikhonov(kappa_target=1.0e10)
        V_star, _ = reg.apply(V)
        assert float(jnp.linalg.cond(V_star)) <= 1.0e10 * (1.0 + 1e-6)

    def test_tighter_target_also_pd(self):
        # A much tighter target should also yield PD (and a larger ridge).
        V = _indefinite_well_conditioned_V()
        V_loose, tau_loose = DiagonalTikhonov(kappa_target=1.0e10).apply(V)
        V_tight, tau_tight = DiagonalTikhonov(kappa_target=1.0e4).apply(V)
        assert np.linalg.eigvalsh(np.asarray(V_tight))[0] > 0.0
        assert float(tau_tight) >= float(tau_loose)

    def test_well_conditioned_psd_still_no_op(self):
        # Non-regression: a PD, well-conditioned V is still returned unchanged
        # at tau = 0 (the signed-spectrum predicate agrees with the old SVD
        # predicate on PSD inputs).
        V = _ill_conditioned_V(target_kappa=1.0e3, dim=4)
        reg = DiagonalTikhonov(kappa_target=1.0e6)
        V_star, tau = reg.apply(V)
        assert float(tau) == 0.0
        assert jnp.allclose(V_star, V)

    def test_apply_jits_on_indefinite_V(self):
        V = _indefinite_well_conditioned_V()
        reg = DiagonalTikhonov(kappa_target=1.0e10)

        @jax.jit
        def compute(r, vv):
            return r.apply(vv)

        Vs_e, tau_e = reg.apply(V)
        Vs_j, tau_j = compute(reg, V)
        assert jnp.allclose(Vs_e, Vs_j, atol=1e-10)
        assert float(tau_e) == pytest.approx(float(tau_j), rel=1e-6)


class TestZeroDiagonalSaturation:
    """Honest-contract pin: a ``V`` the diagonal-ridge family CANNOT repair.

    An exactly-zero diagonal entry (the empirical ``V`` of an all-zero
    mask column, i.e. a zero-support moment) is invariant under the
    canonical ridge: ``V + tau * diag(V)`` leaves ``(1 + tau) * 0 == 0``
    at every ``tau``. No feasible ridge exists, so ``apply()`` saturates
    the bisection at ``_TAU_MAX`` and returns a still-singular ``V_star``
    --- BY (documented) DESIGN, since the routine must stay
    trace-compatible and cannot raise. The estimator surfaces the event
    loudly via the ``v_star_indefinite`` diagnostic (see
    ``tests/test_diagnostics.py::TestVStarIndefiniteDiagnostic``); this
    class pins the regulariser's honest contract rather than a repair it
    cannot deliver.
    """

    @staticmethod
    def _zero_diag_V() -> jnp.ndarray:
        # Moment 2 has zero support: exact-zero row/column, hence an
        # exact-zero diagonal entry. The (0, 1) off-diagonal is nonzero
        # so the diagonal-branch dispatch (additive identity shift, which
        # COULD repair this) does not trigger -- the canonical
        # multiplicative branch is exercised.
        return jnp.array(
            [
                [1.0, 0.3, 0.0],
                [0.3, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

    def test_tau_saturates_and_v_star_not_pd(self):
        from emu_gmm.regularization import _TAU_MAX

        V = self._zero_diag_V()
        V_star, tau = DiagonalTikhonov().apply(V)
        # The bisection saturated at the cap: no feasible tau exists.
        assert float(tau) == pytest.approx(_TAU_MAX)
        # And the returned V_star is NOT positive-definite.
        lam_min = float(np.linalg.eigvalsh(np.asarray(V_star))[0])
        assert lam_min <= 0.0

    def test_downstream_cholesky_nans(self):
        # The concrete downstream hazard this contract implies: the
        # Cholesky of the returned V_star is NaN (jax.scipy returns NaN
        # rather than raising on non-PD input), which silently NaNs the
        # criterion / J / SEs absent the estimator's v_star_indefinite
        # diagnostic.
        V = self._zero_diag_V()
        V_star, _tau = DiagonalTikhonov().apply(V)
        L = jnp.linalg.cholesky(V_star)
        assert bool(jnp.isnan(L).any())
