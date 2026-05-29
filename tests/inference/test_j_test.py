"""Tests for the zero-parameter J-test helper.

The helper evaluates the standard over-identifying-restrictions
statistic at a user-supplied ``theta_null`` without going through
parameter estimation: ``J = m' V^{-1} m ~ chi^2_M`` (dof = M, not
M - K, because no parameters are estimated).

Three regimes are covered:

1. Truth-at-null --- synthetic data drawn from the DGP, ``theta_null``
   set to the truth: ``J`` should be small (no MC bias at this scale).
2. Misspecified ``theta_null`` --- same data, ``theta_null`` far from
   the truth: ``J`` should be large (the moment conditions are
   violated).
3. Shape and label structure --- :class:`JTestResult` carries the right
   types, axes, and dof.
"""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import pytest
import scipy.stats
from emu_gmm import DiagonalTikhonov, JTestResult, SyntheticCovariance, j_test
from emu_gmm.examples.euler import (
    BETA_TRUE,
    GAMMA_TRUE,
    EulerParams,
    euler_residual,
    euler_sampler_factory,
)
from emu_gmm.measures import SyntheticMeasure

N_SIM = 5000


def _measure(seed: int = 0) -> SyntheticMeasure:
    sampler = euler_sampler_factory(N_SIM)
    return SyntheticMeasure(
        key=jax.random.PRNGKey(seed),
        n_sim=N_SIM,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# (a) Truth at null: J should be small.
# ---------------------------------------------------------------------------


class TestJTestAtTruth:
    """At ``theta_null = truth``, the moment conditions hold by construction."""

    def _run(self) -> JTestResult:
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )

    def test_returns_j_test_result(self):
        r = self._run()
        assert isinstance(r, JTestResult)

    def test_J_dof_equals_M(self):
        """Zero estimated parameters → dof equals moment count (M=3)."""
        r = self._run()
        assert r.J_dof == 3

    def test_J_stat_finite_and_small(self):
        """At truth the only source of nonzero J is Monte Carlo noise.

        Under H0, J is asymptotically chi^2_3 with mean 3. With
        N_SIM=5000 we expect J ~= 3 +/- a few; allow 30 to absorb
        sampling variation across CRN-frozen keys.
        """
        r = self._run()
        assert jnp.isfinite(r.J_stat)
        assert r.J_stat < 30.0

    def test_J_pvalue_in_unit_interval(self):
        r = self._run()
        assert jnp.isfinite(r.J_pvalue)
        assert 0.0 <= r.J_pvalue <= 1.0

    def test_pvalue_matches_chi2_sf(self):
        """The reported p-value is exactly the chi^2_dof survival
        function evaluated at J_stat."""
        r = self._run()
        expected = float(scipy.stats.chi2.sf(float(r.J_stat), r.J_dof))
        assert float(r.J_pvalue) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# (b) Misspecified null: J should be large.
# ---------------------------------------------------------------------------


class TestJTestMisspecified:
    """At a ``theta_null`` far from the truth, J should reject."""

    def _run(self) -> JTestResult:
        # gamma = 5.0 is well away from GAMMA_TRUE=2.0; with NU spread
        # (0.5, 1.0, 1.5) the cross-asset slope makes the moment vector
        # depart non-trivially from zero.
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=0.80, gamma=5.0),
        )

    def test_J_stat_large(self):
        """J is well into the chi^2_3 tail."""
        r = self._run()
        # 99.9% quantile of chi^2_3 is ~16.27. We use a comfortable
        # margin: the true population J at this misspecification is
        # huge (the moment is O(1) per coordinate, V scales as
        # 1/n_sim, so J scales as n_sim).
        assert r.J_stat > 100.0

    def test_J_pvalue_tiny(self):
        r = self._run()
        assert r.J_pvalue < 1e-6


# ---------------------------------------------------------------------------
# (c) Shape and label structure.
# ---------------------------------------------------------------------------


class TestJTestStructure:
    """``JTestResult`` fields have the expected types and axes."""

    def _run(self) -> JTestResult:
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )

    def test_V_X_is_named_array(self):
        r = self._run()
        assert isinstance(r.V_X, ha.NamedArray)

    def test_V_X_axes(self):
        r = self._run()
        assert {a.name for a in r.V_X.axes} == {"moments", "moments_dual"}

    def test_V_X_shape(self):
        r = self._run()
        assert r.V_X.array.shape == (3, 3)

    def test_V_X_symmetric(self):
        r = self._run()
        V = r.V_X.array
        assert jnp.allclose(V, V.T, atol=1e-10)

    def test_V_X_positive_definite(self):
        """Regularised V should be PD --- Cholesky should not produce NaN."""
        r = self._run()
        L = jnp.linalg.cholesky(r.V_X.array)
        assert jnp.all(jnp.isfinite(L))

    def test_scalar_fields_have_expected_types(self):
        """J_stat / J_pvalue are 0-d JAX arrays (for jit/vmap
        compatibility); J_dof is a static Python int.

        Downstream consumers that need vanilla Python scalars can call
        ``float(r.J_stat)`` and ``float(r.J_pvalue)`` at the eager
        boundary. ``J_dof`` is static (it derives from the moment count
        ``M`` which is a compile-time shape constant).
        """
        r = self._run()
        # 0-d arrays are scalar-shaped JAX arrays; both jnp.ndarray and
        # the abstract ArrayImpl satisfy hasattr "shape".
        assert hasattr(r.J_stat, "shape") and r.J_stat.shape == ()
        assert hasattr(r.J_pvalue, "shape") and r.J_pvalue.shape == ()
        assert isinstance(r.J_dof, int)


# ---------------------------------------------------------------------------
# Default regularization behaviour.
# ---------------------------------------------------------------------------


class TestRegularizationKwarg:
    """The ``regularization`` kwarg defaults to ``DiagonalTikhonov`` and
    accepts an explicit override."""

    def test_default_runs(self):
        r = j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )
        assert jnp.isfinite(r.J_stat)

    def test_explicit_regularization_accepted(self):
        r = j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
            regularization=DiagonalTikhonov(kappa_target=1e8),
        )
        assert jnp.isfinite(r.J_stat)


# ---------------------------------------------------------------------------
# (d) jit / vmap compatibility.
# ---------------------------------------------------------------------------


class TestJTestJitVmap:
    """``j_test`` composes with ``jax.jit`` and ``jax.vmap``.

    Mirrors the post-#45 contract for :func:`emu_gmm.estimate`: scalar
    statistics are returned as 0-d JAX arrays computed via
    ``jax.scipy.stats.chi2.sf`` (no eager ``float()`` cast or
    ``scipy.stats.chi2.sf`` boundary inside the helper).
    """

    def _inputs(self):
        return {
            "measure": _measure(),
            "covariance": SyntheticCovariance(),
            "model": euler_residual,
        }

    def test_jit_J_stat_finite_and_matches_eager(self):
        inputs = self._inputs()
        theta = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)

        def run(theta):
            return j_test(**inputs, theta_null=theta).J_stat

        eager = float(run(theta))
        jitted = float(jax.jit(run)(theta))
        assert jnp.isfinite(jitted)
        # jit must not change the answer.
        assert jitted == pytest.approx(eager, rel=1e-6, abs=1e-10)

    def test_jit_returns_traced_pvalue(self):
        """``J_pvalue`` is computed via ``jax.scipy.stats.chi2.sf``, so
        it traces; the result is a 0-d JAX array in [0, 1]."""
        inputs = self._inputs()
        theta = EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE)

        def run(theta):
            return j_test(**inputs, theta_null=theta).J_pvalue

        p = jax.jit(run)(theta)
        p_val = float(p)
        assert 0.0 <= p_val <= 1.0

    def test_vmap_over_theta_null_returns_batched_J_stat(self):
        inputs = self._inputs()

        def run(theta):
            return j_test(**inputs, theta_null=theta).J_stat

        batch = EulerParams(
            beta=jnp.array([BETA_TRUE, BETA_TRUE, BETA_TRUE]),
            gamma=jnp.array([GAMMA_TRUE, 1.5, 2.5]),
        )
        J = jax.vmap(run)(batch)
        assert J.shape == (3,)
        assert jnp.all(jnp.isfinite(J))

    def test_vmap_over_theta_null_batched_pvalue(self):
        inputs = self._inputs()

        def run(theta):
            return j_test(**inputs, theta_null=theta).J_pvalue

        batch = EulerParams(
            beta=jnp.array([BETA_TRUE, BETA_TRUE, BETA_TRUE]),
            gamma=jnp.array([GAMMA_TRUE, 1.5, 2.5]),
        )
        p = jax.vmap(run)(batch)
        assert p.shape == (3,)
        assert jnp.all(p >= 0.0) and jnp.all(p <= 1.0)

    def test_jit_then_vmap_composes(self):
        inputs = self._inputs()

        def run(theta):
            return j_test(**inputs, theta_null=theta).J_stat

        batch = EulerParams(
            beta=jnp.array([BETA_TRUE, BETA_TRUE, BETA_TRUE]),
            gamma=jnp.array([GAMMA_TRUE, 1.5, 2.5]),
        )
        eager = jax.vmap(run)(batch)
        jitted = jax.jit(jax.vmap(run))(batch)
        assert jnp.allclose(eager, jitted, rtol=1e-6, atol=1e-10)

    def test_full_result_returned_from_jit(self):
        """The pytree_dataclass JTestResult can be returned directly
        from a jitted function (round-trips as a pytree)."""
        inputs = self._inputs()

        def run(theta):
            return j_test(**inputs, theta_null=theta)

        r_jit = jax.jit(run)(EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE))
        assert isinstance(r_jit, JTestResult)
        assert jnp.isfinite(r_jit.J_stat)
        assert jnp.isfinite(r_jit.J_pvalue)
        assert r_jit.J_dof == 3


# ---------------------------------------------------------------------------
# (e) JTestResult is a pytree_dataclass.
# ---------------------------------------------------------------------------


class TestJTestResultPyTree:
    """:class:`JTestResult` is a ``@jdc.pytree_dataclass`` so it
    round-trips through ``jax.tree_util`` as a PyTree.

    The traced-leaf fields (``J_stat``, ``J_pvalue``, ``V_X``) appear
    as leaves; ``J_dof`` is a static field and does not appear in the
    leaves list.
    """

    def _run(self) -> JTestResult:
        return j_test(
            measure=_measure(),
            covariance=SyntheticCovariance(),
            model=euler_residual,
            theta_null=EulerParams(beta=BETA_TRUE, gamma=GAMMA_TRUE),
        )

    def test_tree_flatten_unflatten_roundtrip(self):
        r = self._run()
        leaves, treedef = jax.tree_util.tree_flatten(r)
        # Reconstruct and verify the rebuilt object preserves field
        # values (J_stat, J_pvalue, V_X array, and the static J_dof).
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, JTestResult)
        assert float(rebuilt.J_stat) == pytest.approx(float(r.J_stat))
        assert float(rebuilt.J_pvalue) == pytest.approx(float(r.J_pvalue))
        assert rebuilt.J_dof == r.J_dof
        assert jnp.allclose(rebuilt.V_X.array, r.V_X.array)

    def test_tree_leaves_are_arrays(self):
        """Every leaf in the flattened pytree is a JAX-compatible
        array. ``J_dof`` is *not* a leaf (it's a static field)."""
        r = self._run()
        leaves = jax.tree_util.tree_leaves(r)
        # All leaves must be JAX arrays (or pytree-array-like).
        for leaf in leaves:
            assert hasattr(leaf, "shape")
        # Static J_dof must not appear in leaves. Use type-and-value
        # match rather than ``in`` (which triggers array __bool__).
        assert not any(isinstance(leaf, int) and leaf == r.J_dof for leaf in leaves)

    def test_tree_map_preserves_structure(self):
        """``jax.tree_util.tree_map`` over the result preserves the
        :class:`JTestResult` type and the static ``J_dof``."""
        r = self._run()
        doubled = jax.tree_util.tree_map(lambda x: x * 2, r)
        assert isinstance(doubled, JTestResult)
        assert float(doubled.J_stat) == pytest.approx(2.0 * float(r.J_stat))
        assert doubled.J_dof == r.J_dof
