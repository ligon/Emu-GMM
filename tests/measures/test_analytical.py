"""Tests for emu_gmm.measures.analytical."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from emu_gmm._internal import axes as axes_mod
from emu_gmm.manifolds import Euclidean, ManifoldLeaf, PSDFixedRank
from emu_gmm.measures.analytical import AnalyticalMeasure
from emu_gmm.types import Measure


@jdc.pytree_dataclass
class _LinearParams:
    a: float
    b: float


def _constant_expectation(model, theta):
    """E[psi] = (1.0, 2.0) regardless of theta."""
    del model, theta
    return jnp.array([1.0, 2.0])


def _theta_dependent_expectation(model, theta):
    """E[psi] = (theta.a, theta.b**2)."""
    del model
    return jnp.array([theta.a, theta.b**2])


def _dummy_psi(x, theta):
    """A placeholder psi that the analytical measure ignores."""
    del x, theta
    return jnp.array([0.0, 0.0])


# ---------------------------------------------------------------------------


class TestExpectation:
    def test_satisfies_measure_protocol(self):
        meas = AnalyticalMeasure(expectation_fn=_constant_expectation)
        assert isinstance(meas, Measure)

    def test_constant_expectation(self):
        meas = AnalyticalMeasure(expectation_fn=_constant_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        m = meas.expectation(_dummy_psi, theta)
        assert m.shape == (2,)
        assert jnp.allclose(m, jnp.array([1.0, 2.0]))

    def test_theta_dependent_expectation(self):
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        m = meas.expectation(_dummy_psi, theta)
        assert m.shape == (2,)
        assert float(m[0]) == pytest.approx(0.5)
        assert float(m[1]) == pytest.approx(4.0)

    def test_handles_namedarray_return(self):
        """expectation_fn may return a haliax NamedArray; expectation strips it."""
        Moments = axes_mod.moments_axis(2)

        def labelled_expectation(model, theta):
            del model
            return ha.named(jnp.array([theta.a, theta.b**2]), (Moments,))

        meas = AnalyticalMeasure(expectation_fn=labelled_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        m = meas.expectation(_dummy_psi, theta)
        assert m.shape == (2,)
        assert not isinstance(m, ha.NamedArray)


# ---------------------------------------------------------------------------


class TestJacobian:
    def test_shape(self):
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        assert G.shape == (2, 2)  # M=2, K=2

    def test_against_analytical_via_ad(self):
        """For f(theta) = (theta.a, theta.b**2):
        d/da = (1, 0); d/db = (0, 2*b).
        """
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        # Row 0 (m_0 = a): d/da = 1, d/db = 0
        assert float(G[0, 0]) == pytest.approx(1.0)
        assert float(G[0, 1]) == pytest.approx(0.0)
        # Row 1 (m_1 = b**2): d/da = 0, d/db = 2*b = 4.0
        assert float(G[1, 0]) == pytest.approx(0.0)
        assert float(G[1, 1]) == pytest.approx(4.0)

    def test_constant_expectation_zero_jacobian(self):
        """A theta-independent expectation has zero Jacobian under AD."""
        meas = AnalyticalMeasure(expectation_fn=_constant_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        assert G.shape == (2, 2)
        assert jnp.allclose(G, jnp.zeros((2, 2)))

    def test_user_supplied_jacobian_used(self):
        """When jacobian_fn is supplied, it is called instead of AD."""
        sentinel = jnp.array([[7.0, 8.0], [9.0, 10.0]])

        def user_jacobian(model, theta):
            del model, theta
            return sentinel

        meas = AnalyticalMeasure(
            expectation_fn=_theta_dependent_expectation,
            jacobian_fn=user_jacobian,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        # AD would give [[1, 0], [0, 4]]; sentinel is distinct, so this
        # confirms jacobian_fn took precedence.
        assert jnp.allclose(G, sentinel)

    def test_user_supplied_jacobian_handles_namedarray(self):
        """A NamedArray returned by jacobian_fn is stripped to plain."""
        Moments = axes_mod.moments_axis(2)
        Params = axes_mod.params_axis(2)
        sentinel = jnp.array([[7.0, 8.0], [9.0, 10.0]])

        def user_jacobian(model, theta):
            del model, theta
            return ha.named(sentinel, (Moments, Params))

        meas = AnalyticalMeasure(
            expectation_fn=_theta_dependent_expectation,
            jacobian_fn=user_jacobian,
        )
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        assert not isinstance(G, ha.NamedArray)
        assert jnp.allclose(G, sentinel)


# ---------------------------------------------------------------------------
# Manifold parameter trees (#41 parity with empirical/synthetic)
# ---------------------------------------------------------------------------

_N_SIDE = 3
_K_RANK = 2
_M_MANIFOLD = _N_SIDE * (_N_SIDE + 1) // 2 + 1  # triu(Y Y') ++ phi = 7
_K_AMBIENT = _N_SIDE * _K_RANK + 1  # vec Y + phi = 7


@jdc.pytree_dataclass
class _ManifoldParams:
    """``PSDFixedRank(3, 2)`` ``Y`` leaf + ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_manifold_theta() -> _ManifoldParams:
    Y = jnp.arange(1.0, 1.0 + _N_SIDE * _K_RANK).reshape(_N_SIDE, _K_RANK) / 10.0
    return _ManifoldParams(
        Y=ManifoldLeaf(Y, PSDFixedRank(_N_SIDE, _K_RANK)),
        phi=ManifoldLeaf(jnp.array([0.7]), Euclidean(1)),
    )


def _manifold_expectation(model, theta):
    """E[psi] = (triu(Y Y') ++ phi): the #41 gauge-fixture moment map."""
    del model
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[jnp.triu_indices(_N_SIDE)]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))])


class TestJacobianManifoldTree:
    """The AD fallback handles manifold trees via ``flatten_params_for_ad``.

    Pre-fix, :meth:`AnalyticalMeasure.jacobian` routed through the v1
    scalar-only ``flatten_params`` and died with "all parameter leaves
    must be 0-d scalars" on any :class:`ManifoldLeaf` tree.
    ``EmpiricalMeasure`` and ``SyntheticMeasure`` were migrated to
    ``flatten_params_for_ad`` in #41; analytical was missed.
    """

    def test_manifold_tree_jacobian_computes(self):
        """A manifold tree yields a finite ambient (M, K) Jacobian."""
        meas = AnalyticalMeasure(expectation_fn=_manifold_expectation)
        theta = _make_manifold_theta()
        G = meas.jacobian(_dummy_psi, theta)
        assert G.shape == (_M_MANIFOLD, _K_AMBIENT)
        assert bool(jnp.all(jnp.isfinite(G)))

    def test_manifold_tree_jacobian_phi_block(self):
        """The phi moment depends on exactly one ambient coordinate.

        Without pinning the flatten ordering: the last moment (phi) has
        derivative 1 w.r.t. exactly one flat coordinate and 0 elsewhere,
        and no Y-moment depends on that coordinate.
        """
        meas = AnalyticalMeasure(expectation_fn=_manifold_expectation)
        theta = _make_manifold_theta()
        G = meas.jacobian(_dummy_psi, theta)
        phi_row = G[-1]
        assert float(jnp.sum(jnp.abs(phi_row))) == pytest.approx(1.0)
        j = int(jnp.argmax(jnp.abs(phi_row)))
        assert float(phi_row[j]) == pytest.approx(1.0)
        assert jnp.allclose(G[:-1, j], 0.0)

    def test_scalar_tree_behavior_unchanged(self):
        """All-scalar trees keep the v1 flatten verbatim (K = n_leaves)."""
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)
        G = meas.jacobian(_dummy_psi, theta)
        assert G.shape == (2, 2)
        assert jnp.allclose(G, jnp.array([[1.0, 0.0], [0.0, 4.0]]))


# ---------------------------------------------------------------------------


class TestJitCompatibility:
    def test_expectation_jits(self):
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.expectation(_dummy_psi, t)

        eager = meas.expectation(_dummy_psi, theta)
        jit_result = compute(meas, theta)
        assert jnp.allclose(eager, jit_result)

    def test_jacobian_jits(self):
        meas = AnalyticalMeasure(expectation_fn=_theta_dependent_expectation)
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.jacobian(_dummy_psi, t)

        G_eager = meas.jacobian(_dummy_psi, theta)
        G_jit = compute(meas, theta)
        assert jnp.allclose(G_eager, G_jit)

    def test_jacobian_jits_with_user_jacobian_fn(self):
        sentinel = jnp.array([[7.0, 8.0], [9.0, 10.0]])

        def user_jacobian(model, theta):
            del model, theta
            return sentinel

        meas = AnalyticalMeasure(
            expectation_fn=_theta_dependent_expectation,
            jacobian_fn=user_jacobian,
        )
        theta = _LinearParams(a=0.5, b=2.0)

        @jax.jit
        def compute(m, t):
            return m.jacobian(_dummy_psi, t)

        G_jit = compute(meas, theta)
        assert jnp.allclose(G_jit, sentinel)


# ---------------------------------------------------------------------------


class TestPyTreeBehaviour:
    def test_is_pytree(self):
        meas = AnalyticalMeasure(expectation_fn=_constant_expectation)
        leaves, _ = jax.tree_util.tree_flatten(meas)
        # Both expectation_fn and jacobian_fn are static; no traced leaves.
        assert len(leaves) == 0
