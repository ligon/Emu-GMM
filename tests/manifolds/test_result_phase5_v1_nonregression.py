r"""Phase-5 v1 bitwise non-regression + tangent_basis_names unit checks.

Phase 5 added a ``components()`` readout, a manifold-aware ``coef_table``
flatten, and positional tangent labels --- all gated on a *non-scalar*
``manifold_spec`` on the result. For a v1 / all-scalar tree (``manifold_spec``
is ``None`` or all leaves scalar) every result-path method must take the v1
branch unchanged:

* ``coef_table`` uses ``flatten_params`` and is indexed by the scalar
  field-names (NOT positional tangent labels);
* ``standard_errors`` is unchanged;
* ``to_pandas()`` Sigma_theta index is the field-names;
* ``components()`` returns the per-leaf scalar tuple in field order.

Plus direct unit tests of ``tangent_basis_names`` (the new helper).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses as jdc
from emu_gmm import types as t
from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal.labels import tangent_basis_names
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


class _StubMeasure:
    def expectation(self, psi, theta):
        return jnp.zeros(2)

    def jacobian(self, psi, theta):
        return jnp.zeros((2, 2))


class _StubCovariance:
    def covariance(self, psi, theta, measure):
        return jnp.eye(2)


class _StubWeighting:
    def whitening_residual(self, m, V, theta):
        return m


class _StubRegularization:
    def apply(self, V):
        return V, 0.0


def _make_scalar_result(manifold_spec=None) -> t.EstimationResult:
    Params = axes_mod.params_axis(2)
    ParamsDual = axes_mod.params_dual_axis(2)
    Moments = axes_mod.moments_axis(3)
    MomentsDual = axes_mod.moments_dual_axis(3)
    sigma = labels_mod.label_matrix(
        jnp.array([[0.01, 0.001], [0.001, 0.02]]), Params, ParamsDual
    )
    v_x = labels_mod.label_matrix(jnp.eye(3) * 0.1, Moments, MomentsDual)
    n_j = labels_mod.label_vector(jnp.array([100.0, 100.0, 100.0]), Moments)
    m_res = labels_mod.label_vector(jnp.array([1e-4, -2e-4, 5e-5]), Moments)
    opt_info = t.OptimizerInfo(
        steps=12, status="converged", final_objective=1.3, backend="stub"
    )
    diagnostics = t.Diagnostics(
        tau_realised=jnp.asarray(0.001),
        kappa_V=jnp.asarray(1e3),
        binding_ridge=jnp.asarray(False),
        cholesky_pivot_min=jnp.asarray(0.05),
        final_objective=jnp.asarray(1.3),
        final_gradient_norm=jnp.asarray(1e-9),
        N_j=n_j,
        moment_residual=m_res,
        optimizer_info=opt_info,
    )
    lc = labels_mod.LabelContext(
        param_names=("beta", "gamma"),
        moment_names=("euler_a", "euler_b", "euler_c"),
    )
    return t.EstimationResult(
        theta_hat=_EulerParams(beta=0.95, gamma=2.0),
        Sigma_theta=sigma,
        V_X=v_x,
        J_stat=jnp.asarray(1.3),
        J_dof=1,
        J_pvalue=jnp.asarray(0.25),
        J_pvalue_adjusted=jnp.asarray(0.25),
        converged=True,
        iterations=12,
        theta_init=_EulerParams(beta=0.9, gamma=1.5),
        measure=_StubMeasure(),
        covariance=_StubCovariance(),
        weighting=_StubWeighting(),
        regularization=_StubRegularization(),
        diagnostics=diagnostics,
        labels=lc,
        manifold_spec=manifold_spec,
    )


class TestV1ManifoldSpecNone:
    def test_coef_table_index_is_field_names(self):
        r = _make_scalar_result(manifold_spec=None)
        tab = r.coef_table
        assert list(tab.index) == ["beta", "gamma"]
        assert list(tab["estimate"].to_numpy()) == [0.95, 2.0]

    def test_to_pandas_sigma_index_field_names(self):
        r = _make_scalar_result(manifold_spec=None)
        sigma = r.to_pandas()["Sigma_theta"]
        assert list(sigma.index) == ["beta", "gamma"]
        assert list(sigma.columns) == ["beta", "gamma"]

    def test_components_returns_scalar_tuple_field_order(self):
        r = _make_scalar_result(manifold_spec=None)
        comps = r.components()
        assert len(comps) == 2
        assert float(comps[0]) == 0.95
        assert float(comps[1]) == 2.0
        # theta property agrees
        comps2 = r.theta.components()
        assert float(comps2[0]) == 0.95

    def test_standard_errors_unchanged(self):
        r = _make_scalar_result(manifold_spec=None)
        se = r.standard_errors
        assert int(se.array.shape[0]) == 2


class TestV1AllScalarSpec:
    """Even with a (non-None) all-scalar spec threaded, the v1 branch holds."""

    def test_all_scalar_spec_uses_field_names(self):
        # A spec whose leaves are all scalar (e.g. a Euclidean()/Positive
        # v1 tree) must NOT trigger positional labels.
        @jdc.pytree_dataclass
        class _P:
            a: ManifoldLeaf
            b: ManifoldLeaf

        p = _P(
            a=ManifoldLeaf(jnp.asarray(0.95), Euclidean()),
            b=ManifoldLeaf(jnp.asarray(2.0), Euclidean()),
        )
        spec = manifold_spec_from_params(p)
        assert all(ls.ambient_shape == () for ls in spec.leaf_specs)
        r = _make_scalar_result(manifold_spec=spec)
        # all-scalar spec -> _is_non_scalar_spec False -> field-name index
        assert list(r.coef_table.index) == ["beta", "gamma"]


class TestTangentBasisNames:
    def test_none_returns_fallback(self):
        assert tangent_basis_names(None, ("a", "b")) == ("a", "b")
        assert tangent_basis_names(None) == ()

    def test_all_scalar_reproduces_field_names(self):
        @jdc.pytree_dataclass
        class _P:
            beta: ManifoldLeaf
            gamma: ManifoldLeaf

        p = _P(
            beta=ManifoldLeaf(jnp.asarray(0.9), Euclidean()),
            gamma=ManifoldLeaf(jnp.asarray(1.5), Euclidean()),
        )
        spec = manifold_spec_from_params(p)
        assert tangent_basis_names(spec) == ("beta", "gamma")

    def test_non_scalar_positional_labels_count_and_order(self):
        @jdc.pytree_dataclass
        class _P:
            Y: ManifoldLeaf
            phi: ManifoldLeaf

        p = _P(
            Y=ManifoldLeaf(jnp.zeros((5, 2)), PSDFixedRank(5, 2)),
            phi=ManifoldLeaf(jnp.zeros((1,)), Euclidean(1)),
        )
        spec = manifold_spec_from_params(p)
        names = tangent_basis_names(spec)
        assert len(names) == 5 * 2 + 1
        # C-order ravel of (5,2): Y[0,0], Y[0,1], Y[1,0], ...
        assert names[0] == "Y[0,0]"
        assert names[1] == "Y[0,1]"
        assert names[2] == "Y[1,0]"
        assert names[-1] == "phi[0]"
