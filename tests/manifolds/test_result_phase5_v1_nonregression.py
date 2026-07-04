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
from emu_gmm._internal import labels as labels_mod
from emu_gmm._internal.labels import tangent_basis_names
from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf


@jdc.pytree_dataclass
class _EulerParams:
    beta: float
    gamma: float


def _make_scalar_result(manifold_spec=None) -> t.OptimizationResult:
    # Post-split, Sigma_theta is ASSEMBLED by the law from the optimization
    # ingredients (moment Jacobian G, weighting Lambda, raw meat V). A minimal
    # but consistent (M=3, D=2) set makes coef_table / se / cov assemble without
    # error; these v1 non-regression checks are about the field-name LABELS, not
    # the numeric Sigma values.
    G = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2), full column rank
    Lam = jnp.eye(3)  # weighting matrix Lambda
    V = jnp.eye(3)  # raw moment covariance (meat)
    lc = labels_mod.LabelContext(
        param_names=("beta", "gamma"),
        moment_names=("euler_a", "euler_b", "euler_c"),
    )
    return t.OptimizationResult(
        theta_hat=_EulerParams(beta=0.95, gamma=2.0),
        moment_jacobian=G,
        weighting_matrix=Lam,
        moment_covariance=V,
        objective_value=jnp.asarray(1.3),
        n_overid=1,
        converged=True,
        iterations=12,
        theta_init=_EulerParams(beta=0.9, gamma=1.5),
        labels=lc,
        manifold_spec=manifold_spec,
    )


class TestV1ManifoldSpecNone:
    def test_coef_table_index_is_field_names(self):
        r = _make_scalar_result(manifold_spec=None)
        tab = r.asymptotic().coef_table
        assert list(tab.index) == ["beta", "gamma"]
        assert list(tab["estimate"].to_numpy()) == [0.95, 2.0]

    def test_to_pandas_sigma_index_field_names(self):
        r = _make_scalar_result(manifold_spec=None)
        law = r.asymptotic()
        # Sigma_theta moved to the law as a plain (D, D) numpy array; the v1
        # field-name labels now ride on the law's coef_table index.
        assert law.cov().shape == (2, 2)
        assert list(law.coef_table.index) == ["beta", "gamma"]

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
        se = r.asymptotic().se()
        assert int(se.shape[0]) == 2


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
        assert list(r.asymptotic().coef_table.index) == ["beta", "gamma"]


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
