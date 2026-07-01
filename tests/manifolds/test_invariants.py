"""Manifold-declared gauge-invariant functionals (the leaf-view foundation).

Each manifold advertises, via ``invariants()``, the canonical gauge-invariant
functionals of a leaf living on it. These are what an ``EstimatorLaw`` routes
through its generic query algebra (``law.leaf(name).se("eigenvalues")``), so the
"which queries make sense" knowledge lives in the geometry, not in an
application-specific law subclass.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from emu_gmm import Euclidean, Interval, Positive, PSDFixedRank
from emu_gmm.inference.functional_se import gamma_eigenvalues, gamma_vech

jax.config.update("jax_enable_x64", True)


class TestFlatManifolds:
    @pytest.mark.parametrize(
        "manifold, x, expected",
        [
            (Euclidean(3), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
            (Euclidean(), np.array(2.5), np.array([2.5])),  # scalar -> length 1
            (Positive(), np.array(2.0), np.array([2.0])),
            (Interval(0.0, 1.0), np.array(0.3), np.array([0.3])),
        ],
    )
    def test_value_is_the_raveled_coordinate(self, manifold, x, expected):
        inv = manifold.invariants()
        assert set(inv) == {"value"}
        np.testing.assert_allclose(np.asarray(inv["value"](x)), expected)


class TestPSDFixedRank:
    _A = np.array(
        [[1.0, 0.2], [0.3, 1.1], [0.5, -0.4], [0.1, 0.9], [-0.2, 0.6]]
    )  # (5, 2)

    def test_keys(self):
        assert set(PSDFixedRank(5, 2).invariants()) == {"eigenvalues", "gamma"}

    def test_eigenvalues_match_functional_se(self):
        inv = PSDFixedRank(5, 2).invariants()
        ev = np.asarray(inv["eigenvalues"](self._A))
        assert ev.shape == (2,)  # k nonzero eigenvalues, structural zeros dropped
        np.testing.assert_allclose(ev, np.asarray(gamma_eigenvalues((self._A,), 2, 0)))

    def test_gamma_vech_match_functional_se(self):
        inv = PSDFixedRank(5, 2).invariants()
        g = np.asarray(inv["gamma"](self._A))
        assert g.shape == (5 * 6 // 2,)  # n(n+1)/2 vech entries
        np.testing.assert_allclose(g, np.asarray(gamma_vech((self._A,), 0)))

    def test_gauge_invariance(self):
        """Gamma-invariants are unchanged under A -> A @ Q for Q in O(k)."""
        inv = PSDFixedRank(5, 2).invariants()
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(2, 2)))
        AQ = self._A @ q
        for name in ("eigenvalues", "gamma"):
            np.testing.assert_allclose(
                np.asarray(inv[name](self._A)),
                np.asarray(inv[name](AQ)),
                atol=1e-10,
            )

    def test_functionals_are_jax_differentiable(self):
        """The functionals must be AD-able (the delta method needs the Jacobian)."""
        inv = PSDFixedRank(5, 2).invariants()
        jac = jax.jacobian(lambda A: inv["eigenvalues"](A))(self._A)
        assert np.all(np.isfinite(np.asarray(jac)))
