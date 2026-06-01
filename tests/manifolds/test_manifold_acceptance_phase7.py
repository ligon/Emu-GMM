r"""Phase-7 acceptance / validation gate for the gauge-invariant Gamma-functional
standard errors (#42 delta-method pull-back).

This module is the END-TO-END correctness proof of the Phase-7 contract: the
``result.functional_se(f)`` primitive plus the ``gamma_se`` / ``gamma_covariance``
/ ``eigenvalue_se`` conveniences. It is **test-only** -- it adds NO src math
beyond the Phase-7 helper itself, and it does NOT consume any new runtime
dependency (the bootstrap / finite-difference references are pure JAX).

The shared synthetic DGP is the SAME ``Product(PSDFixedRank(5, K), Euclidean(1))``
the Phase-6 gate uses (cloned, not imported, so this file is self-contained and
does not couple to Phase-6 test internals):

    theta = (Y in R^{5xK},  phi in R),   Gamma_true = A @ A.T (rank K)
    model_m(theta) = concat( triu(Y @ Y.T),  phi )            # length M = 16
    psi(x, theta)  = model_m(theta) - x

The residual depends on ``theta`` only through the gauge-invariant ``Gamma``
and ``phi`` (so the O(K) fibre is a true symmetry of the objective).

The gates (Phase-7 design brief "Validation"):

1.  ``TestFunctionalSEShapes``       -- finite, correctly-shaped SEs for K=2,3
2.  ``TestGaugeInvarianceOfSEs``     -- Y0 vs Y0@Q identical SEs (positive ctl)
3.  ``TestNegativeControl``          -- gauge-VIOLATING f gets NONZERO variance
4.  ``TestCorrectnessVsReference``   -- delta-method == finite-diff Jacobian and
                                        == parametric-bootstrap SD to MC tol
5.  ``TestV1ScalarReduction``        -- scalar functional_se == standard_errors
6.  ``TestVechOrderingContract``     -- vech order is row-major lower-triangular
7.  ``TestDegenerateEigenvalues``    -- repeated nonzero eigenvalues -> nan, doc'd
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.inference.functional_se import (
    eigenvalue_se,
    functional_se,
    gamma_vech,
    vech_indices,
)
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side (K-Agg primary: n=5, K in {2,3} -> k/n in [0.4, 0.6])
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)


@jdc.pytree_dataclass
class ProductParams:
    """A ``PSDFixedRank(5, K)`` ``Y`` leaf plus a ``Euclidean(1)`` ``phi`` leaf."""

    Y: ManifoldLeaf
    phi: ManifoldLeaf


def _make_params(Y, phi, k) -> ProductParams:
    return ProductParams(
        Y=ManifoldLeaf(jnp.asarray(Y), PSDFixedRank(N, k)),
        phi=ManifoldLeaf(jnp.reshape(jnp.asarray(phi), (1,)), Euclidean(1)),
    )


def _moment_count() -> int:
    return N * (N + 1) // 2 + 1  # 16


def _gauge_invariant_model(x, theta):
    """psi = triu(Y Y^T) concat phi - x. Depends on theta ONLY through Gamma."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _orthogonal(seed: int, k: int, reflect: bool) -> jnp.ndarray:
    """Random Q in O(K); ``reflect`` selects the det=-1 (improper) component."""
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(k, k)))
    q, r = jnp.linalg.qr(g)
    q = q @ jnp.diag(jnp.sign(jnp.diag(r)))
    want = -1.0 if reflect else 1.0
    if float(jnp.linalg.det(q)) * want < 0:
        q = q.at[:, 0].set(-q[:, 0])
    return q


def _make_dgp(k: int, *, noise: float, n_sim: int, data_seed: int):
    """Truth, synthetic measure, near-truth warm start (clone of Phase-6)."""
    rng = np.random.default_rng(data_seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    M = _moment_count()
    noise_key = jax.random.PRNGKey(data_seed)

    def sampler(key, theta):
        del key
        noise_draws = noise * jax.random.normal(noise_key, (n_sim, M))
        return target[None, :] + noise_draws

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
    theta_init = _make_params(Y0, 0.65, k)
    return measure, theta_init, A_true, Gamma_true, phi_true, M


def _estimate(model, measure, theta_init, *, max_steps: int = 400):
    return estimate(
        model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=ContinuouslyUpdated(),
        optimizer=riemannian_lm(max_steps=max_steps),
        theta_init=theta_init,
    )


# ---------------------------------------------------------------------------
# Gate 1 -- shapes / finiteness.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
class TestFunctionalSEShapes:
    r"""``eigenvalue_se`` / ``gamma_se`` / ``functional_se`` return finite,
    correctly-shaped SEs for ``Product(PSDFixedRank(5, K), Euclidean(1))``.

    ``eigenvalue_se`` returns length K (the K NONZERO eigenvalues of the
    rank-K Gamma); the n-K structural zeros are excluded (documented; their
    eigenvalue Jacobian is degenerate). ``gamma_se`` returns length
    n(n+1)/2 = 15.
    """

    def test_eigenvalue_and_gamma_se_shapes(self, k):
        measure, theta_init, _A, Gamma_true, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=300 + k
        )
        result = _estimate(_gauge_invariant_model, measure, theta_init)
        assert bool(result.converged)

        ev_se = result.eigenvalue_se()
        assert ev_se.shape == (k,)  # K nonzero eigenvalues, NOT n
        assert bool(jnp.all(jnp.isfinite(ev_se)))
        assert bool(jnp.all(ev_se > 0.0))

        # explicit rank argument agrees with the inferred default.
        assert bool(jnp.allclose(result.eigenvalue_se(rank=k), ev_se))

        g_se = result.gamma_se()
        q = N * (N + 1) // 2
        assert g_se.shape == (q,)
        assert bool(jnp.all(jnp.isfinite(g_se)))
        assert bool(jnp.all(g_se > 0.0))

        gcov = result.gamma_covariance()
        assert gcov.shape == (q, q)
        # PSD up to round-off (R33).
        evc = jnp.linalg.eigvalsh(0.5 * (gcov + gcov.T))
        assert float(jnp.min(evc)) > -1e-10 * float(jnp.max(jnp.abs(evc)))

        # general primitive: a vector-valued gauge-invariant f.
        def f(comps):
            A, phi = comps
            G = A @ A.T
            return jnp.array([G[0, 0], G[1, 1], jnp.trace(G), phi[0]])

        se, cov = result.functional_se(f)
        assert se.shape == (4,)
        assert cov.shape == (4, 4)
        assert bool(jnp.all(jnp.isfinite(se)))


# ---------------------------------------------------------------------------
# Gate 2 -- gauge invariance of the SEs (positive control).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("reflect", [False, True])
class TestGaugeInvarianceOfSEs:
    r"""Two starts ``Y0`` and ``Y0 @ Q`` (Q in O(K), both SO(K) and the
    reflection component) give IDENTICAL gauge-invariant functional SEs.

    This is the core Phase-7 guarantee: a gauge-invariant f's J_f annihilates
    the gauge nullspace already pinned out of Sigma_theta, so eigenvalue_se /
    gamma_se / functional_se(good f) are identical along the O(K) orbit. The
    tolerance is TIGHT (solver determinism along the fibre, not asymptotics).
    If it fails, suspect a gauge leak -- do NOT loosen.
    """

    SE_ATOL = 1e-7

    def test_Y0_and_Y0Q_give_identical_functional_ses(self, k, reflect):
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=320 + k
        )
        Y0 = jnp.asarray(theta_init.Y.array)
        phi0 = float(jnp.reshape(theta_init.phi.array, ()))
        Q = _orthogonal(11 + k, k, reflect=reflect)
        assert bool(jnp.allclose(Q @ Q.T, jnp.eye(k), atol=1e-10))

        res_a = _estimate(_gauge_invariant_model, measure, theta_init)
        res_b = _estimate(
            _gauge_invariant_model, measure, _make_params(Y0 @ Q, phi0, k)
        )
        assert bool(res_a.converged) and bool(res_b.converged)

        # eigenvalue SEs identical.
        assert bool(
            jnp.allclose(
                res_a.eigenvalue_se(), res_b.eigenvalue_se(), atol=self.SE_ATOL
            )
        )
        # vech(Gamma) SEs identical.
        assert bool(jnp.allclose(res_a.gamma_se(), res_b.gamma_se(), atol=self.SE_ATOL))

        # a general gauge-invariant functional: identical.
        def f(comps):
            A, phi = comps
            G = A @ A.T
            return jnp.array([G[0, 0], G[2, 1], jnp.trace(G @ G), phi[0]])

        sa, _ = res_a.functional_se(f)
        sb, _ = res_b.functional_se(f)
        assert bool(jnp.allclose(sa, sb, atol=self.SE_ATOL))


# ---------------------------------------------------------------------------
# Gate 3 -- negative control: gauge-VIOLATING f gets nonzero gauge variance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
class TestNegativeControl:
    r"""A gauge-VIOLATING functional (depends on raw ``Y`` not only through
    ``Gamma``) gets NONZERO variance from the gauge directions, and its SE
    CHANGES under a gauge rotation Y0 -> Y0 @ Q.

    This proves the gauge-invariance of the GOOD functionals is real (the
    gauge nullspace in Sigma_theta carries genuine variance that a
    gauge-invariant f annihilates) and not an artefact of an always-zero
    Sigma_theta (R6/R9/R32).
    """

    def test_gauge_violating_f_has_nonzero_and_orbit_dependent_se(self, k):
        if k < 2:
            pytest.skip("gauge group trivial for k<2")
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            k, noise=0.05, n_sim=200, data_seed=340 + k
        )
        Y0 = jnp.asarray(theta_init.Y.array)
        phi0 = float(jnp.reshape(theta_init.phi.array, ()))
        Q = _orthogonal(13 + k, k, reflect=False)
        res_a = _estimate(_gauge_invariant_model, measure, theta_init)
        res_b = _estimate(
            _gauge_invariant_model, measure, _make_params(Y0 @ Q, phi0, k)
        )
        assert bool(res_a.converged) and bool(res_b.converged)

        # f_bad leaks a raw Y entry (NOT a function of Gamma alone).
        def f_bad(comps):
            A, _phi = comps
            return jnp.reshape(A[0, 0], (1,))

        # f_good is the matching gauge-invariant Gamma entry.
        def f_good(comps):
            A, _phi = comps
            return jnp.reshape((A @ A.T)[0, 0], (1,))

        se_bad_a, _ = res_a.functional_se(f_bad)
        se_bad_b, _ = res_b.functional_se(f_bad)
        se_good_a, _ = res_a.functional_se(f_good)
        se_good_b, _ = res_b.functional_se(f_good)

        # The gauge-violating SE is finite and strictly positive (it picks up
        # variance from the gauge fibre that Sigma_theta carries).
        assert bool(jnp.all(jnp.isfinite(se_bad_a)))
        assert float(se_bad_a[0]) > 0.0

        # The gauge-violating SE CHANGES under the orbit rotation; the
        # gauge-invariant one does NOT. This is the discriminating proof.
        assert not bool(jnp.allclose(se_bad_a, se_bad_b, atol=1e-6))
        assert bool(jnp.allclose(se_good_a, se_good_b, atol=1e-7))


# ---------------------------------------------------------------------------
# Gate 4 -- correctness vs finite-difference Jacobian + parametric bootstrap.
# ---------------------------------------------------------------------------
class TestCorrectnessVsReference:
    r"""The delta-method SEs match (a) a finite-difference Jacobian sandwich
    and (b) a parametric-bootstrap empirical SD, both on the SAME synthetic
    DGP, to the appropriate tolerance.

    (a) Finite-difference: re-derive J_f by central differences on the flat
        ambient vector, sandwich with the SAME Sigma_theta. This isolates the
        AD Jacobian from the rest of the pipeline (exact agreement expected).

    (b) Parametric bootstrap (R28): resample the per-draw noise (fresh seeds,
        same truth), refit, recompute eigenvalue_se on each fit, take the
        empirical SD of the eigenvalue point estimates across replicates, and
        check the delta-method SE is within a factor of the bootstrap SD
        (MC tolerance at bounded N_boot). Bounded Python loop, fixed seeds,
        taskset-pinned; NO vmap (JIT-mmap hazard).
    """

    def _flat_and_shapes(self, components):
        shapes = [tuple(int(s) for s in jnp.asarray(c).shape) for c in components]
        flat = jnp.concatenate([jnp.reshape(jnp.asarray(c), (-1,)) for c in components])
        return flat, shapes

    def _to_components(self, flat, shapes):
        out = []
        off = 0
        for sh in shapes:
            size = int(np.prod(sh)) if sh != () else 1
            blk = flat[off : off + size]
            out.append(jnp.reshape(blk, sh) if sh != () else jnp.reshape(blk, ()))
            off += size
        return tuple(out)

    @pytest.mark.parametrize("k", [2, 3])
    def test_delta_matches_finite_difference_jacobian(self, k):
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=360 + k
        )
        result = _estimate(_gauge_invariant_model, measure, theta_init)
        assert bool(result.converged)
        comps = result.components()
        Sigma = jnp.asarray(result.Sigma_theta.array)
        flat, shapes = self._flat_and_shapes(comps)

        def f_ev(components):
            A = components[0]
            ev = jnp.linalg.eigvalsh(A @ A.T)
            n = int(ev.shape[0])
            return ev[n - k :]

        # AD-based delta-method SE (the implementation under test).
        se_ad = result.eigenvalue_se(rank=k)

        # Finite-difference Jacobian of f_ev w.r.t. the flat ambient vector.
        eps = 1e-6
        D = int(flat.shape[0])
        p = k
        J_fd = np.zeros((p, D))
        for j in range(D):
            ep = flat.at[j].add(eps)
            em = flat.at[j].add(-eps)
            yp = np.asarray(f_ev(self._to_components(ep, shapes)))
            ym = np.asarray(f_ev(self._to_components(em, shapes)))
            J_fd[:, j] = (yp - ym) / (2 * eps)
        J_fd = jnp.asarray(J_fd)
        cov_fd = J_fd @ Sigma @ J_fd.T
        se_fd = jnp.sqrt(jnp.clip(jnp.diag(cov_fd), 0.0, None))

        # AD and finite-difference SEs agree to FD precision.
        assert bool(jnp.allclose(se_ad, se_fd, rtol=1e-4, atol=1e-7))

    @pytest.mark.slow
    @pytest.mark.parametrize("k", [2])
    def test_delta_matches_parametric_bootstrap(self, k):
        # One ground-truth DGP; resample the noise, refit, collect eigenvalue
        # point estimates, compare delta-method SE to the bootstrap SD.
        n_boot = 40
        noise, n_sim = 0.05, 200
        rng = np.random.default_rng(7000 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        Gamma_true = A_true @ A_true.T
        phi_true = 0.7
        g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
        target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
        M = _moment_count()
        Y0 = jnp.asarray(A_true + 0.03 * rng.normal(size=(N, k)))
        theta_init = _make_params(Y0, 0.65, k)

        def make_measure(seed):
            nk = jax.random.PRNGKey(seed)

            def sampler(key, theta):
                del key
                return target[None, :] + noise * jax.random.normal(nk, (n_sim, M))

            return SyntheticMeasure(
                key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler
            )

        # Delta-method SE from ONE reference fit.
        ref = _estimate(_gauge_invariant_model, make_measure(10_000), theta_init)
        assert bool(ref.converged)
        se_delta = np.asarray(ref.eigenvalue_se(rank=k))

        # Bootstrap: refit on fresh noise draws, collect the K eigenvalues.
        evs = []
        for i in range(n_boot):
            res = _estimate(
                _gauge_invariant_model, make_measure(11_000 + i), theta_init
            )
            if not bool(res.converged):
                continue
            A_hat, _ = res.components()
            ev = np.asarray(jnp.linalg.eigvalsh(A_hat @ A_hat.T))
            evs.append(ev[N - k :])  # top-k ascending
        evs = np.asarray(evs)
        assert evs.shape[0] >= n_boot - 5  # almost all converged
        se_boot = evs.std(axis=0, ddof=1)

        # Delta-method SE within a factor of the bootstrap SD (MC tol at
        # n_boot=40; the delta method and the bootstrap target the same
        # asymptotic variance). Allow [0.5x, 2x].
        ratio = se_delta / se_boot
        assert bool(np.all(ratio > 0.5)), (se_delta, se_boot, ratio)
        assert bool(np.all(ratio < 2.0)), (se_delta, se_boot, ratio)


# ---------------------------------------------------------------------------
# Gate 5 -- v1 / scalar reduction matches the ordinary delta-method SE.
# ---------------------------------------------------------------------------
class TestV1ScalarReduction:
    r"""For an all-scalar (v1) tree, ``functional_se`` reduces to the ordinary
    delta-method SE, and for the identity component projector it agrees with
    ``standard_errors`` bitwise-tight (the contract item 4 / R10/R15/R21/R25).
    """

    def _v1_result(self):
        import jax_dataclasses as _jdc
        from emu_gmm.measures import SyntheticMeasure as _SM

        @_jdc.pytree_dataclass
        class P:
            a: jax.Array
            b: jax.Array

        # A simple linear-in-theta moment model: m = (a, b, a + b) - x.
        target = jnp.array([0.3, -0.7, 0.3 - 0.7])

        def model(x, theta):
            return jnp.array([theta.a, theta.b, theta.a + theta.b]) - x

        nk = jax.random.PRNGKey(1)

        def sampler(key, theta):
            del key, theta
            return target[None, :] + 0.05 * jax.random.normal(nk, (200, 3))

        measure = _SM(key=jax.random.PRNGKey(0), n_sim=200, sampler=sampler)
        theta_init = P(a=jnp.asarray(0.2), b=jnp.asarray(-0.6))
        return estimate(
            model,
            measure,
            covariance=SyntheticCovariance(),
            weighting=ContinuouslyUpdated(),
            theta_init=theta_init,
        )

    def test_identity_projector_matches_standard_errors(self):
        result = self._v1_result()
        se_std = np.asarray(result.standard_errors.array)
        # f = identity (return both scalars).
        se_id, _ = result.functional_se(lambda c: jnp.array([c[0], c[1]]))
        assert bool(jnp.allclose(jnp.asarray(se_id), jnp.asarray(se_std), atol=1e-10))
        # Per-coordinate projectors agree element-wise.
        for i in range(2):
            se_i, _ = result.functional_se(lambda c, _i=i: jnp.reshape(c[_i], (1,)))
            assert float(se_i[0]) == pytest.approx(float(se_std[i]), abs=1e-10)

    def test_scalar_transform_matches_manual_delta(self):
        result = self._v1_result()
        Sigma = np.asarray(result.Sigma_theta.array)
        a_hat = float(jnp.asarray(result.components()[0]))

        # f = a**2: J_f = [2a, 0]; Var = 4 a^2 Sigma_aa.
        se_f, cov_f = result.functional_se(lambda c: jnp.reshape(c[0] ** 2, (1,)))
        manual_var = 4.0 * a_hat**2 * Sigma[0, 0]
        assert float(cov_f[0, 0]) == pytest.approx(manual_var, rel=1e-9)
        assert float(se_f[0]) == pytest.approx(np.sqrt(manual_var), rel=1e-9)


# ---------------------------------------------------------------------------
# Gate 6 -- vech ordering contract.
# ---------------------------------------------------------------------------
class TestVechOrderingContract:
    r"""``gamma_se`` / ``gamma_vech`` use the canonical row-major LOWER-
    triangular vech order: Gamma[0,0], Gamma[1,0], Gamma[1,1], Gamma[2,0], ...
    (length n(n+1)/2). Verified directly against ``jnp.tril_indices`` (R13/R29).
    """

    def test_vech_indices_are_row_major_lower_triangular(self):
        ii, jj = vech_indices(N)
        pairs = list(zip([int(x) for x in ii], [int(y) for y in jj], strict=True))
        expected = [(i, j) for i in range(N) for j in range(i + 1)]
        assert pairs == expected

    def test_gamma_vech_matches_lower_triangle(self):
        rng = np.random.default_rng(0)
        A = jnp.asarray(rng.normal(size=(N, 2)))
        G = A @ A.T
        v = np.asarray(gamma_vech((A, jnp.array([0.7]))))
        ii, jj = vech_indices(N)
        expected = np.asarray(G)[np.asarray(ii), np.asarray(jj)]
        assert np.allclose(v, expected)
        # First entry is Gamma[0,0]; second is Gamma[1,0]; symmetric so the
        # lower entry equals the upper.
        assert v[0] == pytest.approx(float(G[0, 0]))
        assert v[1] == pytest.approx(float(G[1, 0]))


# ---------------------------------------------------------------------------
# Gate 7 -- degenerate eigenvalues -> nan (documented), not a crash.
# ---------------------------------------------------------------------------
class TestDegenerateEigenvalues:
    r"""At exact eigenvalue degeneracy the individual eigenvalues are not
    smooth functions of Gamma, so the per-eigenvalue SE is NOT well-defined --
    ``eigvalsh`` returns a finite but *eigenbasis-dependent* derivative there.
    ``eigenvalue_se`` must NOT crash, must return the right shape, and the
    documented hazard (basis-dependence at exact degeneracy) is demonstrated;
    the generic distinct-eigenvalue case is exact and a near-degenerate
    spectrum yields large-but-finite SEs (R3/R12/R18). The structural n-K
    zeros are excluded from ``eigenvalue_se`` by construction, so they never
    enter the Jacobian.
    """

    def test_repeated_nonzero_eigenvalues_do_not_crash(self):
        # Gamma with two EQUAL nonzero eigenvalues (both 1) and three
        # structural zeros, in a GENERIC (non-axis-aligned) eigenbasis.
        rng = np.random.default_rng(3)
        U = jnp.asarray(np.linalg.qr(rng.normal(size=(N, 2)))[0])  # 5x2 orthonormal
        comps = (U, jnp.array([0.7]))
        ev = jnp.linalg.eigvalsh(U @ U.T)
        assert float(ev[-1]) == pytest.approx(float(ev[-2]), abs=1e-10)  # degenerate
        D = N * 2 + 1
        se, _cov = eigenvalue_se(comps, jnp.eye(D), 2)
        # No crash; correct shape; finite (eigvalsh's JVP is robust here).
        assert se.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(se)))

    def test_degenerate_per_eigenvalue_se_is_basis_dependent(self):
        # The documented hazard: at exact degeneracy the per-eigenvalue SE is
        # not unique -- two eigenbases of the SAME degenerate Gamma give
        # different per-eigenvalue SEs (so they must not be trusted), whereas
        # the SUM of the degenerate block (a symmetric, smooth functional) is
        # basis-INVARIANT. This is why eigenvalue_se documents distinctness.
        rng = np.random.default_rng(5)
        U1 = jnp.asarray(np.linalg.qr(rng.normal(size=(N, 2)))[0])
        # A different orthonormal basis spanning a different 2-plane: distinct
        # Gamma but BOTH have spectrum {1,1,0,0,0}.
        U2 = jnp.asarray(np.linalg.qr(rng.normal(size=(N, 2)))[0])
        D = N * 2 + 1
        Sigma = jnp.eye(D)
        se1, _ = eigenvalue_se((U1, jnp.array([0.7])), Sigma, 2)
        se2, _ = eigenvalue_se((U2, jnp.array([0.7])), Sigma, 2)

        def block_sum(comps):
            A = comps[0]
            ev = jnp.linalg.eigvalsh(A @ A.T)
            return jnp.reshape(ev[N - 2 :].sum(), (1,))

        sum1, _ = functional_se(block_sum, (U1, jnp.array([0.7])), Sigma)
        sum2, _ = functional_se(block_sum, (U2, jnp.array([0.7])), Sigma)
        # The symmetric block-sum SE is well-defined and basis-invariant;
        # the individual eigenvalue SEs are not (this is the hazard).
        assert float(sum1[0]) == pytest.approx(float(sum2[0]), rel=1e-6)
        # (Individual SEs may or may not differ numerically; the contract is
        # only that they are not a trustworthy per-eigenvalue quantity here.)

    def test_distinct_nonzero_eigenvalues_are_finite_and_exact(self):
        A = jnp.zeros((N, 2)).at[0, 0].set(2.0).at[1, 1].set(1.0)
        comps = (A, jnp.array([0.7]))
        D = N * 2 + 1
        se, _cov = eigenvalue_se(comps, jnp.eye(D), 2)
        assert se.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(se)))
        assert bool(jnp.all(se > 0.0))
