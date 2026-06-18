r"""Phase-6 acceptance / validation gate for the manifold pipeline (#12).

This module is the END-TO-END correctness proof of the landed Phase 1-5
manifold infrastructure (ManifoldLeaf + manifold-aware flatten/spec; per-leaf
Riemannian-LM; gauge-aware inference; the result readout). It is
**test-only** -- it adds NO src math. Each gate is designed to *expose* a
defect in a specific Phase 1-5 layer rather than mask it with a loose
tolerance; the docstrings record which layer fails which gate.

The shared synthetic DGP is a ``Product(PSDFixedRank(5, K), Euclidean(1))``:

    theta = (Y in R^{5xK},  phi in R),   Gamma_true = A @ A.T (rank K)
    model_m(theta) = concat( triu(Y @ Y.T),  phi )            # length M = 16
    psi(x, theta)  = model_m(theta) - x

The residual depends on ``theta`` ONLY through the gauge-invariant functionals
``Gamma = Y Y^T`` (its 15 upper-triangular entries, n=5) and ``phi`` (1 entry).
Hence ``(Y Q)(Y Q)^T = Y Y^T`` for any ``Q in O(K)`` -- the unique
gauge-invariant minimiser is ``(Gamma_true, phi_true)`` and the O(K) fibre is a
true symmetry of the objective (R1/R2). Every recovery / agreement assertion
is made on a gauge-INVARIANT functional (``Gamma_hat``, ``eigvalsh(Gamma_hat)``,
``J_stat``, ``phi_hat``), NEVER on raw ``Y`` entries (which differ by an O(K)
rotation; R8/R13/R16).

Per-draw moment noise is i.i.d. ``noise * Normal``. A *noiseless* draw makes the
moment-variance ``V`` singular and the whitening blow up (verified: gradnorm
~1e40 at noise=0); so the DGP carries a small but nonzero noise everywhere.

The gates (see ``docs/manifold-slice-scoping.md`` "Phase 6 -- Validation" and
the acceptance list):

1.  ``TestTightSyntheticRecovery``    -- gate #1 (recovery, principled tol)
2.  ``TestGaugeInvariance``           -- gate #3 (Y0 vs Y0@Q, incl. reflection)
3.  ``TestPymanoptCrossCheck``        -- gate #2 (pymanopt-TR on the quotient)
4.  ``TestJDofChiSquareCalibration``  -- gate #4 (mean(J) ~= J_dof, adversarial)
5.  ``TestFlattenRoundTrip``          -- gate #7 (flatten/unflatten shape preserve)

The v1 bitwise non-regression (gate #5) is exercised by running the full v1
suite unchanged; it is not duplicated here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm._internal.params import (
    flatten_params_with_spec,
    manifold_spec_from_params,
    unflatten_params,
)
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side (K-Agg primary: n=5, K in {2,3} -> k/n in [0.4, 0.6])

# Upper-triangular index pairs of the (5,5) symmetric Gamma: 15 unique entries.
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2)


# ---------------------------------------------------------------------------
# Shared DGP helper (Phase-6 synthetic measure + model).
# ---------------------------------------------------------------------------
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
    # M = n(n+1)/2 unique triu(Gamma) entries + 1 (phi) = 15 + 1 = 16.
    # Over-identified for K=2 (id-dim 10) and K=3 (id-dim 13).
    return N * (N + 1) // 2 + 1


def _gauge_invariant_model(x, theta):
    """psi = triu(Y Y^T) concat phi  -  x. Depends on theta ONLY through
    the gauge-invariant ``Gamma = Y Y^T`` and ``phi`` (R1)."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _orthogonal(seed: int, k: int, reflect: bool) -> jnp.ndarray:
    """Random ``Q`` from QR-of-Gaussian, on a requested O(K) component.

    The Q-factor of a random Gaussian's QR is orthogonal but its determinant
    sign is whatever LAPACK produced (the R-diagonal sign canonicalisation
    fixes R, not det(Q)). We therefore force the determinant EXPLICITLY:
    ``reflect=False`` -> det(Q)=+1 (SO(K), proper rotation); ``reflect=True``
    -> det(Q)=-1 (the improper / reflection component O(K) \\ SO(K)),
    exercising BOTH components of O(K) (R6/R28). Flipping one column toggles
    the determinant sign while preserving orthogonality."""
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(k, k)))
    q, r = jnp.linalg.qr(g)
    q = q @ jnp.diag(jnp.sign(jnp.diag(r)))  # canonicalise QR sign
    want = -1.0 if reflect else 1.0
    if float(jnp.linalg.det(q)) * want < 0:
        q = q.at[:, 0].set(-q[:, 0])  # toggle det sign to the requested one
    return q


def _make_dgp(k: int, *, noise: float, n_sim: int, data_seed: int):
    """Build the truth, the synthetic measure, and a near-truth warm start.

    Returns ``(measure, theta_init, A_true, Gamma_true, phi_true, M)``.
    ``data_seed`` drives BOTH the truth ``A`` and the per-draw noise key, so
    each Monte-Carlo replicate is a fresh, independent synthetic dataset.
    """
    rng = np.random.default_rng(data_seed)
    A_true = jnp.asarray(rng.normal(size=(N, k)))
    Gamma_true = A_true @ A_true.T
    phi_true = 0.7
    g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
    target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
    M = _moment_count()
    assert int(target.shape[0]) == M

    noise_key = jax.random.PRNGKey(data_seed)

    def sampler(key, theta):
        # Exogenous (theta-independent) per-draw targets: truth + i.i.d.
        # noise so V is well-conditioned. CRN-frozen ``noise_key`` -> a
        # deterministic objective surface in theta for a given replicate.
        del key
        noise_draws = noise * jax.random.normal(noise_key, (n_sim, M))
        return target[None, :] + noise_draws

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    # Warm start near the truth so the solver lands on the gauge-invariant
    # minimiser (recovery accuracy is the gate; basin-of-attraction is not).
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
# Gate #1 -- TIGHT synthetic recovery, principled tolerance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
class TestTightSyntheticRecovery:
    r"""Recover ``Gamma_true`` and ``phi_true`` to a tolerance JUSTIFIED from
    the DGP's noise/N -- not loosened-until-green (R2/R5/R23/R27).

    DGP tokens: ``noise = 0.01``, ``n_sim = 200``, ``M = 16``.
    The per-draw moment is ``target + 0.01 * Normal``, so the sample-mean
    moment error per entry has standard error ``sigma_m = 0.01 / sqrt(200)
    = 7.07e-4``. Near the truth the Gamma entries / eigenvalues and ``phi``
    enter the moment vector ~linearly, so the recovered-functional error
    inherits this ``O(sigma_m)`` scale up to a small O(1) factor from the
    nonlinear ``Y |-> Y Y^T`` map and the CU whitening. Empirically the worst
    Gamma / eigenvalue error over the test seeds is ``~1.6e-3`` (~2.3 sigma_m).

      atol(Gamma)  = atol(eigvalsh) = 4e-3   (~= 5.7 * sigma_m; ~2.5x worst obs)
      atol(phi)                     = 2e-3   (phi is linear/low-variance)

    Tightening below ``sigma_m`` would demand a larger ``n_sim`` or lower
    ``noise``; loosening above ``~5 sigma_m`` would let a wandering / biased
    estimator pass. Convergence is asserted on the traced ``done`` flag and a
    small gradient norm, NOT merely ``status=="traced"`` (R22/R32).
    """

    # sigma_m = noise / sqrt(n_sim) = 0.01 / sqrt(200) ~= 7.07e-4.
    ATOL_GAMMA = 4e-3
    ATOL_PHI = 2e-3
    GRADNORM_MAX = 1e-3  # ~14 * sigma_m: a genuinely converged Riemannian LM

    def test_recovers_gamma_and_phi(self, k):
        measure, theta_init, A_true, Gamma_true, phi_true, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=k
        )
        result = _estimate(_gauge_invariant_model, measure, theta_init)

        # REAL convergence (Phase-3 #78 done-flag), not the traced sentinel.
        assert bool(result.converged) is True
        assert float(jnp.asarray(result.diagnostics.final_gradient_norm)) < (
            self.GRADNORM_MAX
        )

        A_hat, phi_hat = result.components()
        assert A_hat.shape == (N, k)
        Gamma_hat = A_hat @ A_hat.T

        # Gauge-INVARIANT recovery: Gamma and its spectrum, never raw A.
        assert bool(jnp.allclose(Gamma_hat, Gamma_true, atol=self.ATOL_GAMMA))
        ev_hat = jnp.linalg.eigvalsh(Gamma_hat)
        ev_true = jnp.linalg.eigvalsh(Gamma_true)
        assert bool(jnp.allclose(ev_hat, ev_true, atol=self.ATOL_GAMMA))
        # rank-K PSD: exactly K positive eigenvalues, the rest ~0.
        assert int(jnp.sum(ev_hat > 1e-3)) == k
        assert float(jnp.reshape(phi_hat, ())) == pytest.approx(
            phi_true, abs=self.ATOL_PHI
        )


# ---------------------------------------------------------------------------
# Gate #3 -- Gauge invariance of the full estimate (point + J).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("reflect", [False, True])
class TestGaugeInvariance:
    r"""Two starts ``Y0`` and ``Y0 @ Q`` for random ``Q in O(K)`` (BOTH
    SO(K) and the reflection component, det<0) give the SAME gauge-invariant
    estimate (R3/R6/R16/R28/R31).

    Because the objective is exactly O(K)-invariant, ``Y0`` and ``Y0 @ Q`` lie
    on the same orbit at every iterate; a correctly gauged solver (per-leaf
    horizontal projection + gauge lambda-floor, Phase 3) and a
    correctly gauged inference block (pinv_eigvalrule dropping exactly
    K(K-1)/2 directions, Phase 4) must land on the SAME ``Gamma_hat`` /
    ``J_stat`` / ``Sigma_theta`` spectrum.

    The tolerances here are TIGHTER than the recovery tolerance (gate #1):
    both runs start from the same orbit, so this checks solver *determinism*
    along the fibre, NOT asymptotic accuracy. A loose tolerance would hide a
    partial gauge leak (R31). If these tight asserts fail, do NOT loosen --
    suspect the Phase-3 horizontal projection or the Phase-4 pinv drop count
    and STOP (R3).
    """

    GAMMA_ATOL = 1e-8  # gauge-invariant point estimate: solver determinism
    J_ATOL = 1e-8  # J is a scalar functional of Gamma at the optimum

    def test_Y0_and_Y0Q_agree_on_quotient(self, k, reflect):
        measure, theta_init, _A, Gamma_true, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=10 + k
        )
        Y0 = jnp.asarray(theta_init.Y.array)
        phi0 = float(jnp.reshape(theta_init.phi.array, ()))
        Q = _orthogonal(7 + k, k, reflect=reflect)
        # Confirm Q is genuinely in O(K) and on the requested component.
        assert bool(jnp.allclose(Q @ Q.T, jnp.eye(k), atol=1e-10))
        det = float(jnp.linalg.det(Q))
        assert det == pytest.approx(-1.0 if reflect else 1.0, abs=1e-8)

        theta_a = theta_init
        theta_b = _make_params(Y0 @ Q, phi0, k)
        res_a = _estimate(_gauge_invariant_model, measure, theta_a)
        res_b = _estimate(_gauge_invariant_model, measure, theta_b)

        assert bool(res_a.converged) and bool(res_b.converged)

        Aa, _ = res_a.components()
        Ab, _ = res_b.components()
        Ga = Aa @ Aa.T
        Gb = Ab @ Ab.T

        # Same gauge-invariant Gamma (NOT raw Y, which differs by O(K)).
        assert bool(jnp.allclose(Ga, Gb, atol=self.GAMMA_ATOL))
        # Same eigenvalue spectrum.
        assert bool(
            jnp.allclose(
                jnp.linalg.eigvalsh(Ga),
                jnp.linalg.eigvalsh(Gb),
                atol=self.GAMMA_ATOL,
            )
        )
        # Same J-statistic.
        assert float(jnp.asarray(res_a.J_stat)) == pytest.approx(
            float(jnp.asarray(res_b.J_stat)), abs=self.J_ATOL
        )
        # Same J_dof (static; gauge dim does not depend on the start).
        assert res_a.J_dof == res_b.J_dof
        # Same Sigma_theta spectrum (gauge-invariant set of eigenvalues): the
        # ambient SEs are gauge-arbitrary, but the eigenvalue SET is not.
        sa = 0.5 * (res_a.Sigma_theta.array + res_a.Sigma_theta.array.T)
        sb = 0.5 * (res_b.Sigma_theta.array + res_b.Sigma_theta.array.T)
        assert bool(
            jnp.allclose(
                jnp.sort(jnp.linalg.eigvalsh(sa)),
                jnp.sort(jnp.linalg.eigvalsh(sb)),
                atol=1e-6,
            )
        )


# ---------------------------------------------------------------------------
# Gate #2 -- pymanopt TrustRegions cross-check ON THE QUOTIENT.
# ---------------------------------------------------------------------------
@pytest.mark.slow  # pymanopt oracle cross-check: heavy, full-suite/nightly gate (#152)
class TestPymanoptCrossCheck:
    r"""Cross-check emu-gmm's Riemannian-LM against a pymanopt TrustRegions
    solve of the *identical* least-squares problem over
    ``Product(PSDFixedRank(5, K), Euclidean(1))`` (R8/R12/R13/R14/R25/R26).

    Identical-problem discipline (R12): both solvers minimise
    ``cost(Y, phi) = 1/2 (m(Y,phi) - x_bar)' W (m(Y,phi) - x_bar)`` with the
    SAME realised empirical moment target ``x_bar`` and the SAME whitening
    ``W = V_X^{-1}`` (read off the emu-gmm result so the objectives coincide
    exactly). ``J_stat == (m - x_bar)' W (m - x_bar) == 2 * cost`` at the
    optimum.

    Comparison is ON THE QUOTIENT only (R8/R13): ``Gamma_hat = Y_hat Y_hat^T``,
    its eigenvalues, and ``J_stat`` -- NEVER raw ``Y_hat`` (the two solvers
    land on different O(K) orbit representatives). pymanopt's ``Product`` and
    ``PSDFixedRank`` are used directly; the jax AD backend keeps everything in
    float64 (R14/R25). Import-gated via ``pytest.importorskip`` -- pymanopt is
    dev/test-only and is NOT a runtime dependency (R9/R17/R36).
    """

    @pytest.mark.parametrize("k", [2, 3])
    def test_quotient_agrees_with_pymanopt_trust_regions(self, k):
        pytest.importorskip("pymanopt")
        import pymanopt
        from pymanopt.manifolds import Euclidean as PymEuclidean
        from pymanopt.manifolds import Product as PymProduct
        from pymanopt.manifolds import PSDFixedRank as PymPSDFixedRank
        from pymanopt.optimizers import TrustRegions

        noise, n_sim, data_seed = 0.02, 200, 2 + k
        rng = np.random.default_rng(data_seed)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        Gamma_true = A_true @ A_true.T
        phi_true = 0.7
        g_true = Gamma_true[_TRIU[:, 0], _TRIU[:, 1]]
        target = jnp.concatenate([g_true, jnp.reshape(jnp.asarray(phi_true), (1,))])
        M = _moment_count()

        # One FROZEN draw set: both solvers see the SAME realised data.
        draw_noise = noise * jax.random.normal(jax.random.PRNGKey(0), (n_sim, M))
        x_bar = jnp.mean(target[None, :] + draw_noise, axis=0)

        def sampler(key, theta):
            del key, theta
            return target[None, :] + draw_noise

        measure = SyntheticMeasure(
            key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler
        )
        Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
        theta_init = _make_params(Y0, 0.65, k)
        res = _estimate(_gauge_invariant_model, measure, theta_init)
        assert bool(res.converged)
        A_emu, _ = res.components()
        Gamma_emu = A_emu @ A_emu.T
        J_emu = float(jnp.asarray(res.J_stat))

        # Whitening used by emu-gmm (so the pymanopt objective is identical).
        W = jnp.linalg.inv(jnp.asarray(res.V_X.array))

        manifold = PymProduct([PymPSDFixedRank(N, k), PymEuclidean(1)])

        @pymanopt.function.jax(manifold)
        def cost(Y, phi):
            g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
            m = jnp.concatenate([g, phi])
            r = m - x_bar
            return 0.5 * r @ W @ r

        problem = pymanopt.Problem(manifold, cost)
        optimizer = TrustRegions(
            verbosity=0, max_iterations=300, min_gradient_norm=1e-10
        )
        out = optimizer.run(problem, initial_point=[np.asarray(Y0), np.array([0.65])])
        Y_pym, _phi_pym = out.point
        Y_pym = np.asarray(Y_pym, dtype=np.float64)
        assert Y_pym.dtype == np.float64  # float64 backend (R25)
        Gamma_pym = Y_pym @ Y_pym.T
        J_pym = 2.0 * float(out.cost)  # J = r'Wr = 2 * (1/2 r'Wr)

        # Agreement ON THE QUOTIENT only.
        assert bool(jnp.allclose(Gamma_emu, jnp.asarray(Gamma_pym), atol=1e-6))
        assert bool(
            jnp.allclose(
                jnp.linalg.eigvalsh(Gamma_emu),
                jnp.linalg.eigvalsh(jnp.asarray(Gamma_pym)),
                atol=1e-6,
            )
        )
        # J agrees to solver-convergence precision (R26): both at the same min.
        assert J_emu == pytest.approx(J_pym, abs=1e-6)


# ---------------------------------------------------------------------------
# Gate #4 -- J_dof chi-square calibration + gauge-violating adversary.
# ---------------------------------------------------------------------------
class TestJDofChiSquareCalibration:
    r"""Monte Carlo: under the well-specified gauge-invariant DGP the
    over-identification statistic ``J_stat`` is centred on ``chi2_{J_dof}``
    with ``J_dof = M - (total_dimension - total_gauge_dim) = M - (5K+1 -
    K(K-1)/2)`` -- distinguishable from the wrong ``M - total_dimension`` and
    ``M`` (R7/R15/R29).

    Bounded MC (R4): ``N_REPS = 24`` Python-loop replicates per K (NO vmap/pmap;
    one ``estimate()`` per replicate). Each replicate is an INDEPENDENT
    synthetic dataset (distinct ``data_seed`` -> distinct truth + noise key).
    ``noise = 0.05`` keeps ``V`` well-conditioned so the chi-square limit is
    clean. The test asserts ``mean(J)`` is within a normal-CI of ``J_dof``
    (the chi2 mean) and rules OUT the two wrong dof values.

    Adversarial half (R7/R10/R15/R18): a deliberately gauge-VIOLATING residual
    that leaks raw ``Y`` entries (depends on ``Y`` NOT only through ``Y Y^T``)
    destroys the null gauge directions. We verify this directly on the
    Euclidean information matrix ``G' G`` (``G = d model / d theta_flat`` at the
    truth, a legitimate delta-method computation): the gauge-invariant model
    has EXACTLY ``K(K-1)/2`` numerically-null directions, while the
    gauge-violating model has STRICTLY FEWER -- proving the package's
    ``gauge_nullspace_dim == K(K-1)/2`` and the ``pinv_eigvalrule`` drop are
    accounting for genuine, not spurious, gauge nullity.
    """

    N_REPS = 24  # bounded MC; ~50s per K on 8 cores

    @pytest.mark.slow
    @pytest.mark.parametrize("k", [2, 3])
    def test_mean_J_centres_on_J_dof(self, k):
        M = _moment_count()
        D = N * k + 1  # total ambient dimension == total_dimension
        gauge = k * (k - 1) // 2
        J_dof = M - (D - gauge)
        wrong_no_gauge = M - D  # if total_gauge_dim were hard-coded 0
        wrong_M = M  # if dof were not reduced at all

        js = []
        observed_dofs = set()
        for i in range(self.N_REPS):
            measure, theta_init, _A, _G, _phi, _M = _make_dgp(
                k, noise=0.05, n_sim=200, data_seed=5000 + 100 * k + i
            )
            res = _estimate(_gauge_invariant_model, measure, theta_init)
            assert bool(res.converged)
            observed_dofs.add(res.J_dof)
            js.append(float(jnp.asarray(res.J_stat)))
        js = np.asarray(js)

        # The package reports the gauge-corrected dof (NOT hard-coded 0).
        assert observed_dofs == {J_dof}
        assert J_dof != wrong_no_gauge or gauge == 0

        # Mean(J) ~= J_dof (mean of chi2_{J_dof}). SE of the sample mean is
        # sqrt(2 * J_dof / N_REPS); allow a 3.5-sigma band (loose enough to
        # avoid MC flakiness at N_REPS=24, tight enough to exclude the wrong
        # dof values, which differ from J_dof by K(K-1)/2 >= 1).
        se_mean = np.sqrt(2.0 * J_dof / self.N_REPS)
        mean_j = float(js.mean())
        assert abs(mean_j - J_dof) < 3.5 * se_mean
        # And mean(J) is closer to J_dof than to either wrong dof value.
        assert abs(mean_j - J_dof) < abs(mean_j - wrong_no_gauge)
        assert abs(mean_j - J_dof) < abs(mean_j - wrong_M)

    @pytest.mark.parametrize("k", [2, 3])
    def test_gauge_violating_residual_kills_null_directions(self, k):
        # Delta-method information matrix G'G at the truth, computed
        # test-side from the model Jacobian. The gauge-invariant model has
        # exactly K(K-1)/2 null directions; a residual that leaks raw Y
        # entries has strictly fewer (the gauge directions gain curvature).
        rng = np.random.default_rng(k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        flat = jnp.concatenate([A_true.reshape(-1), jnp.array([0.7])])
        gauge = k * (k - 1) // 2

        def moment_good(tf):
            Y = tf[: N * k].reshape(N, k)
            phi = tf[N * k]
            g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
            return jnp.concatenate([g, jnp.reshape(phi, (1,))])

        def moment_bad(tf):
            # Append the raw vec(Y) entries as extra moments -> the residual
            # now depends on Y NOT only through Y Y^T (gauge-violating).
            Y = tf[: N * k].reshape(N, k)
            return jnp.concatenate([moment_good(tf), Y.reshape(-1)])

        def null_count(moment_fn):
            G = jax.jacfwd(moment_fn)(flat)
            info = G.T @ G
            ev = jnp.linalg.eigvalsh(0.5 * (info + info.T))
            mx = float(jnp.max(jnp.abs(ev)))
            return int(jnp.sum(jnp.abs(ev) < 1e-8 * mx))

        n_good = null_count(moment_good)
        n_bad = null_count(moment_bad)
        assert n_good == gauge  # exactly the K(K-1)/2 gauge directions are null
        assert n_bad < gauge  # gauge directions are NO LONGER null


# ---------------------------------------------------------------------------
# Gate #7 -- flatten round-trip preserves leaf shapes / tiles the buffer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3])
class TestFlattenRoundTrip:
    r"""Manifold-aware ``flatten_params -> unflatten_params`` of a Product
    param ``(A:(5,K), phi:(1,))`` reconstructs each leaf with its exact shape;
    the LeafSpec block widths tile the flat buffer exactly (R20/R34).

    Guards the Phase-1/2 spec builder: a wrong ``offset`` / ``ambient_shape``
    silently corrupts block boundaries ('looks like it ran').
    """

    def test_block_layout_and_round_trip(self, k):
        rng = np.random.default_rng(1 + k)
        A_true = jnp.asarray(rng.normal(size=(N, k)))
        params = _make_params(A_true, 0.5, k)
        spec = manifold_spec_from_params(params)
        flat, treedef, fspec = flatten_params_with_spec(params)

        # Total dimensions agree and the buffer length is 5K + 1.
        assert fspec.total_ambient_dim == N * k + 1
        assert spec.total_dimension == N * k + 1
        assert int(flat.shape[0]) == N * k + 1

        # Offsets / sizes tile the buffer with no gap or overlap.
        offsets = [ls.offset for ls in spec.leaf_specs]
        sizes = [int(np.prod(ls.ambient_shape)) for ls in spec.leaf_specs]
        assert offsets == [0, N * k]  # Y at 0, phi at N*K
        assert sizes == [N * k, 1]
        assert sum(sizes) == int(flat.shape[0])
        # Each ambient_shape matches its leaf.
        assert tuple(spec.leaf_specs[0].ambient_shape) == (N, k)
        assert int(np.prod(spec.leaf_specs[1].ambient_shape)) == 1

        # Round-trip through the manifold-aware unflatten preserves shapes
        # AND values.
        back = unflatten_params(flat, treedef, manifold_spec=spec)
        assert jnp.asarray(back.Y.array).shape == (N, k)
        assert jnp.asarray(back.phi.array).shape == (1,)
        assert bool(jnp.allclose(jnp.asarray(back.Y.array), A_true))
        assert bool(jnp.allclose(jnp.asarray(back.phi.array), 0.5))

    def test_gauge_dim_flows_from_spec(self, k):
        # total_gauge_dim is K(K-1)/2 (PSDFixedRank) + 0 (Euclidean), summed
        # by the spec builder -- NOT hard-coded (R7/R14).
        params = _make_params(jnp.ones((N, k)), 0.5, k)
        spec = manifold_spec_from_params(params)
        assert spec.total_gauge_dim == k * (k - 1) // 2
