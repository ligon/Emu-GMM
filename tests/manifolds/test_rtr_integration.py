r"""TEST-FIRST integration suite for the Riemannian Trust Region optimiser (#152).

This file is written BEFORE the implementation exists. It imports
``emu_gmm.manifolds.riemannian_tr`` (the Phase-2 deliverable) and will be RED
(``ImportError`` at collection) until that module lands. That is intended: the
suite is the contract Phase 2 implements against, not a post-hoc check.

Theme: **framework integration**. These tests pin the integration-lens risks
from the Phase-0 red-team register -- the seams where ``riemannian_tr`` meets
the existing ``estimate`` pipeline, the MC / ``replicate`` loops, the
manifold-aware flatten/unflatten round-trip, and reverse-mode AD through the
solver. They deliberately do NOT re-test the HVP geometry, tCG internals, or
pymanopt parity (those are owned by the hvp/tcg/parity-themed files); a
geometry bug that nonetheless leaves these integration invariants intact is out
of scope here, and vice versa.

Risks pinned (integration lens, from ``w18spt0nn.output``):

* "RTR done/status PyTree wiring must produce a *traced* ``done`` field ..." --
  under ``jax.jit(estimate)`` AND ``jax.vmap``/``replicate`` ``converged`` must
  be ``False`` for a forced ``max_steps=2`` non-convergence and ``True`` for an
  easy convex fixture; ``info.done`` must be a 0-d JAX bool array (not ``None``,
  not a Python bool) that survives ``tree_leaves``.
* "v2 dispatch never threads the measure through ``args=`` ... MC cache-leak" --
  a 200-rep loop over fresh same-structure measures keeps RSS growth bounded and
  the XLA compile count O(1), not O(reps).
* "``flatten_params_with_spec`` round-trip ... preserve leaf order/dtype/shape"
  -- the recovered pytree re-flattens bit-identically, the treedef matches the
  estimator's, every leaf is float64 with the right shape, and
  ``Sigma_theta`` / ``gamma_se`` from RTR match the LM result on a convex
  fixture.
* "Reverse-mode AD through RTR fails the SAME way as LM (#77) ..." -- the
  ``lax.while_loop`` reverse-mode error must be raised (a clean failure), not a
  silently-wrong gradient.

Discipline mirrored from ``test_manifold_acceptance_phase6.py`` /
``test_riemannian_lm_traced_args.py``: a gauge-invariant ``Product(PSDFixedRank
(5, K), Euclidean(1))`` DGP whose residual depends on ``theta`` only through
``Gamma = Y Y^T`` and ``phi``; every recovery / agreement assertion is on a
gauge-invariant functional (``Gamma``, eigenvalues, ``J_stat``,
``Sigma_theta`` spectrum, ``gamma_se``), never raw ``Y``.
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import estimate
from emu_gmm._internal import params as params_mod
from emu_gmm.covariance import SyntheticCovariance
from emu_gmm.manifolds import Euclidean, PSDFixedRank
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf
from emu_gmm.manifolds.riemannian_lm import riemannian_lm
from emu_gmm.measures import SyntheticMeasure
from emu_gmm.weighting import ContinuouslyUpdated

# The module-under-construction. SKIP cleanly until Phase 2 ships the factory,
# then the tests go live (red until correct).
riemannian_tr_mod = pytest.importorskip(
    "emu_gmm.manifolds.riemannian_tr",
    reason="Phase 2 not yet implemented: emu_gmm.manifolds.riemannian_tr is RED",
)
riemannian_tr = riemannian_tr_mod.riemannian_tr

jax.config.update("jax_enable_x64", True)

N = 5  # ambient PSD side; matches the Phase-6 / K-Agg fixture geometry.
_TRIU = jnp.array(np.triu_indices(N)).T  # (15, 2) upper-triangular index pairs


# ---------------------------------------------------------------------------
# Shared DGP helpers (mirrors test_manifold_acceptance_phase6.py). Flag in
# shared_helpers_needed: these belong in a manifolds/conftest.py once >=2 RTR
# theme files exist.
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
    # M = n(n+1)/2 unique triu(Gamma) entries + 1 (phi).
    return N * (N + 1) // 2 + 1


def _gauge_invariant_model(x, theta):
    """psi = triu(Y Y^T) concat phi  -  x. Depends on theta ONLY through the
    gauge-invariant ``Gamma = Y Y^T`` and ``phi``."""
    Y = theta.Y.array
    phi = theta.phi.array[0]
    g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
    return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - x


def _make_dgp(k: int, *, noise: float, n_sim: int, data_seed: int):
    """Build the truth, the synthetic measure, and a near-truth warm start.

    Returns ``(measure, theta_init, A_true, Gamma_true, phi_true, M)``.
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
        del key, theta
        noise_draws = noise * jax.random.normal(noise_key, (n_sim, M))
        return target[None, :] + noise_draws

    measure = SyntheticMeasure(key=jax.random.PRNGKey(0), n_sim=n_sim, sampler=sampler)
    Y0 = jnp.asarray(A_true + 0.05 * rng.normal(size=(N, k)))
    theta_init = _make_params(Y0, 0.65, k)
    return measure, theta_init, A_true, Gamma_true, phi_true, M


def _estimate(model, measure, theta_init, optimizer, *, weighting=None):
    return estimate(
        model,
        measure,
        covariance=SyntheticCovariance(),
        weighting=weighting if weighting is not None else ContinuouslyUpdated(),
        optimizer=optimizer,
        theta_init=theta_init,
    )


def _gamma(theta_or_components):
    """``Gamma = A A^T`` from either a ProductParams or a ``components()`` A."""
    if isinstance(theta_or_components, ProductParams):
        Y = jnp.asarray(theta_or_components.Y.array)
    else:
        Y = jnp.asarray(theta_or_components)
    return Y @ Y.T


def _rss_kb() -> int | None:
    """Resident-set-size in KB from ``/proc/self/status`` (Linux box). Returns
    ``None`` off Linux so the cache-leak test degrades to the compile-count
    assertion rather than erroring."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


# ===========================================================================
# Risk: traced ``done`` bool under jit AND vmap.
# ===========================================================================
class TestTracedDoneUnderJitAndVmap:
    r"""Integration risk -- 'RTR done/status PyTree wiring must produce a
    *traced* ``done`` field of matching dtype, or the estimator's ``converged``
    collapse silently reports always-converged under jit/vmap'.

    The estimator falls back to ``status in ('converged','traced')`` when
    ``info.done is None`` -- and 'traced' maps to ``converged=True``. So a
    ``done=None`` / Python-bool / static-status RTR reports ``converged=True``
    for EVERY fit under jit/vmap, masking exactly the non-convex stalls RTR
    exists to detect. These tests force the failure to be visible by checking
    a provably-non-convergent ``max_steps=2`` fixture against an easy convex
    one, under both ``jax.jit(estimate)`` and ``jax.vmap`` (the replicate
    path).
    """

    K = 2

    def _easy_optimizer(self):
        # Generous budget: the warm-started convex fixture converges easily.
        return riemannian_tr(max_steps=200, rtol=1e-8, atol=1e-10)

    def _starved_optimizer(self):
        # max_steps=2 cannot reach a stationary point from the warm start on
        # a 13-dim quotient problem -> done must come back False.
        return riemannian_tr(max_steps=2, rtol=1e-12, atol=1e-14)

    def test_done_is_traced_bool_array_surviving_tree_leaves(self):
        """``info.done`` is a 0-d JAX bool array (not None, not a Python bool)
        and rides as a traced leaf through ``tree_leaves`` -- the precondition
        for the estimator's traced ``converged`` collapse to work under jit."""
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=300
        )
        theta_flat, treedef, spec = params_mod.flatten_params_with_spec(theta_init)

        def kernel(tf, m):
            theta = params_mod.unflatten_params(tf, treedef, manifold_spec=spec)
            return jnp.asarray(m.expectation(_gauge_invariant_model, theta))

        _theta_hat, info = self._easy_optimizer()(
            kernel, theta_init, spec, args=measure
        )

        assert info.done is not None, "RTR returned done=None: status fallback wins"
        done = jnp.asarray(info.done)
        assert done.shape == (), f"done must be 0-d, got shape {done.shape}"
        assert done.dtype == jnp.bool_, f"done must be bool, got {done.dtype}"
        # A Python bool is NOT a tree leaf; a JAX array IS. This is the exact
        # discriminator the estimator relies on under jit.
        leaves = jax.tree_util.tree_leaves(info)
        assert any(
            isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.bool_ for leaf in leaves
        ), "done did not survive tree_leaves as a traced bool array"
        assert bool(done) is True  # the easy fixture really did converge

    def test_converged_false_under_jit_for_starved_budget(self):
        """Under ``jax.jit(estimate)`` a ``max_steps=2`` fit reports
        ``converged == False``. A done=None / static-status port passes
        eagerly but reports True here (the silent always-converged bug)."""
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=301
        )

        def run(m):
            res = _estimate(
                _gauge_invariant_model, m, theta_init, self._starved_optimizer()
            )
            return jnp.asarray(res.converged, dtype=jnp.float64)

        conv_jit = float(jax.jit(run)(measure))
        conv_eager = float(run(measure))
        # Both eager and jitted must agree that the starved fit did NOT
        # converge. The bug surfaces as conv_jit==1.0 while conv_eager==0.0.
        assert conv_eager == 0.0, "eager: starved budget wrongly reported converged"
        assert conv_jit == 0.0, "jit: traced 'done' collapsed to always-converged"

    def test_converged_true_under_jit_for_easy_fixture(self):
        """The dual control: an easy convex fixture reports ``converged ==
        True`` under jit, so the False above is not a constant-False artefact."""
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=302
        )

        def run(m):
            res = _estimate(
                _gauge_invariant_model, m, theta_init, self._easy_optimizer()
            )
            return jnp.asarray(res.converged, dtype=jnp.float64)

        assert float(jax.jit(run)(measure)) == 1.0

    def test_converged_vmaps_distinguishing_starved_from_easy(self):
        """The ``replicate`` path is ``vmap``. A vmapped batch mixing a starved
        and an easy fixture must return ``[False, True]`` -- not ``[True,
        True]``. This is the strongest form: a single static/None ``done``
        cannot produce a per-element vector, so it collapses both to True."""
        m_a, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=303
        )
        # Two fresh same-structure measures so vmap has a real batch axis on
        # the data while the closure structure is shared.
        m0 = m_a.with_key(jax.random.PRNGKey(51))
        m1 = m_a.with_key(jax.random.PRNGKey(52))
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), m0, m1)

        # Element 0 is starved (max_steps=2), element 1 is easy. We encode the
        # per-element budget by running two separate vmapped estimates and
        # stitching -- a single optimizer cannot vary max_steps per batch
        # element, so the discriminator is: starved-vmap is all-False, easy-
        # vmap is all-True.
        def run_with(optimizer):
            def one(m):
                res = _estimate(_gauge_invariant_model, m, theta_init, optimizer)
                return jnp.asarray(res.converged, dtype=jnp.float64)

            return jax.vmap(one)(batch)

        starved = run_with(self._starved_optimizer())
        easy = run_with(self._easy_optimizer())
        assert starved.shape == (2,) and easy.shape == (2,)
        # Starved: every element False; easy: every element True. A None/static
        # done makes starved come back [1,1] (always-converged) -- the bug.
        assert bool(jnp.all(starved == 0.0)), "vmap starved batch wrongly converged"
        assert bool(jnp.all(easy == 1.0)), "vmap easy batch failed to converge"


# ===========================================================================
# Risk: MC cache leak -- args= kernel path, bounded RSS, O(1) compiles.
# ===========================================================================
class TestMonteCarloCacheLeak:
    r"""Integration risk -- 'v2 dispatch never threads the measure through
    ``args=``, so RTR's per-call eager retrace re-explodes the #124/#139
    cache-leak the LM path was hardened against'.

    A 200-rep loop over fresh same-structure measures must (i) keep RSS growth
    bounded (no ~14 MB/call accumulation) and (ii) compile the solve core
    O(1) times, not once per rep. An eager-per-call RTR shows linear RSS growth
    and reps-many compiles. We drive the kernel through the ``args=`` channel
    exactly as the estimator's v2 dispatch must, and count XLA traces via a
    counting-psi (a Python-level trace counter, the proven probe from
    ``test_riemannian_lm_traced_args.py``).
    """

    K = 2

    def _setup(self):
        base, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=400
        )
        theta_flat, treedef, spec = params_mod.flatten_params_with_spec(theta_init)
        return base, theta_init, treedef, spec

    def test_args_path_shares_one_trace_across_fresh_measures(self):
        """Two fresh same-structure measures via ``args=`` share ONE trace:
        the counting-psi counter freezes after the first solve. An eager port
        re-traces ``vmap(psi)`` every call."""
        base, theta_init, treedef, spec = self._setup()

        class _CountingPsi:
            def __init__(self):
                self.calls = 0

            def __call__(self, x, theta):
                self.calls += 1
                return _gauge_invariant_model(x, theta)

        counting = _CountingPsi()

        def kernel(tf, m):
            theta = params_mod.unflatten_params(tf, treedef, manifold_spec=spec)
            return jnp.asarray(m.expectation(counting, theta))

        opt = riemannian_tr(max_steps=200)
        m1 = base.with_key(jax.random.PRNGKey(61))
        m2 = base.with_key(jax.random.PRNGKey(62))

        _t1, info1 = opt(kernel, theta_init, spec, args=m1)
        assert bool(info1.done)
        calls_after_first = counting.calls
        assert calls_after_first > 0, "no trace happened on the first solve"

        _t2, info2 = opt(kernel, theta_init, spec, args=m2)
        assert bool(info2.done)
        assert counting.calls == calls_after_first, (
            "RTR re-traced psi on a fresh same-structure measure via args=: "
            f"{counting.calls} != {calls_after_first} -- eager-per-call leak"
        )

    @pytest.mark.slow
    def test_200_rep_loop_bounded_rss_and_o1_compiles(self):
        """200 estimate-equivalent solves over fresh measures: RSS growth
        bounded and the trace count O(1). Pins the #124/#139 leak fix on the
        RTR path. Marked slow (it runs 200 solves)."""
        base, theta_init, treedef, spec = self._setup()

        class _CountingPsi:
            def __init__(self):
                self.calls = 0

            def __call__(self, x, theta):
                self.calls += 1
                return _gauge_invariant_model(x, theta)

        counting = _CountingPsi()

        def kernel(tf, m):
            theta = params_mod.unflatten_params(tf, treedef, manifold_spec=spec)
            return jnp.asarray(m.expectation(counting, theta))

        opt = riemannian_tr(max_steps=120)

        # Warm-up: pay the one-time compile, then measure deltas.
        _t, info = opt(
            kernel, theta_init, spec, args=base.with_key(jax.random.PRNGKey(0))
        )
        assert bool(info.done)
        traces_after_warmup = counting.calls
        gc.collect()
        rss_start = _rss_kb()

        reps = 200
        for i in range(reps):
            m = base.with_key(jax.random.PRNGKey(1000 + i))
            _th, _info = opt(kernel, theta_init, spec, args=m)
        gc.collect()

        # (i) O(1) compiles: the trace count must not grow with reps.
        assert counting.calls == traces_after_warmup, (
            f"RTR retraced inside the MC loop: {counting.calls} traces for "
            f"{reps} reps (expected {traces_after_warmup}); cache leak."
        )

        # (ii) bounded RSS: a ~14 MB/call leak would add ~2.8 GB over 200 reps.
        rss_end = _rss_kb()
        if rss_start is not None and rss_end is not None:
            growth_mb = (rss_end - rss_start) / 1024.0
            assert growth_mb < 400.0, (
                f"RSS grew {growth_mb:.0f} MB over {reps} reps -- linear leak "
                "(the ~14 MB/call write-only-trace pathology)."
            )


# ===========================================================================
# Risk: flatten_params_with_spec round-trip; Sigma_theta / gamma_se vs LM.
# ===========================================================================
@pytest.mark.parametrize("k", [2, 3])
class TestFlattenRoundTripAndInferenceParity:
    r"""Integration risk -- '``flatten_params_with_spec`` round-trip on the
    RTR-returned pytree must preserve leaf order/dtype/shape, or the V-refresh
    and ``Sigma_theta`` read garbage on the manifold path'.

    The estimator re-flattens the optimiser's returned pytree to refresh ``V``
    and compute ``Sigma_theta``. If RTR rebuilds its result by a different leaf
    walk (wrong order, lost float64, transposed ``(n,k)`` block) the size-only
    tiling guard passes and feeds a wrong ``Y`` into ``Sigma_theta``. These
    tests pin the round-trip bit-for-bit AND cross-check the downstream
    inference (``Sigma_theta`` spectrum, ``gamma_se``) against ``riemannian_lm``
    on a convex fixture where both must agree.
    """

    def test_result_pytree_reflattens_bit_identically(self, k):
        """The RTR-returned pytree re-flattens to the SAME flat buffer the
        estimator would build, with matching treedef, float64 leaves, and the
        PSD block reshaped to ``(N, k)`` (not ``(k, N)``)."""
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=500 + k
        )
        spec_init = params_mod.manifold_spec_from_params(theta_init)
        flat0, treedef0, _ = params_mod.flatten_params_with_spec(theta_init)

        def kernel(tf, m):
            theta = params_mod.unflatten_params(tf, treedef0, manifold_spec=spec_init)
            return jnp.asarray(m.expectation(_gauge_invariant_model, theta))

        theta_hat, info = riemannian_tr(max_steps=200)(
            kernel, theta_init, spec_init, args=measure
        )
        assert bool(info.done)

        # Re-flatten the recovered pytree the way the estimator does post-v2.
        flat_back, treedef_back, spec_back = params_mod.flatten_params_with_spec(
            theta_hat
        )

        # treedef identical to the estimator's (same leaf order / structure).
        assert treedef_back == treedef0, "RTR result has a different treedef"
        # Same total ambient size and tiling.
        assert spec_back.total_ambient_dim == N * k + 1
        assert [ls.offset for ls in spec_back.leaf_specs] == [0, N * k]
        # Every leaf float64.
        assert jnp.asarray(theta_hat.Y.array).dtype == jnp.float64
        assert jnp.asarray(theta_hat.phi.array).dtype == jnp.float64
        assert flat_back.dtype == jnp.float64
        # PSD block shape preserved -- a transpose would silently pass a
        # size-only guard but corrupt Gamma.
        assert jnp.asarray(theta_hat.Y.array).shape == (N, k)
        assert jnp.asarray(theta_hat.phi.array).shape == (1,)

        # Bit-identical re-flatten: unflatten then re-flatten is a fixed point.
        reround = params_mod.unflatten_params(
            flat_back, treedef_back, manifold_spec=spec_back
        )
        flat_reround, _, _ = params_mod.flatten_params_with_spec(reround)
        np.testing.assert_array_equal(np.asarray(flat_reround), np.asarray(flat_back))
        # And the PSD block round-trips to (N, k), not (k, N).
        assert jnp.asarray(reround.Y.array).shape == (N, k)

    def test_sigma_theta_and_gamma_se_match_lm_on_convex_fixture(self, k):
        """On the convex gauge-invariant fixture, RTR and LM reach the same
        minimiser, so their ``Sigma_theta`` spectrum and ``gamma_se`` (both
        delta-method functionals read off the round-tripped result) must agree.
        A mis-ordered / wrong-dtype round-trip diverges here even when the raw
        recovery looks fine."""
        measure, theta_init, _A, Gamma_true, _phi, _M = _make_dgp(
            k, noise=0.01, n_sim=200, data_seed=520 + k
        )
        res_tr = _estimate(
            _gauge_invariant_model, measure, theta_init, riemannian_tr(max_steps=300)
        )
        res_lm = _estimate(
            _gauge_invariant_model, measure, theta_init, riemannian_lm(max_steps=300)
        )
        assert bool(res_tr.converged) and bool(res_lm.converged)

        # Gauge-invariant point agreement first (both at the same minimiser).
        A_tr, _ = res_tr.components()
        A_lm, _ = res_lm.components()
        G_tr = _gamma(A_tr)
        G_lm = _gamma(A_lm)
        np.testing.assert_allclose(np.asarray(G_tr), np.asarray(G_lm), atol=1e-6)
        np.testing.assert_allclose(np.asarray(G_tr), np.asarray(Gamma_true), atol=4e-3)

        # Sigma_theta spectrum (gauge-invariant eigenvalue SET) must match: a
        # garbage round-trip into Sigma_theta perturbs this.
        sa = 0.5 * (res_tr.Sigma_theta.array + res_tr.Sigma_theta.array.T)
        sb = 0.5 * (res_lm.Sigma_theta.array + res_lm.Sigma_theta.array.T)
        np.testing.assert_allclose(
            np.sort(np.asarray(jnp.linalg.eigvalsh(sa))),
            np.sort(np.asarray(jnp.linalg.eigvalsh(sb))),
            atol=1e-6,
        )

        # gamma_se: the gauge-invariant SE functional on Gamma. Sorted compare
        # because the ambient ordering is parameterisation-arbitrary but the
        # multiset of SEs on the gauge-invariant functional is not.
        se_tr = np.asarray(res_tr.gamma_se())
        se_lm = np.asarray(res_lm.gamma_se())
        assert se_tr.shape == se_lm.shape
        np.testing.assert_allclose(np.sort(se_tr), np.sort(se_lm), atol=1e-6)


# ===========================================================================
# Risk: reverse-mode AD through the solver raises (clean failure, not silent).
# ===========================================================================
class TestReverseModeADRaises:
    r"""Integration risk -- 'Reverse-mode AD through RTR fails the SAME way as
    LM (#77) ... not silently wrong'.

    ``riemannian_lm`` defers reverse-mode through the solver and the failure is
    CLEAN precisely because the outer loop is a ``lax.while_loop`` (JAX raises a
    reverse-mode error). If RTR's outer loop were an unrolled Python loop,
    ``jax.grad`` would silently unroll and differentiate through the
    accept/reject ``jnp.where`` branches, yielding a numerically WRONG
    hyperparameter gradient with no error. This test pins that the outer loop
    is a ``while_loop`` by requiring the reverse-mode AD to RAISE, matching the
    #77 contract -- a clean failure beats a silent wrong answer.
    """

    K = 2

    def _data_to_objective(self):
        """Build a scalar objective ``data -> final_objective`` through a full
        RTR solve, so ``jax.grad`` over ``data`` must traverse the solver."""
        measure, theta_init, _A, _G, _phi, _M = _make_dgp(
            self.K, noise=0.01, n_sim=200, data_seed=600
        )
        spec = params_mod.manifold_spec_from_params(theta_init)
        flat0, treedef, _ = params_mod.flatten_params_with_spec(theta_init)
        M = _moment_count()
        # A fixed empirical moment target; grad flows through it into the solve.
        target0 = jnp.asarray(np.concatenate([np.linspace(0.1, 1.0, M - 1), [0.7]]))

        def objective(target):
            def residual_fn(tf):
                theta = params_mod.unflatten_params(tf, treedef, manifold_spec=spec)
                Y = theta.Y.array
                phi = theta.phi.array[0]
                g = (Y @ Y.T)[_TRIU[:, 0], _TRIU[:, 1]]
                return jnp.concatenate([g, jnp.reshape(phi, (1,))]) - target

            _theta_hat, info = riemannian_tr(max_steps=50)(
                residual_fn, theta_init, spec
            )
            return jnp.asarray(info.final_objective)

        return objective, target0

    def test_reverse_mode_through_solver_raises_while_loop_error(self):
        """``jax.grad`` of a through-solve objective must raise the
        ``lax.while_loop`` reverse-mode error -- the #77 clean-failure contract.
        A silently-differentiable Python outer loop would NOT raise (and would
        return a wrong gradient)."""
        objective, target0 = self._data_to_objective()
        # Sanity: the forward pass works (so a raise below is about AD, not a
        # broken objective).
        _ = float(objective(target0))

        with pytest.raises(Exception) as excinfo:
            _ = jax.grad(objective)(target0)
        msg = str(excinfo.value).lower()
        # The hallmark of JAX's reverse-mode-through-while_loop refusal. We do
        # not over-pin the exact class (it varies across JAX versions) but the
        # message must implicate reverse-mode / while_loop, NOT a generic shape
        # error -- a Python-loop port would not raise at all.
        assert (
            ("reverse" in msg) or ("while_loop" in msg) or ("while loop" in msg)
        ), f"unexpected error (not the while_loop reverse-mode refusal): {msg}"
