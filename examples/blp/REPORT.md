# Estimating BLP with emu-gmm: a critical comparison against pyblp

**Goal.** Use `emu-gmm` to estimate a Berry–Levinsohn–Pakes (1995)
random-coefficients logit demand model and compare it, quantity by quantity,
against `pyblp` (Conlon & Gortmaker 2020) — the reference Python implementation.
Because `pyblp` is an independent, widely-used, separately-tested codebase,
reproducing its estimates on a real, non-trivial GMM problem is strong evidence
that `emu-gmm`'s core machinery is correct.

**Headline result.** On all three nested specifications of `pyblp`'s Nevo "fake
cereal" tutorial data, `emu-gmm` reproduces `pyblp`'s point estimates, GMM
objective, J-statistic, **and** standard errors to **6–8 significant digits** —
both when warm-started at `pyblp`'s solution and when **cold-started from
`pyblp`'s documented initial values** and left to find the optimum on its own.

---

## 1. How BLP maps onto emu-gmm's interface

BLP's GMM rests on the demand-side moment condition

```
E[ Z_jt · xi_jt(theta) ] = 0,
```

where `xi_jt` is the structural demand error for product `j` in market `t` and
`Z` are instruments. The mapping onto `emu-gmm`'s per-observation
`psi(x, theta) -> R^M` interface:

| BLP object | emu-gmm |
|---|---|
| observation | one **product-market** `(j,t)` row (N = 2256) |
| moment dim `M` | number of instruments (20) |
| `psi(x_n, theta)` | `Z_n · xi_n(theta)` |
| sample moment `g = (1/N) Σ Z_jt xi_jt` | emu's `measure.expectation` (mean over rows) |
| optimal weight `W` | emu whitening with `V_X = W^{-1}` (objective `g'V_X^{-1}g`) |

Three design points carry the work:

1. **The share-inversion contraction is an implicitly-differentiated fixed
   point.** Given the nonlinear parameters `theta2 = (sigma, pi)`, the mean
   utilities `delta_t(theta2)` solve `s(delta_t, theta2) = S_obs_t` per market.
   We solve the Berry contraction with a `lax.while_loop` and attach the exact
   forward-mode derivative via the implicit function theorem
   (`@jax.custom_jvp`): `ddelta = -(ds/ddelta)^{-1} (ds/dtheta2)`. Forward mode
   is exactly what emu's `jacfwd` (`G = measure.jacobian`) and the optimistix LM
   solver consume. *Validated:* the custom JVP matches finite differences to
   ~1e-9 and the contraction recovers `pyblp`'s `delta` to machine precision
   (2.6e-15 share recovery) under float64.

2. **Brand fixed effects cost nothing.** `pyblp`'s `absorb='C(product_ids)'`
   partials 24 brand FE out of the moment. Because the within-brand projection
   `A` is symmetric idempotent, `Z'A(delta - X1 b) = (AZ)'(delta - X1 b)`, so
   **demeaning the instruments alone** (`ZD_tilde`) reproduces the absorbed
   moment with raw `delta`/`prices` and no explicit dummies — the FE are
   annihilated because `AZ` kills any brand-constant term. The only linear
   parameter that survives is the price coefficient.

3. **`beta` is estimated jointly, not concentrated out.** `pyblp` profiles the
   linear parameters; emu treats `(beta, sigma, pi)` as one joint parameter
   vector. The two are the same GMM problem (the profiled optimum is the joint
   optimum), and joint estimation is the clean fit to emu's per-observation
   residual — `psi` never needs a cross-market operation.

The observation being the *product-market* (not the market) is deliberate: it
makes emu's `IIDCovariance` ≡ `pyblp`'s robust covariance and
`ClusteredCovariance(market)` ≡ `pyblp` clustered-by-market, and it matches
`pyblp`'s moment structure 1:1. The per-market contraction does not depend on
the per-row index, so `vmap` traces it **once** per evaluation and gathers it to
the 2256 rows — it stays efficient.

---

## 2. Apples-to-apples alignment

To isolate any discrepancy to the parts emu *owns* (moment construction,
criterion, optimiser, inference) rather than to data wrangling or the
weight-matrix estimation step, the comparison feeds emu **`pyblp`'s own internal
arrays** — the instruments `ZD`, characteristics `X1`/`X2`, integration nodes,
demographics, weights, and shares, extracted directly from the solved
`pyblp.Problem` (`build_reference.py`).

Two scaling/convention facts were pinned empirically and matter for an exact
match:

- **`pyblp`'s reported objective = `N · g'Wg`** (g = mean moment). Equivalently,
  the moment-vector covariance implied by `pyblp`'s weight is `V_X = W^{-1}/N`.
  Feeding emu that `V_X` as both the fixed weight and the covariance makes emu's
  own formulas (`J = m'V_X^{-1}m`, `Sigma_theta = (G'V_X^{-1}G)^{-1}`) reproduce
  `pyblp`'s objective, J, and SE.
- **`pyblp` computes SEs from the weight it *re-optimises* at `theta_hat`
  (`updated_W`)**, not the objective weight (`results.W`). Using `updated_W` for
  the covariance closes the SE gap from ~1% to the 8th digit. (This is a genuine
  finding about how to compare correctly, surfaced by emu's strict
  Measure/Covariance separation — see §4.)

Other conventions: `sigma`'s sign is not identified (it is a standard
deviation), so we compare `|sigma|`; `pyblp` fixes the zero entries of `sigma`
(off-diagonal) and `pi` at zero and estimates only the nonzero ones (9 free `pi`
entries in the Nevo pattern), so emu estimates only those free entries,
scattering them into a zero template.

---

## 3. Results

Each spec is `pyblp`'s and emu's solution of the **identical** GMM problem.
"Warm" = emu started at `pyblp`'s estimate; "cold" = emu started at `pyblp`'s
*documented Nevo initial values* with `beta` profiled in, then optimised
independently (`optimistix_lm`).

### Stage 1 — plain logit (closed-form delta, K=1, dof=19)

| quantity | pyblp | emu (warm) |
|---|---|---|
| `beta_price` | −30.04710289 | −30.04710289 |
| objective | 187.4555 | 187.4555 |
| `beta` SE | 1.00858874 | 1.00858874 |

### Stage 2 — random coefficients, no demographics (K=5, dof=15)

| quantity | pyblp | emu (warm) | emu (cold) |
|---|---|---|---|
| `beta_price` | −30.65209435 | −30.65209435 | −30.65209518 |
| `|sigma|` | [0.19682, 2.44203, 0.01027, 0.17978] | match (8 dig) | match (7 dig) |
| objective | 172.2787 | 172.2787 | 172.2787 |
| `beta` SE | 1.11755193 | 1.11755193 | 1.11755195 |
| iterations | (BFGS) | 2 | 122 |

### Stage 3 — RC + demographics (the headline tutorial spec; K=14, dof=6)

| quantity | pyblp | emu (warm) | emu (cold) |
|---|---|---|---|
| `beta_price` | −60.34397417 | −60.34397417 | −60.34397443 |
| `|sigma|` | [0.54496, 3.06526, 0.00505, 0.07919] | match (8 dig) | match (8 dig) |
| `pi` (9 free) | — | max\|Δ\| 2.4e-11 | max\|Δ\| 5.2e-6 |
| objective | 6.128080 | 6.128080 | 6.128080 |
| `beta` SE | 13.74854692 | 13.74854692 | 13.748547 |
| J / p-value | 6.1281 / 0.409 | 6.1281 / 0.409 | 6.1281 / 0.409 |
| iterations | (BFGS) | 9 | 117 |

The cold-start agreement is the decisive line: from `pyblp`'s documented
starting values, emu's Levenberg–Marquardt optimiser **independently** lands on
`pyblp`'s optimum to 6–8 significant digits across every parameter, the GMM
objective, the J-statistic, and the standard errors. The residual differences
(~1e-6 to 1e-7) are optimiser-tolerance noise: the objective is identical to six
digits.

---

## 4. Performance (indicative)

Wall-clock for the headline `rc_demographics` spec, 4-core CPU, float64
(`benchmark.py`):

| run | time | notes |
|---|---|---|
| **pyblp** solve (BFGS, 2-step) | **44.2 s** | its own nested fixed point + analytic gradients/SEs |
| emu warm, incl. JIT compile | 9.9 s | started at pyblp's `theta_hat` (~9 LM iters) |
| emu warm, compiled (steady) | 4.7 s | |
| emu **cold** from Nevo init, compiled | **11.7 s** | finds the optimum independently (117 LM iters) |

The like-for-like comparison — both finding the optimum from scratch — is emu
~11.7 s vs pyblp ~44 s, **roughly 3.8× faster**; warm steady-state is ~4.7 s.

**This is indicative, not a controlled benchmark.** Caveats: (a) different
optimisers — pyblp uses gradient BFGS (~92 objective evaluations in the
reference run), emu uses Gauss–Newton LM; (b) emu's edge comes from JAX JIT plus
the contraction vectorised across all 94 markets in one `vmap`, and the first
call pays ~5 s of compilation that pyblp never incurs; (c) pyblp does extra work
in that 44 s (two full GMM steps, analytic Jacobians, richer diagnostics); (d)
4 cores only (sandbox limit). Read it as "emu is in the same ballpark or
somewhat faster," not as a precise speedup factor.

## 5. Critical assessment

**What this validates about emu-gmm.** The full estimation pipeline is correct
on a real, independently-implemented, non-trivial GMM problem:

- the moment construction and per-coordinate `N_j` scaling convention
  (CLAUDE.md commitment 9) — emu's `J = m'V_X^{-1}m` with no explicit `N`
  reproduces `pyblp`'s `N·g'Wg` exactly;
- the criterion and the Cholesky-whitening objective — the minimised objective
  agrees to 6 digits;
- the LM optimiser — it finds `pyblp`'s optimum from a cold start, not merely
  confirms a fixed point;
- the information-matrix SE formula `(G'V^{-1}G)^{-1}` (commitment 5) — SEs agree
  to 8 digits **given the same `V`**, including through AD of the BLP contraction
  (`G` is computed by `jacfwd` through the implicitly-differentiated fixed
  point, never a numerical Hessian);
- the manifold-aware flatten/unflatten path — parameters ride as `ManifoldLeaf`
  (Euclidean) blocks and round-trip through `estimate` cleanly.

**A genuine finding, not a bug.** emu's architectural separation of `Measure`
from `CovarianceStrategy` (commitment 1) means emu does *not* infer "the right"
moment covariance from the weight matrix the way a monolithic BLP routine does —
the user picks it. Matching `pyblp`'s reported SEs therefore required two
explicit choices that a `pyblp` user never sees: (a) the moment covariance
implied by `pyblp`'s objective is `W^{-1}/N`, and (b) `pyblp`'s SEs use its
*re-optimised* `updated_W`, not the objective weight. emu's own self-contained
robust covariance (`IIDCovariance`, the textbook `(1/N)Σ g g'`) gives a SE of
~1.55 for the logit price coefficient versus `pyblp`'s 1.009 — **both are
defensible**; they are different covariance estimators, and emu reproduces
`pyblp`'s number exactly when handed `pyblp`'s covariance. This is the
separation working as designed, and it is the kind of thing a consumer should
delegate to the package rather than hand-roll (CLAUDE.md "Downstream contract").

**Scope and limits of this validation.**

- *Tested:* demand-side GMM, Nevo data, logit + RC + RC-with-demographics,
  diagonal `sigma`, CUE-equivalent fixed-weight and a cold-start LM solve, robust
  (heteroskedastic) inference.
- *Not tested here:* the supply side (marginal-cost moments / joint
  demand-supply), the BLP automobile data, optimal-instrument iteration,
  nested-logit, `pyblp`'s alternative `W_type`/`se_type` settings, and emu's own
  2-step `IteratedWeighting` against `pyblp`'s `2s` (the first-step weights
  differ, so they agree only asymptotically — out of scope for a digit-for-digit
  check).
- *Numerical:* float64 throughout (float32 cannot resolve the contraction to the
  tolerance the objective needs); the contraction tolerance is 1e-13.

**Bottom line.** `emu-gmm` reproduces `pyblp` on the Nevo BLP problem to the
limits of optimiser tolerance, from a cold start, across point estimates, the
GMM objective, the J-statistic, and standard errors. The exercise both raises
confidence in emu's correctness and demonstrates that its `Measure` /
`CovarianceStrategy` abstractions are expressive enough to host a
nested-fixed-point structural model without special-casing in the core.

---

## 6. Reproducing

From the repo root (CPU affinity per CLAUDE.md's JIT-mmap note):

```bash
# 1. build the pyblp reference (solves all 3 specs, extracts internal arrays)
XLA_FLAGS=--xla_force_host_platform_device_count=1 taskset -c 0-3 \
    .venv/bin/python examples/blp/build_reference.py

# 2. estimate with emu and compare (warm start at pyblp's solution)
cd examples/blp && XLA_FLAGS=--xla_force_host_platform_device_count=1 taskset -c 0-3 \
    ../../.venv/bin/python run_emu.py all

# optional: the model self-check (contraction vs pyblp delta, IFT vs finite-diff)
../../.venv/bin/python sanity_check.py
```

Files: `build_reference.py` (pyblp side), `blp_model.py` (JAX BLP + contraction),
`blp_data.py` (loader), `run_emu.py` (emu estimation + comparison),
`sanity_check.py` (model de-risking). Requires `pyblp` (`pip install pyblp` into
the poetry venv); everything else is emu-gmm's existing dependency set.
