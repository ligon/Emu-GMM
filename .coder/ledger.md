# Prior-Art Ledger — emu-gmm

> Standing repo ledger (per the `prior-art-ledger` skill): the machinery,
> definitions, and invariants already in force, so a task neither reinvents what
> exists nor contradicts a commitment. Living + git-tracked — edit in place; the
> commit history is the journal. `§N` are the citations used in code comments and
> verification ("OK, anchored on §N"). This distills `CLAUDE.md` — read it (and
> `docs/design.org`) for the full architectural-commitments list before any
> non-trivial change.

**Search tier used:** ripgrep over `src/emu_gmm/__init__.py` (`__all__`) + `src/emu_gmm/`.

## §1 Repo, restated
`emu-gmm` is a JAX-native GMM framework. The name reads as **E_μ** — expectation
under a measure μ. One operator interface, implemented against
**synthetic / analytical / empirical** measures, drives sample estimation,
identification analysis, and simulation-based inference through a single pipeline.
It is meant to be a **spare set of correct interfaces**: the consumer supplies a
per-observation residual `psi(x_i, θ) → R^M` and a per-moment observability
`mask (N,M)`; *everything statistical* is owned by the package.

## §2 Existing machinery — REUSE these (all re-exported at top level)
The consumer's only modelling inputs are `psi` + `mask`. Everything else is a menu
pick or read off the result; do **not** hand-roll any of it.

| surface | symbols | source |
|---------|---------|--------|
| entry points | `estimate(...)`, `build_estimator(...)` | `estimator.py:59` |
| measures | `Synthetic/Analytical/Empirical Measure` | `measures/` |
| covariances | `IID / Clustered / Stratified / DesignAware / Sum / Synthetic / Analytical Covariance` | `covariance/` |
| weighting | `Identity / Fixed / ContinuouslyUpdated (CUE, default) / IteratedWeighting` | `weighting/` |
| regularization | `DiagonalTikhonov`, `ridge_inverse`, `TikhonovPenalty` | `regularization.py`, `numerics.py`, `penalty.py` |
| optimizers | `optimistix_lm / scipy_lm / linear_solver`; manifold: `riemannian_lm` (default), `riemannian_tr` (non-convex only) | `optimizer.py` |
| manifolds | `Euclidean / Positive / Product / PSDFixedRank / ManifoldLeaf`; `ParameterSpace`, `on` | `manifolds/`, `parameter_space.py` |
| **result** | **`EstimationResult`** → `.theta_hat`, `.J_stat/.J_dof/.J_pvalue`, `.Sigma_theta`, `.to_pandas()` (types.py:957), and gauge-invariant `.functional_se(f)` / `.gamma_se()` / `.eigenvalue_se()` (types.py:654/729/753), `.check_inference_validity()` | `types.py` |
| inference helpers | `k_statistic` / `k_confidence_set`, `cluster_bootstrap`, `j_test` | `inference/`, `k_statistic.py:589` |

Well-tested: four acceptance tests (`tests/test_estimator{,_analytical,_empirical,_realdata}.py`) drive the full pipeline; specific guards are named under §4.

## §3 Definitions & conventions in force
- **Measure ⊥ CovarianceStrategy.** `E_μ[ψ]` is a property of the `Measure`; `V(θ)` is a separate `CovarianceStrategy`. Never fold variance construction into a measure (commitment 1).
- **The `(N,M)` mask carries per-moment observability** — the only place missingness lives. Don't pre-aggregate or assume common N (commitment 6/9).
- **Float64 forced at import** (`__init__.py` → `jax.config.update("jax_enable_x64", True)` before any submodule). Stats framework, not deep learning; float32 stops LM certifying convergence and drifts θ̂ at the 3rd digit (commitment 7).
- **CU weighting is the default**; the `∇_θ V` term rides via JAX AD — don't drop it (commitment 4).
- **Information matrix via `G' Λ G`**, never a numerical Hessian (commitment 5).
- Python 3.11+, Poetry, **src/ layout** (don't flatten); JAX-native hot path (pandas/haliax/arrays only at the input boundary, `_internal/labels.py`); `@jdc.pytree_dataclass` for jit/vmap/AD objects, `jdc.static_field()` for recompile-triggering hyperparams. haliax pinned to a **PyPI dev pre-release** (`>=1.4.dev452,<1.5`), not git/1.3. `make quick-check` before every commit.

## §4 Invariants & commitments — the landmines (don't revert; full list in CLAUDE.md)
- **Per-coordinate N_j scaling, NOT textbook common-N.** Moments are per-coordinate means; `V_X` carries the `1/(N_j N_k)` normalization so `Q = m' V_X⁻¹ m → χ²_{M−K}` with **no explicit N**. **Never add an explicit N / √N to the criterion to "match a textbook."** Guard: `tests/test_estimator_unequal_nj.py` (commitment 9).
- **`DiagonalTikhonov` feasibility tests the SIGNED spectrum** (`λ_min > 0` AND `λ_max ≤ κ·λ_min`), not `jnp.linalg.cond` (an absolute-eigenvalue SVD ratio, blind to a small negative eigenvalue). A cond-only rule let a barely-indefinite V through and silently NaN'd `k_statistic`/bootstrap (#111). Don't revert to cond (commitment 3).
- **Design covariances generalize per-*pair*, not per-coordinate** (`StratifiedCovariance` weights `(j,k)` by the pair-specific PSU count `H_{c,jk}`; `DesignAwareCovariance` composes `V_TT + V_SS + estimated V_TS`, ablatable via `from_design_mask(..., couple=False)`). The glue is **not PSD-by-construction** (Tikhonov repairs it). **Known caveat (#145): under fixed stratum-mean heterogeneity the coupled `couple=True` V under-covers — prefer plain `StratifiedCovariance` or `couple=False` until the v2.1 centering pass.** Guard: `tests/covariance/test_stratified.py` F4 (commitment 10).
- **`jax.clear_caches()` per iteration** in any loop that rebuilds closures per call — bare repeated `estimate()` leaks ~14 MB/call via JAX global caches; the `build_estimator` + `replicate` factory path does not leak.
- **64-core JIT-mmap hazard:** several *uncapped* concurrent JAX processes oversubscribe cores → segfault/stall. Bound with `taskset -c` to disjoint halves, don't serialise. (CLAUDE.md has the full recipe + the `pgrep`/`pkill` self-match pitfalls.)

## §5 The downstream contract (this is the reuse decision)
- **Supply only `psi` + `mask`.** Read `theta_hat`, `J`, `Sigma_theta`, SEs, p-values, the k-statistic and bootstraps **off `EstimationResult` / the inference helpers** — never hand-recompute. The bug that motivated this: a consumer computed `J = (√N_j·mean)' V_X⁻¹ (√N_j·mean)` by hand — correct only under equal N_j.
- **A knob missing or an interface wrong? File an `emu-gmm` issue — don't patch around it locally.** Local reimplementation forfeits the single-correct-implementation guarantee. (`../Seasonality` is the model: files #2/#5/#8/#11 rather than hand-rolling.)

## §6 Open questions / deferred (v2+)
SMM matching ((1+1/S) inflation); `ReplicateWeightCovariance` (BRR/jackknife/bootstrap); `EigenvalueFloor`/`NearestPSD` (Higham) regularisers; typed identification-failure taxonomy; quadrature engine for `AnalyticalMeasure`; the `couple=True` centering pass (v2.1, #145). No pymanopt runtime backend (native constructors + dev-only cross-check only).
