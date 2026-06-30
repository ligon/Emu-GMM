# CLAUDE.md — context for agents working in emu-gmm

## What this project is

`emu-gmm` is a JAX-native framework for Generalized Method of Moments (GMM)
estimation. The name reads as $\mathbb{E}_\mu$ — the expectation operator
under a measure $\mu$. The architectural commitment is that one operator
interface, implemented against empirical / analytical / synthetic measures,
drives sample estimation, identification analysis, and simulation-based
inference through a single pipeline.

## Status

v1 is implemented and operational. v2 adds the #79/#80 stratified/design-aware
covariance module and the **Riemannian-manifold epic (#12)**: a
`Product(PSDFixedRank(n, K), Euclidean(...))` parameter is estimable end-to-end
via `RiemannianLM`, with gauge-aware `Sigma_theta` and gauge-invariant standard
errors on functionals of `Gamma = A @ A.T` (`result.eigenvalue_se()`,
`result.gamma_se()`, `result.functional_se(f)`). The manifold types
(`PSDFixedRank`, `Euclidean`, `Product`, `Positive`, `ManifoldLeaf`) are
re-exported at the top level alongside the Measure/Covariance menus. The green
gate is `make check` (ruff + black + mypy + the full pytest suite) — restored
clean 2026-06-09 (#122); run `make quick-check` before every commit and do not
quote literal test counts here (they go stale; the suite is the count). The v2
roadmap and exit criteria live in `docs/implementation-plan.org` Section 13 and
GitHub issue #131. All three
measure paths (synthetic, analytical, empirical) run end-to-end against the
bundled multi-asset Euler example in `src/emu_gmm/examples/euler.py`. The
runnable demo at `examples/run_euler.py` exercises all three contexts and prints
recovery and the J-statistic for each.

## Where the architecture lives

- `docs/design.org` — architecture specification (four review iterations;
  stable).
- `docs/mcar-asymptotics.org` — theoretical companion note: consistency,
  asymptotic normality, and positive-definiteness under MCAR.
- `docs/api-sketch.org` — v1 API surface; module layout, types, and the
  worked Euler-equation example in three variants.
- `docs/implementation-plan.org` — phased task list; Phases 1-7 complete,
  Phase 8 (polish) underway.
- `docs/manifold-epic-progress.org` — the Riemannian-manifold epic (#12)
  record: the 7-phase build (PRs #97–#104), the locked design decisions, and the
  gauge / `Sigma_theta` design brief. `docs/manifold-slice-scoping.md` is the
  original minimal-slice build spec.
- `docs/migration/` — user-facing migration guides (e.g.,
  `manifoldgmm-to-emu-gmm.org`). `docs/design.org` and `docs/api-sketch.org`
  are implementer-facing specs; this CLAUDE.md is the agent-facing index.

Read these before making non-trivial architectural changes. The design has
been through four reviewer iterations; the abstractions are deliberate.

## Where the code lives

- `src/emu_gmm/` — implementation. Public API re-exported at top level.
- `src/emu_gmm/examples/euler.py` — shared model and DGP for the Hansen-
  Singleton-style multi-asset Euler example used by all three acceptance
  tests and by `examples/run_euler.py`.
- `tests/` — mirrors `src/emu_gmm/` structure. Four acceptance tests
  (`test_estimator.py`, `test_estimator_analytical.py`,
  `test_estimator_empirical.py`, and the real-data
  `test_estimator_realdata.py`) drive the full pipeline. The real-data
  one runs against the frozen Seasonality Euler extract in
  `tests/data/` (provenance sidecar + `scripts/freeze_seasonality_extract.py`;
  issue #128) and cross-validates the consumer's published estimates.
- `examples/run_euler.py` — top-level runnable demo; not a test.

## Conventions

- **Python 3.11+**, Poetry for dependencies, `pyproject.toml` for everything.
- **src/ layout** (`src/emu_gmm/`), not flat — workspace standard per
  `../BestPractices.org`. Don't move the package to the top level without
  discussion.
- **JAX-native** in the hot path. No pandas inside compiled boundaries.
  Pandas / haliax / plain arrays all accepted at the input boundary via
  `_internal/labels.py`; converted to plain JAX internally.
- **Named axes via `haliax`** for labelled outputs (Sigma_theta, V_X,
  per-moment diagnostics). NamedArray comes back to the user via
  `EstimationResult`; pandas materialisation via `result.to_pandas()`.
- **`@jdc.pytree_dataclass`** for any object that needs to pass through
  `jit` / `vmap` / AD.
- **Static fields** (`jdc.static_field()`) for hyperparameters that should
  trigger recompilation, not be traced. Mypy doesn't know about
  `jdc.static_field`; suppress with `# type: ignore[attr-defined]` on the
  call site (matches the pattern in `measures/synthetic.py`).
- **Tests** with `pytest`. Slow tests marked `@pytest.mark.slow`. Use
  `make quick-check` for iteration; `make check` for full validation.
  Run quick-check *before* every commit — the lint config has several
  jaxtyping false-positive suppressions in place but new code can still
  fail.
- **Lint/format** via `ruff` and `black`. Pre-commit hooks via
  `pre-commit install`. `pyproject.toml` ignores `F722`, `F821`, `UP037`
  globally — those are jaxtyping shape-string false positives.
- **haliax pinned to the PyPI dev pre-release** (`>=1.4.dev452,<1.5`),
  NOT the git URL and NOT stable 1.3. Stable 1.3 imports
  `jax._src.tree_util.BuiltInKeyEntry` which current JAX no longer
  exposes; the old git pin was the workaround but PyPI rejects direct
  URL deps in published metadata (the v0.3.0 publish failure). haliax
  auto-publishes dev wheels from main, and 1.4.dev452 is built from the
  exact revision (280f49f) the v2 validation ran against — poetry's own
  version stamp confirmed the identity. Rationale documented in
  `pyproject.toml`; revisit when upstream cuts a stable 1.4.
- **Float64 is the default precision.** See "Architectural commitments"
  point 7. Float32 silently degrades both convergence (optimistix LM
  can't certify) and accuracy (theta_hat drifts at the third digit).

## Architectural commitments worth knowing

1. **`Measure` and `CovarianceStrategy` are orthogonal.** Same
   `EmpiricalMeasure` can pair with `IIDCovariance` or `ClusteredCovariance`
   or future `StratifiedCovariance`/`ReplicateWeightCovariance` without
   touching the measure. Don't fold variance construction into measure
   implementations.
2. **`Measure` is a computational interface**, not a strict measure-theoretic
   object. The empirical case uses an $M$-tuple of coordinate-specific
   measures internally; invisible to the rest of the framework.
3. **Pairwise overlap → finite-sample non-PD risk → adaptive regularisation.**
   `DiagonalTikhonov` is the v1 default; anchor-once-then-freeze policy
   keeps the objective smooth in $\theta$. Its feasibility test is on the
   **signed** spectrum (`λ_min > 0` *and* `λ_max ≤ κ_target·λ_min`), not on
   `jnp.linalg.cond` (an SVD ratio of *absolute* eigenvalues, which is blind
   to a small negative eigenvalue). This is what makes the regulariser
   actually deliver the `V ≻ 0` that `_internal/cholesky.py` assumes — a
   conditioning-only rule let a barely-indefinite-but-well-conditioned `V`
   pass through with ~zero ridge and silently NaN'd `k_statistic` /
   `moment_wild_bootstrap` (issue #111). Don't revert it to a `cond`-based
   test.
4. **CU vs Iterated weighting are asymptotically equivalent but not
   identical in finite samples.** CU is the v1 default. Don't silently
   drop the $\nabla_\theta V$ term from the CU gradient; the residual
   function is written so JAX AD captures the dependence automatically.
5. **Information matrix via $G' \Lambda G$**, never via numerical Hessian.
   The two coincide asymptotically; the direct form is cheaper and more
   stable in finite samples.
6. **Labelled outputs via the LabelContext**. The estimator probes the
   model's return value to detect a `haliax.NamedArray` with a `Moments`
   axis and uses its labels; else `moment_names` kwarg; else positional
   `m_0, m_1, ...`. Don't mutate label state from inside the residual
   closure — it rides as a static closure variable.
7. **JAX float64 enabled at package import.** `src/emu_gmm/__init__.py`
   calls `jax.config.update("jax_enable_x64", True)` before any
   sub-module import. JAX defaults to float32 (a deep-learning
   convention); for a statistics framework where Cholesky pivots and
   gradient norms cross many orders of magnitude, float32 is the wrong
   baseline. Symptom under float32: optimistix's LM cannot certify
   convergence at rtol=1e-8 on whitened residuals of magnitude ~0.1
   because the float32 noise floor is ~2e-8. Users wanting float32
   override after import.
8. **Optimiser modularity matters in practice.** Both `optimistix_lm()`
   and `scipy_lm()` work for v1; the framework's `Optimizer` protocol
   makes the swap trivial. Don't tie estimator logic to a specific
   backend.
9. **Per-coordinate $N_j$ scaling — not textbook common-$N$.** Moments are
   per-coordinate means $\bar m_j = \frac1{N_j}\sum_i d_{ij}w_i\psi_j$ and
   $V_X$ carries the matching $\frac1{N_jN_k}$ normalisation, so
   $Q=\bar m'V_X^{-1}\bar m\to\chi^2_{M-K}$ with **no explicit $N$** — all
   the sample-size and per-moment-overlap bookkeeping lives in $V_X$. This
   is algebraically identical to the textbook habit of scaling moment $j$
   by $\sqrt{N_j}$ and weighting by an $O(1)$ covariance, *provided the
   weight is rescaled to match* ($W=\sqrt{N_jN_k}\,V_X$). Scaling moments
   by $\sqrt{N}$ while weighting by $V_X$ equals $N\cdot Q$ **only when all
   $N_j$ are equal** and is otherwise simply wrong. A balanced
   `mask=ones` Monte Carlo cannot catch this (it forces $N_j\equiv N$);
   `tests/test_estimator_unequal_nj.py` is the guard. See `design.org`
   "Scaling convention" and `mcar-asymptotics.org` Thm 3/6. **Never add an
   explicit $N$ (or $\sqrt N$) multiply to the criterion to "match a
   textbook."**
10. **Design covariances generalise the pairwise-overlap rule per *pair*,
    not per coordinate.** `StratifiedCovariance` (between-PSU Neyman, #79)
    centers within each `(stratum x arm)` cell and weights entry $(j,k)$ by
    the *pair-specific* effective PSU count $H_{c,jk}$ — the PSUs supporting
    **both** $j$ and $k$ — exactly mirroring the $N_jN_k$ available-pairs
    rule in commitment 9. Collapsing $H_{c,jk}$ to a per-coordinate $H_{c,j}$
    mis-centers every off-diagonal under unequal observability; a balanced
    `mask=ones` fixture cannot catch it (`tests/covariance/test_stratified.py`
    F4 is the guard). FPC (`fpc=True`, default off) applies one
    *coordinate-independent* per-cell scalar $1-H_{sD,c}/H_s$ to every $(j,k)$
    — the assignment-fraction convention: the FPC is a property of the
    without-replacement assignment, not of observability (which already
    enters via dof/centering), so a per-pair *numerator* would double-count
    it. Resolved jointly with the consumer's design appendix (the per-pair
    numerator was considered and rejected). The
    `DesignAwareCovariance` mixed assembly (#80) *composes* $V_{TT}$ (design,
    delegated) + $V_{SS}$ (cluster, delegated) + an inline $V_{TS}$ cross
    coupling — **estimated, not zeroed**, clustered at the caller's `sampling`
    unit (stratum-level to capture the cross-arm term). The `sampling`
    strategy's `dof_correction` is *inherited by the cross pass* (#119,
    resolved): cross pairs get the per-pair $G_{jk}/(G_{jk}-1)$ over the
    sampling clusters supporting both coordinates, symmetric with $V_{SS}$;
    the design FPC stays inside $V_{TT}$ only. The all-design case
    reduces **bit-for-bit** to `StratifiedCovariance` via a *shared* (not
    copied) engine. The design-exact-$V_{TT}$ **glue is NOT PSD-by-construction**
    (it can be indefinite like `StratifiedCovariance`; `DiagonalTikhonov`
    repairs it — don't over-claim PSD). **Known caveat (#145, #130 MC
    evidence): fixed stratum-mean heterogeneity contaminates the coupled
    assembly** — $V_{TT}$ centers within cells but $V_{SS}$/$V_{TS}$ are
    uncentered, so the composed `couple=True` V under-covers the randomized
    contrast (b1 coverage 0.66, binding 39% in the validation fixture)
    while plain `StratifiedCovariance` stays calibrated; prefer it (or
    `couple=False`) under stratum heterogeneity until the v2.1 centering
    pass. Stratified reduces to
    `ClusteredCovariance` only *in expectation* (centered vs uncentered), not
    bit-for-bit. The $V_{TS}$ cross block is **ablatable in-framework** (#109):
    `from_design_mask(..., couple=False)` returns the block-diagonal
    $V_{TT}\oplus V_{SS}$ ($V_{TS}=0$ counterfactual) and `.cross_block(...)`
    returns the estimated $V_{TS}$ alone, with
    `covariance(couple=True) == covariance(couple=False) + cross_block`. Default
    `couple=True` is bit-for-bit the original; consumers must use these rather
    than hand-rolling a block-diagonal wrapper.

## Deferred to v2+

- **SMM matching**: combining `EmpiricalMeasure` and `SyntheticMeasure`
  in one moment vector with the $(1+1/S)$ asymptotic-variance inflation.
- **`StratifiedCovariance`** (#79) and **`DesignAwareCovariance`** (#80) are
  both implemented: the design-based between-PSU Neyman variance, and the
  composed mixed design/sampling assembly (`V_TT` + `V_SS` + estimated `V_TS`).
- **Riemannian-manifold estimation (#12)** is implemented (PRs #97–#104): native
  `Euclidean` / `PSDFixedRank` / `Product` / `Positive` + `RiemannianLM`, the
  manifold-aware flatten/spec path, gauge-aware `Sigma_theta` (`pinv_eigvalrule`
  dropping the `K(K-1)/2` gauge directions by count), and gauge-invariant
  Gamma-functional SEs (#42). A JAX-native Riemannian TrustRegions solver
  (`riemannian_tr`, top-level exported) **was added 2026-06-17 (#152, reopening
  #9)**, reversing the 2026-06-09 "won't-do": it follows negative curvature for
  non-convex manifold criteria (the K-Aggregators use case) where the
  Gauss–Newton `riemannian_lm` stalls, and is verified faithful to a
  pymanopt-TrustRegions oracle on the quotient
  (`tests/manifolds/test_rtr_pymanopt_parity.py`). `riemannian_lm` stays the
  manifold default (cheaper GN); reach for `riemannian_tr()` only on
  demonstrably non-convex criteria. `riemannian_lm` additionally emits a
  non-convexity **advisory**: on an interactive fit that converges to a genuine
  stationary point whose horizontal true Hessian is indefinite (a saddle), it
  warns and sets `OptimizerInfo.stalled_indefinite` / `.min_curvature`,
  suggesting `riemannian_tr()` (eager-only; never under the vmapped/replicate
  path; a #156 ftol cost-stagnation stop drifts at large gradient and is
  excluded as non-stationary). There is still no pymanopt RUNTIME backend (#3 —
  native constructors only; pymanopt is a dev-only gated cross-check). The
  high-gauge-fraction `k/n > 0.7` regime remains surfaced as a caveat, not
  supported. Validated against a pymanopt-TrustRegions baseline on the
  quotient. The epic's last open item landed with #20:
  `cond_info['exclude_gauge']` is the gauge-aware quotient condition number
  (drop `total_gauge_dim` smallest eigenvalues by count, same rule as
  `pinv_eigvalrule`; extra near-zeros beyond the dropped count still blow it
  up — that is the structural-rank-deficiency signal consumers test for).
  For `total_gauge_dim == 0` it remains the bitwise alias of `raw`.
- **`ReplicateWeightCovariance`**: rest of the design-awareness ladder
  (BRR, jackknife, bootstrap variants; `docs/design.org` Section 2).
- **`EigenvalueFloor`, `NearestPSD` (Higham 1988) regularisers**:
  alternatives to `DiagonalTikhonov`.
- **Structured identification-failure taxonomy**: `design.org` Section 5.4
  describes three pathology classes; v1 surfaces the raw diagnostics but
  doesn't classify failures into typed reports.
- **Quadrature engine for `AnalyticalMeasure`**: user currently supplies
  closed-form `expectation_fn`; v2 might add Gauss-Hermite / Gauss-Legendre
  helpers.
- **Real-data acceptance test on the empirical path**: DONE 2026-06-10
  (#128) — `tests/test_estimator_realdata.py` against the frozen
  Seasonality Euler extract (`tests/data/seasonality_euler_extract.npz`,
  owner-approved derived columns with provenance sidecar).

## Sibling repos worth knowing about

- `../ManifoldGMM/` — the explicit exemplar in `../BestPractices.org`.
  Same conventions, similar problem domain. Reference for src/ layout,
  Makefile patterns, pyproject.toml structure.
- `../K-Aggregators/`, `../CFEDemands/`, `../MetricsMiscellany/` — other
  libraries in the workspace; useful for finding patterns.
- `../BestPractices.org` — workspace conventions for repo organization.

## Don't surprise the user

- Don't add dependencies without flagging them. Current runtime deps:
  jax, jaxtyping, haliax (git), jax_dataclasses, optimistix, numpy,
  scipy, pandas.
- Don't restructure the `docs/` layout or move `src/emu_gmm/` to a flat
  layout without discussion.
- Don't reframe the architectural commitments above without going through
  the user.
- Don't commit work without being asked, unless the user has set up a
  "commit each substantive step" rhythm in the session.
- Don't switch the worked example away from `examples.euler` — three
  acceptance tests and a runnable script all depend on it.
- Don't silently change tolerances on acceptance tests; recovery
  guarantees in the tests are the contract.

## Downstream contract (this applies to consumers like `../Seasonality`)

`emu-gmm` is meant to be a **spare set of correct interfaces**. The
division of labour is deliberate (`design.org` §2): the consumer supplies
a per-observation residual `psi(x_i, theta) -> R^M` and expresses
per-moment observability through the `(N, M)` `mask`; everything
statistical — moment expectation, design-aware `V_X`, the criterion,
`J_stat`, `Sigma_theta`, standard errors, p-values, the K-statistic and
bootstrap helpers — is **owned by the package** and read off
`EstimationResult` / the inference helpers.

- **Don't reinvent package internals downstream.** A consumer must not
  recompute the criterion, `J`, or SEs inline. This is not stylistic: the
  real bug that motivated this section was a downstream repo computing
  `J = (sqrt(N_j)*mean)' V_X^{-1} (sqrt(N_j)*mean)` by hand — correct only
  under equal `N_j`, wrong under missingness (see commitment 9). Delegating
  to the package makes that class of error impossible.
- **Found a problem with an interface? File an `emu-gmm` issue — don't
  patch around it locally.** If a knob is missing or an interface looks
  wrong, the fix belongs in the shared package so every consumer gets it.
  A local reimplementation forfeits the single-correct-implementation
  guarantee and silently forks behaviour. (The `../Seasonality` port is a
  good model here — it files issues `#2/#5/#8/#11/...` rather than
  hand-rolling; the lapse was documentation algebra that assumed equal
  `N_j`, not the code itself.)

## Running on this box — the 64-core JIT-mmap hazard

This host has 64 logical CPUs — **but check before assuming**: sandboxed
sessions may see far fewer (`cat /proc/self/status | grep Cpus_allowed_list`;
a 2026-06-10 session saw only CPUs 0-6 with ~14 GB RAM). `taskset -c` to
cores outside the allowed set fails with EINVAL; ranges that merely
overlap it are silently intersected, so a "pin to 0-31" succeeds while
actually pinning to 0-6.

JAX/XLA sizes its JIT thread pool to the (affinity-visible) CPU
count and `mmap`s executable pages per worker; running several *uncapped*
JAX processes at once (multiple `make quick-check` invocations, or a
multi-agent workflow where each agent runs pytest) oversubscribes the cores
(2 × 64 threads on 64 cores) and stalls or kills the runs. This is the most
likely cause of the mid-session segfault this project has hit, and of
`pytest` runs dying partway with no summary line.

The fix is to **bound the cores each process uses** with CPU affinity, not
to serialise. XLA/Eigen size their pools to the affinity-limited CPU count
(`len(os.sched_getaffinity(0))`), so `taskset` makes a JAX process behave as
an N-core process — fewer threads, fewer mmaps, confined to its cores. Pin
concurrent suites to **disjoint halves** and they coexist at full speed:

```
XLA_FLAGS=--xla_force_host_platform_device_count=1 taskset -c 0-31  .venv/bin/python -m pytest <...>   # suite A
XLA_FLAGS=--xla_force_host_platform_device_count=1 taskset -c 32-63 .venv/bin/python -m pytest <...>   # suite B
```

Validated: two JIT-heavy suites pinned to `0-31` / `32-63` both pass
concurrently (~35 s each), no segfault. This supersedes the old
`OMP_NUM_THREADS=1` single-thread recipe, which was reliable but ~20× slower
(it forced one thread). Use `taskset` halves instead; reach for the
single-thread caps only as a last resort.

- For a **single** run, `taskset -c 0-31` is still worth it: a 32-core run is
  fast and leaves 32 cores free for other work.
- **Multi-agent workflows / delegated agents:** give each agent a distinct
  core range (`0-31` vs `32-63`) rather than telling them to serialise.
- **Pitfall:** do **not** gate on `pgrep -f "pytest"` in a wait loop — the
  loop's own command line contains `pytest`, so it self-matches and never
  exits. (Cost a hung agent once.) Match the concrete process,
  e.g. `pgrep -f "bin/python -m pytest"`, and exclude your own PID — or just
  pin with `taskset` and skip the wait entirely.
- **The self-match hazard applies to `pkill` too, with worse consequences**
  (recurred 2026-06-10 despite the warning above): `pkill -f ladder_mc` in a
  compound command killed its own shell mid-command (exit 144), silently
  skipping everything after it. Use bracket patterns (`pgrep -f
  "[l]adder_mc"`) — and note the bracket trick fails if the LITERAL bracket
  pattern appears elsewhere in your own command line (an `ls
  /tmp/run_ladder_full.sh` in the same compound command re-creates the
  match). Kill by concrete PID, and verify state via files (logs, output
  artifacts), not process greps.
- A backgrounded run whose shell wrapper exits can leave the `pytest` child
  alive — check `ps` before relaunching; "completed" on a piped background
  command does not mean the child died.
- **Jobs longer than ~10 minutes:** detach them (`setsid nohup script >
  log 2>&1 < /dev/null &`) and watch the log, rather than trusting a
  harness background task to outlive its timeout. (Empirically two ~20-min
  gates survived a 10-min nominal timeout, but the contract says they
  shouldn't — don't build multi-hour MC runs on that accident.)
- **Repeated bare `estimate()` calls leak ~14 MB/call** via JAX's global
  caches (fresh closures per call → write-only traces; OOM-killed a 300-rep
  MC study at 9.4 GB, see #139's merge-verification thread). In any loop
  that rebuilds closures per iteration, call `jax.clear_caches()` per
  iteration — it costs nothing there precisely because nothing is re-hit.
  The `build_estimator` + `replicate` factory path does not leak.
- **gitnexus MCP / Kùzu single-writer lock — diagnose by holders, not theory.**
  Each *indexed* repo has its **own** `.gitnexus/lbug` (a single-writer Kùzu
  DB); only the MCP server **binary** and `~/.gitnexus/registry.json` (a
  name→path map) are shared, so the lock is **per-repo**, not cross-repo. The
  recurring `FTS index ensure failed ... read-only database` warning means more
  than one process has *this repo's* `lbug` open at once — and the usual culprit
  is **your own session leaking duplicate servers**: each `/mcp` reconnect spawns
  a fresh `gitnexus mcp` without reaping the old one. Diagnose with the process
  table, not a theory: `lsof .gitnexus/lbug` (or `pgrep -f "gitnexus mcp"` then
  check each PID's `PPid` — they're usually all children of *your own* harness,
  not "another session/repo"). Do **not** kill-by-guess — "newest = live, oldest
  = stale" is unreliable, and killing the live stdio-connected server drops your
  MCP connection (learned the hard way 2026-06-25). Recycle via a `/mcp`
  reconnect (reaps + respawns cleanly), never `kill`. The warning is **cosmetic**:
  the per-Bash staleness hook spawns its *own* short-lived gitnexus process that
  contends for the same single-writer lock, so it can recur even with one server;
  `context` / `cypher` / `impact` work off the read path — only NL `query()`
  (FTS-backed) is the casualty. General rule (applies beyond gitnexus): for any
  *locked / read-only / contended* symptom, enumerate the actual holders before
  theorizing, acting, or sending the user to act.

