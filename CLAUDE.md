# CLAUDE.md — context for agents working in emu-gmm

## What this project is

`emu-gmm` is a JAX-native framework for Generalized Method of Moments (GMM)
estimation. The name reads as $\mathbb{E}_\mu$ — the expectation operator
under a measure $\mu$. The architectural commitment is that one operator
interface, implemented against empirical / analytical / synthetic measures,
drives sample estimation, identification analysis, and simulation-based
inference through a single pipeline.

## Status

v1 is implemented and operational. 226 tests pass; `make quick-check` is
clean (ruff + black + mypy + pytest). All three measure paths (synthetic,
analytical, empirical) run end-to-end against the bundled multi-asset Euler
example in `src/emu_gmm/examples/euler.py`. The runnable demo at
`examples/run_euler.py` exercises all three contexts and prints recovery
and the J-statistic for each.

## Where the architecture lives

- `docs/design.org` — architecture specification (four review iterations;
  stable).
- `docs/mcar-asymptotics.org` — theoretical companion note: consistency,
  asymptotic normality, and positive-definiteness under MCAR.
- `docs/api-sketch.org` — v1 API surface; module layout, types, and the
  worked Euler-equation example in three variants.
- `docs/implementation-plan.org` — phased task list; Phases 1-7 complete,
  Phase 8 (polish) underway.

Read these before making non-trivial architectural changes. The design has
been through four reviewer iterations; the abstractions are deliberate.

## Where the code lives

- `src/emu_gmm/` — implementation. Public API re-exported at top level.
- `src/emu_gmm/examples/euler.py` — shared model and DGP for the Hansen-
  Singleton-style multi-asset Euler example used by all three acceptance
  tests and by `examples/run_euler.py`.
- `tests/` — mirrors `src/emu_gmm/` structure. Three acceptance tests
  (`test_estimator.py`, `test_estimator_analytical.py`,
  `test_estimator_empirical.py`) drive the full pipeline.
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
- **haliax pinned to GitHub master** (not PyPI). PyPI haliax 1.3 imports
  `jax._src.tree_util.BuiltInKeyEntry` which current JAX no longer
  exposes; the git version is compatible. Documented in `pyproject.toml`.
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
   keeps the objective smooth in $\theta$.
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

## Deferred to v2+

- **SMM matching**: combining `EmpiricalMeasure` and `SyntheticMeasure`
  in one moment vector with the $(1+1/S)$ asymptotic-variance inflation.
- **`StratifiedCovariance`, `ReplicateWeightCovariance`**: rest of the
  design-awareness ladder (`docs/design.org` Section 2).
- **`Iterated` weighting**: subsumed by CU for v1; would be ~50 lines.
- **`EigenvalueFloor`, `NearestPSD` (Higham 1988) regularisers**:
  alternatives to `DiagonalTikhonov`.
- **Structured identification-failure taxonomy**: `design.org` Section 5.4
  describes three pathology classes; v1 surfaces the raw diagnostics but
  doesn't classify failures into typed reports.
- **Quadrature engine for `AnalyticalMeasure`**: user currently supplies
  closed-form `expectation_fn`; v2 might add Gauss-Hermite / Gauss-Legendre
  helpers.
- **Real-data acceptance test on the empirical path**: deferred until
  actual data lands. Phase 7 covers the synthetic-data unit-test version.

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
