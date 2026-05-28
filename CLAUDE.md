# CLAUDE.md — context for agents working in emu-gmm

## What this project is

`emu-gmm` is a JAX-native framework for Generalized Method of Moments (GMM)
estimation. The name reads as $\mathbb{E}_\mu$ — the expectation operator
under a measure $\mu$. The architectural conceit is that one operator
interface, implemented against empirical / analytical / synthetic measures,
drives sample estimation, identification analysis, and simulation-based
inference through a single pipeline.

## Where the architecture lives

- `docs/design.org` — architecture specification (current as of the
  fourth revision pass).
- `docs/mcar-asymptotics.org` — theoretical companion note: consistency,
  asymptotic normality, and positive-definiteness under MCAR.
- `docs/api-sketch.org` — v1 API surface; module layout, types, and a
  worked Euler-equation example.

Read these before making non-trivial code changes. The design has been
through four reviewer iterations; the abstractions are deliberate.

## Conventions

- **Python 3.11+**, Poetry for dependencies, `pyproject.toml` for everything.
- **JAX-native** in the hot path. No pandas inside compiled boundaries.
- **Named axes via `haliax`** internally (PyTree-friendly tensors with
  axis labels). Pandas-labelled tensors are valid at the I/O boundary
  via the workspace's `jax-pandas-bridge` if needed.
- **`@jdc.pytree_dataclass`** for any object that needs to pass through
  `jit` / `vmap` / AD.
- **Static fields** (`jdc.static_field()`) for hyperparameters that
  should trigger recompilation, not be traced.
- **Tests** with `pytest`. Slow tests marked `@pytest.mark.slow`. Use
  `make quick-check` for iteration; `make check` for full validation.
- **Lint/format** via `ruff` and `black`. Pre-commit hooks installed
  via `pre-commit install`.

## Architectural commitments worth knowing

1. **`Measure` and `CovarianceStrategy` are orthogonal.** The same
   `EmpiricalMeasure` can pair with any covariance strategy on the
   v1/v2 design-awareness ladder. Don't fold variance construction
   into measure implementations.
2. **`Measure` is a computational interface**, not a strict
   measure-theoretic object. The empirical case uses an $M$-tuple of
   coordinate-specific measures internally; this is invisible to the
   rest of the framework.
3. **Pairwise overlap → finite-sample non-PD risk → adaptive
   regularisation.** `DiagonalTikhonov` is the v1 default; the
   anchor-once-then-freeze policy keeps the objective smooth.
4. **CU vs Iterated weighting are asymptotically equivalent but not
   identical in finite samples.** CU is the v1 default. Don't
   silently drop the $\nabla_\theta V$ term from the CU gradient;
   write the residual function so JAX AD captures the dependence.
5. **Information matrix via $G' \Lambda G$, never via numerical
   Hessian.** The two coincide asymptotically; the direct form is
   cheaper and more stable.

## Sibling repos worth knowing about

- `../ManifoldGMM/` — the explicit exemplar in `../BestPractices.org`.
  Same conventions, similar problem domain.
- `../K-Aggregators/`, `../CFEDemands/`, `../MetricsMiscellany/` —
  other libraries in the workspace; useful for finding patterns.
- `../BestPractices.org` — workspace conventions for repo organization.

## What's not done yet

- `src/emu_gmm/` is a stub. Only `__init__.py` exists.
- No real implementations of `Measure`, `CovarianceStrategy`,
  `WeightingStrategy`, `Optimizer`, or `estimate()`.
- `docs/api-sketch.org` still shows `EmpiricalMeasure` as the worked
  example; should be reworked to use `SyntheticMeasure` for the v1
  demo (no real data → synthetic generator is the natural showcase).

## Don't surprise the user

- Don't add dependencies without flagging them.
- Don't restructure the `docs/` layout without discussion.
- Don't reframe the architectural commitments in §"Architectural
  commitments" without going through the user.
- Don't commit work without being asked, unless the user has set up
  a "commit each substantive step" rhythm in the session.
