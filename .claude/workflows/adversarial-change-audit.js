export const meta = {
  name: 'adversarial-change-audit',
  description: 'Adversarial audit of the statistics-bearing #124/#133/#114 merges before release',
  whenToUse: 'After landing statistics-bearing changes; finds silent-validity defects via lens-specific finders + adversarial refutation.',
  phases: [
    { title: 'Find', detail: 'four lens-specific finders over the change-set' },
    { title: 'Verify', detail: 'three perspective-diverse refuters per finding' },
    { title: 'Synthesize', detail: 'verdict report from surviving findings' },
  ],
}

// ---------------------------------------------------------------------------
// The change-set under audit (passed via args so the workflow is reusable):
// args = { commits: [{sha, label, oneline}], repo }
// ---------------------------------------------------------------------------
const _args = typeof args === 'string' ? JSON.parse(args) : args
const repo = _args.repo
const commits = _args.commits
const commitList = commits.map(c => `- ${c.label}: ${c.sha} (${c.oneline})`).join('\n')

const COMMON = `
You are auditing recently merged work in the JAX GMM library at ${repo}.
Change-set under audit (inspect with \`git -C ${repo} show <sha>\` and read the
current files; the commits are all merged or about to merge):
${commitList}

Background docs: ${repo}/docs/design.org (esp. Section 5), ${repo}/CLAUDE.md,
GitHub issues #124, #133, #114 (\`gh issue view N -R ligon/Emu-GMM --json body,comments\`).

RESOURCE DISCIPLINE (shared 64-core host): static analysis FIRST and by
default. You may run a SMALL python probe only if static analysis is
genuinely inconclusive, prefixed with
\`XLA_FLAGS=--xla_force_host_platform_device_count=1 taskset -c 32-63\`,
using ${repo}/.venv/bin/python, under 60 seconds, NEVER a pytest suite.
Do not modify any files. Do not use bare 'pgrep -f pytest' patterns.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          claim: { type: 'string', description: 'What is wrong, precisely' },
          evidence: { type: 'string', description: 'file:line and reasoning' },
          file: { type: 'string' },
        },
        required: ['title', 'severity', 'claim', 'evidence', 'file'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    reasoning: { type: 'string' },
    severity_opinion: { type: 'string', enum: ['high', 'medium', 'low', 'not-a-bug'] },
  },
  required: ['refuted', 'reasoning', 'severity_opinion'],
}

const LENSES = [
  {
    key: 'stats',
    prompt: `${COMMON}
LENS: STATISTICAL CORRECTNESS. Audit the estimation/inference math:
- The #133 sandwich block in src/emu_gmm/estimator.py (_inference_core):
  bread = G' Lambda G via vmapped whitening of G's columns; meat
  C = L_w^{-1} V L_w^{-T} via whitening V's columns then the transpose's
  columns; Sigma = B^+ meat B^+. Check the algebra line by line: is C
  actually L^{-1} V L^{-T} given whitening_residual semantics? Is the
  UNREGULARIZED V_local the right meat under a binding ridge? Is the
  CU/tau=0 collapse exact in exact arithmetic? Gauge case: does the
  claim "G_riem annihilates gauge directions so bread and meat share
  the nullspace" actually hold for the meat (which sandwiches C between
  Zw' and Zw)? Is pinv-of-bread-only correct, or should the meat also
  be projected?
- Does the J statistic remain consistent with the new Sigma under each
  weighting? Is J_pvalue still computed under non-optimal weightings
  where ||y||^2 has no chi-square limit (known footgun -- has #133 made
  it better, worse, or unchanged)?
- src/emu_gmm/studies/summaries.py: coverage / size_power / bias_sd /
  tau_binding arithmetic, exclusion handling (n_used denominators),
  the SE/MC-SD ratio definition.
- FitRecord semantics: converged as 0/1 float, se from sqrt(diag Sigma).
Report AT MOST your 4 highest-severity findings. Real defects only --
not style. If the math is right, say so with fewer findings.`,
  },
  {
    key: 'tracing',
    prompt: `${COMMON}
LENS: JAX TRACING / CACHING SEMANTICS. Audit the execution-model claims:
- Kernel-identity caching: _residual_kernel / _inference_kernel_jit /
  _fixed_residual_kernel are claimed factory-stable so traces are shared.
  Any path where a fresh closure sneaks in per call? Any eager JAX
  evaluation per call that re-traces (the final_objective hazard was
  fixed via _JITTED_FN_CACHE -- verify; any remaining ones, e.g.
  _effective_n_per_moment, cu_residual_fn glue lambda, chol_kernel)?
- id()-keyed caches (_OPTIMISTIX_FN_CACHE, _JITTED_FN_CACHE,
  _TRACED_SOLVE_CACHE keyed (id(kernel), self, manifold_spec)): id reuse
  after garbage collection -- are the weakref.finalize evictions
  airtight, including for bound methods or lambdas that die early?
- The weak-dtype convert_element_type fix in weighting.py: complete, or
  are there other weak-typed entries (e.g. estimate()'s one-shot path,
  the v2 manifold path)?
- _whiten_cols under vmap: any strategy whose whitening_residual is NOT
  linear in m (or shape-polymorphic) that would silently produce a wrong
  Lambda application rather than an error?
- Donation/aliasing or dtype assumptions in studies/driver.py.
Report AT MOST your 4 highest-severity findings.`,
  },
  {
    key: 'compat',
    prompt: `${COMMON}
LENS: API BACK-COMPAT AND DISPATCH. Audit the compatibility claims:
- Optimizer protocol: args kwarg added to built-ins; _supports_args
  probes by signature -- false positives (an optimizer with **kwargs
  that does NOT understand args), false negatives (wrapped/partial
  optimizers)? What happens if a third-party optimizer accepts args but
  ignores it -- silent wrong results or loud failure?
- Estimator gate: use_traced_path requires type(measure_call) is
  type(template_measure) -- subclass measures silently fall to the
  legacy path (fine) but do factory-time decisions (cache attr name,
  label probe, tau anchor) stay valid for a SUBCLASS that overrides
  expectation_and_contributions? Any path where a same-class measure
  with different STATIC fields (e.g. SyntheticMeasure with a different
  sampler) rides a stale trace?
- outer_loop_driver: new keyword-only kwargs + signature probing -- can
  an old-signature third-party driver receive unexpected kwargs anywhere?
- riemannian_lm args=None byte-identical claim: verify the no-args path
  really is untouched.
- studies/: does replicate() make assumptions about the run-callable
  (e.g. that it accepts (theta_init, measure)) that build_estimator
  callables from older patterns violate?
Report AT MOST your 4 highest-severity findings.`,
  },
  {
    key: 'tests',
    prompt: `${COMMON}
LENS: TEST DISCRIMINATION. For each NEW test in the change-set
(tests/test_estimator_traced_measure.py, tests/test_iterated_traced_args.py,
tests/manifolds/test_riemannian_lm_traced_args.py, tests/test_sigma_sandwich.py,
tests/studies/*), ask: CAN IT FAIL on the defect it claims to pin?
This project has twice found tests that could not fail (a degenerate
fixture where V was exactly zero at the optimum; an Euler fixture where
Identity weighting was 99.996% efficient so there was no gap to detect).
Hunt for more of the same:
- parity tests that compare a quantity to itself through two code paths
  that share the same underlying function (vacuous by construction)?
- retrace counters: does the counting-model count what the test thinks
  (Python executions == trace events)? Could caching elsewhere make the
  counter freeze for the WRONG reason?
- tolerance masking: bands so wide the defect fits inside (MC coverage
  bands, rtol choices vs the actual error magnitudes)?
- fixtures whose parameter regime makes the tested term vanish
  (penalty ~ 0, gap ~ 0, gauge_dim 0 where gauge logic is the subject)?
- the sandwich coverage test: is the heteroskedastic fixture's
  discrimination claim itself verified inside the test, or assumed?
Report AT MOST your 4 highest-severity findings.`,
  },
]

phase('Find')
log('fanning out 4 lens finders over the change-set')
const found = await parallel(
  LENSES.map(l => () =>
    agent(l.prompt, { label: `find:${l.key}`, phase: 'Find', schema: FINDINGS_SCHEMA })
  )
)

// Barrier justified: cap + dedup across ALL findings before paying for
// 3 refuters each.
const sevRank = { high: 0, medium: 1, low: 2 }
let all = []
found.forEach((r, i) => {
  if (!r) return
  const lens = LENSES[i].key
  const top = [...r.findings].sort((a, b) => sevRank[a.severity] - sevRank[b.severity]).slice(0, 4)
  top.forEach(f => all.push({ ...f, lens }))
})
// crude dedup on (file, normalized title head)
const seen = new Set()
const deduped = all.filter(f => {
  const k = `${f.file}::${f.title.toLowerCase().slice(0, 40)}`
  if (seen.has(k)) return false
  seen.add(k)
  return true
})
log(`${all.length} findings reported, ${deduped.length} after dedup; refuting each with 3 lenses`)

phase('Verify')
const REFUTER_LENSES = [
  ['logic', 'Attack the REASONING: is the claimed defect actually a defect given the full code context? Read the surrounding code and docs; look for the guard/dispatch/convention the finder missed.'],
  ['evidence', 'Attack the EVIDENCE: verify the cited file:line actually says what the finding claims (read it); if a tiny probe (<60s, taskset -c 32-63, no pytest) can decide it, run one.'],
  ['materiality', 'Attack the MATERIALITY: even if technically true, does it affect any supported usage, shipped result, or documented contract? Internal-only, unreachable, or cosmetic -> refuted as not-a-bug.'],
]
const verified = await parallel(
  deduped.map(f => () =>
    parallel(
      REFUTER_LENSES.map(([key, angle]) => () =>
        agent(
          `${COMMON}
You are an adversarial REFUTER. A finder reported this finding about the
change-set; your job is to KILL IT if it does not deserve to survive.
${angle}

FINDING [lens=${f.lens}, severity=${f.severity}] ${f.title}
Claim: ${f.claim}
Evidence: ${f.evidence}
File: ${f.file}

Default to refuted=true if you cannot positively confirm the defect.
Set refuted=false ONLY if the defect is real AND the evidence checks out
AND it matters.`,
          { label: `refute:${key}:${f.title.slice(0, 30)}`, phase: 'Verify', schema: VERDICT_SCHEMA }
        )
      )
    ).then(votes => {
      const v = votes.filter(Boolean)
      return { ...f, votes: v, survives: v.filter(x => !x.refuted).length >= 2 }
    })
  )
)
const confirmed = verified.filter(Boolean).filter(f => f.survives)
const killed = verified.filter(Boolean).filter(f => !f.survives)
log(`${confirmed.length} findings survived adversarial refutation; ${killed.length} killed`)

phase('Synthesize')
const report = await agent(
  `${COMMON}
Compose the final audit report for the maintainer. Surviving findings
(each survived >=2 of 3 adversarial refuters):
${JSON.stringify(confirmed, null, 2)}

Killed findings (for the appendix -- one line each, with the kill reason):
${JSON.stringify(killed.map(k => ({ title: k.title, lens: k.lens, reasons: k.votes.filter(v => v.refuted).map(v => v.reasoning.slice(0, 200)) })), null, 2)}

Write: (1) a one-paragraph verdict on whether the change-set is sound to
release; (2) surviving findings ordered by severity with recommended
actions (file an issue / fix before tag / document); (3) the
killed-findings appendix; (4) any pattern-level observation across
findings. Be direct; no padding. Your final message IS the report.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { verdict: report, confirmed, killedCount: killed.length, foundCount: all.length }
