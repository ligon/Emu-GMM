# Vendored: pr-review-toolkit

These agents and the `/review-pr` command are vendored verbatim from the
**`pr-review-toolkit`** plugin of Anthropic's official Claude Code plugin
marketplace (`claude-plugins-official`), so they are available as first-class
**project agents** in this repo without depending on the marketplace being
installed on a given machine.

- Source plugin: `pr-review-toolkit` (author: Anthropic).
- Vendored: 2026-06-25.
- Licence: Apache-2.0 (see `LICENSE` in this directory).

## What's here

Agents (`.claude/agents/`) — invoke via the Agent tool by `name`:

| agent | role |
|---|---|
| `code-reviewer` | CLAUDE.md compliance + general bug/quality review of a diff |
| `code-simplifier` | simplify recently-changed code, reuse existing helpers, preserve behaviour |
| `comment-analyzer` | comment accuracy / rot / doc completeness |
| `pr-test-analyzer` | behavioural test coverage + critical gaps |
| `silent-failure-hunter` | silent failures, swallowed errors, bad fallbacks |
| `type-design-analyzer` | encapsulation + invariant expression of new/changed types |

Command (`.claude/commands/`):

- `review-pr` — orchestrates the agents over the current diff. Invoke as
  **`/review-pr`** (a project command; the upstream docs show the namespaced
  `/pr-review-toolkit:review-pr`, which is the marketplace form — the vendored
  copy is un-namespaced).

## Local caveats

- The agents defer to **this repo's `CLAUDE.md`** as the authority on
  conventions (float64, `jax_dataclasses`, src/ layout, "spare set of correct
  interfaces — don't reinvent"). That overriding instruction is what makes them
  usable here.
- **`code-simplifier`** is the one agent that carries upstream **JS/React-
  flavoured example bullets** (ES modules, arrow-vs-`function`, React component
  patterns) that do not apply to this Python/JAX codebase. Its core mandate
  (preserve behaviour; reuse existing tools; clarity over brevity; follow
  CLAUDE.md) is language-agnostic and still correct here; only the illustrative
  bullets are off-base. Localise to Python/JAX if it proves distracting in use.
- Re-syncing from upstream is a clean re-copy of `agents/*.md` +
  `commands/review-pr.md` + `LICENSE` from the marketplace plugin; keep local
  edits minimal so that stays true.
