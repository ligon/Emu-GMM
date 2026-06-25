<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Emu-GMM** (8837 symbols, 14502 relationships, 126 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Emu-GMM/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Emu-GMM/clusters` | All functional areas |
| `gitnexus://repo/Emu-GMM/processes` | All execution flows |
| `gitnexus://repo/Emu-GMM/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Manifolds area (459 symbols) | `.claude/skills/generated/manifolds/SKILL.md` |
| Work in the Tests area (329 symbols) | `.claude/skills/generated/tests/SKILL.md` |
| Work in the Inference area (179 symbols) | `.claude/skills/generated/inference/SKILL.md` |
| Work in the Covariance area (122 symbols) | `.claude/skills/generated/covariance/SKILL.md` |
| Work in the Studies area (112 symbols) | `.claude/skills/generated/studies/SKILL.md` |
| Work in the Measures area (110 symbols) | `.claude/skills/generated/measures/SKILL.md` |
| Work in the _internal area (75 symbols) | `.claude/skills/generated/internal/SKILL.md` |
| Work in the Emu_gmm area (70 symbols) | `.claude/skills/generated/emu-gmm/SKILL.md` |
| Work in the Examples area (44 symbols) | `.claude/skills/generated/examples/SKILL.md` |
| Work in the Validation area (23 symbols) | `.claude/skills/generated/validation/SKILL.md` |
| Work in the Numerics area (8 symbols) | `.claude/skills/generated/numerics/SKILL.md` |
| Work in the Reviews area (7 symbols) | `.claude/skills/generated/reviews/SKILL.md` |
| Work in the Scripts area (4 symbols) | `.claude/skills/generated/scripts/SKILL.md` |

<!-- gitnexus:end -->
