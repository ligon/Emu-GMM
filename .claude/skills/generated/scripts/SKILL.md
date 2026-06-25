---
name: scripts
description: "Skill for the Scripts area of Emu-GMM. 4 symbols across 1 files."
---

# Scripts

4 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `scripts/`
- Understanding how main work
- Modifying scripts-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `scripts/freeze_seasonality_extract.py` | _git_describe, _run, _column_checksums, main |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `scripts/freeze_seasonality_extract.py:75`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `main` | Function | `scripts/freeze_seasonality_extract.py` | 75 |
| `_git_describe` | Function | `scripts/freeze_seasonality_extract.py` | 46 |
| `_run` | Function | `scripts/freeze_seasonality_extract.py` | 47 |
| `_column_checksums` | Function | `scripts/freeze_seasonality_extract.py` | 62 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _run` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "main"})` — see callers and callees
2. `gitnexus_query({query: "scripts"})` — find related execution flows
3. Read key files listed above for implementation details
