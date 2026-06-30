---
name: reviews
description: "Skill for the Reviews area of Emu-GMM. 7 symbols across 1 files."
---

# Reviews

7 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `docs/`
- Understanding how total, run work
- Modifying reviews-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `docs/reviews/pymanopt_parity_harness.py` | total, _as_numpy, _diff_report, _allclose, _call_emu (+2) |

## Entry Points

Start here when exploring this area:

- **`total`** (Function) — `docs/reviews/pymanopt_parity_harness.py:106`
- **`run`** (Function) — `docs/reviews/pymanopt_parity_harness.py:218`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `total` | Function | `docs/reviews/pymanopt_parity_harness.py` | 106 |
| `run` | Function | `docs/reviews/pymanopt_parity_harness.py` | 218 |
| `_as_numpy` | Function | `docs/reviews/pymanopt_parity_harness.py` | 115 |
| `_diff_report` | Function | `docs/reviews/pymanopt_parity_harness.py` | 123 |
| `_allclose` | Function | `docs/reviews/pymanopt_parity_harness.py` | 150 |
| `_call_emu` | Function | `docs/reviews/pymanopt_parity_harness.py` | 159 |
| `_call_pymanopt` | Function | `docs/reviews/pymanopt_parity_harness.py` | 194 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run → _as_numpy` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "total"})` — see callers and callees
2. `gitnexus_query({query: "reviews"})` — find related execution flows
3. Read key files listed above for implementation details
