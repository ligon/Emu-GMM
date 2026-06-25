---
name: validation
description: "Skill for the Validation area of Emu-GMM. 23 symbols across 1 files."
---

# Validation

23 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `scripts/`
- Understanding how model_for, make_design, draw_measure work
- Modifying validation-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `scripts/validation/ladder_mc.py` | model_for, make_design, draw_measure, covariance_arm, run_arm (+18) |

## Entry Points

Start here when exploring this area:

- **`model_for`** (Function) — `scripts/validation/ladder_mc.py:148`
- **`make_design`** (Function) — `scripts/validation/ladder_mc.py:217`
- **`draw_measure`** (Function) — `scripts/validation/ladder_mc.py:252`
- **`covariance_arm`** (Function) — `scripts/validation/ladder_mc.py:311`
- **`run_arm`** (Function) — `scripts/validation/ladder_mc.py:401`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `model_for` | Function | `scripts/validation/ladder_mc.py` | 148 |
| `make_design` | Function | `scripts/validation/ladder_mc.py` | 217 |
| `draw_measure` | Function | `scripts/validation/ladder_mc.py` | 252 |
| `covariance_arm` | Function | `scripts/validation/ladder_mc.py` | 311 |
| `run_arm` | Function | `scripts/validation/ladder_mc.py` | 401 |
| `run_arm_per_rep_anchor` | Function | `scripts/validation/ladder_mc.py` | 453 |
| `paired_dof_readout` | Function | `scripts/validation/ladder_mc.py` | 586 |
| `jadj_readout` | Function | `scripts/validation/ladder_mc.py` | 611 |
| `h_boundary_readout` | Function | `scripts/validation/ladder_mc.py` | 654 |
| `study_size_iid_vs_cluster` | Function | `scripts/validation/ladder_mc.py` | 690 |
| `study_stratified` | Function | `scripts/validation/ladder_mc.py` | 727 |
| `study_design_aware` | Function | `scripts/validation/ladder_mc.py` | 769 |
| `study_misspec_power` | Function | `scripts/validation/ladder_mc.py` | 814 |
| `study_ridge_binding` | Function | `scripts/validation/ladder_mc.py` | 843 |
| `study_misspec_steps` | Function | `scripts/validation/ladder_mc.py` | 911 |
| `validity` | Function | `scripts/validation/ladder_mc.py` | 381 |
| `has_hidden_invalidity` | Function | `scripts/validation/ladder_mc.py` | 396 |
| `org_row` | Function | `scripts/validation/ladder_mc.py` | 528 |
| `main` | Function | `scripts/validation/ladder_mc.py` | 1022 |
| `_b1_cover_indicator` | Function | `scripts/validation/ladder_mc.py` | 575 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Validity` | intra_community | 4 |
| `Study_design_aware → Covariance_arm` | intra_community | 3 |
| `Study_design_aware → Model_for` | intra_community | 3 |
| `Study_design_aware → Draw_measure` | intra_community | 3 |
| `Study_design_aware → _b1_cover_indicator` | intra_community | 3 |
| `Study_design_aware → _ks` | intra_community | 3 |
| `Study_size_iid_vs_cluster → Model_for` | intra_community | 3 |
| `Study_size_iid_vs_cluster → Draw_measure` | intra_community | 3 |
| `Study_size_iid_vs_cluster → Covariance_arm` | intra_community | 3 |
| `Study_size_iid_vs_cluster → _b1_cover_indicator` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "model_for"})` — see callers and callees
2. `gitnexus_query({query: "validation"})` — find related execution flows
3. Read key files listed above for implementation details
