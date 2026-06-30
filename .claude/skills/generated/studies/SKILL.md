---
name: studies
description: "Skill for the Studies area of Emu-GMM. 112 symbols across 7 files."
---

# Studies

112 symbols | 7 files | Cohesion: 91%

## When to Use

- Working with code in `tests/`
- Understanding how vec, test_exact_arithmetic, test_excludes_but_counts_non_converged work
- Modifying studies-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/studies/test_conditioning.py` | _rec, v, test_masks_event_and_preserves_static_fields, test_partition_identity_including_converged, test_event_share_both_denominators (+26) |
| `tests/studies/test_replicate.py` | _theta0, _make_dgp, _make_run, test_records_carry_leading_rep_axis, test_records_is_a_pytree (+22) |
| `tests/studies/test_summaries.py` | _records, vec, test_exact_arithmetic, test_excludes_but_counts_non_converged, test_theta0_accepts_param_pytree (+19) |
| `src/emu_gmm/studies/summaries.py` | _stacked, _used, size_power, _rates, tau_binding (+5) |
| `src/emu_gmm/studies/conditioning.py` | _unwrap, _n_reps, _event_mask, given, _event_label (+5) |
| `tests/studies/test_study_smoke.py` | _theta0, _balanced_dgp, _run, test_thirty_rep_smoke, test_study_composes_replicate_and_summarizers (+4) |
| `src/emu_gmm/studies/driver.py` | to_pandas |

## Entry Points

Start here when exploring this area:

- **`vec`** (Function) — `tests/studies/test_summaries.py:37`
- **`test_exact_arithmetic`** (Function) — `tests/studies/test_summaries.py:57`
- **`test_excludes_but_counts_non_converged`** (Function) — `tests/studies/test_summaries.py:72`
- **`test_theta0_accepts_param_pytree`** (Function) — `tests/studies/test_summaries.py:80`
- **`test_theta0_wrong_length_raises`** (Function) — `tests/studies/test_summaries.py:86`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `vec` | Function | `tests/studies/test_summaries.py` | 37 |
| `test_exact_arithmetic` | Function | `tests/studies/test_summaries.py` | 57 |
| `test_excludes_but_counts_non_converged` | Function | `tests/studies/test_summaries.py` | 72 |
| `test_theta0_accepts_param_pytree` | Function | `tests/studies/test_summaries.py` | 80 |
| `test_theta0_wrong_length_raises` | Function | `tests/studies/test_summaries.py` | 86 |
| `test_all_non_converged_yields_nans` | Function | `tests/studies/test_summaries.py` | 90 |
| `test_single_used_rep_has_nan_mc_sd` | Function | `tests/studies/test_summaries.py` | 98 |
| `test_exact_coverage` | Function | `tests/studies/test_summaries.py` | 106 |
| `test_excludes_non_converged` | Function | `tests/studies/test_summaries.py` | 117 |
| `test_bad_level_raises` | Function | `tests/studies/test_summaries.py` | 124 |
| `test_exact_rejection_rates` | Function | `tests/studies/test_summaries.py` | 130 |
| `test_excludes_non_converged` | Function | `tests/studies/test_summaries.py` | 143 |
| `test_frequency_and_quantiles` | Function | `tests/studies/test_summaries.py` | 156 |
| `test_all_non_converged` | Function | `tests/studies/test_summaries.py` | 169 |
| `test_exact_ecdf_deviation` | Function | `tests/studies/test_summaries.py` | 178 |
| `test_skewed_pvalues_show_deviation` | Function | `tests/studies/test_summaries.py` | 192 |
| `test_summarizers_accept_bare_fitrecord` | Function | `tests/studies/test_summaries.py` | 202 |
| `v` | Function | `tests/studies/test_conditioning.py` | 41 |
| `test_masks_event_and_preserves_static_fields` | Function | `tests/studies/test_conditioning.py` | 75 |
| `test_partition_identity_including_converged` | Function | `tests/studies/test_conditioning.py` | 87 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Event_share → _n_reps` | intra_community | 3 |
| `Given → _n_reps` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "vec"})` — see callers and callees
2. `gitnexus_query({query: "studies"})` — find related execution flows
3. Read key files listed above for implementation details
