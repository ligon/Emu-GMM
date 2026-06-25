---
name: numerics
description: "Skill for the Numerics area of Emu-GMM. 8 symbols across 1 files."
---

# Numerics

8 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `tests/`
- Understanding how test_tau_positive_and_kappa_hits_target, test_binding_flag_set_when_tau_large, test_inverse_satisfies_M_star_M_inv_eq_I work
- Modifying numerics-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/numerics/test_ridge_inverse.py` | _ill_conditioned_M, test_tau_positive_and_kappa_hits_target, test_binding_flag_set_when_tau_large, test_inverse_satisfies_M_star_M_inv_eq_I, test_inverse_is_symmetric_well_conditioned (+3) |

## Entry Points

Start here when exploring this area:

- **`test_tau_positive_and_kappa_hits_target`** (Function) — `tests/numerics/test_ridge_inverse.py:88`
- **`test_binding_flag_set_when_tau_large`** (Function) — `tests/numerics/test_ridge_inverse.py:101`
- **`test_inverse_satisfies_M_star_M_inv_eq_I`** (Function) — `tests/numerics/test_ridge_inverse.py:109`
- **`test_inverse_is_symmetric_well_conditioned`** (Function) — `tests/numerics/test_ridge_inverse.py:127`
- **`test_inverse_is_symmetric_ill_conditioned`** (Function) — `tests/numerics/test_ridge_inverse.py:133`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_tau_positive_and_kappa_hits_target` | Function | `tests/numerics/test_ridge_inverse.py` | 88 |
| `test_binding_flag_set_when_tau_large` | Function | `tests/numerics/test_ridge_inverse.py` | 101 |
| `test_inverse_satisfies_M_star_M_inv_eq_I` | Function | `tests/numerics/test_ridge_inverse.py` | 109 |
| `test_inverse_is_symmetric_well_conditioned` | Function | `tests/numerics/test_ridge_inverse.py` | 127 |
| `test_inverse_is_symmetric_ill_conditioned` | Function | `tests/numerics/test_ridge_inverse.py` | 133 |
| `test_keys_present` | Function | `tests/numerics/test_ridge_inverse.py` | 171 |
| `test_types_are_python_scalars` | Function | `tests/numerics/test_ridge_inverse.py` | 176 |
| `_ill_conditioned_M` | Function | `tests/numerics/test_ridge_inverse.py` | 12 |

## How to Explore

1. `gitnexus_context({name: "test_tau_positive_and_kappa_hits_target"})` — see callers and callees
2. `gitnexus_query({query: "numerics"})` — find related execution flows
3. Read key files listed above for implementation details
