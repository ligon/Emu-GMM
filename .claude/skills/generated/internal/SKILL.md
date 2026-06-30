---
name: internal
description: "Skill for the _internal area of Emu-GMM. 75 symbols across 12 files."
---

# _internal

75 symbols | 12 files | Cohesion: 91%

## When to Use

- Working with code in `tests/`
- Understanding how test_returns_lower_triangular, test_factorisation_identity, test_matches_scipy work
- Modifying _internal-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/_internal/test_cholesky.py` | _make_pd_5, test_returns_lower_triangular, test_factorisation_identity, test_matches_scipy, test_basic (+8) |
| `tests/_internal/test_params_spec_builder_phase2.py` | _make_product, test_psd5x2_plus_scalar, test_psd5x3_plus_scalar, test_block_boundary_invariant_holds, test_product_spec_hashable (+8) |
| `tests/_internal/test_fn_cache.py` | _make_closure, test_second_lookup_returns_same_object, test_secondary_keys_are_independent, test_distinct_functions_do_not_share, test_entry_dies_with_function (+7) |
| `tests/_internal/test_params_manifold_leaf.py` | _make_product, test_acceptance_shape_and_dtype, test_K3_round_trip, test_block_width_invariant, test_leaf_spec_order_and_offsets (+7) |
| `src/emu_gmm/_internal/labels.py` | _is_pandas_frame, _is_pandas_series, _is_haliax_named, normalise_x, normalise_weights (+1) |
| `src/emu_gmm/_internal/params.py` | flatten_params, _walk_leaf_specs, flatten_params_with_spec, manifold_spec_from_params, flatten_params_for_ad |
| `src/emu_gmm/_internal/cholesky.py` | cholesky, forward_solve, whiten, quadratic_form |
| `src/emu_gmm/_internal/fn_cache.py` | get_or_build, _table, _fallback_table |
| `tests/_internal/test_params_v2.py` | test_usable_as_jit_static_argument, f |
| `tests/_internal/test_params.py` | test_can_jit, double_then_reconstruct |

## Entry Points

Start here when exploring this area:

- **`test_returns_lower_triangular`** (Function) — `tests/_internal/test_cholesky.py:21`
- **`test_factorisation_identity`** (Function) — `tests/_internal/test_cholesky.py:28`
- **`test_matches_scipy`** (Function) — `tests/_internal/test_cholesky.py:33`
- **`test_basic`** (Function) — `tests/_internal/test_cholesky.py:41`
- **`test_matches_scipy`** (Function) — `tests/_internal/test_cholesky.py:48`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_returns_lower_triangular` | Function | `tests/_internal/test_cholesky.py` | 21 |
| `test_factorisation_identity` | Function | `tests/_internal/test_cholesky.py` | 28 |
| `test_matches_scipy` | Function | `tests/_internal/test_cholesky.py` | 33 |
| `test_basic` | Function | `tests/_internal/test_cholesky.py` | 41 |
| `test_matches_scipy` | Function | `tests/_internal/test_cholesky.py` | 48 |
| `test_basic` | Function | `tests/_internal/test_cholesky.py` | 58 |
| `test_matches_scipy` | Function | `tests/_internal/test_cholesky.py` | 65 |
| `test_identity_of_squared_norm` | Function | `tests/_internal/test_cholesky.py` | 77 |
| `test_zero_vector` | Function | `tests/_internal/test_cholesky.py` | 88 |
| `test_gradient_is_V_inv_m` | Function | `tests/_internal/test_cholesky.py` | 94 |
| `test_matches_explicit` | Function | `tests/_internal/test_cholesky.py` | 110 |
| `test_whiten_jits` | Function | `tests/_internal/test_cholesky.py` | 119 |
| `test_second_lookup_returns_same_object` | Function | `tests/_internal/test_fn_cache.py` | 39 |
| `test_secondary_keys_are_independent` | Function | `tests/_internal/test_fn_cache.py` | 46 |
| `test_distinct_functions_do_not_share` | Function | `tests/_internal/test_fn_cache.py` | 54 |
| `test_entry_dies_with_function` | Function | `tests/_internal/test_fn_cache.py` | 63 |
| `test_bound_method_does_not_inherit_plain_function_entry` | Function | `tests/_internal/test_fn_cache.py` | 105 |
| `get_or_build` | Function | `src/emu_gmm/_internal/fn_cache.py` | 77 |
| `test_acceptance_shape_and_dtype` | Function | `tests/_internal/test_params_manifold_leaf.py` | 199 |
| `test_K3_round_trip` | Function | `tests/_internal/test_params_manifold_leaf.py` | 231 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `__call__ → _fallback_table` | cross_community | 5 |
| `__call__ → _fallback_table` | cross_community | 4 |
| `Flatten_params_for_ad → _walk_leaf_specs` | intra_community | 3 |
| `Quadratic_form → Cholesky` | intra_community | 3 |
| `Quadratic_form → Forward_solve` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "test_returns_lower_triangular"})` — see callers and callees
2. `gitnexus_query({query: "_internal"})` — find related execution flows
3. Read key files listed above for implementation details
