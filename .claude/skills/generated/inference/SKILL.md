---
name: inference
description: "Skill for the Inference area of Emu-GMM. 179 symbols across 13 files."
---

# Inference

179 symbols | 13 files | Cohesion: 89%

## When to Use

- Working with code in `tests/`
- Understanding how test_p_value_near_chi_square_calibration, test_p_value_distribution_across_seeds, test_returns_wild_bootstrap_result work
- Modifying inference-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/inference/test_k_statistic.py` | _run, test_J_is_zero, test_K_is_zero, test_S_is_zero, test_df_K_is_p (+46) |
| `tests/inference/test_j_test.py` | _inputs, test_jit_J_stat_finite_and_matches_eager, run, test_jit_returns_traced_pvalue, test_vmap_over_theta_null_returns_batched_J_stat (+26) |
| `tests/inference/test_cluster_bootstrap.py` | _euler_cluster_setup, test_basic_shapes, test_dtype_is_float64, test_key_is_echoed, test_low_boot_n_completes (+20) |
| `tests/inference/test_wild_bootstrap.py` | _build_h0_setup, test_p_value_near_chi_square_calibration, test_p_value_distribution_across_seeds, test_returns_wild_bootstrap_result, test_mammen_sign_path (+12) |
| `tests/inference/test_k_statistic_gauge.py` | _make_params, _orthogonal, _truth, _empirical_measure, fixture (+8) |
| `src/emu_gmm/inference/functional_se.py` | _gamma_from_components, vech_indices, gamma_vech, gamma_se, count_nonzero_eigenvalues (+8) |
| `src/emu_gmm/inference/k_statistic.py` | _to_plain, _safe_outer_divide_jm, _sigma_jm_iid_from_contributions, _sigma_jm_clustered_from_contributions, _N_j_from_measure (+6) |
| `src/emu_gmm/inference/confidence_set.py` | _connected_runs, _interp_edge, _classify, k_confidence_set, _eval |
| `src/emu_gmm/inference/adaptive.py` | maritz_jarrett_quantile_se, evaluate, evaluate, adaptive_bootstrap |
| `src/emu_gmm/inference/wild_bootstrap.py` | _per_obs_signs, _bootstrap_moment, one_replicate |

## Entry Points

Start here when exploring this area:

- **`test_p_value_near_chi_square_calibration`** (Function) — `tests/inference/test_wild_bootstrap.py:155`
- **`test_p_value_distribution_across_seeds`** (Function) — `tests/inference/test_wild_bootstrap.py:182`
- **`test_returns_wild_bootstrap_result`** (Function) — `tests/inference/test_wild_bootstrap.py:374`
- **`test_mammen_sign_path`** (Function) — `tests/inference/test_wild_bootstrap.py:391`
- **`test_invalid_sign_raises`** (Function) — `tests/inference/test_wild_bootstrap.py:407`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_p_value_near_chi_square_calibration` | Function | `tests/inference/test_wild_bootstrap.py` | 155 |
| `test_p_value_distribution_across_seeds` | Function | `tests/inference/test_wild_bootstrap.py` | 182 |
| `test_returns_wild_bootstrap_result` | Function | `tests/inference/test_wild_bootstrap.py` | 374 |
| `test_mammen_sign_path` | Function | `tests/inference/test_wild_bootstrap.py` | 391 |
| `test_invalid_sign_raises` | Function | `tests/inference/test_wild_bootstrap.py` | 407 |
| `test_non_positive_n_boot_raises` | Function | `tests/inference/test_wild_bootstrap.py` | 420 |
| `test_user_supplied_V_overrides_recompute` | Function | `tests/inference/test_wild_bootstrap.py` | 432 |
| `test_analytical_measure_raises_typed_error` | Function | `tests/inference/test_wild_bootstrap.py` | 451 |
| `test_synthetic_measure_raises_typed_error` | Function | `tests/inference/test_wild_bootstrap.py` | 471 |
| `test_jit_returns_traced_scalars` | Function | `tests/inference/test_wild_bootstrap.py` | 521 |
| `run` | Function | `tests/inference/test_wild_bootstrap.py` | 528 |
| `test_vmap_over_keys_returns_batched_pvalues` | Function | `tests/inference/test_wild_bootstrap.py` | 551 |
| `test_jit_then_vmap_composes` | Function | `tests/inference/test_wild_bootstrap.py` | 578 |
| `test_p_value_is_traced_array` | Function | `tests/inference/test_wild_bootstrap.py` | 598 |
| `test_namedarray_V_passes_through` | Function | `tests/inference/test_wild_bootstrap.py` | 637 |
| `test_namedarray_V_from_estimation_result_shape` | Function | `tests/inference/test_wild_bootstrap.py` | 674 |
| `test_basic_shapes` | Function | `tests/inference/test_cluster_bootstrap.py` | 138 |
| `test_dtype_is_float64` | Function | `tests/inference/test_cluster_bootstrap.py` | 160 |
| `test_key_is_echoed` | Function | `tests/inference/test_cluster_bootstrap.py` | 177 |
| `test_low_boot_n_completes` | Function | `tests/inference/test_cluster_bootstrap.py` | 396 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `K_statistic → _safe_outer_divide_jm` | cross_community | 5 |
| `K_statistic → _to_plain` | cross_community | 4 |
| `K_statistic → _N_j_from_measure` | cross_community | 4 |
| `G → _gamma_from_components` | cross_community | 4 |
| `K_statistic → _stats_from_whitened` | cross_community | 3 |
| `K_statistic → Expectation` | cross_community | 3 |
| `Gamma_se → _component_shapes` | cross_community | 3 |
| `Gamma_se → _flatten_components` | cross_community | 3 |
| `Gamma_se → _gamma_from_components` | intra_community | 3 |
| `Gamma_se → Vech_indices` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Emu_gmm | 4 calls |

## How to Explore

1. `gitnexus_context({name: "test_p_value_near_chi_square_calibration"})` — see callers and callees
2. `gitnexus_query({query: "inference"})` — find related execution flows
3. Read key files listed above for implementation details
