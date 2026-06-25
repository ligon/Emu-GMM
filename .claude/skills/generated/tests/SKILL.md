---
name: tests
description: "Skill for the Tests area of Emu-GMM. 329 symbols across 32 files."
---

# Tests

329 symbols | 32 files | Cohesion: 89%

## When to Use

- Working with code in `tests/`
- Understanding how test_estimation_result, test_returns_expected_keys, test_sigma_theta_dataframe work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_weighting.py` | _make_pd, test_returns_m_unchanged, test_ignores_V, test_jits, compute (+26) |
| `tests/test_penalty.py` | _make_measure, _run, test_theta_hat_identical_to_no_penalty_kwarg, test_beta_smaller_with_penalty, test_gamma_smaller_with_penalty (+23) |
| `tests/test_types.py` | _make_result, test_estimation_result, test_returns_expected_keys, test_sigma_theta_dataframe, test_v_x_dataframe (+21) |
| `tests/test_regularization.py` | _ill_conditioned_V, test_ill_conditioned_input_brings_kappa_under_target, test_returns_v_star_eq_v_plus_tau_diag, test_scale_equivariance_under_per_moment_rescaling, test_scale_equivariance_uniform_rescaling (+21) |
| `tests/test_diagnostics.py` | _stub_optimizer_info, test_basic, test_labelled_per_moment_fields, test_no_nans_in_finite_inputs, test_cond_info_passthrough (+11) |
| `tests/test_estimator_traced_measure.py` | _measure, _theta0, test_traced_path_matches_legacy_closure_path, test_traced_path_matches_bare_estimate, test_fresh_same_structure_measures_share_one_trace (+10) |
| `tests/test_sigma_sandwich.py` | _measure, _theta0, _G_and_V, test_cu_tau_zero_collapses_to_classical, test_identity_weighting_is_the_robust_sandwich (+8) |
| `tests/test_estimator_unequal_nj.py` | _unequal_nj_measure, _run, test_N_j_are_unequal, test_recovers_beta, test_recovers_gamma (+8) |
| `tests/test_pinv_eigvalrule.py` | _spd, test_drop_zero_is_bitwise_inv, test_drops_smallest_eigenvalues_not_largest, test_moore_penrose_property, test_rank_is_d_minus_drop (+7) |
| `tests/test_estimator_empirical.py` | _make_measure, _run, test_recovers_beta, test_recovers_gamma, test_converged (+7) |

## Entry Points

Start here when exploring this area:

- **`test_estimation_result`** (Function) — `tests/test_types.py:170`
- **`test_returns_expected_keys`** (Function) — `tests/test_types.py:185`
- **`test_sigma_theta_dataframe`** (Function) — `tests/test_types.py:197`
- **`test_v_x_dataframe`** (Function) — `tests/test_types.py:206`
- **`test_n_j_series`** (Function) — `tests/test_types.py:213`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_estimation_result` | Function | `tests/test_types.py` | 170 |
| `test_returns_expected_keys` | Function | `tests/test_types.py` | 185 |
| `test_sigma_theta_dataframe` | Function | `tests/test_types.py` | 197 |
| `test_v_x_dataframe` | Function | `tests/test_types.py` | 206 |
| `test_n_j_series` | Function | `tests/test_types.py` | 213 |
| `test_moment_residual_series` | Function | `tests/test_types.py` | 221 |
| `test_summary_contains_expected_fields` | Function | `tests/test_types.py` | 227 |
| `test_standard_errors_is_named_array` | Function | `tests/test_types.py` | 245 |
| `test_standard_errors_match_sqrt_diag` | Function | `tests/test_types.py` | 252 |
| `test_standard_errors_cached` | Function | `tests/test_types.py` | 265 |
| `test_coef_table_is_dataframe_with_four_columns` | Function | `tests/test_types.py` | 272 |
| `test_coef_table_populates_all_columns` | Function | `tests/test_types.py` | 280 |
| `test_coef_table_estimate_column_matches_theta_hat` | Function | `tests/test_types.py` | 289 |
| `test_coef_table_std_error_matches_sqrt_diag` | Function | `tests/test_types.py` | 299 |
| `test_coef_table_t_stat_is_estimate_over_std_error` | Function | `tests/test_types.py` | 307 |
| `test_coef_table_p_value_two_sided_normal` | Function | `tests/test_types.py` | 314 |
| `test_to_pandas_exposes_coef_table_as_coefficients` | Function | `tests/test_types.py` | 325 |
| `test_result_labels_type_resolves_to_public_path` | Function | `tests/test_types.py` | 383 |
| `test_round_trip` | Function | `tests/test_types.py` | 416 |
| `test_main_namespace_warns_at_save_time` | Function | `tests/test_types.py` | 444 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Ridge_inverse → _spectrum` | intra_community | 4 |
| `Bisect_step → _spectrum` | intra_community | 3 |
| `Ridge_inverse → _is_diagonal` | cross_community | 3 |
| `Ridge_inverse → _apply_tau` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Covariance | 8 calls |
| Measures | 6 calls |

## How to Explore

1. `gitnexus_context({name: "test_estimation_result"})` — see callers and callees
2. `gitnexus_query({query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
