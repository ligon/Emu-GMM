---
name: covariance
description: "Skill for the Covariance area of Emu-GMM. 122 symbols across 14 files."
---

# Covariance

122 symbols | 14 files | Cohesion: 86%

## When to Use

- Working with code in `tests/`
- Understanding how build_mixed, test_design_aware_mixed_runs_finite_and_symmetric, test_design_aware_VTS_estimated_not_zero_matches_numpy work
- Modifying covariance-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/covariance/test_stratified.py` | _design_and_sampling, build_mixed, _numpy_cluster_cross, _mixed_dac_and_measure, test_design_aware_mixed_runs_finite_and_symmetric (+52) |
| `tests/covariance/test_sum.py` | _numpy_clustered, _numpy_two_way, build_two_way, _two_way, _clustered (+10) |
| `tests/covariance/test_clustered.py` | test_two_clusters_known_values, test_nan_in_psi_at_masked_cells_does_not_poison, test_covariance_jits, compute, test_cached_self_parity_with_correction (+3) |
| `src/emu_gmm/covariance/stratified.py` | covariance, _fpc_factor, _safe_outer_divide, covariance, _cross_corners (+3) |
| `tests/covariance/test_synthetic.py` | test_shape, test_symmetric, test_psd, test_scaling_matches_variance_of_mean, test_larger_n_gives_smaller_V (+2) |
| `tests/covariance/test_analytical.py` | test_constant_covariance, test_theta_dependent_covariance, test_measure_argument_ignored, test_handles_namedarray_return, test_covariance_jits (+1) |
| `src/emu_gmm/covariance/clustered.py` | _safe_outer_divide, covariance, _finite_cluster_correction, _to_plain, psi_at |
| `src/emu_gmm/covariance/iid.py` | _safe_outer_divide, covariance, _to_plain, psi_at |
| `src/emu_gmm/covariance/synthetic.py` | covariance, _to_plain, psi_at |
| `tests/covariance/test_iid.py` | test_covariance_jits, compute, test_two_moment_psi_runs |

## Entry Points

Start here when exploring this area:

- **`build_mixed`** (Function) — `tests/covariance/test_stratified.py:875`
- **`test_design_aware_mixed_runs_finite_and_symmetric`** (Function) — `tests/covariance/test_stratified.py:941`
- **`test_design_aware_VTS_estimated_not_zero_matches_numpy`** (Function) — `tests/covariance/test_stratified.py:949`
- **`test_design_aware_block_structure`** (Function) — `tests/covariance/test_stratified.py:964`
- **`test_design_aware_mixed_cached_self_parity`** (Function) — `tests/covariance/test_stratified.py:987`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `build_mixed` | Function | `tests/covariance/test_stratified.py` | 875 |
| `test_design_aware_mixed_runs_finite_and_symmetric` | Function | `tests/covariance/test_stratified.py` | 941 |
| `test_design_aware_VTS_estimated_not_zero_matches_numpy` | Function | `tests/covariance/test_stratified.py` | 949 |
| `test_design_aware_block_structure` | Function | `tests/covariance/test_stratified.py` | 964 |
| `test_design_aware_mixed_cached_self_parity` | Function | `tests/covariance/test_stratified.py` | 987 |
| `test_design_aware_VTS_uses_sampling_cluster_unit` | Function | `tests/covariance/test_stratified.py` | 1004 |
| `test_design_aware_fpc_enters_VTT_only` | Function | `tests/covariance/test_stratified.py` | 1039 |
| `test_design_aware_couple_false_zeroes_cross_corners` | Function | `tests/covariance/test_stratified.py` | 1070 |
| `test_design_aware_couple_false_keeps_diagonal_blocks_bit_identical` | Function | `tests/covariance/test_stratified.py` | 1089 |
| `test_design_aware_cross_block_decomposition_identity` | Function | `tests/covariance/test_stratified.py` | 1108 |
| `test_design_aware_cross_block_matches_numpy_reference` | Function | `tests/covariance/test_stratified.py` | 1130 |
| `test_design_aware_couple_default_is_true_bitwise_unchanged` | Function | `tests/covariance/test_stratified.py` | 1145 |
| `test_design_aware_all_design_cross_block_is_zero` | Function | `tests/covariance/test_stratified.py` | 1164 |
| `test_design_aware_cross_block_cached_self_parity` | Function | `tests/covariance/test_stratified.py` | 1183 |
| `build_mixed_masked` | Function | `tests/covariance/test_stratified.py` | 1206 |
| `test_design_aware_VTS_inherits_sampling_dof_correction_full_mask` | Function | `tests/covariance/test_stratified.py` | 1252 |
| `test_design_aware_VTS_dof_correction_is_per_pair_under_missingness` | Function | `tests/covariance/test_stratified.py` | 1296 |
| `test_design_aware_dof_decomposition_identity_holds` | Function | `tests/covariance/test_stratified.py` | 1322 |
| `test_design_aware_dof_default_off_is_bitwise_unchanged` | Function | `tests/covariance/test_stratified.py` | 1346 |
| `test_design_aware_mixed_assembly_under_missingness_matches_numpy` | Function | `tests/covariance/test_stratified.py` | 1371 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Moment_wild_bootstrap → _finite_cluster_correction` | cross_community | 3 |
| `Moment_wild_bootstrap → _safe_outer_divide` | cross_community | 3 |
| `Covariance → _fpc_factor` | cross_community | 3 |
| `Covariance → _safe_outer_divide` | cross_community | 3 |
| `Cross_block → _safe_outer_divide` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Measures | 6 calls |
| Tests | 5 calls |

## How to Explore

1. `gitnexus_context({name: "build_mixed"})` — see callers and callees
2. `gitnexus_query({query: "covariance"})` — find related execution flows
3. Read key files listed above for implementation details
