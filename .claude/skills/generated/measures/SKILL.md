---
name: measures
description: "Skill for the Measures area of Emu-GMM. 110 symbols across 14 files."
---

# Measures

110 symbols | 14 files | Cohesion: 80%

## When to Use

- Working with code in `tests/`
- Understanding how test_jacobian_is_public, test_jacobian_returns_average_gradient, test_shape work
- Modifying measures-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/measures/test_empirical.py` | test_shape, test_against_analytical, test_jacobian_respects_mask, test_jacobian_nan_safe, test_per_coordinate_missingness (+16) |
| `src/emu_gmm/measures/empirical.py` | _safe_divide, _check_mask_psi_compatibility, expectation_and_contributions, jacobian, jacobian_contributions (+9) |
| `src/emu_gmm/measures/synthetic.py` | _draws, moments_and_contributions, moment_contributions, jacobian_contributions, jacobian (+7) |
| `tests/measures/test_analytical.py` | test_constant_expectation, test_theta_dependent_expectation, test_handles_namedarray_return, test_shape, test_against_analytical_via_ad (+7) |
| `tests/measures/test_synthetic.py` | test_shape, test_against_analytical, test_theta_dependent_sampler_jacobian, test_expectation_jits, compute (+7) |
| `tests/measures/test_empirical_reverse_ad_nan_safety.py` | expect_at, _nan_laden_measure, test_grad_is_finite_on_partial_residual, half_obj, test_jacrev_through_expectation_is_finite (+4) |
| `tests/measures/test_expectation_and_contributions.py` | _make_measure, test_full_mask_uniform_weights, test_masked_weighted, test_shapes_and_dtypes, test_iid_cached_matches_uncached (+2) |
| `tests/measures/test_moments_and_contributions.py` | _make_measure, test_matches_expectation, test_psi_batch_mean_matches_m, test_dtypes, test_cached_matches_uncached (+1) |
| `tests/measures/test_empirical_namedarray_mask.py` | _df, _expected_mask_np, test_dataframe_mask, test_named_array_mask, test_plain_array_mask (+1) |
| `src/emu_gmm/measures/analytical.py` | expectation, fn, _to_plain, jacobian |

## Entry Points

Start here when exploring this area:

- **`test_jacobian_is_public`** (Function) — `tests/test_moment_restriction_primitives.py:63`
- **`test_jacobian_returns_average_gradient`** (Function) — `tests/test_moment_restriction_primitives.py:190`
- **`test_shape`** (Function) — `tests/measures/test_empirical.py:129`
- **`test_against_analytical`** (Function) — `tests/measures/test_empirical.py:139`
- **`test_jacobian_respects_mask`** (Function) — `tests/measures/test_empirical.py:156`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `EmpiricalMeasure` | Class | `src/emu_gmm/measures/empirical.py` | 191 |
| `test_jacobian_is_public` | Function | `tests/test_moment_restriction_primitives.py` | 63 |
| `test_jacobian_returns_average_gradient` | Function | `tests/test_moment_restriction_primitives.py` | 190 |
| `test_shape` | Function | `tests/measures/test_empirical.py` | 129 |
| `test_against_analytical` | Function | `tests/measures/test_empirical.py` | 139 |
| `test_jacobian_respects_mask` | Function | `tests/measures/test_empirical.py` | 156 |
| `test_jacobian_nan_safe` | Function | `tests/measures/test_empirical.py` | 623 |
| `expectation_and_contributions` | Function | `src/emu_gmm/measures/empirical.py` | 213 |
| `jacobian` | Function | `src/emu_gmm/measures/empirical.py` | 404 |
| `jacobian_contributions` | Function | `src/emu_gmm/measures/empirical.py` | 450 |
| `test_matches_expectation` | Function | `tests/measures/test_moments_and_contributions.py` | 52 |
| `test_psi_batch_mean_matches_m` | Function | `tests/measures/test_moments_and_contributions.py` | 59 |
| `test_dtypes` | Function | `tests/measures/test_moments_and_contributions.py` | 67 |
| `test_cached_matches_uncached` | Function | `tests/measures/test_moments_and_contributions.py` | 78 |
| `test_cached_matches_uncached_varied_params` | Function | `tests/measures/test_moments_and_contributions.py` | 89 |
| `moments_and_contributions` | Function | `src/emu_gmm/measures/synthetic.py` | 97 |
| `moment_contributions` | Function | `src/emu_gmm/measures/synthetic.py` | 170 |
| `jacobian_contributions` | Function | `src/emu_gmm/measures/synthetic.py` | 189 |
| `expect_at` | Function | `tests/measures/test_empirical_reverse_ad_nan_safety.py` | 111 |
| `test_per_coordinate_missingness` | Function | `tests/measures/test_empirical.py` | 61 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Fn → Sampler` | cross_community | 5 |
| `Moment_wild_bootstrap → _check_mask_psi_compatibility` | cross_community | 4 |
| `Moment_wild_bootstrap → _safe_divide` | cross_community | 4 |
| `Moment_wild_bootstrap → _finite_cluster_correction` | cross_community | 3 |
| `Moment_wild_bootstrap → _safe_outer_divide` | cross_community | 3 |
| `Moment_contributions → Sampler` | intra_community | 3 |
| `Jacobian_contributions → Sampler` | intra_community | 3 |
| `Grad_at → _to_plain` | intra_community | 3 |
| `Grad_at → _to_plain` | intra_community | 3 |
| `Fn → Expectation_fn` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Covariance | 14 calls |
| Tests | 7 calls |

## How to Explore

1. `gitnexus_context({name: "test_jacobian_is_public"})` — see callers and callees
2. `gitnexus_query({query: "measures"})` — find related execution flows
3. Read key files listed above for implementation details
