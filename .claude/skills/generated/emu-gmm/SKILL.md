---
name: emu-gmm
description: "Skill for the Emu_gmm area of Emu-GMM. 70 symbols across 9 files."
---

# Emu_gmm

70 symbols | 9 files | Cohesion: 80%

## When to Use

- Working with code in `src/`
- Understanding how outer_loop_driver, from_V0, outer_loop_driver work
- Modifying emu_gmm-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `src/emu_gmm/estimator.py` | _apply_anchored, _residual_core, _residual_kernel, _fixed_residual_kernel, _anchored_chol_core (+19) |
| `src/emu_gmm/types.py` | covariance, _walk_components, components, components, functional_se (+14) |
| `src/emu_gmm/optimizer.py` | _optimistix_wrap, _jitted_fn, __call__, _optimistix_status, _supports_args (+6) |
| `src/emu_gmm/runtime.py` | recommended_env, _backend_initialized, _merge_xla_flags, configure, _device_cap_in_effect (+2) |
| `src/emu_gmm/weighting.py` | _gauge_invariant_signature, from_V0, outer_loop_driver |
| `tests/manifolds/test_gamma_leaf_order_117.py` | _dummy_result, test_no_psd_leaf_raises, test_multiple_psd_leaves_raises |
| `tests/test_weighting_outer_loop_protocol.py` | outer_loop_driver |
| `src/emu_gmm/penalty.py` | penalty |
| `src/emu_gmm/inference/j_test.py` | j_test |

## Entry Points

Start here when exploring this area:

- **`outer_loop_driver`** (Function) — `tests/test_weighting_outer_loop_protocol.py:164`
- **`from_V0`** (Function) — `src/emu_gmm/weighting.py:241`
- **`outer_loop_driver`** (Function) — `src/emu_gmm/weighting.py:426`
- **`covariance`** (Function) — `src/emu_gmm/types.py:115`
- **`residual_fn`** (Function) — `src/emu_gmm/estimator.py:665`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `outer_loop_driver` | Function | `tests/test_weighting_outer_loop_protocol.py` | 164 |
| `from_V0` | Function | `src/emu_gmm/weighting.py` | 241 |
| `outer_loop_driver` | Function | `src/emu_gmm/weighting.py` | 426 |
| `covariance` | Function | `src/emu_gmm/types.py` | 115 |
| `residual_fn` | Function | `src/emu_gmm/estimator.py` | 665 |
| `components` | Function | `src/emu_gmm/types.py` | 487 |
| `components` | Function | `src/emu_gmm/types.py` | 618 |
| `functional_se` | Function | `src/emu_gmm/types.py` | 653 |
| `gamma_covariance` | Function | `src/emu_gmm/types.py` | 711 |
| `gamma_se` | Function | `src/emu_gmm/types.py` | 728 |
| `eigenvalue_se` | Function | `src/emu_gmm/types.py` | 752 |
| `test_no_psd_leaf_raises` | Function | `tests/manifolds/test_gamma_leaf_order_117.py` | 192 |
| `test_multiple_psd_leaves_raises` | Function | `tests/manifolds/test_gamma_leaf_order_117.py` | 198 |
| `build_estimator` | Function | `src/emu_gmm/estimator.py` | 267 |
| `estimate` | Function | `src/emu_gmm/estimator.py` | 1255 |
| `penalty` | Function | `src/emu_gmm/penalty.py` | 59 |
| `whitening_residual` | Function | `src/emu_gmm/types.py` | 151 |
| `expectation` | Function | `src/emu_gmm/types.py` | 99 |
| `apply` | Function | `src/emu_gmm/types.py` | 170 |
| `j_test` | Function | `src/emu_gmm/inference/j_test.py` | 95 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `__call__ → _fallback_table` | cross_community | 5 |
| `Estimate → _all_euclidean` | intra_community | 4 |
| `Estimate → _is_riemannian_optimizer` | intra_community | 4 |
| `_compute_inference → Whitening_residual` | intra_community | 4 |
| `K_statistic → Expectation` | cross_community | 3 |
| `Estimate → _resolve_parameters` | intra_community | 3 |
| `Estimate → _sample_observation` | intra_community | 3 |
| `Estimate → _make_residual_fn` | cross_community | 3 |
| `Functional_se → _walk_components` | intra_community | 3 |
| `Eigenvalue_se → _walk_components` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| _internal | 2 calls |
| Inference | 1 calls |

## How to Explore

1. `gitnexus_context({name: "outer_loop_driver"})` — see callers and callees
2. `gitnexus_query({query: "emu_gmm"})` — find related execution flows
3. Read key files listed above for implementation details
