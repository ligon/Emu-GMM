---
name: manifolds
description: "Skill for the Manifolds area of Emu-GMM. 459 symbols across 36 files."
---

# Manifolds

459 symbols | 36 files | Cohesion: 88%

## When to Use

- Working with code in `tests/`
- Understanding how point, test_mvn_recovery_via_parameter_space_class, test_point_no_default_requires_seed work
- Modifying manifolds-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/manifolds/test_rtr_hvp.py` | test_psd_hvp_self_adjoint_on_horizontal, _make_psd_target, _make_gauge_invariant_Q, _make_gauge_variant_Q, test_psd_additive_recipe_is_gauge_conditional (+30) |
| `src/emu_gmm/manifolds/riemannian_tr.py` | _is_manifold, _project_flat, _raise_index_flat, _hvp_flat, _hvp_per_leaf (+29) |
| `tests/manifolds/test_rtr_reductions.py` | _hvp, _Q_connection, Qp, Qpp, test_hvp_equals_fd_geodesic_second_derivative (+22) |
| `tests/manifolds/test_rtr_gauge_numerical.py` | test_returned_eta_has_no_vertical_component, hvp, test_vertical_only_gradient_is_horizontally_stationary, _target_for, _components (+20) |
| `tests/manifolds/test_rtr_integration.py` | _make_params, _moment_count, _make_dgp, test_converged_true_under_jit_for_easy_fixture, test_result_pytree_reflattens_bit_identically (+20) |
| `tests/manifolds/test_rtr_trust_region.py` | _geodesic_second_derivative_positive, q, test_hvp_positive_carries_affine_connection_term, _make_scale_measure, _info_get (+18) |
| `tests/manifolds/test_rtr_tcg.py` | _psd_params, _gauge_invariant_residual, test_hvp_symmetric_on_horizontal, hvp, test_negative_curvature_sign_matches_dense_hessian (+16) |
| `tests/manifolds/test_parameter_space.py` | _tril_idx, _moment_count, _make_model, _make_data, _sample_moments (+15) |
| `tests/manifolds/test_manifold_acceptance_phase7.py` | _make_params, _moment_count, _orthogonal, _make_dgp, _estimate (+12) |
| `tests/manifolds/test_rtr_pymanopt_parity.py` | _make_params, _orthogonal, _make_synthetic_measure, _nonconvex_target, _estimate_tr (+11) |

## Entry Points

Start here when exploring this area:

- **`point`** (Function) — `src/emu_gmm/parameter_space.py:211`
- **`test_mvn_recovery_via_parameter_space_class`** (Function) — `tests/manifolds/test_parameter_space.py:100`
- **`test_point_no_default_requires_seed`** (Function) — `tests/manifolds/test_parameter_space.py:151`
- **`test_two_random_seeds_recover_same_gauge_invariant_gamma`** (Function) — `tests/manifolds/test_parameter_space.py:186`
- **`test_warm_start_round_trips`** (Function) — `tests/manifolds/test_parameter_space.py:228`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `TestVerticalGradientNegativeControl` | Class | `tests/manifolds/test_rtr_gauge_numerical.py` | 622 |
| `ParameterSpace` | Class | `src/emu_gmm/parameter_space.py` | 110 |
| `Normal` | Class | `tests/manifolds/test_parameter_space.py` | 106 |
| `Bare` | Class | `tests/manifolds/test_parameter_space.py` | 154 |
| `Scalars` | Class | `tests/manifolds/test_parameter_space.py` | 322 |
| `Base` | Class | `tests/manifolds/test_parameter_space.py` | 371 |
| `Derived` | Class | `tests/manifolds/test_parameter_space.py` | 374 |
| `Mixed` | Class | `tests/manifolds/test_parameter_space.py` | 407 |
| `MeanSpace` | Class | `tests/manifolds/test_euclidean_nonscalar_110.py` | 134 |
| `point` | Function | `src/emu_gmm/parameter_space.py` | 211 |
| `test_mvn_recovery_via_parameter_space_class` | Function | `tests/manifolds/test_parameter_space.py` | 100 |
| `test_point_no_default_requires_seed` | Function | `tests/manifolds/test_parameter_space.py` | 151 |
| `test_two_random_seeds_recover_same_gauge_invariant_gamma` | Function | `tests/manifolds/test_parameter_space.py` | 186 |
| `test_warm_start_round_trips` | Function | `tests/manifolds/test_parameter_space.py` | 228 |
| `test_theta_init_alias_is_bitwise_identical_to_parameters` | Function | `tests/manifolds/test_parameter_space.py` | 260 |
| `test_passing_both_parameters_and_theta_init_errors` | Function | `tests/manifolds/test_parameter_space.py` | 291 |
| `test_subclass_merges_inherited_fields` | Function | `tests/manifolds/test_parameter_space.py` | 364 |
| `test_mixed_annotation_preserves_declaration_order` | Function | `tests/manifolds/test_parameter_space.py` | 404 |
| `test_run_callable_first_arg_is_polymorphic` | Function | `tests/manifolds/test_parameter_space.py` | 476 |
| `q` | Function | `tests/manifolds/test_rtr_trust_region.py` | 583 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Body_fun → _is_positive` | cross_community | 7 |
| `_truncated_cg → _is_positive` | cross_community | 7 |
| `__call__ → _is_positive` | cross_community | 6 |
| `Body_fun → _project_flat` | cross_community | 5 |
| `__call__ → _block` | cross_community | 5 |
| `_riemannian_hvp → _is_positive` | cross_community | 5 |
| `_truncated_cg → _project_flat` | cross_community | 5 |
| `__call__ → Q_data` | cross_community | 5 |
| `Body_fun → Pick` | cross_community | 4 |
| `Body_fun → _is_concrete` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Emu_gmm | 8 calls |
| _internal | 2 calls |

## How to Explore

1. `gitnexus_context({name: "point"})` — see callers and callees
2. `gitnexus_query({query: "manifolds"})` — find related execution flows
3. Read key files listed above for implementation details
