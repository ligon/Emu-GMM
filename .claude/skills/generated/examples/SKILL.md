---
name: examples
description: "Skill for the Examples area of Emu-GMM. 44 symbols across 10 files."
---

# Examples

44 symbols | 10 files | Cohesion: 100%

## When to Use

- Working with code in `examples/`
- Understanding how make_panel, within_demean, within_ols work
- Modifying examples-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `examples/coin_plus_noise.py` | coin_plus_noise_sampler_factory, coin_plus_noise_data, run_synthetic, run_empirical, _print_header (+5) |
| `examples/panel_regression.py` | make_panel, within_demean, within_ols, hc0_se, crve_se (+2) |
| `examples/run_euler.py` | _print_header, _print_result, run_synthetic, run_analytical, run_empirical (+1) |
| `examples/hierarchical.py` | simulate, build_measure, make_residual, run, main |
| `examples/twfe.py` | _draw_fixed_effects, make_panel, run_twfe, main |
| `examples/empirical_bootstrap.py` | make_dataset, fit, run, main |
| `examples/fair_coin.py` | make_coin_data, run_fair_coin, main |
| `tests/examples/test_coin_plus_noise.py` | _load_coin_plus_noise_module, coin_module |
| `src/emu_gmm/examples/euler.py` | euler_sampler_factory, euler_data |
| `tests/examples/test_fair_coin.py` | result |

## Entry Points

Start here when exploring this area:

- **`make_panel`** (Function) — `examples/panel_regression.py:71`
- **`within_demean`** (Function) — `examples/panel_regression.py:97`
- **`within_ols`** (Function) — `examples/panel_regression.py:112`
- **`hc0_se`** (Function) — `examples/panel_regression.py:119`
- **`crve_se`** (Function) — `examples/panel_regression.py:125`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `make_panel` | Function | `examples/panel_regression.py` | 71 |
| `within_demean` | Function | `examples/panel_regression.py` | 97 |
| `within_ols` | Function | `examples/panel_regression.py` | 112 |
| `hc0_se` | Function | `examples/panel_regression.py` | 119 |
| `crve_se` | Function | `examples/panel_regression.py` | 125 |
| `emu_panel_fit` | Function | `examples/panel_regression.py` | 141 |
| `main` | Function | `examples/panel_regression.py` | 179 |
| `coin_plus_noise_sampler_factory` | Function | `examples/coin_plus_noise.py` | 124 |
| `coin_plus_noise_data` | Function | `examples/coin_plus_noise.py` | 148 |
| `run_synthetic` | Function | `examples/coin_plus_noise.py` | 164 |
| `run_empirical` | Function | `examples/coin_plus_noise.py` | 181 |
| `main` | Function | `examples/coin_plus_noise.py` | 222 |
| `run_synthetic` | Function | `examples/run_euler.py` | 82 |
| `run_analytical` | Function | `examples/run_euler.py` | 99 |
| `run_empirical` | Function | `examples/run_euler.py` | 115 |
| `main` | Function | `examples/run_euler.py` | 136 |
| `simulate` | Function | `examples/hierarchical.py` | 120 |
| `build_measure` | Function | `examples/hierarchical.py` | 142 |
| `make_residual` | Function | `examples/hierarchical.py` | 176 |
| `run` | Function | `examples/hierarchical.py` | 207 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Coin_plus_noise_sampler_factory` | intra_community | 4 |
| `Main → _draw_fixed_effects` | intra_community | 4 |
| `Main → _print_header` | intra_community | 3 |
| `Main → _print_result` | intra_community | 3 |
| `Main → Simulate` | intra_community | 3 |
| `Main → Build_measure` | intra_community | 3 |
| `Main → Make_residual` | intra_community | 3 |
| `Main → Make_coin_data` | intra_community | 3 |
| `Main → Make_dataset` | intra_community | 3 |
| `Main → Fit` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "make_panel"})` — see callers and callees
2. `gitnexus_query({query: "examples"})` — find related execution flows
3. Read key files listed above for implementation details
