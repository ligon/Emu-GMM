# Prior-Art Ledger — `emu_gmm.studies` / empirical-law queries

Living, git-tracked inventory of the machinery, definitions, and conventions
in force in the Monte-Carlo / repeated-sampling area (`emu_gmm.studies`,
`emu_gmm.inference`) — the "terms of the debate" any new estimator/statistic/
query in this area must cite. Per the `prior-art-ledger` skill: edit in place
(current state, not a log); git history is the journal.

- **Search tier used:** gitnexus 1.6.3 for the reinvention check (index built
  this session: 8,833 nodes / 14,500 edges / 272 clusters); ripgrep + git for
  `path:line` pinning. cq not queried this pass.
- **Scope of this ledger:** the empirical-law surface — repeated-sampling
  records, their summarizers, and the conditional/coupled queries over them.
  First populated to ground **#167** (conditional/coupled queries) under the
  **#144** `EstimatorLaw` umbrella (landed on `main` via **PR #168**, merged
  2026-06-25; the `conditioning.py` `path:line` refs below resolve on `main`).

---

## §1 — Task, restated (repo vocabulary)

`emu_gmm.studies.replicate` runs `key -> dgp(fold_in(key,r)) -> run -> record()`
and stacks per-rep `FitRecord`s into an `MCRecords` (leading `n_reps` axis).
Layer-2 *summarizers* reduce that stack (bias/SD, coverage, size/power, tau,
J-calibration), each **excluding-but-counting** non-converged reps. #167 adds
two missing *query* shapes over the same stack: (a) the **conditional** law —
the sub-record where a diagnostic event holds (`given`) plus its loud size
(`event_share`); and (b) the **coupled** law — two CRN-verified arms and their
paired contrasts (`crn_pair`). These lift three hand-rolled readouts in the
#130 harness (`scripts/validation/ladder_mc.py`) into the package. The umbrella
(#144) is an `EstimatorLaw` interface; #167 is its narrow, point/tuple-only,
additive first increment.

## §2 — Existing machinery (path:line · tested?)

Records / driver (`src/emu_gmm/studies/driver.py`, `src/emu_gmm/types.py`):
- `FitRecord` — `types.py:494`. Stacked per-rep pytree: `theta_flat`,`se`,J-triple,
  `converged`,`tau_realised`,`binding_ridge`,`sigma_meat_indefinite` (0/1 floats),
  static `J_dof`,`param_names`. Tested: `tests/studies/test_summaries.py`,
  `tests/test_types.py`.
- `MCRecords` — `driver.py:46`; `.converged_mask` (`:75`, `>0.5`), `.n_converged`
  (`:80`), `.n_excluded` (`:85`), `.to_pandas` (`:94`). Tested:
  `tests/studies/test_replicate.py`.
- `replicate` — `driver.py:122`; CRN: rep `r` sees `fold_in(key,r)` (`:264`).
  Tested: `tests/studies/test_replicate.py`.
- `monte_carlo_study` / `StudyResult` — `study.py:65` / `:35`. Tested:
  `tests/studies/test_study_smoke.py`.

Summarizers (`src/emu_gmm/studies/summaries.py`, all tested in
`tests/studies/test_summaries.py`):
- `_stacked` (`:34`, unwrap `MCRecords|FitRecord`, **no validation** — trusts caller),
  `_used` (`:41`, converged mask `>0.5` + counts),
  `bias_sd` (`:93`), `coverage` (`:146`), `size_power` (`:198`),
  `tau_binding` (`:245`), `j_calibration` (`:295`).

Bootstrap functional algebra (the #144 "already a query algebra" prior art):
- `BootstrapMean/SE/Quantile/PValue` — `src/emu_gmm/inference/adaptive.py:134/150/173/204`
  (`evaluate(values)->(value,mcse)` + `label`). Tested under `tests/inference/`.

Conditional/coupled queries — **#167, NEW** (`src/emu_gmm/studies/conditioning.py`,
all tested in `tests/studies/test_conditioning.py`):
- `given` (`:127`), `event_share`/`EventShare` (`:190`/`:164`),
  `crn_pair`/`CoupledRecords` (`:293`/`:235`) with `both_finite`(`:245`)/
  `flips`(`:249`)/`paired_diff`(`:268`)/`mean_paired_diff`(`:286`), `Flips`(`:220`),
  helpers `_unwrap`(`:66`)/`_event_mask`(`:82`)/`FLAG_FIELDS`(`:59`).

Consumers retrofitted: `scripts/validation/ladder_mc.py` —
`paired_dof_readout`→`crn_pair`, `jadj_readout`→`given`+`event_share`.

## §3 — Definitions & conventions in force (quote, don't paraphrase)

- **Convergence/flag threshold is `> 0.5`.** `summaries._used` `mask = np.asarray(rec.converged) > 0.5`
  (`summaries.py:43`); `MCRecords.converged_mask` `> 0.5` (`driver.py:86`). #167
  matches it for all flags (`conditioning.py:104,199`). Any new flag selection
  MUST use `> 0.5`, not `> 0`.
- **CRN contract.** "Replicate `r` always sees `jax.random.fold_in(key, r)`. Two
  studies (arms) run with the same master `key` and the same `dgp` therefore
  estimate on **identical draws**" (`driver.py:21-28`). `MCRecords.key` is the
  *master* key only.
- **Per-coordinate `N_j` scaling (CLAUDE.md commitment 9).** Moments are
  per-coordinate means; `V_X` carries `1/(N_jN_k)`; the criterion → `chi^2_{M-K}`
  with **no explicit N**. Never `×N` a criterion "to match a textbook."
- **#140 event-as-product / exclude-but-count.** A replication yields θ̂ *and*
  named-event flags; summarizers exclude-but-count non-converged and surface
  `n_used`/`n_excluded`/`n_valid_se` rather than folding events into rates
  (`summaries.py:8-11`, `bias_sd`/`coverage` NaN-SE accounting).
- **`EstimatorLaw` design dispositions** — `docs/design.org:441` (§ "Prospective
  sketch: EstimatorLaw"): grade-dependence, conditional-vs-marginal surfaced,
  couplings verify their probability space, weak-ID guards. #167 implements the
  point/tuple slice.

## §4 — Invariants & assumptions (the landmines)

1. **Event flags are exact-binary.** `record()` casts bool→float64
   (`types.py:1089-1096`), so `>0.5`≡`>0`≡`==1` **today**. Guard:
   `test_flags_are_exact_binary`. A future fractional/NaN flag would split
   `>0.5` vs `>0` and silently drift the byte-equal retrofit.
2. **CRN alignment needs same DGP, not just same key.** Master-key equality is
   *necessary but not sufficient* (the key can't witness the DGP `split` scheme).
   `crn_pair` therefore requires a matching `coupling_id`
   (`MCRecords.coupling_id`, `driver.py:81`) and refuses otherwise unless
   `assert_coupled=True`. Guards: `test_coupling_id_mismatch_refuses`,
   `test_missing_id_refuses_without_assert`.
3. **Masking destroys rep-index alignment.** `given` returns a masked
   `FitRecord`; a conditioned record can never be paired. `crn_pair` rejects
   non-`MCRecords` (`test_rejects_non_mcrecords`). Conditional-paired contrasts
   go through `flips(..., where=event_mask)`, never `pair(given(...))`.
4. **Static fields ride the treedef.** `given`'s `tree_map(lambda l: l[mask],rec)`
   preserves `J_dof`/`param_names` (not leaves). Guard:
   `test_masks_event_and_preserves_static_fields`.
5. **Conditioning on an estimator-internal flag is selection-conditional, NOT
   nominal.** `coverage(given(rec,"binding_ridge"))` is a within-selection
   diagnostic (the event is a function of the same data as θ̂/se), not a
   coverage guarantee. The blessed use is the within-subset nominal-vs-adjusted
   *p-value* contrast (`jadj`). Documented `conditioning.py` module docstring.

## §5 — Reuse decision (per quantity the task needed)

| quantity | decision | reason |
|---|---|---|
| unwrap `MCRecords\|FitRecord` (`_unwrap`) | **new (parallel)** | `summaries._stacked` is private to layer-2 and *trusts* its caller; `given`/`event_share` are public and must validate + reject junk loudly. Reuse would add a cross-layer private import AND drop the guard. (4 lines; confirmed by code-simplifier pass.) |
| converged mask (`>0.5`) in `event_share` | **reuse (convention)** | identical threshold to `_used`/`converged_mask`; reused as a *convention*, not the symbol (those are `MCRecords`-only / private, but `event_share` also accepts a bare `FitRecord`). |
| `n_converged` within subset | **reuse (convention)** | mirrors `MCRecords.n_converged`; recomputed because the subset is a bare `FitRecord`. |
| conditional selection (`given`) | **new** | no existing conditional-subset query over records; summarizers only marginalize. Justified by #144 §3 (the joint Θ×{0,1}^E law). |
| `event_share` (both denominators) | **new** | no existing "size of a selection" with all-vs-converged denominators; new to avoid the denominator-mismatch footgun (§4). |
| `crn_pair`/`flips`/`paired_diff` | **new** | grep confirms no prior finite-mask-and-pair or directional-flip helper outside the retrofitted readouts; net-new CRN-coupled contrast. |
| `coupling_id` field | **extend** | additive static field on existing `MCRecords` (`driver.py:81`) + passthrough in `replicate`/`monte_carlo_study`; closes the same-DGP gap key-equality can't. |
| `Bootstrap*` retrofit | **deferred (not in #167)** | already a query algebra (`adaptive.py`); #144 marks it for a later additive retrofit, explicitly out of #167 scope. |

Verify-against-ledger result for #167: **OK (anchored on §3/§4)** — no
`REINVENTION` (the `_unwrap`/`>0.5` parallels are convention-reuse, justified
above) and no `CONTRADICTION` (thresholds, CRN, exclude-but-count all honored).
Reinvention check at the gitnexus tier (cypher, this session): the **only**
definitions of `given`/`event_share`/`crn_pair`/`flips`/`paired_diff`/
`both_finite`/`mean_paired_diff`/`Flips` repo-wide are in `conditioning.py`
(plus the `paired_dof_readout` consumer in `ladder_mc.py` and the tests) — no
competing implementation by name or concept-synonym. This corroborates the
ripgrep-floor "new" decisions at the call-graph tier.

## §6 — Open questions for the human

1. **Conditioning hazard exposure (§4.5).** `given` is general enough to feed
   `coverage`/`size_power` a selection-conditional answer. Documented, not
   gated. Leave general (a harness legitimately wants conditional coverage as a
   *diagnostic*) or add a louder guard on the estimator-internal flags?
2. **`coupling_id` provenance.** Currently caller-supplied (the harness uses
   `(repr(design.spec), seed)`). Worth auto-deriving from the `dgp` callable
   identity in `replicate`, or keep explicit (clearer, less magic)?
3. **`h_boundary_readout`** was left out of the #167 retrofit: it reconstructs
   the *mask draw stream* — a different carrier (the missingness-field law), not
   a records query. Track as a future ledger entry when that carrier is built?
4. **Ledger path/format.** Placed at `.coder/ledger.md` (skill default, trackable;
   `.coder/` is not gitignored). Repo prose is Org — convert to
   `.coder/ledger.org` if preferred? Kept Markdown per the skill default.
