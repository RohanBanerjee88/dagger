# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_dilation_smoke.csv
arms present: oracle, real
paired rows loaded: 144

> These CSVs sweep `dilate_overlap_ms` over [0.0, 50.0] ms. This
> report covers the **0 ms baseline only** -- every other value is a
> different pipeline, and averaging across them would label a mean of
> pipelines as a property of one. See the sweep section in
> `run_phase3.py`'s own `.md` for the dilation comparison.

All differences are PAIRED on matched (scene, speaker, depth, system) rows,
so scene difficulty cancels exactly. SEM is the precision of the mean, not
the spread of the data (CLAUDE.md §7) -- the two differ by ~30x here.

## `real` - `oracle` -- total cost of real diarization

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 9 | -4.34 | 3.40 | 44% | 1.3 |
| no_recursion | 2 | 9 | -2.97 | 0.92 | 22% | 3.2 |
| ungated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| ungated_deflation | 2 | 9 | -2.26 | 0.90 | 33% | 2.5 |
| gated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| gated_deflation | 2 | 9 | -2.26 | 0.90 | 33% | 2.5 |
| coarse_to_fine | 1 | 9 | -4.43 | 3.38 | 44% | 1.3 |
| coarse_to_fine | 2 | 9 | -2.26 | 0.88 | 22% | 2.6 |

### by accumulation (`n_accepted_before`, deflation systems only)

The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic
difficulty and hits every system equally, which buried the
accumulation effect for five Phase 2 runs.

Levels are the **`oracle`** arm's accumulation position, so each row reads
"for a speaker `oracle` placed at level k, what did `real` cost it?".
Rows are paired BEFORE they are bucketed -- filtering first would keep only
the speakers whose position coincides across arms, and under real
diarization the ascending-`V_i` sort makes that disagreement the norm
rather than the exception (it is the effect being measured, not noise).

| system | n_accepted_before | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 6 | -3.96 | 2.95 | 33% | 1.3 |
| ungated_deflation | 1 | 6 | -2.55 | 4.06 | 50% | 0.6 |
| ungated_deflation | 2 | 6 | -1.48 | 3.28 | 50% | 0.5 |
| gated_deflation | 0 | 6 | -3.96 | 2.95 | 33% | 1.3 |
| gated_deflation | 1 | 6 | -2.55 | 4.06 | 50% | 0.6 |
| gated_deflation | 2 | 6 | -1.48 | 3.28 | 50% | 0.5 |

## `real_index_order` - `oracle`

(arm not present in this run)

## `real` - `real_index_order`

(arm not present in this run)

## `real_forced_m` - `real`

(arm not present in this run)

