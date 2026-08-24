# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_dilation_sweep.csv
arms present: oracle, real
paired rows loaded: 1200

> These CSVs sweep `dilate_overlap_ms` over [0.0, 200.0, 400.0, 800.0] ms. This
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
| no_recursion | 1 | 75 | -6.73 | 1.20 | 24% | 5.6 |
| no_recursion | 2 | 75 | -2.98 | 0.27 | 9% | 11.2 |
| ungated_deflation | 1 | 75 | -5.69 | 1.28 | 31% | 4.5 |
| ungated_deflation | 2 | 75 | -2.18 | 0.31 | 23% | 7.1 |
| gated_deflation | 1 | 75 | -5.69 | 1.28 | 31% | 4.5 |
| gated_deflation | 2 | 75 | -2.16 | 0.31 | 23% | 7.1 |
| coarse_to_fine | 1 | 75 | -6.82 | 1.21 | 27% | 5.6 |
| coarse_to_fine | 2 | 75 | -2.96 | 0.27 | 9% | 10.8 |

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
| ungated_deflation | 0 | 50 | -5.31 | 1.12 | 20% | 4.8 |
| ungated_deflation | 1 | 50 | -2.80 | 1.30 | 28% | 2.2 |
| ungated_deflation | 2 | 50 | -3.68 | 1.05 | 32% | 3.5 |
| gated_deflation | 0 | 50 | -5.29 | 1.12 | 20% | 4.7 |
| gated_deflation | 1 | 50 | -2.80 | 1.30 | 28% | 2.2 |
| gated_deflation | 2 | 50 | -3.68 | 1.05 | 32% | 3.5 |

## `real_index_order` - `oracle`

(arm not present in this run)

## `real` - `real_index_order`

(arm not present in this run)

## `real_forced_m` - `real`

(arm not present in this run)

