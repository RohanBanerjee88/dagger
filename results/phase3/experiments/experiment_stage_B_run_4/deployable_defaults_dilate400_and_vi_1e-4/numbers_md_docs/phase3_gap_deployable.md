# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_deployable.csv
arms present: oracle, real
paired rows loaded: 2400

> These CSVs sweep `dilate_overlap_ms` over [0.0, 400.0] ms. This
> report covers the **0 ms baseline only** -- every other value is a
> different pipeline, and averaging across them would label a mean of
> pipelines as a property of one. See the sweep section in
> `run_phase3.py`'s own `.md` for the dilation comparison.

All differences are PAIRED on matched (scene, speaker, depth, system) rows,
so scene difficulty cancels exactly. SEM is the precision of the mean, not
the spread of the data (CLAUDE.md §7) -- the two differ by ~30x here.

## `real` - `oracle` -- total cost of real diarization

### overall (un-stratified, whole output track)

Is this arm NET better or worse? The per-depth tables below say
*where* the difference lives; this one says whether it adds up.
Never read it INSTEAD of them (§6.4), and never optimize against it:
it is scale-anchored by the bit-exact solo copy.

#### pooled across depths (the exchange rate)

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 300 | -0.91 | 0.22 | 42% | 4.1 |
| ungated_deflation | 300 | -0.50 | 0.22 | 47% | 2.3 |
| gated_deflation | 300 | -0.47 | 0.22 | 48% | 2.1 |
| coarse_to_fine | 300 | -0.91 | 0.22 | 42% | 4.1 |

#### whole output track, one global scale

Level-sensitive by construction -- read it as a level check,
not as a summary of the per-depth tables.

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 300 | -0.44 | 0.46 | 38% | 1.0 |
| ungated_deflation | 300 | -1.74 | 0.56 | 38% | 3.1 |
| gated_deflation | 300 | -1.94 | 0.55 | 37% | 3.5 |
| coarse_to_fine | 300 | -0.44 | 0.46 | 38% | 1.0 |

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| no_recursion | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |
| ungated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| ungated_deflation | 2 | 150 | -2.20 | 0.23 | 24% | 9.5 |
| gated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| gated_deflation | 2 | 150 | -2.18 | 0.23 | 24% | 9.4 |
| coarse_to_fine | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| coarse_to_fine | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |

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
| ungated_deflation | 0 | 100 | -7.05 | 0.98 | 13% | 7.2 |
| ungated_deflation | 1 | 100 | -2.84 | 0.89 | 33% | 3.2 |
| ungated_deflation | 2 | 100 | -5.79 | 1.06 | 25% | 5.5 |
| gated_deflation | 0 | 100 | -7.06 | 0.98 | 13% | 7.2 |
| gated_deflation | 1 | 102 | -2.83 | 0.88 | 31% | 3.2 |
| gated_deflation | 2 | 98 | -5.82 | 1.08 | 27% | 5.4 |

## `real_index_order` - `oracle`

(arm not present in this run)

## `real` - `real_index_order`

(arm not present in this run)

## `real_forced_m` - `real`

(arm not present in this run)

