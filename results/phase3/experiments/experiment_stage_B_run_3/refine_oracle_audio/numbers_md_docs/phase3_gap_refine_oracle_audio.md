# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_refine_oracle_audio.csv
arms present: oracle, real
paired rows loaded: 1200

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
| no_recursion | 75 | -3.01 | 0.27 | 9% | 11.2 |
| ungated_deflation | 75 | -2.20 | 0.31 | 23% | 7.1 |
| gated_deflation | 75 | -2.19 | 0.31 | 23% | 7.1 |
| coarse_to_fine | 75 | -2.98 | 0.27 | 9% | 10.9 |

#### whole output track, one global scale

Level-sensitive by construction -- read it as a level check,
not as a summary of the per-depth tables.

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 75 | -1.53 | 1.01 | 39% | 1.5 |
| ungated_deflation | 75 | -1.73 | 1.10 | 32% | 1.6 |
| gated_deflation | 75 | -1.81 | 1.08 | 32% | 1.7 |
| coarse_to_fine | 75 | -0.22 | 1.47 | 39% | 0.1 |

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

