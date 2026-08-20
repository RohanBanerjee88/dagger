# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_long2min.csv
arms present: oracle, real, real_forced_m, real_index_order
paired rows loaded: 4800

All differences are PAIRED on matched (scene, speaker, depth, system) rows,
so scene difficulty cancels exactly. SEM is the precision of the mean, not
the spread of the data (CLAUDE.md §7) -- the two differ by ~30x here.

## `real` - `oracle` -- total cost of real diarization

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| no_recursion | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |
| ungated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| ungated_deflation | 2 | 150 | -2.20 | 0.23 | 24% | 9.5 |
| gated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| gated_deflation | 2 | 150 | -2.20 | 0.23 | 24% | 9.5 |
| coarse_to_fine | 1 | 150 | -9.23 | 1.04 | 20% | 8.9 |
| coarse_to_fine | 2 | 150 | -2.92 | 0.23 | 13% | 12.9 |

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
| gated_deflation | 1 | 102 | -2.81 | 0.88 | 32% | 3.2 |
| gated_deflation | 2 | 98 | -5.86 | 1.08 | 26% | 5.4 |

## `real_index_order` - `oracle` -- diarization error alone

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| no_recursion | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |
| ungated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| ungated_deflation | 2 | 150 | -2.46 | 0.18 | 13% | 13.8 |
| gated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| gated_deflation | 2 | 150 | -2.45 | 0.18 | 13% | 13.7 |
| coarse_to_fine | 1 | 150 | -9.23 | 1.04 | 20% | 8.9 |
| coarse_to_fine | 2 | 150 | -2.92 | 0.23 | 13% | 12.9 |

### by accumulation (`n_accepted_before`, deflation systems only)

The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic
difficulty and hits every system equally, which buried the
accumulation effect for five Phase 2 runs.

Levels are the **`oracle`** arm's accumulation position, so each row reads
"for a speaker `oracle` placed at level k, what did `real_index_order` cost it?".
Rows are paired BEFORE they are bucketed -- filtering first would keep only
the speakers whose position coincides across arms, and under real
diarization the ascending-`V_i` sort makes that disagreement the norm
rather than the exception (it is the effect being measured, not noise).

| system | n_accepted_before | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 100 | -6.94 | 0.98 | 12% | 7.1 |
| ungated_deflation | 1 | 100 | -3.11 | 0.88 | 27% | 3.5 |
| ungated_deflation | 2 | 100 | -6.01 | 1.04 | 15% | 5.8 |
| gated_deflation | 0 | 100 | -6.94 | 0.98 | 12% | 7.1 |
| gated_deflation | 1 | 102 | -3.08 | 0.86 | 26% | 3.6 |
| gated_deflation | 2 | 98 | -6.09 | 1.06 | 15% | 5.8 |

## `real` - `real_index_order` -- the reordering V_i induces

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 150 | +0.25 | 0.13 | 50% | 2.0 |
| gated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 150 | +0.25 | 0.13 | 50% | 2.0 |
| coarse_to_fine | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 2 | 150 | +0.00 | 0.00 | 0% | nan |

### by accumulation (`n_accepted_before`, deflation systems only)

The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic
difficulty and hits every system equally, which buried the
accumulation effect for five Phase 2 runs.

Levels are the **`real_index_order`** arm's accumulation position, so each row reads
"for a speaker `real_index_order` placed at level k, what did `real` cost it?".
Rows are paired BEFORE they are bucketed -- filtering first would keep only
the speakers whose position coincides across arms, and under real
diarization the ascending-`V_i` sort makes that disagreement the norm
rather than the exception (it is the effect being measured, not noise).

| system | n_accepted_before | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 100 | -0.09 | 0.10 | 14% | 0.9 |
| ungated_deflation | 1 | 100 | +0.21 | 0.12 | 30% | 1.8 |
| ungated_deflation | 2 | 100 | +0.26 | 0.11 | 31% | 2.3 |
| gated_deflation | 0 | 102 | -0.10 | 0.09 | 14% | 1.0 |
| gated_deflation | 1 | 104 | +0.20 | 0.12 | 30% | 1.8 |
| gated_deflation | 2 | 94 | +0.29 | 0.12 | 32% | 2.4 |

## `real_forced_m` - `real` -- how much of the cost was speaker counting

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 150 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 150 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 2 | 150 | +0.00 | 0.00 | 0% | nan |

### by accumulation (`n_accepted_before`, deflation systems only)

The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic
difficulty and hits every system equally, which buried the
accumulation effect for five Phase 2 runs.

Levels are the **`real`** arm's accumulation position, so each row reads
"for a speaker `real` placed at level k, what did `real_forced_m` cost it?".
Rows are paired BEFORE they are bucketed -- filtering first would keep only
the speakers whose position coincides across arms, and under real
diarization the ascending-`V_i` sort makes that disagreement the norm
rather than the exception (it is the effect being measured, not noise).

| system | n_accepted_before | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 100 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 100 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 100 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 0 | 102 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 1 | 104 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 94 | +0.00 | 0.00 | 0% | nan |

