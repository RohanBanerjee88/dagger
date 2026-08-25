# Phase 3 -- oracle-vs-real gap

sources: phase3_librimix_3spk_vi_on.csv
arms present: oracle, real, real_index_order
paired rows loaded: 3600

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
| no_recursion | 150 | -3.18 | 0.21 | 9% | 15.2 |
| ungated_deflation | 150 | -2.26 | 0.23 | 24% | 9.8 |
| gated_deflation | 150 | -2.24 | 0.23 | 24% | 9.7 |
| coarse_to_fine | 150 | -2.93 | 0.23 | 14% | 13.0 |

#### whole output track, one global scale

Level-sensitive by construction -- read it as a level check,
not as a summary of the per-depth tables.

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 150 | -1.02 | 0.80 | 38% | 1.3 |
| ungated_deflation | 150 | -1.97 | 0.86 | 37% | 2.3 |
| gated_deflation | 150 | -2.18 | 0.84 | 36% | 2.6 |
| coarse_to_fine | 150 | -0.61 | 0.91 | 39% | 0.7 |

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| no_recursion | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |
| ungated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| ungated_deflation | 2 | 150 | -2.20 | 0.23 | 24% | 9.5 |
| gated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| gated_deflation | 2 | 150 | -2.18 | 0.23 | 24% | 9.4 |
| coarse_to_fine | 1 | 150 | -9.23 | 1.04 | 20% | 8.9 |
| coarse_to_fine | 2 | 150 | -2.86 | 0.23 | 14% | 12.6 |

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

## `real_index_order` - `oracle` -- diarization error alone

### overall (un-stratified, whole output track)

Is this arm NET better or worse? The per-depth tables below say
*where* the difference lives; this one says whether it adds up.
Never read it INSTEAD of them (§6.4), and never optimize against it:
it is scale-anchored by the bit-exact solo copy.

#### pooled across depths (the exchange rate)

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 150 | -3.18 | 0.21 | 9% | 15.2 |
| ungated_deflation | 150 | -2.52 | 0.18 | 13% | 14.1 |
| gated_deflation | 150 | -2.28 | 0.19 | 18% | 11.9 |
| coarse_to_fine | 150 | -2.93 | 0.23 | 14% | 13.0 |

#### whole output track, one global scale

Level-sensitive by construction -- read it as a level check,
not as a summary of the per-depth tables.

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 150 | -1.02 | 0.80 | 38% | 1.3 |
| ungated_deflation | 150 | -1.29 | 0.60 | 27% | 2.2 |
| gated_deflation | 150 | -2.15 | 0.67 | 26% | 3.2 |
| coarse_to_fine | 150 | -0.61 | 0.91 | 39% | 0.7 |

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| no_recursion | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |
| ungated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| ungated_deflation | 2 | 150 | -2.46 | 0.18 | 13% | 13.8 |
| gated_deflation | 1 | 150 | -8.25 | 1.07 | 23% | 7.7 |
| gated_deflation | 2 | 150 | -2.22 | 0.19 | 18% | 11.6 |
| coarse_to_fine | 1 | 150 | -9.23 | 1.04 | 20% | 8.9 |
| coarse_to_fine | 2 | 150 | -2.86 | 0.23 | 14% | 12.6 |

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
| gated_deflation | 1 | 102 | -2.98 | 0.87 | 29% | 3.4 |
| gated_deflation | 2 | 98 | -5.83 | 1.07 | 20% | 5.5 |

## `real` - `real_index_order` -- the reordering V_i induces

### overall (un-stratified, whole output track)

Is this arm NET better or worse? The per-depth tables below say
*where* the difference lives; this one says whether it adds up.
Never read it INSTEAD of them (§6.4), and never optimize against it:
it is scale-anchored by the bit-exact solo copy.

#### pooled across depths (the exchange rate)

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 150 | +0.26 | 0.13 | 50% | 2.0 |
| gated_deflation | 150 | +0.04 | 0.12 | 40% | 0.3 |
| coarse_to_fine | 150 | +0.00 | 0.00 | 0% | nan |

#### whole output track, one global scale

Level-sensitive by construction -- read it as a level check,
not as a summary of the per-depth tables.

| system | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|
| no_recursion | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 150 | -0.69 | 0.77 | 46% | 0.9 |
| gated_deflation | 150 | -0.03 | 0.67 | 47% | 0.0 |
| coarse_to_fine | 150 | +0.00 | 0.00 | 0% | nan |

### by overlap depth

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 150 | +0.25 | 0.13 | 50% | 2.0 |
| gated_deflation | 1 | 150 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 150 | +0.03 | 0.12 | 40% | 0.3 |
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
| gated_deflation | 0 | 124 | -0.17 | 0.09 | 13% | 1.8 |
| gated_deflation | 1 | 118 | +0.14 | 0.10 | 25% | 1.4 |
| gated_deflation | 2 | 58 | +0.15 | 0.14 | 26% | 1.1 |

## `real_forced_m` - `real`

(arm not present in this run)

