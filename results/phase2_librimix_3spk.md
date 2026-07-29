# Phase 2 results -- phase2_librimix_3spk

rows scored: 5400

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.16 | 2.96 | -0.59 |
| ungated_deflation | 39.60 | -1.28 | -6.05 |
| gated_deflation | 40.90 | 0.12 | -3.96 |
| coarse_to_fine | 43.17 | 2.35 | -1.33 |

## Diagnostic counts (per system/depth: absent / perfect / failed / scored)

| system | depth | absent (nan) | perfect (+inf) | failed (-inf) | scored |
|---|---|---|---|---|---|
| no_recursion | 1 | 0 | 198 | 0 | 450 |
| no_recursion | 2 | 150 | 0 | 0 | 300 |
| no_recursion | 3 | 0 | 0 | 0 | 450 |
| ungated_deflation | 1 | 0 | 198 | 0 | 450 |
| ungated_deflation | 2 | 150 | 0 | 0 | 300 |
| ungated_deflation | 3 | 0 | 0 | 0 | 450 |
| gated_deflation | 1 | 0 | 198 | 0 | 450 |
| gated_deflation | 2 | 150 | 0 | 0 | 300 |
| gated_deflation | 3 | 0 | 0 | 0 | 450 |
| coarse_to_fine | 1 | 0 | 198 | 0 | 450 |
| coarse_to_fine | 2 | 150 | 0 | 0 | 300 |
| coarse_to_fine | 3 | 0 | 0 | 0 | 450 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 3: coarse_to_fine=-1.33 gated_deflation=-3.96 ungated_deflation=-6.05 -- ordering holds: True
