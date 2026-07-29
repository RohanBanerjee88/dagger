# Phase 2 results -- phase2_librimix_3spk_finetuned

rows scored: 5400

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.33 | 4.14 | 0.53 |
| ungated_deflation | 41.72 | 0.49 | -3.92 |
| gated_deflation | 42.19 | 1.84 | -2.19 |
| coarse_to_fine | 43.36 | 4.21 | 0.21 |

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
depth 3: coarse_to_fine=0.21 gated_deflation=-2.19 ungated_deflation=-3.92 -- ordering holds: True