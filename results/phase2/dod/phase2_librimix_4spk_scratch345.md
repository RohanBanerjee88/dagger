# Phase 2 results -- phase2_librimix_4spk_scratch345

rows scored: 9600

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| no_recursion | 45.17 | 1.33 | -3.47 | -3.35 |
| ungated_deflation | 40.89 | -1.30 | -5.17 | -4.73 |
| gated_deflation | 43.35 | 0.04 | -4.49 | -4.20 |
| coarse_to_fine | 45.18 | 0.81 | -3.88 | -3.59 |

## Diagnostic counts (per system/depth: absent / perfect / failed / scored)

| system | depth | absent (nan) | perfect (+inf) | failed (-inf) | scored |
|---|---|---|---|---|---|
| no_recursion | 1 | 0 | 329 | 0 | 600 |
| no_recursion | 2 | 300 | 0 | 0 | 300 |
| no_recursion | 3 | 150 | 0 | 0 | 450 |
| no_recursion | 4 | 0 | 0 | 0 | 600 |
| ungated_deflation | 1 | 0 | 329 | 0 | 600 |
| ungated_deflation | 2 | 300 | 0 | 0 | 300 |
| ungated_deflation | 3 | 150 | 0 | 0 | 450 |
| ungated_deflation | 4 | 0 | 0 | 0 | 600 |
| gated_deflation | 1 | 0 | 329 | 0 | 600 |
| gated_deflation | 2 | 300 | 0 | 0 | 300 |
| gated_deflation | 3 | 150 | 0 | 0 | 450 |
| gated_deflation | 4 | 0 | 0 | 0 | 600 |
| coarse_to_fine | 1 | 0 | 329 | 0 | 600 |
| coarse_to_fine | 2 | 300 | 0 | 0 | 300 |
| coarse_to_fine | 3 | 150 | 0 | 0 | 450 |
| coarse_to_fine | 4 | 0 | 0 | 0 | 600 |

## Spread (per system/depth)

(p95 saturates at the +-50 dB clip wherever the diagnostic-counts table shows many perfect/failed rows -- at depth 1 most rows are solo copy-through and legitimately +inf, so a p95 of exactly the cap there is expected, not a bug)

| system | depth | n | mean | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| no_recursion | 1 | 600 | 45.17 | 28.03 | 50.00 | 16.68 | 50.00 |
| no_recursion | 2 | 300 | 1.33 | -17.85 | 14.73 | -50.00 | 22.97 |
| no_recursion | 3 | 450 | -3.47 | -26.44 | 8.37 | -50.00 | 22.83 |
| no_recursion | 4 | 600 | -3.35 | -9.58 | 2.69 | -16.35 | 7.70 |
| ungated_deflation | 1 | 600 | 40.89 | 11.85 | 50.00 | 0.20 | 50.00 |
| ungated_deflation | 2 | 300 | -1.30 | -17.90 | 10.36 | -50.00 | 19.17 |
| ungated_deflation | 3 | 450 | -5.17 | -28.31 | 5.32 | -50.00 | 20.40 |
| ungated_deflation | 4 | 600 | -4.73 | -10.26 | 0.94 | -17.50 | 7.65 |
| gated_deflation | 1 | 600 | 43.35 | 21.11 | 50.00 | 2.67 | 50.00 |
| gated_deflation | 2 | 300 | 0.04 | -18.04 | 12.24 | -50.00 | 19.17 |
| gated_deflation | 3 | 450 | -4.49 | -27.90 | 7.39 | -50.00 | 20.40 |
| gated_deflation | 4 | 600 | -4.20 | -10.16 | 1.60 | -16.35 | 7.65 |
| coarse_to_fine | 1 | 600 | 45.18 | 28.68 | 50.00 | 16.68 | 50.00 |
| coarse_to_fine | 2 | 300 | 0.81 | -17.70 | 13.31 | -50.00 | 19.87 |
| coarse_to_fine | 3 | 450 | -3.88 | -27.11 | 7.92 | -50.00 | 23.58 |
| coarse_to_fine | 4 | 600 | -3.59 | -10.25 | 2.68 | -16.35 | 7.33 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|---|
| ungated_deflation | 0 | 49.97 (n=150) | -2.02 (n=50) | -5.87 (n=102) | -3.86 (n=150) |
| ungated_deflation | 1 | 50.00 (n=150) | -5.15 (n=55) | -5.21 (n=92) | -4.76 (n=150) |
| ungated_deflation | 2 | 27.35 (n=150) | -1.85 (n=50) | -7.59 (n=107) | -5.21 (n=150) |
| ungated_deflation | 3 | 36.24 (n=150) | 0.60 (n=145) | -2.93 (n=149) | -5.09 (n=150) |
| gated_deflation | 0 | 47.85 (n=295) | -0.54 (n=113) | -4.40 (n=205) | -3.59 (n=295) |
| gated_deflation | 1 | 41.97 (n=199) | 0.12 (n=110) | -5.17 (n=151) | -4.74 (n=199) |
| gated_deflation | 2 | 33.44 (n=95) | 1.01 (n=66) | -3.96 (n=83) | -4.80 (n=95) |
| gated_deflation | 3 | 32.85 (n=11) | -0.66 (n=11) | -0.65 (n=11) | -5.62 (n=11) |
| no_recursion | n/a | 45.17 (n=600) | 1.33 (n=300) | -3.47 (n=450) | -3.35 (n=600) |
| coarse_to_fine | n/a | 45.18 (n=600) | 0.81 (n=300) | -3.88 (n=450) | -3.59 (n=600) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 1950 paired rows out of 1950/1950 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 600 | 4.29 | 0.00 | 0.00 | 21.38 | 34.2% |
| 2 | 300 | 2.11 | 1.24 | -3.22 | 9.41 | 66.7% |
| 3 | 450 | 1.29 | 0.45 | -3.79 | 7.75 | 56.7% |
| 4 | 600 | 1.14 | 0.54 | -1.78 | 5.63 | 59.2% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 1950 paired rows out of 1950/1950 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 600 | 0.01 | 0.00 | -0.13 | 0.47 | 8.3% |
| 2 | 300 | -0.51 | 0.00 | -3.87 | 1.35 | 19.0% |
| 3 | 450 | -0.41 | 0.00 | -3.49 | 1.14 | 15.3% |
| 4 | 600 | -0.23 | 0.00 | -2.23 | 0.77 | 12.8% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 600 | 218 | 36.3% | 0 | margin=382 |
| coarse_to_fine | 1 | 600 | 214 | 35.7% | 0 | margin=386 |
| gated_deflation | 0 | 600 | 340 | 56.7% | 0 | artifact_score=12, margin=248 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 4: coarse_to_fine=-3.59 gated_deflation=-4.20 ungated_deflation=-4.73 -- ordering holds: True