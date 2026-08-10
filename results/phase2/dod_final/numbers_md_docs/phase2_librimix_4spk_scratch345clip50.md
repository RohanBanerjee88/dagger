# Phase 2 results -- phase2_librimix_4spk_scratch345clip50

rows scored: 9600

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| no_recursion | 45.16 | 1.12 | -3.66 | -3.47 |
| ungated_deflation | 40.97 | -1.02 | -5.00 | -4.76 |
| gated_deflation | 43.27 | 0.08 | -4.49 | -4.23 |
| coarse_to_fine | 45.13 | 0.83 | -3.89 | -3.60 |

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
| no_recursion | 1 | 600 | 45.16 | 28.52 | 50.00 | 17.87 | 50.00 |
| no_recursion | 2 | 300 | 1.12 | -18.21 | 14.17 | -50.00 | 23.21 |
| no_recursion | 3 | 450 | -3.66 | -28.30 | 8.23 | -50.00 | 19.44 |
| no_recursion | 4 | 600 | -3.47 | -9.72 | 2.69 | -18.39 | 4.97 |
| ungated_deflation | 1 | 600 | 40.97 | 13.05 | 50.00 | -4.07 | 50.00 |
| ungated_deflation | 2 | 300 | -1.02 | -18.45 | 10.89 | -47.86 | 18.61 |
| ungated_deflation | 3 | 450 | -5.00 | -28.30 | 5.77 | -50.00 | 18.98 |
| ungated_deflation | 4 | 600 | -4.76 | -10.40 | 0.83 | -20.29 | 4.57 |
| gated_deflation | 1 | 600 | 43.27 | 20.66 | 50.00 | 4.65 | 50.00 |
| gated_deflation | 2 | 300 | 0.08 | -18.51 | 12.52 | -50.00 | 21.31 |
| gated_deflation | 3 | 450 | -4.49 | -28.16 | 6.42 | -50.00 | 18.98 |
| gated_deflation | 4 | 600 | -4.23 | -10.18 | 1.64 | -18.39 | 4.57 |
| coarse_to_fine | 1 | 600 | 45.13 | 28.52 | 50.00 | 17.87 | 50.00 |
| coarse_to_fine | 2 | 300 | 0.83 | -18.21 | 13.33 | -50.00 | 23.21 |
| coarse_to_fine | 3 | 450 | -3.89 | -28.08 | 8.23 | -50.00 | 19.44 |
| coarse_to_fine | 4 | 600 | -3.60 | -10.09 | 2.63 | -18.39 | 5.74 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|---|
| ungated_deflation | 0 | 49.94 (n=150) | -2.42 (n=50) | -5.93 (n=102) | -3.95 (n=150) |
| ungated_deflation | 1 | 50.00 (n=150) | -4.94 (n=55) | -5.15 (n=92) | -4.74 (n=150) |
| ungated_deflation | 2 | 27.59 (n=150) | -2.13 (n=50) | -7.29 (n=107) | -5.21 (n=150) |
| ungated_deflation | 3 | 36.34 (n=150) | 1.33 (n=145) | -2.64 (n=149) | -5.15 (n=150) |
| gated_deflation | 0 | 47.83 (n=290) | -1.22 (n=114) | -4.34 (n=202) | -3.77 (n=290) |
| gated_deflation | 1 | 41.99 (n=204) | 0.75 (n=112) | -5.25 (n=156) | -4.41 (n=204) |
| gated_deflation | 2 | 33.70 (n=95) | 0.94 (n=63) | -3.98 (n=81) | -5.10 (n=95) |
| gated_deflation | 3 | 29.56 (n=11) | 1.71 (n=11) | -0.32 (n=11) | -5.50 (n=11) |
| no_recursion | n/a | 45.16 (n=600) | 1.12 (n=300) | -3.66 (n=450) | -3.47 (n=600) |
| coarse_to_fine | n/a | 45.13 (n=600) | 0.83 (n=300) | -3.89 (n=450) | -3.60 (n=600) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 1950 paired rows out of 1950/1950 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 600 | 4.16 | 0.00 | 0.00 | 20.96 | 34.7% |
| 2 | 300 | 1.85 | 1.08 | -3.27 | 8.96 | 64.3% |
| 3 | 450 | 1.11 | 0.42 | -2.90 | 6.88 | 56.7% |
| 4 | 600 | 1.16 | 0.65 | -1.38 | 5.36 | 62.7% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 1950 paired rows out of 1950/1950 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 600 | -0.03 | 0.00 | -0.34 | 0.15 | 5.8% |
| 2 | 300 | -0.28 | 0.00 | -2.73 | 1.62 | 20.0% |
| 3 | 450 | -0.23 | 0.00 | -2.50 | 1.16 | 15.8% |
| 4 | 600 | -0.13 | 0.00 | -1.59 | 0.87 | 13.0% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 600 | 204 | 34.0% | 0 | margin=396 |
| coarse_to_fine | 1 | 600 | 202 | 33.7% | 0 | margin=398 |
| gated_deflation | 0 | 600 | 339 | 56.5% | 0 | artifact_score=12, margin=249 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 4: coarse_to_fine=-3.60 gated_deflation=-4.23 ungated_deflation=-4.76 -- ordering holds: True