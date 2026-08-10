# Phase 2 results -- phase2_librimix_3spk_scratch345clip50

rows scored: 5400

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.78 | 1.26 | -1.29 |
| ungated_deflation | 40.69 | 0.03 | -2.45 |
| gated_deflation | 42.09 | 0.46 | -2.09 |
| coarse_to_fine | 43.73 | 1.19 | -1.42 |

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

## Spread (per system/depth)

(p95 saturates at the +-50 dB clip wherever the diagnostic-counts table shows many perfect/failed rows -- at depth 1 most rows are solo copy-through and legitimately +inf, so a p95 of exactly the cap there is expected, not a bug)

| system | depth | n | mean | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| no_recursion | 1 | 450 | 43.78 | 27.84 | 50.00 | 16.85 | 50.00 |
| no_recursion | 2 | 300 | 1.26 | -11.65 | 10.30 | -42.06 | 21.68 |
| no_recursion | 3 | 450 | -1.29 | -7.56 | 4.62 | -11.58 | 8.57 |
| ungated_deflation | 1 | 450 | 40.69 | 19.16 | 50.00 | 9.77 | 50.00 |
| ungated_deflation | 2 | 300 | 0.03 | -12.96 | 9.30 | -45.59 | 21.68 |
| ungated_deflation | 3 | 450 | -2.45 | -8.41 | 3.21 | -11.88 | 6.08 |
| gated_deflation | 1 | 450 | 42.09 | 22.92 | 50.00 | 13.22 | 50.00 |
| gated_deflation | 2 | 300 | 0.46 | -12.79 | 9.89 | -45.59 | 21.68 |
| gated_deflation | 3 | 450 | -2.09 | -8.19 | 3.96 | -11.88 | 6.14 |
| coarse_to_fine | 1 | 450 | 43.73 | 27.81 | 50.00 | 16.85 | 50.00 |
| coarse_to_fine | 2 | 300 | 1.19 | -13.33 | 9.99 | -42.06 | 21.68 |
| coarse_to_fine | 3 | 450 | -1.42 | -7.57 | 4.31 | -11.58 | 8.57 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| ungated_deflation | 0 | 49.89 (n=150) | -0.85 (n=75) | -1.79 (n=150) |
| ungated_deflation | 1 | 36.92 (n=150) | -2.01 (n=79) | -2.65 (n=150) |
| ungated_deflation | 2 | 35.25 (n=150) | 1.58 (n=146) | -2.91 (n=150) |
| gated_deflation | 0 | 46.29 (n=241) | 0.41 (n=140) | -1.50 (n=241) |
| gated_deflation | 1 | 38.30 (n=164) | 0.07 (n=117) | -2.67 (n=164) |
| gated_deflation | 2 | 33.39 (n=45) | 1.71 (n=43) | -3.10 (n=45) |
| no_recursion | n/a | 43.78 (n=450) | 1.26 (n=300) | -1.29 (n=450) |
| coarse_to_fine | n/a | 43.73 (n=450) | 1.19 (n=300) | -1.42 (n=450) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 1200 paired rows out of 1200/1200 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 450 | 3.04 | 0.00 | 0.00 | 14.12 | 44.9% |
| 2 | 300 | 1.16 | 0.47 | -2.12 | 5.99 | 59.0% |
| 3 | 450 | 1.03 | 0.37 | -1.64 | 5.02 | 55.6% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 1200 paired rows out of 1200/1200 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 450 | -0.05 | 0.00 | -0.88 | 0.49 | 11.1% |
| 2 | 300 | -0.07 | 0.00 | -2.07 | 1.80 | 24.0% |
| 3 | 450 | -0.12 | 0.00 | -1.93 | 1.27 | 20.0% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 450 | 208 | 46.2% | 0 | margin=242 |
| coarse_to_fine | 1 | 450 | 206 | 45.8% | 0 | margin=244 |
| gated_deflation | 0 | 450 | 302 | 67.1% | 0 | artifact_score=6, margin=142 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 3: coarse_to_fine=-1.42 gated_deflation=-2.09 ungated_deflation=-2.45 -- ordering holds: True