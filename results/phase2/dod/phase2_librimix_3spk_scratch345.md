# Phase 2 results -- phase2_librimix_3spk_scratch345

rows scored: 5400

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.90 | 1.60 | -1.19 |
| ungated_deflation | 40.75 | -0.25 | -2.47 |
| gated_deflation | 42.24 | 0.44 | -2.03 |
| coarse_to_fine | 43.90 | 1.18 | -1.47 |

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
| no_recursion | 1 | 450 | 43.90 | 27.74 | 50.00 | 19.17 | 50.00 |
| no_recursion | 2 | 300 | 1.60 | -11.93 | 11.57 | -41.43 | 20.98 |
| no_recursion | 3 | 450 | -1.19 | -7.88 | 4.68 | -10.69 | 8.42 |
| ungated_deflation | 1 | 450 | 40.75 | 19.63 | 50.00 | 9.59 | 50.00 |
| ungated_deflation | 2 | 300 | -0.25 | -11.85 | 9.82 | -46.82 | 20.98 |
| ungated_deflation | 3 | 450 | -2.47 | -8.15 | 3.25 | -11.82 | 6.94 |
| gated_deflation | 1 | 450 | 42.24 | 23.57 | 50.00 | 13.96 | 50.00 |
| gated_deflation | 2 | 300 | 0.44 | -11.85 | 10.37 | -46.82 | 20.98 |
| gated_deflation | 3 | 450 | -2.03 | -8.07 | 3.86 | -10.69 | 6.94 |
| coarse_to_fine | 1 | 450 | 43.90 | 27.76 | 50.00 | 18.37 | 50.00 |
| coarse_to_fine | 2 | 300 | 1.18 | -12.28 | 10.88 | -41.43 | 20.98 |
| coarse_to_fine | 3 | 450 | -1.47 | -8.52 | 4.53 | -14.37 | 8.32 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| ungated_deflation | 0 | 49.91 (n=150) | -0.32 (n=75) | -1.76 (n=150) |
| ungated_deflation | 1 | 36.80 (n=150) | -2.39 (n=79) | -2.79 (n=150) |
| ungated_deflation | 2 | 35.52 (n=150) | 0.95 (n=146) | -2.86 (n=150) |
| gated_deflation | 0 | 46.29 (n=245) | 0.96 (n=142) | -1.36 (n=245) |
| gated_deflation | 1 | 38.06 (n=163) | -0.25 (n=118) | -2.85 (n=163) |
| gated_deflation | 2 | 34.78 (n=42) | 0.67 (n=40) | -2.84 (n=42) |
| no_recursion | n/a | 43.90 (n=450) | 1.60 (n=300) | -1.19 (n=450) |
| coarse_to_fine | n/a | 43.90 (n=450) | 1.18 (n=300) | -1.47 (n=450) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 1200 paired rows out of 1200/1200 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 450 | 3.15 | 0.00 | 0.00 | 15.37 | 43.1% |
| 2 | 300 | 1.42 | 0.59 | -2.52 | 7.55 | 59.7% |
| 3 | 450 | 1.00 | 0.29 | -2.11 | 5.43 | 53.6% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 1200 paired rows out of 1200/1200 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 450 | -0.00 | 0.00 | -0.93 | 0.72 | 12.2% |
| 2 | 300 | -0.42 | 0.00 | -3.30 | 1.67 | 24.7% |
| 3 | 450 | -0.29 | 0.00 | -2.72 | 1.40 | 19.3% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 450 | 222 | 49.3% | 0 | margin=228 |
| coarse_to_fine | 1 | 450 | 220 | 48.9% | 0 | margin=230 |
| gated_deflation | 0 | 450 | 294 | 65.3% | 0 | artifact_score=5, margin=151 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 3: coarse_to_fine=-1.47 gated_deflation=-2.03 ungated_deflation=-2.47 -- ordering holds: True