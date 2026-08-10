# Phase 2 results -- phase2_librimix_5spk_scratch345clip50

rows scored: 15000

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|
| no_recursion | 45.80 | 0.51 | -3.32 | -5.63 | -4.87 |
| ungated_deflation | 40.72 | -1.75 | -5.10 | -6.78 | -6.19 |
| gated_deflation | 43.88 | -0.55 | -4.05 | -6.12 | -5.51 |
| coarse_to_fine | 45.79 | 0.10 | -3.51 | -5.72 | -4.95 |

## Diagnostic counts (per system/depth: absent / perfect / failed / scored)

| system | depth | absent (nan) | perfect (+inf) | failed (-inf) | scored |
|---|---|---|---|---|---|
| no_recursion | 1 | 0 | 467 | 0 | 750 |
| no_recursion | 2 | 451 | 0 | 0 | 299 |
| no_recursion | 3 | 300 | 0 | 0 | 450 |
| no_recursion | 4 | 151 | 0 | 0 | 599 |
| no_recursion | 5 | 0 | 0 | 0 | 750 |
| ungated_deflation | 1 | 0 | 467 | 0 | 750 |
| ungated_deflation | 2 | 451 | 0 | 0 | 299 |
| ungated_deflation | 3 | 300 | 0 | 0 | 450 |
| ungated_deflation | 4 | 151 | 0 | 0 | 599 |
| ungated_deflation | 5 | 0 | 0 | 0 | 750 |
| gated_deflation | 1 | 0 | 467 | 0 | 750 |
| gated_deflation | 2 | 451 | 0 | 0 | 299 |
| gated_deflation | 3 | 300 | 0 | 0 | 450 |
| gated_deflation | 4 | 151 | 0 | 0 | 599 |
| gated_deflation | 5 | 0 | 0 | 0 | 750 |
| coarse_to_fine | 1 | 0 | 467 | 0 | 750 |
| coarse_to_fine | 2 | 451 | 0 | 0 | 299 |
| coarse_to_fine | 3 | 300 | 0 | 0 | 450 |
| coarse_to_fine | 4 | 151 | 0 | 0 | 599 |
| coarse_to_fine | 5 | 0 | 0 | 0 | 750 |

## Spread (per system/depth)

(p95 saturates at the +-50 dB clip wherever the diagnostic-counts table shows many perfect/failed rows -- at depth 1 most rows are solo copy-through and legitimately +inf, so a p95 of exactly the cap there is expected, not a bug)

| system | depth | n | mean | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| no_recursion | 1 | 750 | 45.80 | 29.99 | 50.00 | 17.80 | 50.00 |
| no_recursion | 2 | 299 | 0.51 | -21.95 | 15.11 | -50.00 | 23.02 |
| no_recursion | 3 | 450 | -3.32 | -23.78 | 9.22 | -50.00 | 16.45 |
| no_recursion | 4 | 599 | -5.63 | -27.19 | 4.66 | -50.00 | 15.21 |
| no_recursion | 5 | 750 | -4.87 | -10.67 | 1.24 | -14.68 | 4.27 |
| ungated_deflation | 1 | 750 | 40.72 | 4.75 | 50.00 | -11.92 | 50.00 |
| ungated_deflation | 2 | 299 | -1.75 | -23.95 | 12.91 | -50.00 | 22.79 |
| ungated_deflation | 3 | 450 | -5.10 | -25.37 | 6.46 | -46.25 | 16.45 |
| ungated_deflation | 4 | 599 | -6.78 | -26.41 | 3.43 | -50.00 | 15.21 |
| ungated_deflation | 5 | 750 | -6.19 | -11.69 | -1.27 | -15.60 | 3.64 |
| gated_deflation | 1 | 750 | 43.88 | 21.22 | 50.00 | -2.53 | 50.00 |
| gated_deflation | 2 | 299 | -0.55 | -23.43 | 12.53 | -50.00 | 25.71 |
| gated_deflation | 3 | 450 | -4.05 | -25.37 | 7.52 | -46.25 | 16.45 |
| gated_deflation | 4 | 599 | -6.12 | -26.67 | 4.34 | -50.00 | 15.21 |
| gated_deflation | 5 | 750 | -5.51 | -10.99 | -0.01 | -15.60 | 4.27 |
| coarse_to_fine | 1 | 750 | 45.79 | 30.22 | 50.00 | 17.80 | 50.00 |
| coarse_to_fine | 2 | 299 | 0.10 | -21.95 | 13.65 | -50.00 | 23.62 |
| coarse_to_fine | 3 | 450 | -3.51 | -23.78 | 8.13 | -50.00 | 16.45 |
| coarse_to_fine | 4 | 599 | -5.72 | -27.23 | 4.82 | -50.00 | 16.45 |
| coarse_to_fine | 5 | 750 | -4.95 | -10.89 | 1.22 | -15.66 | 4.34 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 49.81 (n=150) | -2.84 (n=40) | -3.62 (n=73) | -5.89 (n=111) | -4.97 (n=150) |
| ungated_deflation | 1 | 49.97 (n=150) | -4.22 (n=43) | -6.40 (n=75) | -5.37 (n=110) | -5.89 (n=150) |
| ungated_deflation | 2 | 50.00 (n=150) | -6.71 (n=45) | -6.97 (n=82) | -7.99 (n=115) | -6.31 (n=150) |
| ungated_deflation | 3 | 19.49 (n=150) | -7.86 (n=33) | -7.65 (n=78) | -7.99 (n=114) | -7.00 (n=150) |
| ungated_deflation | 4 | 34.31 (n=150) | 2.42 (n=138) | -2.69 (n=142) | -6.61 (n=149) | -6.78 (n=150) |
| gated_deflation | 0 | 48.07 (n=336) | -1.57 (n=106) | -2.97 (n=180) | -5.41 (n=258) | -4.97 (n=336) |
| gated_deflation | 1 | 43.60 (n=266) | -0.79 (n=113) | -5.12 (n=166) | -6.39 (n=212) | -5.71 (n=266) |
| gated_deflation | 2 | 36.20 (n=115) | 1.86 (n=59) | -3.27 (n=79) | -7.38 (n=100) | -6.18 (n=115) |
| gated_deflation | 3 | 30.90 (n=30) | 1.00 (n=18) | -7.47 (n=22) | -6.84 (n=26) | -7.00 (n=30) |
| gated_deflation | 4 | 22.24 (n=3) | -12.14 (n=3) | -5.20 (n=3) | 0.22 (n=3) | -8.58 (n=3) |
| no_recursion | n/a | 45.80 (n=750) | 0.51 (n=299) | -3.32 (n=450) | -5.63 (n=599) | -4.87 (n=750) |
| coarse_to_fine | n/a | 45.79 (n=750) | 0.10 (n=299) | -3.51 (n=450) | -5.72 (n=599) | -4.95 (n=750) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 2848 paired rows out of 2848/2848 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 750 | 5.07 | 0.00 | 0.00 | 29.46 | 28.8% |
| 2 | 299 | 1.85 | 1.19 | -3.56 | 9.25 | 64.9% |
| 3 | 450 | 1.59 | 0.98 | -2.89 | 8.56 | 63.1% |
| 4 | 599 | 1.06 | 0.58 | -2.65 | 6.11 | 58.9% |
| 5 | 750 | 1.24 | 0.71 | -1.14 | 5.19 | 64.8% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 2848 paired rows out of 2848/2848 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 750 | -0.01 | 0.00 | -0.28 | 0.06 | 5.7% |
| 2 | 299 | -0.41 | 0.00 | -2.93 | 1.18 | 15.1% |
| 3 | 450 | -0.19 | 0.00 | -2.46 | 1.21 | 16.4% |
| 4 | 599 | -0.09 | 0.00 | -1.71 | 1.11 | 13.7% |
| 5 | 750 | -0.08 | 0.00 | -1.23 | 0.69 | 12.1% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 750 | 201 | 26.8% | 0 | margin=549 |
| coarse_to_fine | 1 | 750 | 200 | 26.7% | 0 | margin=550 |
| gated_deflation | 0 | 750 | 355 | 47.3% | 0 | artifact_score=27, margin=368 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 5: coarse_to_fine=-4.95 gated_deflation=-5.51 ungated_deflation=-6.19 -- ordering holds: True