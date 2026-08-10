# Phase 2 results -- phase2_librimix_5spk_scratch345

rows scored: 15000

(means clip +-inf to +-50 dB rather than dropping them -- see the diagnostic-counts table for how often that happens per system/depth)

| system | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|
| no_recursion | 45.82 | 0.65 | -3.08 | -5.47 | -4.76 |
| ungated_deflation | 40.52 | -1.95 | -5.41 | -6.83 | -6.17 |
| gated_deflation | 43.92 | -0.67 | -4.29 | -6.11 | -5.50 |
| coarse_to_fine | 45.83 | 0.11 | -3.31 | -5.64 | -4.91 |

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
| coarse_to_fine | 1 | 0 | 468 | 0 | 750 |
| coarse_to_fine | 2 | 451 | 0 | 0 | 299 |
| coarse_to_fine | 3 | 300 | 0 | 0 | 450 |
| coarse_to_fine | 4 | 151 | 0 | 0 | 599 |
| coarse_to_fine | 5 | 0 | 0 | 0 | 750 |

## Spread (per system/depth)

(p95 saturates at the +-50 dB clip wherever the diagnostic-counts table shows many perfect/failed rows -- at depth 1 most rows are solo copy-through and legitimately +inf, so a p95 of exactly the cap there is expected, not a bug)

| system | depth | n | mean | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| no_recursion | 1 | 750 | 45.82 | 30.17 | 50.00 | 20.31 | 50.00 |
| no_recursion | 2 | 299 | 0.65 | -22.96 | 14.84 | -50.00 | 24.47 |
| no_recursion | 3 | 450 | -3.08 | -22.25 | 9.23 | -50.00 | 16.33 |
| no_recursion | 4 | 599 | -5.47 | -26.59 | 4.37 | -50.00 | 15.34 |
| no_recursion | 5 | 750 | -4.76 | -10.93 | 1.41 | -15.50 | 4.10 |
| ungated_deflation | 1 | 750 | 40.52 | 3.91 | 50.00 | -14.77 | 50.00 |
| ungated_deflation | 2 | 299 | -1.95 | -24.57 | 12.85 | -44.39 | 21.12 |
| ungated_deflation | 3 | 450 | -5.41 | -25.64 | 6.61 | -46.81 | 16.28 |
| ungated_deflation | 4 | 599 | -6.83 | -25.51 | 2.97 | -50.00 | 15.34 |
| ungated_deflation | 5 | 750 | -6.17 | -11.59 | -0.76 | -15.50 | 4.10 |
| gated_deflation | 1 | 750 | 43.92 | 21.00 | 50.00 | -9.54 | 50.00 |
| gated_deflation | 2 | 299 | -0.67 | -23.75 | 12.48 | -43.11 | 22.72 |
| gated_deflation | 3 | 450 | -4.29 | -25.04 | 7.50 | -50.00 | 16.28 |
| gated_deflation | 4 | 599 | -6.11 | -24.79 | 3.73 | -50.00 | 15.34 |
| gated_deflation | 5 | 750 | -5.50 | -10.96 | 0.30 | -15.50 | 4.10 |
| coarse_to_fine | 1 | 750 | 45.83 | 30.17 | 50.00 | 20.31 | 50.00 |
| coarse_to_fine | 2 | 299 | 0.11 | -22.96 | 14.54 | -50.00 | 21.68 |
| coarse_to_fine | 3 | 450 | -3.31 | -22.25 | 8.56 | -50.00 | 16.28 |
| coarse_to_fine | 4 | 599 | -5.64 | -25.01 | 4.31 | -50.00 | 13.41 |
| coarse_to_fine | 5 | 750 | -4.91 | -11.02 | 1.41 | -15.50 | 4.41 |

## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted into the residual before this speaker was extracted)

(within-scene control: depth is held fixed down each column, so a decline across rows is accumulation, not intrinsic overlap difficulty)

| system | n_accepted_before | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|---|
| ungated_deflation | 0 | 49.85 (n=150) | -2.21 (n=40) | -3.60 (n=73) | -5.88 (n=111) | -4.89 (n=150) |
| ungated_deflation | 1 | 49.98 (n=150) | -4.63 (n=43) | -6.41 (n=75) | -5.31 (n=110) | -5.93 (n=150) |
| ungated_deflation | 2 | 50.00 (n=150) | -7.03 (n=45) | -7.26 (n=82) | -7.94 (n=115) | -6.35 (n=150) |
| ungated_deflation | 3 | 18.41 (n=150) | -7.80 (n=33) | -7.92 (n=78) | -7.84 (n=114) | -6.98 (n=150) |
| ungated_deflation | 4 | 34.36 (n=150) | 2.03 (n=138) | -3.37 (n=142) | -7.05 (n=149) | -6.72 (n=150) |
| gated_deflation | 0 | 48.04 (n=351) | -1.81 (n=110) | -2.98 (n=189) | -5.37 (n=265) | -4.93 (n=351) |
| gated_deflation | 1 | 43.52 (n=251) | -0.64 (n=110) | -4.98 (n=155) | -5.86 (n=204) | -5.77 (n=251) |
| gated_deflation | 2 | 36.42 (n=116) | 1.47 (n=60) | -5.17 (n=82) | -8.01 (n=101) | -6.12 (n=116) |
| gated_deflation | 3 | 29.93 (n=29) | 1.02 (n=16) | -7.25 (n=21) | -8.88 (n=26) | -7.26 (n=29) |
| gated_deflation | 4 | 19.95 (n=3) | -11.48 (n=3) | -6.19 (n=3) | -0.59 (n=3) | -8.89 (n=3) |
| no_recursion | n/a | 45.82 (n=750) | 0.65 (n=299) | -3.08 (n=450) | -5.47 (n=599) | -4.76 (n=750) |
| coarse_to_fine | n/a | 45.83 (n=750) | 0.11 (n=299) | -3.31 (n=450) | -5.64 (n=599) | -4.91 (n=750) |

## Paired difference: coarse_to_fine - ungated_deflation

(joined on (scene, speaker, depth); 2848 paired rows out of 2848/2848 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 750 | 5.31 | 0.00 | 0.00 | 29.95 | 28.1% |
| 2 | 299 | 2.05 | 1.44 | -4.38 | 10.35 | 65.9% |
| 3 | 450 | 2.10 | 1.06 | -2.75 | 10.00 | 63.8% |
| 4 | 599 | 1.20 | 0.45 | -2.80 | 6.79 | 56.6% |
| 5 | 750 | 1.27 | 0.64 | -1.34 | 5.75 | 61.2% |

## Paired difference: coarse_to_fine - no_recursion

(joined on (scene, speaker, depth); 2848 paired rows out of 2848/2848 scoreable per system. A positive mean with a ~50% win rate means the margin comes from a few large wins, not broad superiority -- that was the Phase 1 result, so it is reported here.)

| depth | pairs | mean diff | median | p5 | p95 | win rate |
|---|---|---|---|---|---|---|
| 1 | 750 | 0.01 | 0.00 | -0.11 | 0.36 | 7.1% |
| 2 | 299 | -0.54 | 0.00 | -4.03 | 0.92 | 13.0% |
| 3 | 450 | -0.23 | 0.00 | -2.29 | 1.32 | 14.0% |
| 4 | 599 | -0.17 | 0.00 | -2.20 | 0.91 | 11.5% |
| 5 | 750 | -0.15 | 0.00 | -1.55 | 0.59 | 8.9% |

## Confidence gate

| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |
|---|---|---|---|---|---|---|
| coarse_to_fine | 0 | 750 | 196 | 26.1% | 0 | margin=554 |
| coarse_to_fine | 1 | 750 | 193 | 25.7% | 0 | margin=557 |
| gated_deflation | 0 | 750 | 353 | 47.1% | 0 | artifact_score=23, margin=374 |

## Ordering check (3+ speaker overlaps, deepest available depth)
depth 5: coarse_to_fine=-4.91 gated_deflation=-5.50 ungated_deflation=-6.17 -- ordering holds: True