# Phase 3 results -- phase3_librimix_3spk_long2min

rows scored: 4800
arms: oracle, real, real_forced_m, real_index_order

(means clip +-inf to +-50 dB rather than dropping them)

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 47.62 | 1.73 |
| oracle | ungated_deflation | 46.71 | 0.24 |
| oracle | gated_deflation | 46.71 | 0.24 |
| oracle | coarse_to_fine | 47.70 | 1.04 |
| real | no_recursion | 38.47 | -1.38 |
| real | ungated_deflation | 38.47 | -1.96 |
| real | gated_deflation | 38.47 | -1.95 |
| real | coarse_to_fine | 38.47 | -1.87 |
| real_forced_m | no_recursion | 38.47 | -1.38 |
| real_forced_m | ungated_deflation | 38.47 | -1.96 |
| real_forced_m | gated_deflation | 38.47 | -1.95 |
| real_forced_m | coarse_to_fine | 38.47 | -1.87 |
| real_index_order | no_recursion | 38.47 | -1.38 |
| real_index_order | ungated_deflation | 38.47 | -2.22 |
| real_index_order | gated_deflation | 38.47 | -2.21 |
| real_index_order | coarse_to_fine | 38.47 | -1.87 |

## Diarization quality (per arm)

| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |
|---|---|---|---|---|---|---|---|---|
| oracle | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 |
| real | 50 | 0.113 | 0.105 | 0.000 | 0.008 | 0.758 | 0 | 0 |
| real_forced_m | 50 | 0.113 | 0.105 | 0.000 | 0.008 | 0.758 | 0 | 0 |
| real_index_order | 50 | 0.113 | 0.105 | 0.000 | 0.008 | 0.758 | 0 | 0 |

## Cluster discovery vs. enrollment

`clusters` is what the diarizer produced; `enrolled` (the `m` column) is
how many survived enrollment. A cluster with no predicted-solo region
cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a
single-speaker system and its system comparisons are vacuous.

| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |
|---|---|---|---|---|---|
| oracle | 50 | 3.00 | 3.00 | 0 | 0 |
| real | 50 | 3.00 | 3.00 | 0 | 0 |
| real_forced_m | 50 | 3.00 | 3.00 | 0 | 0 |
| real_index_order | 50 | 3.00 | 3.00 | 0 | 0 |

## Gate decisions (per arm x system, by reason)

| arm | system | decisions | accepted | margin | variance (V_i) | vad | artifact | too short | no clip |
|---|---|---|---|---|---|---|---|---|---|
| oracle | gated_deflation | 150 | 149 (99%) | 1 | 0 | 0 | 0 | 0 | 0 |
| oracle | coarse_to_fine | 300 | 215 (72%) | 85 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 150 | 147 (98%) | 3 | 0 | 0 | 0 | 0 | 0 |
| real | coarse_to_fine | 300 | 212 (71%) | 88 | 0 | 0 | 0 | 0 | 0 |
| real_forced_m | gated_deflation | 150 | 147 (98%) | 3 | 0 | 0 | 0 | 0 | 0 |
| real_forced_m | coarse_to_fine | 300 | 212 (71%) | 88 | 0 | 0 | 0 | 0 | 0 |
| real_index_order | gated_deflation | 150 | 147 (98%) | 3 | 0 | 0 | 0 | 0 | 0 |
| real_index_order | coarse_to_fine | 300 | 212 (71%) | 88 | 0 | 0 | 0 | 0 | 0 |

`V_i` rejections: **0**; `mean_variance` nonzero in **1332/1800** decisions, max 0.000324

`V_i` has been structurally 0 in every prior run (one solo run -> one clip
-> variance over a single sample). A nonzero count above is the first time
the enrollment-variance check has had anything to measure.

## Oracle-vs-real gap (paired, per system x depth)

### `real` - `oracle` -- total cost of real diarization

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

### `real_index_order` - `oracle` -- diarization error alone

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

### `real` - `real_index_order` -- the reordering V_i induces

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

### `real_forced_m` - `real` -- how much of the cost was speaker counting

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


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=1.04 gated_deflation=0.24 ungated_deflation=0.24 -- ordering holds: True
- `real` depth 2: coarse_to_fine=-1.87 gated_deflation=-1.95 ungated_deflation=-1.96 -- ordering holds: True
- `real_forced_m` depth 2: coarse_to_fine=-1.87 gated_deflation=-1.95 ungated_deflation=-1.96 -- ordering holds: True
- `real_index_order` depth 2: coarse_to_fine=-1.87 gated_deflation=-2.21 ungated_deflation=-2.22 -- ordering holds: True

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
