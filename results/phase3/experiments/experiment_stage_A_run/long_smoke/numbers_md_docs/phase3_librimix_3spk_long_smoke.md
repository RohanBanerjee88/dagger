# Phase 3 results -- phase3_librimix_3spk_long_smoke

rows scored: 480
arms: oracle, real, real_forced_m, real_index_order

(means clip +-inf to +-50 dB rather than dropping them)

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 46.38 | 0.42 |
| oracle | ungated_deflation | 45.61 | -0.88 |
| oracle | gated_deflation | 45.61 | -0.88 |
| oracle | coarse_to_fine | 46.51 | -0.27 |
| real | no_recursion | 42.26 | -2.53 |
| real | ungated_deflation | 42.26 | -3.27 |
| real | gated_deflation | 42.26 | -3.27 |
| real | coarse_to_fine | 42.26 | -2.73 |
| real_forced_m | no_recursion | 42.26 | -2.53 |
| real_forced_m | ungated_deflation | 42.26 | -3.27 |
| real_forced_m | gated_deflation | 42.26 | -3.27 |
| real_forced_m | coarse_to_fine | 42.26 | -2.73 |
| real_index_order | no_recursion | 42.26 | -2.53 |
| real_index_order | ungated_deflation | 42.26 | -3.27 |
| real_index_order | gated_deflation | 42.26 | -3.27 |
| real_index_order | coarse_to_fine | 42.26 | -2.73 |

## Diarization quality (per arm)

| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |
|---|---|---|---|---|---|---|---|---|
| oracle | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 |
| real | 5 | 0.111 | 0.105 | 0.000 | 0.006 | 0.754 | 0 | 0 |
| real_forced_m | 5 | 0.111 | 0.105 | 0.000 | 0.006 | 0.754 | 0 | 0 |
| real_index_order | 5 | 0.111 | 0.105 | 0.000 | 0.006 | 0.754 | 0 | 0 |

## Cluster discovery vs. enrollment

`clusters` is what the diarizer produced; `enrolled` (the `m` column) is
how many survived enrollment. A cluster with no predicted-solo region
cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a
single-speaker system and its system comparisons are vacuous.

| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |
|---|---|---|---|---|---|
| oracle | 5 | 3.00 | 3.00 | 0 | 0 |
| real | 5 | 3.00 | 3.00 | 0 | 0 |
| real_forced_m | 5 | 3.00 | 3.00 | 0 | 0 |
| real_index_order | 5 | 3.00 | 3.00 | 0 | 0 |

## Gate decisions (per arm x system, by reason)

| arm | system | decisions | accepted | margin | variance (V_i) | vad | artifact | too short | no clip |
|---|---|---|---|---|---|---|---|---|---|
| oracle | gated_deflation | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| oracle | coarse_to_fine | 30 | 18 (60%) | 12 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| real | coarse_to_fine | 30 | 18 (60%) | 12 | 0 | 0 | 0 | 0 | 0 |
| real_forced_m | gated_deflation | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| real_forced_m | coarse_to_fine | 30 | 18 (60%) | 12 | 0 | 0 | 0 | 0 | 0 |
| real_index_order | gated_deflation | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| real_index_order | coarse_to_fine | 30 | 18 (60%) | 12 | 0 | 0 | 0 | 0 | 0 |

`V_i` rejections: **0**; `mean_variance` nonzero in **126/180** decisions, max 0.000254

`V_i` has been structurally 0 in every prior run (one solo run -> one clip
-> variance over a single sample). A nonzero count above is the first time
the enrollment-variance check has had anything to measure.

## Oracle-vs-real gap (paired, per system x depth)

### `real` - `oracle` -- total cost of real diarization

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 15 | -4.12 | 2.68 | 40% | 1.5 |
| no_recursion | 2 | 15 | -2.94 | 0.57 | 13% | 5.2 |
| ungated_deflation | 1 | 15 | -3.34 | 2.84 | 47% | 1.2 |
| ungated_deflation | 2 | 15 | -2.38 | 0.59 | 20% | 4.1 |
| gated_deflation | 1 | 15 | -3.34 | 2.84 | 47% | 1.2 |
| gated_deflation | 2 | 15 | -2.38 | 0.59 | 20% | 4.1 |
| coarse_to_fine | 1 | 15 | -4.25 | 2.68 | 40% | 1.6 |
| coarse_to_fine | 2 | 15 | -2.46 | 0.55 | 13% | 4.4 |

### `real_index_order` - `oracle` -- diarization error alone

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 15 | -4.12 | 2.68 | 40% | 1.5 |
| no_recursion | 2 | 15 | -2.94 | 0.57 | 13% | 5.2 |
| ungated_deflation | 1 | 15 | -3.34 | 2.84 | 47% | 1.2 |
| ungated_deflation | 2 | 15 | -2.38 | 0.40 | 13% | 5.9 |
| gated_deflation | 1 | 15 | -3.34 | 2.84 | 47% | 1.2 |
| gated_deflation | 2 | 15 | -2.38 | 0.40 | 13% | 5.9 |
| coarse_to_fine | 1 | 15 | -4.25 | 2.68 | 40% | 1.6 |
| coarse_to_fine | 2 | 15 | -2.46 | 0.55 | 13% | 4.4 |

### `real` - `real_index_order` -- the reordering V_i induces

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 15 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 15 | -0.00 | 0.34 | 60% | 0.0 |
| gated_deflation | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 15 | -0.00 | 0.34 | 60% | 0.0 |
| coarse_to_fine | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 2 | 15 | +0.00 | 0.00 | 0% | nan |

### `real_forced_m` - `real` -- how much of the cost was speaker counting

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 15 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 15 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 15 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 1 | 15 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 2 | 15 | +0.00 | 0.00 | 0% | nan |


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=-0.27 gated_deflation=-0.88 ungated_deflation=-0.88 -- ordering holds: False
- `real` depth 2: coarse_to_fine=-2.73 gated_deflation=-3.27 ungated_deflation=-3.27 -- ordering holds: False
- `real_forced_m` depth 2: coarse_to_fine=-2.73 gated_deflation=-3.27 ungated_deflation=-3.27 -- ordering holds: False
- `real_index_order` depth 2: coarse_to_fine=-2.73 gated_deflation=-3.27 ungated_deflation=-3.27 -- ordering holds: False

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
