# Phase 3 results -- phase3_librimix_3spk_vion_smoke

rows scored: 216
arms: oracle, real, real_index_order

(means clip +-inf to +-50 dB rather than dropping them)

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 46.91 | 1.08 |
| oracle | ungated_deflation | 45.63 | -0.73 |
| oracle | gated_deflation | 45.63 | -0.73 |
| oracle | coarse_to_fine | 47.00 | 0.04 |
| real | no_recursion | 42.57 | -1.88 |
| real | ungated_deflation | 42.57 | -2.99 |
| real | gated_deflation | 42.57 | -2.99 |
| real | coarse_to_fine | 42.57 | -2.23 |
| real_index_order | no_recursion | 42.57 | -1.88 |
| real_index_order | ungated_deflation | 42.57 | -2.99 |
| real_index_order | gated_deflation | 42.57 | -2.99 |
| real_index_order | coarse_to_fine | 42.57 | -2.23 |

## Overall SI-SDR (un-stratified, whole output track)

One row per (scene, speaker, system); no depth stratification. Read it
WITH the per-depth tables, never instead of them -- and never optimize
against it: it is anchored by the bit-exact solo copy, so it rewards
fixing a level error over fixing a shape error.

| arm | system | 0 ms |
|---|---|---|
| oracle | no_recursion | -13.17±2.26 |
| oracle | ungated_deflation | -11.49±1.91 |
| oracle | gated_deflation | -11.49±1.91 |
| oracle | coarse_to_fine | -14.83±2.08 |
| real | no_recursion | -17.83±2.89 |
| real | ungated_deflation | -12.97±1.87 |
| real | gated_deflation | -12.97±1.87 |
| real | coarse_to_fine | -20.01±3.35 |
| real_index_order | no_recursion | -17.83±2.89 |
| real_index_order | ungated_deflation | -14.95±2.56 |
| real_index_order | gated_deflation | -14.95±2.56 |
| real_index_order | coarse_to_fine | -20.01±3.35 |


## Diarization quality (per arm)

| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |
|---|---|---|---|---|---|---|---|---|
| oracle | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 |
| real | 3 | 0.099 | 0.098 | 0.000 | 0.000 | 0.787 | 0 | 0 |
| real_index_order | 3 | 0.099 | 0.098 | 0.000 | 0.000 | 0.787 | 0 | 0 |

## Cluster discovery vs. enrollment

`clusters` is what the diarizer produced; `enrolled` (the `m` column) is
how many survived enrollment. A cluster with no predicted-solo region
cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a
single-speaker system and its system comparisons are vacuous.

| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |
|---|---|---|---|---|---|
| oracle | 3 | 3.00 | 3.00 | 0 | 0 |
| real | 3 | 3.00 | 3.00 | 0 | 0 |
| real_index_order | 3 | 3.00 | 3.00 | 0 | 0 |

## Gate decisions (per arm x system, by reason)

| arm | system | decisions | accepted | margin | variance (V_i) | vad | artifact | too short | no clip |
|---|---|---|---|---|---|---|---|---|---|
| oracle | gated_deflation | 9 | 9 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| oracle | coarse_to_fine | 18 | 12 (67%) | 6 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 9 | 8 (89%) | 0 | 1 | 0 | 0 | 0 | 0 |
| real | coarse_to_fine | 18 | 12 (67%) | 4 | 2 | 0 | 0 | 0 | 0 |
| real_index_order | gated_deflation | 9 | 8 (89%) | 0 | 1 | 0 | 0 | 0 | 0 |
| real_index_order | coarse_to_fine | 18 | 12 (67%) | 4 | 2 | 0 | 0 | 0 | 0 |

`V_i` rejections: **6**; `mean_variance` nonzero in **54/81** decisions, max 0.000144

`V_i` has been structurally 0 in every prior run (one solo run -> one clip
-> variance over a single sample). A nonzero count above is the first time
the enrollment-variance check has had anything to measure.

## Oracle-vs-real gap (paired, per system x depth)

### `real` - `oracle` -- total cost of real diarization

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 9 | -4.34 | 3.40 | 44% | 1.3 |
| no_recursion | 2 | 9 | -2.97 | 0.92 | 22% | 3.2 |
| ungated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| ungated_deflation | 2 | 9 | -2.26 | 0.90 | 33% | 2.5 |
| gated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| gated_deflation | 2 | 9 | -2.26 | 0.90 | 33% | 2.5 |
| coarse_to_fine | 1 | 9 | -4.43 | 3.38 | 44% | 1.3 |
| coarse_to_fine | 2 | 9 | -2.26 | 0.88 | 22% | 2.6 |

### `real_index_order` - `oracle` -- diarization error alone

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 9 | -4.34 | 3.40 | 44% | 1.3 |
| no_recursion | 2 | 9 | -2.97 | 0.92 | 22% | 3.2 |
| ungated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| ungated_deflation | 2 | 9 | -2.26 | 0.60 | 22% | 3.8 |
| gated_deflation | 1 | 9 | -3.06 | 3.79 | 56% | 0.8 |
| gated_deflation | 2 | 9 | -2.26 | 0.60 | 22% | 3.8 |
| coarse_to_fine | 1 | 9 | -4.43 | 3.38 | 44% | 1.3 |
| coarse_to_fine | 2 | 9 | -2.26 | 0.88 | 22% | 2.6 |

### `real` - `real_index_order` -- the reordering V_i induces

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 9 | +0.00 | 0.00 | 0% | nan |
| no_recursion | 2 | 9 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 1 | 9 | +0.00 | 0.00 | 0% | nan |
| ungated_deflation | 2 | 9 | +0.00 | 0.48 | 56% | 0.0 |
| gated_deflation | 1 | 9 | +0.00 | 0.00 | 0% | nan |
| gated_deflation | 2 | 9 | +0.00 | 0.48 | 56% | 0.0 |
| coarse_to_fine | 1 | 9 | +0.00 | 0.00 | 0% | nan |
| coarse_to_fine | 2 | 9 | +0.00 | 0.00 | 0% | nan |


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=0.04 gated_deflation=-0.73 ungated_deflation=-0.73 -- ordering holds: False
- `real` depth 2: coarse_to_fine=-2.23 gated_deflation=-2.99 ungated_deflation=-2.99 -- ordering holds: False
- `real_index_order` depth 2: coarse_to_fine=-2.23 gated_deflation=-2.99 ungated_deflation=-2.99 -- ordering holds: False

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
