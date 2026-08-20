# Phase 3 results -- phase3_librimix_3spk_dilation_sweep

rows scored: 4800
arms: oracle, real

(means clip +-inf to +-50 dB rather than dropping them)

> Dilation sweep over [0.0, 200.0, 400.0, 800.0] ms. **Every table below the
> sweep section reports the 0 ms baseline only** -- mixing dilation
> values into one mean would average across different pipelines.

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 47.03 | 1.69 |
| oracle | ungated_deflation | 45.98 | 0.24 |
| oracle | gated_deflation | 45.98 | 0.24 |
| oracle | coarse_to_fine | 47.12 | 1.10 |
| real | no_recursion | 40.30 | -1.29 |
| real | ungated_deflation | 40.30 | -1.94 |
| real | gated_deflation | 40.30 | -1.92 |
| real | coarse_to_fine | 40.30 | -1.86 |

## Overlap dilation sweep

Paired against the 0 ms baseline on matched (scene, speaker, depth, system).
`recall`/`false alarm` score the DERIVED overlap mask against the true
overlap region -- the mask that decides copy-vs-extract, which is what
dilation actually moves (DER is unchanged by construction: dilation does
not touch `activity`).

### `oracle`

| dilate (ms) | recall | false alarm | scenes lost | d1 vs 0ms | d2 vs 0ms |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.000 | 0/25 | +0.00±0.00 (n=300) | +0.00±0.00 (n=300) |
| 200 | 1.000 | 0.008 | 0/25 | -32.56±0.67 (n=300) | +0.01±0.01 (n=300) |
| 400 | 1.000 | 0.016 | 0/25 | -37.51±0.60 (n=300) | +0.01±0.01 (n=300) |
| 800 | 1.000 | 0.032 | 0/25 | -43.26±0.41 (n=300) | +0.01±0.02 (n=300) |

### `real`

| dilate (ms) | recall | false alarm | scenes lost | d1 vs 0ms | d2 vs 0ms |
|---|---|---|---|---|---|
| 0 | 0.774 | 0.000 | 0/25 | +0.00±0.00 (n=300) | +0.00±0.00 (n=300) |
| 200 | 0.876 | 0.000 | 0/25 | -2.40±0.43 (n=300) | +1.10±0.08 (n=300) |
| 400 | 0.946 | 0.003 | 0/25 | -14.63±0.93 (n=300) | +1.90±0.12 (n=300) |
| 800 | 0.989 | 0.015 | 0/25 | -29.85±0.91 (n=300) | +2.19±0.15 (n=300) |


## Diarization quality (per arm)

| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |
|---|---|---|---|---|---|---|---|---|
| oracle | 25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 |
| real | 25 | 0.110 | 0.102 | 0.000 | 0.007 | 0.774 | 0 | 0 |

## Cluster discovery vs. enrollment

`clusters` is what the diarizer produced; `enrolled` (the `m` column) is
how many survived enrollment. A cluster with no predicted-solo region
cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a
single-speaker system and its system comparisons are vacuous.

| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |
|---|---|---|---|---|---|
| oracle | 25 | 3.00 | 3.00 | 0 | 0 |
| real | 25 | 3.00 | 3.00 | 0 | 0 |

## Gate decisions (per arm x system, by reason)

| arm | system | decisions | accepted | margin | variance (V_i) | vad | artifact | too short | no clip |
|---|---|---|---|---|---|---|---|---|---|
| oracle | gated_deflation | 75 | 75 (100%) | 0 | 0 | 0 | 0 | 0 | 0 |
| oracle | coarse_to_fine | 150 | 101 (67%) | 49 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 75 | 74 (99%) | 1 | 0 | 0 | 0 | 0 | 0 |
| real | coarse_to_fine | 150 | 104 (69%) | 46 | 0 | 0 | 0 | 0 | 0 |

`V_i` rejections: **0**; `mean_variance` nonzero in **222/450** decisions, max 0.000254

`V_i` has been structurally 0 in every prior run (one solo run -> one clip
-> variance over a single sample). A nonzero count above is the first time
the enrollment-variance check has had anything to measure.

## Oracle-vs-real gap (paired, per system x depth)

### `real` - `oracle` -- total cost of real diarization

| system | depth | n | mean (dB) | SEM | win rate | \|t\| |
|---|---|---|---|---|---|---|
| no_recursion | 1 | 75 | -6.73 | 1.20 | 24% | 5.6 |
| no_recursion | 2 | 75 | -2.98 | 0.27 | 9% | 11.2 |
| ungated_deflation | 1 | 75 | -5.69 | 1.28 | 31% | 4.5 |
| ungated_deflation | 2 | 75 | -2.18 | 0.31 | 23% | 7.1 |
| gated_deflation | 1 | 75 | -5.69 | 1.28 | 31% | 4.5 |
| gated_deflation | 2 | 75 | -2.16 | 0.31 | 23% | 7.1 |
| coarse_to_fine | 1 | 75 | -6.82 | 1.21 | 27% | 5.6 |
| coarse_to_fine | 2 | 75 | -2.96 | 0.27 | 9% | 10.8 |


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=1.10 gated_deflation=0.24 ungated_deflation=0.24 -- ordering holds: False
- `real` depth 2: coarse_to_fine=-1.86 gated_deflation=-1.92 ungated_deflation=-1.94 -- ordering holds: True

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
