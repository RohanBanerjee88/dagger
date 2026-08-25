# Phase 3 results -- phase3_librimix_3spk_deployable

rows scored: 4800
arms: oracle, real

(means clip +-inf to +-50 dB rather than dropping them)

> Dilation sweep over [0.0, 400.0] ms. **Every table below the
> sweep section reports the 0 ms baseline only** -- mixing dilation
> values into one mean would average across different pipelines.

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 47.62 | 1.73 |
| oracle | ungated_deflation | 46.71 | 0.24 |
| oracle | gated_deflation | 46.71 | 0.24 |
| oracle | coarse_to_fine | 47.62 | 1.73 |
| real | no_recursion | 38.47 | -1.38 |
| real | ungated_deflation | 38.47 | -1.96 |
| real | gated_deflation | 38.47 | -1.94 |
| real | coarse_to_fine | 38.47 | -1.38 |

## Overall SI-SDR (un-stratified)

One row per (scene, speaker, system); no depth stratification. Read
these WITH the per-depth tables, never instead of them (§6.4), and
never optimize against any of them -- optimizing a whole-output number
is what voided Stage B's refinement ceiling.

### Pooled across depths -- THE EXCHANGE RATE

Scale fitted per depth (so a per-region level error is discounted
exactly as the per-depth tables discount it), then error energies
pooled weighted by each depth's true speech. Provably bounded by the
best and worst depth, so it can weigh a depth-1 loss against a depth-2
gain. **This is the number to compare sweep points on.**

| arm | system | 0 ms | 400 ms |
|---|---|---|---|
| oracle | no_recursion | 6.10±0.27 | 3.47±0.34 |
| oracle | ungated_deflation | 4.61±0.31 | 2.37±0.38 |
| oracle | gated_deflation | 4.61±0.31 | 2.37±0.38 |
| oracle | coarse_to_fine | 6.10±0.27 | 3.47±0.34 |
| real | no_recursion | 2.92±0.24 | 4.82±0.25 |
| real | ungated_deflation | 2.35±0.27 | 3.63±0.30 |
| real | gated_deflation | 2.37±0.27 | 3.67±0.30 |
| real | coarse_to_fine | 2.92±0.24 | 4.82±0.25 |

### Whole output track, one global scale

The literal `si_sdr(output, target)`. SCALE-ANCHORED by the bit-exact
solo copy, so a pure LEVEL error in the overlap region is charged at
full price while every per-depth row discounts it -- which is why this
can land BELOW every depth it appears to summarise (it did so in 271 of
288 rows on 2026-08-23). Kept because it is the only score here that
can see a level error at all; not a summary of the tables above.

| arm | system | 0 ms | 400 ms |
|---|---|---|---|
| oracle | no_recursion | -17.07±0.82 | -16.60±0.84 |
| oracle | ungated_deflation | -13.92±0.75 | -13.41±0.71 |
| oracle | gated_deflation | -13.99±0.75 | -13.47±0.71 |
| oracle | coarse_to_fine | -17.07±0.82 | -16.60±0.84 |
| real | no_recursion | -18.08±0.82 | -16.46±0.72 |
| real | ungated_deflation | -15.90±0.71 | -14.92±0.74 |
| real | gated_deflation | -16.16±0.72 | -15.18±0.76 |
| real | coarse_to_fine | -18.08±0.82 | -16.46±0.72 |

### Level disagreement across depths (dB)

`20*log10(max alpha / min alpha)` over the per-depth fitted scales. 0
means one consistent level explains the whole output. Large values are
the mechanism behind any gap between the two tables above, and are
invisible to every scale-invariant metric in this project.

| arm | system | 0 ms | 400 ms |
|---|---|---|---|
| oracle | no_recursion | 8.55±0.17 | 9.44±0.27 |
| oracle | ungated_deflation | 13.29±0.41 | 14.23±0.46 |
| oracle | gated_deflation | 13.21±0.40 | 14.15±0.46 |
| oracle | coarse_to_fine | 8.55±0.17 | 9.44±0.27 |
| real | no_recursion | 6.48±0.23 | 8.21±0.20 |
| real | ungated_deflation | 10.63±0.44 | 12.05±0.39 |
| real | gated_deflation | 10.32±0.43 | 11.81±0.37 |
| real | coarse_to_fine | 6.48±0.23 | 8.21±0.20 |


## Overlap dilation sweep

Paired against the 0 ms baseline on matched (scene, speaker, depth, system).
`recall`/`false alarm` score the DERIVED overlap mask against the true
overlap region -- the mask that decides copy-vs-extract, which is what
dilation actually moves (DER is unchanged by construction: dilation does
not touch `activity`).

### `oracle`

| dilate (ms) | recall | false alarm | scenes lost | d1 vs 0ms | d2 vs 0ms |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.000 | 0/50 | +0.00±0.00 (n=600) | +0.00±0.00 (n=600) |
| 400 | 1.000 | 0.016 | 0/50 | -38.65±0.44 (n=600) | +0.02±0.01 (n=600) |

### `real`

| dilate (ms) | recall | false alarm | scenes lost | d1 vs 0ms | d2 vs 0ms |
|---|---|---|---|---|---|
| 0 | 0.758 | 0.000 | 0/50 | +0.00±0.00 (n=600) | +0.00±0.00 (n=600) |
| 400 | 0.937 | 0.003 | 0/50 | -12.95±0.62 (n=600) | +1.98±0.08 (n=600) |


## Diarization quality (per arm)

| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |
|---|---|---|---|---|---|---|---|---|
| oracle | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 |
| real | 50 | 0.113 | 0.105 | 0.000 | 0.008 | 0.758 | 0 | 0 |

## Cluster discovery vs. enrollment

`clusters` is what the diarizer produced; `enrolled` (the `m` column) is
how many survived enrollment. A cluster with no predicted-solo region
cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a
single-speaker system and its system comparisons are vacuous.

| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |
|---|---|---|---|---|---|
| oracle | 50 | 3.00 | 3.00 | 0 | 0 |
| real | 50 | 3.00 | 3.00 | 0 | 0 |

## Gate decisions (per arm x system, by reason)

| arm | system | decisions | accepted | margin | variance (V_i) | vad | artifact | too short | no clip |
|---|---|---|---|---|---|---|---|---|---|
| oracle | gated_deflation | 150 | 149 (99%) | 1 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 150 | 121 (81%) | 3 | 26 | 0 | 0 | 0 | 0 |

`V_i` rejections: **26**; `mean_variance` nonzero in **148/300** decisions, max 0.000324

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
| gated_deflation | 2 | 150 | -2.18 | 0.23 | 24% | 9.4 |
| coarse_to_fine | 1 | 150 | -9.16 | 1.04 | 18% | 8.8 |
| coarse_to_fine | 2 | 150 | -3.11 | 0.21 | 9% | 14.8 |


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=1.73 gated_deflation=0.24 ungated_deflation=0.24 -- ordering holds: True
- `real` depth 2: coarse_to_fine=-1.38 gated_deflation=-1.94 ungated_deflation=-1.96 -- ordering holds: True

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
