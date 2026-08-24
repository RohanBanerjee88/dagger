# Phase 3 results -- phase3_librimix_3spk_dilation_v2

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

| arm | system | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|---|
| oracle | no_recursion | 6.03±0.41 | 4.73±0.39 | 3.63±0.39 | 1.76±0.46 |
| oracle | ungated_deflation | 4.58±0.44 | 3.51±0.41 | 2.57±0.40 | 0.92±0.46 |
| oracle | gated_deflation | 4.58±0.44 | 3.51±0.41 | 2.57±0.40 | 0.92±0.46 |
| oracle | coarse_to_fine | 5.44±0.41 | 4.33±0.37 | 3.31±0.37 | 1.55±0.43 |
| real | no_recursion | 3.02±0.36 | 4.29±0.37 | 4.91±0.38 | 3.55±0.40 |
| real | ungated_deflation | 2.38±0.40 | 3.18±0.44 | 3.54±0.46 | 2.21±0.42 |
| real | gated_deflation | 2.39±0.40 | 3.23±0.44 | 3.59±0.46 | 2.21±0.42 |
| real | coarse_to_fine | 2.45±0.36 | 3.72±0.38 | 4.20±0.38 | 3.18±0.38 |

### Whole output track, one global scale

The literal `si_sdr(output, target)`. SCALE-ANCHORED by the bit-exact
solo copy, so a pure LEVEL error in the overlap region is charged at
full price while every per-depth row discounts it -- which is why this
can land BELOW every depth it appears to summarise (it did so in 271 of
288 rows on 2026-08-23). Kept because it is the only score here that
can see a level error at all; not a summary of the tables above.

| arm | system | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|---|
| oracle | no_recursion | -16.25±1.06 | -15.75±1.01 | -15.58±1.03 | -15.14±1.02 |
| oracle | ungated_deflation | -13.29±1.01 | -13.01±0.97 | -12.93±1.00 | -12.62±0.97 |
| oracle | gated_deflation | -13.29±1.01 | -13.01±0.97 | -12.93±1.00 | -12.62±0.97 |
| oracle | coarse_to_fine | -17.73±1.38 | -16.91±1.23 | -16.58±1.23 | -16.01±1.19 |
| real | no_recursion | -17.78±1.11 | -17.35±1.11 | -15.84±0.95 | -15.69±1.02 |
| real | ungated_deflation | -15.02±0.88 | -15.42±1.01 | -13.85±0.81 | -13.28±0.93 |
| real | gated_deflation | -15.10±0.88 | -15.58±1.02 | -14.29±0.90 | -13.28±0.93 |
| real | coarse_to_fine | -17.95±1.24 | -16.71±1.07 | -16.13±1.02 | -17.31±1.23 |

### Level disagreement across depths (dB)

`20*log10(max alpha / min alpha)` over the per-depth fitted scales. 0
means one consistent level explains the whole output. Large values are
the mechanism behind any gap between the two tables above, and are
invisible to every scale-invariant metric in this project.

| arm | system | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|---|
| oracle | no_recursion | 8.78±0.25 | 9.15±0.27 | 9.51±0.28 | 10.33±0.35 |
| oracle | ungated_deflation | 13.66±0.58 | 14.03±0.59 | 14.38±0.60 | 15.24±0.63 |
| oracle | gated_deflation | 13.66±0.58 | 14.03±0.59 | 14.38±0.60 | 15.24±0.63 |
| oracle | coarse_to_fine | 8.20±0.30 | 8.57±0.31 | 8.86±0.34 | 9.70±0.40 |
| real | no_recursion | 6.76±0.31 | 7.80±0.29 | 8.51±0.27 | 9.21±0.31 |
| real | ungated_deflation | 11.03±0.64 | 11.68±0.59 | 12.47±0.55 | 13.55±0.56 |
| real | gated_deflation | 10.88±0.63 | 11.49±0.59 | 12.15±0.53 | 13.55±0.56 |
| real | coarse_to_fine | 5.95±0.41 | 7.04±0.39 | 7.52±0.38 | 8.50±0.33 |


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
