# Phase 3 results -- phase3_librimix_3spk_refine_ceiling

rows scored: 1200
arms: oracle, real

(means clip +-inf to +-50 dB rather than dropping them)

> **NOT A DEPLOYABLE RESULT.** `refine.oracle_ceiling` was ON, so
> `coarse_to_fine` accepted refinement candidates by comparing SI-SDR
> against the CLEAN SOURCES. Its rows are an upper bound on what any
> acceptance rule could achieve, not a system that could be shipped.
> The other three systems never refine and are unaffected, so they
> remain a valid control within this run.

## Absolute SI-SDR (per arm x system x depth)

| arm | system | depth 1 | depth 2 |
|---|---|---|---|
| oracle | no_recursion | 47.03 | 1.69 |
| oracle | ungated_deflation | 45.98 | 0.24 |
| oracle | gated_deflation | 45.98 | 0.24 |
| oracle | coarse_to_fine | 47.08 | 1.83 |
| real | no_recursion | 40.30 | -1.29 |
| real | ungated_deflation | 40.30 | -1.94 |
| real | gated_deflation | 40.30 | -1.92 |
| real | coarse_to_fine | 40.30 | -1.12 |

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

| arm | system | 0 ms |
|---|---|---|
| oracle | no_recursion | 6.03±0.41 |
| oracle | ungated_deflation | 4.58±0.44 |
| oracle | gated_deflation | 4.58±0.44 |
| oracle | coarse_to_fine | 6.17±0.41 |
| real | no_recursion | 3.02±0.36 |
| real | ungated_deflation | 2.38±0.40 |
| real | gated_deflation | 2.39±0.40 |
| real | coarse_to_fine | 3.19±0.36 |

### Whole output track, one global scale

The literal `si_sdr(output, target)`. SCALE-ANCHORED by the bit-exact
solo copy, so a pure LEVEL error in the overlap region is charged at
full price while every per-depth row discounts it -- which is why this
can land BELOW every depth it appears to summarise (it did so in 271 of
288 rows on 2026-08-23). Kept because it is the only score here that
can see a level error at all; not a summary of the tables above.

| arm | system | 0 ms |
|---|---|---|
| oracle | no_recursion | -16.25±1.06 |
| oracle | ungated_deflation | -13.29±1.01 |
| oracle | gated_deflation | -13.29±1.01 |
| oracle | coarse_to_fine | -15.86±1.04 |
| real | no_recursion | -17.78±1.11 |
| real | ungated_deflation | -15.02±0.88 |
| real | gated_deflation | -15.10±0.88 |
| real | coarse_to_fine | -18.31±1.21 |

### Level disagreement across depths (dB)

`20*log10(max alpha / min alpha)` over the per-depth fitted scales. 0
means one consistent level explains the whole output. Large values are
the mechanism behind any gap between the two tables above, and are
invisible to every scale-invariant metric in this project.

| arm | system | 0 ms |
|---|---|---|
| oracle | no_recursion | 8.78±0.25 |
| oracle | ungated_deflation | 13.66±0.58 |
| oracle | gated_deflation | 13.66±0.58 |
| oracle | coarse_to_fine | 8.94±0.24 |
| real | no_recursion | 6.76±0.31 |
| real | ungated_deflation | 11.03±0.64 |
| real | gated_deflation | 10.88±0.63 |
| real | coarse_to_fine | 7.00±0.31 |


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
| oracle | coarse_to_fine | 150 | 27 (18%) | 45 | 0 | 0 | 0 | 0 | 0 |
| real | gated_deflation | 75 | 74 (99%) | 1 | 0 | 0 | 0 | 0 | 0 |
| real | coarse_to_fine | 150 | 49 (33%) | 38 | 0 | 0 | 0 | 0 | 0 |

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
| coarse_to_fine | 1 | 75 | -6.78 | 1.21 | 24% | 5.6 |
| coarse_to_fine | 2 | 75 | -2.95 | 0.26 | 11% | 11.3 |


## Ordering check (deepest available depth, per arm)

- `oracle` depth 2: coarse_to_fine=1.83 gated_deflation=0.24 ungated_deflation=0.24 -- ordering holds: False
- `real` depth 2: coarse_to_fine=-1.12 gated_deflation=-1.92 ungated_deflation=-1.94 -- ordering holds: True

## Caveats

- LibriMix scenes have hard segment boundaries, no reverb and no
  conversational turn-taking, so a real diarizer's error profile here is
  not its profile on AMI. Treat this gap as a LOWER BOUND on the
  real-corpora gap Phase 4 will measure.
- The deflation systems' `real` vs `oracle` difference mixes diarization
  error with the reordering a now-nonzero `V_i` causes; the
  `real_index_order` rows above separate the two.
