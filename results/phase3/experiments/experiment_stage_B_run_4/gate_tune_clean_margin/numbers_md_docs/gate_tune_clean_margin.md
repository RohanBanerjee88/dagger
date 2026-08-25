# Confidence-gate threshold selection (dev split)

rows: 900  |  populations: honest=150, contaminated=150, correct=150, swapped=150, clean_correct=150, clean_swapped=150

Thresholds are swept INDEPENDENTLY, each against what it is meant to detect -- never jointly against SI-SDR, which for `gated_deflation` is a dial between `ungated_deflation` and `no_recursion` rather than a quality measure. See this script's module docstring.

## Detection sweeps (labelled populations)

### `mean_variance` -- contaminated vs honest

(n=150 honest, n=150 contaminated; a value > the threshold is rejected. detection = contaminated caught, false rej. = honest wrongly rejected. J = detection - false rej.)

| threshold | detection | false rej. | J |
|---|---|---|---|
| 1e-05 | 33.3% | 0.0% | +0.333 |
| 0.0001 | 4.0% | 0.0% | +0.040 |
| 0.0005 | 0.0% | 0.0% | +0.000 |
| 0.001 | 0.0% | 0.0% | +0.000 |
| 0.005 | 0.0% | 0.0% | +0.000 |
| 0.01 | 0.0% | 0.0% | +0.000 |
| 0.05 | 0.0% | 0.0% | +0.000 |

population medians -- honest: 0.00000, contaminated: 0.00000

**suggested `mean_variance`: 1e-05** (highest J = +0.333). Judgement still applies -- if two candidates are within noise, prefer the one that rejects less healthy data, since a false rejection costs real quality on every scene while a missed detection costs only on contaminated ones.

### `margin` -- swapped vs correct

(n=150 correct, n=150 swapped; a value < the threshold is rejected. detection = swapped caught, false rej. = correct wrongly rejected. J = detection - false rej.)

| threshold | detection | false rej. | J |
|---|---|---|---|
| -0.2 | 0.0% | 0.0% | +0.000 |
| -0.1 | 0.7% | 0.7% | +0.000 |
| 0 | 1.3% | 0.7% | +0.007 |
| 0.05 | 1.3% | 0.7% | +0.007 |
| 0.1 | 1.3% | 0.7% | +0.007 |
| 0.15 | 5.3% | 0.7% | +0.047 |
| 0.2 | 8.7% | 4.0% | +0.047 |
| 0.3 | 26.0% | 16.7% | +0.093 |

population medians -- correct: 0.44470, swapped: 0.40648

**NO USABLE THRESHOLD.** The best candidate reaches only J = +0.093 (< 0.1), i.e. this diagnostic barely separates swapped from correct at any value in the grid, so no threshold here would be a real detector. Do NOT copy a value out of this table. Check first whether the fault fixture is doing its job (are the two medians above actually different?) and whether the grid brackets the observed range; only then suspect the diagnostic itself.

## Q1b -- is the margin broken, or just starved by `G`?

Same contrast as `tau_margin` above (n=150 per population), but the
clean source is substituted for `G`'s output -- i.e. what the margin would
score if the extractor were perfect. **NOT DEPLOYABLE**; it is a bound.

* **Separates** -> the formula is sound and purely gated on `G`'s quality.
  It recovers when the extractor does, and no gate redesign is warranted;
  `tau_margin`'s J = +0.046 was a statement about this checkpoint, not
  about `M_i`.
* **Does not separate** -> the margin is broken independently of `G` and
  needs REPLACING rather than re-tuning.


### `margin` -- clean_swapped vs clean_correct

(n=150 clean_correct, n=150 clean_swapped; a value < the threshold is rejected. detection = clean_swapped caught, false rej. = clean_correct wrongly rejected. J = detection - false rej.)

| threshold | detection | false rej. | J |
|---|---|---|---|
| -0.2 | 0.7% | 0.0% | +0.007 |
| -0.1 | 2.7% | 0.0% | +0.027 |
| 0 | 10.0% | 0.0% | +0.100 |
| 0.05 | 13.3% | 0.0% | +0.133 |
| 0.1 | 18.7% | 0.0% | +0.187 |
| 0.15 | 26.0% | 0.0% | +0.260 |
| 0.2 | 32.0% | 0.0% | +0.320 |
| 0.3 | 46.7% | 1.3% | +0.453 |

population medians -- clean_correct: 0.55680, clean_swapped: 0.31409

**suggested `margin`: 0.3** (highest J = +0.453). Judgement still applies -- if two candidates are within noise, prefer the one that rejects less healthy data, since a false rejection costs real quality on every scene while a missed detection costs only on contaminated ones.

## Rate sweeps (no fault population)

### `vad_coverage` -- rejection rate on healthy estimates

(n=150; a value < the threshold is rejected. No fault population exists for this check, so there is no detection rate to trade against -- pick a value in the tail that fires on genuine failures without cutting into normal operation.)

| threshold | healthy rejected |
|---|---|
| 0 | 0.0% |
| 0.25 | 0.0% |
| 0.5 | 0.0% |
| 0.75 | 0.0% |

observed on healthy data: min 0.8238, p5 0.8718, median 0.9486, p95 1.0000, max 1.0000

### `artifact_score` -- rejection rate on healthy estimates

(n=150; a value > the threshold is rejected. No fault population exists for this check, so there is no detection rate to trade against -- pick a value in the tail that fires on genuine failures without cutting into normal operation.)

| threshold | healthy rejected |
|---|---|
| 0.7 | 94.7% |
| 0.8 | 5.3% |
| 0.9 | 0.0% |
| 1 | 0.0% |

observed on healthy data: min 0.6657, p5 0.6967, median 0.7422, p95 0.7989, max 0.8221

## What the current (untuned) thresholds do here

| threshold | value | rejects |
|---|---|---|
| `max_mean_variance` | 0.05 | 0.0% of healthy estimates |
| `tau_margin` | 0.1 | 0.7% of healthy estimates |
| `min_vad_coverage` | 0.5 | 0.0% of healthy estimates |
| `max_artifact_score` | 0.9 | 0.0% of healthy estimates |

## Next step

Freeze one set of thresholds, put the SAME values in every Phase 2 eval config (`gate_cfg` drives both `gated_deflation` and `coarse_to_fine`'s refinement from one dict, so per-system tuning is not available), and confirm with a single `scripts/run_phase2.py` run. The sweeps above are exact for a round-0 decision only -- later deflation decisions change the audio downstream of them.
