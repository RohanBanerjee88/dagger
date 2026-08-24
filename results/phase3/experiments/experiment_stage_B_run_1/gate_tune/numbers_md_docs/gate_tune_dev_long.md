# Confidence-gate threshold selection (dev split)

rows: 601  |  populations: honest=151, contaminated=148, correct=151, swapped=151

Thresholds are swept INDEPENDENTLY, each against what it is meant to detect -- never jointly against SI-SDR, which for `gated_deflation` is a dial between `ungated_deflation` and `no_recursion` rather than a quality measure. See this script's module docstring.

## Detection sweeps (labelled populations)

### `mean_variance` -- contaminated vs honest

(n=151 honest, n=148 contaminated; a value > the threshold is rejected. detection = contaminated caught, false rej. = honest wrongly rejected. J = detection - false rej.)

| threshold | detection | false rej. | J |
|---|---|---|---|
| 1e-05 | 100.0% | 98.0% | +0.020 |
| 0.0001 | 45.3% | 7.9% | +0.373 |
| 0.0005 | 0.0% | 0.0% | +0.000 |
| 0.001 | 0.0% | 0.0% | +0.000 |
| 0.005 | 0.0% | 0.0% | +0.000 |
| 0.01 | 0.0% | 0.0% | +0.000 |
| 0.05 | 0.0% | 0.0% | +0.000 |

population medians -- honest: 0.00005, contaminated: 0.00010

**suggested `mean_variance`: 0.0001** (highest J = +0.373). Judgement still applies -- if two candidates are within noise, prefer the one that rejects less healthy data, since a false rejection costs real quality on every scene while a missed detection costs only on contaminated ones.

### `margin` -- swapped vs correct

(n=151 correct, n=151 swapped; a value < the threshold is rejected. detection = swapped caught, false rej. = correct wrongly rejected. J = detection - false rej.)

| threshold | detection | false rej. | J |
|---|---|---|---|
| -0.2 | 0.0% | 0.0% | +0.000 |
| -0.1 | 0.0% | 0.0% | +0.000 |
| 0 | 0.7% | 0.0% | +0.007 |
| 0.05 | 1.3% | 0.7% | +0.007 |
| 0.1 | 2.0% | 1.3% | +0.007 |
| 0.15 | 2.6% | 1.3% | +0.013 |
| 0.2 | 6.0% | 2.6% | +0.033 |
| 0.3 | 19.9% | 15.2% | +0.046 |

population medians -- correct: 0.43873, swapped: 0.41965

**NO USABLE THRESHOLD.** The best candidate reaches only J = +0.046 (< 0.1), i.e. this diagnostic barely separates swapped from correct at any value in the grid, so no threshold here would be a real detector. Do NOT copy a value out of this table. Check first whether the fault fixture is doing its job (are the two medians above actually different?) and whether the grid brackets the observed range; only then suspect the diagnostic itself.

## Rate sweeps (no fault population)

### `vad_coverage` -- rejection rate on healthy estimates

(n=151; a value < the threshold is rejected. No fault population exists for this check, so there is no detection rate to trade against -- pick a value in the tail that fires on genuine failures without cutting into normal operation.)

| threshold | healthy rejected |
|---|---|
| 0 | 0.0% |
| 0.25 | 0.0% |
| 0.5 | 0.0% |
| 0.75 | 0.7% |

observed on healthy data: min 0.7274, p5 0.9304, median 0.9846, p95 1.0000, max 1.0000

### `artifact_score` -- rejection rate on healthy estimates

(n=151; a value > the threshold is rejected. No fault population exists for this check, so there is no detection rate to trade against -- pick a value in the tail that fires on genuine failures without cutting into normal operation.)

| threshold | healthy rejected |
|---|---|
| 0.7 | 96.0% |
| 0.8 | 18.5% |
| 0.9 | 0.7% |
| 1 | 0.0% |

observed on healthy data: min 0.6717, p5 0.7125, median 0.7647, p95 0.8405, max 0.9149

## What the current (untuned) thresholds do here

| threshold | value | rejects |
|---|---|---|
| `max_mean_variance` | 0.05 | 0.0% of healthy estimates |
| `tau_margin` | 0.1 | 1.3% of healthy estimates |
| `min_vad_coverage` | 0.5 | 0.0% of healthy estimates |
| `max_artifact_score` | 0.9 | 0.7% of healthy estimates |

## Next step

Freeze one set of thresholds, put the SAME values in every Phase 2 eval config (`gate_cfg` drives both `gated_deflation` and `coarse_to_fine`'s refinement from one dict, so per-system tuning is not available), and confirm with a single `scripts/run_phase2.py` run. The sweeps above are exact for a round-0 decision only -- later deflation decisions change the audio downstream of them.
