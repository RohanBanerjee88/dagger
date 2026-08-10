# Phase 2 -- accumulation analysis (cross-run)

sources: phase2_librimix_3spk_scratch345clip50.csv, phase2_librimix_4spk_scratch345clip50.csv, phase2_librimix_5spk_scratch345clip50.csv

rows: 23992 scoreable  |  m: [3, 4, 5]  |  depths: [1, 2, 3, 4, 5]

## Mean SI-SDR by scene speaker count `m`, at fixed depth

(read across a row: that is the accumulation axis. Reading down a column is the intrinsic-difficulty axis and is what the per-run depth tables already show.)

| depth | system | m=3 | m=4 | m=5 |
|---|---|---|---|---|
| 1 | no_recursion | 43.78 | 45.16 | 45.80 |
| 1 | ungated_deflation | 40.69 | 40.97 | 40.72 |
| 1 | gated_deflation | 42.09 | 43.27 | 43.88 |
| 1 | coarse_to_fine | 43.73 | 45.13 | 45.79 |
| 2 | no_recursion | 1.26 | 1.12 | 0.51 |
| 2 | ungated_deflation | 0.03 | -1.02 | -1.75 |
| 2 | gated_deflation | 0.46 | 0.08 | -0.55 |
| 2 | coarse_to_fine | 1.19 | 0.83 | 0.10 |
| 3 | no_recursion | -1.29 | -3.66 | -3.32 |
| 3 | ungated_deflation | -2.45 | -5.00 | -5.10 |
| 3 | gated_deflation | -2.09 | -4.49 | -4.05 |
| 3 | coarse_to_fine | -1.42 | -3.89 | -3.51 |
| 4 | no_recursion | -- | -3.47 | -5.63 |
| 4 | ungated_deflation | -- | -4.76 | -6.78 |
| 4 | gated_deflation | -- | -4.23 | -6.12 |
| 4 | coarse_to_fine | -- | -3.60 | -5.72 |
| 5 | no_recursion | -- | -- | -4.87 |
| 5 | ungated_deflation | -- | -- | -6.19 |
| 5 | gated_deflation | -- | -- | -5.51 |
| 5 | coarse_to_fine | -- | -- | -4.95 |

## Control-corrected degradation with `m` (per fixed depth)

(`raw` = mean at m=max minus mean at m=min. `excess` = raw minus `no_recursion`'s raw over the same span -- no_recursion runs no deflation, so its m-dependence is eval-set difficulty and cancels out. Near-zero excess = accumulation-free; strongly negative = accumulating.)

| depth | m span | system | raw (dB) | excess vs control (dB) |
|---|---|---|---|---|
| 1 | 3->5 | no_recursion | +2.01 | control |
| 1 | 3->5 | ungated_deflation | +0.03 | -1.99 |
| 1 | 3->5 | gated_deflation | +1.79 | -0.23 |
| 1 | 3->5 | coarse_to_fine | +2.06 | +0.05 |
| 2 | 3->5 | no_recursion | -0.75 | control |
| 2 | 3->5 | ungated_deflation | -1.77 | -1.02 |
| 2 | 3->5 | gated_deflation | -1.01 | -0.26 |
| 2 | 3->5 | coarse_to_fine | -1.09 | -0.34 |
| 3 | 3->5 | no_recursion | -2.03 | control |
| 3 | 3->5 | ungated_deflation | -2.65 | -0.62 |
| 3 | 3->5 | gated_deflation | -1.96 | +0.06 |
| 3 | 3->5 | coarse_to_fine | -2.10 | -0.07 |
| 4 | 4->5 | no_recursion | -2.16 | control |
| 4 | 4->5 | ungated_deflation | -2.01 | +0.15 |
| 4 | 4->5 | gated_deflation | -1.89 | +0.28 |
| 4 | 4->5 | coarse_to_fine | -2.12 | +0.05 |

## Accumulation-specific gap: coarse_to_fine - ungated_deflation

| depth | m=3 | m=4 | m=5 |
|---|---|---|---|
| 1 | 3.04 | 4.16 | 5.07 |
| 2 | 1.16 | 1.85 | 1.85 |
| 3 | 1.03 | 1.11 | 1.59 |
| 4 | -- | 1.16 | 1.06 |
| 5 | -- | -- | 1.24 |