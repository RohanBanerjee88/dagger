# Phase 2 -- accumulation analysis (cross-run)

sources: phase2_librimix_3spk_scratch345.csv, phase2_librimix_4spk_scratch345.csv, phase2_librimix_5spk_scratch345.csv

rows: 23992 scoreable  |  m: [3, 4, 5]  |  depths: [1, 2, 3, 4, 5]

## Mean SI-SDR by scene speaker count `m`, at fixed depth

(read across a row: that is the accumulation axis. Reading down a column is the intrinsic-difficulty axis and is what the per-run depth tables already show.)

| depth | system | m=3 | m=4 | m=5 |
|---|---|---|---|---|
| 1 | no_recursion | 43.90 | 45.17 | 45.82 |
| 1 | ungated_deflation | 40.75 | 40.89 | 40.52 |
| 1 | gated_deflation | 42.24 | 43.35 | 43.92 |
| 1 | coarse_to_fine | 43.90 | 45.18 | 45.83 |
| 2 | no_recursion | 1.60 | 1.33 | 0.65 |
| 2 | ungated_deflation | -0.25 | -1.30 | -1.95 |
| 2 | gated_deflation | 0.44 | 0.04 | -0.67 |
| 2 | coarse_to_fine | 1.18 | 0.81 | 0.11 |
| 3 | no_recursion | -1.19 | -3.47 | -3.08 |
| 3 | ungated_deflation | -2.47 | -5.17 | -5.41 |
| 3 | gated_deflation | -2.03 | -4.49 | -4.29 |
| 3 | coarse_to_fine | -1.47 | -3.88 | -3.31 |
| 4 | no_recursion | -- | -3.35 | -5.47 |
| 4 | ungated_deflation | -- | -4.73 | -6.83 |
| 4 | gated_deflation | -- | -4.20 | -6.11 |
| 4 | coarse_to_fine | -- | -3.59 | -5.64 |
| 5 | no_recursion | -- | -- | -4.76 |
| 5 | ungated_deflation | -- | -- | -6.17 |
| 5 | gated_deflation | -- | -- | -5.50 |
| 5 | coarse_to_fine | -- | -- | -4.91 |

## Control-corrected degradation with `m` (per fixed depth)

(`raw` = mean at m=max minus mean at m=min. `excess` = raw minus `no_recursion`'s raw over the same span -- no_recursion runs no deflation, so its m-dependence is eval-set difficulty and cancels out. Near-zero excess = accumulation-free; strongly negative = accumulating.)

| depth | m span | system | raw (dB) | excess vs control (dB) |
|---|---|---|---|---|
| 1 | 3->5 | no_recursion | +1.92 | control |
| 1 | 3->5 | ungated_deflation | -0.22 | -2.14 |
| 1 | 3->5 | gated_deflation | +1.68 | -0.24 |
| 1 | 3->5 | coarse_to_fine | +1.93 | +0.01 |
| 2 | 3->5 | no_recursion | -0.95 | control |
| 2 | 3->5 | ungated_deflation | -1.70 | -0.75 |
| 2 | 3->5 | gated_deflation | -1.11 | -0.16 |
| 2 | 3->5 | coarse_to_fine | -1.07 | -0.12 |
| 3 | 3->5 | no_recursion | -1.90 | control |
| 3 | 3->5 | ungated_deflation | -2.94 | -1.04 |
| 3 | 3->5 | gated_deflation | -2.26 | -0.36 |
| 3 | 3->5 | coarse_to_fine | -1.84 | +0.06 |
| 4 | 4->5 | no_recursion | -2.12 | control |
| 4 | 4->5 | ungated_deflation | -2.11 | +0.01 |
| 4 | 4->5 | gated_deflation | -1.91 | +0.21 |
| 4 | 4->5 | coarse_to_fine | -2.05 | +0.07 |

## Accumulation-specific gap: coarse_to_fine - ungated_deflation

| depth | m=3 | m=4 | m=5 |
|---|---|---|---|
| 1 | 3.15 | 4.29 | 5.31 |
| 2 | 1.42 | 2.11 | 2.05 |
| 3 | 1.00 | 1.29 | 2.10 |
| 4 | -- | 1.14 | 1.20 |
| 5 | -- | -- | 1.27 |