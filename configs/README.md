# Configs

Laid out as `configs/phase<N>/{dod,experiments}/`.

- **`dod/`** — the configs that produce that phase's **Definition of Done** result, the numbers
  reported in CLAUDE.md §5. If you want to reproduce a headline figure, it is in here. Keep this
  directory small: one training config and its evals per phase.
- **`experiments/`** — everything else. Pilots, smoke tests, diagnostics, parameter sweeps, and
  superseded runs kept because their results are cited in CLAUDE.md's status notes. A config lives
  here if its numbers are *evidence for a decision* rather than *a reported result*.

Nothing is deleted when a run is superseded — CLAUDE.md's Phase 2 history refers to specific
configs by name, and those references have to keep resolving.

## What produces each DoD

| phase | configs | produces |
|---|---|---|
| 0 | `phase0/dod/phase0_librimix.yaml`, `phase0_wsj0mix.yaml` | mixture → oracle regions → copied-solo output → metrics, both loaders |
| 1 | `phase1/dod/phase1_librimix_3spk_train.yaml` (`--system proposed\|blind`), then `phase1_librimix_3spk_eval.yaml` | proposed 4.40 dB vs blind 2.05 dB overlap SI-SDR (2026-07-13) |
| 2 | `phase2/dod/phase2_librimix_curriculum_3_4_5_train_scratch.yaml`, then `phase2_librimix_{3,4,5}spk_eval_scratch.yaml` | the ordering + accumulation figures (see the Phase 2 close-out block in CLAUDE.md) |

Phase 2's training config is deliberately **not warm-started** — every earlier Phase 2 checkpoint
was initialized from the one before it, so reproducing a result meant replaying the whole chain.
The `dod/` config trains from random init, so the reported checkpoint comes from one command
(CLAUDE.md §7).

## Notable experiment groups (Phase 2)

| group | configs | what it settled |
|---|---|---|
| curriculum pilots / full | `*_curriculum_3_4_5_train_{pilot,full}.yaml` + `*_eval_curriculum*.yaml` | multi-depth training widens the accumulation gap; fixed-depth fine-tuning narrows it |
| depth-5 pilot | `phase2_librimix_5spk_train_pilot.yaml`, `*_eval_pilot.yaml` | normalization + depth-5 exposure raises the floor but narrows the gap |
| enrollment-budget sweep | `phase2_librimix_5spk_enroll_budget_{150,300,500,800}.yaml` | starving enrollment shrinks the refinement deficit only by shutting the gate off (degenerate) |
| heterogeneous sources | `phase2_librimix_3spk_{hetero,homo}.yaml` | refinement is still net-negative when the enrolled recording differs from the extracted one |
| gate tuning | `phase2_gate_tune_dev.yaml` | dev-split threshold selection; deferred to Phase 3, since `V_i` is structurally zero under oracle diarization |
| smokes | `*_smoke*.yaml` | 5-scene sanity checks — run one before any long job |

## Conventions

- Every result comes from a `scripts/run_phaseN.py` (or `train_phase1.py`) plus one of these files.
  No number in CLAUDE.md should be unreproducible from a single command.
- `eval.tag` distinguishes runs against different checkpoints so one never silently overwrites
  another's results in `results/`.
- Scripts' `--config` defaults still point at the config they pointed at before this reorganization,
  so bare invocations behave as they always did.
