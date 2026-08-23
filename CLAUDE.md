# CLAUDE.md — Project Guide & Human Checklist

> **How to use this file.** It lives at the repo root. Claude Code reads it automatically
> at the start of every session, so it never loses the plot. You (the human) read it to
> check each step is done right before moving on. When something changes, update this file
> first — it is the single source of truth.

---

## 0. What we are building (one paragraph)

We recover **one clean audio track per speaker** from a single-channel recording where people
talk over each other. We use **speaker diarization** to find the moments where each person
talks *alone*, turn those moments into a voice "fingerprint" (an embedding), and then use that
fingerprint to pull each speaker out of the overlapping parts. The twist that makes this work:
we **never subtract voices from a running residual** in the audio we output — every speaker is
extracted directly from the original mixture. That single choice is what makes our error stay
small no matter how many people overlap.

Working repo name: **`dagger`** (provisional — verify it's free on GitHub + PyPI before committing).
License: **Apache-2.0** (needs `LICENSE` + `NOTICE` at root).

---

## 1. THE ONE RULE YOU MUST NEVER BREAK

**Every speaker's output waveform is computed from the untouched overlap mixture `x_O`.**

```
ŝ_i(t) = x(t)·w_Ei(t)  +  G(x_O(t), ē_i)·w_Oi(t)
                              ^^^^ always the ORIGINAL mixture, never a residual
```

Recursion is allowed, but **only** to (a) refine the embedding `ē_i` and (b) decide the order
we process speakers in. Recursion must **never** feed a subtracted/residual signal into `G`
to produce audio the listener hears.

**Why this matters (the theorem in plain words):** if you extract from a residual, each step's
mistake gets baked into the next step, and the error grows roughly *linearly* with the number
of overlapping speakers. If you extract from the original mixture every time, each speaker's
error stands alone and **does not accumulate**. This is the entire point of the paper. If you
ever see code subtracting `ŝ` from `x_O` and feeding the result back into `G` for output —
**stop, that's the bug we exist to avoid.**

---

## 2. Facts that are mathematically settled (do not "re-derive" and break them)

These are proven in `docs/diarization_full_mathematical_theory.pdf`. Treat them as ground truth.

- **Solo regions are clean.** Where only speaker *i* is active, `x = s_i + n`. So a solo clip
  is a valid enrollment sample. ✔
- **Copy, don't separate, on solo regions.** Running a network on already-clean audio only adds
  artifacts. Solo parts are copied straight through. ✔
- **Error accumulation is NOT "monotone."** An earlier draft claimed error "grows monotonically
  every step." **That's false.** The correct statement is three regimes: worst-case *linear*,
  independent-errors *√m*, realistic-correlated *linear*. Never reintroduce the "monotone" claim. ✔
- **The reconstruction loss needs a noise term.** `‖x_O − Σ ŝ_i‖²` alone fights the separation
  loss whenever noise ≠ 0. Use `‖x_O − Σ ŝ_i − n̂‖²` (a noise head) OR train on noise-free data. ✔
- **Leakage uses a MARGIN, not raw similarity.** Raw `cos(ŝ_i, e_j)` is always positive (voices
  aren't orthogonal). Use the margin `M_i = cos(ŝ_i, e_i) − max_{j≠i} cos(ŝ_i, e_j)`. ✔
- **The gate can't check its own enrollment.** If enrollment is contaminated, the confidence
  score happily passes it. Guard it *before* the gate with the enrollment-variance check `V_i`. ✔
- **Soft masks at seams.** Hard on/off masks click and starve the network of context. Use smooth
  crossfaded windows (`w_Ei + w_Oi = 1`). ✔

**Symbol cheat-sheet** (matches the theory doc): `x` mixture · `s_i` clean speaker *i* ·
`ŝ_i` our estimate · `n` noise · `a_i(t)` diarization activity · `E_i` solo region ·
`O_i` overlap region for *i* · `x_O` overlap mixture · `φ` speaker encoder · `e_i`/`ē_i`
embedding / mean embedding · `G` extractor · `ε_i` extraction error · `M_i` identity margin ·
`V_i` enrollment variance · `ν` noise floor · `τ` gate threshold.

---

## 3. Tech stack (what each module uses)

| Module | Tool / model | Notes |
|---|---|---|
| Diarization | **pyannote.audio 4.0.x + `community-1`** (full repo id: `pyannote/speaker-diarization-community-1`) | Gated on HF (free); token in `DAGGER_HF_TOKEN`. The short name `pyannote/community-1` is NOT a real repo and 404s. Its 4.x pipeline returns a `DiarizeOutput`, and you must read `.speaker_diarization` — **never** `.exclusive_speaker_diarization`, which allows at most one speaker per instant and would silently delete every overlap. Ungated fallback: NeMo Sortformer. |
| Speaker encoder `φ` | **NVIDIA NeMo TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`) | Freeze pretrained weights first; fine-tune late. Revised from the original WeSpeaker+ReDimNet2 pick during Phase 1 planning: consolidates with NeMo Sortformer (the diarizer fallback below) on one framework. Checkpoint is CC-BY-4.0 (attributed in `NOTICE`); toolkit is Apache-2.0. |
| **Eval-only** encoder | `microsoft/wavlm-base-plus-sv` via `transformers` | MUST differ from `φ` — chosen over Kiwano for a simple pip-installable path; architecturally unrelated to TitaNet. |
| Extractor `G` | **TF-GridNet + cross-attention fusion**, original implementation from the published architecture (informed by the USEF-TSE paper, arXiv:2409.02615) | Not vendored from USEF-TSE (CC-BY-NC 4.0, incompatible with this repo's Apache-2.0) or WeSep (no license file). Conv-TasNet+FiLM fast baseline was skipped in favor of building this directly. |
| Signal metrics | **torchmetrics** (SI-SDR, SDR) | |
| Intelligibility | **Whisper large-v3** for WER | |
| Refinement precedent | TS-SEP / EvoTSE (for related-work + comparison) | Our novelty = the honest gate + accumulation-free proof. |

> Verify exact APIs/versions against current docs when scaffolding — these move fast.

---

## 4. Repo layout (target)

```
dagger/
├── CLAUDE.md                 # this file
├── LICENSE                   # Apache-2.0
├── NOTICE                    # attribution (Apache-2.0 expects this)
├── README.md
├── pyproject.toml
├── configs/                  # yaml configs per experiment
├── dagger/
│   ├── diarize/              # pyannote wrapper → activity matrix a_i(t), regions E_i/O_i
│   ├── enroll/               # top-K solo segments, φ embedding, mean ē_i, variance V_i
│   ├── extract/              # G: TF-GridNet + cross-attn fusion
│   ├── refine/               # confidence-gated embedding refinement + speaker ordering
│   ├── gate/                 # margin M_i, VAD, artifact score, threshold τ
│   ├── reconstruct/          # soft-mask stitching (partition of unity)
│   ├── losses/               # SI-SDR, speaker-consistency, noise-head recon, artifact
│   └── metrics/              # SI-SDR/SDR, margin (eval encoder), Whisper WER
├── scripts/                  # run_phaseN.py entrypoints
├── tests/                    # unit tests, esp. the "no residual in audio path" guard
└── docs/
    ├── diarization_full_mathematical_theory.pdf   # the full proof doc (12 sections + master
    │                                               # theorem) -- the why behind every module
    └── diarization_corrected_proofs.pdf           # shorter errata companion: the same
                                                    # monotone-growth correction, condensed
```

---

## 5. Phase-by-phase plan (the checklist)

> Work **one phase at a time**. Do not start a phase until the previous one's "Definition of
> done" is green. Each phase says what to build, how to know it worked, and what a screw-up
> looks like.

### ☑ Phase 0 — Plumbing (no learning yet) — DoD MET 2026-07-03

**Goal:** data flows end to end with *oracle* diarization, and metrics compute.

**Build:** dataset loaders (WSJ0-2mix, LibriMix); an **oracle diarization path** that reads
ground-truth RTTM instead of running a diarizer; the metrics harness (SI-SDR, speaker margin,
Whisper WER); the "copy solo regions" reconstruction with no extractor yet.

**How to check it worked:** feeding ground-truth sources through the harness gives near-perfect
scores; copying solo regions recovers solo audio exactly.

**Red flags:** metric values that look impossibly good on overlaps (you're leaking ground truth),
or sample-rate/framerate misalignment between diarization masks and audio frames.

**Definition of done:** one command runs mixture → (oracle regions) → copied-solo output →
metrics, on a handful of files, with sane numbers.

**Status (marked complete 2026-08-18, retroactively).** Landed in `c8312aa` (2026-07-03) with
`scripts/run_phase0.py` + `configs/phase0/dod/phase0_{librimix,wsj0mix}.yaml` and six test modules
(`tests/phase0/`). The box went unticked for a year of project time purely as bookkeeping — every
later phase is built on this path and could not have produced its numbers if it were wrong.

*The DoD criterion is met, and later runs measure it continuously.* "Copying solo regions recovers
solo audio exactly" shows up directly as `+inf` SI-SDR rows (a perfect estimate has zero error
energy): the Phase 3 long-scene oracle arm produced **792 `+inf` rows**, and its depth-1 mean is
**47.62 dB** with `±inf` clipped to ±50. That is the Phase 0 guarantee being re-verified on real
data every time a later phase runs, which is stronger evidence than a one-off Phase 0 table.

*One deviation from §7 worth knowing:* `run_phase0.py` prints to stdout and writes no CSV/`.md`, so
unlike Phases 1-3 there is no committed `results/phase0/`. The command is reproducible; its output
was never captured. Not worth fixing retroactively — the `+inf` evidence above supersedes it — but
do not go looking for a results file that does not exist.

### ☑ Phase 1 — Identity conditioning (validates: targeting beats blind separation) — DoD MET 2026-07-13

**Goal:** extract each speaker from the mixture using their embedding. **No recursion yet.**

**Build:** `enroll/` (top-K solo clips → `φ` → mean `ē_i`); `extract/` (`G(x_O, ē_i)`);
wire into reconstruction. Add a **blind-separation baseline** for comparison.

**How to check it worked:** on 2-speaker clean data you approach the literature bar
(~23 dB SI-SDR on WSJ0-2mix); on **3+ speakers** your method holds up while the blind baseline
visibly merges speakers.

**Red flags:** speaker-similarity computed with the *training* encoder (metric hygiene violation);
`G` receiving `x·1_Oi` with hard masks (fix later, but note it); enrollment taken from overlap
by accident.

**Definition of done:** proposed > blind on 3-speaker mixtures, oracle diarization, table saved.

**⚠ KNOWN ISSUE (found 2026-07-11, first Phase 1 runs): `stagger_offsets` starves speakers of
solo time on 3+ speakers.** Chain placement starts utterance *i+1* at `(1 − overlap)` into
utterance *i* (`dagger/data/mixing.py`), so solo time depends on *random length ratios*: with
`overlap: 0.5`, the middle speaker of a 3-mix gets a solo window only if `L2 > L1`, and the last
only if `L3 > ~0.5·L2`. Result on Libri3Mix: **~70–80% of scenes are skipped at enrollment**
(`NoSoloRegionError`, caught and logged — not a crash), silently shrinking the effective
training/eval sets and biasing survivors toward `L2 > L1` orderings. The mixing docstring's
promise ("every mixture has a solo lead-in, an overlap middle, and a solo tail") only holds for
2 speakers. The skip-on-no-solo behavior itself is correct and stays (unenrollable speakers are
real; Phase 3 must handle them) — the bug is that our own generator manufactures them.
*Fixed (2026-07-11):* `stagger_offsets` now takes `min_solo` (samples) and pushes each start
just late enough that every speaker keeps a contiguous solo window of
`min(min_solo, own length)`; both loaders pass it via the `min_solo_ms` config key
(default 1000 ms — above enrollment's 500 ms `min_clip_ms`). `min_solo_ms: 0` restores the
legacy length-ratio-dependent behavior. Trade-off: the guarantee takes precedence over
`overlap`, so adjacent short utterances may overlap less than requested.
*Phase 2 heads-up:* a chain-staggered scene where the middle speaker has solo time **cannot**
contain a depth-3 overlap (s2's solo requires s1 to end before s3 starts). The depth-stratified
experiment needs both per-speaker solos *and* deep overlaps, so Phase 2 placement must become a
small scheduler (e.g., per speaker: one guaranteed solo segment + one deliberately deep
overlapped segment) — the solo-aware offset fix above is not sufficient for Phase 2.

**⚠ KNOWN ISSUE (found 2026-07-11, first full training run): degenerate loss terms drowned the
training signal for `G`.** Symptoms: training loss flat at ~40 (i.e. −40 dB SI-SDR — worse than
outputting the mixture, impossible for a real comparison) for 25 epochs, and proposed vs blind
eval rows agreeing within ~0.3 dB — two different architectures both stuck near passthrough.
Cause: uniform-random 4 s training crops usually miss a given speaker's overlap window, and
SI-SDR is *scale-invariant*, so a ~zero windowed target can't express "output silence" — the
term degenerates to `−10·log10(eps)` (~+80) with a garbage gradient that swamps the scoreable
terms. The `min_solo` fix made this *more* common (it reduces overlap by design). Same disease,
milder, in the blind system's PIT loss (silent speaker in crop).
*Fixed (2026-07-11):* (a) `scripts/train_phase1.py` masks out (crop, speaker) terms whose
windowed target has ~zero energy and averages over scoreable terms only (skipped terms also
skip their forward pass); (b) `dagger/losses/pit.py` masks silent target speakers out of the
per-item mean and drops all-silent items; (c) `dagger/data/torch_adapter.py` centers each crop
on a random overlap sample (uniform over overlap samples via precomputed run boundaries;
uniform-start fallback when a scene has no overlap), and `require_overlap=True` (used by
proposed training) drops zero-overlap scenes with a logged count — such scenes still exist
because the `min_solo` guarantee can push short utterances fully clear of the chain; at eval
they appear as `overlap: n/a` rows (speaker is 100% copy-path), which is correct behavior.
Healthy-training signature going forward: loss starts ~5–15 and *trends down*; a flat loss
near +40 means degenerate terms are back.

**⚠ OPEN ISSUE (found 2026-07-11, second full training run, after the loss fixes): the proposed
extractor collapses to passthrough — its embedding conditioning is not being used.** Evidence:
with the fixed losses, the **blind** baseline now trains cleanly (loss 0.36 → −2.26 over 25
epochs; eval overlap SI-SDR 0.01 → 2.26 dB), but the **proposed** system's loss oscillates
around 0.1–0.7 with no trend, and its eval `overlap(prop)` column is identical (±0.01 dB)
across two independently trained runs — only possible if both runs output ≈ a scaled copy of
`x_O` (SI-SDR is scale-invariant, so every `c·x_O` scores exactly what the mixture scores).
Mechanism: one output head is asked for three different answers from the *same* input,
disambiguated only by `ē_i`; if the conditioning pathway is too weak to matter early, the three
per-speaker gradients on identical input cancel and "output the mixture" is a stable resting
point (the blind system escapes because its 3 PIT-matched heads can specialize without
conflicting gradients). No wiring bug found in `extract/tfgridnet_crossattn.py` /
`extract/crossattn.py` / `extract/tfgridnet.py` — the suspicion is *dosage*, not design:
config injects the fusion before only **1 of 6** blocks (`cross_attn_blocks: 1`; the backbone
already supports fusing before every block), the 192-d embedding is compressed to only
`n_tokens: 4` key/value tokens, and raw (unnormalized) TitaNet embeddings feed `token_proj`
(early noisy FiLM scales incentivize the optimizer to mute the pathway).
*Next actions (diagnose before spending GPU):* (1) embedding-sensitivity probe on the saved
checkpoint — `G(x, e_A)` vs `G(x, e_B)` vs `G(x, random)`; near-identical outputs confirm the
collapse; (2) overfit-4-scenes test — if proposed cannot drive its loss strongly negative even
when memorizing, the conditioning pathway is underpowered; (3) if confirmed, remedies in order:
`cross_attn_blocks: 6` (one YAML line), L2-normalize the embedding before `token_proj`,
`n_tokens: 8`, then retrain. Note the current blind-beats-proposed table is **not** a DoD
verdict: blind gets oracle best-permutation matching, and proposed's 0.13 dB is a passthrough
artifact, not a measurement of a working extractor — the Phase 1 comparison hasn't actually
been run yet.
*Diagnosed (2026-07-12): the architecture is fine — the failure is optimization (passthrough
is a plateau the optimizer must escape, and at lr 1e-3 it never does).* Evidence chain:
(a) the dosage remedies (`cross_attn_blocks: 6`, L2-norm in the fusion module, `n_tokens: 8`)
were applied and a fresh capped run (400 scenes / 25 epochs) *still* landed at 0.14 dB overlap —
third run within 0.01 dB of passthrough; (b) the embedding-sensitivity probe
(`scripts/probe_phase1_conditioning.py`) on that checkpoint showed the pathway ALIVE but not
steering (outputs change ~5% when swapping embeddings; SI-SDR vs `x_O` = 35.8 dB ≈ scaled
mixture copy; diag−offdiag margin 0.16 dB ≈ 0); (c) the overfit-4-scenes run
(`configs/phase1/experiments/phase1_overfit4_diag.yaml`, lr 3e-4, 600 single-batch epochs) sat at the passthrough
plateau for ~170 steps, then **escaped**: final loss ~−2 to −3, and the probe on its checkpoint
returned STEERS (outputs change 86% across embeddings, passthrough down to 8.1 dB, diag
+2.38 dB vs offdiag −3.04 dB — pointing G at speaker j actively suppresses speaker i).
Structural remedies (aux speaker-consistency loss, silent-target energy terms, mixture
dropout) are NOT needed on current evidence. *Remedy applied (2026-07-12):* `lr: 3e-4` in the
train config and gradient clipping (`train.grad_clip`, default max-norm 5.0, both systems) in
`scripts/train_phase1.py` — the overfit log showed single unclipped steps (+1-to-+2 loss
spikes) repeatedly erasing hundreds of steps of descent. Retrain pending. Expect a plateau
phase near ~0.3–0.5 loss before escape; a run has failed only if it is still flat at the END,
not because it starts flat.
*RESOLVED (2026-07-13): Phase 1 DoD met.* The lr 3e-4 + grad-clip retrain escaped the plateau
(400 scenes/25 epochs: 1.75 dB vs blind's 1.03; probe on *test* scenes: STEERS). Scaled runs
(Kaggle batch, one T4 each: `limit: 2000, epochs: 30, batch 4, lr 3e-4, grad_clip 10`;
`torch_adapter` now stores crops compactly — float32 audio + uint8 masks, ~3× less host RAM —
after the prepared-scenes list OOMed ~30 GB at 2000 scenes) produced the DoD table
(150 test scenes, 450 rows):
**proposed 4.40 dB vs blind 2.05 dB mean overlap SI-SDR (+2.35 dB)**; probe: passthrough
2.89 dB, diag +5.80 vs off-diag −6.95 (12.8 dB steering margin — grew with data: 5.6 dB at
400 scenes). Caveats recorded honestly: (a) per-row win rate is only 50% (paired std 7.51 dB) —
the mean margin comes from magnitude asymmetry (proposed's wins are much larger than its
losses); Phase 2's depth stratification should locate where the big wins live; (b) both
systems are undertrained (2000 of ~34k Libri3Mix train-360 recipes; loss still descending at
cutoff) so all numbers are lower bounds; (c) the 2-speaker WSJ0-2mix literature-bar check is
deferred — no LDC license — substitute Libri2Mix if ever needed. Reproduce: train both systems
with `configs/phase1/dod/phase1_librimix_3spk_train.yaml` (`--system proposed|blind`), eval with
`scripts/run_phase1.py --config configs/phase1/dod/phase1_librimix_3spk_eval.yaml` (limit 150).

### ☑ Phase 2 — THE money experiment (validates: accumulation-free reconstruction) — DoD MET 2026-08-10

**Goal:** prove the central claim empirically.

**Build:** three more systems on top of Phase 1 — (a) **ungated deflation** (extract from
residual), (b) **gated deflation**, (c) **coarse-to-fine** (recursion refines embeddings only;
audio always from `x_O`). Add confidence gate (`M_i`, VAD, artifact) and embedding refinement.

**How to check it worked — the plot that makes the paper:** stratify every metric by
**overlap depth |K|**. Deflation should **degrade roughly linearly** with depth; coarse-to-fine
should stay **flat**.

**Red flags:** coarse-to-fine secretly reading a residual for output (re-read §1); gate using
raw leakage instead of the margin; refinement with no gate accepting bad embeddings.

**Definition of done:** the depth-stratified plot clearly shows flat (ours) vs sloped (deflation),
and ordering proposed ≥ gated > ungated on 3+ speakers.

**Status (2026-07-22): code + unit tests landed.** All Phase 2 components are implemented and
unit-tested (synthetic fixtures only, no GPU/data needed; `pytest tests/phase2/` green, full suite
194 passed at the time). See the "First real run" note below for what happened once this ran
against real data. System naming, to avoid confusion with Phase 1's "proposed" (= this phase's
`no_recursion`):

- `no_recursion` — exactly Phase 1's proposed path, unchanged.
- `ungated_deflation` / `gated_deflation` — the deliberate residual anti-pattern (CLAUDE.md §1),
  built only for comparison. Isolated in `dagger/reconstruct/deflation.py`, which is the ONLY
  place in the repo allowed to feed a residual into `G` (via `_extract_from_residual`, which
  bypasses `Extractor.extract()`'s guard on purpose). `dagger/refine/coarse_to_fine.py` never
  imports it (enforced by an `ast`-based test) and never imports `TrackedSignal`/`Provenance` at
  all, so it is structurally incapable of building a residual.
- `coarse_to_fine` — "ours": recursion refines the embedding only (`dagger/refine/coarse_to_fine.py`),
  every round's audio comes from the unmodified, guarded `reconstruct_all`.

No retraining for Phase 2 — all four systems condition the one Phase 1 checkpoint
(`checkpoints/phase1/proposed_librimix_3spk.pt`) differently at inference time.

**New scene scheduler** (`dagger.data.mixing.schedule_solo_then_overlap`, config key
`dataset.placement: scheduled`, default remains `"chain"` = Phase 0/1 behavior unchanged): gives
every speaker a guaranteed non-overlapping solo slot, then places every speaker's remaining audio
starting at the same synchronized offset, reaching depth == num_speakers before tapering as
shorter clips end. Needed because the Phase 0/1 chain-staggered mixer cannot give a speaker both
solo time and participation in a 3-way overlap in the same scene (see the Phase 1 "heads-up" note
above). Works with existing Libri3Mix metadata — no new corpus/metadata needed.

**First real run (2026-07-26): ordering criterion met, flat-vs-sloped criterion only weakly
met — DoD NOT yet called.** `scripts/run_phase2.py` against real Libri3Mix (150 scheduled-placement
test scenes, oracle diarization) + the Phase 1 checkpoint (`checkpoints/phase1/proposed_librimix_3spk.pt`,
untuned default gate thresholds), mean SI-SDR by depth:

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.16 | 2.96 | -0.59 |
| ungated_deflation | 39.60 | -1.28 | -6.05 |
| gated_deflation | 40.90 | 0.12 | -3.96 |
| coarse_to_fine | 43.17 | 2.35 | -1.33 |

**Ordering holds cleanly:** `coarse_to_fine (-1.33) > gated_deflation (-3.96) > ungated_deflation
(-6.05)` at depth 3, and the depth-2→3 *drop* increases monotonically in the theoretically-correct
order too (no_recursion -3.55 dB, coarse_to_fine -3.68 dB, gated_deflation -4.09 dB,
ungated_deflation -4.77 dB — accumulation-free systems degrade least, ungated deflation degrades
most). **"Flat vs sloped" is only directionally supported, not the clean "clearly flat vs clearly
sloped" plot CLAUDE.md's DoD language wants** — every system (including the two accumulation-free
ones) drops sharply from depth 1 to depth 2, and keeps declining depth 2→3; the theoretically-
predicted differences ride on top of that shared decline as a modest few-dB effect, not a
dramatically different shape. Leading hypothesis: the Phase 1 checkpoint was trained under
`dataset.placement: chain` (the default), which — per this same section's Phase 1 "heads-up" note —
structurally produces very few genuine depth-3 overlaps, so `G` is comparatively out-of-distribution
at depth 3 for *every* system, compressing everyone toward a shared floor and muting the visible
gap between architectures even though the relative-degradation ordering still comes through.
Not a data-leakage or diarization-alignment red flag (CLAUDE.md §5 red flags) — those would produce
impossibly-good numbers, not a shared floor.

**Bug found + fixed while reading the first results:** `_write_results` (and the plot script)
originally filtered with `np.isfinite(si_sdr)`, which silently drops both `nan` (correct — speaker
not active at that depth, nothing to score) *and* `±inf` (wrong — `si_sdr()` legitimately returns
`+inf` for a perfect estimate / `-inf` for a silent one against real target energy; those are
informative outcomes, not undefined ones). 44% of depth-1 rows were exactly `+inf` (solo
copy-through, expected) and were vanishing from the reported depth-1 mean, understating it (32–40
dB shown vs. 39.6–43.2 dB correct). Depth 2/3 — the comparison that actually matters for the DoD —
had zero `±inf` rows and were unaffected. Fixed: both scripts now clip `±inf` to `±50 dB` before
averaging instead of dropping it, and the `.md` output gained a diagnostic-counts table (absent /
perfect / failed / scored, per system/depth) so this class of bug is visible directly in the report
next time.

**Next step:** most likely worth training (or continuing training) `G` with `dataset.placement:
scheduled` (or a chain+scheduled mix) so it gets real depth-3 exposure, before re-running the
depth-stratified eval — that should sharpen the flat-vs-sloped contrast if the compressed-floor
hypothesis above is right. Gate thresholds in `configs/phase2/experiments/phase2_librimix_3spk_eval.yaml`
(`tau_margin`, `max_mean_variance`, `min_vad_coverage`, `max_artifact_score`) are still untuned
defaults and haven't been revisited given the above.

**Fine-tune attempt (2026-07-27): checkpoint recovered at epoch 25/30, not the full 30.**
`scripts/train_phase1.py` gained an optional `train.init_checkpoint` key so a run can warm-start
from an existing checkpoint's weights instead of random init; `configs/phase2/experiments/phase2_librimix_3spk_train_scheduled.yaml`
fine-tunes `checkpoints/phase1/proposed_librimix_3spk.pt` on `dataset.placement: scheduled` scenes
(same 2000-scene/30-epoch scale as the original Phase 1 run, for comparability) to give `G` real
depth-3 exposure. First attempt on Kaggle hit the platform's ~12h (43200s) max execution ceiling
and was SIGKILLed (exit 137) during epoch 30 -- the observed per-epoch cost (~1460s steady-state
+ ~580s one-time setup) put the full 30-epoch plan at ~44,400s, just over the cap. The mid-run
checkpoint at epoch 25 (`checkpoint_every: 5`, overwriting) survived in the canceled version's
output and was used as-is (loss -2.95 there vs. -3.28 at epoch 29, so not far off where it was
headed) rather than re-running with a lower epoch count. Evaluated via
`configs/phase2/experiments/phase2_librimix_3spk_eval_finetuned.yaml` (`eval.tag: finetuned`, writes
`results/phase2_librimix_3spk_finetuned.{csv,md}` alongside the original
`results/phase2_librimix_3spk.{csv,md}` for a before/after comparison) -- see the next status note
for what this produced, once it's run.

**Fine-tuned-checkpoint result (2026-07-29): ordering replicated on a second, independent
checkpoint, but the fine-tune did NOT sharpen the accumulation-specific gap -- DoD still not
called.** 150 scheduled-placement test scenes, same eval harness, checkpoint =
`checkpoints/phase2/proposed_librimix_3spk_scheduled.pt` (epoch-25 fine-tune above). Confirmed
deterministic: rerunning the identical config against the identical checkpoint reproduced the
table to the decimal (no RNG anywhere in the eval path -- inference-mode extractor, no shuffling,
deterministic scene placement/enrollment/deflation-order/gate).

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.33 | 4.14 | 0.53 |
| ungated_deflation | 41.72 | 0.49 | -3.92 |
| gated_deflation | 42.19 | 1.84 | -2.19 |
| coarse_to_fine | 43.36 | 4.21 | 0.21 |

Absolute floor rose for every system at depth 2/3 (roughly +1 to +2 dB each, e.g. `no_recursion`
depth 3: -0.59 -> +0.53) -- confirms half the hypothesis: `G` really was undertrained on true
depth-3 overlap, and the scheduled-placement exposure helped. Ordering held again
(`coarse_to_fine (0.21) > gated_deflation (-2.19) > ungated_deflation (-3.92)` at depth 3) --
a second, independently fine-tuned checkpoint reproducing the same correctly-signed order is a
real replication, not nothing.

**But the accumulation-specific signal itself did not grow -- if anything it shrank for the worst
offender.** Isolating the "extra drop beyond `no_recursion`'s own depth-2-to-3 drop" (the same
technique used to read the first run, see above):

| | extra drop -- BEFORE | extra drop -- AFTER (fine-tuned) |
|---|---|---|
| `coarse_to_fine` | 0.13 dB | 0.39 dB |
| `gated_deflation` | 0.53 dB | 0.42 dB |
| `ungated_deflation` | 1.22 dB | **0.80 dB** |

Fine-tuning raised everyone's floor roughly equally rather than disproportionately helping the
accumulation-free systems. Leading explanation: Theorem 3's deflation penalty is `L*||E_(m-1)||`
-- fine-tuning likely shrank the intrinsic error `ε` (why absolute scores rose) but there's no
reason it would also shrink the extractor's sensitivity `L` to a corrupted residual, and a more
capable/responsive extractor could plausibly be *more* sensitive to corruption, roughly
offsetting the `ε` improvement in the deflation-specific term. Also plausible this is partly
150-scene run-to-run noise (~1 dB swings have been seen before in this pipeline, e.g. the blind
baseline's training-seed variance). **Conclusion: retraining alone is not the lever that sharpens
this experiment.** The still-open options from the "next step" note above (push to 4-5 speaker
overlap depth, since Theorem 2's accumulation term grows faster than linearly and shouldn't
depend on `ε`/`L` improving; and/or the min/max-band reporting TODO) are more promising than a
further retrain.

**Reporting TODO — DONE (verify before re-adding): add empirical min/max (or a percentile band),
not just the mean, to the depth-stratified table/plot.** The motivation stands — a single mean
can't distinguish "consistently mediocre" from "usually great with a few bad scenes," and for a
finite sample the infimum/supremum are just the observed min/max, with a 5th/95th percentile band
more robust to one freak-bad scene than the raw min. **Both halves have since landed:**
`_write_results` emits a "Spread (per system/depth)" table with `n / mean / p5 / p95 / min / max`,
and `plot_phase2_depth.py` takes `--band p5p95`. Note the trap this TODO sets, now spelled out in
§7: that band is about **spread**, while the figures' default error bars are **SEM**, about the
precision of the mean — they differ by ~30x on this data and must never be drawn unlabeled.
Separately (bigger, not scheduled yet): Theorem 2's `‖E_m‖ ≤ mε`
is a *proven* worst-case bound, not a measurement — a further validation step would be estimating
`ε`/`L` empirically and checking our measured worst case actually falls under what that formula
predicts; that fits Phase 4's full results section better than a Phase 2 bolt-on.

**4-5 speaker overlap depth (2026-07-30): first zero-shot look, 50 scenes -- ordering holds at
every depth, gap trends wider, 150-scene rerun pending.** Took the "push to 4-5 speaker overlap
depth" option from the note above rather than another retrain: `scripts/extend_librimix_metadata.py`
synthesizes 4/5-speaker LibriMix metadata by borrowing extra (speaker, utterance, gain) triples from
other rows of the existing `libri3mix_test.csv` (same LibriSpeech audio already on disk, no new
corpus, deterministic given `--seed`) -- nothing in `dagger/data/mixing.py`, `reconstruct/`, `gate/`,
or `refine/` is capped at 3 speakers (all key off `activity.shape[0]`), so this is purely a metadata
exercise. `configs/phase2/experiments/phase2_librimix_4spk_eval.yaml` / `_5spk_eval.yaml` run the SAME scheduled-
placement fine-tuned checkpoint (`checkpoints/phase2/proposed_librimix_3spk_scheduled.pt`)
zero-shot -- no retraining -- against these deeper scenes.

First run (`limit: 50`, tags `zeroshot4`/`zeroshot5`) result, coarse_to_fine vs. ungated_deflation
gap by depth:

| depth | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| gap (dB) | 4.92 | 4.23 | 4.76 | 5.66 |

The ordering `coarse_to_fine > gated_deflation > ungated_deflation` now holds cleanly at **every**
depth 2 through 5 (previously only checked at the deepest available depth), and the gap trends
wider rather than staying flat/noise-level -- the first run where the accumulation-specific signal
looks like it's actually growing with depth, which is what the push to deeper overlap was for.
Two caveats before calling this real: (a) absolute SI-SDR at depth 4-5 is deeply negative (-4 to
-10 dB) for every system -- expected, not alarming, since this checkpoint was only ever fine-tuned
on scenes reaching real depth-3 overlap, so depth 4-5 is genuinely out-of-distribution for it (same
mechanism as the pre-scheduled-placement depth-3 shared-floor collapse, see the "first real run"
note above); read this as a relative/ordering result, not an absolute-quality one; (b) 50 scenes is
thin enough that the gap's wobble (4.92 -> 4.23 -> 4.76 -> 5.66) could partly be sampling noise
rather than genuine monotonic widening. **Next: rerun both configs at `limit: 150`** (already set in
both configs, matching the existing 3-speaker eval scale) to see whether the ordering and the
widening trend survive a larger sample -- not yet run as of this update.

**Depth-agnostic extractor investigation (2026-07-30): Stage 1 (waveform-level energy
normalization) implemented and unit-tested; not yet run on real data.** The depth-4/5 zero-shot
collapse above (-4 to -10 dB) affects `no_recursion` too, which has no gate/deflation logic at
all -- ruling out the gate and pointing at the raw extractor `G`'s training exposure. Two research
passes (a full codebase read of `dagger/extract/tfgridnet.py`/`crossattn.py`/`tfgridnet_crossattn.py`
and `dagger/data/torch_adapter.py`/`mixing.py`, plus a literature search on TF-GridNet/USEF-TSE and
the broader separation literature) converged on two findings: (a) `G` is architecturally
speaker-count-agnostic already (no layer sized to a number of concurrent speakers) -- the failure
is purely distributional; (b) there is no waveform-level energy normalization anywhere in the
pipeline, so the raw mixture amplitude scales with however many concurrent sources happen to be
summed, and every training scene so far summed at most 3 -- a 5-way overlap is a genuinely unseen
amplitude/density regime for the LSTM/attention nonlinearities inside each TF-GridNet block (the
per-block `GroupNorm` renormalizes *after* those nonlinearities run, not before, so it doesn't fully
protect against this). Separately, TF-GridNet and USEF-TSE are both only ever trained/evaluated at
a fixed speaker count in their own papers, and fixed-N-trained separators generalizing poorly to a
different N is a known, established phenomenon in the broader field (e.g. Cone of Silence,
arXiv:2010.06007) -- this repo's own pattern of one fine-tune per fixed depth reproduces exactly
that setup. The literature's standard mitigation (recursive one-and-rest extraction from a residual)
is explicitly incompatible with CLAUDE.md §1's central rule and is a non-starter here.

Two-stage remedy planned (`/Users/adityaarchunananand/.claude/plans/ok-while-i-am-humble-rain.md`):
**Stage 1** (this update) removes the amplitude confound; **Stage 2** (not started) trains `G` on a
*mix* of depths in one run instead of one fixed depth per run, addressing the fixed-N-training root
cause directly. Some intrinsic difficulty increase with depth is still expected regardless (more
concurrent voices genuinely means more simultaneous spectral masking) -- "depth-agnostic" means
much better zero-shot transfer, not identical absolute quality at every depth.

Stage 1 landed: new `dagger/audio/normalize.py` (`active_rms` computed only over the *active*/nonzero
region of a waveform, not the whole tensor -- training crops are short and overlap-centered while
inference feeds whole scenes masked to just the overlap region, so a whole-tensor RMS would conflate
"how much of this tensor is zero-padding" with "how many sources are summed"; `normalize`/`denormalize`
built on it). Wired into `_TFGridNetCrossAttnModule.forward` (`dagger/extract/tfgridnet_crossattn.py`)
only -- normalize on entry, run the backbone/fusion/mask computation unchanged, denormalize on exit.
This is the single call site both training and inference already funnel through, so no other file
needed changes. Verified by hand this gives an *exact* property (not just an improvement claim):
`active_rms(k*x) = k*active_rms(x)` for `k>0`, so the network's internal computation is bit-identical
regardless of input scale, meaning `module(k*x, e) == k*module(x, e)` exactly up to floating point --
added as a mechanical unit test rather than a fuzzy check. Full suite green (202/202, no regressions);
a synthetic 20-step training loop (forward/loss/backward/optimizer-step, alternating input amplitude
>10x between steps to mimic the depth-3-vs-5 energy swing) stayed numerically stable throughout.
Could not run the literal `configs/phase1/experiments/phase1_smoke.yaml` CPU check -- needs real LibriMix/LibriSpeech
audio under `DAGGER_DATA_ROOT`, which only exists on Kaggle, not the local dev machine -- the
synthetic loop is the closest available substitute and is arguably a harsher check.

This change invalidates every existing checkpoint (the input distribution `input_conv`/GroupNorm/FiLM
adapted to shifts). Per the plan, no dedicated GPU run was spent re-validating this in isolation --
the already-queued `configs/phase2/experiments/phase2_librimix_5spk_train_pilot.yaml` (warm-started from
`checkpoints/phase2/proposed_librimix_3spk_scheduled.pt`) was the vehicle for the first run against
the new code.

**Decision-gate result (2026-07-31): floor rose sharply at depth 4-5, but the gap narrowed, not
widened -- reconfirms the 2026-07-29 finding via a different route, and points at Stage 2, not more
single-depth fine-tuning.** Training ran clean (loss 4.80 -> 1.59 monotonically over 15 epochs; one
gradient-norm spike to 4134 at epoch 13 was absorbed cleanly by `grad_clip: 10.0` with no loss
disruption -- clipping working exactly as intended, not a sign to lower `lr`). Eval via
`configs/phase2/experiments/phase2_librimix_4spk_eval_pilot.yaml` / `_5spk_eval_pilot.yaml` (150 test scenes each),
compared against the pre-Stage-1 zero-shot numbers logged above:

| system | depth 2 (old→new) | depth 3 (old→new) | depth 4 (old→new) | depth 5 (old→new) |
|---|---|---|---|---|
| no_recursion | 4.60→1.85 (-2.75) | -1.21→-2.45 (-1.24) | -5.19→-3.48 (+1.71) | -4.08→-4.80 (-0.72) |
| ungated_deflation | -0.88→-1.32 (-0.44) | -5.68→-4.81 (+0.87) | -9.94→-5.08 (+4.86) | -10.22→-6.56 (+3.66) |
| gated_deflation | 1.53→0.44 (-1.09) | -3.72→-3.36 (+0.36) | -8.41→-4.27 (+4.14) | -7.44→-5.55 (+1.89) |
| coarse_to_fine | 4.04→1.43 (-2.61) | -1.45→-2.85 (-1.40) | -5.18→-4.04 (+1.14) | -4.56→-5.35 (-0.79) |

Ordering (`coarse_to_fine > gated_deflation > ungated_deflation`) still holds cleanly at both depth 4
and depth 5 (`ordering holds: True` printed both times) -- the core thesis is intact. But the
accumulation-specific gap (coarse_to_fine minus ungated_deflation) shrank at every depth rather than
widening: 4.92→2.75 (depth 2), 4.23→1.96 (depth 3), 4.76→1.30 (depth 4), 5.66→1.21 (depth 5).
`ungated_deflation` improved the most because it started worst (most room to improve) -- exactly the
"training raises everyone's floor, disproportionately helping whoever started worst, which narrows
rather than sharpens the gap" pattern already logged on 2026-07-29, now reproduced via depth-5
exposure + normalization together instead of more epochs at the same depth. New wrinkle this run
surfaced: depth 2-3 quietly *regressed* for the non-deflation systems (no_recursion, coarse_to_fine)
by 1-3 dB -- plausibly because `schedule_solo_then_overlap` scenes spend most of their overlap-zone
time near peak depth before tapering, so this fine-tune's data skewed heavily toward deep overlap,
specializing the network toward deep overlap at some cost to shallow-overlap quality it already had.

Caveat: this run bundled two changes at once (normalization + new depth-5 training exposure), a
deliberate simplification to avoid a throwaway GPU run -- so today's numbers credit "normalization +
this training" together, not normalization in isolation. **Conclusion: a second single-depth
fine-tune (even combined with normalization) keeps reproducing the same raise-the-floor/narrow-the-
gap pattern regardless of exactly how the extra training is delivered.** Stage 2 (train on a genuine
*mix* of depths in one run, not one more depth-specific fine-tune) is next.

**Stage 2 (2026-07-31): multi-depth curriculum training implemented and unit-tested; not yet run on
real data.** `scripts/train_phase1.py`'s `train_proposed` now accepts `dataset:` as either a single
dict (unchanged behavior) or a *list* of per-depth dataset configs -- each entry builds its own
loader via the existing, unmodified `build_dataset`/`build_scene_crop_dataset`; one shared model and
optimizer see batches drawn from all loaders, interleaved in a per-epoch-shuffled order (fresh
iterators + a shuffled draw-order list built from each loader's length, so every individual batch
still comes from exactly one loader -- internally uniform in `num_speakers` -- meaning no custom
collate/padding was needed anywhere). A single-entry list reduces to exactly the prior single-dataset
code path, not a special case. Checkpoints now optionally record `trained_n_src` for provenance.
Verified with two new tests (`tests/phase2/test_train_phase1_curriculum.py`, `build_dataset` and
`TitaNetEncoder` faked so no real corpus/GPU is needed): a 2-depth (`n_src` 2 and 3) curriculum run
trains without error and its saved checkpoint records both depths; a single-dict config still trains
and records exactly one depth (the backward-compatibility check). Full suite green (204/204).

`configs/phase2/experiments/phase2_librimix_curriculum_3_4_5_train_pilot.yaml` is the first real run to try: `dataset:`
as a 3-entry list (n_src 3/4/5, `placement: scheduled`, ~130 scenes/depth, 15 epochs -- matching the
existing single-depth pilot's total scene budget rather than multiplying it by 3, since this is an
extrapolation from the n_src=3 timing anchor, not a measurement), warm-started from the depth-5 pilot
checkpoint (already Stage-1-normalized and depth-5-exposed, so this run only has to add "a mix of
depths" on top of that). Needs n_src=4 training-split metadata generated first (`n_src=5` was already
generated for the depth-5 pilot but Kaggle sessions don't persist, so regenerate both if starting
fresh) -- the command is in the config's header comment. Matching eval configs
(`phase2_librimix_{3,4,5}spk_eval_curriculum.yaml`, tag `curriculum345`) are ready to compare against
both the pre-Stage-1 zero-shot numbers and the single-depth pilot's numbers logged above.

**Decision-gate result (2026-07-31): first clean widening signal in the whole investigation.**
Evaluated the curriculum checkpoint on all three test sets (native 3-speaker, 4-speaker, 5-speaker).
Ordering (`coarse_to_fine > gated_deflation > ungated_deflation`) holds cleanly at every depth 2
through 5 across all three runs, no exceptions. Comparing the accumulation-specific gap
(coarse_to_fine minus ungated_deflation) against the single-depth-5 pilot on the two test sets where
a matching pilot number exists (the pilot was never evaluated on the native 3-speaker set):

| depth | eval set | pilot gap | curriculum gap | change |
|---|---|---|---|---|
| 2 | 4spk-test | 4.10 | 5.34 | +1.24 |
| 3 | 4spk-test | 2.47 | 3.37 | +0.90 |
| 4 | 4spk-test | 1.75 | 2.34 | +0.59 |
| 2 | 5spk-test | 4.22 | 5.07 | +0.85 |
| 3 | 5spk-test | 3.77 | 4.49 | +0.72 |
| 4 | 5spk-test | 2.58 | 2.61 | +0.03 (flat) |
| 5 | 5spk-test | 1.88 | 2.38 | +0.50 |

Six of seven widened, one flat, zero narrowed -- the first time any intervention in this investigation
has widened the gap rather than narrowing it (a fixed-depth fine-tune, and normalization bundled with
one, both narrowed it). Mechanism: the widening is driven mostly by the deflation systems getting
*worse* under curriculum training (ungated dropped 0.5-1.0 dB at most depths, gated a smaller but
consistent 0.1-0.2 dB), not by `coarse_to_fine` improving sharply -- `no_recursion`/`coarse_to_fine`
held essentially flat at every depth (≤0.3 dB moves), avoiding the depth-2/3 regression the
single-depth pilot caused. Plausible explanation: `G` is never trained on residual/corrupted input
(deflation only hands it one at *inference* time); curriculum training makes `G` a sharper extractor
for genuine clean-mixture input across a range of depths, and that specialization doesn't transfer to
-- and may actively hurt -- robustness against the residual corruption deflation feeds it.

Caveat: the native-3-speaker-test comparison against the original 2026-07-29 depth-3-only checkpoint
(gap dropped 4.13→2.28) is confounded -- that baseline predates Stage 1, so the comparison mixes
"curriculum vs. single-depth" with "pre- vs. post-normalization code" and a specialist-vs-generalist
checkpoint tradeoff. Not read as contradicting the clean 4spk/5spk-test comparisons above, which hold
code and eval harness fixed and vary only the training data.

**Absolute quality is still poor at depth 4-5** (roughly -4 to -8 dB across systems) even with the
gap now widening correctly -- this pilot was sized as a cheap mechanism check (130 scenes/depth, 15
epochs), not a quality-maximizing run. Next candidate: scale the curriculum run up (more scenes/
epochs per depth, mirroring the 400→2000-scene jump that produced Phase 1's actual DoD-worthy
numbers) now that the mechanism itself is validated as directionally correct.

---

#### What "curriculum training" means here (for anyone reading this repo cold)

Not curriculum learning in the usual easy→hard-ordering sense. In this repo it means one training
run whose batches are drawn from **several overlap depths at once** instead of a single fixed
speaker count.

Background: `G` is architecturally speaker-count-agnostic (nothing is sized to a number of
concurrent speakers -- every module keys off `activity.shape[0]`), but a network trained only on
3-speaker mixtures is *distributionally* specialized to them, and transfers poorly to 4 or 5. That
is a known effect in the separation literature, and every run here before 2026-07-31 reproduced it.

The implementation (`scripts/train_phase1.py`): `dataset:` in the config may be a single dict
(one depth, the original behavior) **or a list** of per-depth entries. Each entry builds its own
loader through the unmodified `build_dataset`; one shared model and optimizer then see batches
interleaved from all loaders in a per-epoch-shuffled order. Every individual batch still comes from
exactly one loader, so it is internally uniform in `num_speakers` and no custom collate or padding
was needed. A single-entry list reduces to exactly the prior single-depth code path.

Why it mattered: fixed-depth fine-tuning repeatedly raised the absolute floor while *narrowing* the
accumulation-specific gap (2026-07-29, 2026-07-31), because it disproportionately helped whichever
system started worst. Curriculum training was the first intervention to widen the gap instead --
it makes `G` a sharper extractor for genuine clean-mixture input across a range of depths, and that
sharpening does not transfer to the corrupted-residual input that deflation feeds it at inference.

---

**PHASE 2 CLOSE-OUT (2026-08-04): DoD met on the ordering + flat-vs-sloped criteria, with the axis
corrected; one pending run to regenerate everything from a scratch-trained checkpoint.**

*The axis correction -- the single most important lesson of this phase.* Five runs plotted against
overlap depth read as "directionally supported, never clean." The reason: **depth is not the
accumulation counter.** `reconstruct_all_deflation` deflates once per scene over all `m` speakers,
so the error a speaker inherits is set by how many prior estimates were subtracted before it --
not by how many voices happen to be concurrent in whichever region is later scored. Depth measures
*intrinsic difficulty*, which hits all four systems equally and buries a between-system effect.
Two variables were being summed into one column. The instrumentation added on 2026-08-02
(`m`, `deflation_index`, `n_accepted_before` per row; gate decisions in a separate `_gate.csv` at
their own grain) separates them, and the effect is immediate. No retraining was involved -- the
quantity was always in the pipeline, merely never recorded.

*Ordering.* `coarse_to_fine > gated_deflation > ungated_deflation` holds at every depth 2-5 across
all three eval sets, without exception.

*The two figures.*

- **Primary -- `n_accepted_before` at the deepest depth, deflation systems only.** SI-SDR against
  the number of prior estimates already subtracted into the residual: the exact index of Theorem
  3's `L*||E_(m-1)||` penalty. Monotone at every `m`, with balanced n=150 per level (every speaker
  participates in the peak-depth region by construction). At m=5, depth 5:
  `-3.84 -> -6.67 -> -8.20 -> -8.94 -> -9.14`. The accumulation-free systems do not appear because
  they take zero deflation steps -- for them the property is proven structurally (`refine/` never
  imports `deflation`, never builds a residual, enforced by an `ast`-based test), not measured. A
  measurement cannot demonstrate the absence of a mechanism better than the architecture does.
  Generate at the DEEPEST depth: shallower depths have unbalanced n across levels (only some
  speakers have samples there) and the curve wobbles for that reason alone.
- **Secondary -- `m` sweep at fixed depth 2, all four systems.** Less direct but far more legible,
  and it shows the same effect *between* scenes. `no_recursion` is flat (+0.16 dB from m=3 to m=5),
  which is what licenses the cross-eval-set comparison at all; `ungated_deflation` falls 2.35 dB,
  `gated_deflation` 1.66. Depth 2 specifically, because it is the only depth where the control is
  genuinely flat and the figure needs no correction to be honest.

*The accumulation decline is sub-linear, not linear* (m=5 steps: -2.83, -1.53, -0.74, -0.20).
Theorem 2's `||E_m|| <= m*eps` is an upper bound and the measurement sits well under it. Report
this before a reader computes `m*eps` and asks. An enrollment-order confound exists (deflation
order is ascending `V_i`, so level 0 is always the best-enrolled speaker) and is bounded at
**~0.45 dB against a 5.30 dB effect** -- under 10%, estimated from gated's level-0 population,
which includes later-in-order speakers whose predecessors were all rejected.

*Absolute SI-SDR is negative at depths 4-5 for every system, including `no_recursion`.* This is the
extractor's operating point, not the reconstruction strategy: the curriculum checkpoint was
650 scenes/depth x 15 epochs against Phase 1's 2000 x 30 at a single depth. Phase 2's claims are
relative -- every comparison holds scenes, mixture, and checkpoint fixed and varies only the
reconstruction strategy -- and the accumulation result replicated on **three independently
constructed corpora whose absolute levels span 5 dB** (`no_recursion` at the deepest depth: -4.03,
-0.23, +0.75). It is not an artifact of a weak checkpoint. Do NOT attribute the negative absolutes
to refinement: `no_recursion` has none and is equally negative.

*Refinement: a real bug, then an earned negative result.* `coarse_to_fine` sat below `no_recursion`
at every depth. Cause found in `refine/coarse_to_fine.py`: the gate was called with
`embedding_self=blended`, where `blended = 0.5*e_i + 0.5*raw` contains the very embedding being
judged, so `identity_margin` computed `cos(theta/2)` instead of `cos(theta)` -- inflating every
candidate, and inflating the *worst* ones most (at 90 degrees, 0.00 becomes 0.71). Signature: a
98-99% accept rate that did not move with speaker count, while the identically-thresholded deflation
gate tracked difficulty correctly (71.8 -> 61.2 -> 54.1% at m=3/4/5). Fixed by passing
`embeddings[i]`; accept rate became 60.2/49.3/39.1% and the deficit roughly halved. The residual
deficit is real, and refinement was then tested in three regimes, all with clean (oracle)
enrollment:

| regime | result |
|---|---|
| stock, 1 s enrollment | -0.2 to -1.1 dB |
| starved, 150-800 ms (`enroll.budget_ms`) | deficit shrinks, but only via gate shutdown (94% ties) -- **degenerate, not evidence** |
| heterogeneous, 4 s (solo from utterance A, overlap from B) | **-0.36 / -0.69 dB with the gate healthy at 59.6%** |

The heterogeneous arm is the decisive one: enrollment stayed long and clean so extraction quality
held and the gate stayed open, and refinement still lost (122 losses against 72 wins at depth 2,
ties only 35%). Its matched control (`same-chapter` pairing, identical geometry) came in at
-0.77 / -0.83, so heterogeneity helped by 0.14-0.41 dB -- directionally consistent with the
mechanism, but ~1.5 sigma across two different scene sets, so not established. **Conclusion:
refinement is net-harmful when enrollment is correct, and is reported as an optional stage, off by
default (`refine.rounds: 0` makes `coarse_to_fine` bit-identical to `no_recursion`).** The one
untested regime is *contaminated* enrollment, where the baseline is actually broken -- that is
Phase 3, and it is why `V_i` exists and has never once fired. Also untested: a variance-weighted
blend instead of the fixed `0.5/0.5`, which would down-weight a noisy candidate automatically and
needs no benefit test. A negative result on one update rule is not one on the family.

*New finding -- the confidence gate is not an accumulation detector.* On the heterogeneous corpus
(4 s enrollment) `gated_deflation` accepted **98.4%** and collapsed onto `ungated_deflation`
(-4.11 vs -4.13 at depth 3); the control, also 4 s, gave 97.6%. With a good reference embedding,
residual-corrupted estimates still score high margins -- `M_i` detects **wrong-speaker**, not
**degraded-same-speaker**. Its rejections in the stock runs were driven substantially by enrollment
noise keeping margins near `tau_margin`, not by recognizing residual damage. Consistent with the
earlier decomposition that **80-96% of gated's advantage was a shift in the accumulation
distribution**, not better estimates. Do not present "gated beats ungated" as a robust property:
it holds only when the gate actually rejects, which depends on the operating point.

*Gate threshold tuning stays deferred to Phase 3* (`scripts/tune_gate.py`,
`configs/phase2/experiments/phase2_gate_tune_dev.yaml`, and the `offset` dataset key exist and are tested). Three of
the four thresholds behave sensibly at their defaults, and the fourth cannot be tuned at all today:
`schedule_solo_then_overlap` gives one solo run, `select_topk_solo_clips` returns one clip, and
`var` across one clip is identically 0, so `V_i` is structurally dead until real diarization
supplies multiple segments of differing quality. Tuning must happen on the dev split (`offset:
650`), never on test -- and note `gate_cfg` is shared, so raising `tau_margin` also tightens
`gated_deflation` toward `no_recursion`, the degenerate direction.

> **2026-08-20, on the two halves of that paragraph.** The `V_i` sentence is the ONLY statement of
> it in this file that was fully correct: "structurally dead **until real diarization supplies
> multiple segments of differing quality**" is exactly what happened, and the caveat that later
> notes dropped is what made it right. Keep the qualifier when quoting it.
> But **"three of the four thresholds behave sensibly at their defaults" did NOT survive.**
> `tau_margin: 0.1` scores Youden's J = +0.046 against swapped conditioning ("no usable
> threshold"), and `max_mean_variance: 0.05` is 500x above `V_i`'s usable range. "Behaves sensibly"
> meant "fires at a plausible-looking rate", which is not evidence that a check detects anything --
> that is what a fault fixture is for, and none had been run. (`offset: 650` above is also stale;
> it is now 1000.) See Stage B Session B Q3.

*Pending:* one scratch-trained curriculum run at larger scale
(`configs/phase2/dod/phase2_librimix_curriculum_3_4_5_train_scratch.yaml`, no `init_checkpoint`, so the
reported checkpoint comes from one command rather than a chain of prior fine-tunes), then re-run
the three evals and generate both figures. Expect absolute quality similar to or slightly below the
warm-started checkpoint at equal compute -- that is the price of reproducibility, and it does not
affect the relative claims. Figure generation needs **no GPU**: `plot_phase2_depth.py` and
`aggregate_phase2.py` are CPU-only and read the existing CSVs.

---

**SCRATCH-TRAINED DoD RUN (2026-08-09): provenance is clean and the ordering claim fully
replicates, but the accumulation MAGNITUDE weakened ~3x -- traced to a gradient-clipping defect
that only bites when training from random init. One re-run pending; report the checkpoint-robust
claims now, hold the magnitude.**

The run above was executed. `configs/phase2/dod/phase2_librimix_curriculum_3_4_5_train_scratch.yaml`
completed **all 10/10 epochs** (loss 4.69 -> 1.67, monotone) and the reported checkpoint is the
epoch-10 save, not an interrupted `checkpoint_every: 2` save; no committed config was edited at
runtime. Wall clock 18:27 -> 00:58 = **6h31m** training against the config's 7h32m budget, plus
~2.2h of evals -- ~8.7h of the ~12h ceiling, so **~3.3h of headroom went unused** (room for ~5 more
epochs). Evals: `configs/phase2/dod/phase2_librimix_{3,4,5}spk_eval_scratch.yaml`, 150 test scenes
each, tag `scratch345`, results in `results/phase2/dod/`. NOTE: `run_phase2.py` writes CSVs and `.md`s FLAT into its `eval.results_dir`; the `numbers_csv/` / `numbers_md_docs/` / `graphs/` split under both `dod/` and `dod_final/` is manual curation applied afterwards, so a re-run drops flat files that need sorting. The §7 "one command regenerates this"
requirement is therefore **met** for the first time in Phase 2.

#### Report these -- they are checkpoint-robust and this run is their strongest evidence

1. **Ordering: 9 of 9, no exceptions.** `coarse_to_fine > gated_deflation > ungated_deflation` at
   every depth 2-5 across all three eval sets. This now holds on a checkpoint built from random
   init by a single command, in addition to every warm-started checkpoint before it. This is the
   DoD's primary criterion and it is met.
2. **Refinement is net-harmful under clean enrollment -- replicated.** `coarse_to_fine -
   no_recursion` is negative at all 9 slices (-0.15 to -0.54 dB), losses ~2x wins, with the gate
   *healthy*: accept rate 49.3 / 36.3 / 26.1% at m=3/4/5, tracking difficulty downward. No sign of
   the degenerate 94%-tie gate shutdown seen in the starved-enrollment arm, so this is an earned
   negative, not an artifact. Confirms `refine.rounds: 0` as the default.
3. **`V_i` is confirmed structurally dead.** Zero variance rejections in 5,400 gate decisions;
   `mean_variance` is *exactly* 0.0 in 69-82% of them and <= 2e-4 otherwise, against a 0.05
   threshold. Exactly as predicted -- it stays a Phase 3 item and cannot be tuned until real
   diarization supplies multiple enrollment segments of differing quality.
   > **PARTLY CORRECTED (2026-08-20).** "Structurally 0 under oracle diarization" is right and
   > stays -- one solo run gives one clip and the variance over one sample is 0 by definition. But
   > "dead" does not follow: once real diarization supplies several clips, `V_i` scores J = +0.373
   > at a **1e-4** threshold. The `0.05` this compares against is 500x too high. See Stage B
   > Session B Q3.

#### The accumulation magnitude weakened, and the terminal step is a new finding

Primary quantity (`ungated_deflation`, `n_accepted_before`, m=5 at depth 5, n=150 per level --
balanced by construction):

| checkpoint | lvl 0 | lvl 1 | lvl 2 | lvl 3 | lvl 4 | total |
|---|---|---|---|---|---|---|
| warm-started (2026-08-04) | -3.84 | -6.67 | -8.20 | -8.94 | -9.14 | **-5.30 dB**, monotone |
| scratch (this run) | -4.89 | -5.93 | -6.35 | -6.98 | -6.72 | **-1.83 dB**, upticks at the end |

Paired within-scene, the *total* 0->4 decline is solid (**t = -4.5**), but per-step only 0->1 is
individually significant (t = -2.2) and the final step is **+0.26 dB (t = +0.8)**.

**The terminal uptick is not noise -- it replicates on all three corpora** (3spk last step -0.07,
t=-0.2; 4spk +0.12, t=+0.3; 5spk +0.26, t=+0.8). There is a mechanism, and it should be *stated*
in the paper rather than left for a reader to find: the last deflation step is the **one-and-rest
endpoint**. Its residual is `x_O` minus *every* other estimate, which already approximates that
speaker's own source, so `G` has an unusually easy job. Theorem 2's `||E_m|| <= m*eps` is an upper
bound and the terminal step is a benign special case of it -- accumulation is monotone through the
*body* of the chain and flattens at the endpoint. Both figures encode this by drawing the final
segment **dashed**.

**The enrollment-order confound is now negligible and points the conservative way.** Re-estimated
by the same technique as 2026-08-04 (gated's `n_accepted_before == 0` population split by
`deflation_index`): **+0.07 dB at 5spk** (vs ~0.45 dB before) and **-1.05 dB at 3spk** -- negative
meaning later-in-order speakers score *better*, which would understate the measured decline. So the
shrinkage is real, not a confound artifact.

#### Root cause of the weakening: `grad_clip` saturated into a hidden LR schedule

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| median grad norm | 1.10 | 3.70 | 6.86 | 7.86 | 8.99 | 11.12 | 12.93 | 14.15 | 14.76 | **15.14** |
| clipped % | 4 | 11 | 20 | 27 | 37 | 61 | 76 | 86 | 92 | **93** |
| loss | 4.69 | 4.25 | 3.71 | 3.27 | 2.95 | 2.64 | 2.26 | 1.94 | 1.74 | 1.67 |

`_grad_norm_summary`'s own docstring (`scripts/train_phase1.py`) states the rule this violates:
"a median at/above the clip means every step is being rescaled -- i.e. the clip is acting as a
hidden LR reduction -- so raise it." The median crossed `grad_clip: 10.0` at epoch 6 and ended at
1.5x it; the last four epochs were effectively normalized-gradient descent at a shrinking step
size. **The loss "converging" is confounded with exactly that** -- per-epoch improvement decays
-0.31 / -0.20 / -0.07 at epochs 8/9/10, precisely where the clip rate hits 86/92/93%. From this
log, "converged" and "throttled" are indistinguishable.

*Why it bit here and never before:* `si_sdr_loss` has gradient magnitude that **grows** as the
estimate approaches the target (error energy sits in the denominator), so a fixed clip threshold
becomes progressively more binding the better the model gets -- it is self-limiting by
construction. `grad_clip: 10.0` was inherited from Phase 1, where clipping was rare spike
protection (2026-07-12). The warm-start chain never traversed the high-gradient region from random
init, so it never tripped this. **Training from scratch is the specific thing that exposed it.**
Genuine spikes do still occur (max 1520 / 2183 / 896), so clipping is still needed -- just not at a
threshold that catches 93% of ordinary steps.

Consistent with this, absolute quality came in **below** the config's "similar or slightly below"
prediction -- `no_recursion` at each set's deepest depth: 3spk -1.19 (was +0.75), 4spk -3.35 (was
-0.23), 5spk -4.76 (was -4.03), i.e. 0.7-3.1 dB worse.

**The honest cross-checkpoint statement, and it belongs in the paper:** the *ordering* is robust --
it has held on every checkpoint ever trained here. The accumulation *magnitude* is not: it varies
~3x (5.30 vs 1.83 dB) between two checkpoints trained on the same curriculum recipe differing only
in warm-start. Claim the ordering; report the magnitude as a range with the checkpoint named.

*Re-run queued:* same config, `grad_clip` raised (~50, or a running-percentile clip, which is the
principled fix given the SI-SDR gradient growth), optionally `epochs: 13` to spend the idle 3.3h.
That settles whether the 3x shrinkage is a checkpoint artifact or a property of the measurement.
Either outcome is publishable; only the unexplained version is not.

#### Corrected figures (regenerated 2026-08-09, CPU-only, from the committed script)

Both live in `results/phase2/dod/graphs/` and are regenerated by one command each (see the reporting-code
section below). Two reporting defects were found while regenerating them from `scratch345`, both
now fixed in `scripts/plot_phase2_depth.py` itself rather than in the output:

- **`fig1_accumulation_within_scene.png`** -- the headline slice (m=5, depth 5), both deflation
  systems, +-1 SEM error bars, terminal step dashed.
  ```
  python scripts/plot_phase2_depth.py \
      results/phase2/dod/numbers_csv/phase2_librimix_5spk_scratch345.csv \
      --x-axis n_accepted_before --depth 5 \
      --out results/phase2/dod/graphs/fig1_accumulation_within_scene.png
  ```
  *Defect fixed:* the original plotted `gated_deflation`'s level-4 point (**n=3**, SEM 2.68) as a
  dramatic plunge to -8.89 -- the figure's most visually striking feature was its least reliable
  point. Points below `--min-n` (default 25) are now drawn hollow, annotated with their `n`,
  excluded from the trend line, and listed on stdout.
- **`fig2_accumulation_across_scenes.png`** -- paired SI-SDR vs `no_recursion` at fixed depth 2,
  swept over m, all three test systems.
  ```
  python scripts/plot_phase2_depth.py \
      results/phase2/dod/numbers_csv/phase2_librimix_{3,4,5}spk_scratch345.csv \
      --x-axis m --depth 2 \
      --out results/phase2/dod/graphs/fig2_accumulation_across_scenes.png
  ```
  *Defect fixed:* the original m-sweep was licensed by `no_recursion` being flat at depth 2
  (+0.16 dB on the warm-started checkpoint). On this checkpoint the control slopes **-0.95 dB**,
  nearly as steeply as two of the three test systems, so the original figure was four near-parallel
  declining lines -- visually "everything degrades with m," exactly the depth/accumulation
  conflation the axis correction existed to eliminate. Replacing mean-subtraction with **paired
  differencing per (scene, speaker, depth)** removes between-corpus difficulty *exactly*, putting
  the control at 0 by construction -- and this is now the **default** for `--x-axis m`, so it
  cannot be forgotten. The figure also shows the ordering as vertical separation at every m, so it
  carries both claims:

  | system (paired vs control, depth 2) | m=3 | m=4 | m=5 |
  |---|---|---|---|
  | `ungated_deflation` | -1.85 +-0.17 | -2.62 +-0.23 | -2.59 +-0.26 |
  | `gated_deflation` | -1.16 +-0.14 | -1.29 +-0.16 | -1.31 +-0.18 |
  | `coarse_to_fine` | -0.42 +-0.11 | -0.51 +-0.11 | -0.54 +-0.12 |

  Read: ungated's penalty deepens 3->4 (~2.7 sigma) then saturates; gated's and coarse_to_fine's
  are flat. `coarse_to_fine`'s constant ~-0.5 dB is the refinement deficit, not accumulation.

#### Reporting-code hardening (2026-08-09)

Every figure defect this project shipped had the same shape: **the `.md` tables were right and the
figure was wrong.** The tables already carried `n`, spread, diagnostic counts and the ordering
check; the figure carried a bare mean line. And the one protection that did exist
(`plot_phase2_depth.py --band`) was an opt-in flag nobody remembered to pass. So the fix is not
more flags -- it is **safe defaults, with flags that change thresholds rather than switch
correctness off.** Five changes, all CPU-only, full suite green (292 passed):

1. **One aggregation layer: `dagger/metrics/phase2_scores.py`.** `SI_SDR_CAP_DB` and the
   nan-drop / `+-inf`-clip rule used to be defined three times -- once each in `run_phase2.py`,
   `aggregate_phase2.py`, `plot_phase2_depth.py` -- each with a comment claiming it matched the
   others. That duplication already cost us: the 2026-07-26 `+-inf` bug had to be fixed twice. All
   three now import `load_score_rows` / `clip_score` / `mean_sem` / `group_values` /
   `paired_differences` / `control_slope` / `terminal_x_values`. Verified behaviour-preserving:
   `aggregate_phase2.py` reproduces `phase2_accumulation.md` byte-identically.
2. **Plot defaults flipped.** `--band {sem,p5p95,none}` now defaults to **`sem`**, so error bars
   are on unless switched off, and the chosen band is named in the figure footnote -- SEM and a
   p5-p95 band differ by ~30x here and a reader assumes the smaller one is the error bar.
   `--min-n` defaults to **25**: thinner points are drawn hollow, annotated with their `n`, kept
   off the trend line, and listed on stdout.
3. **The stale precondition is now computed, not remembered.** `control_slope()` checks whether
   `no_recursion` is flat on the `m` axis; over `CONTROL_FLATNESS_TOLERANCE_DB` (0.3 dB) the script
   prints a loud WARNING naming the drift and pointing at paired mode. Better still, `--mode`
   defaults to `paired-vs-control` for `--x-axis m`, which removes the precondition entirely.
   (Confirmed live: forcing `--mode raw` on the `scratch345` CSVs prints the -0.95 dB warning.)
4. **The terminal one-and-rest step is dashed automatically**, detected as `x == m-1`, and only
   when *every* contributing row at that position is terminal -- so a bucket mixing m=4 and m=5
   scenes stays an ordinary point rather than being dashed on a guess.
5. **`tests/phase2/test_reporting_guards.py`** (16 tests) encodes all three shipped defects as
   fixtures: a `+-inf` row that must clip rather than vanish, an n=3 cell beside an n=150 one that
   must stay distinguishable, and a control sloping -0.95 dB that must be detected. Each would have
   gone green before its fix. **This is the actual insurance** -- all three defects survived real
   runs because *nothing failed* when they were present.

*Still open (small):* `run_phase2.py` and `aggregate_phase2.py` build their `.md` tables with their
own local mean helpers rather than the shared `mean_sem`. Harmless today -- the numbers agree --
but it is the same duplication shape one level down.

---

**PHASE 2 DoD MET (2026-08-10). The grad-clip re-run replicated the accumulation curve to within
0.08 dB per level, so the ~3x shrinkage is a property of the training recipe, NOT an optimizer
artifact. Ordering now holds 18 of 18 across two independent scratch checkpoints. Claim the
ordering; report the magnitude as a range with the checkpoint named.**

#### The two DoD runs in one paragraph (read this first)

Phase 2 needed a checkpoint whose numbers came from **one command** (§7). The first such run
(2026-08-09) delivered that, and its ordering result was clean -- but its accumulation magnitude
came in at 1.83 dB against the 5.30 dB measured earlier on the warm-started checkpoint, a 3x
discrepancy. Reading its log turned up a cause: `grad_clip: 10.0` had quietly become a hidden
learning-rate schedule (93% of steps clipped by the end), so the run's apparent convergence could
not be distinguished from throttling. That made the 1.83 dB unreportable -- not wrong, but with an
unexamined optimizer defect sitting directly upstream of it. The second run (2026-08-10) changed
that one setting to 50.0 and nothing else. Clipping vanished (0-2%), the throttling signature
vanished with it -- **and the accumulation curve came back essentially identical, 1.81 dB, matching
to within 0.08 dB at every level.** So the defect was real but was never what produced the small
magnitude. With the confound eliminated and the number replicated on a second independent
checkpoint, both DoD criteria are settled on evidence rather than on a single run, and the phase
closes.

#### What "training recipe" means here

Three levels, and the distinction is the whole conclusion above, so it is worth being exact:

- **run** -- one execution of one config. Differs from another run of the same config only by
  random init and nondeterminism (training here is unseeded).
- **checkpoint** -- the weights a run produces. One per run.
- **recipe** -- the config *plus its initialization history*: trained from scratch vs. warm-started
  off a chain of earlier checkpoints, which depths the data mixes, how many scenes and epochs, the
  lr/clip schedule.

The measured accumulation magnitude is **stable across runs of the same recipe** (1.83 vs 1.81 dB
across two scratch-trained curriculum runs, despite different init, different clipping regimes and
a 0.34 gap in final training loss) and **differs between recipes** (~1.8 dB scratch vs 5.30 dB for
the warm-started chain, same curriculum config otherwise). That is why the honest claim is "the
magnitude is a property of the recipe," and why the paper names the checkpoint beside the number
rather than quoting a single figure as a property of the method.

#### Why a second DoD run existed at all

The 2026-08-09 run's provenance was clean but its `grad_clip: 10.0` had saturated into a hidden LR
schedule (93% of steps clipped by epoch 10, median gradient norm 1.5x the threshold -- see the
block above). That left the 1.83 dB accumulation figure with an unexamined optimizer defect sitting
directly upstream of it. Reporting a 3x discrepancy against the warm-started checkpoint's 5.30 dB
while a known throttling bug went unaddressed was the one outcome that was not publishable, so the
run was repeated with **exactly one variable changed**: `grad_clip: 10.0 -> 50.0`. `epochs` stayed
at 10 deliberately -- changing both would have made "the effect returned" unattributable between
the clip and the extra training, and those two readings imply different claims.

Artifacts kept distinct on purpose so the comparison stays checkable:
`checkpoints/phase2/proposed_librimix_curriculum_3_4_5_scratch_clip50.pt`, `eval.tag:
scratch345clip50`, both tags side by side in `results/phase2/dod_final/`.

#### The clip fix worked on both its stated goals

| | clip 10 (2026-08-09) | clip 50 (2026-08-10) |
|---|---|---|
| clipped % by epoch | 4 -> 93 | 0 -> 0 (peak 2% at epoch 9) |
| median grad norm, final | 15.14 (1.5x threshold) | 11.94 (0.24x threshold) |
| final-epoch loss improvement | **-0.07** (5x below its own run average) | **-0.37** (on its average) |
| final loss | 1.67 | 2.01 |

The old run's terminal deceleration -- what made "converged" and "throttled" indistinguishable --
is gone; the new run was still descending steeply when it hit the 10-epoch cutoff. So the
2026-08-09 log's flattening WAS an artifact. That is worth keeping even though it changed nothing
downstream: it was not knowable before this run.

The higher final loss is not a regression. Epoch 1 was already 4.88 vs 4.69 at 0% vs 4% clipping,
i.e. before the clip could matter -- training is unseeded, and Phase 1 saw identical configs score
2.26 and 1.03 dB on consecutive days. Also, Adam plus 93%-clipping is a *different* optimizer, not
a slowed one: clipping the global norm to a constant makes the pre-Adam gradient direction-only at
fixed magnitude. "The old run reached a lower loss" therefore does not mean clipping helped.

#### The result: replication, not restoration

`ungated_deflation`, `n_accepted_before`, m=5 at depth 5, n=150 per level (balanced by construction):

| level | 0 | 1 | 2 | 3 | 4 | total |
|---|---|---|---|---|---|---|
| clip 10 | -4.89 | -5.93 | -6.35 | -6.98 | -6.72 | **-1.83 dB** |
| clip 50 | -4.97 | -5.89 | -6.31 | -7.00 | -6.78 | **-1.81 dB** |
| delta | -0.08 | +0.04 | +0.04 | -0.02 | -0.06 | +0.02 |

Two checkpoints, different random init, 93% vs ~1% clipping, training loss 1.67 vs 2.01 -- and the
curve reproduces to 0.08 dB at every level.

*Not the same checkpoint evaluated twice.* The Phase 2 eval path has no RNG anywhere (confirmed
2026-07-29 by reproducing a table to the decimal), so an identical checkpoint gives *identical*
numbers, not numbers differing by 0.04-0.08. And the agreement is this tight -- rather than the
~0.45 dB two independent draws at SEM 0.32 would give -- because both evals run the identical 150
test scenes with identical placement, enrollment and deflation order, so scene-level variance
cancels exactly and only the model difference survives.

*The 0.08 dB figure is about the PRIMARY quantity only -- do not generalise it.* The secondary
one (fig2: paired vs. `no_recursion` at fixed depth 2) reproduces its shape and ordering but not
its magnitude, coming in consistently shallower on the clip-50 checkpoint:

| depth 2, paired vs control | m=3 | m=4 | m=5 |
|---|---|---|---|
| `ungated_deflation` clip 10 -> clip 50 | -1.85 -> -1.23 | -2.62 -> -2.14 | -2.59 -> -2.26 |
| `gated_deflation` | -1.16 -> -0.80 | -1.29 -> -1.04 | -1.31 -> -1.06 |
| `coarse_to_fine` | -0.42 -> -0.07 | -0.51 -> -0.28 | -0.54 -> -0.41 |

Expected, and the reason is structural rather than noise. The primary quantity contrasts *one
system against itself* at different accumulation levels, so whatever the checkpoint is like
largely cancels. The secondary contrasts *two different systems*, so it measures how much a
corrupted residual costs -- which is exactly Theorem 3's extractor sensitivity `L`, a property of
the checkpoint. Two checkpoints having different `L` is what the theory predicts. What replicates
across both is the shape (ungated deepens 3->4 then saturates; gated and coarse_to_fine flat) and
the vertical ordering at every `m`.

**What it settles:**

1. **The grad-clip throttling was not the cause of the shrinkage.** Hypothesis dead. Worth killing
   -- the alternative was shipping the 3x discrepancy with a known defect upstream of it.
2. **The accumulation magnitude is insensitive to training level across this range.** The clip-50
   checkpoint is *less* converged, and this project's own twice-logged pattern (2026-07-29,
   2026-07-31: more training -> higher floor -> NARROWER gap) predicted a wider gap. It did not
   appear.
3. **The magnitude is a property of the training recipe, not of any run.** ~1.8 dB for
   scratch-trained curriculum checkpoints (now replicated); 5.30 dB for the warm-started chain. The
   difference is training *history*, and note it cuts AGAINST the "more training narrows the gap"
   pattern, since the warm-started chain had far more cumulative training and showed the wider gap.
   State this as an observation; we have no mechanism for it.

Headline the scratch number: it has clean single-command provenance *and* it is replicated. Report
5.30 dB as the observed warm-started case with the checkpoint named.

#### Final DoD scorecard

- **Ordering: 18 of 18.** `coarse_to_fine > gated_deflation > ungated_deflation` at every depth 2-5
  across all three eval sets, on *both* scratch checkpoints. clip-50 depth-2 margins, tightest
  first: 5spk 0.10 / -0.55 / -1.75; 4spk 0.83 / 0.08 / -1.02; 3spk 1.19 / 0.46 / 0.03.
- **Accumulation: monotone through the body of the chain, replicated** (table above).
- **The terminal one-and-rest step replicated a third time** (+0.21 dB here, +0.26 before). Its
  residual is `x_O` minus every other estimate, which already approximates that speaker's own
  source, so `G` has an unusually easy job. A benign special case of Theorem 2's `||E_m|| <= m*eps`
  upper bound -- accumulation is monotone through the body and flattens at the endpoint. Both
  figures draw it dashed.
- **Refinement is net-harmful under clean enrollment: negative at all 9 slices** (-0.07 to -0.41 dB;
  the clip-10 run gave -0.15 to -0.54), with the gate healthy -- accept rate 46.2 / 34.0 / 26.8% at
  m=3/4/5, tracking difficulty downward, no degenerate tie-shutdown. `refine.rounds: 0` stays the
  default.
- **`V_i` is structurally dead, again.** Zero variance rejections in 5,400 gate decisions;
  `mean_variance` is *exactly* 0.0 in 77% of them and never exceeds 4.23e-4 against a 0.05
  threshold. Rejections are margin (3,138) and artifact_score (45) only. Phase 3 item -- it cannot
  be tuned until real diarization supplies multiple enrollment segments of differing quality.
  > **PARTLY CORRECTED (2026-08-20):** structurally 0 under *oracle* regions, yes -- but not dead.
  > At a 1e-4 threshold on real diarization it scores J = +0.373; `0.05` is 500x too high. And the
  > 3,138 `margin` rejections counted here are now suspect from the other direction: the margin
  > scored J = +0.046 ("no usable threshold") on the dev sweep, and the conditioning probe has
  > since CONFIRMED that verdict -- clip50 steers at 9.41 dB, so the fault fixture was valid and
  > the margin genuinely fails to separate wrong-speaker output. **Treat these 3,138 rejections as
  > close to noise**, and with them the Phase 2 gated-vs-ungated comparison: `gated_deflation`
  > differs from `ungated_deflation` only where the margin fires, and the margin is not detecting
  > what it is supposed to detect. See Stage B Session B Q3 and its follow-up.

#### Absolute quality: a training-budget fact, and a Phase 4 input

`no_recursion` at each set's deepest depth: 3spk -1.29, 4spk -3.47, 5spk -4.87 (the clip-10 run:
-1.19 / -3.35 / -4.76). Negative at depth 4-5 for *every* system including the one with no
deflation logic at all, so this is the extractor's operating point, not the reconstruction
strategy. Note also that a 0.34 training-loss difference moved eval quality by only ~0.1 dB.

The cause is budget, and it is arithmetic -- steps scale as scenes x epochs / batch:

| | scenes | epochs | steps | at n_src=3 |
|---|---|---|---|---|
| Phase 1 DoD (4.40 dB) | 2000 | 30 | 15,000 | 15,000 |
| this curriculum run | 2400 (3x800) | 10 | 6,000 | **2,000** |

40% of Phase 1's total steps, spread over three depths, so the 3-speaker case saw roughly 13% of
what the Phase 1 checkpoint saw (metric slices are not identical -- directional, not like-for-like).
And the lever does not exist within one session: wall time scales as scenes x epochs, so total
steps scale as time. Rebalancing buys nothing (400/depth x 25 epochs = 7,500 steps in ~7.9h), and
batch 8 OOMs a 15 GB T4. Materially better absolute quality costs more *sessions* -- roughly 19h
(two, warm-started) to reach a plateau, ~3x Phase 1's cost to match its per-depth exposure.

**This does not weaken Phase 2's claims, which are all relative** -- every comparison holds scenes,
mixture and checkpoint fixed and varies only the reconstruction strategy. The accumulation result
now survives three corpora spanning ~5 dB of absolute level (`no_recursion` at the deepest depth:
-4.03, -0.23, +0.75 on the warm-started checkpoint) *and* two checkpoints at different training
losses agreeing to 0.08 dB. A result that survives a 5 dB shift in the operating point is not an
artifact of the operating point. Carry the budget arithmetic into Phase 4, where absolute numbers
are the deliverable.

#### Reproduce

```
python scripts/train_phase1.py --config configs/phase2/dod/phase2_librimix_curriculum_3_4_5_train_scratch.yaml --system proposed
python scripts/run_phase2.py  --config configs/phase2/dod/phase2_librimix_{3,4,5}spk_eval_scratch.yaml   # one at a time
python scripts/aggregate_phase2.py results/.../phase2_librimix_{3,4,5}spk_scratch345clip50.csv --out .../phase2_accumulation_scratch345clip50.md
python scripts/plot_phase2_depth.py .../phase2_librimix_5spk_scratch345clip50.csv --x-axis n_accepted_before --depth 5 --out .../fig1_...png
python scripts/plot_phase2_depth.py .../phase2_librimix_{3,4,5}spk_scratch345clip50.csv --x-axis m --depth 2 --out .../fig2_...png
```

Results live in `results/phase2/dod_final/` (`numbers_csv/`, `numbers_md_docs/`, `graphs/`), both
tags side by side. All six clip-50 CSVs are present, and the aggregation + both figures were
**re-derived locally from the committed scripts** on 2026-08-10, off the GPU box entirely:
`aggregate_phase2.py` reproduced the Kaggle-produced `phase2_accumulation_scratch345clip50.md`
byte-identically (modulo a trailing newline). So §7's "one command regenerates this" holds for the
whole evidence set, not just the checkpoint.

### ☐ Phase 3 — Real diarization + robustness

**Goal:** survive imperfect, real diarization.

**Build:** swap oracle path for **pyannote 4.0 community-1**; add mask augmentation during
training (Gaussian noise on activity masks, segment flipping, synthetic overlap injection);
turn on the **`V_i` enrollment-rejection** to catch solo regions that were secretly overlaps.

**How to check it worked:** run the oracle-vs-real ablation — the gap between them tells you how
much diarization error costs. `V_i` rejection should catch contaminated enrollments (test by
deliberately feeding a known-overlap clip as "solo").

**Red flags:** big unexplained quality drop with no oracle-vs-real breakdown (you can't attribute
it); `V_i` never firing (threshold too loose) or firing on everything (too tight).

**Definition of done:** end-to-end results with real diarization, plus the oracle-vs-real gap table.

**Split into two stages (2026-08-14).** Stage A needs **no GPU training**: build the diarizer
seam, the DER metric and the paired oracle-vs-real harness, and measure the gap on the existing
Phase 2 checkpoint. Stage B then designs mask augmentation against the *measured* error profile.
The ordering is deliberate — the first five Phase 2 runs were spent sizing a remedy before the
disease was measured, and this phase does not repeat that.

**STAGE A CODE LANDED (2026-08-14): implemented, unit-tested offline, NOT yet run against real
data or pyannote.** Full suite 350 passed / 1 skipped (292 baseline + 58 new; the skip is a
`pyannote.metrics` cross-check that needs the `[diarize]` extra). Nothing here has seen a real
diarizer yet — treat every claim below as "the code does this", not "the data says this".

*What was built.* A `Diarizer` ABC (`dagger/diarize/base.py`) copying the one abstraction in this
repo that already survived a backend swap (`SpeakerEncoder`), with `OracleDiarizer` and
`PyannoteDiarizer` (`dagger/diarize/pyannote_diarizer.py`, imports deferred) behind it;
`dagger/diarize/regions.py` collapsing the 6-times-duplicated
`activity_matrix -> solo_overlap_regions -> overlap_depth` dance; `dagger/diarize/mapping.py`
(Hungarian cluster->speaker matching); `dagger/metrics/der.py`; `scripts/run_phase3.py` +
`scripts/aggregate_phase3.py`; `configs/phase3/{dod,experiments}/`; a new `[diarize]` extra
(NOT folded into `[ml]` — CI installs `dev,data` only and pyannote pulls a gated checkpoint).
The six existing `activity_matrix` call sites are **untouched**, so no committed number can shift.

*The one structural refactor.* `score_scene`/`_make_gate_fn`/`_accepted_before` moved from
`scripts/run_phase2.py` into **`dagger/eval/systems.py`**, which `run_phase2.py` re-exports so
every module-level name stays importable at the old path. Phase 3 runs the same four systems
twice per scene, and a second copy is exactly how the 2026-07-26 `+-inf` bug came to need fixing
twice. **Verified equivalent**: the two helpers are AST-identical to their originals, and the
pre-refactor `score_scene` (pulled from git) produces **identical** score rows and gate rows to
the new one on synthetic scenes. The corpus-based byte-identical CSV check still has to run on
the GPU box.

*Four arms, each varying one thing* (`scripts/run_phase3.py`), all scoring the same scenes in the
same pass so rows pair exactly on `(scene, speaker, depth, system)`: `oracle` (mandatory, §6.2 —
the script exits if it is absent), `real` (free speaker-count estimation, the honest condition),
`real_forced_m` (`num_speakers=m`, isolating segmentation from counting error), and
`real_index_order` — see below.

*`real_index_order` exists because of a BIAS, not tidiness.* Deflation order is ascending `V_i`
(`run_phase2.py:189`, now `dagger/eval/systems.py::deflation_order`). Under oracle diarization
`V_i` is identically 0 and Python's sort is stable, so **that sort has never actually done
anything**. Real diarization fragments solo regions, `V_i` becomes real, and the sort becomes a
real permutation — and position in it *is* `n_accepted_before`, the dominant driver of SI-SDR for
the deflation systems (~1.8 dB across levels). It is not symmetric: whichever speaker's solo
region best survives diarization earns the lowest `V_i` and is promoted to level 0, the
least-damaged slot — and that speaker's extraction was already the easiest. So the real arm is
*flattered* and the gap is biased **downwards**. Read the decomposition as
`real_index_order - oracle` = diarization error alone, `real - real_index_order` = the reordering
`V_i` induces. `no_recursion` and `coarse_to_fine` take no deflation order, so they must be
**identical** across those two arms — asserted as a free check on the arm machinery.

*Sample rate.* `dataset.diarizer_sample_rate: 16000` makes the loader build a second, NATIVE
16 kHz mixture from the same placement (`Scene.mixture_hi`). Upsampling the 8 kHz mixture instead
invents nothing above 4 kHz, so DER would be inflated for a reason unrelated to diarization
difficulty — and that inflation would land straight in the headline gap. The 8 kHz scene is
bit-identical with the key absent (pinned by test). Restricted to integer rate ratios so
placements scale exactly.

*Three bugs found while building, each of which would have produced a plausible-looking wrong
number with nothing failing — the signature of every reporting defect this project has shipped:*

1. **The gap table would have been EMPTY.** Score rows keyed `speaker` on the cluster id, so real
   rows read `SPEAKER_00` and oracle rows `s1`; every paired lookup missed. Rows now carry the
   *attributed ground-truth* label, with `cluster` as separate Phase 3-only provenance
   (`run_phase2.py`'s writer uses `extrasaction="ignore"`, so its schema and bytes are untouched).
2. **A spurious cluster killed the whole scene.** An invented cluster usually lies inside another
   speaker's speech, so it has no solo region and `enroll_speaker` raises. Skipping the scene
   would discard scenes *in proportion to how badly the diarizer did* — the same selection bias
   that silently shrank Phase 1's dataset. `score_scene` gained `on_unenrollable`, defaulting to
   `"raise"` (Phase 2 behaviour untouched); Phase 3 passes `"drop"` for every arm.
3. **A silent speaker lost its row.** The scene builders skip zero-length chunks, so with
   discovered labels the oracle arm would drop a row and shift every downstream index. The oracle
   now binds its label set (`Diarizer.binds_scene_speakers`); only the real path discovers. Also
   corrected: oracle row order matches `scene.speakers` because both segment builders iterate the
   speaker *list* (`activity.py:145,177`), NOT because of solo-slot timing.

*DER is overlap-aware on purpose*: it scores per-frame speaker **counts**, not one label per
frame. A single-label-per-frame DER scores every overlap frame as at best partially right by
construction, which for a project about overlap would mostly measure its own simplification.
`overlap_recall` is broken out because overlap is the only region where `G` runs at all (solo is
copied through), so it should track the SI-SDR gap far better than aggregate DER.

*`V_i` should come alive here for the first time.* It has been **exactly 0.0** in every run this
project has ever done, structurally: `schedule_solo_then_overlap` gives one solo run -> one clip
-> variance over one sample. A real diarizer fragments solo regions, enrollment draws k>1 clips,
and the variance becomes real. **The single load-bearing check on the smoke run is a nonzero
`mean_variance` in `_gate.csv`.** Note this also makes the deflation-order sort live, which is
what arm 4 exists to separate.

*Not yet done / known gaps.* Nothing has run against pyannote or real data. The clip50 checkpoint
the DoD config points at is ~~**not on the HF Hub** (only the Phase 1 and Phase 2-finetuned ones
are), so a fresh Kaggle session must attach it as a dataset or re-upload it.~~ *(STALE. It has
since been published as `AdityaAA2004/dagger-phase2-final-model`, Hub filename
`phase2_final_model_weights.pt`; see README.md's checkpoint table. `checkpoints/` is gitignored, so
every checkpoint must be fetched via `hf_hub_download` and copied to the path the configs expect --
nothing arrives with a clone.)* ~~The dev split at
`offset: 650` (`configs/phase2/experiments/phase2_gate_tune_dev.yaml`) **overlaps** rows
[650, 800) of the clip50 training run, which used `limit: 800` — move it to `offset: 1000` before
any Stage B gate tuning, or thresholds get tuned on scenes `G` trained on.~~ *(Fixed 2026-08-18 in
Stage B Session A: now `offset: 1000`. The config's own header had claimed `limit: 650`, which is
what hid the overlap.)* Two limitations stay
recorded rather than fixed, because Phase 3 cannot vary them: LibriMix's synthetic geometry (hard
boundaries, no reverb, no turn-taking) makes this gap a **lower bound** on the real-corpora gap
Phase 4 will see; and the resampler fallback warning exists so a future corpus without
`mixture_hi` cannot silently reintroduce band-limited diarizer input.

**FIRST KAGGLE ATTEMPT (2026-08-15): the A5 refactor guard PASSED on real data; three
environment/API defects found and fixed; no Phase 3 numbers yet.**

1. **A5 guard passed — byte-identical.** `run_phase2.py` against
   `configs/phase2/dod/phase2_librimix_3spk_eval_scratch.yaml` with the clip50 checkpoint
   reproduced the committed `phase2_librimix_3spk_scratch345clip50.csv` **byte for byte**, so
   moving `score_scene` into `dagger/eval/systems.py` is now verified against the corpus, not
   just synthetically. This box is checked; do not spend another ~29 min on it unless
   `dagger/eval/` or `run_phase2.py` changes.
2. **The model id in the configs was wrong.** `pyannote/community-1` 404s; the real repo is
   **`pyannote/speaker-diarization-community-1`**. It fooled a pre-flight that constructed
   `PyannoteDiarizer()` with no argument (picking up the corrected `DEFAULT_MODEL`) while the
   run read `diarizer.model` from the config — so the check passed and the run failed.
   Both `configs/phase3/**` files now carry the full id.
3. **pyannote 4.x returns `DiarizeOutput`, not `Annotation`** — `output.itertracks(...)` raises
   `AttributeError`. `dagger/diarize/pyannote_diarizer.py::_to_segments` now unwraps
   defensively (wrapper or bare annotation; 2- or 3-tuples). **It reads
   `.speaker_diarization` and must NEVER read `.exclusive_speaker_diarization`**: the exclusive
   variant allows at most one speaker per instant, so taking it would silently delete every
   overlap — the overlap mask would collapse, `G` would barely run, `overlap_recall` would read
   ~0, and the run would produce a complete, wrong results table rather than an error. Pinned by
   `tests/phase3/test_pyannote_output.py` (357 passed).
4. **Kaggle in-kernel numpy breaks after the pip installs.** `pip install -U numba numba-cuda`
   then `.[diarize]` leaves numpy mixed on disk, so a lazily-imported submodule (`numpy.char`
   via `scipy.signal`) mismatches the numpy already in memory from `import torch`. Subprocesses
   are fine — the 25-min `run_phase2.py` eval ran through the same import. **Notebook rule: do
   Phase 3 work via `!python`, never in-kernel.** Kaggle batch cannot restart the kernel.

**STAGE A RESULT (2026-08-18): the oracle-vs-real gap is measured. Scene LENGTH was the cause of
every earlier collapse -- not geometry, not the diarizer.** `configs/phase3/experiments/phase3_librimix_3spk_long.yaml`,
50 two-minute scenes, chain placement at `overlap: 0.3`, checkpoint
`checkpoints/phase2/proposed_librimix_curriculum_3_4_5_scratch_clip50.pt` (unchanged, no retraining).

#### Why four runs were needed to get one number

pyannote returned ~2 clusters for 3-speaker scenes in every SHORT geometry, and the cluster count
did not move with geometry at all -- only the drop rate did:

| corpus | scene | clusters | enrolled | degenerate (m=1) | DER |
|---|---|---|---|---|---|
| scheduled, 1 s solo (2026-08-16) | ~10 s | ~2.0 | 1.00 | 149/150 | 0.309 |
| chain (2026-08-16) | ~20 s | 2.04 | 1.57 | 70/150 | 0.309 |
| scheduled, 3 s solo (2026-08-16) | ~19 s | 2.07 | 1.80 | 42/150 | 0.332 |
| **chain, 2 min (2026-08-18)** | **~120 s** | **3.00** | **3.00** | **0/50** | **0.113** |

The first three are NOT usable measurements: with `m=1` there is nothing to deflate, gate or
refine, so all four systems collapse onto one another and every "gap" is really the cost of losing
a speaker outright. Standard diarization benchmarks run minutes (CALLHOME 2-5 min, AMI tens of
minutes); at 10-20 s we were an order of magnitude below the operating point pyannote is built for.
`scripts/build_long_scene_metadata.py` concatenates consecutive same-speaker utterances
(`pathA|pathB|...`, a shape the loader already supported) to reach it.

**`overlap: 0.3`, never 0.5.** Chain places the middle speaker between the other two, and at 0.5
its solo window closes *exactly* -- measured 32 s / 1 s / 33 s of solo on a 129 s scene. That is the
same starvation, so a long-scene run at 0.5 reproduces the failure it was built to escape. At 0.3:
45 s / 26 s / 45 s, 25% of the scene overlapped. `build_long_scene_metadata.py` refuses `>= 0.5`
for 3+ speakers.

#### The numbers (50 scenes, 4800 paired rows, all four arms at clusters = enrolled = 3.00)

Absolute SI-SDR, and the paired gap (`real - oracle`, matched on scene/speaker/depth/system):

| system | oracle d1 | oracle d2 | real d1 | real d2 | gap d1 | gap d2 |
|---|---|---|---|---|---|---|
| no_recursion | 47.62 | 1.73 | 38.47 | -1.38 | -9.16 +-1.04 | **-3.11 +-0.21** |
| ungated_deflation | 46.71 | 0.24 | 38.47 | -1.96 | -8.25 +-1.07 | -2.20 +-0.23 |
| gated_deflation | 46.71 | 0.24 | 38.47 | -1.95 | -8.25 +-1.07 | -2.20 +-0.23 |
| coarse_to_fine | 47.70 | 1.04 | 38.47 | -1.87 | -9.23 +-1.04 | -2.92 +-0.23 |

Win rates 9-24% with `|t|` 7.7-14.8: the real arm loses *consistently*, not through a few
catastrophic scenes. That is the opposite of Phase 1's +2.35 dB at a 50% win rate, and it means
the mean is describing the typical row.

Diarization quality: **DER 0.113** = miss 0.105 + false alarm 0.000 + confusion 0.008,
`overlap_recall` 0.758, zero missed speakers, zero spurious clusters.

**Depth 3 does not exist in this corpus.** Chain at overlap 0.3 makes s1 and s3 disjoint, so the
table stops at depth 2. Getting long scenes AND depth 3 needs scheduled placement with a long solo
zone (e.g. `per-speaker 60 s` with `min_solo_ms: 15000` -> 45 s solo zone + ~45 s of genuine 3-way
overlap) -- a metadata + config change, no new code. Not run.

#### What the gap is made of: the ACTIVITY masks, not the deflation order

Settled directly from the arms, not inferred:

* `no_recursion` takes **no deflation order at all**, and its gap is the LARGEST at -3.11 dB.
* `real - real_index_order` is **exactly +0.00** for `no_recursion` and `coarse_to_fine`.

So the loss is entirely attributable to the predicted masks. The mechanism is in the DER
decomposition: `overlap_recall` 0.758 means ~24% of true overlap frames are called non-overlap, and
in those frames the pipeline takes the **solo-copy path** -- copying the mixture verbatim instead of
extracting, so every other speaker leaks in at full level. Miss dominates (0.105) and confusion has
nearly vanished (0.008), which is the inverse of the short-scene profile (confusion 0.171-0.187).

**The `V_i` reordering confound is ~11% of the effect, not negligible.** `real - real_index_order`
= +0.25 dB (|t| 2.0) for both deflation systems at depth 2, against a -2.46 dB gap. Arm 4 earned
its place; without it that 0.25 dB would sit inside the headline number unattributed. Note the
sign: variance-ordering *flatters* the real arm, exactly as predicted when the arm was designed.

#### Three results that are clean, and three cautions on reading the tables

*Clean:*

1. **`real_forced_m` is now IDENTICAL to `real`** -- same DER to 3 decimals, same SI-SDR to 2, +0.00
   on all 300 paired rows. At 2 minutes pyannote picks 3 unaided, so forcing it changes nothing.
   Not a caching artifact: the two arms use different cache keys and run the pipeline separately,
   and `real_index_order` *does* differ, so the machinery can produce differences. **The
   speaker-counting problem is gone, and this arm can be dropped from future long-scene runs**
   (~25% of runtime measuring a constant).
2. **Refinement is net-harmful under CONTAMINATED enrollment too** -- the one regime Phase 2 never
   tested, and the reason `refine.rounds: 2` was kept on here. oracle depth 2: 1.73 -> 1.04
   (-0.69 dB); real depth 2: -1.38 -> -1.87 (-0.49 dB). It costs *less* when diarization is
   imperfect (paired gap -2.92 vs -3.11, i.e. 0.19 dB less degradation), so contamination does move
   it in the predicted direction -- nowhere near enough to turn positive. `refine.rounds: 0` stays
   the default.
3. **Ordering holds at depth 2 in all four arms** -- but see caution 2 before quoting it.

*Cautions:*

1. **The gate is degenerate at 98-99% accept**, so `gated_deflation` and `ungated_deflation` are
   the same system (oracle depth 2: 0.24 vs 0.24). The gated/ungated half of the ordering claim is
   vacuous in this run. Same pattern as Phase 2's heterogeneous corpus (98.4% accept).
2. Rejections are **margin only** (3 for gated, 88 for coarse_to_fine); vad, artifact and variance
   are 0 across all 1800 decisions.
3. ~~**The `real - oracle` accumulation table is a biased subset.**~~ **FIXED 2026-08-18 (Stage B
   A1) -- both accumulation tables are now usable; the numbers below supersede the originals.**
   `scripts/aggregate_phase3.py`'s `_table` filtered on `n_accepted_before` *before* pairing, so a
   speaker counted only if its accumulation position matched across arms -- and under `real` the
   `V_i` sort reorders, so it usually did not (n = 34/28/16, against 98/92/92 for
   `real_index_order`, whose order matches oracle by construction). The discarded rows were
   discarded *because of* the effect being measured, so the survivors were biased, not merely few.
   `_table` now pairs first and buckets each pair by the **reference arm's** position, which is
   also the only reading that means anything: "for a speaker the oracle placed at level k, what did
   real diarization cost it?"

   | `real - oracle`, ungated_deflation | level 0 | level 1 | level 2 |
   |---|---|---|---|
   | before (biased) | n=34, -5.51 ±1.30 | n=28, -3.83 ±2.12 | n=16, -6.36 ±2.50 |
   | after (paired) | **n=100, -7.05 ±0.98** | **n=100, -2.84 ±0.89** | **n=100, -5.79 ±1.06** |

   Levels are now balanced at n≈100 by construction, and SEM roughly halves at every level, so the
   table gained precision as well as correctness. **The independent check that the fix is right
   rather than merely different:** `real - oracle` and `real_index_order - oracle` should nearly
   coincide, since the reordering `V_i` induces is only ~0.25 dB (see above). Before the fix they
   disagreed by 1-2 dB at every level; after it they agree to ~0.2 dB
   (-7.05/-2.84/-5.79 vs -6.94/-3.11/-6.01), with the already-correctly-paired arm barely moving
   (98/92/92 -> 100/100/100). The depth table is **unchanged**, as it must be: depth is derived
   from the reference activity for both arms, so it always paired correctly.
   Regression-tested in `tests/phase3/test_accumulation_stratification.py`.

#### ~~`V_i` is dead -- fourth confirmation~~ **WRONG. See Stage B Session B Q3 (2026-08-20).**

> **CORRECTED.** The measurements below are right; the conclusion drawn from them is not.
> `tune_gate.py` on a dev split later scored `V_i` at **Youden's J = +0.373** at a threshold of
> **1e-4**. The signal was always there -- `max_mean_variance: 0.05` sits **500x above the entire
> usable range**, so of course nothing ever fired. "It never crosses the threshold" was read as
> "the check cannot work" when it meant "the threshold is in the wrong place." Note the sentence
> below explicitly rules out the tuning explanation, which is exactly the inference that was wrong.

Nonzero in 1332/1800 decisions, max **0.000324** against a `max_mean_variance: 0.05` threshold
(~150x below), **zero rejections**. The oracle arm shows 0/45 nonzero in the smoke while real shows
42/45, so the mechanism works -- real diarization genuinely fragments enrollment into multiple
clips -- the magnitude simply never crosses anything. ~~This is no longer explainable as a tuning
problem~~ (it was exactly a tuning problem) or as a consequence of degenerate scenes: it now holds
with 3.00 clusters enrolled, 0 degenerate scenes and DER 0.113.

#### Absolute quality is the EXTRACTOR's operating point, not diarization's

The *oracle* arm reaches only **1.73 dB at depth 2**. Whatever is wrong there was already wrong
with perfect masks. Cause is the training budget documented in Phase 2's close-out: the curriculum
checkpoint got ~40% of Phase 1's total steps spread across three depths, so the 3-speaker case saw
roughly 13% of Phase 1's exposure. Phase 3 changed no training, so nothing here could have moved
it. Do NOT read -1.38 dB as "real diarization ruins quality" -- read it as -3.11 dB below an
already-low ceiling.

#### Reproduce

```
python scripts/build_long_scene_metadata.py \
    --librispeech-root $DAGGER_DATA_ROOT/LibriSpeech/test-clean \
    --output $DAGGER_DATA_ROOT/metadata/Libri3Mix/libri3mix_test_long.csv \
    --n-src 3 --num-scenes 50 --per-speaker-sec 50 --overlap 0.3
python scripts/run_phase3.py --config configs/phase3/experiments/phase3_librimix_3spk_long.yaml
python scripts/aggregate_phase3.py results/phase3/experiments/phase3_librimix_3spk_long2min.csv \
    --out results/phase3/experiments/phase3_gap_long2min.md
```

The four CSVs/`.md`s this produces are now **committed** under `results/phase3/experiments/`, so
§7's "one command regenerates this" finally holds for Phase 3 as it does for Phase 2. The
aggregation step above is CPU-only and was re-run against the committed CSVs on 2026-08-18 with the
fixed `_table` -- that is where the corrected accumulation numbers in caution 3 come from.

Note `run_phase3.py` was run with the pre-Stage-B config, so the committed CSVs have **no
`dilate_ms` column**. That is handled, not a gap: `load_score_rows` reads a missing column as
`0.0` (= undilated), which is exactly what that run was.

---

#### FOUR OUTSTANDING ITEMS (2026-08-18) -- what Stage A left open

> **Status after Stage B Session A (2026-08-18):** all four now have their *code* landed and
> unit-tested offline; none has been *run*, because every one of them needs TitaNet, the extractor
> and (for 1 and 3) pyannote on real audio. Item 4's script fix is the exception -- it is CPU-only,
> so it has been applied to the committed Stage A CSVs and its corrected numbers are in the
> cautions table above. See "STAGE B -- SESSION A LANDED" below for what exists now.

**1. The gate has never actually been tuned.** `scripts/tune_gate.py` and
`configs/phase2/experiments/phase2_gate_tune_dev.yaml` have existed since Phase 2 and have never
been run, because `V_i` was structurally 0 and there was nothing to sweep. Stage A produces the
first regime where it is nonzero, so the sweep is finally meaningful -- and the point of running it
is **to convert an inference into a measurement**: today's claim that `V_i` cannot work rests on
"max 3.2e-4 against a 0.05 threshold", whereas `tune_gate.py` reports Youden's J against a
deliberately contaminated-enrollment population and refuses to recommend below J = 0.1. A
documented J ~ 0 is a far stronger negative result than small-looking numbers.
*Two prerequisites, both now MET in code:* (a) a **dev split** -- generate long-scene metadata from
`dev-clean`, not `test-clean`; tuning on the 50 test scenes and then reporting them is leakage (§8)
*(done: `configs/phase3/experiments/phase3_gate_tune_dev_long.yaml`; the stale `offset: 650` in the
Phase 2 dev config is also fixed to 1000)*; (b) remember `gate_cfg` is SHARED between
`gated_deflation` and refinement, so raising
`tau_margin` slides gated toward `no_recursion` -- the degenerate direction. In this run only
`tau_margin` fires at all; vad, artifact and variance are inert across 1800 decisions.

**2. Absolute quality is poor and it is NOT a Phase 3 problem.** Oracle depth 2 is 1.73 dB, so the
ceiling is already low with perfect masks; this is the training-budget arithmetic in Phase 2's
close-out (~13% of Phase 1's per-depth exposure for the 3-speaker case). Fixing it means
retraining, which should be **bundled with Stage B's mask-augmentation run rather than spent
separately** -- one GPU session serves both. Two things to change in that run beyond augmentation:
train on **long** scenes (so train and eval geometry match), and restore **depth 3+**, which this
corpus lacks because chain at overlap 0.3 makes s1 and s3 disjoint. Scheduled placement with a long
solo zone gives both (e.g. per-speaker 60 s, `min_solo_ms: 15000` -> 45 s solo + ~45 s of true
3-way overlap); that is a metadata + config change, no new code.

**3. The low win rate is the ACTIVITY masks, and the mitigation is cheap.** Already settled from
the arms (see the Stage A result above): `no_recursion` takes no deflation order and carries the
largest gap, and `real - real_index_order` is exactly +0.00 for the two order-independent systems.
The mechanism is missed overlap -- `overlap_recall` 0.758 means ~24% of true overlap frames are
called solo, and there the pipeline **copies the mixture verbatim** instead of extracting.
*The untested mitigation:* the costs are **asymmetric**. Calling a solo region "overlap" is mild
(`G` runs where a copy would have done); calling an overlap region "solo" is catastrophic (a raw
mixture is emitted as a speaker's track). So the derived overlap mask should be biased toward
inclusion -- dilate predicted overlap by some milliseconds before building `x_O`. One config key,
one knob, sweepable on dev, **no retraining**. If missed overlap really is the dominant cost this
should recover a meaningful share of the 3.11 dB, and it is the cheapest open lever in the phase.

**4. "Refinement is net-negative" needs a CEILING, not more repetitions.**
*(Status 2026-08-20: the ceiling was built and run, but scored the wrong slice, so it is not yet
answered -- see Stage B Session B Q2. The reasoning below stands unchanged and is why the re-run
matters.)* It is now negative in
every regime tested -- clean enrollment (Phase 2, -0.07 to -0.54 dB), heterogeneous enrollment
(Phase 2, -0.36/-0.69), and contaminated real-diarization enrollment (Stage A, -0.49 dB). But
"never positive" is not provable by accumulating negatives, and the mechanism explains why the
tests keep landing negative: refinement blends `0.5*e_enrolled + 0.5*e_from_extracted_overlap`, and
the second term is embedded from `G`'s output, currently ~1.7 dB at depth 2. For the blend to help,
the extracted-audio embedding must beat the enrollment embedding -- but bad enrollment produces bad
conditioning produces a bad candidate, so the two requirements fight.
*The decisive experiment -- an ORACLE-REFINEMENT upper bound.* Run refinement but accept a
candidate only when it is genuinely closer to the true speaker (measured against `scene.sources`
with the **eval** encoder, §6.3). Scoring-time only, never deployable; the point is the ceiling.
**If oracle-refinement is still net-negative, refinement has no headroom on this extractor and that
is a publishable negative result with a stated mechanism. If it is positive but the real gate
cannot find it, the acceptance RULE is what is broken, not refinement.** One eval pass, no
training, and it settles which of the two we have been looking at for three phases.
*If the ceiling turns out positive, the levers in order:* variance-weighted blend instead of the
fixed 0.5/0.5 (already flagged in Phase 2's close-out as untested and needing no benefit test, since
a noisy candidate down-weights itself); embed only the cleanest part of the extracted audio
(lowest-depth frames) rather than the longest overlap run; and better `G` -- at 10-15 dB the
extracted audio would embed near-cleanly and refinement's premise would finally hold. Current read:
refinement is probably gated on extractor quality rather than on the update rule, but the ceiling
run is what turns that from opinion into evidence.

#### STAGE B -- full work list (4/5/6 done; 1/3 RUN; 2 run but VOID, re-run queued; 7/8/9 open)

Ordered so that everything needing no GPU comes first, and the one training run is entered with its
design already decided by measurement rather than by guess.

*No GPU -- do these first:*

1. ☒ **DONE. Overlap-dilation sweep** (item 3) -- **recovers 91% of the depth-2 gap**
   (-2.98 -> -0.28 dB at 800 ms, 52% win rate vs oracle), no retraining. Also priced "copy, don't
   separate" at 43.9 dB and closed the Phase 1 context limitation at +0.02 dB. Operating point NOT
   yet chosen: needs the un-stratified metric first. See Session B Q1.
2. ☒ **RUN, THEN VOIDED. Oracle-refinement ceiling** (item 4) -- the rule scored the whole waveform
   while the table reported depth 2, so its number was not a bound. Fixed 2026-08-20; **re-run
   queued (~1.7 h)**. The headroom question is currently *unknown*, not answered. See Session B Q2.
3. ☒ **DONE. `tune_gate.py` on a long-scene dev split** (item 1) -- `V_i` scores **J = +0.373 at a
   1e-4 threshold**, overturning four "structurally dead" conclusions; the shipped `0.05` is 500x
   too high. `tau_margin` scores J = +0.046 ("no usable threshold"), and the conditioning probe
   CONFIRMED the fixture was valid (clip50 steers at 9.41 dB), so the margin is genuinely not a
   detector. Only `max_mean_variance: 1e-4` is worth freezing; `tau_margin` needs replacing rather
   than re-tuning, and cannot be evaluated on this checkpoint anyway. See Session B Q3.
4. ☒ **DONE. Fixed `aggregate_phase3._table`** so the accumulation stratification pairs before
   filtering. Applied to the committed Stage A CSVs; corrected numbers in the cautions table above.
5. ☒ **DONE. Dropped `real_forced_m`** from `phase3_librimix_3spk_long.yaml`: identical to `real`
   to 3 decimals, ~25% of runtime measuring a constant. Kept in the SHORT-scene configs, where
   counting still fails.
6. ☒ **DONE. Moved the gate-tune dev split** to `offset: 1000`. Note the config header's
   justification was itself **wrong** -- it claimed the curriculum runs used `limit: 650`, but
   `configs/phase2/dod/phase2_librimix_curriculum_3_4_5_train_scratch.yaml` uses `limit: 800` on
   all three depths, so the old `offset: 650` overlapped 150 training scenes. 1000 leaves a
   200-row margin. No committed number is affected: this split has never been run.

*The one GPU session -- design it from the measurements above:*

7. ☑ **Mask augmentation** -- **code landed, RUN PENDING** (`dagger/data/mask_augment.py`, wired
   into `build_scene_crop_dataset`'s `_prepare`, not `__getitem__`: `_prepare` is where enrollment
   happens, so augmenting there contaminates enrollment realistically, which is what exercises the
   whole gate path; `w_overlap` is precomputed from clean masks and is now recomputed after
   augmentation). **Targets the measured error profile, not the assumed one:** at 2-minute scenes
   the profile is miss 0.105 / confusion 0.008, so it simulates **dropped speaker activity inside
   overlapped regions**, NOT the label-swap and boundary-jitter mix originally planned from the
   short-scene runs where confusion dominated (0.171-0.187). Strength is deliberately NOT
   pre-committed -- pick it from Session B's dilation result.
8. **Train on long, multi-depth scenes** (item 2) so train and eval geometry match and depth 3+
   exists, and so the same session repairs the absolute-quality ceiling.
9. **The 2x2:** {baseline ckpt, mask-augmented ckpt} x {oracle, real}, on the long-scene corpus.

*Deferred, with reasons:*

10. **Whisper WER** -- Phase 4 per §3; `dagger/metrics/__init__.py` already promises it.
11. **Real corpora (AMI-SDM etc.)** -- Phase 4. Note the Stage A gap is NOT a clean lower bound on
    AMI: LibriMix is easier acoustically (no reverb, hard boundaries) but its short synthetic scenes
    were *harder* for speaker counting, and AMI gives each speaker minutes of clean speech. The
    `Diarizer` ABC and the four-arm harness transfer directly, since AMI ships reference
    annotations and so supports the mandatory oracle arm (§6.2).

---

**STAGE B -- SESSION A LANDED (2026-08-18): all Stage B code written and unit-tested offline; the
three measurement runs are queued but NOT run.** Suite **434 passed / 1 skipped** (baseline 381 --
note the "357" quoted in the Stage A note was already stale). Nothing below has seen real audio
except item 4, which is CPU-only and has been applied to the committed Stage A CSVs.

*The compute model this phase now works under.* Everything runs on Kaggle; "CPU" means a Kaggle CPU
session (burns no GPU quota), "GPU" a GPU session. **The local dev machine cannot run any model
work at all** -- no NeMo, no pyannote, no `DAGGER_DATA_ROOT`, and the clip50 checkpoint is not in
`checkpoints/` (only the Phase 1 and Phase 2-finetuned ones are). So "no GPU" in the work list
above never meant "runs locally": items 1, 2, 3 and 7 all need TitaNet + the extractor + (for 1
and 3) pyannote on real audio. Work is therefore grouped by *which session type it needs*:

| session | items | what it costs |
|---|---|---|
| **A** (local / Kaggle CPU) | 4, 5, 6 + the code for 1, 2, 3, 7 | none -- done |
| **B** (Kaggle GPU, no training) | run 1, 2, 3 | one short session |
| **C** (Kaggle GPU, training) | 7, 8, 9 | one long session, designed from B |

*Session B's three runs, all against Stage A's chain@0.3 corpus and the clip50 checkpoint so every
row pairs against the committed baseline:*

* `configs/phase3/experiments/phase3_librimix_3spk_dilation_sweep.yaml` --
  `dilate_overlap_ms: [0, 10, 25, 50, 100, 200]`.
* `configs/phase3/experiments/phase3_librimix_3spk_refine_ceiling.yaml` -- `refine.oracle_ceiling: true`.
* `configs/phase3/experiments/phase3_gate_tune_dev_long.yaml` -- `tune_gate.py` on a **dev-clean**
  long-scene split.

The sweep's `0 ms` point is the Stage A baseline re-run and doubles as a reproduction check, at no
extra cost since it is one value in the swept list.

#### Three design decisions worth knowing before reading the code

1. **The dilation sweep runs INSIDE one invocation, not one run per value.** `dilate_overlap_ms`
   accepts a list, and `score_scene_all_arms` loops it over the cached `Regions`. Dilation is pure
   post-processing, so pyannote runs **once per scene** however many values are swept. That is a
   cost saving, but the correctness reason matters more: pyannote's clustering is not guaranteed
   bit-reproducible, so re-running it per sweep point could let the *regions themselves* drift
   between the points being compared -- turning a one-variable comparison into a two-variable one.
   Pinned by a test that counts diarizer invocations.
2. **The refinement ceiling scores AUDIO, not embeddings.** The natural formulation -- accept iff
   the candidate embedding is closer to the true speaker's -- does not typecheck, and the way it
   fails is instructive: candidates live in `phi`'s TitaNet space, an eval-encoder reference in
   WavLM space, and the only way to make them comparable is to embed the true source with `phi` --
   exactly the training-encoder-as-metric violation §6.3 forbids. So the rule accepts iff the
   *reconstruction* from the candidate scores higher SI-SDR against the clean source. No encoder is
   involved, so §6.3 cannot be violated, and it bounds the quantity actually reported rather than a
   proxy for it. Cost is one extra `reconstruct_all` per round -- each speaker's output depends
   only on its own embedding (§1), so every candidate evaluates in one batched call.
   *Reading it:* `coarse_to_fine` minus `no_recursion` **within the same run** (the other three
   systems never refine, so they are a valid control), plus the new `ceiling_accept_gate_would_reject`
   reason count, which IS the headroom the deployable gate could not reach. If that count is ~0,
   the gate was already making the right calls and refinement's deficit is not an acceptance
   problem at all.
3. **`tune_gate.py` needed a real diarizer to make `V_i` measurable, not merely to be tidy.** Under
   oracle regions `V_i` is *structurally* 0 -- one solo run -> one clip -> variance over a single
   sample -- while the contaminated fixture (enrolling from a speaker's overlap region, which has
   several runs) is nonzero. Sweeping "identically 0" against "anything at all" separates them
   perfectly and reports a spectacular Youden's J for a property that does not exist in deployment.
   The real question is whether contaminated variance clears the floor a real diarizer's fragmented
   solo regions already produce on honest enrollment (Stage A: nonzero in 1332/1800, max 3.24e-4).
   Only a real-diarization honest population can answer it. The `tests/phase3/test_tune_gate_real_regions.py`
   suite asserts the oracle honest population is exactly 0, so the premise is checked rather than
   assumed.

#### Three defects the new tests caught before any GPU time was spent

Each is the same shape as every reporting defect this project has shipped: a plausible number, and
nothing failing.

1. **`np.convolve(..., mode="same")` returns `max(len(signal), len(kernel))` samples.** A dilation
   half-width wider than the scene produced an over-long mask. It happened to raise on the
   broadcast against `activity`, but would have silently mis-sized a squarer array -- and the sweep
   reaches that regime *legitimately*, since a large value is exactly how the enrollment-starvation
   limit gets measured. Replaced with an exact O(n) prefix-sum window.
2. **Every existing Phase 3 table would have averaged across dilation values.** None of them filter
   on `dilate_ms`, so a six-point sweep would have blended six different pipelines into one cell
   labelled "depth 2" -- the same two-variables-in-one-column mistake that cost Phase 2 five runs on
   the depth axis. Every table below the sweep section now reports the **0 ms baseline only**, and
   `aggregate_phase3.py` holds `dilate_ms` fixed inside the pairing key so an arm difference is
   always taken at one dilation.
3. **The new config guards ran after `build_dataset`.** `main()` already documents the opposite
   intent for arm validation ("a config typo should not first require DAGGER_DATA_ROOT to be
   mounted"); the refine guard now sits with it. All four guards verified to refuse before the
   corpus mounts.

#### Regression evidence

* `aggregate_phase2.py` still reproduces `phase2_accumulation_scratch345clip50.md`
  **byte-identically** (modulo the trailing newline already documented in Phase 2's close-out), so
  the `dagger/metrics/phase2_scores.py` changes did not disturb any committed Phase 2 number.
  `plot_phase2_depth.py` still runs and still flags the n=3 thin point.
* `SCORE_FIELDS` and `GATE_FIELDS` are **untouched**, so the Phase 2 CSV schema is unchanged. Every
  new parameter (`score_scene`'s `refine_oracle_ceiling`, `refine_embeddings`' `accept_fn`,
  `measure_scene`'s `diarizer`, `build_scene_crop_dataset`'s `mask_augment`) defaults to the prior
  behaviour, and `accept_fn=None` is pinned bit-identical by test.
* `dilate_ms` is Phase 3-only: `load_score_rows` reads a missing column as `0.0`, so the committed
  Stage A CSVs (which predate it) load unchanged.

#### Still open going into Session B

* **`refine.rounds`.** Both Session B configs keep `rounds: 2` to match the Stage A baseline
  exactly. `refine.rounds: 0` remains the DEFAULT for reported systems -- refinement is net-harmful
  in every regime measured -- and B2 is the run that decides whether that becomes final.
* **Depth 3 does not exist in the Session B corpus.** Chain at overlap 0.3 makes s1 and s3
  disjoint. That is deliberate (comparability with the committed baseline); depth 3 arrives with
  the scheduled long-solo geometry in Session C.
* **Absolute quality is still the extractor's operating point**, not diarization's -- the *oracle*
  arm reaches only 1.73 dB at depth 2. Nothing in Session A changed training, so nothing here could
  have moved it. It is repaired, if at all, by Session C's training budget.

---

**STAGE B -- SESSION B RESULT (2026-08-20): dilation recovers 91% of the oracle-vs-real gap at
depth 2 with no retraining; `V_i` WORKS and four prior "it is dead" conclusions were a threshold
error; the refinement ceiling was mis-specified and must be re-run.** All three runs completed in
one Kaggle batch (~9 h). Artifacts in `results/phase3/experiments/experiment_stage_B_run_1/`.
Checkpoint unchanged: `checkpoints/phase2/proposed_librimix_curriculum_3_4_5_scratch_clip50.pt`.

The reproduction gate passed first: a 3-scene smoke reproduced the committed Stage A output
**144/144 rows byte-identically** at `dilate_ms == 0`, so every number below sits on verified code.
That gate was the one fatal assertion in the batch, deliberately.

#### Q1 -- overlap dilation: the gap is a MASK problem, and it is mostly fixable for free

Two runs. First a **mask-only curve** (no extractor, no encoder -- pure geometry, 3 min for 25
scenes, ~1/30th the cost of measuring the same points in SI-SDR), which repriced the whole
experiment before any GPU time went into it:

| dilate (ms) | 0 | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|---|---|
| overlap recall | 0.774 | 0.788 | 0.802 | 0.828 | 0.876 | 0.946 | 0.989 | 0.998 | 1.000 |
| false alarm | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0003 | 0.0026 | 0.0149 | 0.0451 | 0.1077 |
| min solo left | 24.6 s | | | | 23.0 s | 22.0 s | 20.8 s | 19.3 s | 16.3 s |
| scenes starved | 0/25 | | | | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |

**Recall reaches 1.000.** The missed overlap is boundary-recoverable, not bulk non-detection --
just at a ~400-800 ms scale (marginal gain per doubling peaks at +7.02 pts over 200->400 ms, then
collapses to +0.87 at 800->1600). That scale matches pyannote's minimum-duration constraints
trimming short overlap incursions. **Enrollment never starves**, at any value: these are 2-minute
scenes with 20-40 s of solo per speaker, so even a 3.2 s dilation barely dents it. The binding
constraint is false alarm, not enrollment -- the opposite of what the config was written to guard
against, whose grid `[0, 10, 25, 50, 100, 200]` would have topped out at recall 0.876 and captured
about half the available gain. Do the cheap geometry curve before the expensive sweep.

Then the SI-SDR sweep on the re-chosen grid (25 scenes, `no_recursion`, absolute dB):

| dilate | oracle d1 | oracle d2 | real d1 | real d2 |
|---|---|---|---|---|
| 0 | 47.03 | 1.69 | 40.30 | -1.29 |
| 200 | 13.85 | 1.70 | 37.92 | 0.02 |
| 400 | 8.92 | 1.71 | 25.50 | 1.02 |
| 800 | 3.16 | 1.71 | 10.27 | **1.41** |

Paired against the **oracle 0 ms baseline** -- the actual target, since that is the ideal the real
arm is trying to reach:

| dilate | depth 2 | win | depth 1 | win |
|---|---|---|---|---|
| 0 | -2.98 +-0.27 | 9% | -6.73 +-1.20 | 24% |
| 200 | -1.67 +-0.20 | 12% | -9.11 +-1.48 | 24% |
| 400 | -0.67 +-0.17 | 32% | -21.52 +-1.74 | 8% |
| 800 | **-0.28 +-0.15** | **52%** | -36.76 +-1.22 | 0% |

The 0 ms row reproduces Stage A (-2.98 on 25 scenes vs -3.11 on 50). **At 800 ms the real arm hits
a 52% win rate against oracle diarization at depth 2** -- statistically indistinguishable from
perfect masks, with no retraining. 91% of the gap, from one config key.

**Three findings, and the second is the one for the paper:**

1. **Missed overlap was the whole story, and it is recoverable at the mask level.** Stage A
   attributed the gap to the predicted masks; this confirms it and fixes most of it.
2. **The oracle arm turned CLAUDE.md §2's "copy, don't separate" from an assertion into a
   measurement.** Dilating *exact* masks drops depth 1 from **47.03 -> 3.16 dB**: a **43.9 dB**
   price for running `G` over already-clean audio. That rule has been asserted since Phase 0 and
   had never been priced. This is why the oracle arm was swept too rather than only the real one --
   it isolates the pure over-extraction cost with no diarization error in it.
3. **Extra context for `G` is worth +0.02 dB.** Oracle depth 2 is flat across the sweep
   (1.69 -> 1.71). Phase 1's recorded limitation -- "``x_O`` is built with a hard binary mask, so
   ``G`` never sees context beyond the overlap region... scheduled to be revisited" -- is a
   non-issue. Closed, free.

**The catch, and why the dB numbers mislead here.** Depth 1 collapses. But depth-1 SI-SDR starts at
47 dB, where the error energy is ~1e-4 of signal, so corrupting even ~1% of samples multiplies
error energy a hundredfold and costs 20 dB. **Near-perfect scores are fragile**, and the dB scale
therefore massively over-weights degradation that is perceptually inaudible (47 dB and 25 dB both
sound flawless) while under-weighting the depth-2 gain that is the difference between unusable and
borderline usable.

**This exposes a real reporting gap: there is no un-stratified, whole-output SI-SDR anywhere in
this project**, and it is the number that decides the dilation operating point. §6.4 forbids
reporting an aggregate *instead of* stratification -- not alongside it. Until it exists, the
optimum is unknown; crude energy arithmetic puts it near 200-400 ms, but that is arithmetic, not a
measurement. **Add it before choosing a value.** See §7.

#### Q2 -- the refinement ceiling was MIS-SPECIFIED. Its number is void; the re-run is queued.

The run executed and the ceiling was active (23 `ceiling_accept_gate_would_reject`, 139
`ceiling_reject_gate_would_accept`). Deficit `coarse_to_fine - no_recursion` at depth 2, matched
control on the same 25 scenes:

| arm | standard gate | oracle ceiling |
|---|---|---|
| oracle | -0.59 +-0.15 (\|t\|=4.0) | -0.24 +-0.12 (\|t\|=2.0) |
| real | -0.57 +-0.19 (\|t\|=3.1) | -0.37 +-0.17 (\|t\|=2.2) |

**Read as a ceiling this is nonsense, and the nonsense is what exposed the bug.** An acceptance
rule that only ever accepts improvements cannot do worse than accepting nothing -- and accepting
nothing IS `no_recursion`, because refinement round 0 starts from the enrollment embeddings. The
deficit must be >= 0. It was negative. A bound that cannot lose, losing.

*Cause.* `make_oracle_accept_fn` scored candidates on the **whole waveform**; the table reports
`si_sdr_by_depth` at **depth 2**. Different objectives, so the monotonicity argument never
transferred. The mechanism is SI-SDR's scale invariance: it fits a scalar before measuring the
residual, and which samples you include decides what that scalar becomes. Over the whole waveform
~75% of the audio is a bit-exact solo copy, which pins the scalar near 1 and makes any *level*
error in the overlap region cost full price; over the depth-2 slice the scalar floats and absorbs
that level error for free. So a candidate that fixes `G`'s **level** while worsening its **shape**
wins the first comparison and loses the second. That is what it kept picking.

*Confirmed directly, not inferred.* Splitting speakers by whether the ceiling accepted anything:

| | speakers | mean depth-2 delta |
|---|---|---|
| ceiling accepted nothing | 48 (oracle) / 44 (real) | **+0.000** (max abs delta 0.00e+00) |
| ceiling accepted >= 1 round | 27 (oracle) / 31 (real) | **-0.674 / -0.886 dB** |

Every accepted change -- each an improvement by the rule's own measure -- made the reported metric
worse, and untouched speakers moved by *exactly* zero.

**What survives from this run:**

* The refinement plumbing is verified: untouched speakers are bit-identical to `no_recursion`.
* The standard gate's deficit (-0.59 / -0.57 dB) replicates Phase 2 and Stage A on a third corpus.
* **A genuine finding about the metric:** optimizing whole-waveform SI-SDR *actively harms*
  extraction quality by 0.67-0.89 dB. Worth remembering before anything here is ever optimized
  against a whole-output number.
* The accept rate decomposition: the real gate accepts **70%**, the mis-specified oracle accepted
  **31%**.

**What does NOT survive:** the -0.24 / -0.37 figures are not a ceiling, and the question "does
refinement have headroom?" is **unknown**, not answered. `refine.rounds: 0` stays the default on
the strength of the standard-gate result, not on the ceiling.

*Fixed 2026-08-20.* `make_oracle_accept_fn(row_sources, scoring_depth, min_depth=2)` now masks with
the same `si_sdr_regionwise` that `si_sdr_by_depth` uses, so the rule optimizes exactly what is
reported. `score_scene` passes the `scoring_depth` it already computes; no config changes, so the
re-run is the same command (~1.7 h). Guarantee and its one limit: what cannot decrease is SI-SDR
*pooled over depths >= min_depth*. The chain corpus has a single overlap depth, so pooled IS the
reported per-depth number there; with several overlap depths an individual depth could still move
while the pool improves. Six tests added (440 passed / 1 skipped), including
`TestTheCeilingCannotLose` -- the end-to-end monotonicity guard that would have caught this, and
which did not exist -- plus a fixture that pits the two objectives against each other explicitly
(whole-waveform 13.06 vs 20.97, depth-2 slice 33.98 vs 13.98 on the same pair of waveforms). Note
the first version of *that* fixture did not actually disagree, and a deliberate guard-the-guard
test caught it; without it the regression test would have passed for the wrong reason.

#### Q3 -- `V_i` WORKS. The threshold was wrong by 500x. And the margin does NOT work.

First run of `tune_gate.py` in the project's history, on a long-scene **dev-clean** split
(50 scenes, verified disjoint from test: 40 test speakers, 39 dev, 0 shared).

**`mean_variance` (`V_i`) -- contaminated vs honest enrollment:**

| threshold | detection | false rej. | J |
|---|---|---|---|
| 1e-5 | 100.0% | 98.0% | +0.020 |
| **1e-4** | **45.3%** | **7.9%** | **+0.373** |
| >= 5e-4 | 0.0% | 0.0% | 0.000 |

Medians: honest **0.00005**, contaminated **0.00010**. n = 151 honest / 148 contaminated.
J = +0.373, comfortably over `tune_gate.py`'s 0.1 refusal bar.

**Every config in this repo ships `max_mean_variance: 0.05` -- 500x above the entire usable range.**
That is the whole explanation for four separate "V_i is structurally dead" conclusions (Phase 2
close-out x2, Stage A, and the config comments). Each observation was correct -- `mean_variance` is
nonzero but never exceeds ~3e-4 -- and each *inference* from it was wrong. The check was never
dead; it was never switched on.

Two things to be precise about, because "V_i works" is easy to over-read:
* it catches **under half** of contaminated enrollments at a 7.9% false-rejection cost -- a partial
  detector, not a solved problem;
* the usable window is **one order of magnitude wide** (1e-5 and 5e-4 both give J ~ 0), so the
  value will not transfer across corpora or checkpoints without re-tuning. Re-run `tune_gate.py`
  whenever either changes.

**`margin` (`M_i`) -- swapped vs correct conditioning:** best J = **+0.046** (at 0.3: 19.9%
detection, 15.2% false rejection) -> `NO USABLE THRESHOLD`. Medians: correct **0.43873**, swapped
**0.41965**. So the one check believed to be working is barely separating the populations --
consistent with `gated_deflation` collapsing onto `ungated_deflation` at 98-99% accept, and with
the refinement gate rubber-stamping 70%.

*That result was held pending a fixture check, and the check has now run -- see below. The fixture
was valid, so the margin result stands.*

Finding 1 never had that dependency: the `V_i` fixture contaminates *enrollment*, which never
routes through `G`.

#### Q3 follow-up (2026-08-20) -- the conditioning probe: the fixture was VALID

The margin sweep only tests the margin if `G` actually *responds* to its conditioning. If the
checkpoint did not steer, `G(x_O, e_j)` and `G(x_O, e_i)` would be the same waveform, the fixture
would have handed the margin two identical populations, and J ~ 0 would be arithmetic rather than a
finding. `scripts/probe_phase1_conditioning.py` on **clip50, 15 scenes of the same dev-long corpus
`tune_gate` used** (no pyannote -- the probe uses oracle regions; ~7 min):

| | passthrough vs `x_O` | diagonal | off-diagonal | **steering margin** |
|---|---|---|---|---|
| Phase 1 DoD checkpoint (recorded) | 2.89 | **+5.80** | -6.95 | **12.75 dB** |
| clip50 (this probe) | 2.45 | **+2.36** | -7.05 | **9.41 dB** |
| Phase 1's collapsed run (recorded) | 35.8 | -- | -- | 0.16 dB |

`VERDICT: conditioning STEERS`. Relative output difference between two speaker embeddings is
**0.9816** (working reference ~0.86, collapsed ~0.05), and at 2.45 dB against `x_O` the output is
nothing like a scaled copy of the mixture. **So the swapped population was genuinely different
audio, and `tau_margin`'s J = +0.046 stands: it is not a detector.**

**One sharp reading of that table: suppression is IDENTICAL (-6.95 vs -7.05); the entire 3.3 dB
shortfall is in the diagonal.** clip50 learned to reject the wrong speaker but not to reconstruct
the right one -- exactly the signature of the undertraining Phase 2's close-out documented
(~13% of Phase 1's per-depth exposure for the 3-speaker case). Note this is a *lower* bound on the
disparity, since the Phase 1 numbers were measured on a different corpus.

#### The unifying mechanism: everything that EMBEDS `G`'s OUTPUT is gated on `G`'s QUALITY

The probe forces a more precise diagnosis than "the margin is broken", and it explains four
separate standing puzzles at once.

The probe measures steering in **waveform** space; the margin operates in **embedding** space, on
`G`'s output. Those can disagree, and here they do: the output is clearly the right speaker
acoustically (+2.36 vs -7.05 dB), yet TitaNet embeds it to something whose cosine to the right
enrollment (0.43873) barely beats its cosine to the wrong one (0.41965). At **+2.36 dB the output
is still mostly distortion**, so its embedding is dominated by artifacts rather than identity and
lands roughly equidistant from every enrollment. That is precisely the "raw similarity is always
positive, voices aren't orthogonal" problem §2 introduced the margin to solve -- reappearing
*inside* the margin, because both of its terms are contaminated the same way.

**The margin formula is not wrong. It is gated on extractor quality** -- exactly like refinement
is. And once stated, the same root cause covers:

| observation | explanation |
|---|---|
| refinement net-harmful in every regime tested | its candidate is embedded from ~2 dB audio |
| `tau_margin` J = +0.046 | the margin is computed on ~2 dB audio |
| `gated_deflation` ~ `ungated_deflation` (98-99% accept, Phase 2 and Stage A) | the gate's only live check cannot discriminate |
| the ceiling accepted 31% where the real gate accepted 70% | the gate is close to noise |

**This makes a testable prediction: all four recover together as `G` improves.** That is a claim
about the training budget, not about the gate design or the blend rule -- so it argues for spending
Session C on training rather than on redesigning the acceptance rule, and it means a gate redesign
evaluated on *this* checkpoint would be measuring the extractor either way.

Two caveats worth keeping attached. `V_i` is **not** in this family -- it embeds enrollment clips
from the mixture, never `G`'s output, which is why it is the one check that works at this
checkpoint. And the prediction is currently a mechanism, not a measurement: it is confirmed only by
re-probing after a better-trained checkpoint exists.

Rejections were `margin` only (67 for `coarse_to_fine`); vad and artifact stayed inert, as in every
prior run.

#### Next actions, in order

1. ☒ **DONE. Conditioning probe on clip50** -- steers at 9.41 dB, so the margin fixture was valid
   and J = +0.046 stands. Produced the unifying mechanism above.
2. ☒ **DONE (2026-08-20). Un-stratified whole-output SI-SDR.** `score_scene` now returns a third
   list, written to **`{stem}_overall.csv`** -- its own file at its own grain, one row per
   (scene, speaker, system), NO depth column. The separation is structural on purpose: every
   per-depth table here groups by `depth`, and a whole-output row carrying one would be absorbed
   as "another depth" (the `+-inf` drop, the `dilate_ms` sweep and the ceiling's objective are the
   three times this project has shipped that shape of defect). `run_phase3.py` gains an "Overall
   SI-SDR" section, which is the ONLY table in the file where dilation sweep points can be
   compared against each other -- every other one is baseline-only by design.
   `run_phase2.py` drops the third list, so its committed CSVs stay byte-identical.
   **It is a reporting number only**: scale-anchored by the bit-exact solo copy, so it rewards
   fixing a level error over a shape error, and using it as a selection target is precisely what
   voided the Stage B ceiling.
3. **Re-run B2** with the corrected ceiling (~1.7 h). The headroom question is currently unknown.
   Expect it to stay negative: the unifying mechanism predicts refinement cannot pay while `G`
   sits at ~2 dB, whatever the acceptance rule. A ceiling that is negative *for a stated reason*
   is a stronger result than one that is merely negative.
4. **Re-run the dilation sweep** reading the new overall metric.
   *Config ready (2026-08-20):* `configs/phase3/experiments/phase3_librimix_3spk_dilation_v2.yaml`,
   which differs from the run-1 config by `eval.tag` ALONE -- a second file rather than a re-run of
   the first, because re-running it would overwrite the committed run-1 results, and run 1's
   dilation table is the phase's headline finding. What changed is the CODE, not the experiment:
   run 1 predates `_overall.csv`, which is exactly why its operating point could not be chosen.
   The grid stays `[0, 200, 400, 800]` so every per-depth row is directly comparable and the only
   new information is the overall column. **Budget: ~6.7 h** (25 x 2 x 4 units at 120 s/unit), so
   this does NOT fit in a session alongside B2 (~1.7 h) and `vi_on` (~5.0 h) -- give it its own,
   or drop `limit` to 15 (~4.0 h).
5. **Set `max_mean_variance: 1e-4`** and re-run a gated comparison -- `V_i` firing for the first
   time changes what `gated_deflation` means. Note this is now the ONLY live check in the gate,
   since `tau_margin` is confirmed inert, so `gated_deflation` currently differs from
   `ungated_deflation` by almost nothing.
   *Config ready (2026-08-20):* `configs/phase3/experiments/phase3_librimix_3spk_vi_on.yaml`,
   which differs from the Stage A baseline by **exactly one line** so rows pair and the effect is
   attributable to the threshold alone. The baseline config was deliberately NOT edited -- it
   generated the committed Stage A CSVs, and retro-editing it would break §7's "one command
   regenerates this". Expect the gated systems to possibly get *worse*: `V_i` is a partial
   detector (45.3% detection at 7.9% false rejection), and a false rejection costs quality on
   every scene while a missed detection costs only on contaminated ones.
6. ☒ **DONE (2026-08-20). Reconciled the run-1 configs with what actually ran, and taught
   `aggregate_phase3.py` to read `_overall.csv`.** Two separate things, both §7 hygiene:

   *The reproducibility defect.* The Stage B run-1 results did NOT regenerate from their committed
   configs. The Kaggle notebook mutated them in-kernel to fit a measured 120 s/unit cost into the
   session budget, wrote the mutated copies to `/kaggle/working`, and never brought them back:

   | | config said | run actually did |
   |---|---|---|
   | dilation grid | `[0,10,25,50,100,200]` | `[0,200,400,800]` |
   | dilation arms / limit | 3 / 50 | 2 / 25 |
   | ceiling arms / limit | 3 / 50 | 2 / 25 |

   Both configs now hold exactly what ran, verified programmatically against the committed CSVs.
   **The lesson is about the harness, not the configs:** a notebook that overrides a config
   in-kernel breaks §7 silently, because nothing fails and the results look fine. Future runs
   should write the effective config back beside the results, or vary the config file itself.
   (The ceiling config carries a note that run 1's numbers are void for an unrelated reason -- the
   mis-specified objective -- so re-running it now yields the corrected answer from the same
   config, which is the intended behaviour: the config did not change, the code did.)

   *The aggregation gap.* `aggregate_phase3.py` read only the per-depth CSV, so the new metric
   never reached a gap table. It now loads the sibling `_overall.csv` automatically (not as a CLI
   argument -- naming both invites mismatching them) and renders an "overall" section per
   comparison, paired on `(source, scene, speaker, system, dilate_ms)`. Dilation joins that key for
   the same reason it joins the per-depth one: pairing `real` at 400 ms against `oracle` at 0 ms
   would measure the knob and the diarizer together. A missing sibling degrades gracefully, since
   every CSV written before 2026-08-20 lacks one.

7. **Session C's case is stronger than it was.** The mechanism above says the margin, the gate,
   refinement and the ceiling are all downstream of `G`'s quality, so they cannot be fixed
   independently of it -- and a gate redesign evaluated on this checkpoint would be measuring the
   extractor regardless. Budget arithmetic is in Phase 2's close-out.

---

**STAGE B -- VERIFICATION PASS (2026-08-23): all five checks ran; four pass, and the fifth passed
only vacuously and was hiding a real defect. Test E's byte-identity failure is CLOSED as benign
(line endings, exact arithmetic below). The un-stratified metric added in `9f62f4b` is
LEVEL-DOMINATED and could not have chosen the dilation operating point -- diagnosed, fixed, and the
fix is the first thing in this project that can see an extractor level error at all.**

Artifacts: `results/phase3/experiments/verify_4_questions_run/` (12 files) and the executed
notebook `dagger.ipynb` at the repo root, outputs embedded. ~90 min of Kaggle GPU, all inference,
no training. Offline suite inside the same session: 453 passed.

#### The scorecard

| test | verdict | evidence |
|---|---|---|
| A -- reconciled config reproduces run 1 | **PASS** | 576 shared rows, 0 mismatched, exact string equality (46.6 min) |
| B -- `_overall.csv` grain + pooling | **VACUOUS PASS** | `checked = 0`; see below |
| C -- `aggregate_phase3` reads the sibling | **PASS** | loads when present, degrades when absent |
| D -- does `V_i` fire at 1e-4? | **PASS, read the rate** | 6 rejections / 81 decisions, max 0.000144 (17.7 min) |
| E -- Phase 2 byte-identity | **PASS (soft)** | 0/5400 si_sdr differ, `max abs delta = 0.000e+00` (25.4 min) |

Test A also settles the A5 guard question for the 3-tuple refactor: the per-depth rows are
untouched, verified against the committed corpus rather than synthetically.

#### Test E: benign, and the arithmetic is exact

Committed `8745fbfb...` vs fresh `84d69f89...`, 489764 vs 495166 bytes. The committed file has
**0 CR**, 5400 LF, and **no trailing newline**; the fresh one is 5401 CRLF-terminated lines. That
is 5401 extra CR + 1 extra LF = **5402**, which is the observed delta exactly. Values are
bit-identical on all 5400 rows.

*The guard needs narrowing, not abandoning.* A byte-level comparison across environments tests the
csv dialect, the float repr and the CUDA stack as much as it tests the code. Replace it with
**every shared key present and `max |delta si_sdr| < 1e-3 dB`**, which still catches a logic change
while surviving a driver upgrade or a line-ending difference. (Not yet applied.)

#### Test B passed while verifying NOTHING, and that is the story of this session

The pooling check skips any speaker lacking a `+inf` depth row:

```python
if not (finite and any(d == math.inf for d in depths)):
    continue
```

These 2-minute chain scenes have **zero** `+inf` rows -- the whole 25-scene sweep has only 40, and
none in the 3 smoke scenes -- because crossfade ramps sit inside the depth-1 region, so a solo copy
scores 38-88 dB rather than exactly perfect. The guard therefore `continue`d all 288 rows and
printed `PASS -- 0 speakers verified`. The repo's own copy of this test
(`tests/phase3/test_overall_metric.py`) has `assert checked > 0` and was never at risk; the
notebook's copy dropped it.

**Had it run it would have failed.** 271 of 288 overall rows sit BELOW the worst per-depth value for
the same speaker, by 5-20 dB.

#### The `_overall.csv` defect: not a wiring bug, a metric that measures the wrong thing

Three findings, each ruling out a candidate rather than assuming:

1. **It is nearly blind to depth 1.** Dilation collapses the solo block by 42-81 dB per speaker;
   the overall number moves by -2.30 to +1.93 dB, in *both* directions. Across all 288 rows,
   `corr(overall, depth1) = -0.206` and `corr(overall, depth2) = +0.532`.
2. **Not the documented scale-anchoring caveat, at this magnitude.** Simulated: at this corpus's
   geometry (solo ~75% of each speaker's speech) the whole-track number bottoms out near **-4.5 dB**
   even with the overlap output 100x too loud. Measured mean is **-13.71**, min **-45.37**.
3. **Not VAD truncation.** The first hypothesis was that `active_mask`'s -40 dB threshold drops soft
   onsets, leaving target energy at `depth == 0` where the reconstruction emits exactly 0. Dead:
   `dagger/data/librimix.py:174,180` uses `segments_from_chunks`/`segments_from_placement`, so
   activity is the exact placement window and depth 0 holds no energy on either side. Confirmed
   directly -- a local chain-placed repro has **0 samples at depth 0**.

*Root cause, reproduced locally and proven with a one-variable sweep.* SI-SDR fits ONE scalar over
whatever samples it is handed. Score the whole track and the near-exact solo copy pins that scalar
near 1, so a pure **level** error in the overlap region is charged at full price; score each depth
separately and the scalar floats per region and absorbs the same error for free. Holding the
estimate's SHAPE fixed and moving only its overlap gain:

| overlap gain | depth 1 | depth 2 | whole track |
|---|---|---|---|
| 1x | 39.15 | -0.00 | **+5.23** |
| 3x | 39.15 | -0.00 | -1.41 |
| 10x | 39.15 | -0.00 | -5.36 |
| 1000x | 39.15 | -0.00 | **-7.51** |

Every per-depth score is bit-identical down the column while the whole-track number falls 13 dB.
**So the whole-track metric is dominated by an extractor LEVEL error, not by the quality tradeoff
it was added to weigh** -- and it therefore could not have chosen the dilation operating point,
which was its entire purpose. Note it would still have picked 800 ms, matching the per-depth story:
a broken metric returning the plausible answer, which is how this defect class keeps surviving real
runs here.

*This also means an extractor level error has been present and unmeasurable all along.* Every
SI-SDR in this project is scale-invariant, so nothing could see it. It shows up only as a
disagreement between two metrics, which is how it stayed hidden until these two existed side by
side.

#### The fix (landed, suite 471 passed / 1 skipped, up from 453)

`dagger/metrics/sisdr.py` gains two functions, and the overall row gains two columns beside the
existing `si_sdr` (which is KEPT -- it is the only score here that can see a level error):

* **`si_sdr_pooled_by_depth` -> `si_sdr_pooled`. THE EXCHANGE RATE.** Fits the scale per depth,
  then pools error energies weighted by each depth's *true speech*:
  `10*log10( sum_k T_k / sum_k (T_k/r_k) )`. Being a weighted mediant of the per-depth ratios it is
  **provably bounded** by the best and worst of them, and invariant to a per-region gain. This is
  the number to compare sweep points on.
  *Weights come from the TARGET, never the estimate.* The first implementation pooled projection
  energies and a test caught it immediately: that lets a region the extractor happens to output
  loudly pull the pooled number toward its own score, reintroducing the exact level sensitivity the
  function exists to remove, one level up.
* **`depth_scale_factors` -> `level_error_db`.** `20*log10(max alpha / min alpha)` across depths.
  Verified to recover a known gain exactly (9.54 / 20.00 / 40.00 / 60.00 dB for 3x/10x/100x/1000x)
  while `si_sdr_pooled` stays bit-stable. This column is deliberately **not** put through
  `clip_score`: its +-50 dB cap exists to tame `+-inf` SI-SDR, and truncating a 60 dB level error to
  50 would be a silently wrong diagnostic in the one column added to stop level error hiding.

`SCORE_FIELDS` and `GATE_FIELDS` are **untouched** and `run_phase2.py` is unmodified, so Phase 2
CSVs stay byte-identical. Old CSVs (every run before 2026-08-23) lack the new columns; the loader
**drops** the key rather than defaulting it -- a default would make "never measured" look like
"measured as zero" -- and both the `.md` section and the gap table say the run predates the column
instead of rendering an empty table. Pinned by `tests/phase3/test_pooled_depth_metric.py` (13 tests,
including a characterization of the level-dominated behaviour so it can never again be mistaken for
a bug) plus back-compat and no-clip tests in `test_overall_metric.py`.

The false claim that started it is corrected at source: `test_overall_metric.py`'s docstring used to
say the whole-track number "genuinely pools the depths". It does not, and reading it that way is
what let this ship.

#### Test D: `V_i` fires, and the rate is the point

6 of 81 decisions = **7.4%**, on ordinary honest enrollment, max `0.000144` against the `1e-4`
threshold. The dev sweep's predicted **false-rejection** rate was 7.9%. Those match, so on this
evidence the firings may be *entirely* false rejections. `vi_on` is unblocked by the notebook's own
criterion (nonzero), but expect it to measure a cost rather than a win -- consistent with what
Session B already predicted for a partial detector (45.3% detection at 7.9% false rejection).

#### Where the four Stage A items stand after this session

| # | item | status |
|---|---|---|
| 1 | gate has never been tuned | **Answered** (Session B). `vi_on` (~5.0 h) bookable; expect a cost. |
| 2 | absolute quality | **Untouched.** Session C. Critical path. |
| 3 | activity masks / dilation | **Mechanism answered** (91% recovery). Operating point was blocked on a defective metric; the metric is now fixed but **the sweep must be re-run to read it**. |
| 4 | refinement ceiling | **Unknown.** Re-run queued, ~1.7 h, unaffected by any of this. |

#### Next actions, in order

1. **Narrow the A5 guard** to parsed-value tolerance. The diagnosis is in; the condition the
   previous note attached to this ("apply only once the diagnostic says which case") is satisfied.
2. **B2 ceiling re-run (~1.7 h).** Cheapest, answers the only genuinely open question of the four,
   and depends on none of the above.
3. **`dilation_v2` (~6.7 h at limit 25) now has a metric that can decide it.** Read
   `si_sdr_pooled`, not `si_sdr`. Watch `level_error_db` in the same table: if it is large, the
   extractor has a level problem worth fixing before any of this is tuned further.
4. **`vi_on` (~5.0 h at limit 50).**
5. **Fix the notebook's Test B guard** if it is reused: drop the `+inf` precondition (require two
   finite depths instead) and print `checked` before the verdict. The general property to assert is
   the bound on `si_sdr_pooled`; the whole-track number has no such bound and must not be tested as
   though it does.

#### Still open, unchanged by this session

* **Session C's training run**, blocked on two decisions: (a) do training masks come from oracle +
  `mask_augment`, or from real pyannote output cached once in a CPU session? (b) warm-start from
  clip50, or scratch? `mask_augment.py` is written and unit-tested but has never touched a training
  run, and its motivation is partly undercut by dilation -- augmentation makes `G` robust to bad
  masks, while dilation makes the masks good.
* **A `CONTEXT.md` glossary** was requested and still does not exist. ADR candidates: §1's
  no-residual rule, eval-encoder-not-training-encoder, the mask-source question, and now
  **which-slice-a-metric-scores**, which has caused three defects (the ceiling's objective, this
  one, and Test B's guard).
* **The Session C memory constraint:** `build_scene_crop_dataset._prepare` keeps each whole scene
  resident at ~35 bytes/sample, so a 2-minute scene is ~34 MB. 800 scenes is 27 GB (Kaggle's
  ceiling) and Phase 2's 2400-scene curriculum count would need **81 GB**. The model only ever sees
  4 s crops, so long scenes buy realistic enrollment and overlap density, not longer inputs.

### ☐ Phase 4 — Real corpora + full ablation

**Goal:** the results section.

**Build:** evaluate on **AMI-SDM, AliMeeting, NOTSOFAR-1** (and optionally DiPCo). Run the full
6-way ablation (blind / diarization-only / TSE-no-recursion / ungated / gated / coarse-to-fine).

**How to check it worked:** the six-system ordering holds on real data; WER (Whisper) improves
for the proposed system.

**Definition of done:** all tables + the depth plot reproduce with one script per experiment.

---

## 6. Guardrails Claude must respect (and you should watch for)

1. **No residual in the audio path.** (See §1. This is the whole thesis.)
2. **Oracle diarization first, always.** Never report a real-diarization number without the
   oracle number beside it — otherwise you can't tell if a failure is the diarizer, `φ`, or `G`.
3. **Eval encoder ≠ training encoder.** Non-negotiable for speaker metrics.
4. **Stratify — and on the right axis.** Never report an aggregate average. But note (2026-08-04)
   that **overlap depth is the intrinsic-difficulty axis, not the evidence axis**: it hits all four
   systems equally and buried the accumulation effect for five runs. The headline axis is
   **accumulation** (`m` / `n_accepted_before`). Report both; claim on accumulation.
5. **Keep the noise term in the recon loss** (or train noise-free). Don't let the losses fight.
6. **One phase at a time.** Green "definition of done" before proceeding.
7. **Add a unit test that fails if any output tensor was produced from a residual.** Cheap
   insurance against the one mistake that would silently invalidate the paper.

---

## 7. Conventions & housekeeping

- **License:** Apache-2.0. Put `LICENSE` and a short `NOTICE` at root; keep a license header
  policy for source files if you want attribution carried downstream.
- **Naming:** keep repo name, package name, and import name identical (`dagger`).
- **Reproducibility:** every result comes from a `scripts/run_phaseN.py` + a `configs/*.yaml`.
  No numbers that can't be regenerated by one command.
- **Sample rate:** dev at 8 kHz (fast) on WSJ0/LibriMix; 16 kHz for real corpora + Whisper.
- **Commit discipline:** small commits per module; tests green before merge.
- **Statistical reporting.** Four idioms drifted in undefined (`std`, `p5/p95`, `sigma`, `SEM`);
  they are not interchangeable, and this is the committed definition of which to use when.
  Full one-line glossary in `docs/research-glossary.md` § "Statistics & reporting" — but that file
  is **gitignored personal notes**, so anything the paper depends on lives here.
  - **Always carry `n`** beside a mean. Phase 2's first fig1 drew an **n=3** point as its headline
    trend; nothing on the chart said so.
  - **`sd`** = spread of individual scenes (describes the data). **`SEM` = `sd/√n`** = precision of
    the *mean* (shrinks with more data). At n=150, sd 3.9 dB → SEM 0.32 dB, ~12× tighter.
  - **`p5/p95` band** answers "how consistent?"; **SEM** answers "how well do we know the average?"
    They differ by ~30× on this data. **Never plot both unlabeled** — a reader assumes the smaller
    one is the error bar. (This is the trap in the Phase 2 close-out's open min/max-band TODO:
    that TODO is about spread; the figure error bars are about precision. Both belong; label them.)
  - **Prefer paired differences** on matched `(scene, speaker, depth)` rows over subtracting two
    means — pairing cancels scene difficulty *exactly* instead of hoping it averages out. This is
    what rescued Phase 2's fig2 when the `no_recursion` control stopped being flat.
  - **Report the win rate beside the mean.** A positive mean at a ~50% win rate is a few large
    wins, not broad superiority — that was literally Phase 1's result.
  - **Near-perfect SI-SDR is FRAGILE, and dB over-weights damage to it.** At 47 dB the error energy
    is ~1e-4 of signal, so corrupting ~1% of samples multiplies it a hundredfold and costs ~20 dB —
    while 47 dB and 25 dB are equally inaudible. Depth-1 (solo-copy) rows live in exactly this
    regime. A large dB drop there is not automatically a large harm; say so rather than letting the
    number speak. Stage B's dilation sweep is the worked case.
  - **Which SLICE you score decides what SI-SDR rewards, because it is scale-invariant.** SI-SDR
    fits a scalar before measuring the residual, and the samples you include determine that scalar.
    Score the whole output and a bit-exact solo copy pins it near 1, so a *level* error in the
    overlap region costs full price; score the overlap slice alone and the scalar floats and
    absorbs that error for free. The two rank estimates differently — by 8 and 20 dB in opposite
    directions on Stage B's test fixture. **Any rule that selects, gates, or optimizes must score
    the same slice the claim is reported on.** Getting this wrong voided Stage B's refinement
    ceiling; `dagger/refine/oracle_ceiling.py` documents the failure.
  - **The un-stratified number exists now, and there are TWO of them — use the right one.**
    Stratify-only leaves "is dilating net better?" unanswerable, since the gain and the cost land
    in different rows with no exchange rate between them; §6.4 forbids an aggregate *instead of*
    stratification, not alongside it. But the obvious implementation is a trap, and we fell in it
    (2026-08-23):
    - **`si_sdr_pooled`** is the **exchange rate**. It fits the scale *per depth*, then pools error
      energies weighted by each depth's true speech. It is provably bounded by the best and worst
      depth and invariant to a per-region gain. **Compare configurations on this.**
    - **`si_sdr`** over the whole track fits ONE scale, which the near-exact solo copy pins near 1 —
      so a pure *level* error in the overlap region is charged at full price while every per-depth
      row discounts it. It landed **below every depth it appeared to summarise in 271 of 288 rows**
      and correlated **−0.21** with depth 1. Keep it: it is the only score here that can see a level
      error at all. Never read it as a summary of the per-depth tables.
    - **`level_error_db`** is that level disagreement, made explicit. Every SI-SDR in this project
      is scale-invariant, so a systematic level error is invisible to all of them and shows up only
      as a discrepancy between the two numbers above.
  - **Weight a pooled metric by the TARGET, never by the estimate.** Weighting by the estimate's
    energy lets a region the system happens to output loudly pull the pooled score toward its own
    value — reintroducing level sensitivity one level up. The target is ground truth; how much real
    speech sits at each depth is a property of the scene, not of the system.
  - **A guard that verifies zero rows is not a passing guard.** Test B (2026-08-23) required a
    `+inf` depth row that this corpus never produces, skipped all 288 rows, and printed PASS while
    the property it checked was violated in 271 of them. **Always print the count you verified, and
    assert it is nonzero.**
  - **`|t| ≳ 2`** as a reading heuristic for "unlikely to be chance." Effect size and significance
    are separate questions and both get reported: Phase 2's accumulation decline is solidly
    non-chance (`t = −4.5`) *and* 3× smaller than the prior checkpoint's.

---

## 8. If you're unsure

- **Why does a module exist?** → `docs/diarization_full_mathematical_theory.pdf`, matched section numbers.
- **Is this change safe?** → re-check §1 and §2. If it touches the audio path or the loss,
  be extra careful.
- **Numbers look too good?** → suspect ground-truth leakage or metric-encoder reuse first.
- **What does this term mean?** → `docs/research-glossary.md` (jargon, one line each, with code
  pointers) and `docs/research-practices.md` (*why* research works this way). Both are gitignored
  personal notes — convenience, not source of truth. Anything a claim rests on belongs in this
  file: the statistical reporting rules are §7, the settled maths is §2.

---

*Last updated: 2026-08-20 — Phase 3 Stage B **Session B** landed, plus its follow-up probe (see
§5 Phase 3; suite **440 passed / 1 skipped**). Four results, two of which overturn standing
conclusions. (1) **Overlap dilation recovers 91% of the oracle-vs-real gap at depth 2**
(-2.98 → **-0.28 dB**, 52% win rate against oracle diarization) with no retraining — and the oracle
arm priced §2's "copy, don't separate" at **43.9 dB** for the first time. (2) **`V_i` works**:
J = +0.373 at a **1e-4** threshold, so four prior "structurally dead" conclusions were a threshold
error — the shipped `0.05` sits 500× above the usable range. (3) **`tau_margin` is NOT a detector**
(J = +0.046), and the conditioning probe confirmed that verdict rather than excusing it: clip50
steers at 9.41 dB, so the fault fixture was valid. Phase 2's gated-vs-ungated comparison is
weakened accordingly. (4) The **refinement ceiling was mis-specified** — it scored the whole
waveform while the table reported depth 2 — so its number is void and the headroom question is
*unknown*; fixed and queued for re-run. The probe also produced the phase's most useful synthesis:
**every check that embeds `G`'s output is gated on `G`'s quality**, which explains the dead margin,
the rubber-stamping gate and net-harmful refinement as one root cause, and predicts they recover
together only when the extractor does — so Session C's training budget, not a gate redesign, is the
lever. Stage A's headline is unchanged and now explained. Phase 1 and Phase 2 DoDs remain met.

**2026-08-23 — Stage B VERIFICATION PASS complete** (see "STAGE B — VERIFICATION PASS" in §5;
suite now **471 passed / 1 skipped**). All five checks ran. **Test E is closed as benign**: values
bit-identical on all 5400 rows, the byte delta is exactly line endings (5401 CR + 1 LF = 5402), so
the A5 guard needs narrowing to a value tolerance, not abandoning. **Test B passed vacuously** — its
guard required a `+inf` depth row this corpus never produces, so it verified 0 of 288 rows — and it
was hiding a real defect: the un-stratified `si_sdr` added in `9f62f4b` is **level-dominated**, sat
below every depth it appeared to summarise in **271 of 288 rows**, and correlated **−0.21** with
depth 1. Root cause is not a wiring bug but the one-scale-per-slice property in §7: reproduced by
holding an estimate's shape fixed and moving only its overlap gain, which leaves every per-depth
score bit-identical while the whole-track number falls 13 dB. **Fixed**: `si_sdr_pooled` (the
bounded, gain-invariant exchange rate — compare configurations on this) and `level_error_db` (the
extractor level error, which every scale-invariant metric in this project was structurally unable
to see). Phase 2 CSVs are unaffected — `SCORE_FIELDS`/`GATE_FIELDS` untouched, `run_phase2.py`
unmodified. **`V_i` fires** at 1e-4 (6/81), but at ~the predicted false-rejection rate, so `vi_on`
likely measures a cost. Of the four Stage A items: 1 answered, 2 untouched (Session C, critical
path), 3 mechanism answered but the operating point needs the dilation sweep RE-RUN to read the new
metric, 4 still unknown (B2, ~1.7 h, cheapest and unblocked).

Per-phase history lives in §5, not here — this footer is deliberately kept to a few lines so it
cannot drift out of sync with it.*
