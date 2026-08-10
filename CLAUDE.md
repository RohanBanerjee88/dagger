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
| Diarization | **pyannote.audio 4.0.x + `community-1`** | Gated on HF (free). Ungated fallback: NeMo Sortformer. |
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

### ☐ Phase 0 — Plumbing (no learning yet)

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

*Last updated: 2026-08-10 — Phase 2 DoD MET (see the block at the end of §5 Phase 2).
Ordering holds 18/18 across two independent scratch checkpoints; the accumulation
magnitude replicated at 1.81 vs 1.83 dB, so the grad-clip defect was not its cause.
Next: Phase 3 (real diarization). Per-phase history lives in §5, not here — this
footer is deliberately kept to a few lines so it cannot drift out of sync with it.*
