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

These are proven in `docs/theory.pdf`. Treat them as ground truth.

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
    └── theory.pdf            # the full proof doc — the why behind every module
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
(`configs/phase1_overfit4_diag.yaml`, lr 3e-4, 600 single-batch epochs) sat at the passthrough
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
with `configs/phase1_librimix_3spk_train.yaml` (`--system proposed|blind`), eval with
`scripts/run_phase1.py --config configs/phase1_librimix_3spk_eval.yaml` (limit 150).

### ☐ Phase 2 — THE money experiment (validates: accumulation-free reconstruction)

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
hypothesis above is right. Gate thresholds in `configs/phase2_librimix_3spk_eval.yaml`
(`tau_margin`, `max_mean_variance`, `min_vad_coverage`, `max_artifact_score`) are still untuned
defaults and haven't been revisited given the above.

**Fine-tune attempt (2026-07-27): checkpoint recovered at epoch 25/30, not the full 30.**
`scripts/train_phase1.py` gained an optional `train.init_checkpoint` key so a run can warm-start
from an existing checkpoint's weights instead of random init; `configs/phase2_librimix_3spk_train_scheduled.yaml`
fine-tunes `checkpoints/phase1/proposed_librimix_3spk.pt` on `dataset.placement: scheduled` scenes
(same 2000-scene/30-epoch scale as the original Phase 1 run, for comparability) to give `G` real
depth-3 exposure. First attempt on Kaggle hit the platform's ~12h (43200s) max execution ceiling
and was SIGKILLed (exit 137) during epoch 30 -- the observed per-epoch cost (~1460s steady-state
+ ~580s one-time setup) put the full 30-epoch plan at ~44,400s, just over the cap. The mid-run
checkpoint at epoch 25 (`checkpoint_every: 5`, overwriting) survived in the canceled version's
output and was used as-is (loss -2.95 there vs. -3.28 at epoch 29, so not far off where it was
headed) rather than re-running with a lower epoch count. Evaluated via
`configs/phase2_librimix_3spk_eval_finetuned.yaml` (`eval.tag: finetuned`, writes
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

**Reporting TODO (not yet implemented): add empirical min/max (or a percentile band), not just the
mean, to the depth-stratified table/plot.** Right now `_write_results` only reports the mean
SI-SDR per system/depth, which hides the spread — a single mean can't distinguish "consistently
mediocre" from "usually great with a few bad scenes." For a finite sample like ours, the
infimum/supremum of the observed scores are just the min/max (no additional subtlety vs. the
general infinite-set concept); in practice a 5th/95th percentile band is more robust to one
freak-bad scene than the raw min. This is a cheap addition (re-aggregate the existing CSV, no new
experiment) and should land in both `scripts/run_phase2.py`'s `_write_results` and
`scripts/plot_phase2_depth.py`. Separately (bigger, not scheduled yet): Theorem 2's `‖E_m‖ ≤ mε`
is a *proven* worst-case bound, not a measurement — a further validation step would be estimating
`ε`/`L` empirically and checking our measured worst case actually falls under what that formula
predicts; that fits Phase 4's full results section better than a Phase 2 bolt-on.

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
4. **Stratify by overlap depth.** That's the evidence, not aggregate averages.
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

---

## 8. If you're unsure

- **Why does a module exist?** → `docs/theory.pdf`, matched section numbers.
- **Is this change safe?** → re-check §1 and §2. If it touches the audio path or the loss,
  be extra careful.
- **Numbers look too good?** → suspect ground-truth leakage or metric-encoder reuse first.

---

*Last updated: 2026-07-29 — Phase 2 fine-tuned-checkpoint result: ordering criterion replicated on
a second, independently fine-tuned checkpoint (coarse_to_fine > gated_deflation > ungated_deflation
at depth 3) and the absolute depth-2/3 floor rose ~1-2 dB for every system (confirms `G` was
undertrained on true depth-3 overlap), but the accumulation-specific gap did NOT sharpen as hoped
-- retraining alone is not the lever. DoD still NOT called; next candidates are pushing to 4-5
speaker overlap depth or the min/max-band reporting TODO (see the Phase 2 status notes above).
Previously: 2026-07-26 — Phase 2 first real run: ordering criterion met but "flat vs sloped" only
weakly/directionally supported; also fixed a real aggregation bug found while reading the results
(±inf silently dropped instead of clipped, understating depth-1). Before that: 2026-07-22 — Phase 2
code + unit tests landed (scene scheduler, depth utility, gate module, deflation baselines,
coarse-to-fine refinement, `si_sdr_by_depth`, `scripts/run_phase2.py`/`plot_phase2_depth.py`; full
suite 194 passed). Before that: 2026-07-13 — PHASE 1 DoD MET: proposed 4.40 dB vs blind 2.05 dB
overlap SI-SDR (3-spk Libri3Mix, oracle diarization, 150 test scenes) after scaled 2000-scene batch
runs; probe steering margin 12.8 dB.*
