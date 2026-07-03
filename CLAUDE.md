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
| Speaker encoder `φ` | **WeSpeaker** toolkit, **ReDimNet2** encoder | Freeze pretrained weights first; fine-tune late. |
| **Eval-only** encoder | Different model (WavLM-based / **Kiwano**) | MUST differ from `φ`, or we cheat on speaker metrics. |
| Extractor `G` | **TF-GridNet + cross-attention fusion** (USEF-TSE recipe), via **WeSep** | Conv-TasNet+FiLM = fast baseline. |
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

### ☐ Phase 1 — Identity conditioning (validates: targeting beats blind separation)

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

*Last updated: keep this line current whenever the plan changes.*
