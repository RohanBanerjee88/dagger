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

**Worked example — five people talking at once, and what the two designs actually do.**
Take one 5-speaker scene and follow the *fifth* speaker the system processes:

| | deflation (the anti-pattern) | ours |
|---|---|---|
| what `G` is fed | `x_O − ŝ₁ − ŝ₂ − ŝ₃ − ŝ₄` — four estimates' errors baked in | `x_O`, untouched |
| measured SI-SDR | **−6.78 dB** | **−4.97 dB** |

Both numbers are from the same file, same scene, same checkpoint
(`phase2_librimix_5spk_scratch345clip50.csv`, m=5 at depth 5, n=150 per level): a speaker with
four prior subtractions scores **1.81 dB worse** than the same system's speaker with none. Ours has
no such gradient to measure, because nothing is ever subtracted.

---

## 2. Facts that are mathematically settled (do not "re-derive" and break them)

These are proven in `docs/diarization_full_mathematical_theory.pdf`. Treat them as ground truth.

- **Solo regions are clean.** Where only speaker *i* is active, `x = s_i + n`. So a solo clip
  is a valid enrollment sample. ✔
  *Example:* in a 2-minute 3-speaker scene each speaker gets ~45 s alone; enrollment takes the top
  3 such clips (≥500 ms each) and averages their embeddings into `ē_i`.
- **Copy, don't separate, on solo regions.** Running a network on already-clean audio only adds
  artifacts. Solo parts are copied straight through. ✔
  *Example, and this one is now PRICED:* forcing `G` over already-clean audio by dilating oracle
  masks to 800 ms drops depth 1 from **55.34 dB to 3.16 dB** — a **52 dB** penalty for separating
  what you could have copied (`phase3_librimix_3spk_dilation_v2.csv`, oracle/`no_recursion`).
- **Error accumulation is NOT "monotone."** An earlier draft claimed error "grows monotonically
  every step." **That's false.** The correct statement is three regimes: worst-case *linear*,
  independent-errors *√m*, realistic-correlated *linear*. Never reintroduce the "monotone" claim. ✔
- **The reconstruction loss needs a noise term.** `‖x_O − Σ ŝ_i‖²` alone fights the separation
  loss whenever noise ≠ 0. Use `‖x_O − Σ ŝ_i − n̂‖²` (a noise head) OR train on noise-free data. ✔
- **Leakage uses a MARGIN, not raw similarity.** Raw `cos(ŝ_i, e_j)` is always positive (voices
  aren't orthogonal). Use the margin `M_i = cos(ŝ_i, e_i) − max_{j≠i} cos(ŝ_i, e_j)`. ✔
  *Example of the formula being right and still not working:* on the dev sweep, correct
  conditioning scored **0.43873** and deliberately swapped conditioning **0.41965** — a 0.019 gap
  against a 0.1 threshold. The margin is sound; at `G`'s current ~2 dB the same distortion
  contaminates both cosines, so it separates nothing.
  *Now MEASURED rather than argued (2026-08-25):* on the **clean source** the same contrast gives
  **0.55680 vs 0.31409** — a 0.243 gap, J = **+0.453**, 0.0% false rejection up to τ=0.2. So the
  formula discriminates fine; only `G`'s output starves it. See Stage B Session 3 Q1b.
- **The gate can't check its own enrollment.** If enrollment is contaminated, the confidence
  score happily passes it. Guard it *before* the gate with the enrollment-variance check `V_i`. ✔
  *Example:* under real diarization, `V_i` rejected **156 of 1,350** enrollments (11.6%) at the
  tuned 1e-4 threshold and cost nothing — +0.012 to +0.235 dB
  (`experiment_stage_B_run_3/vi_on/`). Under *oracle* regions it is structurally 0 and fires never.
- **Soft masks at seams.** Hard on/off masks click and starve the network of context. Use smooth
  crossfaded windows (`w_Ei + w_Oi = 1`). ✔
  *Example, and the sting in its tail:* `fade_ms: 5` puts a 40-sample ramp inside the depth-1
  region, so an oracle solo copy scores 38–88 dB rather than `+inf`. Assuming otherwise produced a
  false Phase 0 claim and a verification check that verified 0 of 288 rows (2026-08-23).

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
├── journal/                  # per-phase working record (how each verdict was reached)
│   ├── phase1.md
│   ├── phase2.md
│   └── phase3.md
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

*The DoD criterion is met, and later runs measure it continuously.* **Worked example** — take one
speaker in one 2-minute scene of `phase3_librimix_3spk_long2min.csv`: about 45 s of it is
solo, and the pipeline copies those samples straight from the mixture. Scored on their own, those
depth-1 samples come back at **47.62 dB** mean (oracle arm, `no_recursion`, n=150) — error energy
about 1/58,000 of the speaker's own signal. That is the Phase 0 guarantee being re-verified on real
data every time a later phase runs, which is stronger evidence than a one-off Phase 0 table.

> **CORRECTED 2026-08-25.** This paragraph previously claimed the oracle arm produced **792 `+inf`
> rows**. It does not: that file has **zero** `+inf` rows in the oracle arm, and 96 in total, all of
> them in the three *real*-diarization arms. The reason is the crossfade — `fade_ms: 5` puts a ramp
> inside the depth-1 region, so an oracle solo copy is near-exact but not bit-exact and scores
> 38–88 dB instead of `+inf`. Read 47.62 dB as the evidence, not an `+inf` count. The same wrong
> assumption cost a whole verification check later (Test B, 2026-08-23, which required a `+inf` row
> that this corpus never produces and so verified 0 of 288 rows while printing PASS).

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


> **Full working record: [`journal/phase1.md`](journal/phase1.md)** — the `stagger_offsets` starvation bug, the degenerate-loss diagnosis, the passthrough collapse and the lr/grad-clip fix that resolved it. Read it before retraining `G` or changing the loss.

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


> **Full working record: [`journal/phase2.md`](journal/phase2.md)** — five runs, the depth-vs-accumulation axis correction, the refinement negative result, the grad-clip investigation, and the reporting-code hardening. Read it before quoting any accumulation number or regenerating the figures.

### ☐ Phase 3 — Real diarization + robustness

> #### WHERE PHASE 3 STANDS (2026-08-25) — read this before anything below it
>
> Phase 3 opened five questions. Everything in this section is the working record of answering
> them; this table is the answer. **2 closed, 3 open.**
>
> | # | question | status | evidence | what's left |
> |---|---|---|---|---|
> | 1 | **Confidence gate** | 🟠 **4 of 4 DIAGNOSED, 1 of 4 tuned** | `V_i` tuned (J=+0.373 @1e-4), 196 firings, costs nothing. Margin: sound but starved (+0.453 clean vs +0.046 on `G`). **VAD: same verdict** — J **+0.680** on clean audio vs **+0.000** on `G`, recomputed from a committed CSV with no new run. **Artifact: DEFECT** — `spectral_flatness` averaged silent frames (which score exactly 1.0), so it reported each speaker's duty cycle, not artifacts: clean 0.7376 vs `G` 0.7422, and `max_artifact_score: 0.9` sits **above pure white noise (0.847)** | run `phase3_gate_faults.yaml` to place the two thresholds; three of the four now trace back to `G`, i.e. to Q2 |
> | 2 | **Absolute quality** | 🔴 **UNTOUCHED — critical path** | oracle depth 2 = **1.69 dB**, unmoved since Stage A; ~13% of Phase 1's per-depth exposure | Stage C retrain, now also landing the **three unimplemented loss terms** (`L_recon`, `L_spk`, `L_art`) — see `journal/phase3.md` |
> | 3 | **Solo/overlap masks** | ✅ **CLOSED** | **400 ms**, an interior optimum on `si_sdr_pooled`. At 50 scenes the gap goes **-3.18 -> -1.28 dB**, win 9% -> 23%: **60% of the diarization cost recovered** | make it the default; re-report Stage A under it |
> | 4 | **Refinement** | ✅ **CLOSED — both axes** | rule axis <= **+0.18 dB**; perfect candidate + open gate = **+0.002 dB** | nothing; `rounds: 0` final |
> | 5 | **Output level** | 🟠 **defect confirmed, ROOT CAUSE FOUND** | `G` emits overlap at **2.86x** (alpha_2 median 3.16, max 19.5); replicates 8.88 / 8.78 / **9.02 dB** at 3 / 25 / 50 scenes. Cause: `L_recon` is the only loss term that constrains level and it was never implemented | **folds into Q2** — Stage C adds `L_recon`; no inference-time patch (they all smuggle in a corpus assumption) |
>
> *Q1's two new diagnoses are falsifiable too, and both predictions are on record before the run
> (`journal/phase3.md` § Stage B Session 4): with the energy gate on, healthy `artifact_score` should fall
> **0.742 -> ~0.35-0.45** — if it does not move, the duty-cycle diagnosis is wrong — and the
> `fault_clean_` arm should separate where the `fault_g_` arm does not.*
>
> *Two of the closures are NEGATIVE results with stated mechanisms, which is the stronger kind.*
> Q4: refinement's premise was wrong, not its implementation — it assumed the extracted overlap
> beats "the raw mixture", but enrollment never used the mixture, it used the already-clean SOLO
> region. Q1's margin: the formula was never broken, and that closure is **falsifiable** — re-run
> the clean-margin probe after Stage C and the extracted gap should migrate 0.019 -> 0.243.
>
> *Q5 is not blocked on GPU, it is blocked on the estimator.* `match_level_to_mixture` computes the
> MMSE/Wiener gain, which deliberately attenuates (0.6x at 1.7 dB SNR), and its `MAX_RESCALE = 8`
> clamp refuses precisely the speakers whose error is worst.

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


> **Full working record: [`journal/phase3.md`](journal/phase3.md)** — Stage A's oracle-vs-real gap, Stage B Sessions A/B/1/2/3/4, the verification pass, the level-error discovery, and the two void runs with their post-mortems. Read it before touching the gate, the dilation default, the level fix, or Stage C.

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
- **EARNED VALUES ARE NOT DEFAULTS — every new config must state these three explicitly.**
  The shipped defaults have each been *disproven* by a run in this repo, and none of them was
  changed, deliberately: editing a default silently re-scores every committed result that inherited
  it. The cost of that choice is that a config written from an old template inherits a value we have
  measured to be wrong, and nothing complains.
  | key | inherited default | what a run measured | set it to |
  |---|---|---|---|
  | `dilate_overlap_ms` | `0` | interior optimum on `si_sdr_pooled`; recovers 60% of the diarization gap | **`400`** |
  | `max_mean_variance` | `0.05` | 500x above `V_i`'s entire usable range; at 1e-4 it scores J=+0.373 and costs nothing | **`1e-4`** |
  | `artifact_min_energy_db` | absent (`None`) | ungated, `spectral_flatness` reports duty cycle: clean 0.7376 vs `G` 0.7422 | **`-40.0`**, or `null` *deliberately* |
  The third is enforced rather than trusted: `scripts/tune_gate.py` refuses to recommend a
  `max_artifact_score` unless the config states `artifact_min_energy_db`, because a threshold tuned
  with the gate on is a threshold on a *different measurement* and the two must always travel
  together. `null` is an accepted answer — the requirement is a decision on the record, not a
  particular value. The other two have no such enforcement yet and rely on this table.
- **Sample rate:** dev at 8 kHz (fast) on WSJ0/LibriMix; 16 kHz for real corpora + Whisper.
- **Commit discipline:** small commits per module; tests green before merge.
- **Results layout.** `run_phaseN.py` writes its five files FLAT into `eval.results_dir`; the
  filing is manual curation applied afterwards, and it is the same shape in Phase 2 and Phase 3:
  `results/phaseN/<run>/<experiment>/{numbers_csv,numbers_md_docs}/` (plus `graphs/` where figures
  exist). One sub-folder per experiment, and each experiment's aggregate `phase3_gap_*.md` lives
  with the experiment it aggregates — not in a separate `aggregate/` folder, which separates a
  number from the rows it came from. Two consequences worth remembering: a re-run drops flat files
  that need re-filing, and **moving a results file breaks any path quoted in this document** — grep
  for the old path when you re-file (the Stage A reproduce block broke exactly this way on
  2026-08-23). Notebooks that read a committed CSV as a baseline hold these paths too.
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
  - **A scale-invariant metric cannot see a level error, and this project's are ALL
    scale-invariant.** `G` was measured emitting the overlap region at **2.86x** the true amplitude
    (2026-08-23) after three phases in which no number could have revealed it — SI-SDR fits the
    scale out before measuring error, so every per-depth score reads identically at 1x, 3x, 10x and
    1000x. Root cause was the training objective: `si_sdr_loss` is scale-invariant, so nothing ever
    constrained the output level. **A defect invisible to the whole measurement suite does not
    appear as a bad number — it appears as no number at all**, and surfaces only as a disagreement
    between two metrics with *different* scale behaviour. Keep `level_error_db` reported for exactly
    that reason, and be suspicious of any quantity no committed metric could contradict.
  - **Every number in this file must be re-derivable from a committed CSV.** Audited 2026-08-25
    against `results/`: Phase 1's 4.40/2.05/+2.35, Phase 2's 18/18 ordering, its -4.97→-6.78
    accumulation curve, its +0.21 terminal uptick, its 0-variance-rejections-in-5400, and Stage A's
    DER 0.113 / recall 0.758 / -3.11 dB all reproduce **exactly**. Two did not: the Phase 0
    "792 `+inf` rows" claim (actual: **0** in the oracle arm) and the refinement accept rates
    (46.2/34.0/26.8 → **46.0/33.8/26.7**). Both are corrected in place. The lesson is the first
    one's shape: it was a *plausible* number nobody could have checked without opening the file,
    and it then propagated into a test guard that silently verified nothing.
  - **A guard that verifies zero rows is not a passing guard.** Test B (2026-08-23) required a
    `+inf` depth row that this corpus never produces, skipped all 288 rows, and printed PASS while
    the property it checked was violated in 271 of them. **Always print the count you verified, and
    assert it is nonzero.**
  - **`|t| ≳ 2`** as a reading heuristic for "unlikely to be chance." Effect size and significance
    are separate questions and both get reported: Phase 2's accumulation decline is solidly
    non-chance (`t = −4.5`) *and* 3× smaller than the prior checkpoint's.

---

## 8. If you're unsure

- **How did we get to this number?** → `journal/phase{1,2,3}.md`, the full working record per phase.
  §5 carries each phase's verdict and status table; the journal carries how it was reached — the
  failed runs, the void runs, and the diagnoses. Read the relevant one before changing anything that
  phase measured.
- **Why does a module exist?** → `docs/diarization_full_mathematical_theory.pdf`, matched section numbers.
  Note this file and the glossary below are **gitignored and local-only** — a fresh clone does not
  have them, so nothing a claim rests on may live there.
- **Is this change safe?** → re-check §1 and §2. If it touches the audio path or the loss,
  be extra careful.
- **Numbers look too good?** → suspect ground-truth leakage or metric-encoder reuse first.
- **What does this term mean?** → `docs/research-glossary.md` (jargon, one line each, with code
  pointers) and `docs/research-practices.md` (*why* research works this way). Both are gitignored
  personal notes — convenience, not source of truth. Anything a claim rests on belongs in this
  file: the statistical reporting rules are §7, the settled maths is §2.


---

## 9. Defects this project has shipped (read this before trusting a result)

Extracted 2026-08-25 from the phase journals, which is where each of these was diagnosed. They are
here because a lesson that prevents a mistake has to be in the file that gets read, not behind a
pointer. §7 covers the *statistical* half (which slice, scale-invariance, zero-row guards); this
section is the *engineering* half.

**Every defect below has the same signature: a plausible number, and nothing failing.** Not one
announced itself as an error. That is the thing to be paranoid about — a run that crashes costs an
afternoon, a run that returns a believable wrong number costs a conclusion.

- **A flag can be read, validated, warned about — and never forwarded.** `run_phase3.py`'s `main()`
  computed `refine.oracle_audio` and `extractor.rescale_to_mixture`, printed guidance about them,
  and passed neither to the scorer. Both silently defaulted to `False`, and **~3.4 h of GPU**
  produced output bit-identical to a plain run, read as a result (2026-08-24).
  *Why nothing caught it:* every test drove `score_scene` directly, so the **config-to-call-site
  wiring was covered nowhere**. Unit-testing the function you wrote does not test the path the
  program takes to it. Guarded now by
  `tests/phase3/test_run_phase3_arms.py::TestConfigFlagsReachScoreScene`.
  **When adding a flag, add a run that would visibly differ, and assert it does.**

- **"It never fires" is not evidence that a check is useless.** `V_i` was declared "structurally
  dead" **four separate times** on the observation that it never crossed `max_mean_variance: 0.05`.
  The threshold was **500x above its entire usable range**; at 1e-4 it scores J = +0.373 and costs
  nothing. Each observation was right and each inference was wrong.
  **A check that never fires needs a fault fixture, not a conclusion.** `min_vad_coverage` (0
  rejections in 10,950 decisions) and `max_artifact_score` (45) are in exactly that position today
  and are still unexamined.

- **Measuring one point on an axis and calling it a property of the design.** Twice:
  refinement was declared net-harmful after four regimes that all varied ENROLLMENT quality at one
  fixed, poor extractor; `tau_margin` was declared "not a detector" from J = +0.046 measured on
  `G`'s ~2 dB output. Both collapsed when the unswept axis was finally swept — the margin scores
  **J = +0.453** on clean audio.
  **Before writing "X does not work", name the axes X depends on and say which ones you varied.**

- **A test that cannot fail is not a passing test.** The oracle-audio test passed vacuously *twice*
  before it was right: first a constant-gain stand-in extractor made `coarse_to_fine` independent of
  its embeddings, then SI-SDR's scale invariance made gain-only steering unobservable. Same shape as
  Test B verifying 0 of 288 rows while printing PASS.
  **Deliberately break the thing under test and confirm the test goes red.** Every guard added since
  has been verified that way.

- **A notebook that overrides a config in-kernel breaks §7 silently.** Stage B run 1 mutated its
  configs to fit a session budget, wrote the mutated copies to `/kaggle/working`, and never brought
  them back. The committed results stopped regenerating from the committed configs, and nothing
  failed. **Vary the config file, or write the effective config back beside the results.**

- **Byte-identity is the wrong guard across environments.** The Phase 2 A5 check failed on hashes
  while all 5400 values were bit-identical — the delta was 5401 CR + 1 LF, i.e. line endings. A byte
  comparison tests the csv dialect, the float repr and the CUDA stack as much as it tests the code.
  **Compare parsed values within a tolerance** (every shared key present, `max |delta| < 1e-3 dB`).

- **The tables were right and the FIGURE was wrong, every time.** Each figure defect shipped here had
  the `.md` carrying `n`, spread and diagnostic counts while the plot carried a bare mean line — an
  **n=3** point drawn as a headline trend, a control that had stopped being flat. The fix was not
  more flags but **safe defaults**: error bars on unless switched off, thin points drawn hollow and
  excluded from trend lines, preconditions computed rather than remembered.

- **Two variables in one column.** Depth measures intrinsic difficulty and hits all four systems
  equally; accumulation is the evidence axis. Plotting against depth buried the effect for **five
  runs** before the axes were separated — and no retraining was involved, the quantity had been in
  the pipeline the whole time, merely never recorded. The same mistake nearly recurred when a
  six-point dilation sweep would have averaged six pipelines into one cell labelled "depth 2".

- **A deferral justified on one axis can be silently wrong on another.** `L_recon`
  (`||x_O - sum s_hat_i - n_hat||^2`) was deliberately deferred, with a written rationale and a
  deadline: LibriMix is an anechoic sum with no noise term, so §2's "or train on noise-free data"
  branch applies, and `dagger/losses/__init__.py` records that the noise head "MUST land before
  Phase 3 trains on real/noisy corpora". **Every word of that is correct** — and it is entirely
  about NOISE. Nobody noticed the same term is also the only one that constrains output LEVEL:
  `L_sep` is scale-invariant and structurally cannot. So `G` drifted to **2.86x** too loud in the
  overlap region and stayed there for three phases (found 2026-08-23, root cause 2026-08-25).
  The deferral note was reviewed repeatedly and never looked wrong, because on its own axis it
  never was. **When deferring a component, list what it does — plural — and say which of those
  jobs the deferral gives up.** A one-axis justification for a multi-job component is the shape
  to distrust.

- **A defect invisible to the whole measurement suite appears as no number at all.** `G` emitted the
  overlap region at **2.86x** the correct amplitude for three phases. Every metric here is
  scale-invariant, so no single number could ever have shown it; it surfaced only as a
  *disagreement* between two metrics with different scale behaviour, which required the second
  metric to exist first. **Be suspicious of any quantity no committed metric could contradict.**

- **A check can be well-named, correctly implemented, and measuring something else entirely.**
  `spectral_flatness` did exactly what its docstring said — geometric over arithmetic mean of the
  magnitude spectrum, averaged across frames — and the arithmetic was right. But a digitally silent
  frame has every bin pinned at the `eps` floor, so geo == arith and it scores exactly **1.0**,
  maximally "artifact-like" while containing nothing. And `crossfade_windows` guarantees
  `w_Ei + w_Oi == activity_i`, so the estimate is **exactly zero wherever speaker `i` is inactive** —
  over half a scheduled 3-speaker scene. For three phases `max_artifact_score` was therefore
  reporting each speaker's DUTY CYCLE. Measured 2026-08-28: the clean source scores **0.7376** and
  `G`'s ~2 dB output **0.7422**, a 0.005 gap, while pure white noise reaches only 0.847 — so the
  shipped threshold of **0.9 sat above white noise** and could not have rejected an estimate that
  was 100% garbage. Note what did NOT catch it: the value was in range, plausible, stable across
  runs, and had a passing unit test (`tone < noise`, which stays true). What caught it was comparing
  the check's reading on *clean* audio against its reading on *bad* audio and finding them equal.
  **For every diagnostic, ask what it reads on a known-good input and a known-bad one; if those two
  numbers are close, the check is not measuring what it is named after.** Its sibling
  `vad_coverage` was immune for one reason worth copying: it takes an `expected_active` mask and
  scores only there.

- **Two populations compared must come out of the same function.** While wiring the gate fixtures,
  `_fault_row` forwarded `artifact_min_energy_db` and `_row` did not, so healthy rows would have
  carried whole-track flatness (~0.74) and every fault the energy-gated value (~0.4). The sweep
  would have reported that *every* corruption LOWERS flatness — including additive noise, which
  provably raises it — and the sign-disagreement guard would have printed a headline finding for
  someone to believe. **Nothing would have failed**, and neither the inertness guard nor the
  non-empty guard is capable of catching it, because a constant offset between two populations is
  indistinguishable from a real effect. The only defence is asking, at every comparison, whether
  both numbers were produced by the same call with the same arguments.

---

*Where to start: §5's per-phase status tables are the current state; `journal/phase{1,2,3}.md` is
how each was reached; §9 is what has gone wrong before. Recency lives in `git log` and in each
journal's tail, not here — this file deliberately carries no dated changelog, because a third copy
of the state is where claims go stale. The footer this replaced (removed 2026-08-25) had grown to
71 lines, still announced itself as "Last updated: 2026-08-20", and contained no number that was
not already in §5 and the journal.*
