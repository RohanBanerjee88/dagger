# Phase 3 — Real diarization + robustness: the working record

Moved verbatim out of `CLAUDE.md` on 2026-08-25 (relocation only, no edits). The
**five-question status table** stays in CLAUDE.md §5 and is the answer; this is the
working record of getting there — Stage A, Stage B Sessions A/B/1/2/3/4, the
verification pass, and every void run and its post-mortem.

Read before: touching the gate, the dilation default, the level fix, or Stage C.

---

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
*What -3.11 dB is, on one recording:* the same speaker's overlapped seconds go from **+1.73 dB**
(oracle masks) to **-1.38 dB** (pyannote's masks). The mechanism is in the DER split -- 24% of true
overlap frames are called solo, and there the pipeline **copies the raw mixture** into that
speaker's track, so all three voices arrive at full level for those moments.
*Verified against the committed CSV 2026-08-25:* `phase3_librimix_3spk_long2min.csv` gives
oracle 47.62/1.73, real 38.47/-1.38, paired d2 -3.11 +-0.21 at 9% win, and `_diar.csv` gives
DER 0.113 = miss 0.105 + FA 0.000 + confusion 0.008 with overlap_recall 0.758. All exact.

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
python scripts/aggregate_phase3.py \
    results/phase3/experiments/experiment_stage_A_run/long2min/numbers_csv/phase3_librimix_3spk_long2min.csv \
    --out results/phase3/experiments/experiment_stage_A_run/long2min/numbers_md_docs/phase3_gap_long2min.md
```

The four CSVs/`.md`s this produces are now **committed** under
`results/phase3/experiments/experiment_stage_A_run/`, so
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
> **ALL THREE LEVERS ARE NOW CLOSED (2026-08-25).** The ceiling came in positive but tiny
> (+0.14 / +0.18 dB, Session 1), bounding the whole acceptance-rule family. The third lever --
> "better `G` would make the premise hold" -- was tested directly by embedding the candidate from
> the CLEAN SOURCE, i.e. the perfect-extractor limit, and scored **+0.002 dB** (Session 3, Q4b).
> The first two are weight/segment choices over two already-clean embeddings of the same speaker,
> so the same mechanism covers them. `refine.rounds: 0` is final, and the code default now matches.

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
   too high. `tau_margin` scores J = +0.046 ("no usable threshold") ON `G`'S OUTPUT -- but **+0.453
   on clean audio** (Session 3 Q1b), so the formula is sound and merely starved; the conditioning probe
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
  the scheduled long-solo geometry in Stage C.
* **Absolute quality is still the extractor's operating point**, not diarization's -- the *oracle*
  arm reaches only 1.73 dB at depth 2. Nothing in Session A changed training, so nothing here could
  have moved it. It is repaired, if at all, by Stage C's training budget.

---

**STAGE B -- SESSION B RESULT (2026-08-20): dilation recovers 91% of the oracle-vs-real gap at
depth 2 with no retraining; `V_i` WORKS and four prior "it is dead" conclusions were a threshold
error; the refinement ceiling was mis-specified and must be re-run.** All three runs completed in
one Kaggle batch (~9 h). Artifacts in `results/phase3/experiments/experiment_stage_B_run_1/`
(one sub-folder per experiment: `dilation_sweep/`, `dilation_smoke/`, `refine_ceiling/`,
`gate_tune/`, each split into `numbers_csv/` and `numbers_md_docs/`).
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
audio, and `tau_margin`'s J = +0.046 stands as a measurement of this checkpoint.**

> **SCOPE CORRECTED (2026-08-25).** This sentence originally ended "it is not a detector", which
> reads as a verdict on `M_i` itself. It is not one. The probe establishes that the *fixture* was
> valid, not that the *formula* is broken -- those are different claims, and conflating them is the
> same error this file made about refinement. Measured since: on the clean source the margin scores
> J = **+0.453** against J = +0.046 on `G`'s output. The formula is sound and purely starved by the
> extractor. See Stage B Session 3 Q1b.

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
Stage C on training rather than on redesigning the acceptance rule, and it means a gate redesign
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
5. ☒ **DONE (2026-08-24). Set `max_mean_variance: 1e-4`** and re-run a gated comparison -- `V_i`
   fired 156 times in 1,350 decisions (11.6%) and cost **nothing**: +0.012 to +0.235 dB, never
   negative. The prediction below ("expect the gated systems to possibly get *worse*") was WRONG,
   and the reasoning that produced it -- a partial detector's false rejections should cost quality
   on every scene -- does not hold here, because rejecting a refinement candidate or a deflation
   subtraction is cheap when both were near-worthless to begin with. Note this is now the ONLY live check in the gate,
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

7. **Stage C's case is stronger than it was.** The mechanism above says the margin, the gate,
   refinement and the ceiling are all downstream of `G`'s quality, so they cannot be fixed
   independently of it -- and a gate redesign evaluated on this checkpoint would be measuring the
   extractor regardless. Budget arithmetic is in Phase 2's close-out.

---

**STAGE B -- VERIFICATION PASS (2026-08-23): all five checks ran; four pass, and the fifth passed
only vacuously and was hiding a real defect. Test E's byte-identity failure is CLOSED as benign
(line endings, exact arithmetic below). The un-stratified metric added in `9f62f4b` is
LEVEL-DOMINATED and could not have chosen the dilation operating point -- diagnosed, fixed, and the
fix is the first thing in this project that can see an extractor level error at all.**

Artifacts: `results/phase3/experiments/verify_4_questions_run/` (12 files, split
`repro/` and `vion_smoke/`) and the executed
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
| 1 | gate has never been tuned | **`V_i` ANSWERED** (Session B tuned it, Session 2 priced it: FREE, +0.235 dB where it bites). The **margin is NOT** answered -- J=+0.046 is one point on the extractor axis. |
| 2 | absolute quality | **Untouched.** Stage C. Critical path. |
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

* **Stage C's training run**, blocked on two decisions: (a) do training masks come from oracle +
  `mask_augment`, or from real pyannote output cached once in a CPU session? (b) warm-start from
  clip50, or scratch? `mask_augment.py` is written and unit-tested but has never touched a training
  run, and its motivation is partly undercut by dilation -- augmentation makes `G` robust to bad
  masks, while dilation makes the masks good.
* **A `CONTEXT.md` glossary** was requested and still does not exist. ADR candidates: §1's
  no-residual rule, eval-encoder-not-training-encoder, the mask-source question, and now
  **which-slice-a-metric-scores**, which has caused three defects (the ceiling's objective, this
  one, and Test B's guard).
* **The Stage C memory constraint:** `build_scene_crop_dataset._prepare` keeps each whole scene
  resident at ~35 bytes/sample, so a 2-minute scene is ~34 MB. 800 scenes is 27 GB (Kaggle's
  ceiling) and Phase 2's 2400-scene curriculum count would need **81 GB**. The model only ever sees
  4 s crops, so long scenes buy realistic enrollment and overlap density, not longer inputs.


---

**LEVEL-CHECK RUN (2026-08-23, same day, later): `G` EMITS THE OVERLAP REGION 2.86x TOO LOUD. A
systematic gain error that has been present and structurally unmeasurable for three phases, found
the moment two metrics with different scale behaviour sat side by side. `si_sdr_pooled` is
validated against a known result in the same run. Nothing committed moved.**

`levelcheck.ipynb` (repo root, outputs embedded), 3 scenes x 2 arms x 0 ms only = 6 units, ~25 min
including setup. Artifacts under `results/verify/` in the session; suite 475 passed inside it.

#### The three results

**1. `level_error_db` = +9.14 dB median (2.86x amplitude), and it is worse for deflation.**

| arm | system | level error (median) | pooled |
|---|---|---|---|
| oracle | `no_recursion` | 8.88 | 5.25 |
| oracle | `coarse_to_fine` | 7.98 | 4.21 |
| oracle | `gated_deflation` | **11.48** | 3.44 |
| oracle | `ungated_deflation` | **11.48** | 3.44 |

Range 4.58 to 20.68 dB. `gated` and `ungated` are identical to the decimal -- the degenerate gate
again. Read the oracle rows, not `real`: under real regions the scoring depths and the audio-path
masks diverge, so mask error mixes into the level reading.

**2. `si_sdr_pooled` works, and is validated rather than merely plausible.** Bounded in **72 of 72**
rows -- the property Test B verified zero of. Paired `real - oracle` on pooled gives `no_recursion`
**-2.97 dB** against Stage A's depth-2 gap of **-3.11 dB**: a new metric landing within 0.14 dB of
the established headline is strong evidence it measures the right thing. Where the whole-track
number gave -13.17, pooled gives +5.25.

**3. Both A5 guards clean.** Phase 3 per-depth rows: 144 shared, 0 differing, `max |delta| = 0.000e+00`.
Phase 2 (Test E re-run, 25.3 min): 5400 rows, 0 differing, `max |delta| = 0.000e+00`, byte delta
again pure line endings.

#### Root cause: the training objective never constrained the level

`dagger/losses/sisdr.py:20` fits `scale` out before measuring error. **SI-SDR is scale-invariant, so
nothing in training ever told `G` what level to emit at**, and it drifted to 2.86x. Not a bug in the
audio path, and the two candidates that looked likelier were both eliminated:

* **Not the Stage-1 normalize/denormalize round trip.** It is scale-*equivariant*
  (`module(k*x, e) == k*module(x, e)`, pinned by test), so it preserves whatever scale the network
  produces -- it cannot set one.
* **Not the crossfade.** `w_Ei + w_Oi = activity_i` is a partition of unity, and the measured
  `alpha ~ 1` at depth 1 confirms the copy path is unity gain.

*Why it took three phases.* Every metric in this project is scale-invariant, so no single number
could ever have shown it. It is visible only as a DISAGREEMENT between two metrics with different
scale behaviour -- which required the second metric to exist first. A defect invisible to the whole
measurement suite does not show up as a bad number; it shows up as no number at all.

#### What it does and does not cost

* **No committed SI-SDR number is affected.** Every per-depth score is scale-invariant: in the
  local sweep, depth 2 reads `-0.00` at gains of 1x, 3x, 10x, 100x and 1000x alike.
* **What it does cost:** a ~9 dB jump at every solo->overlap seam (audible, ugly), plus anything
  NOT scale-invariant -- a sum-to-mixture check, the noise-head reconstruction loss. The Phase 4
  Whisper-WER concern was overstated in the run's own printed verdict: Whisper normalizes its input.
* ~~**It is self-limiting under dilation.**~~ **WITHDRAWN 2026-08-24.** The prediction was that at
  800 ms nearly everything goes through `G`, the gain becomes uniform, and a uniform gain is just a
  volume knob. Session 1 measured the opposite: `level_error_db` GROWS with dilation
  (8.78 -> 10.33 dB). No mechanism yet. Q3 and Q5 are therefore independent -- dilating does not
  mitigate the level error.
* **`dilation_v2` is NOT blocked** -- the run's canned verdict said it was, and that was wrong.
  `si_sdr_pooled` is invariant to a per-region gain by construction, so the dilation operating point
  read on it is uncontaminated.

#### The Phase 2 implication, stated carefully

Deflation SUBTRACTS each estimate into a running residual. An estimate 2.86x too loud
**over-subtracts**, and the error compounds down the chain -- which is a mechanism for the +2.6 dB
higher level error the deflation systems show. So some fraction of the measured accumulation penalty
may be a calibration artifact rather than intrinsic accumulation.

This does **not** threaten the ordering or CLAUDE.md §1's structural argument: `coarse_to_fine`
never builds a residual and is immune by construction. It does mean the accumulation *magnitude*
has a candidate fixable component, and it is one hypothesis for the recipe-dependent 1.8-vs-5.3 dB
spread already on record. Testable: correct the level, re-measure the `n_accepted_before` curve.

#### Code landed (suite 487 passed / 1 skipped, up from 475)

All three default OFF, so every committed number still reproduces.

* **`dagger/reconstruct/stitch.py::match_level_to_mixture`** + `rescale_to_mixture` on
  `reconstruct_speaker`/`reconstruct_all`/`reconstruct_all_deflation` and `score_scene`. Takes the
  least-squares scalar `c = <g, x_O> / <g, g>` over the emitted region. Uses **only the mixture, no
  ground truth**, so unlike the two `oracle` flags it IS deployable. Exact in the ideal case (the
  other speakers are near-orthogonal to `s_i`, so `c ~ 1/k`), and with real noise it shrinks
  slightly -- the conservative direction. Refuses to act on a non-positive projection (sign flip) or
  beyond `MAX_RESCALE = 8.0` (the scalar is then noise). The deflation path rescales against the
  **running residual**, not `x_O`: that is the mixture each speaker was actually extracted from.
* **`refine_embeddings(candidate_audio=...)`** -- embeds the candidate from the clean sources: the
  perfect-extractor bound. Under it the gate judges the same clean clip, deliberately: judging
  clean candidates with a gate looking at ~2 dB output rejected 100% of them, and the bound would
  then measure the gate, which is `accept_fn`'s job. `run_phase3.py` refuses both oracle flags at
  once for the same reason.
* Configs: `phase3_librimix_3spk_refine_oracle_audio.yaml` (differs from the ceiling config by
  `refine.oracle_audio` and `eval.tag` alone, so the two bounds pair row by row) and
  `phase3_librimix_3spk_rescale.yaml` (`extractor.rescale_to_mixture: true`, `refine.rounds: 0`).
* `tests/phase3/test_level_rescale_and_oracle_audio.py` (12 tests), including bit-identity with the
  flags off and an end-to-end check that `level_error_db` collapses when the rescale is on.

*One test-design note worth keeping.* The oracle-audio test first asserted the OUTPUT changes, and
passed vacuously twice: a constant-gain stand-in extractor made `coarse_to_fine` independent of its
embeddings, and then SI-SDR's scale invariance made gain-only steering unobservable. It now asserts
on the GATE MARGIN (observable whether or not the gate accepts) plus a second end-to-end test with a
permissive gate. Same shape as Test B: an assertion that cannot fail is not a passing assertion.

#### The five open questions after this run

| # | question | state | closes with |
|---|---|---|---|
| 1 | confidence gate | **2 of 4 checks settled.** `V_i` tuned (J=+0.373 @1e-4, free). Margin diagnosed: sound but starved (+0.453 clean vs +0.046 on `G`). **`min_vad_coverage` and `max_artifact_score` NEVER TUNED** -- 0 and 45 rejections in 10,950 decisions, and no fault population exists to sweep them against | fault fixtures for VAD + artifact; margin folds into Q2 |
| 2 | absolute quality | **Untouched.** oracle d2 = 1.73 dB; ~13% of Phase 1's per-depth exposure | Stage C retrain |
| 3 | solo/overlap masks | **Mechanism answered** (91% recovered @800 ms); operating point open, now unblocked -- *ANSWERED 2026-08-24: 400 ms, see Session 1* | `dilation_v2`, 6.7 h, read `si_sdr_pooled` |
| 4 | refinement window | **NEITHER END KNOWN** -- see below | `refine_ceiling` 1.7 h + `refine_oracle_audio` 1.7 h |
| 5 | output level | **Root cause known**, fix implemented, unmeasured | `rescale` config |

*On #4, a correction to how this file has framed it.* "Refinement is net-harmful, -0.07 to -0.54 dB"
is an EFFECT SIZE, not a bound. The question is where refinement *works*, and that window has two
axes: enrollment must be bad enough that improvement is possible, and `G` must be good enough that
the candidate beats the enrollment. All four regimes tested to date -- clean, starved,
heterogeneous, contaminated-real -- varied ENROLLMENT and held extractor quality fixed at one poor
point. **The binding axis has never been swept.** The two bound runs bracket it: `oracle_ceiling`
bounds the acceptance rule, `oracle_audio` bounds the extractor.

#### Session plan

* **Session 1 (~8.9 h):** `refine_ceiling` (1.7 h) + `dilation_v2` (6.7 h). No new code needed.
* **Session 2 (~8.9 h):** `refine_oracle_audio` (1.7 h) + `rescale` (1.7 h) + `vi_on` (5.0 h).
* **Session 3:** Stage C training, for #2.

One notebook with a `SESSION` constant, committed once per session; `levelcheck.ipynb`'s cells 1-4
are the proven setup header. Keep the one-fatal-assertion discipline: the reproduction gate aborts,
everything else diagnoses inline and reports a verdict.


---

**STAGE B -- SESSION 1 (2026-08-24): Q3 ANSWERED (dilate 400 ms, recovering 63% of the
diarization cost) and Q4's ACCEPTANCE-RULE AXIS CLOSED (a perfect rule is worth +0.18 dB, so the
rule is not what is broken). Q5 confirmed at 25-scene scale and worse than predicted. Two runs,
~8.7 h, both from committed configs unmodified.**

Artifacts: `results/phase3/experiments/experiment_stage_B_run_2/{dilation_v2,refine_ceiling}/`,
each split `numbers_csv/` + `numbers_md_docs/` per §7. Checkpoint unchanged (clip50); no training.

#### Q3 -- the dilation operating point is 400 ms, and it is an INTERIOR optimum

`real` / `no_recursion` / **`si_sdr_pooled`** (the exchange rate -- NOT the whole-track `si_sdr`):

| dilate | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|
| `real` pooled | 3.02 | 4.29 | **4.91** | 3.55 |
| `oracle` pooled | 6.03 | 4.73 | 3.63 | 1.76 |
| paired `real - oracle@0ms` | -3.01 +-0.27 | -1.74 +-0.20 | **-1.12 +-0.20** | -2.48 +-0.28 |
| win rate vs oracle@0ms | 9% | 12% | **23%** | 16% |

**63% of the oracle-vs-real gap recovered** (-3.01 -> -1.12 dB) with no retraining.

*On one recording:* pyannote misses ~24% of the overlap frames in a 2-minute scene -- mostly brief
incursions at the edges of a turn. In those moments the pipeline believes the speaker is alone and
**copies the raw mixture into their track**, so you hear all three people at full volume. Widening
the derived overlap mask by 400 ms on each side catches nearly all of them. Push to 800 ms and it
reverses: `G` now runs over genuinely solo audio, and the depth-1 copy falls 41.81 -> 10.27 dB for
less depth-2 gain than that costs.

*Why this supersedes run 1's read.* Run 1 had no pooled metric and its per-depth story pointed at
800 ms; on `si_sdr_pooled` 800 ms is clearly past the peak. The optimum is **interior**, which is
the first time an un-stratified number here has produced one rather than a monotone -- a metric
that only ever says "more is better" is not weighing anything. This is the payoff of the
2026-08-23 metric fix, and it changed the answer.

*Two sanity checks that the mechanism is the intended one.* The `oracle` arm declines
**monotonically** (6.03 -> 1.76): dilating already-perfect masks buys no recall and only pays false
alarm, exactly as it should. And at 400 ms `real` (4.91) **overtakes** `oracle` (3.63) at the same
dilation -- recovery of missed overlap outrunning the over-extraction cost, which can only happen
in the arm that had missed overlap to recover.

Per-depth, `no_recursion`, showing the trade the pooled number is netting:

| | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|
| oracle d1 / d2 | 55.34 / 1.69 | 14.57 / 1.70 | 9.13 / 1.71 | 3.16 / 1.71 |
| real d1 / d2 | 41.81 / -1.29 | 39.24 / 0.02 | 25.95 / 1.02 | 10.27 / 1.41 |

Read the `real` row: depth 2 climbs **-1.29 -> +1.41** while depth 1 falls 41.81 -> 10.27. Neither
column alone answers "is this better"; that is precisely the exchange-rate problem, and pooled nets
them at +1.89 dB in favour of 400 ms.

**Action: `dilate_overlap_ms: 400` becomes the default**, and the Stage A gap should be re-reported
with it.

#### Q4 -- the acceptance-rule axis is CLOSED. A perfect rule is worth +0.18 dB.

Ceiling deficit `coarse_to_fine - no_recursion`, paired, n=75:

| arm | depth 2 | \|t\| | win |
|---|---|---|---|
| oracle | **+0.141** +-0.049 | 2.9 | 25% |
| real | **+0.177** +-0.034 | 5.3 | 41% |

**Positive, as it must be** -- an acceptance rule that only ever commits improvements cannot do
worse than committing nothing, and committing nothing IS `no_recursion`. Run 1's version came in
NEGATIVE, which is what exposed the mis-specified objective (it scored the whole waveform while the
table reported depth 2). The 2026-08-20 fix is confirmed working on real data.

**But the magnitude is the result.** +0.18 dB is the ABSOLUTE MAXIMUM any acceptance rule could
deliver on this extractor, against the deployable gate's measured **-0.5 dB**.

*On one recording:* refinement re-listens to that speaker's ~16 s of extracted overlap, builds a
fresh voiceprint from it, and re-extracts. Give it a rule allowed to peek at the clean source and
commit only genuine improvements, and the speaker's track gets **0.18 dB** better. Play the two
files: you cannot tell them apart. That is the ceiling, not the achievement. And the gate is
barely missing that headroom: `ceiling_accept_gate_would_reject` fired **4 of 150** (oracle) and
**6 of 150** (real) -- about 4%.

*The gate's actual failure is the other direction.* `ceiling_reject_gate_would_accept` fired **78**
and **63** times: the gate waves through a large majority of candidates the ground truth says are
worse. It is too permissive, not too strict -- exactly what a margin scoring J = +0.046 predicts,
and it means raising `tau_margin` is the wrong lever (it would tighten the wrong direction).

**Conclusion: refinement's deficit is NOT an acceptance problem.** The levers Phase 2 listed for a
"positive ceiling the gate cannot find" -- variance-weighted blend, embedding the lowest-depth
frames -- are all rule improvements, and the ceiling says the entire rule axis is worth <= 0.18 dB.
`refine.rounds: 0` stands. **One axis remains: the extractor** (`refine_oracle_audio`, Session 2).

#### Q5 -- level error replicates at scale, and my "self-limiting" prediction was WRONG

Median `level_error_db`, `no_recursion` vs the deflation systems:

| arm / system | 0 ms | 200 ms | 400 ms | 800 ms |
|---|---|---|---|---|
| oracle `no_recursion` | 8.78 | 9.15 | 9.51 | 10.33 |
| oracle `ungated`/`gated` | **13.66** | 14.03 | 14.38 | 15.24 |
| oracle `coarse_to_fine` | 8.20 | 8.57 | 8.86 | 9.70 |

* **Replicates.** 8.78 dB here against 8.88 on the 3-scene probe.
* *What it sounds like on one recording:* for ~45 s the speaker plays at the correct volume,
  because those samples are a verbatim copy of the mixture. Then the overlap starts and the track
  jumps to **2.8x amplitude -- a +9 dB step** -- and drops back when it ends. It sounds like they
  start shouting the instant someone talks over them, twice a minute. For a deflation system the
  step is 4.8x.
* **The deflation gap WIDENED**: +4.88 dB over `no_recursion` at 25 scenes, against +2.60 dB at 3
  scenes. Stronger support for the over-subtraction mechanism -- an estimate 2.86x too loud
  over-subtracts into the residual and the error compounds down the chain.
* **It is NOT self-limiting under dilation.** The LEVEL-CHECK note above predicted the gain would
  become uniform as more of the track goes through `G`, making it a harmless volume knob. It does
  the opposite: 8.78 -> 10.33 dB as dilation grows. That prediction is withdrawn; there is no
  mechanism for the increase yet. It also means dilating to 400 ms does not mitigate Q5, and the
  two fixes are independent.

#### Q1 and Q2 -- one confirming point, and one untouched

* **Q1.** The ceiling run's gate accepted 27/150 (oracle) and 49/150 (real) while the ground-truth
  rule rejected 78 and 63 of what it accepted. Consistent with the dead margin; no new information
  about `V_i`, which `vi_on` still has to price.
* **Q2.** oracle depth 2 = **1.69 dB**, against Stage A's 1.73. Nothing in this session changed
  training, so nothing could have moved it. Note Q4 now routes back here: refinement cannot pay
  until the extractor improves, whatever the acceptance rule.

#### The five questions after Session 1

| # | question | state | closes with |
|---|---|---|---|
| 1 | confidence gate | **2 of 4 settled** (Session 3). `V_i` free; margin sound but starved. **VAD + artifact never tuned** (0 / 45 firings in 10,950) | fault fixtures for the other two |
| 2 | absolute quality | **Untouched.** oracle d2 = 1.69 dB | Stage C retrain -- CRITICAL PATH |
| 3 | solo/overlap masks | **ANSWERED: 400 ms, 63% of the gap recovered** | make it the default; re-report Stage A |
| 4 | refinement window | **Rule axis CLOSED (<= +0.18 dB).** Extractor axis open | `refine_oracle_audio`, 1.7 h |
| 5 | output level | **Confirmed at scale**, worse for deflation, grows with dilation | `rescale`, 1.7 h |

**Session 2 (~8.9 h), all configs already committed:** `refine_oracle_audio` (1.7 h) + `rescale`
(1.7 h) + `vi_on` (5.0 h). *Outcome (2026-08-24): only `vi_on` was valid -- a wiring defect voided
the other two. See "STAGE B -- SESSION 2".*


---

**STAGE B -- SESSION 2 (2026-08-24): TWO OF THE THREE RUNS ARE VOID. A wiring defect in
`run_phase3.py` meant `refine.oracle_audio` and `extractor.rescale_to_mixture` were read from the
config, validated, warned about -- and never passed to the scorer. Q1 is answered and the answer
REVERSES the prediction: switching `V_i` on costs nothing and helps slightly. Q4b and Q5 remain
open.**

Artifacts: `results/phase3/experiments/experiment_stage_B_run_3/{refine_oracle_audio,rescale,vi_on}/`.

#### The defect, and why nothing caught it

`main()` computed both flags and forwarded neither:

```python
score_scene_all_arms(
    ..., refine_oracle_ceiling=refine_oracle_ceiling,   # forwarded
    dilation_failures=dilation_failures,                # <- the other two never appear
)
```

Both defaulted to `False`. Proven, not inferred: the `refine_oracle_audio` run is **bit-identical
on all 300 rows**, `coarse_to_fine` included, to Session 1's plain `dilation_v2` at 0 ms. The
`rescale` run is identical for every system except `coarse_to_fine`, and that differs only because
its config sets `refine.rounds: 0`.

*Everything around the bug was correct*, which is why it survived: the configs were right, the
`match_level_to_mixture` and `candidate_audio` code was right and unit-tested, and `score_scene`
accepted both arguments properly. Only the single call site dropped them. **Every existing test
drove `score_scene` directly**, so the config-to-call wiring was covered nowhere -- the same shape
as Test B's vacuous guard: a plausible result, and nothing failing.

Fixed, plus `tests/phase3/test_run_phase3_arms.py::TestConfigFlagsReachScoreScene`, which asserts
that every flag `main()` reads is also forwarded. Verified to FAIL on the reverted code and pass on
the fix (suite 491 passed / 1 skipped, up from 487).

**Cost: ~3.4 h of GPU that produced no new information.** Re-run both from the same configs; the
configs did not change, the code did.

#### Q1 -- ANSWERED, and it reverses the prediction: `V_i` is free

First run in the project's history with a live `V_i`. 1,350 gate decisions, 50 scenes, 3 arms:

| reason | n | share |
|---|---|---|
| accepted | 962 | 71.3% |
| margin | 232 | 17.2% |
| **enrollment_variance** | **156** | **11.6%** |

`mean_variance` median 4.49e-05, max 3.24e-04, and 156/1350 above the tuned 1e-4 threshold. **This
is the first time `enrollment_variance` has ever appeared as a rejection reason.**

*What it catches, on one recording:* enrollment picks 3 "solo" clips for speaker A to build their
voiceprint. If pyannote's solo region for A actually has a second of B bleeding in, one of those
clips is really A+B -- and the resulting voiceprint is a blend, so `G` then extracts a blend for
A's entire overlap section. `V_i` is the disagreement *between* A's own clips: high variance means
one of them is not the same person as the others. It cannot see this under oracle diarization,
where a speaker's solo region is one contiguous run and yields exactly one clip -- variance over a
single sample is 0 by definition, which is why this took three phases to become measurable.

Paired against the Stage A baseline, depth 2, n=150 per cell:

| arm | `gated_deflation` | `coarse_to_fine` |
|---|---|---|
| oracle | +0.000 | +0.000 |
| real | +0.012 (\|t\| 0.4) | +0.057 (\|t\| 2.3) |
| real_index_order | **+0.235 (\|t\| 3.6)** | +0.057 (\|t\| 2.3) |

**Switching `V_i` on is not a cost.** It is neutral-to-slightly-positive, and the one clearly
non-chance effect (+0.235 dB, |t| 3.6) is in the arm where deflation order is *not* already sorted
by `V_i` -- which is the arm where rejecting a contaminated enrollment actually changes something.

*The controls confirm the plumbing:* `no_recursion` and `ungated_deflation` are **exactly +0.000**
in all three arms. Neither runs the gate, so a threshold change cannot reach them. And the `oracle`
arm is +0.000 for every system, because `V_i` is structurally 0 under oracle regions -- the same
fact that made this untunable for three phases.

**Action: `max_mean_variance: 1e-4` becomes the default.** The earlier prediction in this file --
"expect it to measure a cost rather than a win", reasoned from the 7.9% false-rejection rate -- was
wrong. The false rejections are real (the 11.6% firing rate matches the dev sweep almost exactly)
but they cost nothing measurable, because rejecting a *refinement* candidate or a deflation
subtraction is cheap when both were near-worthless anyway.

#### Q1's other half is NOT answered, and this file has been overstating it

`V_i` is settled. **`tau_margin` is not**, and the distinction matters:

* **`V_i` is not gated on `G`.** It embeds enrollment clips taken from the *mixture*, never `G`'s
  output. Its J = +0.373 is durable across any change to the extractor.
* **The margin is.** Its J = +0.046 was measured with `G` at ~2 dB, where the same distortion
  contaminates both cosines and the margin collapses whatever the extraction. That is one point on
  the extractor axis -- exactly the error already corrected for refinement (Q4). "The margin is not
  a detector" is only established AT THIS CHECKPOINT.

*The cheap way to settle it*, mirroring `refine_oracle_audio`: compute the margin on the **clean
source** instead of `G`'s output, correct vs. swapped conditioning. Separates -> the formula is
sound and purely gated on `G`, so it recovers when Q2 does. Fails to separate -> the margin is
broken as a formula and needs replacing. No training, one `tune_gate.py` variant.

#### Q4b and Q5 -- still open, both void

Nothing was measured. The re-run is the same two commands.

*What Q5's re-run should show, so it can be read rather than eyeballed:* `level_error_db` falling
from ~8.9 dB toward 0; `si_sdr_pooled` **barely moving** (it is gain-invariant by construction, so
a large shift means the rescale is changing SHAPE, which it must not); per-depth `si_sdr` unchanged
(scale-invariant); and the deflation systems gaining most, if over-subtraction is the mechanism.

Note the correction is the least-squares (MMSE) scalar, so it under-corrects slightly rather than
over-corrects -- expect `level_error_db` small but nonzero, not 0. And with the measured `alpha_2`
reaching 19.5x on some speakers, `MAX_RESCALE = 8.0` will decline to act on a few rows rather than
rescale on a noisy projection.

#### The five questions after Session 2

| # | question | state | closes with |
|---|---|---|---|
| 1 | confidence gate | **2 of 4 settled.** `V_i` free, +0.235 dB where it bites -- default 1e-4. Margin sound but starved. **VAD + artifact unexamined** | fault fixtures for VAD + artifact |
| 2 | absolute quality | **Untouched.** oracle d2 = 1.69 dB | Stage C retrain -- CRITICAL PATH |
| 3 | solo/overlap masks | **ANSWERED: 400 ms, 63% of the gap** | make it the default |
| 4 | refinement window | Rule axis CLOSED (<= +0.18 dB). **Extractor axis VOID, not measured** | re-run `refine_oracle_audio` |
| 5 | output level | Measured at 8.78 dB; **the fix is VOID, not measured** | re-run `rescale` |


---

**STAGE B -- SESSION 3, Q1b (2026-08-25): THE MARGIN FORMULA IS SOUND. It was starved by `G`, not
broken -- so no gate redesign is warranted, and Q1's second half folds into Q2.**

First run of the clean-margin probe (`configs/phase3/experiments/phase3_gate_tune_clean_margin.yaml`,
50 dev-clean long scenes, oracle regions, n=150 per population). Artifacts in
`results/phase3/experiments/experiment_stage_B_run_4/`.

| population pair | median gap | best Youden's J |
|---|---|---|
| extracted -- `correct` vs `swapped` (Session B) | 0.43873 vs 0.41965 = **0.019** | **+0.046** |
| clean source -- `clean_correct` vs `clean_swapped` | 0.55680 vs 0.31409 = **0.243** | **+0.453** at tau=0.3 |

False rejection is **0.0%** all the way to tau=0.2. The gap is **13x wider** on clean audio and J is
**10x higher**.

*On one recording:* hand the margin A's clean 16 s of overlap and B's clean 16 s, both judged
against A's voiceprint. It separates them cleanly. Hand it `G`'s versions of the same two clips and
it cannot tell them apart -- at ~2 dB both outputs are mostly the same distortion, and that
distortion contaminates `cos(s_hat, e_A)` and `cos(s_hat, e_B)` equally.

#### What this settles, and what it corrects

`tau_margin`'s "NO USABLE THRESHOLD" verdict was **a statement about this checkpoint, not about
`M_i`**. This file recorded it as a property of the formula in several places, which was the same
error already corrected for refinement (2026-08-24): measuring one point on the extractor axis and
reading it as a property of the design.

Three standing observations now have one explanation rather than three:

* `tau_margin` scoring J = +0.046 (Session B),
* `gated_deflation` collapsing onto `ungated_deflation` at 98-99% accept (Phase 2, Stage A),
* the refinement gate accepting 78 and 63 candidates the ground truth called worse (Session 1).

All three are the same margin, computed on the same ~2 dB output. **They recover together when `G`
does.** That is a claim about the training budget (Q2), and it means a gate redesign evaluated on
this checkpoint would have been measuring the extractor either way.

#### The `V_i` contrast is the control, and it holds

`V_i` embeds enrollment clips taken from the *mixture*, never `G`'s output -- so it is the one check
not gated on extractor quality, and it is the one check that works today (J = +0.373, and Session 2
priced it at zero cost). The clean-margin result completes the picture the unifying mechanism
predicted: everything that embeds `G`'s output is starved; the one thing that does not, works.

#### THE GATE HAS FOUR CHECKS, AND ONLY TWO HAVE BEEN EXAMINED

Counted across every gate decision this project has ever recorded -- **10,950** rows spanning the
Phase 2 clip50 DoD, Stage A, and Stage B Sessions 1-3:

| check | config key | rejections | status |
|---|---|---|---|
| identity margin `M_i` (the "Leak" term of the theory's Eq. 11) | `tau_margin` | 4,106 | diagnosed: sound, starved by `G` |
| enrollment variance `V_i` | `max_mean_variance` | 196 | **tuned** (0.05 -> 1e-4) and priced (free) |
| speech coverage | `min_vad_coverage` | **0** | **NEVER TUNED** |
| artifact score | `max_artifact_score` | **45** | **NEVER TUNED** |

*Why the last two were never swept, and it is structural rather than an oversight.*
`scripts/tune_gate.py` gives `V_i` and the margin **detection** sweeps because each has a labelled
fault population -- contaminated enrollment, and swapped conditioning. VAD and artifact get only
**rate** sweeps, explicitly marked "no fault population": there is nothing to detect *against*, so
no Youden's J can be computed and no threshold can be justified.

**"0 rejections in 10,950" does not mean the check is useless**, and this file has made exactly that
inference before and been wrong by 500x. It is equally consistent with (a) the failure mode never
occurring in these corpora, or (b) `min_vad_coverage: 0.5` / `max_artifact_score: 0.9` sitting in
the wrong place entirely -- which is precisely what "`V_i` is structurally dead" turned out to be.
Closing this properly needs a fault fixture per check: a silenced or truncated output for VAD, a
deliberately distorted one for artifact. That is the same move that converted `V_i` from a dead
check into a tuned one, and it needs no GPU beyond one `tune_gate.py` variant.

**Do not re-tune `tau_margin` on this checkpoint.** The clean sweep's suggested 0.3 is the value for
a *perfect* extractor. On today's output that same threshold buys nothing: the Session B sweep
measured 19.9% detection against 15.2% false rejection there, i.e. J = +0.046 -- it rejects roughly
as much good as bad, because the two distributions are 0.019 apart and almost entirely overlapping.
No threshold anywhere separates them, so moving the cut only trades one error for the other. Re-run
this probe after Stage C and tune then.

*And note which way the gate errs, because it is easy to state backwards.* Both populations sit
**ABOVE** the shipped `tau_margin: 0.1` (medians 0.43873 and 0.41965), so on this corpus the cut
falls below almost everything and the gate accepts -- the dev sweep measures only **1.3% false
rejection at 0.1**. The accept rate elsewhere varies with corpus and enrollment quality (Stage A and
Phase 2's heterogeneous corpus: **98-99%**; the clip50 DoD run: `gated_deflation` **67.1%** at m=3
and **47.3%** at m=5), so "accepts nearly everything" is not universal. What IS consistent is the
direction of the error: where ground truth could check it, the gate was too PERMISSIVE --
`ceiling_reject_gate_would_accept` fired **78 and 63 of 150** while
`ceiling_accept_gate_would_reject` fired only **4 and 6** (Session 1). That is why raising
`tau_margin` is the wrong lever: the signal is absent, the cut is not misplaced.

---

## STAGE C — THE TRAINING RUN: three unimplemented losses land here

**Decided 2026-08-25.** The theory's objective (`docs/diarization_full_mathematical_theory.pdf`
§10) is four terms:

```
L = λ1·L_sep  +  λ2·L_spk  +  λ3·L_recon  +  λ4·L_art
```

Only `L_sep` has ever been implemented (`dagger/losses/sisdr.py`). Stage C implements the other
three. Their statuses going in are **not the same**, and the difference is worth keeping.

### 1. `L_recon` — the noise-head reconstruction loss. OVERDUE, not missing.

```
L_recon = || x_O − Σ_i ŝ_i − n̂ ||²
```

This one was **deliberately deferred with a stated deadline**, recorded in
`dagger/losses/__init__.py`:

> *"deliberately deferred, not yet implemented: WSJ0-2mix/LibriMix scenes are anechoic sums with no
> noise term by construction, so Phase 1 trains noise-free (CLAUDE.md §2's explicit 'or train on
> noise-free data' branch). **This noise head MUST land before Phase 3 trains on real/noisy
> corpora**, or the reconstruction loss will fight the separation loss whenever noise != 0
> (guardrail §6.5)."*

Stage C is the first Phase 3 training run, so **that deadline is now**. The deferral was correct:
§2 permits either branch, and LibriMix genuinely has no noise term.

**But the justification was about NOISE, and the term does a second job nobody named: it constrains
the output LEVEL.** `L_recon` is not scale-invariant — if every `ŝ_i` came out 2.86x too loud,
`Σ ŝ_i` would be 2.86x the mixture and this term would be enormous. `L_sep` (SI-SDR) fits the scale
out before measuring error and therefore cannot see level at all.

*So this is the root cause of Q5.* `G` emits the overlap region at a median **2.86x** the true
amplitude (`level_error_db` +9.14, replicated at 3/25/50 scenes) precisely because the one loss term
that would have constrained it was dropped -- for a reason that was valid on the axis it was argued
on, and silent on the axis that mattered. **Q5 is therefore not a separate question with an
inference-time fix; it is a consequence of an unimplemented loss, and it closes here.**

*Why the inference-time patches were abandoned.* `match_level_to_mixture` (committed, default OFF)
computes `c = <g, x_O>/<g, g>`, which is the **MMSE/Wiener gain** -- it deliberately attenuates, to
0.597 at 1.7 dB SNR, so it lands the output at 0.59x instead of 1.0x; and its `MAX_RESCALE = 8`
clamp refuses exactly the speakers whose error is worst (`alpha_2` reaches 19.5). The deeper problem
is general: **any estimator that minimises squared error against the mixture shrinks**
(errors-in-variables attenuation), so the whole family -- including a joint multi-speaker fit -- has
the same flaw. The alternative, matching `G`'s overlap RMS to the speaker's solo-region RMS, works
on LibriMix but assumes a speaker's level is stationary between regions. That is false on AMI-SDM:
the Lombard effect (people raise their voice when talked over), head movement at a single distant
mic, reverberation, and solo/overlap runs minutes apart all break it. **A training-time constraint
is dataset-independent by construction; every inference-time patch smuggles in a corpus assumption.**

### 2. `L_spk` — speaker consistency. NEVER SCHEDULED.

Pushes `φ(ŝ_i)` toward `ē_i`: trains `G` to produce output that *embeds* like the right speaker.

Named in CLAUDE.md §4's target layout and in `dagger/losses/__init__.py` ("remain unimplemented"),
but **no phase's Build step has ever scheduled it**. That is a gap in the plan, not a deferral.

*Why it belongs in Stage C specifically.* Q1b measured the identity margin at J = **+0.046** on
`G`'s output against J = **+0.453** on the clean source -- the formula is sound and starved, because
`G`'s ~2 dB output embeds poorly. `L_spk` is the loss that directly optimises that quantity. The
check we could not make work and the loss we never implemented are the same quantity, which is not
a coincidence to leave unrecorded.

### 3. `L_art` — artifact. NEVER SCHEDULED, AND NEVER DEFINED. **DECISION OPEN.**

> **Status 2026-08-28: the formula is NOT chosen.** What follows is (a) the finding that no
> definition exists anywhere, (b) the consistency constraint the theory doc supplies for its own
> terms, which any candidate must satisfy, and (c) **one candidate** that satisfies it, recorded as
> an option rather than a decision. **Stage C does not proceed until this is settled.**
>
> Note what the candidate rests on and what it does not. Its *consistency* argument is solid — it
> vanishes at the truth, which is the doc's own test. Its *premise* — that spectral flatness is the
> right thing to measure at all — rests on a diagnostic found broken three days earlier
> (Stage B Session 4), and has never been checked against a real artifact population.
> `phase3_gate_faults.yaml` is the run that would check it.

Worse than `L_spk`'s status. `L_spk` at least has a stated meaning ("push `φ(ŝ_i)` toward `ē_i`").
`L_art` had **no formula anywhere** — not in `dagger/losses/`, not in this journal, and *not in the
theory doc either*. The word "Artifact" appears exactly **twice** in
`docs/diarization_full_mathematical_theory.pdf` and is defined in neither place:

* §9 Eq. (16): `C_i = α·Sim(φ(ŝ_i), e_i) + β·Speech(ŝ_i) − γ·Leak(ŝ_i) − δ·Artifact(ŝ_i)`
* §10: `L = λ1·L_sep + λ2·L_spk + λ3·L_recon + λ4·L_art`

§10 writes `L_sep` and `L_recon` out in full and leaves `L_art` a bare symbol. (`Speech(ŝ_i)` is in
the same position.) So there was nothing to look up, and Stage C was gated on deriving it.

**The constraint the doc supplies for its own terms.** Proposition 7 and Corollary 1 establish a
test every term in this objective must pass: *it must vanish at the truth, or it fights `L_sep`*.
That is exactly the noise argument — `L_recon` without `n̂` is nonzero at `ŝ_i = s_i`, so no estimate
zeroes both, and the repair is `n̂` so the truth becomes a common minimiser.

**The obvious candidate fails that test.** `L_art = F(ŝ_i)` for `F` = spectral flatness would drive
`F -> 0`, but clean speech measures `F ≈ 0.37` (Stage B Session 4), not 0. Minimising it moves `ŝ_i` AWAY
from `s_i`. That is Proposition 7's conflict one term over, and it would have been nastier than the
noise case: `L_sep` and `L_art` would trade against each other with no committed metric able to say
which was winning.

**CANDIDATE (not chosen) — deviation from the target, i.e. Corollary 1's repair reapplied:**

```
L_art = ( F(ŝ_i ⊙ w_Oi) − F(s_i ⊙ w_Oi) )²
```

* Exactly 0 at `ŝ_i = s_i`, so `(s_i, n)` stays a common minimiser of all four terms.
* `F(s_i ⊙ w_Oi)` is constant in θ — computed once under `no_grad` per (item, speaker).
* `F` is the **energy-gated** flatness from Stage B Session 4, in torch: frames above −40 dB of the clip
  peak, `exp(mean(log(|X|+ε)))/mean(|X|)`. Differentiable; the frame-selection mask is detached.
* Scored on the **overlap-windowed slice** — the same slice the training loop already scores, and
  the same quantity `max_artifact_score` reads, so the gate check is this term's read-out. That is
  the attribution rule this section already sets for the other two terms.
* Needs the clean target at training time, which `batch["sources"]` already carries. No pipeline
  change, same as `L_recon`.

#### THE CANDIDATE WAS MEASURED 2026-08-28 AND IT ORDERS BAD OUTPUTS WRONG

Before adopting it, `F` was scored on one speech-like target (`F(s_i) = 0.3268`) against three
estimates. SI-SDR is included as the ground-truth ordering:

| estimate | SI-SDR | `F(ŝ_i)` | candidate `L_art` |
|---|---|---|---|
| A  good extraction | **+17.0 dB** | 0.5836 | **0.06592** |
| B  broadband hiss | +4.9 dB | 0.7495 | 0.17861 |
| C  80% spectral holes | **−37.8 dB** | 0.3078 | **0.00036** |

**Row C is a destroyed signal and the loss charges it essentially nothing — 180x LESS than the good
17 dB extraction in row A.** The term has a pathological minimum: it would prefer that `G` wreck the
signal via spectral holes over extracting it well.

*Mechanism.* `F` is hypersensitive upward (mild additive noise fills the spectral valleys, and the
geometric mean is dominated by its smallest bins, so a 17 dB estimate already reads 0.58) and nearly
blind downward (`punch_holes` resynthesises with 4x overlap-add, so neighbours refill what was
zeroed; the output is phase-wrecked but its flatness statistics land back near the target's). So `F`
mostly tracks the broadband noise floor — which `L_sep` already penalises well — and misses the
structural corruption `L_art` was meant to add value on.

**The lesson, and it generalises past this term.** The consistency derivation was *correct* and
still is: the candidate does vanish at the truth, which is the doc's own test. What was never
checked is whether `F` **orders bad outputs sensibly**, and vanishing-at-truth says nothing about
that — a term can be zero in the right place and have its gradient point somewhere useless
everywhere else. **Any candidate for `L_art` must pass BOTH tests: zero at the truth, AND monotone
against SI-SDR across a set of deliberately damaged estimates.** The three-row table above is the
cheap screen; run it before adopting anything here. This is the same blindness Stage B Session 4 found in
the same diagnostic (0.7376 clean vs 0.7422 on `G`'s output), one layer up and now with a sign that
rewards damage.

*If a flatness-based candidate is chosen anyway*, the argument for **two-sided rather than a
one-sided hinge** — and the reason is not symmetry-for-its-own-sake.
`relu(F(ŝ)−F(s))²` assumes every artifact RAISES flatness. Stage B Session 4 put that in doubt: musical
noise is sparse and tonal and may lower it, which is what `phase3_gate_faults.yaml` exists to
measure. The doc names the downward failure too — §5's reading says a network on clean audio
*"can only add artifacts (over-suppression, phase distortion)"*, and over-suppression makes output
LESS flat than the target. A two-sided term catches both families and assumes neither.

*A design property worth naming, not just a convenience:* two-sided also decouples Stage C from
the gate_faults run. A one-sided loss would need the direction answer to fix its sign, so a
training run would depend on a GPU run that has not happened. **A formula that requires knowing
which way a failure goes is more fragile than one that penalises departure from ground truth in
either direction.**

### Two implementation constraints found 2026-08-28, neither previously recorded

**`L_recon` collides with the loop's memory workaround, and the collision is invisible from either
side.** `scripts/train_phase1.py` backwards **per speaker** on purpose — *"summing all speakers'
losses before `backward()` holds `num_speakers` full TF-GridNet graphs and OOMs on 16 GB GPUs"* —
while `L_recon = ||x_O − Σ_i ŝ_i − n̂||²` needs every `ŝ_i` alive at once. The line-211 comment is
correct about what existed then; §10 is correct about the theory; nobody wrote down their
intersection. (Same shape as the `L_recon` deferral being right on the noise axis and silent on the
level axis.)

*It resolves exactly, not approximately.* Add a `no_grad` pass to build `Σ_j ŝ_j` as values, then
keep the per-speaker backward with the other speakers detached:

```
r = x_O − Σ_j ŝ_j − n̂            # computed once; all terms detached-valued
L_recon_i = ||r||²  with only ŝ_i attached
```

`detach` changes no *values*, so `r_i == r` for every `i`, and gradients accumulate across
`backward()` calls — giving `Σ_i 2r·(−∂ŝ_i/∂θ)`, which is the **exact** gradient of the true
`L_recon`. Cost is 2x forward compute, zero extra memory, and the existing loop survives unchanged.

**`L_spk` has no gradient path today.** `TitaNetEncoder.embed` resamples 8 kHz -> 16 kHz through
`dagger.data.audio_io.resample`, which is **numpy** (`dagger/enroll/encoder.py:143-145`), so
gradients cannot flow from `ŝ_i` back through `φ`. **Decided: build a torch-differentiable `φ`
path** — `torchaudio.functional.resample` plus the NeMo model's forward, weights frozen, gradients
flowing to the input only. The alternative (a cheap spectral proxy) was rejected because it
optimises a different quantity than the margin check reads, which would make the 0.019 -> 0.243
prediction untestable: a null result could not distinguish "the loss failed" from "the proxy was
unrelated to the margin". Note this does not violate §6.3 — that rule is eval-encoder ≠ training
encoder, and `φ` IS the training encoder.

### How to add three terms at once without losing attribution

Adding all three to one retrain means a quality change cannot be attributed to any of them. Two
cheap protections, both of which have precedent in this phase's failures:

* **Make every `λ` a config key**, so `λ_spk = λ_art = 0` reproduces a pure-`L_sep` run exactly.
  A term that cannot be switched off cannot be ablated, and §9's "a flag that is read but never
  forwarded" says to add a run that would visibly differ and assert that it does.
* **Read the result on the metric each term targets**, not only on depth-2 SI-SDR:
  `L_recon` -> `level_error_db` (should collapse toward 0, and this is the falsifiable part);
  `L_spk` -> the clean-vs-extracted margin gap (0.019 should migrate toward 0.243);
  `L_art`  -> `max_artifact_score`, which has fired 45 times in 10,950 decisions and is untuned.

*The `L_recon` prediction is the sharpest test Stage C carries:* if `level_error_db` does not fall,
the level error is NOT explained by the missing loss and the whole diagnosis above is wrong.

### Stage C decisions — SETTLED 2026-08-28

**Training masks: oracle + `mask_augment`.** Not cached pyannote output. The deciding argument is
where a diarization error *lands*: with augmentation it degrades the INPUT mask, which is the thing
`G` is meant to become robust to. With real pyannote output it would have to pass through
cluster-to-speaker mapping first — training needs `(x_O, ē_i) -> s_i` and a diarizer cluster is not
a labelled speaker — so every mapping error becomes a corrupted TARGET, which is strictly worse
than a corrupted input. `dagger/data/mask_augment.py` is also already built, tested and seeded, and
simulates the *measured* 2-minute profile (miss 0.105, confusion 0.008, overlap recall 0.758)
rather than an assumed one. That module's own docstring records the cost of getting this wrong
once already: label-swap augmentation designed off short-scene DER that turned out broken.

**Initialization: from scratch.** Not warm-started from `..._scratch_clip50.pt`. Warm-starting is
cheaper and would converge inside a session budget, but that checkpoint carries **the exact
pathology under test** — it emits the overlap region at 2.86x. Asking `L_recon` to undo a learned
bias means a null result cannot distinguish "the loss is too weak" from "the initialization was too
sticky", which converts Stage C's sharpest falsifiable prediction (*`level_error_db` should
collapse toward 0; if it does not, the missing-loss diagnosis is wrong*) into an uninterpretable
one. The extra GPU buys the ability to believe the answer either way.

**Corollary for the attribution baseline.** Stage C already changes three things at once, and the
protection above is that every `λ` is a config key so `λ_spk = λ_art = 0` reproduces a pure-`L_sep`
run exactly. The mask source must obey the same rule: **run the baseline arm with `mask_augment`
off**, so the arm being compared against differs from Phase 2 in the losses and in nothing else.
Otherwise four simultaneous changes produce one unattributable number.

**The constraint that bounds both** (not a decision): `build_scene_crop_dataset._prepare` holds each
whole scene at ~35 bytes/sample, so 800 two-minute scenes is 27 GB and Phase 2's 2400-scene count
would need 81 GB.

---

## STAGE B — SESSION 4: Q1's last two checks get a fault population (2026-08-28)

**What this session did.** Built the manufactured fault fixtures that `min_vad_coverage` and
`max_artifact_score` have never had, and in doing so found that one of the two checks was not
measuring what its name says. No GPU was used; every number below comes from committed CSVs or
from CPU probes of the diagnostics themselves.

### Finding 1 — `vad_coverage` is a detector, and a labelled fault population was already committed

`clean_swapped` (Session 3's Q1b arm) puts speaker *j*'s clean source into speaker *i*'s overlap
window. Where *j* is not talking there, it is silence — which is precisely the "output contains no
speech where it should" failure `min_vad_coverage` exists to catch. Recomputed from
`experiment_stage_B_run_4/.../gate_tune_clean_margin.csv`, no new run:

| `min_vad_coverage` | detection | false rej. | J |
|---|---|---|---|
| 0.5 | 65.3% | 0.0% | +0.653 |
| **0.6** | **68.0%** | **0.0%** | **+0.680** |
| 0.7 | 70.7% | 13.3% | +0.573 |

The same contrast **on `G`'s output** (`swapped` vs `correct`) scores **J = +0.000** at every
threshold, and −0.047 at 0.9.

**That is the margin's verdict again, one check over: sound diagnostic, starved by `G`.** It is the
third instance of §9's "measuring one point on an axis and calling it a property of the design",
and the unswept axis was the extractor all three times. The data to see it had been sitting in a
committed file since Session 3; nobody ran the sweep because `_rate_sweep` was the only thing
pointed at that column.

### Finding 2 — `max_artifact_score` was reporting each speaker's DUTY CYCLE

`spectral_flatness` averaged over every frame of the estimate, and a digitally silent frame has
every bin pinned at the `eps` floor, so its geometric and arithmetic means are equal and it scores
exactly **1.0** — maximally "artifact-like" while containing nothing. The estimate it is handed is
`x*w_Ei + G(x_O,e_i)*w_Oi`, and `crossfade_windows` guarantees `w_Ei + w_Oi == activity_i`:
**outside speaker `i`'s activity BOTH windows are zero and the estimate is exactly zero.** In a
scheduled 3-speaker scene that is over half the track.

Two measurements, and it is the disagreement between them that matters:

| | flatness |
|---|---|
| clean source (`clean_correct`) | **0.7376** |
| `G`'s ~2 dB output (`correct`) | **0.7422** |
| pure white noise (CPU probe) | 0.847 |
| digital silence (CPU probe) | 1.000 |

A **0.005** gap between pristine audio and near-garbage is no dynamic range at all, and the shipped
`max_artifact_score: 0.9` sits **above pure white noise** — the check could not have rejected an
estimate that was 100% noise. It has fired 45 times in 10,950 decisions, and those 45 were almost
certainly the quietest speakers rather than the most damaged ones.

*Same signature as Q5's level error:* a defect no committed metric could contradict, surfacing only
as a disagreement between two quantities rather than as a bad number.

### Finding 3 — `active_mask`'s −40 dB floor sets what a VAD fixture can be

Measured on a synthetic harmonic, region-selective attenuation against `vad_coverage`:

| attenuation | −20 dB | −30 dB | −40 dB | −50 dB |
|---|---|---|---|---|
| coverage | 1.000 | 1.000 | 0.116 | 0.025 |

The first fixture written was `quiet_30db`, which is **inert** — it sits entirely on the wrong side
of the cliff and would have produced a detection column of 0% that read as a broken generator. The
catalogue now straddles the floor (20/30/40/50 dB), so the two above it are deliberate NEGATIVE
controls and the sweep *locates* the floor instead of assuming it.

Note also what is NOT a valid fixture: a uniform whole-signal gain. `active_mask` thresholds each
frame against the clip's own peak, so global attenuation is invisible — the same scale-invariance
that hid Q5's 2.86x error from every SI-SDR here. A region-selective gain works only because the
loud, untouched solo copy holds the peak reference in place. `test_gate_faults.py` pins both halves.

### What landed

* `dagger/gate/faults.py` (**NOT DEPLOYABLE**) — `drop_span`, `attenuate`, `add_noise`,
  `punch_holes`. Every fixture GRADED: a binary "emit silence" fault scores J = 1.0 at every
  candidate and therefore places none, which is the degenerate sweep oracle-region `V_i` produced.
* `spectral_flatness(min_energy_db=...)` — **opt-in, default `None`**, so every committed
  `artifact_score` column stays regenerable by its own generator. Threaded to four call sites via
  one named reader, `dagger.gate.confidence.artifact_min_energy_db`.
* `scripts/tune_gate.py` — `G` now runs **once per speaker** and every fault population is a numpy
  corruption of the cached tensor, so ten populations cost one extraction. Graded detection tables,
  a direction report, and four guards: non-empty populations, grid-endpoint, per-fixture inertness,
  and fault-sign disagreement.
* `configs/phase3/experiments/phase3_gate_faults.yaml`.

### Why the suggestion criterion is not Youden's J

J needs one faulty population; a graded family has five, and they disagree. On a worked example the
J against `dropout_75` picks 0.55 while the J against `dropout_25` picks 0.80 — same data, opposite
answers, and which severities appear in the table is arbitrary. Both detection and false rejection
are monotone in the threshold, so there is no interior optimum to search for; an argmax over a
monotone curve finds sampling noise, not a peak. The rule is instead **the tightest threshold whose
false rejection stays within `max_false_rejection` (5%)**, which puts the value judgment — a false
rejection costs quality on every scene, a missed detection only on faulty ones — on the record
rather than inside J's unchosen 1:1 weighting.

### Predictions on record, so the run can falsify them

1. With `min_energy_db: -40`, healthy `artifact_score` falls **0.742 → ~0.35–0.45**. If it does not
   move, the duty-cycle diagnosis is wrong.
2. The `fault_g_` arm stays a non-detector for both checks; the `fault_clean_` arm separates. Q1
   then reads "4 of 4 diagnosed, 2 of them starved by `G`".
3. `add_noise` and `punch_holes` may move `artifact_score` in **opposite** directions. If they do,
   `max_artifact_score` needs replacing, not tuning.

### Verification

Suite **516 passed, 1 skipped**. Each fixture was then deliberately neutered to confirm its guard
reddens — §9's "a test that cannot fail is not a passing test", discharged rather than asserted:

```
punch_holes  neutered -> RED: test_punch_holes_actually_changes_the_signal
drop_span    neutered -> RED: test_dropout_lowers_vad_coverage_monotonically
attenuate    neutered -> RED: test_attenuation_is_detectable_only_below_active_masks_floor
add_noise    neutered -> RED: test_additive_noise_raises_flatness_monotonically
energy_gate  neutered -> RED: test_silent_frames_inflate_ungated_flatness_and_the_gate_removes_them
```

The report generators were also exercised on synthetic rows before spending GPU on them: the graded
table, the empty-clean-arm path, and the sign-disagreement guard all render correctly.

### Two defects caught during implementation, both worth recording

* **`_row` and `_fault_row` briefly measured different quantities.** `_fault_row` forwarded
  `artifact_min_energy_db`; `_row` did not. Healthy populations would have carried whole-track
  flatness (~0.74) and every fault the gated value (~0.4), so the sweep would have reported that
  *every* corruption LOWERS flatness — including additive noise, which provably raises it — and
  `_direction_report` would have printed "FAULTS DISAGREE IN SIGN" for you to read as Finding 3.
  **Nothing would have failed.** Neither the inertness guard nor the non-empty guard catches this;
  what catches it is asking whether two compared numbers came out of the same function.
* **The pairing guard made the two older `tune_gate` configs un-runnable.** Both now carry
  `artifact_min_energy_db: null` — the guard's point is a decision on the record, not a particular
  value, and §7 requires those committed results to stay regenerable.

### Q1 after this session

| check | config key | status |
|---|---|---|
| identity margin `M_i` | `tau_margin` | diagnosed: sound, starved by `G` |
| enrollment variance `V_i` | `max_mean_variance` | **tuned** (0.05 → 1e-4) and priced (free) |
| speech coverage | `min_vad_coverage` | **diagnosed: sound, starved** (J +0.680 clean, +0.000 on `G`); fixture built, threshold pending the run |
| artifact score | `max_artifact_score` | **defect found** — measures duty cycle; fix opt-in, fixture built, threshold pending the run |

**4 of 4 now diagnosed; 1 of 4 tuned.** Three of the four trace back to `G`'s quality, which is Q2.
