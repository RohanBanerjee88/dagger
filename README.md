# dagger

**Diarization-guided, accumulation-free target speaker extraction.**

Recover one clean audio track per speaker from a single-channel recording where
people talk over each other. Speaker diarization finds where each person talks
*alone*; those solo moments become a voice embedding; that embedding pulls each
speaker out of the overlapping parts.

**The one rule** (see [`CLAUDE.md`](CLAUDE.md) §1): every speaker's output is
extracted from the *untouched* overlap mixture `x_O` — never from a running
residual. That single choice keeps each speaker's error independent, so error
does not accumulate with overlap depth.

```
s_hat_i(t) = x(t)·w_Ei(t) + G(x_O(t), e_bar_i)·w_Oi(t)
                                ^^^^ always the ORIGINAL mixture
```

## Status — Phase 0 + 1 done, Phase 2 in progress

**Phase 0 — plumbing, oracle diarization:**

- `dagger/audio/provenance.py` — provenance tracking that makes the "no residual
  in the audio path" rule mechanically enforceable (the extractor refuses
  residual inputs).
- `dagger/diarize/` — **oracle** diarization: read ground-truth RTTM → activity
  matrix `a_i(t)` → solo regions `E_i` / overlap mask, and the overlap mixture
  `x_O`.
- `dagger/extract/` — extractor interface + Phase 0 `NullExtractor`.
- `dagger/reconstruct/` — soft-mask stitching with partition of unity
  (`w_Ei + w_Oi = a_i`) and crossfaded seams.
- `dagger/metrics/` — SI-SDR (overall and region-wise).
- `dagger/data/` — real-corpus loaders (**LibriMix**, **WSJ0-2mix**). Mixtures are
  built *on the fly* from source utterances (storage-lean: only the sources live
  on the mounted volume, never the mixtures) and *staggered* into a solo →
  overlap → solo layout so the copy-solo / extract-overlap split is exercised.
  Oracle activity is derived from the clean sources.

Still open in Phase 0 (deferred to later phases per CLAUDE.md §5): the speaker
-margin and Whisper-WER metrics.

**Phase 1 — identity conditioning:** DoD met. `dagger/enroll/` (top-K solo
clips → speaker embedding `phi` → mean `e_bar_i`) and `dagger/extract/` (the
proposed extractor `G`: TF-GridNet + cross-attention fusion, an original
implementation informed by the USEF-TSE architecture) are wired into
reconstruction, alongside a blind-separation baseline for comparison. See
[Results](#results) below.

**Phase 2 — accumulation-free reconstruction (the money experiment):** in
progress, DoD not yet called. `dagger/reconstruct/deflation.py` (the
ungated/gated residual-deflation anti-patterns, built only for comparison —
see CLAUDE.md §1), `dagger/refine/coarse_to_fine.py` (recursion refines the
embedding only; audio always comes from the guarded, unmodified `x_O`), and
`dagger/gate/` (confidence gate: identity margin, VAD, artifact score) are
implemented and wired into a depth-stratified evaluation. **Ordering
criterion met:** `coarse_to_fine > gated_deflation > ungated_deflation` on
overlap depth 3, replicated on an independently fine-tuned checkpoint.
**Flat-vs-sloped criterion only weakly/directionally supported so far** — see
[Results](#results) and [Limitations](#limitations).

## Quickstart

```bash
pip install -e .            # numpy + pyyaml (core)
```

```bash
pip install -e '.[dev]'     # + pytest
pytest                       # 146 tests, numpy/torch-CPU only, no GPU or corpus needed
```

The no-residual-in-audio-path guard (CLAUDE.md §1) has its own dedicated file,
`tests/test_no_residual_in_audio_path.py` — the cheapest insurance in the repo
against the one mistake that would invalidate the paper's central claim.
Torch-dependent tests (`tests/phase1/`) skip cleanly if `torch` isn't
installed; install it (`pip install torch`) to exercise them.

To run the end-to-end Phase 0 demo on a real corpus you need audio on a mounted
volume (see below):

```bash
pip install -e '.[data]'    # soundfile + scipy + python-dotenv
cp .env.example .env        # set DAGGER_DATA_ROOT (and, for WSJ0, its access key)
python scripts/run_phase0.py --config configs/phase0/dod/phase0_librimix.yaml
python scripts/run_phase0.py --config configs/phase0/dod/phase0_wsj0mix.yaml
```

The run reports SI-SDR split by region: solo interiors are recovered bit-exactly
(`inf`), while overlap regions are poor by design until Phase 1 adds the
extractor `G`.

### Remote-compute data setup

- Mount the corpus (LibriSpeech for LibriMix; WSJ0 for WSJ0-2mix) on the compute
  node and point `DAGGER_DATA_ROOT` at it in `.env`. Nothing is committed to the
  repo — `.env` and `data/` are gitignored.
- Each dataset config's `metadata` path (a LibriMix CSV or a `mix_2_spk` list) is
  resolved under `DAGGER_DATA_ROOT`.
- **WSJ0-2mix** is LDC-licensed and has no API key. When the corpus is mounted,
  no credential is needed. To fetch it from a private mirror instead, set
  `DAGGER_WSJ0_ACCESS_KEY` in `.env` (the single authorization hook).

## Results

### Phase 1 — proposed vs. blind separation (3-speaker LibriMix, oracle diarization)

| system | mean overlap SI-SDR |
|---|---|
| proposed | 4.40 dB |
| blind | 2.05 dB |

150 test scenes / 450 speaker-rows (`results/phase1_librimix_3spk.csv`). Reproduce:

```bash
pip install -e '.[data,ml]'
python scripts/run_phase1.py --config configs/phase1/dod/phase1_librimix_3spk_eval.yaml
```

This requires the trained checkpoint at `checkpoints/phase1/proposed_librimix_3spk.pt`
(see [Pretrained weights](#pretrained-weights)), or retrain it yourself:

```bash
python scripts/train_phase1.py --config configs/phase1/dod/phase1_librimix_3spk_train.yaml --system proposed
python scripts/train_phase1.py --config configs/phase1/dod/phase1_librimix_3spk_train.yaml --system blind
```

### Phase 2 — depth-stratified accumulation-free vs. deflation (3-speaker LibriMix, oracle diarization)

Mean SI-SDR by overlap depth, fine-tuned checkpoint, 150 test scenes (5400 rows):

| system | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| no_recursion | 43.33 dB | 4.14 dB | 0.53 dB |
| ungated_deflation | 41.72 dB | 0.49 dB | -3.92 dB |
| gated_deflation | 42.19 dB | 1.84 dB | -2.19 dB |
| coarse_to_fine | 43.36 dB | 4.21 dB | 0.21 dB |

`coarse_to_fine > gated_deflation > ungated_deflation` at depth 3, replicating the
theoretically-predicted ordering (`results/phase2_librimix_3spk_finetuned.csv`). See
[Limitations](#limitations) for why this isn't a called DoD yet. Reproduce:

```bash
pip install -e '.[data,ml]'
python scripts/run_phase2.py --config configs/phase2/experiments/phase2_librimix_3spk_eval_finetuned.yaml
```

### Pretrained weights

All three checkpoints are hosted on the Hugging Face Hub under Apache-2.0 (same
license as this repo). They do **not** bundle NVIDIA's TitaNet-Large speaker
encoder (`nvidia/speakerverification_en_titanet_large`, CC-BY-4.0) — that
model is loaded separately at runtime via NeMo; see `NOTICE` for the full
attribution.

| Checkpoint | HF repo | Used by |
|---|---|---|
| Phase 1 "proposed" extractor | [`AdityaAA2004/dagger-phase1-proposed-librimix-3spk`](https://huggingface.co/AdityaAA2004/dagger-phase1-proposed-librimix-3spk) | `configs/phase1/dod/phase1_librimix_3spk_eval.yaml`, `configs/phase2/experiments/phase2_librimix_3spk_eval.yaml` |
| Phase 2 fine-tuned extractor | [`AdityaAA2004/dagger-phase2-proposed-librimix-3spk-finetuned`](https://huggingface.co/AdityaAA2004/dagger-phase2-proposed-librimix-3spk-finetuned) | `configs/phase2/experiments/phase2_librimix_3spk_eval_finetuned.yaml` |
| **Phase 2 final model** (the reported one) | [`AdityaAA2004/dagger-phase2-final-model`](https://huggingface.co/AdityaAA2004/dagger-phase2-final-model) | `configs/phase2/dod/phase2_librimix_{3,4,5}spk_eval_scratch.yaml`, `configs/phase3/**` |

**Start with the Phase 2 final model** unless you specifically want to reproduce an
earlier phase's table. It is the checkpoint every Phase 2 DoD number and every Phase 3
result is computed from: trained from random init by a single command (no warm-start
chain), on a multi-depth curriculum interleaving 3-, 4- and 5-speaker mixtures.

Fetch and cache a checkpoint locally with:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="AdityaAA2004/dagger-phase2-final-model",
    filename="phase2_final_model_weights.pt",
)
```

Note the Hub filename (`phase2_final_model_weights.pt`) differs from the path the configs
expect (`checkpoints/phase2/proposed_librimix_curriculum_3_4_5_scratch_clip50.pt`), so
either copy it into place or override `extractor.checkpoint` with the cached path:

```bash
mkdir -p checkpoints/phase2
cp "$ckpt_path" checkpoints/phase2/proposed_librimix_curriculum_3_4_5_scratch_clip50.pt
```

## Limitations

- **The +2.35 dB mean margin is driven by magnitude, not consistency.** Row-by-row
  (450 speaker-scenes, 396 with a scoreable overlap region), `proposed` beats `blind`
  only 49.7% of the time (197/396) — the mean advantage comes from `proposed`'s wins
  being much larger than its losses (paired std 7.52 dB against a mean of 2.35 dB), not
  from winning more often. Phase 2's depth-stratified evaluation is designed to test
  whether these large wins concentrate at deeper overlap, which would explain the
  asymmetry rather than leave it as unexplained variance.
- **Both systems are undertrained.** The reported numbers use 2000 of ~34,000 available
  Libri3Mix train-360 scenes; training loss was still descending at cutoff. Treat the
  reported SI-SDR values as lower bounds, not converged performance.
- **The 2-speaker WSJ0-2mix literature-bar check (~23 dB SI-SDR) is deferred** — no LDC
  license currently available; Libri2Mix would substitute if needed.
- **Real (non-oracle) diarization is untested.** All current results use ground-truth
  RTTM; Phase 3 swaps in a real diarizer and reports the oracle-vs-real gap.
- **Phase 2's "flat vs. sloped" criterion is only directionally supported, not a clean
  result.** The ordering `coarse_to_fine > gated_deflation > ungated_deflation` holds at
  depth 3, and every system's depth-2→3 drop is larger the more deflation-prone the
  system is (worst for `ungated_deflation`, best for `coarse_to_fine`) — but all four
  systems still drop sharply from depth 1 to depth 2, so the accumulation-specific
  effect rides on top of a shared decline rather than producing a dramatically
  different shape between systems.
- **Fine-tuning on scheduled-placement (true depth-3) scenes raised every system's
  absolute floor by ~1-2 dB, but did not sharpen the accumulation-specific gap.** That
  gap stayed roughly flat (`ungated_deflation`'s extra drop beyond `no_recursion`'s own
  went from 1.22 dB to 0.80 dB) — retraining alone isn't the lever that makes this
  effect larger; see `CLAUDE.md`'s Phase 2 section for the fuller analysis.
- **The 4-5 speaker overlap-depth extension is zero-shot and out-of-distribution.**
  The ordering holds cleanly at every depth 2-5 and the gap trends wider (a promising
  sign the effect is real), but absolute SI-SDR at depth 4-5 is deeply negative for
  every system, since the checkpoint was never trained past real depth-3 overlap —
  read this as a relative/ordering result only. Only 50 scenes scored so far; a
  150-scene rerun is planned but not yet executed.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
