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

## Status — Phase 0 (plumbing) + Phase 1 (identity conditioning) done

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
python scripts/run_phase0.py --config configs/phase0_librimix.yaml
python scripts/run_phase0.py --config configs/phase0_wsj0mix.yaml
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
python scripts/run_phase1.py --config configs/phase1_librimix_3spk_eval.yaml
```

This requires the trained checkpoint at `checkpoints/phase1/proposed_librimix_3spk.pt`
(see [Pretrained weights](#pretrained-weights)), or retrain it yourself:

```bash
python scripts/train_phase1.py --config configs/phase1_librimix_3spk_train.yaml --system proposed
python scripts/train_phase1.py --config configs/phase1_librimix_3spk_train.yaml --system blind
```

### Pretrained weights

Both checkpoints are hosted on the Hugging Face Hub under Apache-2.0 (same
license as this repo). They do **not** bundle NVIDIA's TitaNet-Large speaker
encoder (`nvidia/speakerverification_en_titanet_large`, CC-BY-4.0) — that
model is loaded separately at runtime via NeMo; see `NOTICE` for the full
attribution.

| Checkpoint | HF repo | Used by |
|---|---|---|
| Phase 1 "proposed" extractor | [`AdityaAA2004/dagger-phase1-proposed-librimix-3spk`](https://huggingface.co/AdityaAA2004/dagger-phase1-proposed-librimix-3spk) | `configs/phase1_librimix_3spk_eval.yaml`, `configs/phase2_librimix_3spk_eval.yaml` |
| Phase 2 fine-tuned extractor | [`AdityaAA2004/dagger-phase2-proposed-librimix-3spk-finetuned`](https://huggingface.co/AdityaAA2004/dagger-phase2-proposed-librimix-3spk-finetuned) | `configs/phase2_librimix_3spk_eval_finetuned.yaml` |

Fetch and cache either checkpoint locally with:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="AdityaAA2004/dagger-phase1-proposed-librimix-3spk",
    filename="proposed_librimix_3spk.pt",
)

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

## Where things are going

The full phase-by-phase plan, the mathematically-settled facts, and the
guardrails live in [`CLAUDE.md`](CLAUDE.md) — the single source of truth for this
repo. The proof behind every module is in
[`docs/diarization_full_mathematical_theory.pdf`](docs/diarization_full_mathematical_theory.pdf).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
