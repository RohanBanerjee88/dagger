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

## Status — Phase 0 (plumbing, oracle diarization)

Implemented and tested:

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

## Quickstart

```bash
pip install -e .            # numpy + pyyaml (core)
pytest                      # incl. the no-residual-in-audio-path guard
```

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

## Where things are going

The full phase-by-phase plan, the mathematically-settled facts, and the
guardrails live in [`CLAUDE.md`](CLAUDE.md) — the single source of truth for this
repo. The proof behind every module is in `docs/theory.pdf`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
