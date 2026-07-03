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

Still open in Phase 0: real dataset loaders (WSJ0-2mix, LibriMix) — the runnable
demo below uses a self-contained synthetic scene instead.

## Quickstart

```bash
pip install -e .            # numpy + pyyaml
python scripts/run_phase0.py --config configs/phase0.yaml
pytest                      # incl. the no-residual-in-audio-path guard
```

The Phase 0 run reports SI-SDR split by region: solo interiors are recovered
bit-exactly (`inf`), while overlap regions are poor by design until Phase 1 adds
the extractor `G`.

## Where things are going

The full phase-by-phase plan, the mathematically-settled facts, and the
guardrails live in [`CLAUDE.md`](CLAUDE.md) — the single source of truth for this
repo. The proof behind every module is in `docs/theory.pdf`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
