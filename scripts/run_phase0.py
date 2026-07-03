#!/usr/bin/env python3
"""Phase 0 entrypoint: mixture -> oracle regions -> copied-solo output -> metrics.

Runs the whole plumbing path with *oracle* diarization and the null extractor
(no learning yet). It builds a synthetic scene from ``configs/phase0.yaml``,
derives the ground-truth activity/solo/overlap regions, reconstructs each
speaker by copying solo regions (overlap left empty until Phase 1), and reports
SI-SDR overall and split by solo vs. overlap.

Phase 0 definition of done (CLAUDE.md §5): solo regions score essentially
perfectly. Overlap regions score poorly here *by design* — there is no
extractor yet. Reproduce with::

    python scripts/run_phase0.py --config configs/phase0.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.audio.provenance import original_mixture
from dagger.diarize.oracle import (
    Segment,
    activity_matrix,
    overlap_mixture,
    solo_overlap_regions,
)
from dagger.extract.base import NullExtractor
from dagger.metrics.sisdr import si_sdr_regionwise
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all


def build_scene(cfg: dict) -> tuple[np.ndarray, np.ndarray, list[Segment], list[str], int]:
    """Synthesize per-speaker clean sources, the mixture, and oracle segments."""
    sr = int(cfg["sample_rate"])
    scene = cfg["scene"]
    n = int(round(float(scene["duration"]) * sr))
    t = np.arange(n) / sr

    segments: list[Segment] = []
    speakers: list[str] = []
    sources = []
    for spk in scene["speakers"]:
        speakers.append(spk["name"])
        active = np.zeros(n, dtype=np.float64)
        for start, dur in spk["segments"]:
            segments.append(Segment(spk["name"], float(start), float(dur)))
            lo = int(round(float(start) * sr))
            hi = int(round((float(start) + float(dur)) * sr))
            active[max(0, lo):min(n, hi)] = 1.0
        # a clean tone, gated to the speaker's active spans
        sources.append(np.sin(2 * np.pi * float(spk["freq"]) * t) * active)

    sources = np.stack(sources, axis=0)  # [S, T], row i aligned to speakers[i]
    mixture = sources.sum(axis=0)
    return mixture, sources, segments, speakers, sr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mixture, sources, segments, speakers, sr = build_scene(cfg)
    n = mixture.shape[0]
    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * sr))

    activity, speakers = activity_matrix(segments, num_samples=n, sample_rate=sr, speakers=speakers)
    solo, overlap = solo_overlap_regions(activity)

    x = original_mixture(mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")  # untouched overlap mixture

    outputs = reconstruct_all(
        x=x, x_O=x_O, activity=activity, solo=solo,
        embeddings=None, extractor=NullExtractor(), fade=fade,
    )

    print(f"scene: {len(speakers)} speakers, {n} samples @ {sr} Hz, fade={fade} samples")
    print(f"overlap frames: {int(overlap.sum())} / {n}")
    print()
    header = (
        f"{'speaker':>8} | {'solo-interior':>13} | {'solo+seams':>10} | {'overlap':>8}"
    )
    print(header)
    print("-" * len(header))
    for i, spk in enumerate(speakers):
        w_Ei, _ = crossfade_windows(solo[i], activity[i], fade=fade)
        interior = np.isclose(w_Ei, 1.0)              # copy is bit-exact here
        solo_i = solo[i].astype(bool)                  # includes crossfade seams
        overlap_i = activity[i].astype(bool) & overlap.astype(bool)

        interior_score = si_sdr_regionwise(outputs[i], sources[i], interior)
        solo_score = si_sdr_regionwise(outputs[i], sources[i], solo_i)
        ov_score = si_sdr_regionwise(outputs[i], sources[i], overlap_i)

        def fmt(v: float) -> str:
            if np.isnan(v):
                return "n/a"
            return "inf" if v == float("inf") else f"{v:.2f}"

        print(f"{spk:>8} | {fmt(interior_score):>13} | {fmt(solo_score):>10} | {fmt(ov_score):>8}")

    print()
    print("Phase 0 check (SI-SDR, dB): 'solo-interior' should read 'inf' — solo audio")
    print("is copied bit-exact from the mixture. 'solo+seams' is slightly lower where")
    print("the crossfade tapers (no extractor yet to fill w_Oi). 'overlap' is poor by")
    print("design until Phase 1 adds the extractor G.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
