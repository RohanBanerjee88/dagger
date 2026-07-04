#!/usr/bin/env python3
"""Phase 0 entrypoint: mixture -> oracle regions -> copied-solo output -> metrics.

Runs the whole plumbing path with *oracle* diarization and the null extractor
(no learning yet) over a handful of real mixtures. A dataset loader
(``configs/phase0_*.yaml`` -> LibriMix / WSJ0-2mix) mixes source utterances on
the fly (storage-lean — only the sources live on the mounted volume), derives
ground-truth activity/solo/overlap regions, reconstructs each speaker by copying
solo regions (overlap left empty until Phase 1), and reports SI-SDR overall and
split by solo vs. overlap.

Phase 0 definition of done (CLAUDE.md §5): solo regions score essentially
perfectly. Overlap regions score poorly here *by design* — there is no extractor
yet. Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/run_phase0.py --config configs/phase0_librimix.yaml
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
from dagger.data import Scene, build_dataset
from dagger.data.paths import load_env
from dagger.diarize.oracle import (
    activity_matrix,
    overlap_mixture,
    solo_overlap_regions,
)
from dagger.extract.base import NullExtractor
from dagger.metrics.sisdr import si_sdr_regionwise
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all


def _fmt(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "inf" if v == float("inf") else f"{v:.2f}"


def score_scene(scene: Scene, fade: int) -> list[tuple[str, float, float, float]]:
    """Run the Phase 0 path on one scene; return per-speaker region SI-SDRs."""
    n = scene.mixture.shape[0]
    activity, speakers = activity_matrix(
        scene.segments, num_samples=n, sample_rate=scene.sample_rate, speakers=scene.speakers
    )
    # Phase 0 red flag (§5): masks must line up with the waveform exactly.
    assert activity.shape[1] == n, "activity mask length != waveform length"

    solo, overlap = solo_overlap_regions(activity)
    x = original_mixture(scene.mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")  # untouched overlap mixture

    outputs = reconstruct_all(
        x=x, x_O=x_O, activity=activity, solo=solo,
        embeddings=None, extractor=NullExtractor(), fade=fade,
    )

    results = []
    for i, spk in enumerate(speakers):
        w_Ei, _ = crossfade_windows(solo[i], activity[i], fade=fade)
        interior = np.isclose(w_Ei, 1.0)                       # copy is bit-exact here
        solo_i = solo[i].astype(bool)                          # includes crossfade seams
        overlap_i = activity[i].astype(bool) & overlap.astype(bool)
        results.append((
            spk,
            si_sdr_regionwise(outputs[i], scene.sources[i], interior),
            si_sdr_regionwise(outputs[i], scene.sources[i], solo_i),
            si_sdr_regionwise(outputs[i], scene.sources[i], overlap_i),
        ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0_librimix.yaml")
    args = parser.parse_args()

    load_env()  # populate DAGGER_DATA_ROOT / credentials from .env if present
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_rate = int(cfg["sample_rate"])
    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * sample_rate))

    dataset = build_dataset(cfg)
    print(f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  @ {sample_rate} Hz  fade={fade} samples")
    print()
    header = f"{'scene':>12} | {'speaker':>8} | {'solo-interior':>13} | {'solo+seams':>10} | {'overlap':>8}"
    print(header)
    print("-" * len(header))

    solo_scores: list[float] = []
    overlap_scores: list[float] = []
    for scene in dataset:
        for spk, interior_s, solo_s, ov_s in score_scene(scene, fade):
            print(f"{scene.name[:12]:>12} | {spk:>8} | {_fmt(interior_s):>13} | {_fmt(solo_s):>10} | {_fmt(ov_s):>8}")
            if not np.isnan(solo_s) and solo_s != float("inf"):
                solo_scores.append(solo_s)
            if not np.isnan(ov_s):
                overlap_scores.append(ov_s)

    print()
    if solo_scores:
        print(f"mean solo+seams SI-SDR (finite): {np.mean(solo_scores):.2f} dB")
    if overlap_scores:
        print(f"mean overlap    SI-SDR         : {np.mean(overlap_scores):.2f} dB (poor by design; no extractor yet)")
    print()
    print("Phase 0 check (CLAUDE.md §5): 'solo-interior' reads 'inf' — solo audio is")
    print("copied bit-exact from the mixture. 'overlap' is poor by design until Phase 1")
    print("adds the extractor G.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
