#!/usr/bin/env python3
"""Embedding-sensitivity probe for the proposed extractor (CLAUDE.md Phase 1).

Diagnoses the conditioning-collapse issue on a trained checkpoint by asking
three questions, per scene, on the real overlap mixture ``x_O``:

1. **Is the conditioning pathway alive?** Compare ``G(x_O, e_i)`` across the
   scene's speakers and against a random embedding. Near-identical waveforms
   (relative difference ~0) mean the embedding is ignored -> the collapse.
2. **Is the output just the mixture?** SI-SDR(G(x_O, e_i), x_O) over overlap
   samples. SI-SDR is scale-invariant, so a high value (>~15 dB) means the
   output is ~a scaled copy of x_O -> passthrough.
3. **Does the embedding steer toward the right speaker?** The matrix
   SI-SDR(G(x_O, e_i), s_j) over overlap: a working conditioner has
   diagonal > off-diagonal.

Interpretation (printed at the end):
- pathway dead            -> escalate: aux speaker-consistency loss / mixture dropout
- alive but not steering  -> optimization problem: lower LR, grad clipping, retrain
- steering                -> conditioning works; it just needs more training

Reproduce with::

    DAGGER_DATA_ROOT=/kaggle/working/data python scripts/probe_phase1_conditioning.py \\
        --config configs/phase1_librimix_3spk_eval.yaml --scenes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.audio.provenance import original_mixture
from dagger.data import build_dataset
from dagger.data.paths import load_env
from dagger.diarize.oracle import activity_matrix, overlap_mixture, solo_overlap_regions
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.metrics.sisdr import si_sdr_regionwise


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rel_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Relative L2 difference between two waveforms over masked samples."""
    m = mask.astype(bool)
    a, b = a[m], b[m]
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_librimix_3spk_eval.yaml")
    parser.add_argument("--checkpoint", default=None, help="override the config's checkpoint path")
    parser.add_argument("--scenes", type=int, default=5, help="probe-able scenes to use")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = _device(args.device)

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = args.checkpoint or extractor_cfg.pop("checkpoint", None)
    extractor_cfg.pop("checkpoint", None)
    extractor_cfg.pop("blind_checkpoint", None)
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        print(f"checkpoint not found: {checkpoint_path!r} (pass --checkpoint)")
        return 1
    embed_dim = int(extractor_cfg.get("embed_dim", 192))

    load_env()
    dataset = build_dataset(cfg)
    enroll_cfg = cfg.get("enroll", {})
    enroll_k = int(enroll_cfg.get("k", 3))
    min_clip_ms = float(enroll_cfg.get("min_clip_ms", 500.0))

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(
        checkpoint_path=checkpoint_path, device=device, **extractor_cfg
    )
    rng = np.random.default_rng(0)

    rel_diffs_speaker: list[float] = []   # G(x_O,e_i) vs G(x_O,e_j), i != j
    rel_diffs_random: list[float] = []    # G(x_O,e_i) vs G(x_O, random)
    passthrough_scores: list[float] = []  # SI-SDR(G(x_O,e_i), x_O) on overlap
    diag_scores: list[float] = []         # SI-SDR(G(x_O,e_i), s_i) on overlap
    offdiag_scores: list[float] = []      # SI-SDR(G(x_O,e_i), s_j) on overlap, i != j

    probed = 0
    for scene in dataset:
        if probed >= args.scenes:
            break
        n = scene.mixture.shape[0]
        activity, speakers = activity_matrix(
            scene.segments, num_samples=n, sample_rate=scene.sample_rate,
            speakers=scene.speakers,
        )
        solo, overlap = solo_overlap_regions(activity)
        if not overlap.any():
            print(f"[probe] skipping scene {scene.name!r}: no overlap samples")
            continue
        x = original_mixture(scene.mixture, label="x")
        x_O = overlap_mixture(x, overlap, label="x_O")

        try:
            embeddings = np.stack(
                [
                    enroll_speaker(
                        scene.mixture, solo[i], activity[i], scene.sample_rate,
                        encoder, k=enroll_k, min_clip_ms=min_clip_ms,
                    ).embedding
                    for i in range(len(speakers))
                ],
                axis=0,
            )
        except NoSoloRegionError as exc:
            print(f"[probe] skipping scene {scene.name!r}: {exc}")
            continue

        outputs = [extractor.extract(x_O, embeddings[i]) for i in range(len(speakers))]
        y_random = extractor.extract(x_O, rng.normal(size=embed_dim))

        ov = overlap.astype(bool)
        for i in range(len(speakers)):
            rel_diffs_random.append(_rel_diff(outputs[i], y_random, ov))
            passthrough_scores.append(si_sdr_regionwise(outputs[i], np.asarray(x_O), ov))
            for j in range(len(speakers)):
                mask_j = activity[j].astype(bool) & ov
                score = si_sdr_regionwise(outputs[i], scene.sources[j], mask_j)
                if np.isfinite(score):
                    (diag_scores if i == j else offdiag_scores).append(score)
                if i < j:
                    rel_diffs_speaker.append(_rel_diff(outputs[i], outputs[j], ov))

        probed += 1
        print(
            f"[probe] scene {scene.name!r}: "
            f"reldiff(e_i,e_j)={np.mean(rel_diffs_speaker[-len(speakers)*(len(speakers)-1)//2:]):.4f}  "
            f"passthrough={np.mean(passthrough_scores[-len(speakers):]):.2f} dB"
        )

    if probed == 0:
        print("no probe-able scenes (all skipped); increase dataset limit or check data root")
        return 1

    rd_spk = float(np.mean(rel_diffs_speaker))
    rd_rnd = float(np.mean(rel_diffs_random))
    pt = float(np.mean(passthrough_scores))
    dg = float(np.mean(diag_scores)) if diag_scores else float("nan")
    od = float(np.mean(offdiag_scores)) if offdiag_scores else float("nan")

    print()
    print(f"=== probe summary over {probed} scenes ===")
    print(f"mean rel. output diff, speaker-vs-speaker embeddings: {rd_spk:.4f}")
    print(f"mean rel. output diff, speaker-vs-random embedding:   {rd_rnd:.4f}")
    print(f"mean SI-SDR(output, x_O) on overlap (passthrough):    {pt:.2f} dB")
    print(f"mean SI-SDR(output_i, s_i) on overlap (diagonal):     {dg:.2f} dB")
    print(f"mean SI-SDR(output_i, s_j) on overlap (off-diag):     {od:.2f} dB")
    print()

    if rd_spk < 1e-3:
        print("VERDICT: conditioning pathway DEAD -- outputs ignore the embedding.")
        print("  -> escalate: auxiliary speaker-consistency loss and/or mixture dropout.")
    elif np.isfinite(dg) and np.isfinite(od) and dg - od >= 1.0:
        print("VERDICT: conditioning STEERS toward the right speaker (diag > off-diag).")
        print("  -> the architecture works; train longer / on more data.")
    else:
        print("VERDICT: conditioning ALIVE but not steering (outputs differ, no diag margin).")
        print("  -> optimization problem: lower LR (e.g. 3e-4), add grad clipping, retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
