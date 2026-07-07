#!/usr/bin/env python3
"""Phase 1 entrypoint: mixture -> oracle regions -> proposed vs. blind -> metrics.

Mirrors ``scripts/run_phase0.py``'s structure (same ``load_env``/
``build_dataset``/per-scene loop), extended with:

* real enrollment (:mod:`dagger.enroll`) instead of ``embeddings=None``;
* the proposed extractor (:class:`~dagger.extract.tfgridnet_crossattn.
  TFGridNetCrossAttnExtractor`) instead of :class:`~dagger.extract.base.NullExtractor`;
* the blind-separation baseline (:class:`~dagger.extract.blind.BlindSeparator`)
  for comparison, permutation-matched via
  :func:`dagger.metrics.sisdr.si_sdr_best_permutation` (its output order is
  unconstrained, unlike the proposed extractor's);
* the eval-only speaker-similarity margin
  (:mod:`dagger.metrics.speaker_similarity`) as a diagnostic column.

Writes the Phase 1 definition-of-done artifact to
``results/phase1_<dataset>_<n_src>spk.{csv,md}``. Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/run_phase1.py \\
        --config configs/phase1_librimix_3spk_eval.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.audio.provenance import original_mixture
from dagger.data import Scene, build_dataset
from dagger.data.paths import load_env
from dagger.diarize.oracle import activity_matrix, overlap_mixture, solo_overlap_regions
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.extract.blind import BlindSeparator
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.metrics.sisdr import si_sdr_best_permutation, si_sdr_regionwise
from dagger.metrics.speaker_similarity import EvalSpeakerEncoder, eval_enroll_and_margin
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all


def _fmt(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "inf" if v == float("inf") else f"{v:.2f}"


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_scene(
    scene: Scene,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    encoder: TitaNetEncoder,
    proposed_extractor: TFGridNetCrossAttnExtractor,
    blind_separator: BlindSeparator | None,
    eval_encoder: EvalSpeakerEncoder | None,
) -> list[dict]:
    """Run the Phase 1 path on one scene; return one row dict per speaker."""
    n = scene.mixture.shape[0]
    activity, speakers = activity_matrix(
        scene.segments, num_samples=n, sample_rate=scene.sample_rate, speakers=scene.speakers
    )
    solo, overlap = solo_overlap_regions(activity)
    x = original_mixture(scene.mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")

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

    proposed_outputs = reconstruct_all(
        x=x, x_O=x_O, activity=activity, solo=solo,
        embeddings=embeddings, extractor=proposed_extractor, fade=fade,
    )

    blind_outputs = None
    if blind_separator is not None:
        blind_raw = blind_separator.separate(x, num_speakers=len(speakers))
        _, perm = si_sdr_best_permutation(blind_raw, scene.sources)
        blind_outputs = blind_raw[list(perm)]

    margins = None
    if eval_encoder is not None:
        try:
            margins = eval_enroll_and_margin(
                scene.mixture, solo, activity, proposed_outputs, scene.sample_rate,
                eval_encoder, k=enroll_k, min_clip_ms=min_clip_ms,
            )
        except NoSoloRegionError as exc:
            # Diagnostic-only: a failure here must not discard this scene's
            # already-computed, real SI-SDR rows -- just drop the margin column.
            print(f"[eval-margin] skipping margin for scene {scene.name!r}: {exc}")

    rows = []
    for i, spk in enumerate(speakers):
        w_Ei, _ = crossfade_windows(solo[i], activity[i], fade=fade)
        interior = np.isclose(w_Ei, 1.0)
        solo_i = solo[i].astype(bool)
        overlap_i = activity[i].astype(bool) & overlap.astype(bool)
        rows.append({
            "scene": scene.name,
            "speaker": spk,
            "solo_interior": si_sdr_regionwise(proposed_outputs[i], scene.sources[i], interior),
            "solo_seams": si_sdr_regionwise(proposed_outputs[i], scene.sources[i], solo_i),
            "overlap_proposed": si_sdr_regionwise(proposed_outputs[i], scene.sources[i], overlap_i),
            "overlap_blind": (
                si_sdr_regionwise(blind_outputs[i], scene.sources[i], overlap_i)
                if blind_outputs is not None else float("nan")
            ),
            "eval_margin": margins[i] if margins is not None else float("nan"),
        })
    return rows


def _write_results(rows: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    fieldnames = [
        "scene", "speaker", "solo_interior", "solo_seams",
        "overlap_proposed", "overlap_blind", "eval_margin",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    proposed_scores = [r["overlap_proposed"] for r in rows if np.isfinite(r["overlap_proposed"])]
    blind_scores = [r["overlap_blind"] for r in rows if np.isfinite(r["overlap_blind"])]
    lines = [
        f"# Phase 1 results -- {stem}", "",
        f"scenes/speakers scored: {len(rows)}", "",
        f"mean overlap SI-SDR (proposed): {np.mean(proposed_scores):.2f} dB" if proposed_scores else "mean overlap SI-SDR (proposed): n/a",
        f"mean overlap SI-SDR (blind): {np.mean(blind_scores):.2f} dB" if blind_scores else "mean overlap SI-SDR (blind): n/a",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} and {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_librimix_3spk_eval.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_rate = int(cfg["sample_rate"])
    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * sample_rate))
    device = _device(args.device)

    dataset = build_dataset(cfg)
    enroll_cfg = cfg.get("enroll", {})
    enroll_k = int(enroll_cfg.get("k", 3))
    min_clip_ms = float(enroll_cfg.get("min_clip_ms", 500.0))

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = extractor_cfg.pop("checkpoint", None)
    blind_checkpoint_path = extractor_cfg.pop("blind_checkpoint", None)

    encoder = TitaNetEncoder(device=device)
    proposed_extractor = TFGridNetCrossAttnExtractor(
        checkpoint_path=checkpoint_path, device=device, **extractor_cfg
    )

    blind_separator = None
    if blind_checkpoint_path and Path(blind_checkpoint_path).is_file():
        blind_separator = BlindSeparator(
            checkpoint_path=blind_checkpoint_path, device=device, **extractor_cfg
        )
    else:
        print("no blind checkpoint found; skipping the blind-baseline column")

    eval_encoder: EvalSpeakerEncoder | None
    try:
        eval_encoder = EvalSpeakerEncoder(device=device)
    except Exception as exc:  # noqa: BLE001 -- diagnostic-only, must not block the run
        print(f"could not load the eval encoder ({exc}); skipping the eval-margin column")
        eval_encoder = None

    print(f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  @ {sample_rate} Hz  fade={fade} samples")
    header = f"{'scene':>12} | {'speaker':>8} | {'solo-interior':>13} | {'solo+seams':>10} | {'overlap(prop)':>13} | {'overlap(blind)':>14} | {'margin':>7}"
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []  # (scene name, reason)
    for scene in dataset:
        try:
            scene_rows = score_scene(
                scene, fade, enroll_k, min_clip_ms, encoder, proposed_extractor,
                blind_separator, eval_encoder,
            )
        except NoSoloRegionError as exc:
            # Only the benign "no usable solo audio" enrollment case is skip-worthy.
            # A plain ValueError (e.g. BlindSeparator's num_speakers mismatch, or the
            # overlap-contamination guard in dagger.enroll.topk) is a real bug and
            # must propagate rather than be silently folded into this skip summary.
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")
            continue
        for row in scene_rows:
            print(
                f"{row['scene'][:12]:>12} | {row['speaker']:>8} | "
                f"{_fmt(row['solo_interior']):>13} | {_fmt(row['solo_seams']):>10} | "
                f"{_fmt(row['overlap_proposed']):>13} | {_fmt(row['overlap_blind']):>14} | "
                f"{_fmt(row['eval_margin']):>7}"
            )
            rows.append(row)

    if skipped:
        print(
            f"[enroll] skipped {len(skipped)}/{len(dataset)} scenes during enrollment "
            f"(see per-scene messages above): {[name for name, _ in skipped]}"
        )

    n_src = cfg["dataset"].get("n_src", 2)
    stem = f"phase1_{cfg['dataset']['name']}_{n_src}spk"
    results_dir = Path(cfg.get("eval", {}).get("results_dir", "results"))
    _write_results(rows, results_dir, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
