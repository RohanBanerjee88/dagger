#!/usr/bin/env python3
"""Phase 2 entrypoint: the depth-stratified accumulation-free experiment.

Compares four systems, all conditioning the SAME trained extractor `G` from
Phase 1 (no retraining -- see CLAUDE.md §5 Phase 2), on scenes built by the
Phase 2 scene scheduler (:func:`dagger.data.mixing.schedule_solo_then_overlap`,
enabled via ``dataset.placement: scheduled``) so every speaker gets guaranteed
solo time AND the scene reaches a genuine depth-3 overlap:

* ``no_recursion``      -- Phase 1's proposed path, unchanged (the
                            accumulation-free baseline with no gate/refinement).
* ``ungated_deflation``  -- the deliberate anti-pattern (CLAUDE.md §1):
                            iteratively subtracts each estimate from a running
                            residual and re-extracts from it.
* ``gated_deflation``    -- same, but a confidence-gate rejection leaves the
                            residual untouched for the next speaker.
* ``coarse_to_fine``     -- recursion refines the embedding only; audio always
                            comes from the unmodified, guarded
                            ``reconstruct_all`` (this is "ours").

Every metric is stratified by overlap depth |K| (CLAUDE.md §5: "stratify every
metric by overlap depth |K|" -- that's the evidence, not aggregate averages).
Writes a long-format CSV (``scene, speaker, system, depth, si_sdr``) plus a
summary ``.md`` to ``results/phase2_<dataset>_<n_src>spk.{csv,md}``. Reproduce
with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/run_phase2.py \\
        --config configs/phase2_librimix_3spk_eval.yaml
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
from dagger.diarize.oracle import activity_matrix, overlap_depth, overlap_mixture, solo_overlap_regions
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.gate.confidence import confidence_gate
from dagger.metrics.sisdr import si_sdr_by_depth
from dagger.reconstruct.deflation import reconstruct_all_deflation
from dagger.reconstruct.stitch import reconstruct_all
from dagger.refine.coarse_to_fine import refine_embeddings

SYSTEMS = ("no_recursion", "ungated_deflation", "gated_deflation", "coarse_to_fine")


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_gate_fn(embeddings, variances, activity, overlap, encoder, sample_rate, gate_cfg):
    num_speakers = embeddings.shape[0]

    def gate_fn(i: int, estimate: np.ndarray):
        others = [embeddings[j] for j in range(num_speakers) if j != i]
        expected_active = activity[i].astype(bool) & overlap.astype(bool)
        return confidence_gate(
            estimate, sample_rate, embeddings[i], others, encoder,
            variances[i], expected_active,
            tau_margin=gate_cfg["tau_margin"],
            max_mean_variance=gate_cfg["max_mean_variance"],
            min_vad_coverage=gate_cfg["min_vad_coverage"],
            max_artifact_score=gate_cfg["max_artifact_score"],
        )

    return gate_fn


def score_scene(
    scene: Scene,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    encoder: TitaNetEncoder,
    extractor: TFGridNetCrossAttnExtractor,
    gate_cfg: dict,
    refine_rounds: int,
) -> list[dict]:
    """Run all four Phase 2 systems on one scene; return one row per (speaker, depth, system)."""
    n = scene.mixture.shape[0]
    activity, speakers = activity_matrix(
        scene.segments, num_samples=n, sample_rate=scene.sample_rate, speakers=scene.speakers
    )
    solo, overlap = solo_overlap_regions(activity)
    depth = overlap_depth(activity)
    x = original_mixture(scene.mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")
    num_speakers = len(speakers)

    enrollments = [
        enroll_speaker(
            scene.mixture, solo[i], activity[i], scene.sample_rate,
            encoder, k=enroll_k, min_clip_ms=min_clip_ms,
        )
        for i in range(num_speakers)
    ]
    embeddings = np.stack([e.embedding for e in enrollments], axis=0)
    variances = np.stack([e.variance for e in enrollments], axis=0)

    # Deflation processing order: ascending mean enrollment variance, i.e. the
    # most-confidently-enrolled speaker goes first (order is load-bearing only
    # for the deflation systems -- coarse_to_fine's batched refinement is
    # order-independent, see dagger.refine.coarse_to_fine's docstring).
    order = sorted(range(num_speakers), key=lambda i: float(np.mean(variances[i])))

    outputs: dict[str, np.ndarray] = {}
    outputs["no_recursion"] = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)
    outputs["ungated_deflation"], _ = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=None, fade=fade,
    )
    gate_fn = _make_gate_fn(embeddings, variances, activity, overlap, encoder, scene.sample_rate, gate_cfg)
    outputs["gated_deflation"], _ = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=gate_fn, fade=fade,
    )
    refined_embeddings, _ = refine_embeddings(
        x, x_O, activity, solo, embeddings, variances, extractor, encoder, scene.sample_rate,
        rounds=refine_rounds, fade=fade,
        tau_margin=gate_cfg["tau_margin"], max_mean_variance=gate_cfg["max_mean_variance"],
        min_vad_coverage=gate_cfg["min_vad_coverage"], max_artifact_score=gate_cfg["max_artifact_score"],
    )
    outputs["coarse_to_fine"] = reconstruct_all(
        x, x_O, activity, solo, refined_embeddings, extractor, fade=fade
    )

    rows = []
    for system_name in SYSTEMS:
        system_outputs = outputs[system_name]
        for i, spk in enumerate(speakers):
            by_depth = si_sdr_by_depth(system_outputs[i], scene.sources[i], depth)
            for k, score in by_depth.items():
                rows.append({
                    "scene": scene.name, "speaker": spk, "system": system_name,
                    "depth": k, "si_sdr": score,
                })
    return rows


def _write_results(rows: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    fieldnames = ["scene", "speaker", "system", "depth", "si_sdr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    depths = sorted({r["depth"] for r in rows})
    lines = [f"# Phase 2 results -- {stem}", "", f"rows scored: {len(rows)}", ""]
    header = "| system | " + " | ".join(f"depth {k}" for k in depths) + " |"
    sep = "|---" * (len(depths) + 1) + "|"
    lines += [header, sep]
    means: dict[str, dict[int, float]] = {}
    for system_name in SYSTEMS:
        per_depth = {}
        for k in depths:
            scores = [
                r["si_sdr"] for r in rows
                if r["system"] == system_name and r["depth"] == k and np.isfinite(r["si_sdr"])
            ]
            per_depth[k] = float(np.mean(scores)) if scores else float("nan")
        means[system_name] = per_depth
        cells = " | ".join(f"{per_depth[k]:.2f}" for k in depths)
        lines.append(f"| {system_name} | {cells} |")

    lines += ["", "## Ordering check (3+ speaker overlaps, deepest available depth)"]
    if depths:
        deepest = depths[-1]
        ctf, gated, ungated = (
            means["coarse_to_fine"][deepest], means["gated_deflation"][deepest], means["ungated_deflation"][deepest],
        )
        ok = ctf >= gated > ungated
        lines.append(
            f"depth {deepest}: coarse_to_fine={ctf:.2f} gated_deflation={gated:.2f} "
            f"ungated_deflation={ungated:.2f} -- ordering holds: {ok}"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} and {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_librimix_3spk_eval.yaml")
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
    gate_cfg = cfg.get("gate", {})
    refine_rounds = int(cfg.get("refine", {}).get("rounds", 2))

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(checkpoint_path=checkpoint_path, device=device, **extractor_cfg)

    print(f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  @ {sample_rate} Hz  fade={fade} samples")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for scene in dataset:
        try:
            scene_rows = score_scene(
                scene, fade, enroll_k, min_clip_ms, encoder, extractor, gate_cfg, refine_rounds,
            )
        except NoSoloRegionError as exc:
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")
            continue
        rows.extend(scene_rows)
        print(f"scored scene {scene.name!r} ({len(scene_rows)} rows)")

    if skipped:
        print(
            f"[enroll] skipped {len(skipped)}/{len(dataset)} scenes during enrollment "
            f"(see per-scene messages above): {[name for name, _ in skipped]}"
        )

    n_src = cfg["dataset"].get("n_src", 2)
    stem = f"phase2_{cfg['dataset']['name']}_{n_src}spk"
    results_dir = Path(cfg.get("eval", {}).get("results_dir", "results"))
    _write_results(rows, results_dir, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
