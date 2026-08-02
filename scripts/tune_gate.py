#!/usr/bin/env python3
"""Confidence-gate threshold selection on a dev split (CLAUDE.md §2, Phase 2).

The four thresholds in every Phase 2 eval config (``tau_margin``,
``max_mean_variance``, ``min_vad_coverage``, ``max_artifact_score``) have never
been chosen -- they are defaults that have ridden along since the first Phase 2
run. This script picks them, and deliberately does NOT pick them by maximizing
SI-SDR.

Why not SI-SDR: ``gated_deflation`` interpolates between ``ungated_deflation``
(a gate that accepts everything) and ``no_recursion`` (a gate that rejects
everything -- the residual never updates, so every speaker extracts from a
pristine ``x_O``). Its SI-SDR is therefore a dial between the worst system and
the best one, and "optimizing" it just slides a comparison baseline toward the
system it exists to lose to. Tighten far enough and it beats ``coarse_to_fine``,
destroying the ordering claim for a reason that has nothing to do with the
theory. So each threshold is swept against its own intrinsic criterion:

* ``max_mean_variance`` (``V_i``) -- against DELIBERATELY CONTAMINATED
  enrollment. Each speaker is enrolled twice: honestly from its solo region,
  and again from its *overlap* region passed in as though it were solo. Only
  ``V_i`` can catch this: a contaminated enrollment produces a confident margin,
  which is exactly why CLAUDE.md §2 says "the gate can't check its own
  enrollment" and why this check short-circuits the others.
* ``tau_margin`` (``M_i``) -- against DELIBERATELY SWAPPED conditioning. Every
  speaker's output is recomputed with a neighbour's embedding, then gated as if
  it were still that speaker. A working margin separates the two populations.
* ``min_vad_coverage`` / ``max_artifact_score`` -- no labelled populations
  exist (there is no ground truth for what fraction of estimates *should* fail
  spectral flatness), so these are reported as rejection rates on healthy data:
  the useful question is whether they fire at all, and whether they cut into
  normal operation.

The first two are detection problems, so each candidate value yields a
(detection, false-rejection) pair -- an ROC sweep, with Youden's J
(detection - false_rejection) reported as the suggested operating point.

Sweeps are INDEPENDENT, not a joint grid: a joint sweep needs one objective,
and the only one available is the SI-SDR that must not be used.

Note the sweep is exact for a first/round-0 decision, which is what this script
measures. Downstream deflation decisions change which estimates enter the
residual, hence the audio, hence later diagnostics -- so confirm a chosen
config with one full ``scripts/run_phase2.py`` run rather than trusting the
sweep for those.

    DAGGER_DATA_ROOT=/kaggle/working/data python scripts/tune_gate.py \\
        --config configs/phase2_gate_tune_dev.yaml
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
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.gate.confidence import gate_diagnostics
from dagger.reconstruct.stitch import reconstruct_all

DIAGNOSTIC_FIELDS = [
    "scene", "speaker", "m", "population",
    "mean_variance", "margin", "vad_coverage", "artifact_score",
]

# Which population each threshold is supposed to reject, and which it must not.
# "honest"/"correct" are the healthy populations; a threshold that rejects them
# is costing real quality.
HONEST, CONTAMINATED = "honest", "contaminated"
CORRECT, SWAPPED = "correct", "swapped"

# Below this Youden's J, the best candidate is not meaningfully separating the
# two populations and no value in the grid is a real detector. Reported as a
# refusal to recommend rather than a low-confidence recommendation: a sweep over
# two identical distributions still produces a "best" row, and copying it into a
# config would launder noise into a threshold. 0.1 is a deliberately low bar --
# it is asking for "better than nearly nothing", not for a good detector.
MIN_USEFUL_YOUDEN_J = 0.1


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _contaminated_mask(activity_i: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """The speaker's OVERLAP region, shaped like a solo mask.

    Handed to ``enroll_speaker`` in place of ``solo[i]``, this enrolls the
    speaker from audio that contains other voices -- the exact failure mode
    ``V_i`` exists to catch, and the one a real diarizer produces when it
    mislabels an overlap segment as single-speaker.
    """
    return (activity_i > 0).astype(np.float64) * (overlap > 0).astype(np.float64)


def measure_scene(
    scene: Scene,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    encoder: TitaNetEncoder,
    extractor: TFGridNetCrossAttnExtractor,
) -> list[dict]:
    """Measure all four gate diagnostics on one scene, for every population.

    Returns one row per (speaker, population). Populations are:
    ``honest``/``contaminated`` (differ in how the speaker was enrolled) and
    ``correct``/``swapped`` (differ in which embedding the extractor was
    conditioned on). ``honest`` and ``correct`` describe the same healthy run
    and share its numbers; they are emitted as separate rows so each sweep can
    select its own comparison pair without re-deriving it.
    """
    n = scene.mixture.shape[0]
    activity, speakers = activity_matrix(
        scene.segments, num_samples=n, sample_rate=scene.sample_rate, speakers=scene.speakers
    )
    solo, overlap = solo_overlap_regions(activity)
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

    # Contaminated enrollment: same speaker, same K, but drawn from overlap.
    # A speaker with too little overlap to form a clip simply has no
    # contaminated counterpart -- recorded as absent rather than faked.
    contaminated_variances: list[np.ndarray | None] = []
    for i in range(num_speakers):
        try:
            result = enroll_speaker(
                scene.mixture, _contaminated_mask(activity[i], overlap), activity[i],
                scene.sample_rate, encoder, k=enroll_k, min_clip_ms=min_clip_ms,
            )
        except NoSoloRegionError:
            contaminated_variances.append(None)
        else:
            contaminated_variances.append(result.variance)

    # Correct conditioning, and the same speakers re-extracted with a
    # neighbour's embedding. Rolling the embedding matrix keeps speaker i's own
    # windows and activity while feeding G the wrong identity -- the
    # off-diagonal of the Phase 1 conditioning probe, reused as a gate stimulus.
    outputs_correct = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)
    outputs_swapped = (
        reconstruct_all(x, x_O, activity, solo, np.roll(embeddings, 1, axis=0), extractor, fade=fade)
        if num_speakers > 1 else None
    )

    rows: list[dict] = []
    for i, spk in enumerate(speakers):
        others = [embeddings[j] for j in range(num_speakers) if j != i]
        expected_active = activity[i].astype(bool) & overlap.astype(bool)

        def _row(population: str, estimate: np.ndarray, variance: np.ndarray) -> dict:
            diagnostics = gate_diagnostics(
                estimate, scene.sample_rate, embeddings[i], others, encoder,
                variance, expected_active,
            )
            return {
                "scene": scene.name, "speaker": spk, "m": num_speakers,
                "population": population,
                "mean_variance": diagnostics.mean_variance,
                "margin": diagnostics.margin,
                "vad_coverage": diagnostics.vad_coverage,
                "artifact_score": diagnostics.artifact_score,
            }

        healthy = _row(HONEST, outputs_correct[i], variances[i])
        rows.append(healthy)
        rows.append({**healthy, "population": CORRECT})

        if contaminated_variances[i] is not None:
            rows.append(_row(CONTAMINATED, outputs_correct[i], contaminated_variances[i]))
        if outputs_swapped is not None:
            rows.append(_row(SWAPPED, outputs_swapped[i], variances[i]))
    return rows


def _values(rows: list[dict], population: str, field: str) -> np.ndarray:
    values = [r[field] for r in rows if r["population"] == population]
    return np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)


def _detection_sweep(
    rows: list[dict], field: str, grid: list[float], *, healthy: str, faulty: str, reject_below: bool
) -> list[str]:
    """ROC sweep for a threshold with two labelled populations.

    ``reject_below`` says which side of the threshold is a rejection:
    ``tau_margin`` rejects values BELOW it, ``max_mean_variance`` rejects values
    ABOVE it. Detection is the fraction of the faulty population rejected;
    false rejection is the fraction of the healthy population rejected.
    """
    healthy_values = _values(rows, healthy, field)
    faulty_values = _values(rows, faulty, field)
    direction = "<" if reject_below else ">"
    lines = [
        "", f"### `{field}` -- {faulty} vs {healthy}", "",
        f"(n={len(healthy_values)} {healthy}, n={len(faulty_values)} {faulty}; "
        f"a value {direction} the threshold is rejected. detection = {faulty} caught, "
        f"false rej. = {healthy} wrongly rejected. J = detection - false rej.)", "",
        "| threshold | detection | false rej. | J |",
        "|---|---|---|---|",
    ]
    if healthy_values.size == 0 or faulty_values.size == 0:
        lines.append("| -- | (a population is empty; nothing to sweep) | -- | -- |")
        return lines

    best = None
    for threshold in grid:

        if reject_below:
            detection = float(np.mean(faulty_values < threshold))
            false_rejection = float(np.mean(healthy_values < threshold))
        else:
            detection = float(np.mean(faulty_values > threshold))
            false_rejection = float(np.mean(healthy_values > threshold))
        youden = detection - false_rejection
        if best is None or youden > best[1]:
            best = (threshold, youden)
        lines.append(
            f"| {threshold:g} | {100 * detection:.1f}% | {100 * false_rejection:.1f}% | {youden:+.3f} |"
        )

    lines += [
        "",
        f"population medians -- {healthy}: {np.median(healthy_values):.5f}, "
        f"{faulty}: {np.median(faulty_values):.5f}",
    ]
    if best[1] < MIN_USEFUL_YOUDEN_J:
        # Guard against the quiet failure: if the fault fixture did not actually
        # produce a different distribution, every candidate scores J~0 and the
        # "best" one is just whichever came first in the grid. Recommending it
        # would launder noise into a config value.
        lines += [
            "",
            f"**NO USABLE THRESHOLD.** The best candidate reaches only J = {best[1]:+.3f} "
            f"(< {MIN_USEFUL_YOUDEN_J}), i.e. this diagnostic barely separates {faulty} from "
            f"{healthy} at any value in the grid, so no threshold here would be a real "
            "detector. Do NOT copy a value out of this table. Check first whether the fault "
            f"fixture is doing its job (are the two medians above actually different?) and "
            "whether the grid brackets the observed range; only then suspect the diagnostic "
            "itself.",
        ]
        return lines

    lines += [
        "",
        f"**suggested `{field}`: {best[0]:g}** (highest J = {best[1]:+.3f}). Judgement still "
        "applies -- if two candidates are within noise, prefer the one that rejects less "
        "healthy data, since a false rejection costs real quality on every scene while a "
        "missed detection costs only on contaminated ones.",
    ]
    return lines


def _rate_sweep(rows: list[dict], field: str, grid: list[float], *, reject_below: bool) -> list[str]:
    """Rejection-rate sweep for a threshold with no labelled fault population.

    Reports what fraction of HEALTHY estimates each candidate would reject, on
    its own. This is also the only sound way to ask whether the threshold is
    inert: the gate's ``reason`` field records the FIRST check that failed, so a
    threshold can look like it never fires simply because an earlier one fired
    first.
    """
    values = _values(rows, CORRECT, field)
    direction = "<" if reject_below else ">"
    lines = [
        "", f"### `{field}` -- rejection rate on healthy estimates", "",
        f"(n={len(values)}; a value {direction} the threshold is rejected. No fault "
        "population exists for this check, so there is no detection rate to trade "
        "against -- pick a value in the tail that fires on genuine failures without "
        "cutting into normal operation.)", "",
        "| threshold | healthy rejected |",
        "|---|---|",
    ]
    if values.size == 0:
        lines.append("| -- | (no values) |")
        return lines
    for threshold in grid:
        rate = float(np.mean(values < threshold)) if reject_below else float(np.mean(values > threshold))
        lines.append(f"| {threshold:g} | {100 * rate:.1f}% |")
    lines += [
        "",
        f"observed on healthy data: min {values.min():.4f}, p5 {np.percentile(values, 5):.4f}, "
        f"median {np.median(values):.4f}, p95 {np.percentile(values, 95):.4f}, max {values.max():.4f}",
    ]
    return lines


def _current_config_section(rows: list[dict], gate_cfg: dict) -> list[str]:
    """What the thresholds currently in the config do on this dev split."""
    lines = [
        "", "## What the current (untuned) thresholds do here", "",
        "| threshold | value | rejects |",
        "|---|---|---|",
    ]
    checks = [
        ("max_mean_variance", "mean_variance", HONEST, False),
        ("tau_margin", "margin", CORRECT, True),
        ("min_vad_coverage", "vad_coverage", CORRECT, True),
        ("max_artifact_score", "artifact_score", CORRECT, False),
    ]
    for key, field, population, reject_below in checks:
        if key not in gate_cfg:
            continue
        values = _values(rows, population, field)
        if values.size == 0:
            lines.append(f"| `{key}` | {gate_cfg[key]:g} | (no values) |")
            continue
        threshold = float(gate_cfg[key])
        rate = float(np.mean(values < threshold)) if reject_below else float(np.mean(values > threshold))
        lines.append(f"| `{key}` | {threshold:g} | {100 * rate:.1f}% of healthy estimates |")
    return lines


def _write(rows: list[dict], lines: list[str], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} and {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_gate_tune_dev.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_rate = int(cfg["sample_rate"])
    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * sample_rate))
    device = _device(args.device)

    enroll_cfg = cfg.get("enroll", {})
    enroll_k = int(enroll_cfg.get("k", 3))
    min_clip_ms = float(enroll_cfg.get("min_clip_ms", 500.0))

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = extractor_cfg.pop("checkpoint", None)
    gate_cfg = cfg.get("gate", {})
    tune_cfg = cfg.get("tune", {})

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(checkpoint_path=checkpoint_path, device=device, **extractor_cfg)

    # `dataset` may be a single dict or a list of per-depth entries, matching
    # scripts/train_phase1.py's curriculum support -- thresholds tuned at one
    # speaker count would otherwise specialize to it.
    dataset_cfgs = cfg["dataset"] if isinstance(cfg["dataset"], list) else [cfg["dataset"]]

    rows: list[dict] = []
    skipped = 0
    for dataset_cfg in dataset_cfgs:
        dataset = build_dataset({**cfg, "dataset": dataset_cfg})
        print(f"dataset: n_src={dataset_cfg.get('n_src')}  scenes: {len(dataset)}")
        for scene in dataset:
            try:
                scene_rows = measure_scene(
                    scene, fade, enroll_k, min_clip_ms, encoder, extractor,
                )
            except NoSoloRegionError as exc:
                skipped += 1
                print(f"[enroll] skipping scene {scene.name!r}: {exc}")
                continue
            rows.extend(scene_rows)
            print(f"measured scene {scene.name!r} ({len(scene_rows)} rows)")

    if not rows:
        raise SystemExit("no scenes could be measured -- check the dev split config")

    grids = {
        "tau_margin": tune_cfg.get("tau_margin_grid", [-0.2, -0.1, 0.0, 0.1, 0.2]),
        "max_mean_variance": tune_cfg.get("max_mean_variance_grid", [0.01, 0.05, 0.1, 0.5]),
        "min_vad_coverage": tune_cfg.get("min_vad_coverage_grid", [0.0, 0.25, 0.5, 0.75]),
        "max_artifact_score": tune_cfg.get("max_artifact_score_grid", [0.7, 0.8, 0.9, 1.0]),
    }

    counts = {p: sum(1 for r in rows if r["population"] == p) for p in
              (HONEST, CONTAMINATED, CORRECT, SWAPPED)}
    lines = [
        "# Confidence-gate threshold selection (dev split)", "",
        f"rows: {len(rows)}  |  populations: " +
        ", ".join(f"{k}={v}" for k, v in counts.items()) +
        (f"  |  scenes skipped at enrollment: {skipped}" if skipped else ""), "",
        "Thresholds are swept INDEPENDENTLY, each against what it is meant to detect -- "
        "never jointly against SI-SDR, which for `gated_deflation` is a dial between "
        "`ungated_deflation` and `no_recursion` rather than a quality measure. "
        "See this script's module docstring.", "",
        "## Detection sweeps (labelled populations)",
    ]
    lines += _detection_sweep(
        rows, "mean_variance", grids["max_mean_variance"],
        healthy=HONEST, faulty=CONTAMINATED, reject_below=False,
    )
    lines += _detection_sweep(
        rows, "margin", grids["tau_margin"],
        healthy=CORRECT, faulty=SWAPPED, reject_below=True,
    )
    lines += ["", "## Rate sweeps (no fault population)"]
    lines += _rate_sweep(rows, "vad_coverage", grids["min_vad_coverage"], reject_below=True)
    lines += _rate_sweep(rows, "artifact_score", grids["max_artifact_score"], reject_below=False)
    lines += _current_config_section(rows, gate_cfg)
    lines += [
        "", "## Next step", "",
        "Freeze one set of thresholds, put the SAME values in every Phase 2 eval config "
        "(`gate_cfg` drives both `gated_deflation` and `coarse_to_fine`'s refinement from "
        "one dict, so per-system tuning is not available), and confirm with a single "
        "`scripts/run_phase2.py` run. The sweeps above are exact for a round-0 decision "
        "only -- later deflation decisions change the audio downstream of them.",
    ]

    results_dir = Path(tune_cfg.get("results_dir", "results"))
    stem = "gate_tune"
    tag = tune_cfg.get("tag")
    if tag:
        stem = f"{stem}_{tag}"
    _write(rows, lines, results_dir, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
