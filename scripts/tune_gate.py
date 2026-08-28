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
        --config configs/phase2/experiments/phase2_gate_tune_dev.yaml
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
from dagger.diarize.regions import scene_regions
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.gate.confidence import gate_diagnostics
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all
from dagger.gate import faults
from dagger.gate.artifact import spectral_flatness, vad_coverage
from dagger.gate.confidence import artifact_min_energy_db as read_artifact_min_energy_db
from dagger.gate.confidence import gate_diagnostics
import zlib
DIAGNOSTIC_FIELDS = [
    "scene", "speaker", "m", "population",
    "mean_variance", "margin", "vad_coverage", "artifact_score",
]

# Which population each threshold is supposed to reject, and which it must not.
# "honest"/"correct" are the healthy populations; a threshold that rejects them
# is costing real quality.
HONEST, CONTAMINATED = "honest", "contaminated"
CORRECT, SWAPPED = "correct", "swapped"
#: The Q1b pair (2026-08-25): the same correct/swapped contrast, but with a
#: PERFECT extractor substituted for ``G``. See :func:`measure_scene`.
CLEAN_CORRECT, CLEAN_SWAPPED = "clean_correct", "clean_swapped"

# Below this Youden's J, the best candidate is not meaningfully separating the
# two populations and no value in the grid is a real detector. Reported as a
# refusal to recommend rather than a low-confidence recommendation: a sweep over
# two identical distributions still produces a "best" row, and copying it into a
# config would launder noise into a threshold. 0.1 is a deliberately low bar --
# it is asking for "better than nearly nothing", not for a good detector.
MIN_USEFUL_YOUDEN_J = 0.1

#: Manufactured fault populations (:mod:`dagger.gate.faults`, NOT DEPLOYABLE).
#: Each entry is (severity label, corruption). Severity is baked in here rather tahn taken from config.
#: In this way, the population names in the CSV are self-describing -- a commited 'results' file that says `fault_g_dropout_50` needs no companion
#: config to be read three months later.
VAD_FAULTS = (
    ("dropout_25", lambda s,r,g: faults.drop_span(s, r, 0.25, rng=g)),
    ("dropout_50", lambda s,r,g: faults.drop_span(s, r, 0.50, rng=g)),
    ("dropout_75", lambda s,r,g: faults.drop_span(s, r, 0.75, rng=g)),
    ("dropout_90", lambda s,r,g: faults.drop_span(s, r, 0.90, rng=g)),
    ("quiet_20db", lambda s, r, g: faults.attenuate(s, r, -20.0)),
    # Attenuation is graded ACROSS active_mask's -40 dB floor, measured
    # 2026-08-28: -20 and -30 dB leave coverage at 1.000, -40 dB drops it to
    # 0.116, -50 dB bottoms out at 0.025. The first two are therefore DELIBERATE
    # NEGATIVE CONTROLS -- faults this check should NOT catch -- so a 0% column
    # and an INERT flag for them is the expected result, not a broken fixture.
    # A family sitting entirely on one side of the cliff cannot locate it, which
    # is exactly what the original lone `quiet_30db` did.
    ("quiet_30db", lambda s,r,g: faults.attenuate(s,r, -30.0)),
    ("quiet_40db", lambda s, r, g: faults.attenuate(s, r, -40.0)),
    ("quiet_50db", lambda s, r, g: faults.attenuate(s, r, -50.0)),
)

ARTIFACT_FAULTS = (
    ("snr20", lambda s,r,g: faults.add_noise(s,r, 20.0, rng=g)),
    ("snr10", lambda s,r,g: faults.add_noise(s,r, 10.0, rng=g)),
    ("snr0", lambda s,r,g: faults.add_noise(s,r, 0.0, rng=g)),
    ("holes_50", lambda s,r,g: faults.punch_holes(s,r, 0.50, rng=g)),
    ("holes_75", lambda s,r,g: faults.punch_holes(s,r, 0.75, rng=g))
)

ALL_FAULTS = VAD_FAULTS + ARTIFACT_FAULTS
#: Which healthy population each fault arm is scored against.
FAULT_ARMS = (("fault_g_", CORRECT), ("fault_clean_", CLEAN_CORRECT))

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
    enroll_budget_ms: float | None,
    encoder: TitaNetEncoder,
    extractor: TFGridNetCrossAttnExtractor,
    diarizer=None,
    artifact_min_energy_db: float | None = None
) -> list[dict]:
    """Measure all four gate diagnostics on one scene, for every population.

    Returns one row per (speaker, population). Populations are:
    ``honest``/``contaminated`` (differ in how the speaker was enrolled) and
    ``correct``/``swapped`` (differ in which embedding the extractor was
    conditioned on). ``honest`` and ``correct`` describe the same healthy run
    and share its numbers; they are emitted as separate rows so each sweep can
    select its own comparison pair without re-deriving it.

    ``diarizer`` decides where the regions come from; ``None`` means oracle,
    which is what every config predating this argument gets.

    **Why a real diarizer matters here, and why it is not a tidy-up.** Under
    oracle diarization ``V_i`` is *structurally* zero: the scene scheduler gives
    each speaker exactly one solo run, enrollment therefore draws one clip, and
    the variance across one sample is 0 by definition. So the honest population
    is pinned at exactly 0 while the contaminated fixture below (which enrolls
    from a speaker's *overlap* region, and so can draw several clips) is
    nonzero. A sweep between "identically 0" and "anything at all" separates the
    two perfectly and reports a spectacular Youden's J -- for a property that
    does not exist in deployment.

    The question worth answering is whether contaminated variance clears the
    variance a *real* diarizer's fragmented solo regions already produce on
    honest enrollment (Phase 3 Stage A measured that floor: nonzero in
    1332/1800 decisions, max 3.24e-4). Only a real-diarization honest
    population can answer it, which is why this argument exists.
    """
    if diarizer is None:
        activity, speakers = activity_matrix(
            scene.segments, num_samples=scene.mixture.shape[0],
            sample_rate=scene.sample_rate, speakers=scene.speakers,
        )
        solo, overlap = solo_overlap_regions(activity)
    else:
        regions = scene_regions(scene, diarizer)
        activity, speakers = regions.activity, regions.speakers
        solo, overlap = regions.solo, regions.overlap
    x = original_mixture(scene.mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")
    num_speakers = len(speakers)

    enrollments = [
        enroll_speaker(
            scene.mixture, solo[i], activity[i], scene.sample_rate,
            encoder, k=enroll_k, min_clip_ms=min_clip_ms, budget_ms=enroll_budget_ms,
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
                budget_ms=enroll_budget_ms,
            )
        except NoSoloRegionError:
            contaminated_variances.append(None)
        else:
            contaminated_variances.append(result.variance)

    # `G` is run ONCE per speaker and the raw output kept, so every fault
    # population below is a numpy corruption of a cached tensor rather than
    # another forward pass. Ten populations therefore cost one extraction.
    g_out = np.stack([
        np.asarray(extractor.extract(x_O, embeddings[i]), dtype=np.float64)
        for i in range(num_speakers)
    ])
    x_samples_ = np.asarray(x, dtype=np.float64)
    outputs_correct = np.stack([
        _stitch(x_samples_, g_out[i], solo[i], activity[i], fade)
        for i in range(num_speakers)
    ])

    # Swapped conditioning keeps `reconstruct_all`: it needs a genuine forward pass with no rolled embeddings,
    # and no fault populating is built from it, so there is nothing to gain from caching its `g_out` the way `correct` does.
    outputs_swapped = (
        reconstruct_all(x, x_O, activity, solo, np.roll(embeddings, 1, axis=0), extractor, fade)
        if num_speakers > 1 else None
    )

    # ---- Q1b: the same contrast with a PERFECT extractor ---------------------
    # `tau_margin` scored J = +0.046 on the `correct`/`swapped` pair above, and
    # this file concluded "not a detector". But the margin is computed on `G`'s
    # OUTPUT, which sits near 2 dB and is mostly distortion -- and that
    # distortion contaminates cos(s_hat, e_i) and cos(s_hat, e_j) equally, so
    # the margin can collapse toward 0 whether the extraction was right or
    # wrong. That verdict is therefore one point on the EXTRACTOR axis, not a
    # property of the formula (the same error this project made about
    # refinement, corrected 2026-08-24).
    #
    # These two populations substitute the clean source for `G`'s output,
    # keeping the reconstruction otherwise identical (`x*w_Ei + s*w_Oi` is
    # exactly `reconstruct_all` with a perfect `G`). Read the sweep as:
    #
    #   separates  -> the FORMULA is sound and purely gated on `G`'s quality.
    #                 It recovers when the extractor does; no gate redesign.
    #   does not   -> the margin is broken independently of `G` and needs
    #                 replacing, not re-tuning.
    #
    # NOT DEPLOYABLE: it reads `scene.sources`. It is a bound, like the two
    # oracle refinement flags.
    #
    # Requires row i to BE source i, which holds under oracle regions and not
    # under a real diarizer's anonymous clusters. Skipped loudly rather than
    # silently when it does not hold -- a probe that quietly measures nothing is
    # the failure mode that cost this project a whole verification check.
    clean_correct = clean_swapped = None
    if list(speakers) == list(scene.speakers):
        sources = np.asarray(scene.sources, dtype=np.float64)
        rolled = np.roll(sources, 1, axis=0)  # matches np.roll(embeddings, 1)
        x_samples = np.asarray(x, dtype=np.float64)
        clean_correct = np.zeros_like(sources)
        clean_swapped = np.zeros_like(sources)
        for i in range(num_speakers):
            w_Ei, w_Oi = crossfade_windows(solo[i], activity[i], fade=fade)
            clean_correct[i] = x_samples * w_Ei + sources[i] * w_Oi
            clean_swapped[i] = x_samples * w_Ei + rolled[i] * w_Oi
    else:
        print(f"[tune_gate] scene {scene.name!r}: clean-margin arm SKIPPED -- "
              f"rows are diarizer clusters {list(speakers)[:3]}..., not "
              f"scene speakers. Run this probe with oracle regions.")

    rows: list[dict] = []
    for i, spk in enumerate(speakers):
        others = [embeddings[j] for j in range(num_speakers) if j != i]
        expected_active = activity[i].astype(bool) & overlap.astype(bool)

        def _row(population: str, estimate: np.ndarray, variance: np.ndarray) -> dict:
            diagnostics = gate_diagnostics(
                estimate, scene.sample_rate, embeddings[i], others, encoder,
                variance, expected_active, 
                # Must match `_fault_row`. Without this the healthy populations
                # carry whole-track flatness (~0.74) while every fault carries
                # the energy-gated one (~0.4), so the artifact sweep silently
                # compares two different quantities and reads the offset as a
                # fault effect -- in the direction opposite to the real one.
                artifact_min_energy_db=artifact_min_energy_db,
            )
            return {
                "scene": scene.name, "speaker": spk, "m": num_speakers,
                "population": population,
                "mean_variance": diagnostics.mean_variance,
                "margin": diagnostics.margin,
                "vad_coverage": diagnostics.vad_coverage,
                "artifact_score": diagnostics.artifact_score,
            }

        def _fault_row(population: str, estimate: np.ndarray) -> dict:
            """VAD + artifact only; margin and V_i recorded as nan.

            Deliberate: a fault population needs no margin, and computing one
            would add ~18 TitaNet calls per speaker (~8,100 on a 150-scene dev
            split) to answer a question Q1 does not ask. The nans are honest --
            `apply_thresholds` is NaN-safe and `_values` filters them -- but the
            `.md` says so explicitly rather than leaving it to be inferred.
            """
            return {
                "scene": scene.name, "speaker": spk, "m": num_speakers,
                "population": population,
                "mean_variance": float("nan"),
                "margin": float("nan"),
                "vad_coverage": vad_coverage(estimate, expected_active, scene.sample_rate),
                "artifact_score": spectral_flatness(estimate, min_energy_db=artifact_min_energy_db),
            }


        healthy = _row(HONEST, outputs_correct[i], variances[i])
        rows.append(healthy)
        rows.append({**healthy, "population": CORRECT})

        if contaminated_variances[i] is not None:
            rows.append(_row(CONTAMINATED, outputs_correct[i], contaminated_variances[i]))
        if outputs_swapped is not None:
            rows.append(_row(SWAPPED, outputs_swapped[i], variances[i]))
        if clean_correct is not None and num_speakers > 1:
            rows.append(_row(CLEAN_CORRECT, clean_correct[i], variances[i]))
            rows.append(_row(CLEAN_SWAPPED, clean_swapped[i], variances[i]))

        # ---- manufactured fault populations (NOT DEPLOYABLE) ----------------
        # Corrupt what `G` produced, then stitch through the SAME windows. The
        # solo half stays a clean copy of the mixture, so what is being measured
        # is "G failed", not "the pipeline broke".
        w_Ei_i, w_Oi_i = crossfade_windows(solo[i], activity[i], fade=fade)
        region = w_Oi_i > 0
        bases = [("fault_g_", g_out[i])]
        if clean_correct is not None and num_speakers > 1:
            bases.append(("fault_clean_", np.asarray(scene.sources, dtype=np.float64)[i]))
        for prefix, base in bases:
            for name, corrupt in ALL_FAULTS:
                corrupted = corrupt(base, region, _fault_rng(scene.name, i))
                rows.append(_fault_row(prefix + name, _stitch(x_samples_, corrupted, solo[i], activity[i], fade)))

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

def _graded_detection_sweep(
        rows: list[dict],
        field: str,
        grid: list[float],
        *,
        healthy: str,
        faulty: list[tuple[str, str]],
        reject_below: bool,
        arm: str,
        budget: float
    ) -> list[str]:

    """Detection table for a GRADED family of manufactured faults.

    One column per severity, all sharing one false-rejection column computed on ``healthy``.
    This is the table that actually places a threshold: a single (detection, false rejection) pair says only whether a fault is caught, while the graded row
    says HOW BAD a fault has to be before it is caught, and whether the threshold is too aggressive on healthy data. 

    **The suggestion criterion is NOT Youden's J, and that is deliberate.** J weights a missed detection and a false rejection equally, which is incoherent for a graded family.
    Its answer depends on how many severities happen to be in table, so addinga milder fixture would move the recommendation with no new evidence.
    Instead: the tightest threshold whose false rejection stays within ``budget``. Both are monotone in the threshold, so that is simply the extreme admissible value -- stated as a rule rather than discovered by search,
    because a search over a monotone function invites reading noise as an optimum. J is still printed per severity so this remains comparable with the V_i and margin sweeps above.
    """
    healthy_values = _values(rows, healthy, field)
    direction = "<" if reject_below else ">"
    labels = [label for label, _ in faulty]
    lines = [
        "", f"### `{field}` -- graded faults, {arm}", "",
        f"(n={len(healthy_values)} healthy `{healthy}`; a value {direction} the threshold is "
        f"rejected. Columns are detection per severity; false rej. is shared. "
        f"Suggestion = tightest threshold with false rej. <= {100 * budget:.0f}%.)", "",
        "| threshold | false rej. | " + " | ".join(f"`{label}`" for label in labels) + " |",
        "|---" * (2 + len(labels)) + "|",
    ]
    populations = {label: _values(rows, population, field) for label, population in faulty}
    empty = [label for label, values in populations.items() if values.size == 0]
    if healthy_values.size == 0 or len(empty) == len(labels):
        lines.append(f"| -- | (no rows: healthy n={len(healthy_values)}, empty faults {empty}) |"
                     + " -- |" * len(labels))
        return lines

    def _rate(values: np.ndarray, threshold: float) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.mean(values < threshold)) if reject_below else float(np.mean(values > threshold))

    suggested = None
    for threshold in grid:
        false_rej = _rate(healthy_values, threshold)
        detections = [_rate(populations[label], threshold) for label in labels]
        if false_rej <= budget:
            # Monotone in `threshold`, so the last admissible value IS the
            # tightest one; no search, no tie-breaking, nothing to read noise into.
            if suggested is None or (threshold > suggested[0]) == reject_below:
                suggested = (threshold, false_rej, detections)
        lines.append(
            f"| {threshold:g} | {100 * false_rej:.1f}% | "
            + " | ".join("--" if np.isnan(c) else f"{100 * c:.1f}%" for c in detections) + " |"
        )

    lines += ["", "Youden's J per severity, at each candidate:", "",
              "| threshold | " + " | ".join(f"`{label}`" for label in labels) + " |",
              "|---" * (1 + len(labels)) + "|"]
    for threshold in grid:
        false_rejection = _rate(healthy_values, threshold)
        lines.append(f"| {threshold:g} | " + " | ".join(
            "--" if np.isnan(_rate(populations[label], threshold))
            else f"{_rate(populations[label], threshold) - false_rejection:+.3f}"
            for label in labels) + " |")

    if empty:
        lines += ["", f"_(no rows for {empty} -- this arm did not run for those faults.)_"]

    if suggested is None:
        lines += ["", f"**NO ADMISSIBLE THRESHOLD.** Every candidate in the grid rejects more than "
                      f"{100 * budget:.0f}% of healthy `{healthy}`. Either the grid does not reach far "
                      "enough into the safe tail, or this check cannot be applied to this arm without "
                      "cutting into normal operation. Do NOT copy a value out of this table."]
        return lines

    threshold, false_rejection, detections = suggested
    endpoint = threshold in (grid[0], grid[-1])
    detected = [f"`{label}` {100 * c:.0f}%" for label, c in zip(labels, detections) if not np.isnan(c)]
    lines += ["", f"**suggested `{field}`: {threshold:g}** -- {100 * false_rejection:.1f}% false "
                  f"rejection, catching " + ", ".join(detected) + "."]
    if endpoint:
        # The V_i mistake, made impossible to repeat quietly: its threshold sat
        # 500x outside the diagnostic's entire usable range for four sessions,
        # and every observation of "it never fires" was correct while every
        # inference drawn from it was wrong.
        lines += ["", "**GRID DID NOT BRACKET.** The suggestion is the first or last value in the "
                      "grid, so the real optimum may lie outside it entirely. Widen "
                      f"`{field}_grid` and re-run before copying this into a config."]
    return lines

def _direction_report(
    rows: list[dict],
    field: str,
    *,
    healthy: str,
    faulty: list[tuple[str, str]],
    arm: str
) -> list[str]:
    """Which WAY each fault moves the diagnostic, and whether they agree.
    
    The check that makes artifact half worth running. ``max_artifact_score`` rejects values ABOVE it, which presumes every artifact raises flatness.
    Additive noise does. Musical noise is sparse and tonal and may LOWER it. If the family disagrees in sign, a single one-sided threshold cannot catch both and the honest finding is tht the check needs to be replaced.
    """
    healthy_values = _values(rows, healthy, field)
    lines = ["", f"### `{field}` -- direction of each fault, {arm}", "",
             "| fault | median | shift vs healthy |", "|---|---|---|"]
    if healthy_values.size == 0:
        return lines + ["| -- | (healthy population empty) | -- |"]

    baseline = float(np.median(healthy_values))
    lines.append(f"| _healthy_ `{healthy}` | {baseline:.5f} | -- |")
    shifts = {}
    for label, population in faulty:
        values = _values(rows, population, field)
        if values.size == 0:
            lines.append(f"| `{label}` | (no rows) | -- |")
            continue
        shift = float(np.median(values) - baseline)
        shifts[label] = shift
        flag = "  **INERT**" if abs(shift) < 1e-3 else ""
        lines.append(f"| `{label}` | {np.median(values):.5f} | {shift:+.5f}{flag} |")
    signs = {np.sign(s) for s in shifts.values() if abs(s) >= 1e-3}
    if len(signs) > 1:
        lines += ["", "**FAULTS DISAGREE IN SIGN.** Some corruptions raise this diagnostic and "
                      "others lower it, so no single one-sided threshold catches both families. "
                      "This check needs REPLACING rather than tuning -- do not pick a value from "
                      "the table above."]

    inert = [label for label, shift in shifts.items() if abs(shift) < 1e-3]
    if inert:
        lines += ["", f"**INERT FIXTURES: {inert}.** These corruptions did not move the median at "
                      "all. Before concluding anything about the check, confirm the fixture is "
                      "doing its job -- a fixture that changes nothing produces a sweep that looks "
                      "like a well-behaved negative result."]
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

def _stitch(x_samples: np.ndarray, overlap_signal: np.ndarray, solo_i, activity_i, fade: int):
    """``x*w_Ei + overlap*w_Oi`` -- reconstruct_speaker's body, with the extractor
    call lifted out so one ``G`` forward pass can feed every fault population.

    Guarded by ``test_gate_faults.py::test_stitch_matches_reconstruct_all``: this
    MUST reproduce ``reconstruct_all(...)[i]`` exactly, or the healthy `correct`
    population silently stops being the thing every prior run measured.
    """
    w_Ei, w_Oi = crossfade_windows(solo_i, activity_i, fade=fade)
    return np.asarray(x_samples, dtype=np.float64) * w_Ei + np.asarray(overlap_signal, dtype=np.float64) * w_Oi


def _fault_rng(scene_name: str, speaker_index: int) -> np.random.Generator:
    """Deterministic per (scene, speaker), so a re-run reproduces the CSV.

    ``zlib.crc32`` rather than ``hash()``: Python's string hash is salted per
    process, so a seed derived from it would make this script non-reproducible
    while looking perfectly deterministic in any single session.
    """
    return np.random.default_rng(zlib.crc32(f"{scene_name}:{speaker_index}".encode()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2/experiments/phase2_gate_tune_dev.yaml")
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
    # Honored here too so one `enroll:` block means the same thing in every
    # script; see dagger.enroll.topk.select_topk_solo_clips.
    enroll_budget_raw = enroll_cfg.get("budget_ms")
    enroll_budget_ms = None if enroll_budget_raw is None else float(enroll_budget_raw)

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = extractor_cfg.pop("checkpoint", None)
    gate_cfg = cfg.get("gate", {})
    artifact_energy_db = read_artifact_min_energy_db(gate_cfg)
    # The pairing guard. `max_artifact_score` tuned WITH energy-gating is a
    # threshold on a different quantity than one tuned without it, so this
    # script refuses to mint a number unless the config has stated which. An
    # explicit `artifact_min_energy_db: null` satisfies it -- the requirement is
    # a decision on the record, not a particular value.
    if "artifact_min_energy_db" not in gate_cfg:
        raise SystemExit(
            "gate.artifact_min_energy_db is not set in this config.\n"
            "  Set it to -40.0 to score only frames with speech in them (recommended: the\n"
            "  whole-track mean is dominated by each speaker's duty cycle -- clean audio and\n"
            "  G's ~2 dB output score 0.7376 vs 0.7422, a 0.005 gap), or to `null` to keep\n"
            "  the pre-2026-08-26 behaviour deliberately. Either way it must be explicit,\n"
            "  because the max_artifact_score this run recommends is only valid alongside it."
        )
    print(f"artifact flatness: min_energy_db={artifact_energy_db!r}")

    tune_cfg = cfg.get("tune", {})

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(checkpoint_path=checkpoint_path, device=device, **extractor_cfg)

    # Absent -> oracle regions, i.e. every pre-existing config is unchanged.
    # See measure_scene's docstring for why a real diarizer is what makes the
    # `V_i` sweep mean anything.
    diarizer_cfg = dict(cfg.get("diarizer", {}))
    diarizer = None
    if diarizer_cfg:
        from dagger.diarize.pyannote_diarizer import DEFAULT_MODEL, PyannoteDiarizer

        diarizer = PyannoteDiarizer(
            model=str(diarizer_cfg.get("model", DEFAULT_MODEL)),
            device=device,
            num_speakers=diarizer_cfg.get("num_speakers"),
        )
        print(f"diarizer: {diarizer_cfg.get('model', DEFAULT_MODEL)} (honest population is REAL)")
    else:
        print("diarizer: oracle -- V_i is structurally 0, so its sweep is not meaningful")

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
                    scene, fade, enroll_k, min_clip_ms, enroll_budget_ms, encoder, extractor,
                    diarizer=diarizer, artifact_min_energy_db=artifact_energy_db,
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

    max_false_rejection = float(tune_cfg.get("max_false_rejection", 0.05))

    counts = {p: sum(1 for r in rows if r["population"] == p) for p in
              (HONEST, CONTAMINATED, CORRECT, SWAPPED, CLEAN_CORRECT, CLEAN_SWAPPED)}
    counts.update({prefix + name: sum(1 for r in rows if r["population"] == prefix + name)
                   for prefix, _ in FAULT_ARMS for name, _ in ALL_FAULTS})
    # A guard that verifies zero rows is not a passing guard (CLAUDE.md §7):
    # Test B once skipped all 288 rows and printed PASS while the property it
    # checked was violated in 271 of them. The `G` arm must always be populated;
    # the clean arm legitimately is not (m=1, or diarizer clusters).
    missing = [prefix + name for name, _ in ALL_FAULTS
               for prefix, _ in (("fault_g_", None),) if counts.get(prefix + name, 0) == 0]
    if missing:
        raise SystemExit(f"fault populations produced NO rows: {missing}. The fixtures did not run; "
                         "every sweep below them would be vacuous.")
    
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
    # Q1b: the same margin sweep with a PERFECT extractor. Rendered only when
    # the clean arm actually ran -- an empty table would read as "the margin
    # found nothing" rather than "this probe did not execute".
    n_clean = sum(1 for r in rows if r["population"] == CLEAN_CORRECT)
    lines += ["", "## Q1b -- is the margin broken, or just starved by `G`?", ""]
    if n_clean == 0:
        lines += [
            "_(clean-margin arm did not run: rows are diarizer clusters, not scene",
            "speakers. Re-run this probe with oracle regions -- the question is about",
            "the FORMULA, and a real diarizer only adds a confound.)_", "",
        ]
    else:
        lines += [
            f"Same contrast as `tau_margin` above (n={n_clean} per population), but the",
            "clean source is substituted for `G`'s output -- i.e. what the margin would",
            "score if the extractor were perfect. **NOT DEPLOYABLE**; it is a bound.", "",
            "* **Separates** -> the formula is sound and purely gated on `G`'s quality.",
            "  It recovers when the extractor does, and no gate redesign is warranted;",
            "  `tau_margin`'s J = +0.046 was a statement about this checkpoint, not",
            "  about `M_i`.",
            "* **Does not separate** -> the margin is broken independently of `G` and",
            "  needs REPLACING rather than re-tuning.", "",
        ]
        lines += _detection_sweep(
            rows, "margin", grids["tau_margin"],
            healthy=CLEAN_CORRECT, faulty=CLEAN_SWAPPED, reject_below=True,
        )
    lines += ["", "## Rate sweeps (no fault population)"]
    lines += _rate_sweep(rows, "vad_coverage", grids["min_vad_coverage"], reject_below=True)
    lines += _rate_sweep(rows, "artifact_score", grids["max_artifact_score"], reject_below=False)
    lines += ["", "## Graded fault sweeps (manufactured populations -- NOT DEPLOYABLE)", "",
            "Faults are injected into `G`'s output *before* stitching, so the solo half stays "
            "a clean copy of the mixture and what is measured is \"`G` failed\", not \"the "
            "pipeline broke\". Margin and `V_i` are `nan` on these rows by design -- see "
            "`_fault_row`.", ""]
    for prefix, healthy_pop in FAULT_ARMS:
        arm = "on `G`'s output" if prefix == "fault_g_" else "on the CLEAN source (a bound)"
        vad_faulty = [(name, prefix + name) for name, _ in VAD_FAULTS]
        art_faulty = [(name, prefix + name) for name, _ in ARTIFACT_FAULTS]
        lines += _graded_detection_sweep(
            rows, "vad_coverage", grids["min_vad_coverage"], healthy=healthy_pop,
            faulty=vad_faulty, reject_below=True, arm=arm, budget=max_false_rejection)
        lines += _direction_report(rows, "vad_coverage", healthy=healthy_pop,
                                   faulty=vad_faulty, arm=arm)
        lines += _graded_detection_sweep(
            rows, "artifact_score", grids["max_artifact_score"], healthy=healthy_pop,
            faulty=art_faulty, reject_below=False, arm=arm, budget=max_false_rejection)
        lines += _direction_report(rows, "artifact_score", healthy=healthy_pop,
                                   faulty=art_faulty, arm=arm)

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
