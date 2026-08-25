#!/usr/bin/env python3
"""Phase 3 entrypoint: what does real diarization actually cost?

Guardrail §6.2 is the whole design: *never report a real-diarization number
without the oracle number beside it*, because otherwise a quality drop cannot be
attributed to the diarizer vs. ``phi`` vs. ``G``. So every arm below scores the
**same scenes in the same pass**, through the same
:func:`dagger.eval.systems.score_scene` that Phase 2 uses — rows then pair
exactly on ``(scene, speaker, depth, system)``, and CLAUDE.md §7's "prefer paired
differences over subtracting two means" applies without hoping scene difficulty
averages out.

Four arms, each varying exactly one thing from its neighbour:

* ``oracle``           -- ground-truth regions. The mandatory companion number.
* ``real``             -- pyannote, estimating the speaker count itself. The
                          headline: real conditions, counting errors included.
* ``real_forced_m``    -- pyannote told how many speakers there are. Removes
                          miscounting, leaving segmentation error alone.
* ``real_index_order`` -- ``real``, but deflation processes speakers in index
                          order instead of by ascending ``V_i``.

That last arm exists because of a bias, not a tidy-up. Deflation order is
``ascending V_i``, and under oracle diarization ``V_i`` is identically zero, so
the sort has always been a no-op. Real diarization makes it a real permutation,
and position in that order IS ``n_accepted_before`` -- the dominant driver of
SI-SDR for the deflation systems. Worse, it is biased: whichever speaker's solo
region best survives diarization earns the lowest ``V_i`` and is promoted to the
least-damaged slot, so the real arm is flattered and the gap is understated. See
:func:`dagger.eval.systems.deflation_order`. Read the decomposition as:

* ``real_index_order - oracle``  -- diarization error alone
* ``real - real_index_order``    -- the reordering ``V_i`` induces
* ``real_forced_m - real``       -- how much of the cost was speaker *counting*

Three files are written to ``<eval.results_dir>/phase3_<dataset>_<n>spk[_<tag>]``:

* ``.csv``       -- score rows, Phase 2's ``SCORE_FIELDS`` plus a ``diarization``
                    column naming the arm.
* ``_gate.csv``  -- gate decisions plus ``diarization``. This is where ``V_i``
                    finally shows a nonzero ``mean_variance``: it is structurally
                    0 under oracle diarization (one solo run -> one clip ->
                    variance over one sample), and only a real diarizer's
                    fragmented solo regions bring it to life.
* ``_diar.csv``  -- per (scene, arm) diarization quality: DER and its components,
                    plus ``overlap_recall`` and the missed/spurious speaker counts.

Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/run_phase3.py \\
        --config configs/phase3/dod/phase3_librimix_3spk_oracle_vs_real.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.data import build_dataset
from dagger.data.paths import load_env
from dagger.diarize.base import DiarizationFailedError
from dagger.diarize.mapping import map_clusters_to_speakers
from dagger.diarize.oracle import OracleDiarizer
from dagger.diarize.regions import dilate_overlap, scene_regions
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError
from dagger.eval.systems import (
    GATE_FIELDS, OVERALL_FIELDS, SCORE_FIELDS, SYSTEMS, score_scene,
)
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.metrics.der import diarization_error_rate
from dagger.metrics.phase2_scores import SI_SDR_CAP_DB, clip_score, mean_sem

#: Arm name -> (uses the real diarizer?, forced speaker count?, deflation order).
#: `None` for num_speakers means "let the diarizer estimate".
ARMS: dict[str, tuple[bool, bool, str]] = {
    "oracle": (False, False, "variance"),
    "real": (True, False, "variance"),
    "real_forced_m": (True, True, "variance"),
    "real_index_order": (True, False, "index"),
}

DEFAULT_ARMS = ("oracle", "real", "real_forced_m", "real_index_order")

# `cluster` is Phase 3-only provenance: which predicted cluster a row came from.
# `speaker` is the GROUND-TRUTH speaker it was attributed to, which is what the
# paired comparisons key on -- keying on the cluster id would make every
# oracle-vs-real pair miss (oracle rows are `s1..`, real rows `SPEAKER_00..`)
# and silently produce an empty gap table.
PHASE3_SCORE_FIELDS = (
    ["diarization", "dilate_ms"] + SCORE_FIELDS + ["cluster", "n_clusters"]
)
PHASE3_GATE_FIELDS = ["diarization", "dilate_ms"] + GATE_FIELDS + ["cluster"]
PHASE3_OVERALL_FIELDS = (
    ["diarization", "dilate_ms"] + OVERALL_FIELDS + ["cluster", "n_clusters"]
)

DIAR_FIELDS = [
    "scene", "diarization", "dilate_ms", "m", "der", "miss_rate", "false_alarm_rate",
    "confusion_rate", "overlap_recall", "n_ref", "n_pred",
    "n_missed_speakers", "n_spurious_clusters", "total_speech",
    "mask_overlap_recall", "mask_overlap_false_alarm",
]


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_diarizer(diarizer_cfg: dict, num_speakers: int | None, device: str):
    """Construct the configured real diarizer.

    Imported here rather than at module scope so the ``[diarize]`` extra is only
    needed by runs that actually use a real arm -- an oracle-only run (and every
    offline test) stays importable without pyannote installed.
    """
    name = str(diarizer_cfg.get("name", "pyannote")).lower()
    if name != "pyannote":
        raise ValueError(
            f"Unknown diarizer {name!r}. Only 'pyannote' is wired up; NeMo "
            "Sortformer is a drop-in behind dagger.diarize.base.Diarizer."
        )
    from dagger.diarize.pyannote_diarizer import DEFAULT_MODEL, PyannoteDiarizer

    return PyannoteDiarizer(
        model=str(diarizer_cfg.get("model", DEFAULT_MODEL)),
        device=device,
        num_speakers=num_speakers,
    )


def _diarization_row(scene, regions, arm: str, dilate_ms: float = 0.0) -> dict:
    """DER and components for one arm on one scene, against oracle activity.

    DER itself is computed from ``regions.activity``, which ``dilate_overlap``
    leaves untouched -- so every DER column is identical across dilation values
    by construction. That is correct: dilation is an audio-path decision, not a
    better activity estimate, and letting it move DER would let the knob grade
    its own homework.

    The two ``mask_overlap_*`` columns are what the sweep is actually read on.
    They score the DERIVED overlap mask (the thing dilation changes, and the
    thing that decides copy-vs-extract) against the true overlap region:
    recall is the share of genuinely-overlapped frames the pipeline will extract
    on, false alarm the share of non-overlapped frames it will needlessly run
    ``G`` over. Stage A measured recall 0.758, and the whole asymmetry argument
    is that buying recall with false alarm is a good trade.
    """
    reference_regions = scene_regions(scene, OracleDiarizer())
    reference = reference_regions.activity
    result = diarization_error_rate(reference, regions.activity)

    true_overlap = reference_regions.overlap > 0
    predicted_overlap = regions.overlap > 0
    n_true = int(true_overlap.sum())
    n_clean = int((~true_overlap).sum())
    return {
        "scene": scene.name,
        "diarization": arm,
        "dilate_ms": dilate_ms,
        "mask_overlap_recall": (
            float((predicted_overlap & true_overlap).sum() / n_true)
            if n_true else float("nan")
        ),
        "mask_overlap_false_alarm": (
            float((predicted_overlap & ~true_overlap).sum() / n_clean)
            if n_clean else float("nan")
        ),
        "m": int(reference.shape[0]),
        "der": result.der,
        "miss_rate": result.miss_rate,
        "false_alarm_rate": result.false_alarm_rate,
        "confusion_rate": result.confusion_rate,
        "overlap_recall": result.overlap_recall,
        "n_ref": result.n_ref,
        "n_pred": result.n_pred,
        "n_missed_speakers": result.n_missed_speakers,
        "n_spurious_clusters": result.n_spurious_clusters,
        "total_speech": result.total_speech,
    }


def score_scene_all_arms(
    scene,
    arms: list[str],
    diarizer_cfg: dict,
    device: str,
    *,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    enroll_budget_ms: float | None,
    encoder,
    extractor,
    gate_cfg: dict,
    refine_rounds: int,
    diarizer_cache: dict,
    arm_failures: dict | None = None,
    dilate_ms_values: list[float] = (0.0,),
    refine_oracle_ceiling: bool = False,
    refine_oracle_audio: bool = False,
    rescale_to_mixture: bool = False,
    dilation_failures: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run every arm over one scene. Returns ``(score, gate, diarization)`` rows.

    The real diarizer runs **once per (uses-real, forced-m) combination**, not
    once per arm: ``real`` and ``real_index_order`` differ only in deflation
    order, which is a reconstruction-time choice. Re-running the diarizer for
    each would waste the expensive step and, since pyannote's clustering is not
    guaranteed bit-reproducible, could make the two arms differ in their regions
    as well as their order -- destroying the one-variable-at-a-time property the
    whole decomposition rests on.

    ``dilate_ms_values`` sweeps :func:`~dagger.diarize.regions.dilate_overlap`
    **inside** this per-scene function, for exactly the same two reasons. It is
    pure post-processing on the cached ``Regions``, so pyannote still runs once
    per scene no matter how many values are swept -- and running it per value
    would let the regions themselves drift between the points being compared.
    ``[0.0]`` (the default) is the undilated baseline and adds no work.
    """
    score_rows: list[dict] = []
    gate_rows: list[dict] = []
    diar_rows: list[dict] = []
    overall_rows: list[dict] = []

    regions_cache: dict[tuple[bool, bool], object] = {}

    for arm in arms:
        uses_real, forced_m, order_policy = ARMS[arm]
        key = (uses_real, forced_m)
        if key not in regions_cache:
            if uses_real:
                num_speakers = len(scene.speakers) if forced_m else None
                cache_key = ("pyannote", num_speakers)
                if cache_key not in diarizer_cache:
                    diarizer_cache[cache_key] = _build_diarizer(
                        diarizer_cfg, num_speakers, device
                    )
                diarizer = diarizer_cache[cache_key]
            else:
                diarizer = OracleDiarizer()
            try:
                regions_cache[key] = scene_regions(scene, diarizer)
            except DiarizationFailedError as exc:
                # Skip THIS ARM on THIS SCENE, not the whole scene. A forced
                # speaker count is unsatisfiable on a scene whose segmentation
                # yields fewer embedding windows than m -- and `real_forced_m`
                # is a diagnostic arm, so letting it discard the scene would
                # cost the headline `real - oracle` comparison a row for a
                # reason that has nothing to do with either of them.
                #
                # Paired comparisons intersect on keys, so they simply use the
                # rows that exist; `n` is reported per row in the gap table, so
                # a thinner forced-m comparison is visible rather than silent.
                if arm_failures is not None:
                    arm_failures[arm] = arm_failures.get(arm, 0) + 1
                print(f"[diarize] scene {scene.name!r}: skipping arm {arm!r} -- {exc}")
                continue
        base_regions = regions_cache[key]

        for dilate_ms in dilate_ms_values:
            samples = int(round(dilate_ms / 1000.0 * scene.sample_rate))
            regions = dilate_overlap(base_regions, samples)

            try:
                rows, gates, overalls = score_scene(
                    scene, fade, enroll_k, min_clip_ms, enroll_budget_ms,
                    encoder, extractor, gate_cfg, refine_rounds,
                    regions=regions, order_policy=order_policy,
                    # A real diarizer routinely invents a cluster lying entirely
                    # inside another speaker's speech, which can never be
                    # enrolled. Failing the whole scene on that would discard
                    # scenes in proportion to how badly the diarizer did -- the
                    # selection bias that silently shrank Phase 1's dataset.
                    # Applied to EVERY arm so the arms stay comparable; it is a
                    # no-op for oracle, where the scheduler guarantees solo audio.
                    on_unenrollable="drop",
                    refine_oracle_ceiling=refine_oracle_ceiling,
                    refine_oracle_audio=refine_oracle_audio,
                    rescale_to_mixture=rescale_to_mixture,
                )
            except NoSoloRegionError as exc:
                # Dilation shrinks solo regions, so a large enough value can
                # leave a scene with no enrollable speaker at all. Caught HERE,
                # per (arm, dilation), rather than letting it propagate: at the
                # scene level it would drop the scene from EVERY sweep point
                # including the 0 ms baseline, which is the one point that must
                # stay comparable to the committed Stage A numbers. Counted so
                # the knob's cost is reported instead of inferred from a
                # shrinking `n`.
                if dilation_failures is not None:
                    dilation_failures[(arm, dilate_ms)] = (
                        dilation_failures.get((arm, dilate_ms), 0) + 1
                    )
                print(
                    f"[enroll] scene {scene.name!r} arm {arm!r} "
                    f"dilate={dilate_ms:g} ms: {exc}"
                )
                continue

            for row in rows:
                score_rows.append({"diarization": arm, "dilate_ms": dilate_ms, **row})
            for row in gates:
                gate_rows.append({"diarization": arm, "dilate_ms": dilate_ms, **row})
            for row in overalls:
                overall_rows.append({"diarization": arm, "dilate_ms": dilate_ms, **row})
            diar_rows.append(_diarization_row(scene, regions, arm, dilate_ms))

    return score_rows, gate_rows, diar_rows, overall_rows


def _clipped(rows: list[dict]) -> list[float]:
    return [c for c in (clip_score(r["si_sdr"]) for r in rows) if c is not None]


def _gap_section(rows: list[dict], arms: list[str], depths: list[int]) -> list[str]:
    """Paired arm-vs-arm differences -- the DoD table.

    Paired on ``(scene, speaker, depth, system)`` so scene difficulty cancels
    *exactly* rather than being averaged over, and reported with ``n``, SEM, win
    rate and ``|t|`` together: CLAUDE.md §7 requires all four, because a positive
    mean at a ~50% win rate is a few large wins rather than broad superiority
    (which was literally Phase 1's result).
    """
    comparisons = [
        ("real", "oracle", "total cost of real diarization"),
        ("real_index_order", "oracle", "diarization error alone"),
        ("real", "real_index_order", "the reordering V_i induces"),
        ("real_forced_m", "real", "how much of the cost was speaker counting"),
    ]
    lines = ["", "## Oracle-vs-real gap (paired, per system x depth)", ""]
    any_reported = False
    for left, right, blurb in comparisons:
        if left not in arms or right not in arms:
            continue
        any_reported = True
        lines += [
            f"### `{left}` - `{right}` -- {blurb}", "",
            "| system | depth | n | mean (dB) | SEM | win rate | \\|t\\| |",
            "|---|---|---|---|---|---|---|",
        ]
        for system_name in SYSTEMS:
            for depth in depths:
                def keyed(arm: str) -> dict:
                    return {
                        (r["scene"], r["speaker"], r["depth"]): c
                        for r in rows
                        if r["diarization"] == arm
                        and r["system"] == system_name
                        and r["depth"] == depth
                        for c in [clip_score(r["si_sdr"])]
                        if c is not None
                    }

                a, b = keyed(left), keyed(right)
                diffs = [a[k] - b[k] for k in a if k in b]
                if not diffs:
                    continue
                mean, sem, n = mean_sem(diffs)
                wins = sum(1 for d in diffs if d > 0) / n
                t = abs(mean / sem) if sem and not np.isnan(sem) and sem > 0 else float("nan")
                lines.append(
                    f"| {system_name} | {depth} | {n} | {mean:+.2f} | {sem:.2f} | "
                    f"{wins:.0%} | {t:.1f} |"
                )
        lines.append("")
    if not any_reported:
        lines += ["(no arm pair available in this run)", ""]
    return lines


def _diar_section(diar_rows: list[dict], arms: list[str]) -> list[str]:
    """DER decomposition per arm.

    Components are listed separately rather than only their sum: DER adds three
    unlike failures, and CLAUDE.md §6.4 forbids reporting an aggregate as
    evidence. ``overlap_recall`` is called out because overlap is the only region
    where ``G`` runs at all -- solo frames are copied straight through -- so it
    should track the SI-SDR gap far better than total DER does.
    """
    lines = [
        "", "## Diarization quality (per arm)", "",
        "| arm | n | DER | miss | false alarm | confusion | overlap recall | missed spk | spurious |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        subset = [r for r in diar_rows if r["diarization"] == arm]
        if not subset:
            continue

        def avg(field: str) -> float:
            values = [r[field] for r in subset if not np.isnan(r[field])]
            return float(np.mean(values)) if values else float("nan")

        lines.append(
            f"| {arm} | {len(subset)} | {avg('der'):.3f} | {avg('miss_rate'):.3f} | "
            f"{avg('false_alarm_rate'):.3f} | {avg('confusion_rate'):.3f} | "
            f"{avg('overlap_recall'):.3f} | "
            f"{sum(r['n_missed_speakers'] for r in subset)} | "
            f"{sum(r['n_spurious_clusters'] for r in subset)} |"
        )
    return lines


def _enrollment_section(rows: list[dict], arms: list[str]) -> list[str]:
    """How many clusters the diarizer found vs. how many survived enrollment.

    These are different numbers and the difference is load-bearing. A cluster
    with no *predicted-solo* region cannot be enrolled and is dropped, and a real
    diarizer produces such clusters routinely -- one nested entirely inside
    another speaker's speech, say. On 2026-08-16 that silently reduced a
    3-speaker scene to a ONE-speaker reconstruction in 149/150 scenes, which made
    all four systems identical (nothing to deflate, gate, or refine) and cost a
    whole run to diagnose from the raw CSV.

    `m == 1` here is the alarm: with one track there is no system comparison
    left, whatever the SI-SDR column says.
    """
    lines = [
        "", "## Cluster discovery vs. enrollment", "",
        "`clusters` is what the diarizer produced; `enrolled` (the `m` column) is",
        "how many survived enrollment. A cluster with no predicted-solo region",
        "cannot be enrolled and is dropped. If `enrolled` is 1, that arm is a",
        "single-speaker system and its system comparisons are vacuous.", "",
        "| arm | scenes | clusters (mean) | enrolled (mean) | dropped | scenes with enrolled=1 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        per_scene: dict[str, tuple[int, int]] = {}
        for r in rows:
            if r["diarization"] != arm:
                continue
            per_scene[r["scene"]] = (int(r["n_clusters"]), int(r["m"]))
        if not per_scene:
            continue
        clusters = [c for c, _ in per_scene.values()]
        enrolled = [m for _, m in per_scene.values()]
        dropped = sum(c - m for c, m in per_scene.values())
        degenerate = sum(1 for m in enrolled if m <= 1)
        flag = "  **<- degenerate**" if degenerate > len(enrolled) // 2 else ""
        lines.append(
            f"| {arm} | {len(per_scene)} | {np.mean(clusters):.2f} | "
            f"{np.mean(enrolled):.2f} | {dropped} | {degenerate}{flag} |"
        )
    return lines


def _gate_section(gate_rows: list[dict], arms: list[str]) -> list[str]:
    """Gate decisions per arm, broken out BY REASON.

    Two things this phase must be able to see, neither visible from an accept
    rate alone:

    * ``enrollment_variance`` firing at all. `V_i` has been exactly 0.0 in every
      run this project has ever done, so a nonzero count here is the first time
      the enrollment gate has ever done anything.
    * ``overlap_clip_too_short``. Under real diarization a predicted
      overlap-only run can be a few samples, which refinement must skip. Folding
      that into the rejection count would report boundary jitter as if the gate
      had judged the audio and found it wanting.
    """
    lines = [
        "", "## Gate decisions (per arm x system, by reason)", "",
        "| arm | system | decisions | accepted | " + " | ".join(
            ["margin", "variance (V_i)", "vad", "artifact", "too short", "no clip"]
        ) + " |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    reasons = ["margin", "enrollment_variance", "vad_coverage", "artifact_score",
               "overlap_clip_too_short", "no_overlap_clip"]
    for arm in arms:
        for system_name in ("gated_deflation", "coarse_to_fine"):
            subset = [r for r in gate_rows
                      if r["diarization"] == arm and r["system"] == system_name]
            if not subset:
                continue
            accepted = sum(1 for r in subset if r["accepted"] is True)
            counts = [sum(1 for r in subset if r["reason"] == reason) for reason in reasons]
            rate = f"{accepted} ({100.0 * accepted / len(subset):.0f}%)"
            lines.append(
                f"| {arm} | {system_name} | {len(subset)} | {rate} | "
                + " | ".join(str(c) for c in counts) + " |"
            )

    variance_fired = sum(1 for r in gate_rows if r["reason"] == "enrollment_variance")
    values = [float(r["mean_variance"]) for r in gate_rows
              if r["mean_variance"] not in (None, "", "None")
              and not np.isnan(float(r["mean_variance"]))]
    nonzero = sum(1 for v in values if v > 0)
    lines += [
        "", f"`V_i` rejections: **{variance_fired}**; `mean_variance` nonzero in "
        f"**{nonzero}/{len(values)}** decisions"
        + (f", max {max(values):.3g}" if values else ""),
        "",
        "`V_i` has been structurally 0 in every prior run (one solo run -> one clip",
        "-> variance over a single sample). A nonzero count above is the first time",
        "the enrollment-variance check has had anything to measure.",
    ]
    return lines


def _dilation_section(
    rows: list[dict],
    diar_rows: list[dict],
    depths: list[int],
    dilation_failures: dict | None,
    n_scenes: int,
) -> list[str]:
    """The overlap-dilation sweep: what the knob buys, and what it costs.

    Read the two halves TOGETHER. Dilation trades overlap-mask false alarms
    (cheap: ``G`` runs where a copy would have done) for overlap-mask recall
    (the expensive failure: an unseparated mixture emitted as a speaker's
    track). But it also shrinks solo regions, which is the same audio enrollment
    draws from -- so a value that improves SI-SDR while dropping speakers is not
    a win, it is a smaller, easier problem being scored.

    Rows are paired against the 0 ms baseline on matched
    ``(scene, speaker, depth, system)``, per CLAUDE.md §7 -- so scene difficulty
    cancels exactly rather than being averaged over.
    """
    values = sorted({r["dilate_ms"] for r in rows})
    if len(values) < 2:
        return []

    lines = [
        "", "## Overlap dilation sweep", "",
        "Paired against the 0 ms baseline on matched (scene, speaker, depth, system).",
        "`recall`/`false alarm` score the DERIVED overlap mask against the true",
        "overlap region -- the mask that decides copy-vs-extract, which is what",
        "dilation actually moves (DER is unchanged by construction: dilation does",
        "not touch `activity`).", "",
    ]

    for arm in sorted({r["diarization"] for r in rows}):
        lines += [
            f"### `{arm}`", "",
            "| dilate (ms) | recall | false alarm | scenes lost | "
            + " | ".join(f"d{k} vs 0ms" for k in depths) + " |",
            "|---" * (4 + len(depths)) + "|",
        ]
        for value in values:
            diar_subset = [
                r for r in diar_rows
                if r["diarization"] == arm and r["dilate_ms"] == value
            ]

            def _avg(field: str) -> float:
                vals = [r[field] for r in diar_subset if not np.isnan(r[field])]
                return float(np.mean(vals)) if vals else float("nan")

            lost = (dilation_failures or {}).get((arm, value), 0)
            cells = []
            for depth in depths:
                def keyed(at: float) -> dict:
                    return {
                        (r["scene"], r["speaker"], r["system"]): c
                        for r in rows
                        if r["diarization"] == arm and r["dilate_ms"] == at
                        and r["depth"] == depth
                        for c in [clip_score(r["si_sdr"])]
                        if c is not None
                    }

                here, base = keyed(value), keyed(0.0)
                diffs = [here[k] - base[k] for k in here if k in base]
                if not diffs:
                    cells.append("n/a")
                    continue
                mean, sem, n = mean_sem(diffs)
                cells.append(f"{mean:+.2f}±{sem:.2f} (n={n})")
            lines.append(
                f"| {value:g} | {_avg('mask_overlap_recall'):.3f} | "
                f"{_avg('mask_overlap_false_alarm'):.3f} | {lost}/{n_scenes} | "
                + " | ".join(cells) + " |"
            )
        lines.append("")
    return lines


def _overall_section(overall_rows: list[dict], arms: list[str]) -> list[str]:
    """The un-stratified number: what a listener actually gets.

    Reported ALONGSIDE the per-depth tables, never instead of them (CLAUDE.md
    §6.4). It exists because stratify-only leaves "is this configuration net
    better?" unanswerable -- a gain and its cost land in different rows with no
    exchange rate between them.

    Swept dilation values each get their own row here, and that is the whole
    point: this is the only table in the file where the sweep points can be
    compared against each other on one number.
    """
    if not overall_rows:
        return []

    dilations = sorted({r["dilate_ms"] for r in overall_rows})

    def value_of(row: dict, field: str) -> float | None:
        """``clip_score``'s ±50 dB cap exists to tame ``±inf`` SI-SDR. A level
        DISAGREEMENT is not an SI-SDR and has no such pathology, so capping it
        would quietly report a 60 dB level error as 50 -- a truncated
        diagnostic, which is the exact defect class this column was added to
        expose. Drop nan/inf there instead of clipping."""
        raw = row[field]
        if field == "level_error_db":
            v = float(raw)
            return None if not math.isfinite(v) else v
        return clip_score(raw)

    def table(field: str) -> list[str]:
        # Rows written before 2026-08-23 carry only `si_sdr`. Rendering an empty
        # table for a column that did not exist yet would read as "the metric
        # says nothing here" rather than "this run predates the metric".
        if not any(field in r for r in overall_rows):
            return [f"_(no `{field}` in this run -- it predates the column.)_"]
        header = "| arm | system |" + "".join(f" {d:g} ms |" for d in dilations)
        out = [header, "|---" * (2 + len(dilations)) + "|"]
        for arm in arms:
            for system_name in SYSTEMS:
                cells = []
                for d in dilations:
                    vals = [
                        value_of(r, field) for r in overall_rows
                        if r["diarization"] == arm and r["system"] == system_name
                        and r["dilate_ms"] == d and field in r
                    ]
                    vals = [v for v in vals if v is not None]
                    if not vals:
                        cells.append("n/a")
                        continue
                    mean, sem, _n = mean_sem(vals)
                    cells.append(f"{mean:.2f}±{sem:.2f}")
                if all(c == "n/a" for c in cells):
                    continue
                out.append(
                    f"| {arm} | {system_name} |" + "".join(f" {c} |" for c in cells)
                )
        return out

    lines = [
        "", "## Overall SI-SDR (un-stratified)", "",
        "One row per (scene, speaker, system); no depth stratification. Read",
        "these WITH the per-depth tables, never instead of them (§6.4), and",
        "never optimize against any of them -- optimizing a whole-output number",
        "is what voided Stage B's refinement ceiling.", "",
        "### Pooled across depths -- THE EXCHANGE RATE", "",
        "Scale fitted per depth (so a per-region level error is discounted",
        "exactly as the per-depth tables discount it), then error energies",
        "pooled weighted by each depth's true speech. Provably bounded by the",
        "best and worst depth, so it can weigh a depth-1 loss against a depth-2",
        "gain. **This is the number to compare sweep points on.**", "",
    ]
    lines += table("si_sdr_pooled")
    lines += [
        "", "### Whole output track, one global scale", "",
        "The literal `si_sdr(output, target)`. SCALE-ANCHORED by the bit-exact",
        "solo copy, so a pure LEVEL error in the overlap region is charged at",
        "full price while every per-depth row discounts it -- which is why this",
        "can land BELOW every depth it appears to summarise (it did so in 271 of",
        "288 rows on 2026-08-23). Kept because it is the only score here that",
        "can see a level error at all; not a summary of the tables above.", "",
    ]
    lines += table("si_sdr")
    lines += [
        "", "### Level disagreement across depths (dB)", "",
        "`20*log10(max alpha / min alpha)` over the per-depth fitted scales. 0",
        "means one consistent level explains the whole output. Large values are",
        "the mechanism behind any gap between the two tables above, and are",
        "invisible to every scale-invariant metric in this project.", "",
    ]
    lines += table("level_error_db")
    return lines + [""]


def _ordering_section(rows: list[dict], arms: list[str], depths: list[int]) -> list[str]:
    """Does the Phase 2 ordering survive real diarization?

    Nearly free here, and a genuinely useful robustness result: Phase 2
    established `coarse_to_fine > gated_deflation > ungated_deflation` 18/18
    under oracle regions, and whether that holds when the masks are wrong is a
    different question from how much absolute quality it costs.
    """
    lines = ["", "## Ordering check (deepest available depth, per arm)", ""]
    if not depths:
        return lines + ["(no rows)"]
    deepest = depths[-1]
    for arm in arms:
        means = {}
        for system_name in SYSTEMS:
            values = _clipped(
                [
                    r for r in rows
                    if r["diarization"] == arm
                    and r["system"] == system_name
                    and r["depth"] == deepest
                ]
            )
            means[system_name] = float(np.mean(values)) if values else float("nan")
        ok = means["coarse_to_fine"] >= means["gated_deflation"] > means["ungated_deflation"]
        lines.append(
            f"- `{arm}` depth {deepest}: coarse_to_fine={means['coarse_to_fine']:.2f} "
            f"gated_deflation={means['gated_deflation']:.2f} "
            f"ungated_deflation={means['ungated_deflation']:.2f} -- ordering holds: {ok}"
        )
    return lines


def _write_results(
    rows: list[dict],
    gate_rows: list[dict],
    diar_rows: list[dict],
    overall_rows: list[dict],
    out_dir: Path,
    stem: str,
    arms: list[str],
    arm_failures: dict | None = None,
    n_scenes: int | None = None,
    dilation_failures: dict | None = None,
    refine_oracle_ceiling: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    gate_csv_path = out_dir / f"{stem}_gate.csv"
    diar_csv_path = out_dir / f"{stem}_diar.csv"
    overall_csv_path = out_dir / f"{stem}_overall.csv"
    md_path = out_dir / f"{stem}.md"

    for path, fields, payload in (
        (csv_path, PHASE3_SCORE_FIELDS, rows),
        (gate_csv_path, PHASE3_GATE_FIELDS, gate_rows),
        (diar_csv_path, DIAR_FIELDS, diar_rows),
        # Its own file, at its own grain -- see OVERALL_FIELDS. Keeping it out of
        # `{stem}.csv` is what makes it structurally impossible for a
        # depth-stratified table to absorb it as an extra depth.
        (overall_csv_path, PHASE3_OVERALL_FIELDS, overall_rows),
    ):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(payload)

    depths = sorted({r["depth"] for r in rows})
    # Every table below the sweep section reports the UNDILATED baseline only.
    # Dilation is a per-row condition, not a per-run one, so leaving it mixed in
    # would average an absolute SI-SDR across several different pipelines and
    # call the result "depth 2" -- the same conflation-of-two-variables mistake
    # that cost Phase 2 five runs on the wrong axis.
    all_rows = rows
    dilate_values = sorted({r["dilate_ms"] for r in rows})
    rows = [r for r in rows if r["dilate_ms"] == 0.0]
    baseline_gate_rows = [r for r in gate_rows if r["dilate_ms"] == 0.0]
    baseline_diar_rows = [r for r in diar_rows if r["dilate_ms"] == 0.0]

    lines = [
        f"# Phase 3 results -- {stem}", "", f"rows scored: {len(all_rows)}",
        f"arms: {', '.join(arms)}", "",
        f"(means clip +-inf to +-{SI_SDR_CAP_DB:.0f} dB rather than dropping them)",
        "",
    ]
    if refine_oracle_ceiling:
        lines += [
            "> **NOT A DEPLOYABLE RESULT.** `refine.oracle_ceiling` was ON, so",
            "> `coarse_to_fine` accepted refinement candidates by comparing SI-SDR",
            "> against the CLEAN SOURCES. Its rows are an upper bound on what any",
            "> acceptance rule could achieve, not a system that could be shipped.",
            "> The other three systems never refine and are unaffected, so they",
            "> remain a valid control within this run.", "",
        ]
    if len(dilate_values) > 1:
        lines += [
            f"> Dilation sweep over {dilate_values} ms. **Every table below the",
            "> sweep section reports the 0 ms baseline only** -- mixing dilation",
            "> values into one mean would average across different pipelines.", "",
        ]
    lines += ["## Absolute SI-SDR (per arm x system x depth)", ""]
    header = "| arm | system | " + " | ".join(f"depth {k}" for k in depths) + " |"
    lines += [header, "|---" * (len(depths) + 2) + "|"]
    for arm in arms:
        for system_name in SYSTEMS:
            cells = []
            for depth in depths:
                values = _clipped(
                    [
                        r for r in rows
                        if r["diarization"] == arm
                        and r["system"] == system_name
                        and r["depth"] == depth
                    ]
                )
                cells.append(f"{float(np.mean(values)):.2f}" if values else "n/a")
            lines.append(f"| {arm} | {system_name} | " + " | ".join(cells) + " |")

    if arm_failures:
        # An arm that could not run on many scenes is a thinner comparison than
        # the others, and the reader must see that BEFORE the gap table rather
        # than infer it from a smaller `n`.
        lines += [
            "", "## Arm coverage", "",
            "Scenes where an arm could not be computed (arm skipped, scene kept for",
            "the others). A forced speaker count is unsatisfiable when the pipeline's",
            "segmentation yields fewer embedding windows than `m`.", "",
            "| arm | scenes failed | of | % |", "|---|---|---|---|",
        ]
        for arm in arms:
            count = arm_failures.get(arm, 0)
            total = n_scenes if n_scenes else 0
            pct = f"{100.0 * count / total:.0f}%" if total else "n/a"
            lines.append(f"| {arm} | {count} | {total} | {pct} |")

    lines += _overall_section(overall_rows, arms)
    lines += _dilation_section(
        all_rows, diar_rows, depths, dilation_failures, n_scenes or 0
    )
    lines += _diar_section(baseline_diar_rows, arms)
    lines += _enrollment_section(rows, arms)
    lines += _gate_section(baseline_gate_rows, arms)
    lines += _gap_section(rows, arms, depths)
    lines += _ordering_section(rows, arms, depths)

    lines += [
        "", "## Caveats", "",
        "- LibriMix scenes have hard segment boundaries, no reverb and no",
        "  conversational turn-taking, so a real diarizer's error profile here is",
        "  not its profile on AMI. Treat this gap as a LOWER BOUND on the",
        "  real-corpora gap Phase 4 will measure.",
        "- The deflation systems' `real` vs `oracle` difference mixes diarization",
        "  error with the reordering a now-nonzero `V_i` causes; the",
        "  `real_index_order` rows above separate the two.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}, {gate_csv_path}, {diar_csv_path}, "
          f"{overall_csv_path} and {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase3/dod/phase3_librimix_3spk_oracle_vs_real.yaml",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_rate = int(cfg["sample_rate"])
    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * sample_rate))

    # Validate the arm set BEFORE touching the environment or the corpus: a
    # config typo should not first require DAGGER_DATA_ROOT to be mounted and a
    # checkpoint to load before it reports itself.
    diarizer_cfg = dict(cfg.get("diarizer", {}))
    arms = list(diarizer_cfg.pop("arms", DEFAULT_ARMS))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown diarizer arms {unknown}; expected {sorted(ARMS)}")

    # Scalar or list; a list sweeps inside one run so the diarizer is paid for
    # once. 0 is always included so every sweep carries its own baseline and the
    # comparison never depends on a different run's numbers.
    dilate_raw = diarizer_cfg.pop("dilate_overlap_ms", 0.0)
    dilate_ms_values = sorted(
        {float(v) for v in (dilate_raw if isinstance(dilate_raw, (list, tuple)) else [dilate_raw])}
        | {0.0}
    )
    if any(v < 0 for v in dilate_ms_values):
        raise SystemExit("diarizer.dilate_overlap_ms values must be >= 0.")
    if "oracle" not in arms:
        # Guardrail §6.2: a real-diarization number without its oracle companion
        # cannot be attributed to the diarizer rather than to phi or G.
        raise SystemExit(
            "the 'oracle' arm is mandatory (CLAUDE.md §6.2: never report a "
            "real-diarization number without the oracle number beside it)."
        )

    # Same rule as the arm check above: refuse an incoherent refinement config
    # here, not after the corpus has mounted and the checkpoint has loaded.
    refine_cfg = cfg.get("refine", {})
    refine_rounds = int(refine_cfg.get("rounds", 0))
    refine_oracle_ceiling = bool(refine_cfg.get("oracle_ceiling", False))
    refine_oracle_audio = bool(refine_cfg.get("oracle_audio", False))
    rescale_to_mixture = bool(cfg.get("extractor", {}).get("rescale_to_mixture", False))
    if refine_oracle_audio and refine_rounds == 0:
        raise SystemExit(
            "refine.oracle_audio is on but refine.rounds is 0 -- refinement "
            "never runs, so the perfect-extractor bound would measure nothing."
        )
    if refine_oracle_audio and refine_oracle_ceiling:
        raise SystemExit(
            "refine.oracle_audio and refine.oracle_ceiling are both on. They "
            "bound DIFFERENT axes -- the extractor and the acceptance rule -- "
            "and enabling both measures neither: a run that bounds two things "
            "at once cannot attribute its result to either."
        )
    if refine_oracle_ceiling and refine_rounds == 0:
        raise SystemExit(
            "refine.oracle_ceiling is on but refine.rounds is 0 -- refinement "
            "never runs, so the ceiling would measure nothing while producing a "
            "complete-looking table."
        )

    device = _device(args.device)

    dataset = build_dataset(cfg)
    enroll_cfg = cfg.get("enroll", {})
    enroll_k = int(enroll_cfg.get("k", 3))
    min_clip_ms = float(enroll_cfg.get("min_clip_ms", 500.0))
    enroll_budget_raw = enroll_cfg.get("budget_ms")
    enroll_budget_ms = None if enroll_budget_raw is None else float(enroll_budget_raw)

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = extractor_cfg.pop("checkpoint", None)
    gate_cfg = cfg.get("gate", {})

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(
        checkpoint_path=checkpoint_path, device=device, **extractor_cfg
    )

    print(
        f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  "
        f"@ {sample_rate} Hz  fade={fade} samples  arms={arms}  "
        f"dilate_ms={dilate_ms_values}"
    )
    if refine_oracle_ceiling:
        print(
            "\n*** refine.oracle_ceiling is ON. `coarse_to_fine` accepts a\n"
            "*** refinement candidate by comparing SI-SDR against the CLEAN\n"
            "*** SOURCES. This is a scoring-time upper bound and is NOT a\n"
            "*** deployable system -- never report its rows as a result.\n"
        )

    rows: list[dict] = []
    gate_rows: list[dict] = []
    diar_rows: list[dict] = []
    overall_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    diarizer_cache: dict = {}
    arm_failures: dict[str, int] = {}
    dilation_failures: dict[tuple[str, float], int] = {}
    for scene in dataset:
        try:
            scene_rows, scene_gate_rows, scene_diar_rows, scene_overall_rows = score_scene_all_arms(
                scene, arms, diarizer_cfg, device,
                fade=fade, enroll_k=enroll_k, min_clip_ms=min_clip_ms,
                enroll_budget_ms=enroll_budget_ms, encoder=encoder,
                extractor=extractor, gate_cfg=gate_cfg,
                refine_rounds=refine_rounds, diarizer_cache=diarizer_cache,
                arm_failures=arm_failures,
                dilate_ms_values=dilate_ms_values,
                refine_oracle_ceiling=refine_oracle_ceiling,
                refine_oracle_audio=refine_oracle_audio,
                rescale_to_mixture=rescale_to_mixture,
                dilation_failures=dilation_failures,
            )
        except NoSoloRegionError as exc:
            # Benign and expected under real diarization: a predicted cluster may
            # simply have no solo audio to enroll from. Skipping the whole SCENE
            # (not just that speaker) keeps every arm covering the same rows, so
            # the paired differences stay paired.
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")
            continue
        rows.extend(scene_rows)
        gate_rows.extend(scene_gate_rows)
        diar_rows.extend(scene_diar_rows)
        overall_rows.extend(scene_overall_rows)
        print(f"scored scene {scene.name!r} ({len(scene_rows)} rows across {len(arms)} arms)")

    if skipped:
        print(
            f"[enroll] skipped {len(skipped)}/{len(dataset)} scenes during enrollment "
            f"(see per-scene messages above): {[name for name, _ in skipped]}"
        )
    if arm_failures:
        print("\n[diarize] per-arm scene failures (arm skipped, scene kept):")
        for arm, count in sorted(arm_failures.items()):
            pct = 100.0 * count / max(1, len(dataset))
            print(f"    {arm}: {count}/{len(dataset)} scenes ({pct:.0f}%)")
        print(
            "    An arm that failed on a large fraction of scenes is not a usable\n"
            "    comparison -- read its `n` in the gap table before quoting it."
        )

    n_src = cfg["dataset"].get("n_src", 2)
    stem = f"phase3_{cfg['dataset']['name']}_{n_src}spk"
    tag = cfg.get("eval", {}).get("tag")
    if tag:
        stem = f"{stem}_{tag}"
    results_dir = Path(cfg.get("eval", {}).get("results_dir", "results"))
    _write_results(rows, gate_rows, diar_rows, overall_rows, results_dir, stem, arms,
                   arm_failures=arm_failures, n_scenes=len(dataset),
                   dilation_failures=dilation_failures,
                   refine_oracle_ceiling=refine_oracle_ceiling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
