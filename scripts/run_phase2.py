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

Three files are written to
``results/phase2_<dataset>_<n_src>spk[_<eval.tag>]`` (``eval.tag``, optional,
distinguishes runs against different checkpoints so one run never overwrites
another's results):

* ``.csv``       -- long-format scores, one row per (speaker, depth, system);
                    see ``SCORE_FIELDS``. Alongside ``depth`` (concurrent
                    voices at a sample) it carries ``m`` (speakers in the
                    scene) and ``deflation_index``/``n_accepted_before`` (how
                    many prior estimates were subtracted into the residual
                    before this speaker was extracted). Those last two are the
                    counter Theorem 3's accumulation penalty is indexed by --
                    depth is not, since deflation runs once per scene over all
                    ``m`` speakers regardless of which depth region is later
                    scored.
* ``_gate.csv``  -- confidence-gate decisions, one row per (speaker, system,
                    round); see ``GATE_FIELDS``.
* ``.md``        -- human-readable summary tables.

Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/run_phase2.py \\
        --config configs/phase2/experiments/phase2_librimix_3spk_eval.yaml
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

# The two systems that run dagger.reconstruct.deflation, i.e. the only ones for
# which `deflation_index`/`n_accepted_before` mean anything. Membership is
# explicit rather than a name test (`endswith("deflation")`) so adding a system
# can't silently opt it in.
DEFLATION_SYSTEMS = ("ungated_deflation", "gated_deflation")

SCORE_FIELDS = [
    "scene", "speaker", "system", "m", "depth", "si_sdr",
    "deflation_index", "n_accepted_before", "refine_rounds",
]
# Gate decisions live in their own file, at their own grain: one per (scene,
# speaker, system, round). Folding them into the score rows would duplicate each
# decision across however many depths that speaker happens to span, so any
# accept rate counted from those rows would be inflated by a depth-dependent
# factor. Keeping the two grains in separate files makes that mistake hard.
GATE_FIELDS = [
    "scene", "speaker", "system", "round",
    "accepted", "mean_variance", "margin", "vad_coverage", "artifact_score", "reason",
]

# si_sdr() legitimately returns +-inf (a perfect estimate / a silent estimate
# against real target energy -- see dagger.metrics.sisdr.si_sdr's docstring),
# not "undefined." Only nan ("speaker not active here, nothing to score") should
# be excluded from an aggregate mean; +-inf are real, informative outcomes and
# get clipped to this cap so one perfect/failed row can't make the mean itself
# +-inf. A plain np.isfinite() filter would silently drop both alongside nan --
# that was a real Phase 2 bug (44% of depth-1 rows are exactly +inf, being
# solo copy-through, and were vanishing from the reported depth-1 mean).
SI_SDR_CAP_DB = 50.0


def _clip_score(value: float) -> float | None:
    """``None`` for nan (exclude); ``value`` clipped to +-SI_SDR_CAP_DB otherwise."""
    if np.isnan(value):
        return None
    return float(np.clip(value, -SI_SDR_CAP_DB, SI_SDR_CAP_DB))


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


def _accepted_before(order: list[int], accept_sequence: np.ndarray) -> list[int]:
    """Speaker-indexed count of accepted estimates subtracted into the residual
    *before* each speaker was extracted -- the accumulation counter Theorem 3's
    ``L*||E_(m-1)||`` penalty is indexed by.

    ``accept_sequence`` is speaker-indexed (see
    :func:`dagger.reconstruct.deflation.reconstruct_all_deflation`), so the
    prefix has to be taken over ``order``, not over speaker ids. Ungated
    deflation accepts unconditionally, so there this must equal each speaker's
    position in ``order``; gated deflation can only be lower.
    """
    return [
        sum(int(accept_sequence[j]) for j in order[: order.index(i)])
        for i in range(len(order))
    ]


def score_scene(
    scene: Scene,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    enroll_budget_ms: float | None,
    encoder: TitaNetEncoder,
    extractor: TFGridNetCrossAttnExtractor,
    gate_cfg: dict,
    refine_rounds: int,
) -> tuple[list[dict], list[dict]]:
    """Run all four Phase 2 systems on one scene.

    Returns ``(score_rows, gate_rows)`` -- one score row per (speaker, depth,
    system), and one gate row per (speaker, system, round) for the two systems
    that actually run the confidence gate. See ``SCORE_FIELDS``/``GATE_FIELDS``
    for why the two are kept apart.
    """
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
            encoder, k=enroll_k, min_clip_ms=min_clip_ms, budget_ms=enroll_budget_ms,
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
    deflation_idx: dict[int, int] = {i: j for j, i in enumerate(order)}  # speaker -> position in order

    outputs: dict[str, np.ndarray] = {}
    outputs["no_recursion"] = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)

    # Each deflation call binds its OWN result names: reusing one pair across
    # both calls would let the gated run's telemetry overwrite the ungated
    # run's and end up reported against the wrong system.
    outputs["ungated_deflation"], _, ungated_accepts = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=None, fade=fade,
    )
    gate_fn = _make_gate_fn(embeddings, variances, activity, overlap, encoder, scene.sample_rate, gate_cfg)
    outputs["gated_deflation"], gated_gate_results, gated_accepts = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=gate_fn, fade=fade,
    )

    n_accepted_before = {
        "ungated_deflation": _accepted_before(order, ungated_accepts),
        "gated_deflation": _accepted_before(order, gated_accepts),
    }
    # Ungated accepts unconditionally, so its accumulation count must be exactly
    # each speaker's position in `order`. Deriving it from the returned accept
    # sequence rather than hardcoding that lets this assertion catch a
    # regression in the deflation loop itself.
    assert n_accepted_before["ungated_deflation"] == [deflation_idx[i] for i in range(num_speakers)]

    refined_embeddings, round_results = refine_embeddings(
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
        is_deflation = system_name in DEFLATION_SYSTEMS
        for i, spk in enumerate(speakers):
            by_depth = si_sdr_by_depth(system_outputs[i], scene.sources[i], depth)
            for k, score in by_depth.items():
                rows.append({
                    "scene": scene.name,
                    "speaker": spk,
                    "system": system_name,
                    "m": num_speakers,
                    "depth": k,
                    "si_sdr": score,
                    # None (not 0) for the accumulation-free systems: 0 is a
                    # legitimate deflation_index (the first speaker in `order`),
                    # so writing it here would make "no accumulation" and "not
                    # applicable" indistinguishable in the CSV.
                    "deflation_index": deflation_idx[i] if is_deflation else None,
                    "n_accepted_before": n_accepted_before[system_name][i] if is_deflation else None,
                    "refine_rounds": refine_rounds if system_name == "coarse_to_fine" else 0,
                })

    gate_rows = []
    for i, spk in enumerate(speakers):
        result = gated_gate_results[i]
        if result is not None:
            gate_rows.append({
                "scene": scene.name, "speaker": spk, "system": "gated_deflation", "round": 0,
                "accepted": result.accepted, "mean_variance": result.mean_variance,
                "margin": result.margin, "vad_coverage": result.vad_coverage,
                "artifact_score": result.artifact_score, "reason": result.reason,
            })
    for round_index, per_speaker in enumerate(round_results):
        for i, spk in enumerate(speakers):
            result = per_speaker[i]
            row = {
                "scene": scene.name, "speaker": spk, "system": "coarse_to_fine",
                "round": round_index,
            }
            if result is None:
                # No overlap-only region to re-embed from, so no gate decision
                # was made. Recorded rather than skipped -- a missing row would
                # be silently indistinguishable from a rejection when accept
                # rates are counted.
                row.update({
                    "accepted": None, "mean_variance": None, "margin": None,
                    "vad_coverage": None, "artifact_score": None,
                    "reason": "no_overlap_clip",
                })
            else:
                row.update({
                    "accepted": result.accepted, "mean_variance": result.mean_variance,
                    "margin": result.margin, "vad_coverage": result.vad_coverage,
                    "artifact_score": result.artifact_score, "reason": result.reason,
                })
            gate_rows.append(row)

    return rows, gate_rows


def _select(rows: list[dict], **equals) -> list[dict]:
    """Rows matching every ``field=value`` constraint."""
    return [r for r in rows if all(r[field] == value for field, value in equals.items())]


def _clipped(rows: list[dict]) -> list[float]:
    """Scoreable SI-SDR values: nan dropped, +-inf clipped to +-SI_SDR_CAP_DB."""
    return [c for c in (_clip_score(r["si_sdr"]) for r in rows) if c is not None]


def _spread_section(rows: list[dict], depths: list[int]) -> list[str]:
    """Mean alone can't distinguish "consistently mediocre" from "usually great
    with a few bad scenes" -- and the Phase 1 precedent (a +2.35 dB mean at a
    50% win rate) is exactly that failure. Percentiles are preferred to raw
    min/max as the headline spread because one freak-bad scene moves min but
    not p5; both are reported.
    """
    lines = [
        "", "## Spread (per system/depth)", "",
        f"(p95 saturates at the +-{SI_SDR_CAP_DB:.0f} dB clip wherever the diagnostic-counts "
        "table shows many perfect/failed rows -- at depth 1 most rows are solo copy-through "
        "and legitimately +inf, so a p95 of exactly the cap there is expected, not a bug)",
        "",
        "| system | depth | n | mean | p5 | p95 | min | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for system_name in SYSTEMS:
        for k in depths:
            values = _clipped(_select(rows, system=system_name, depth=k))
            if not values:
                lines.append(f"| {system_name} | {k} | 0 | -- | -- | -- | -- | -- |")
                continue
            a = np.asarray(values, dtype=np.float64)
            lines.append(
                f"| {system_name} | {k} | {len(values)} | {a.mean():.2f} | "
                f"{np.percentile(a, 5):.2f} | {np.percentile(a, 95):.2f} | "
                f"{a.min():.2f} | {a.max():.2f} |"
            )
    return lines


def _accumulation_section(rows: list[dict], depths: list[int]) -> list[str]:
    """SI-SDR by how many prior estimates were deflated into the residual.

    This is the axis Theorem 3's ``L*||E_(m-1)||`` penalty is indexed by, and
    unlike a cross-eval-set sweep over ``m`` it is a WITHIN-SCENE control: one
    m-speaker scene contributes rows at accumulation 0..m-1 with the acoustics,
    the enrollment, and the checkpoint all held fixed. If deflation accumulates
    error, ungated_deflation declines along this axis at every fixed depth,
    while the accumulation-free systems (shown as the n/a reference row) have
    no such axis at all.
    """
    lines = [
        "", "## SI-SDR by accumulation (`n_accepted_before` -- prior estimates subtracted "
        "into the residual before this speaker was extracted)", "",
        "(within-scene control: depth is held fixed down each column, so a decline "
        "across rows is accumulation, not intrinsic overlap difficulty)", "",
        "| system | n_accepted_before | " + " | ".join(f"depth {k}" for k in depths) + " |",
        "|---|---" + "|---" * len(depths) + "|",
    ]

    def _row(label: str, subset: list[dict]) -> str:
        cells = []
        for k in depths:
            values = _clipped(_select(subset, depth=k))
            cells.append(f"{np.mean(values):.2f} (n={len(values)})" if values else "--")
        return f"| {label} | " + " | ".join(cells) + " |"

    for system_name in DEFLATION_SYSTEMS:
        subset = _select(rows, system=system_name)
        for count in sorted({r["n_accepted_before"] for r in subset}):
            lines.append(_row(f"{system_name} | {count}", _select(subset, n_accepted_before=count)))
    for system_name in SYSTEMS:
        if system_name not in DEFLATION_SYSTEMS:
            lines.append(_row(f"{system_name} | n/a", _select(rows, system=system_name)))
    return lines


def _paired_section(rows: list[dict], depths: list[int], system_a: str, system_b: str) -> list[str]:
    """Per-row paired difference ``system_a - system_b``, joined on
    ``(scene, speaker, depth)``.

    Both systems score the same speakers over the same depth array, so the key
    sets are identical and the join is lossless -- asserted below rather than
    assumed, since a silent partial join would quietly change what the win rate
    is a rate *of*.
    """
    def index(system_name: str) -> dict[tuple, float]:
        out = {}
        for r in _select(rows, system=system_name):
            score = _clip_score(r["si_sdr"])
            if score is not None:
                out[(r["scene"], r["speaker"], r["depth"])] = score
        return out

    idx_a, idx_b = index(system_a), index(system_b)
    shared = set(idx_a) & set(idx_b)
    lines = [
        "", f"## Paired difference: {system_a} - {system_b}", "",
        f"(joined on (scene, speaker, depth); {len(shared)} paired rows out of "
        f"{len(idx_a)}/{len(idx_b)} scoreable per system. A positive mean with a "
        "~50% win rate means the margin comes from a few large wins, not broad "
        "superiority -- that was the Phase 1 result, so it is reported here.)", "",
        "| depth | pairs | mean diff | median | p5 | p95 | win rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in depths:
        diffs = [idx_a[key] - idx_b[key] for key in shared if key[2] == k]
        if not diffs:
            lines.append(f"| {k} | 0 | -- | -- | -- | -- | -- |")
            continue
        a = np.asarray(diffs, dtype=np.float64)
        win_rate = float(np.mean(a > 0)) * 100.0
        lines.append(
            f"| {k} | {len(diffs)} | {a.mean():.2f} | {np.median(a):.2f} | "
            f"{np.percentile(a, 5):.2f} | {np.percentile(a, 95):.2f} | {win_rate:.1f}% |"
        )
    return lines


def _gate_section(gate_rows: list[dict]) -> list[str]:
    """Confidence-gate accept rates, per system and round.

    Counted from the gate rows (one decision per speaker/round), never from the
    score rows -- a decision duplicated across every depth a speaker spans would
    inflate this by a depth-dependent factor. A ~100% accept rate means the gate
    is rubber-stamping and its thresholds are inert; a ~0% rate means
    gated_deflation has degenerated into no_recursion. Both are degenerate, and
    neither is visible from SI-SDR alone.
    """
    lines = [
        "", "## Confidence gate", "",
        "| system | round | decisions | accepted | accept rate | no clip | reasons (rejections) |",
        "|---|---|---|---|---|---|---|",
    ]
    for system_name in sorted({r["system"] for r in gate_rows}):
        subset = _select(gate_rows, system=system_name)
        for round_index in sorted({r["round"] for r in subset}):
            per_round = _select(subset, round=round_index)
            decided = [r for r in per_round if r["accepted"] is not None]
            accepted = [r for r in decided if r["accepted"]]
            no_clip = len(per_round) - len(decided)
            rate = f"{100.0 * len(accepted) / len(decided):.1f}%" if decided else "--"
            reasons: dict[str, int] = {}
            for r in decided:
                if not r["accepted"]:
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            summary = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "--"
            lines.append(
                f"| {system_name} | {round_index} | {len(decided)} | {len(accepted)} | "
                f"{rate} | {no_clip} | {summary} |"
            )
    return lines


def _write_results(rows: list[dict], gate_rows: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    gate_csv_path = out_dir / f"{stem}_gate.csv"
    md_path = out_dir / f"{stem}.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(gate_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GATE_FIELDS)
        writer.writeheader()
        writer.writerows(gate_rows)

    depths = sorted({r["depth"] for r in rows})
    lines = [
        f"# Phase 2 results -- {stem}", "", f"rows scored: {len(rows)}", "",
        f"(means clip +-inf to +-{SI_SDR_CAP_DB:.0f} dB rather than dropping them -- "
        "see the diagnostic-counts table for how often that happens per system/depth)",
        "",
    ]
    header = "| system | " + " | ".join(f"depth {k}" for k in depths) + " |"
    sep = "|---" * (len(depths) + 1) + "|"
    lines += [header, sep]
    means: dict[str, dict[int, float]] = {}
    diagnostics: dict[str, dict[int, dict[str, int]]] = {}
    for system_name in SYSTEMS:
        per_depth = {}
        per_depth_diag = {}
        for k in depths:
            raw = [
                r["si_sdr"] for r in rows if r["system"] == system_name and r["depth"] == k
            ]
            clipped = [c for c in (_clip_score(v) for v in raw) if c is not None]
            per_depth[k] = float(np.mean(clipped)) if clipped else float("nan")
            per_depth_diag[k] = {
                "nan": sum(1 for v in raw if np.isnan(v)),
                "perfect": sum(1 for v in raw if v == float("inf")),
                "failed": sum(1 for v in raw if v == float("-inf")),
                "scored": len(clipped),
            }
        means[system_name] = per_depth
        diagnostics[system_name] = per_depth_diag
        cells = " | ".join(f"{per_depth[k]:.2f}" for k in depths)
        lines.append(f"| {system_name} | {cells} |")

    lines += [
        "", "## Diagnostic counts (per system/depth: absent / perfect / failed / scored)",
        "", "| system | depth | absent (nan) | perfect (+inf) | failed (-inf) | scored |",
        "|---|---|---|---|---|---|",
    ]
    for system_name in SYSTEMS:
        for k in depths:
            d = diagnostics[system_name][k]
            lines.append(f"| {system_name} | {k} | {d['nan']} | {d['perfect']} | {d['failed']} | {d['scored']} |")

    lines += _spread_section(rows, depths)
    lines += _accumulation_section(rows, depths)
    # The thesis (accumulation-free beats deflation) and the control comparison
    # (does refinement actually help, or does it cost quality vs. not recursing
    # at all?) -- both draw audio from the same reconstruct_all, so a negative
    # second table isolates the embedding refinement as the cause.
    lines += _paired_section(rows, depths, "coarse_to_fine", "ungated_deflation")
    lines += _paired_section(rows, depths, "coarse_to_fine", "no_recursion")
    lines += _gate_section(gate_rows)

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
    print(f"wrote {csv_path}, {gate_csv_path} and {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2/experiments/phase2_librimix_3spk_eval.yaml")
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
    # Cap on how much solo audio each enrollment clip contributes. None (the
    # default) means whole clips -- every config predating this key is
    # unaffected. Used by the enrollment-budget sweep to test whether
    # coarse-to-fine refinement pays once the solo embedding is the weaker
    # estimate; capping the clip leaves scene geometry untouched, so sweep
    # points stay row-for-row comparable.
    enroll_budget_raw = enroll_cfg.get("budget_ms")
    enroll_budget_ms = None if enroll_budget_raw is None else float(enroll_budget_raw)

    extractor_cfg = dict(cfg.get("extractor", {}))
    checkpoint_path = extractor_cfg.pop("checkpoint", None)
    gate_cfg = cfg.get("gate", {})
    refine_rounds = int(cfg.get("refine", {}).get("rounds", 2))

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(checkpoint_path=checkpoint_path, device=device, **extractor_cfg)

    print(f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  @ {sample_rate} Hz  fade={fade} samples")

    rows: list[dict] = []
    gate_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for scene in dataset:
        try:
            scene_rows, scene_gate_rows = score_scene(
                scene, fade, enroll_k, min_clip_ms, enroll_budget_ms,
                encoder, extractor, gate_cfg, refine_rounds,
            )
        except NoSoloRegionError as exc:
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")
            continue
        rows.extend(scene_rows)
        gate_rows.extend(scene_gate_rows)
        print(f"scored scene {scene.name!r} ({len(scene_rows)} rows)")

    if skipped:
        print(
            f"[enroll] skipped {len(skipped)}/{len(dataset)} scenes during enrollment "
            f"(see per-scene messages above): {[name for name, _ in skipped]}"
        )

    n_src = cfg["dataset"].get("n_src", 2)
    stem = f"phase2_{cfg['dataset']['name']}_{n_src}spk"
    tag = cfg.get("eval", {}).get("tag")
    if tag:
        # Distinguishes runs against different checkpoints/configs (e.g. the
        # original Phase 1 checkpoint vs. a scheduled-placement fine-tune) so
        # a later run never silently overwrites an earlier one's results --
        # the stem alone doesn't depend on which checkpoint was used.
        stem = f"{stem}_{tag}"
    results_dir = Path(cfg.get("eval", {}).get("results_dir", "results"))
    _write_results(rows, gate_rows, results_dir, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
