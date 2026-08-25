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

from dagger.data import build_dataset
from dagger.data.paths import load_env
from dagger.enroll.encoder import TitaNetEncoder
from dagger.enroll.topk import NoSoloRegionError
from dagger.extract.tfgridnet_crossattn import TFGridNetCrossAttnExtractor
from dagger.metrics.phase2_scores import SI_SDR_CAP_DB, clip_score

# The four systems and the per-scene scoring body now live in dagger.eval.systems,
# shared with scripts/run_phase3.py, which runs the same comparison twice per
# scene (oracle regions vs. real ones). Re-exported here so every name stays
# importable at this path -- the reporting tests load this script by path and
# reach for them. Same reasoning as dagger.metrics.phase2_scores one level down:
# a second copy is how the 2026-07-26 +-inf bug came to need fixing twice.
from dagger.eval.systems import (  # noqa: E402  (kept beside its siblings)
    DEFLATION_SYSTEMS,
    GATE_FIELDS,
    SCORE_FIELDS,
    SYSTEMS,
    accepted_before as _accepted_before,
    deflation_order,
    make_gate_fn as _make_gate_fn,
    score_scene,
)

_clip_score = clip_score


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        # extrasaction="ignore": score_scene also returns a Phase 3-only
        # `cluster` column (which predicted cluster a row came from). Ignoring it
        # keeps this file's schema and bytes exactly as they were.
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(gate_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GATE_FIELDS, extrasaction="ignore")
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
    refine_rounds = int(cfg.get("refine", {}).get("rounds", 0))

    encoder = TitaNetEncoder(device=device)
    extractor = TFGridNetCrossAttnExtractor(checkpoint_path=checkpoint_path, device=device, **extractor_cfg)

    print(f"dataset: {cfg['dataset']['name']}  scenes: {len(dataset)}  @ {sample_rate} Hz  fade={fade} samples")

    rows: list[dict] = []
    gate_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for scene in dataset:
        try:
            # Third element is the un-stratified overall rows (Phase 3 writes
            # them to `_overall.csv`). Deliberately dropped here: Phase 2's
            # committed CSVs must stay byte-identical, so this script's
            # outputs are left exactly as they were.
            scene_rows, scene_gate_rows, _ = score_scene(
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
