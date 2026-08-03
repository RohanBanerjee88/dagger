#!/usr/bin/env python3
"""Enrollment-budget sweep: does coarse-to-fine refinement pay when the solo
embedding is starved?

Coarse-to-fine blends the solo-derived enrollment embedding toward one computed
from *extracted overlap* audio. That only helps when the solo clip is the weaker
estimate. At the default ~1 s of clean solo it isn't: refinement costs 0.2-1.1 dB
against ``no_recursion``, which is the same pipeline without the blend. This
script tabulates that cost against ``enroll.budget_ms`` to find the crossover --
or to establish there isn't one, which is equally reportable.

Read the ``coarse_to_fine - no_recursion`` column. Both systems degrade as the
budget shrinks (a worse embedding means worse extraction for everyone), so the
absolute numbers falling is expected and uninformative; the question is purely
whether the DIFFERENCE crosses zero.

Two properties make this comparison unusually clean:

* ``budget_ms`` truncates the enrollment clip and changes nothing else, so every
  sweep point has byte-identical scenes, ``x_O``, and depth arrays. Rows are
  therefore paired across sweep points, not merely averaged.
* ``no_recursion`` and ``coarse_to_fine`` draw audio from the SAME
  ``reconstruct_all`` call and differ only in which embeddings go in, so their
  difference isolates refinement with nothing else varying.

    python scripts/aggregate_enroll_budget.py \\
        full=results/phase2_librimix_5spk_curriculum345full.csv \\
        800=results/phase2_librimix_5spk_budget800.csv \\
        500=results/phase2_librimix_5spk_budget500.csv \\
        300=results/phase2_librimix_5spk_budget300.csv \\
        150=results/phase2_librimix_5spk_budget150.csv \\
        --out results/phase2_enroll_budget.md
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# Matches scripts/run_phase2.py's SI_SDR_CAP_DB.
SI_SDR_CAP_DB = 50.0

TREATMENT = "coarse_to_fine"
CONTROL = "no_recursion"


def _load(csv_path: Path) -> dict[str, dict[tuple, float]]:
    """``{system: {(scene, speaker, depth): si_sdr}}`` for the two systems compared."""
    out: dict[str, dict[tuple, float]] = {TREATMENT: {}, CONTROL: {}}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            system = row["system"]
            if system not in out:
                continue
            value = float(row["si_sdr"])
            if math.isnan(value):  # speaker not active at this depth
                continue
            value = max(-SI_SDR_CAP_DB, min(SI_SDR_CAP_DB, value))
            out[system][(row["scene"], row["speaker"], int(row["depth"]))] = value
    return out


def _paired(points: list[tuple[str, dict]], depth: int | None) -> list[dict]:
    rows = []
    for label, systems in points:
        treatment, control = systems[TREATMENT], systems[CONTROL]
        keys = set(treatment) & set(control)
        if depth is not None:
            keys = {key for key in keys if key[2] == depth}
        if not keys:
            rows.append({"label": label, "n": 0})
            continue
        diffs = np.asarray([treatment[key] - control[key] for key in sorted(keys)], dtype=np.float64)
        rows.append({
            "label": label,
            "n": diffs.size,
            "mean": float(diffs.mean()),
            "median": float(np.median(diffs)),
            # Ties are the norm here: a speaker whose refinement was rejected in
            # every round keeps its enrollment embedding, so its output is
            # bit-identical to no_recursion's. Reporting win/tie/loss separately
            # keeps those out of the "loss" bucket, where a bare win rate puts
            # them.
            "wins": int((diffs > 0).sum()),
            "ties": int((diffs == 0).sum()),
            "losses": int((diffs < 0).sum()),
            "control_mean": float(np.mean([control[key] for key in sorted(keys)])),
        })
    return rows


def _section(points: list[tuple[str, dict]], depth: int | None) -> list[str]:
    title = f"depth {depth}" if depth is not None else "all depths pooled"
    lines = [
        "", f"## {TREATMENT} - {CONTROL} ({title})", "",
        "| enroll budget (ms) | n | mean diff | median | win / tie / loss | control mean |",
        "|---|---|---|---|---|---|",
    ]
    rows = _paired(points, depth)
    for row in rows:
        if not row["n"]:
            lines.append(f"| {row['label']} | 0 | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {row['label']} | {row['n']} | {row['mean']:+.2f} | {row['median']:+.2f} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} | {row['control_mean']:.2f} |"
        )

    scored = [row for row in rows if row["n"]]
    if scored:
        best = max(scored, key=lambda row: row["mean"])
        crossed = [row["label"] for row in scored if row["mean"] > 0]
        lines += [""]
        if crossed:
            lines.append(
                f"**Crossover found.** Refinement is net-positive at budget(s): "
                f"{', '.join(crossed)}. Best is {best['label']} ms at {best['mean']:+.2f} dB."
            )
        else:
            lines.append(
                f"**No crossover.** Refinement stays net-negative at every budget swept; "
                f"the least-bad is {best['label']} ms at {best['mean']:+.2f} dB. If the trend "
                "is still rising at the smallest budget, extend the sweep downward before "
                "concluding; if it is flat or falling, starving enrollment is not the lever "
                "and the next test is heterogeneous-source scenes (solo from one utterance, "
                "overlap from another of the same speaker)."
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "points", nargs="+",
        help="LABEL=path.csv, one per sweep point (label is the budget in ms, or 'full')",
    )
    parser.add_argument("--out", type=Path, default=Path("results/phase2_enroll_budget.md"))
    parser.add_argument(
        "--depths", type=int, nargs="*", default=None,
        help="depths to tabulate (default: every depth present, plus a pooled table)",
    )
    args = parser.parse_args()

    points = []
    for item in args.points:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=path.csv, got {item!r}")
        label, _, path = item.partition("=")
        points.append((label, _load(Path(path))))

    depths = args.depths
    if depths is None:
        depths = sorted({key[2] for _, systems in points for key in systems[CONTROL]})

    lines = [
        "# Phase 2 -- enrollment-budget sweep", "",
        "Does coarse-to-fine refinement pay once the solo embedding is starved?", "",
        "Both systems degrade as the budget shrinks -- only the DIFFERENCE matters. "
        "`budget_ms` truncates the enrollment clip and nothing else, so scenes are "
        "identical across sweep points and rows are paired.", "",
        "points: " + ", ".join(label for label, _ in points), "",
    ]
    for depth in depths:
        lines += _section(points, depth)
    lines += _section(points, None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
