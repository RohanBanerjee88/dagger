#!/usr/bin/env python3
"""Phase 3's DoD table: the oracle-vs-real gap, decomposed.

CPU-only, reads the CSVs ``scripts/run_phase3.py`` wrote. Separate from the
run script for the same reason ``aggregate_phase2.py`` is: regenerating a table
after a reporting fix must not require re-running a GPU eval.

Every number here is a **paired** difference on matched
``(source, scene, speaker, depth, system)`` rows, via
:func:`dagger.metrics.phase2_scores.paired_by_field`. Both arms scored the
identical scenes in the identical pass, so pairing cancels scene difficulty
exactly and every non-arm source of variance with it.

Four decompositions, each isolating one thing (see ``run_phase3.py``'s docstring
for why the last two exist at all):

* ``real - oracle``            -- the total cost of real diarization (headline)
* ``real_index_order - oracle`` -- diarization error alone
* ``real - real_index_order``  -- the reordering a nonzero ``V_i`` induces
* ``real_forced_m - real``     -- how much of the cost was speaker *counting*

Reported per (system x depth) and per (system x ``n_accepted_before``) with
``n``, mean, SEM, win rate and ``|t|`` together — CLAUDE.md §7 requires all four:
SEM is the precision of the mean (not the spread), and a positive mean at a ~50%
win rate is a few large wins rather than broad superiority.

Usage::

    python scripts/aggregate_phase3.py results/phase3/dod/phase3_librimix_3spk_oracle_vs_real.csv \\
        --out results/phase3/dod/phase3_oracle_vs_real.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.metrics.phase2_scores import (
    SYSTEMS,
    load_score_rows,
    mean_sem,
    paired_by_field,
)

#: (left arm, right arm, what the difference isolates).
COMPARISONS = [
    ("real", "oracle", "total cost of real diarization"),
    ("real_index_order", "oracle", "diarization error alone"),
    ("real", "real_index_order", "the reordering V_i induces"),
    ("real_forced_m", "real", "how much of the cost was speaker counting"),
]


def _stats(diffs: list[float]) -> tuple[float, float, int, float, float]:
    """``(mean, sem, n, win_rate, |t|)`` for a list of paired differences."""
    mean, sem, n = mean_sem(diffs)
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    wins = sum(1 for d in diffs if d > 0) / n
    t = abs(mean / sem) if sem and not np.isnan(sem) and sem > 0 else float("nan")
    return mean, sem, n, wins, t


def _table(rows: list[dict], left: str, right: str, x_field: str) -> list[str]:
    """One comparison, stratified over ``x_field`` (``depth`` or accumulation)."""
    xs = sorted({r[x_field] for r in rows if r.get(x_field) is not None})
    lines = [
        f"| system | {x_field} | n | mean (dB) | SEM | win rate | \\|t\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    any_row = False
    for system_name in SYSTEMS:
        for x in xs:
            subset = [
                r for r in rows
                if r["system"] == system_name and r.get(x_field) == x
            ]
            diffs = paired_by_field(subset, "diarization", left, right)
            if not diffs:
                continue
            any_row = True
            mean, sem, n, wins, t = _stats(diffs)
            lines.append(
                f"| {system_name} | {x} | {n} | {mean:+.2f} | {sem:.2f} | "
                f"{wins:.0%} | {t:.1f} |"
            )
    if not any_row:
        return ["(no paired rows for this comparison)", ""]
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="run_phase3.py score CSV(s)")
    parser.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = parser.parse_args()

    rows = load_score_rows(args.csv, required_columns=("m", "diarization"))
    arms = sorted({r["diarization"] for r in rows if r["diarization"]})
    if "oracle" not in arms:
        raise SystemExit(
            "no 'oracle' rows in these CSVs. Guardrail §6.2: a real-diarization "
            "number is not interpretable without the oracle beside it."
        )

    lines = [
        "# Phase 3 -- oracle-vs-real gap", "",
        f"sources: {', '.join(p.name for p in args.csv)}",
        f"arms present: {', '.join(arms)}",
        f"paired rows loaded: {len(rows)}", "",
        "All differences are PAIRED on matched (scene, speaker, depth, system) rows,",
        "so scene difficulty cancels exactly. SEM is the precision of the mean, not",
        "the spread of the data (CLAUDE.md §7) -- the two differ by ~30x here.", "",
    ]

    for left, right, blurb in COMPARISONS:
        if left not in arms or right not in arms:
            lines += [f"## `{left}` - `{right}`", "", "(arm not present in this run)", ""]
            continue
        lines += [f"## `{left}` - `{right}` -- {blurb}", ""]
        lines += ["### by overlap depth", ""]
        lines += _table(rows, left, right, "depth")
        lines += [
            "### by accumulation (`n_accepted_before`, deflation systems only)", "",
            "The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic",
            "difficulty and hits every system equally, which buried the",
            "accumulation effect for five Phase 2 runs.", "",
        ]
        lines += _table(rows, left, right, "n_accepted_before")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
