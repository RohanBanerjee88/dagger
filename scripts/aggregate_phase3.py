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
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.metrics.phase2_scores import (
    SYSTEMS,
    clip_score,
    load_score_rows,
    mean_sem,
    paired_rows_by_field,
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
    """One comparison, stratified over ``x_field`` (``depth`` or accumulation).

    **Pairs first, buckets second.** Filtering on ``x_field`` before pairing --
    which this did until 2026-08-18 -- keeps only the speakers whose ``x_field``
    value coincides across the two arms. On ``depth`` that is a no-op, since
    depth is derived from the reference activity for both arms and they always
    agree. On ``n_accepted_before`` it silently discards most of the sample, and
    discards it *because of* the very effect being measured: deflation order is
    ascending ``V_i``, which real diarization turns from a no-op into a real
    permutation. Measured cost: n = 34/28/16 against 98/92/92 for the arm whose
    order matches oracle by construction.

    Buckets are the **reference (right) arm's** value, which is also the only
    reading that carries meaning: "for a speaker the reference arm placed at
    level k, what did the left arm cost it?" Bucketing on the left arm would ask
    a question about a position the diarizer itself chose.
    """
    # `dilate_ms` joins the pairing key so an arm-vs-arm difference is always
    # taken at the SAME dilation. A swept CSV holds several different pipelines;
    # pairing `real` at 50 ms against `oracle` at 0 ms would silently measure the
    # knob and the diarizer together and report the sum as the diarizer's cost.
    pairs = paired_rows_by_field(
        rows, "diarization", left, right,
        key_fields=("source", "scene", "speaker", "depth", "system", "dilate_ms"),
    )
    # Blank is not zero for the accumulation-free systems -- 0 means "nothing was
    # deflated before me", which is a different statement from "this system never
    # deflates" (see _OPTIONAL_INT_FIELDS in dagger.metrics.phase2_scores). Such
    # rows carry no accumulation level and are dropped rather than bucketed at 0.
    buckets: dict[tuple[str, int], list[float]] = {}
    for left_row, right_row in pairs:
        x = right_row.get(x_field)
        if x is None:
            continue
        buckets.setdefault((left_row["system"], int(x)), []).append(
            left_row["si_sdr"] - right_row["si_sdr"]
        )

    lines = [
        f"| system | {x_field} | n | mean (dB) | SEM | win rate | \\|t\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    any_row = False
    for system_name in SYSTEMS:
        for x in sorted(x for s, x in buckets if s == system_name):
            diffs = buckets[(system_name, x)]
            any_row = True
            mean, sem, n, wins, t = _stats(diffs)
            lines.append(
                f"| {system_name} | {x} | {n} | {mean:+.2f} | {sem:.2f} | "
                f"{wins:.0%} | {t:.1f} |"
            )
    if not any_row:
        return ["(no paired rows for this comparison)", ""]
    return lines + [""]


def _load_overall(csv_paths):
    """Load the sibling ``_overall.csv`` for each score CSV, if it exists.

    Not a CLI argument: the un-stratified file always sits beside the score file
    that produced it, and making the caller name both invites the two being
    mismatched -- an overall table paired against a different run's per-depth
    table would be a plausible-looking, wrong report.

    Silently absent is fine and expected: every Phase 3 CSV written before
    2026-08-20 predates the metric.
    """
    rows = []
    for path in csv_paths:
        sibling = path.with_name(path.stem + "_overall.csv")
        if not sibling.is_file():
            continue
        with open(sibling, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    row["si_sdr"] = float(row["si_sdr"])
                except (TypeError, ValueError):
                    continue
                # Added 2026-08-23. Absent in earlier runs, so the key is
                # DROPPED rather than defaulted -- a default would make a run
                # that never measured this indistinguishable from one that
                # measured it as zero.
                for optional in ("si_sdr_pooled", "level_error_db"):
                    if optional not in row:
                        continue
                    try:
                        row[optional] = float(row[optional])
                    except (TypeError, ValueError):
                        del row[optional]
                row["dilate_ms"] = float(row.get("dilate_ms") or 0.0)
                row["source"] = path.name
                rows.append(row)
    return rows


def _overall_table(
    rows: list[dict], left: str, right: str, field: str = "si_sdr_pooled"
) -> list[str]:
    """The un-stratified gap: ``left - right``, paired on
    (source, scene, speaker, system, dilate_ms) -- no depth, because there is
    no depth. Says whether a configuration is NET better; the per-depth tables
    say where the difference lives. §6.4 wants both and forbids only reporting
    this instead of them.

    ``field`` defaults to ``si_sdr_pooled`` -- the depth-pooled exchange rate,
    which fits the scale per depth and is provably bounded by the best and
    worst of them. The literal whole-track ``si_sdr`` is available too, but it
    is scale-anchored by the bit-exact solo copy and so is dominated by LEVEL
    error rather than by the quality tradeoff this table is read for; on
    2026-08-23 it sat below every depth it appeared to summarise in 271 of 288
    rows. Runs written before that carry only ``si_sdr``.
    """
    if not rows:
        return ["(no `_overall.csv` beside these score CSVs -- this metric was",
                "added 2026-08-20, so earlier runs do not have it)", ""]
    if not any(field in r for r in rows):
        return [f"(no `{field}` column -- this run predates it; see the "
                "whole-track table below)", ""]

    def key(r):
        return (r["source"], r["scene"], r["speaker"], r["system"], r["dilate_ms"])

    lhs = {key(r): r[field] for r in rows if r.get("diarization") == left and field in r}
    rhs = {key(r): r[field] for r in rows if r.get("diarization") == right and field in r}

    buckets: dict[str, list[float]] = {}
    for k in lhs:
        if k in rhs:
            a, b = clip_score(lhs[k]), clip_score(rhs[k])
            if a is not None and b is not None:
                buckets.setdefault(k[3], []).append(a - b)

    lines = ["| system | n | mean (dB) | SEM | win rate | \\|t\\| |",
             "|---|---|---|---|---|---|"]
    any_row = False
    for system_name in SYSTEMS:
        diffs = buckets.get(system_name)
        if not diffs:
            continue
        any_row = True
        mean, sem, n, wins, tstat = _stats(diffs)
        lines.append(f"| {system_name} | {n} | {mean:+.2f} | {sem:.2f} | "
                     f"{wins:.0%} | {tstat:.1f} |")
    if not any_row:
        return ["(no paired overall rows for this comparison)", ""]
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="run_phase3.py score CSV(s)")
    parser.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = parser.parse_args()

    rows = load_score_rows(args.csv, required_columns=("m", "diarization"))
    overall_rows = _load_overall(args.csv)
    arms = sorted({r["diarization"] for r in rows if r["diarization"]})
    if "oracle" not in arms:
        raise SystemExit(
            "no 'oracle' rows in these CSVs. Guardrail §6.2: a real-diarization "
            "number is not interpretable without the oracle beside it."
        )

    dilations = sorted({r["dilate_ms"] for r in rows})
    if len(dilations) > 1:
        # A sweep CSV can be aggregated, but only one pipeline at a time: the
        # tables below would otherwise blend several dilation values into one
        # "depth 2" cell. run_phase3.py's own report has the sweep section.
        rows = [r for r in rows if r["dilate_ms"] == 0.0]

    lines = [
        "# Phase 3 -- oracle-vs-real gap", "",
        f"sources: {', '.join(p.name for p in args.csv)}",
        f"arms present: {', '.join(arms)}",
        f"paired rows loaded: {len(rows)}", "",]
    if len(dilations) > 1:
        lines += [
            f"> These CSVs sweep `dilate_overlap_ms` over {dilations} ms. This",
            "> report covers the **0 ms baseline only** -- every other value is a",
            "> different pipeline, and averaging across them would label a mean of",
            "> pipelines as a property of one. See the sweep section in",
            "> `run_phase3.py`'s own `.md` for the dilation comparison.", "",
        ]
    lines += [
        "All differences are PAIRED on matched (scene, speaker, depth, system) rows,",
        "so scene difficulty cancels exactly. SEM is the precision of the mean, not",
        "the spread of the data (CLAUDE.md §7) -- the two differ by ~30x here.", "",
    ]

    for left, right, blurb in COMPARISONS:
        if left not in arms or right not in arms:
            lines += [f"## `{left}` - `{right}`", "", "(arm not present in this run)", ""]
            continue
        lines += [f"## `{left}` - `{right}` -- {blurb}", ""]
        lines += [
            "### overall (un-stratified, whole output track)", "",
            "Is this arm NET better or worse? The per-depth tables below say",
            "*where* the difference lives; this one says whether it adds up.",
            "Never read it INSTEAD of them (§6.4), and never optimize against it:",
            "it is scale-anchored by the bit-exact solo copy.", "",
        ]
        lines += ["#### pooled across depths (the exchange rate)", ""]
        lines += _overall_table(overall_rows, left, right, "si_sdr_pooled")
        lines += ["#### whole output track, one global scale", "",
                  "Level-sensitive by construction -- read it as a level check,",
                  "not as a summary of the per-depth tables.", ""]
        lines += _overall_table(overall_rows, left, right, "si_sdr")
        lines += ["### by overlap depth", ""]
        lines += _table(rows, left, right, "depth")
        lines += [
            "### by accumulation (`n_accepted_before`, deflation systems only)", "",
            "The axis CLAUDE.md §6.4 says to claim on: depth measures intrinsic",
            "difficulty and hits every system equally, which buried the",
            "accumulation effect for five Phase 2 runs.", "",
            f"Levels are the **`{right}`** arm's accumulation position, so each row reads",
            f"\"for a speaker `{right}` placed at level k, what did `{left}` cost it?\".",
            "Rows are paired BEFORE they are bucketed -- filtering first would keep only",
            "the speakers whose position coincides across arms, and under real",
            "diarization the ascending-`V_i` sort makes that disagreement the norm",
            "rather than the exception (it is the effect being measured, not noise).", "",
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
