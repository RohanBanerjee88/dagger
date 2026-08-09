#!/usr/bin/env python3
"""Cross-run Phase 2 aggregation: the accumulation axis.

``scripts/run_phase2.py`` reports one eval set at a time, stratified by overlap
depth |K|. That is the right axis for *intrinsic difficulty* -- more concurrent
voices genuinely is harder, which is why every system, including the ones with
no deflation logic at all, degrades with depth. It is the WRONG axis for the
accumulation claim: ``dagger.reconstruct.deflation`` deflates once per scene
over all ``m`` speakers, so the accumulated residual error a speaker inherits is
set by ``m`` (and its position in the deflation order), not by how many voices
happen to be concurrent in whichever region is later scored. A "depth 2" row in
the 5-speaker eval was produced by a 5-step deflation chain; a "depth 2" row in
the 3-speaker eval by a 3-step chain.

This script concatenates several runs' CSVs on their ``m`` column and holds
depth fixed while sweeping ``m``, which separates the two effects.

The three eval sets are different audio, so a raw cross-set comparison also
picks up whatever difficulty difference exists between them. ``no_recursion``
is the control: it runs no deflation, so any m-dependence in it is set
difficulty, and subtracting it leaves the accumulation-specific part. Both the
raw and the control-corrected numbers are printed -- the correction is a
subtraction the reader should be able to check, not one to take on trust.

    python scripts/aggregate_phase2.py \\
        results/phase2_librimix_3spk_curriculum345full.csv \\
        results/phase2_librimix_4spk_curriculum345full.csv \\
        results/phase2_librimix_5spk_curriculum345full.csv \\
        --out results/phase2_accumulation.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dagger.metrics import phase2_scores

# Reading, clipping and the +-inf rule are shared with scripts/run_phase2.py and
# scripts/plot_phase2_depth.py -- see dagger.metrics.phase2_scores on why these
# are no longer three separate copies.
SYSTEMS = phase2_scores.SYSTEMS
CONTROL_SYSTEM = phase2_scores.CONTROL_SYSTEM
SI_SDR_CAP_DB = phase2_scores.SI_SDR_CAP_DB


def _load(csv_paths: list[Path]) -> list[dict]:
    """Concatenate score CSVs, keeping only scoreable rows.

    Requires the ``m`` column, i.e. a CSV written after the accumulation
    instrumentation landed -- an older 5-column file is rejected loudly rather
    than silently aggregated into a single meaningless m-group.
    """
    return phase2_scores.load_score_rows(csv_paths, required_columns=("m",))


def _mean(rows: list[dict], system: str, m: int, depth: int) -> float | None:
    values = [
        r["si_sdr"] for r in rows
        if r["system"] == system and r["m"] == m and r["depth"] == depth
    ]
    return float(np.mean(values)) if values else None


def _cross_tab(rows: list[dict], ms: list[int], depths: list[int]) -> list[str]:
    """Mean SI-SDR at fixed depth, swept over scene speaker count."""
    lines = [
        "## Mean SI-SDR by scene speaker count `m`, at fixed depth", "",
        "(read across a row: that is the accumulation axis. Reading down a "
        "column is the intrinsic-difficulty axis and is what the per-run "
        "depth tables already show.)", "",
        "| depth | system | " + " | ".join(f"m={m}" for m in ms) + " |",
        "|---|---" + "|---" * len(ms) + "|",
    ]
    for depth in depths:
        for system in SYSTEMS:
            cells = []
            for m in ms:
                value = _mean(rows, system, m, depth)
                cells.append(f"{value:.2f}" if value is not None else "--")
            if set(cells) == {"--"}:
                continue
            lines.append(f"| {depth} | {system} | " + " | ".join(cells) + " |")
    return lines


def _excess_section(rows: list[dict], ms: list[int], depths: list[int]) -> list[str]:
    """Degradation from the smallest to the largest ``m`` at fixed depth, minus
    the control's degradation over the same span.

    What survives the subtraction is the part attributable to deflating more
    speakers rather than to the eval sets differing. The accumulation-free
    systems should land near zero; ungated_deflation should not.
    """
    lines = [
        "", "## Control-corrected degradation with `m` (per fixed depth)", "",
        f"(`raw` = mean at m=max minus mean at m=min. `excess` = raw minus "
        f"`{CONTROL_SYSTEM}`'s raw over the same span -- {CONTROL_SYSTEM} runs no "
        "deflation, so its m-dependence is eval-set difficulty and cancels out. "
        "Near-zero excess = accumulation-free; strongly negative = accumulating.)", "",
        "| depth | m span | system | raw (dB) | excess vs control (dB) |",
        "|---|---|---|---|---|",
    ]
    for depth in depths:
        available = [m for m in ms if _mean(rows, CONTROL_SYSTEM, m, depth) is not None]
        if len(available) < 2:
            continue
        lo, hi = available[0], available[-1]
        control_raw = _mean(rows, CONTROL_SYSTEM, hi, depth) - _mean(rows, CONTROL_SYSTEM, lo, depth)
        for system in SYSTEMS:
            low, high = _mean(rows, system, lo, depth), _mean(rows, system, hi, depth)
            if low is None or high is None:
                continue
            raw = high - low
            excess = raw - control_raw
            label = "control" if system == CONTROL_SYSTEM else f"{excess:+.2f}"
            lines.append(f"| {depth} | {lo}->{hi} | {system} | {raw:+.2f} | {label} |")
    return lines


def _gap_section(rows: list[dict], ms: list[int], depths: list[int]) -> list[str]:
    """The headline accumulation-specific gap, by (depth, m)."""
    lines = [
        "", "## Accumulation-specific gap: coarse_to_fine - ungated_deflation", "",
        "| depth | " + " | ".join(f"m={m}" for m in ms) + " |",
        "|---" + "|---" * len(ms) + "|",
    ]
    for depth in depths:
        cells = []
        for m in ms:
            ctf = _mean(rows, "coarse_to_fine", m, depth)
            ung = _mean(rows, "ungated_deflation", m, depth)
            cells.append(f"{ctf - ung:.2f}" if ctf is not None and ung is not None else "--")
        if set(cells) == {"--"}:
            continue
        lines.append(f"| {depth} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/phase2_accumulation.md"))
    args = parser.parse_args()

    rows = _load(args.csv_paths)
    ms = sorted({r["m"] for r in rows})
    depths = sorted({r["depth"] for r in rows})

    if len(ms) < 2:
        print(
            f"warning: only one scene speaker count present (m={ms[0]}). The "
            "accumulation sweep needs at least two eval sets with different "
            "`m` -- the tables below will have a single column."
        )

    sources = sorted({r["source"] for r in rows})
    lines = [
        "# Phase 2 -- accumulation analysis (cross-run)", "",
        f"sources: {', '.join(sources)}", "",
        f"rows: {len(rows)} scoreable  |  m: {ms}  |  depths: {depths}", "",
    ]
    lines += _cross_tab(rows, ms, depths)
    lines += _excess_section(rows, ms, depths)
    lines += _gap_section(rows, ms, depths)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
