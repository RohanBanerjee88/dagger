#!/usr/bin/env python3
"""Render Phase 2's stratified plot from ``scripts/run_phase2.py``'s CSV(s).

Three x-axes are available, and they answer different questions -- picking the
wrong one is why this plot was hard to read for several runs:

* ``--x-axis depth`` (default) -- mean SI-SDR vs. overlap depth |K|. This is the
  INTRINSIC DIFFICULTY axis: more concurrent voices means more simultaneous
  spectral masking, so *every* system slopes downward here, including
  ``no_recursion``, which runs no deflation at all. Do not read a downward slope
  on this axis as evidence of error accumulation.
* ``--x-axis m`` -- SI-SDR vs. the number of speakers in the scene, at a fixed
  depth (``--depth``). This is the ACCUMULATION axis: deflation runs once per
  scene over all ``m`` speakers, so this is what Theorem 3's ``L*||E_(m-1)||``
  penalty is indexed by. Needs several eval sets, so pass every run's CSV.
  Defaults to ``--mode paired-vs-control`` (see below).
* ``--x-axis n_accepted_before`` -- SI-SDR vs. how many prior estimates were
  subtracted into the residual before this speaker was extracted. The sharpest
  of the three: it varies WITHIN a scene, so acoustics, enrollment, and
  checkpoint are all held fixed. Only the deflation systems have this column.

The expected shape is flat-vs-sloped on the latter two axes -- the
accumulation-free systems (``no_recursion``, ``coarse_to_fine``) roughly flat,
``ungated_deflation`` sloping down and ``gated_deflation`` between them.

**Reporting defaults are safe rather than opt-in.** Every figure defect this
project shipped had the same shape: the ``.md`` tables were right and the figure
was wrong, because the protection existed as a flag nobody remembered to pass.
So error bars are ON, thin-``n`` points are excluded from the trend ON, and the
control-flatness precondition is checked ON. Flags below change *thresholds*;
they do not switch correctness off.

Requires the ``viz`` extra (``pip install -e .[viz]``).

    python scripts/plot_phase2_depth.py results/phase2_librimix_5spk.csv

    python scripts/plot_phase2_depth.py results/phase2_librimix_{3,4,5}spk.csv \\
        --x-axis m --depth 2 --out results/phase2_accumulation.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dagger.metrics.phase2_scores import (
    CONTROL_SYSTEM,
    SYSTEMS,
    control_slope,
    group_values,
    load_score_rows,
    mean_sem,
    paired_differences,
    terminal_x_values,
)

AXIS_LABELS = {
    "depth": "overlap depth |K|  (intrinsic difficulty)",
    "m": "speakers in scene m  (accumulation)",
    "n_accepted_before": "prior estimates deflated  (accumulation, within-scene)",
}

#: Categorical hues in fixed order, one per system, never cycled. Validated
#: colorblind-safe against a light surface (worst adjacent pair dE 9.1 protan).
#: Two of them sit under 3:1 contrast, which obliges visible labels -- every
#: series is direct-labeled below, which is the relief.
SERIES_COLORS = {
    "no_recursion": "#2a78d6",
    "ungated_deflation": "#eb6834",
    "gated_deflation": "#1baf7a",
    "coarse_to_fine": "#eda100",
}
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3e0"

#: A control this un-flat means a raw cross-eval-set sweep is reading eval-set
#: difficulty as much as accumulation. Chosen against the two observed values:
#: +0.16 dB (2026-08-04, the figure was honest) and -0.95 dB (2026-08-09, it was
#: not, and nothing caught it because the check lived in someone's head).
CONTROL_FLATNESS_TOLERANCE_DB = 0.3


def _series_points(values_by_x: dict[int, list[float]], min_n: int):
    """Split one series into ``(kept, thin)`` -- ``(x, mean, sem, n)`` tuples.

    ``thin`` points are real measurements with too few samples to carry a trend.
    They are returned rather than dropped so the caller can still show them, and
    show *why* they are set apart: Phase 2's first primary figure drew an ``n=3``
    mean as its most dramatic feature, with nothing on the canvas saying so.
    """
    kept, thin = [], []
    for x in sorted(values_by_x):
        mean, sem, n = mean_sem(values_by_x[x])
        (kept if n >= min_n else thin).append((x, mean, sem, n))
    return kept, thin


def _paired_by_x(rows: list[dict], system: str, x_field: str, depth: int | None):
    """``{x: [paired difference vs. the control, ...]}`` for one system."""
    control_rows = [r for r in rows if r["system"] == CONTROL_SYSTEM]
    system_rows = [r for r in rows if r["system"] == system]
    if depth is not None:
        control_rows = [r for r in control_rows if r["depth"] == depth]
        system_rows = [r for r in system_rows if r["depth"] == depth]

    by_x: dict[int, list[float]] = {}
    for x in sorted({r[x_field] for r in system_rows if r.get(x_field) is not None}):
        subset = [r for r in control_rows + system_rows if r.get(x_field) == x]
        diffs = paired_differences(subset, system, CONTROL_SYSTEM)
        if diffs:
            by_x[int(x)] = diffs
    return by_x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--x-axis", choices=sorted(AXIS_LABELS), default="depth",
        help="which stratification to plot; see the module docstring on why this matters",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="hold overlap depth fixed (recommended with --x-axis m, so the "
             "accumulation axis isn't confounded by intrinsic difficulty)",
    )
    parser.add_argument(
        "--band", choices=("sem", "p5p95", "none"), default="sem",
        help="uncertainty shown per point. 'sem' (default) = +-1 standard error "
             "of the MEAN, i.e. how precisely the average is known. 'p5p95' = the "
             "middle 90%% of individual scores, i.e. how much scenes vary. They "
             "differ by ~30x on this data and answer different questions -- "
             "whichever is drawn is named in the axis label and legend.",
    )
    parser.add_argument(
        "--min-n", type=int, default=25,
        help="points with fewer samples are drawn hollow and annotated with their "
             "n, but excluded from the connecting trend line (default: 25)",
    )
    parser.add_argument(
        "--mode", choices=("auto", "raw", "paired-vs-control"), default="auto",
        help="'raw' plots each system's own mean. 'paired-vs-control' plots the "
             "per-row difference against no_recursion, matched on "
             "(scene, speaker, depth), which cancels eval-set difficulty exactly "
             "and puts the control at 0 by construction. 'auto' (default) picks "
             "paired for --x-axis m, raw otherwise.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    x_field = args.x_axis
    mode = args.mode
    if mode == "auto":
        mode = "paired-vs-control" if x_field == "m" else "raw"
    if mode == "paired-vs-control" and x_field == "n_accepted_before":
        raise SystemExit(
            "--mode paired-vs-control needs a column the control also has, and "
            "no_recursion never deflates, so it has no 'n_accepted_before'. Use "
            "--mode raw for this axis."
        )

    if x_field == "m" and args.depth is None:
        print(
            "warning: --x-axis m without --depth mixes every depth into each "
            "point, so intrinsic difficulty rides on top of the accumulation "
            "signal. Pass --depth to hold it fixed."
        )

    rows = load_score_rows(args.csv_paths, required_columns=("m", "depth", x_field))

    # The precondition that silently went stale between two runs. Only meaningful
    # on the cross-eval-set axis, and only when raw means are being compared --
    # pairing removes the need for it entirely.
    if mode == "raw" and x_field == "m":
        slope = control_slope(rows, x_field, depth=args.depth)
        if slope is not None and abs(slope[0]) > CONTROL_FLATNESS_TOLERANCE_DB:
            drift, lo, hi = slope
            print(
                f"WARNING: the {CONTROL_SYSTEM} control moves {drift:+.2f} dB from "
                f"m={lo} to m={hi} (tolerance +-{CONTROL_FLATNESS_TOLERANCE_DB}). It "
                "runs no deflation, so that is eval-set difficulty, and a raw sweep "
                "here shows difficulty and accumulation summed. Re-run with "
                "--mode paired-vs-control, which cancels it exactly."
            )

    if mode == "paired-vs-control":
        series = {
            system: _paired_by_x(rows, system, x_field, args.depth)
            for system in SYSTEMS if system != CONTROL_SYSTEM
        }
    else:
        series = group_values(rows, x_field, depth=args.depth)
    series = {system: by_x for system, by_x in series.items() if by_x}
    if not series:
        raise SystemExit("no scoreable rows matched -- check --depth and the CSV paths")

    terminal = terminal_x_values(rows, x_field)

    default_name = args.csv_paths[0].stem + f"_{x_field}"
    if args.depth is not None:
        default_name += f"_depth{args.depth}"
    if mode == "paired-vs-control":
        default_name += "_vs_control"
    out = args.out or args.csv_paths[0].with_name(default_name + ".png")

    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    thin_notes: list[str] = []

    for system in SYSTEMS:
        if system not in series:
            continue
        color = SERIES_COLORS[system]
        kept, thin = _series_points(series[system], args.min_n)
        if not kept:
            thin_notes.append(f"{system}: every point below n={args.min_n}, not drawn")
            continue

        xs = [p[0] for p in kept]
        means = [p[1] for p in kept]
        yerr = [p[2] for p in kept] if args.band == "sem" else None
        ax.errorbar(
            xs, means, yerr=yerr, color=color, ls="none", marker="o", ms=6,
            capsize=3, elinewidth=1, label=system, zorder=3,
        )

        # The terminal deflation step is the one-and-rest endpoint, not a trend
        # point (see terminal_x_values). Drawn dashed so a flattening there does
        # not read as the accumulation claim failing.
        is_terminal = xs[-1] in terminal.get(system, set()) and len(xs) >= 2
        body_x, body_y = (xs[:-1], means[:-1]) if is_terminal else (xs, means)
        ax.plot(body_x, body_y, color=color, lw=2, zorder=3)
        if is_terminal:
            ax.plot(xs[-2:], means[-2:], color=color, lw=2, ls=(0, (2.5, 2)), zorder=3)

        if args.band == "p5p95":
            lo = [float(np.percentile(series[system][x], 5)) for x in xs]
            hi = [float(np.percentile(series[system][x], 95)) for x in xs]
            ax.fill_between(xs, lo, hi, alpha=0.15, color=color, zorder=1)

        ax.annotate(
            system, (xs[-1], means[-1]), textcoords="offset points",
            xytext=(8, 0), fontsize=8.5, color=color, va="center",
        )

        for x, mean, _, n in thin:
            ax.plot([x], [mean], marker="o", ms=5, mfc="none", mec=color,
                    mew=1.2, ls="none", zorder=3)
            ax.annotate(f"n={n}", (x, mean), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=MUTED)
            thin_notes.append(f"{system} at {x}: mean {mean:.2f} dB from n={n}")

    if mode == "paired-vs-control":
        ax.axhline(0, color=MUTED, lw=1.1, ls="--", zorder=2)
        ax.annotate(
            f"{CONTROL_SYSTEM} (control, 0 by construction)",
            (min(min(s) for s in series.values()), 0),
            textcoords="offset points", xytext=(2, -11), fontsize=8,
            color=MUTED, va="top",
        )

    # The band is named in a footnote rather than the axis label: SEM and a
    # p5-p95 band differ by ~30x here, so which one is drawn has to be stated,
    # but stating it in the ylabel makes the label long enough to clip.
    band_note = {
        "sem": "Error bars: +-1 SEM (precision of the mean, sd/sqrt(n)) -- NOT the "
               "spread of individual scenes.",
        "p5p95": "Shaded band: p5-p95 of INDIVIDUAL scores (spread) -- not the "
                 "precision of the mean, which is ~30x tighter.",
        "none": "No uncertainty shown.",
    }[args.band]
    footnote = f"{band_note}  Hollow markers: n < {args.min_n}, shown but excluded "
    footnote += "from the trend line."
    if any(terminal.get(s) for s in series):
        footnote += ("\nDashed final segment: the terminal one-and-rest step, where the "
                     "residual approximates the speaker's own source -- not a trend point.")

    ax.set_xlabel(AXIS_LABELS[x_field])
    ax.set_ylabel(
        f"paired SI-SDR vs. {CONTROL_SYSTEM} (dB)"
        if mode == "paired-vs-control" else "mean SI-SDR (dB)"
    )
    title = "Phase 2: accumulation-free vs. deflation"
    if args.depth is not None:
        title += f" (depth {args.depth})"
    ax.set_title(title, loc="left")

    ticks = sorted({x for by_x in series.values() for x in by_x})
    ax.set_xticks(ticks)
    # Right-hand headroom for the direct labels, which sit outside the last point.
    span = (ticks[-1] - ticks[0]) or 1
    ax.set_xlim(ticks[0] - 0.15 * span, ticks[-1] + 0.45 * span)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=min(4, len(series)), fontsize=8,
    )
    fig.text(0.005, -0.14, footnote, fontsize=7.3, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.28)

    print(f"wrote {out}")
    print(f"  mode={mode}  band={args.band}  min_n={args.min_n}")
    for note in thin_notes:
        print(f"  thin: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
