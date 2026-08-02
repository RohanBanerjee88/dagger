#!/usr/bin/env python3
"""Render Phase 2's stratified plot from ``scripts/run_phase2.py``'s CSV(s).

Three x-axes are available, and they answer different questions -- picking the
wrong one is why this plot was hard to read for several runs:

* ``--x-axis depth`` (default) -- mean SI-SDR vs. overlap depth |K|. This is the
  INTRINSIC DIFFICULTY axis: more concurrent voices means more simultaneous
  spectral masking, so *every* system slopes downward here, including
  ``no_recursion``, which runs no deflation at all. Do not read a downward slope
  on this axis as evidence of error accumulation.
* ``--x-axis m`` -- mean SI-SDR vs. the number of speakers in the scene, at a
  fixed depth (``--depth``). This is the ACCUMULATION axis: deflation runs once
  per scene over all ``m`` speakers, so this is what Theorem 3's
  ``L*||E_(m-1)||`` penalty is indexed by. Needs several eval sets, so pass
  every run's CSV. ``scripts/aggregate_phase2.py`` produces the matching tables,
  including the ``no_recursion`` control correction for eval-set difficulty.
* ``--x-axis n_accepted_before`` -- mean SI-SDR vs. how many prior estimates
  were subtracted into the residual before this speaker was extracted. The
  sharpest of the three: it varies WITHIN a scene, so acoustics, enrollment, and
  checkpoint are all held fixed. Only the deflation systems have this column.

The expected shape, then, is flat-vs-sloped on the latter two axes -- the
accumulation-free systems (``no_recursion``, ``coarse_to_fine``) roughly flat,
``ungated_deflation`` sloping down and ``gated_deflation`` between them. Requires
the ``viz`` extra (``pip install -e .[viz]``).

    python scripts/plot_phase2_depth.py results/phase2_librimix_5spk.csv --band

    python scripts/plot_phase2_depth.py results/phase2_librimix_{3,4,5}spk.csv \\
        --x-axis m --depth 2 --out results/phase2_accumulation.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

SYSTEMS = ("no_recursion", "ungated_deflation", "gated_deflation", "coarse_to_fine")

# Matches scripts/run_phase2.py's SI_SDR_CAP_DB: si_sdr() legitimately returns
# +-inf (a perfect / a totally failed estimate -- see dagger.metrics.sisdr.si_sdr),
# not "undefined." Only nan (speaker not active here) is excluded; +-inf are
# clipped to this cap rather than silently dropped, so a handful of perfect
# solo-copy rows can't vanish from the plotted mean.
SI_SDR_CAP_DB = 50.0

AXIS_LABELS = {
    "depth": "overlap depth |K|  (intrinsic difficulty)",
    "m": "speakers in scene m  (accumulation)",
    "n_accepted_before": "prior estimates deflated  (accumulation, within-scene)",
}


def _load_values(
    csv_paths: list[Path], x_field: str, depth_filter: int | None
) -> dict[str, dict[int, list[float]]]:
    """``{system: {x: [si_sdr, ...]}}`` for the requested x-axis.

    Rows whose ``x_field`` is empty are skipped: the accumulation columns are
    deliberately blank for the accumulation-free systems, since 0 is a real
    value there ("nothing deflated before me") and writing it would conflate
    "no accumulation" with "not applicable".
    """
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or x_field not in reader.fieldnames:
                raise SystemExit(
                    f"{path} has no {x_field!r} column -- it predates the accumulation "
                    "instrumentation. Re-run scripts/run_phase2.py to regenerate it."
                )
            for row in reader:
                si_sdr = float(row["si_sdr"])
                if math.isnan(si_sdr):  # speaker not active here, nothing to score
                    continue
                if depth_filter is not None and int(row["depth"]) != depth_filter:
                    continue
                raw_x = row[x_field]
                if raw_x is None or raw_x == "":
                    continue
                si_sdr = max(-SI_SDR_CAP_DB, min(SI_SDR_CAP_DB, si_sdr))
                values[row["system"]][int(raw_x)].append(si_sdr)
    return values


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
        "--band", action="store_true",
        help="shade the 5th-95th percentile range, not just the mean",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    if args.x_axis == "m" and args.depth is None:
        print(
            "warning: --x-axis m without --depth mixes every depth into each "
            "point, so intrinsic difficulty rides on top of the accumulation "
            "signal. Pass --depth to hold it fixed."
        )

    values = _load_values(args.csv_paths, args.x_axis, args.depth)
    if not values:
        raise SystemExit("no scoreable rows matched -- check --depth and the CSV paths")

    default_name = args.csv_paths[0].stem + f"_{args.x_axis}"
    if args.depth is not None:
        default_name += f"_depth{args.depth}"
    out = args.out or args.csv_paths[0].with_name(default_name + ".png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for system in SYSTEMS:
        if system not in values:
            continue
        xs = sorted(values[system])
        means = [float(np.mean(values[system][x])) for x in xs]
        line, = ax.plot(xs, means, marker="o", label=system)
        if args.band:
            lo = [float(np.percentile(values[system][x], 5)) for x in xs]
            hi = [float(np.percentile(values[system][x], 95)) for x in xs]
            ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())

    ax.set_xlabel(AXIS_LABELS[args.x_axis])
    ax.set_ylabel("mean SI-SDR (dB)" + (" with p5-p95 band" if args.band else ""))
    title = "Phase 2: accumulation-free vs. deflation"
    if args.depth is not None:
        title += f" (depth {args.depth})"
    ax.set_title(title)
    ax.set_xticks(sorted({x for system in values for x in values[system]}))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
