#!/usr/bin/env python3
"""Render the Phase 2 depth-stratified plot from ``scripts/run_phase2.py``'s CSV.

"The plot that makes the paper" (CLAUDE.md §5 Phase 2): mean SI-SDR vs. overlap
depth |K|, one line per system. The accumulation-free systems (``no_recursion``,
``coarse_to_fine``) should stay flat as depth grows; the deflation systems
should slope downward, ``ungated_deflation`` more steeply than
``gated_deflation``. Requires the ``viz`` extra (``pip install -e .[viz]``).

    python scripts/plot_phase2_depth.py results/phase2_librimix_3spk.csv \\
        --out results/phase2_librimix_3spk_depth.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SYSTEMS = ("no_recursion", "ungated_deflation", "gated_deflation", "coarse_to_fine")


def _load_means(csv_path: Path) -> dict[str, dict[int, float]]:
    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            si_sdr = float(row["si_sdr"])
            if si_sdr != si_sdr or si_sdr in (float("inf"), float("-inf")):  # skip nan/inf
                continue
            system, depth = row["system"], int(row["depth"])
            sums[system][depth] += si_sdr
            counts[system][depth] += 1
    return {
        system: {depth: sums[system][depth] / counts[system][depth] for depth in sums[system]}
        for system in sums
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    means = _load_means(args.csv_path)
    out = args.out or args.csv_path.with_name(args.csv_path.stem + "_depth.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for system in SYSTEMS:
        if system not in means:
            continue
        depths = sorted(means[system])
        ax.plot(depths, [means[system][d] for d in depths], marker="o", label=system)
    ax.set_xlabel("overlap depth |K|")
    ax.set_ylabel("mean SI-SDR (dB)")
    ax.set_title("Phase 2: accumulation-free vs. deflation, by overlap depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
