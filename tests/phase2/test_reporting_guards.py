"""Guards for the three reporting defects Phase 2 actually shipped.

Every one of them survived a real run and was caught by a human re-deriving the
numbers by hand, not by anything failing. That is the point of this file: each
test below encodes one defect as a fixture that would have gone green before the
fix and goes red if the behaviour regresses.

The defects, in the order they were found:

1. **2026-07-26 -- ``+-inf`` silently dropped.** ``np.isfinite()`` filtered out
   ``nan`` (correct) *and* ``+-inf`` (wrong -- a perfect or totally failed
   estimate is informative). 44% of depth-1 rows were exactly ``+inf``, so the
   reported depth-1 mean was understated by 7-8 dB.
2. **2026-08-09 -- an ``n=3`` cell drawn as the headline trend.** The primary
   figure's most dramatic feature was ``gated_deflation``'s level-4 point, a mean
   over three rows with a 2.68 dB SEM, plotted identically to n=150 points.
3. **2026-08-09 -- a stale precondition.** The cross-eval-set ``m`` sweep was
   only legible because the ``no_recursion`` control happened to be flat
   (+0.16 dB) when the figure was designed. On the next checkpoint it sloped
   -0.95 dB and the figure quietly became four near-parallel declining lines.

Pure functions over small fixtures -- no corpus, GPU, encoder or extractor.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from dagger.metrics.phase2_scores import (
    SI_SDR_CAP_DB,
    clip_score,
    control_slope,
    group_values,
    load_score_rows,
    mean_sem,
    paired_differences,
    terminal_x_values,
)

SCORE_FIELDS = [
    "scene", "speaker", "system", "m", "depth", "si_sdr",
    "deflation_index", "n_accepted_before", "refine_rounds",
]


def _row(scene, speaker, system, si_sdr, *, m=3, depth=2, n_accepted_before=None):
    return {
        "scene": scene, "speaker": speaker, "system": system, "m": m,
        "depth": depth, "si_sdr": si_sdr, "deflation_index": n_accepted_before,
        "n_accepted_before": n_accepted_before, "refine_rounds": 0,
    }


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: ("" if v is None else v) for k, v in row.items()
            })
    return path


class TestInfinityIsClippedNotDropped:
    """Defect 1: a perfect row must saturate the mean, never vanish from it."""

    def test_plus_inf_clips_to_the_cap(self):
        assert clip_score(float("inf")) == SI_SDR_CAP_DB
        assert clip_score(float("-inf")) == -SI_SDR_CAP_DB

    def test_nan_alone_is_excluded(self):
        assert clip_score(float("nan")) is None

    def test_loader_keeps_inf_and_drops_nan(self, tmp_path):
        path = _write_csv(tmp_path / "scores.csv", [
            _row("sc0", "s0", "no_recursion", float("inf")),
            _row("sc1", "s0", "no_recursion", float("nan")),
            _row("sc2", "s0", "no_recursion", 0.0),
        ])

        rows = load_score_rows([path])

        assert len(rows) == 2, "nan dropped, +inf kept"
        # The buggy np.isfinite() filter would have left only the 0.0 row and
        # reported a mean of 0.00 instead of 25.00.
        mean, _, n = mean_sem([r["si_sdr"] for r in rows])
        assert (mean, n) == (SI_SDR_CAP_DB / 2, 2)


class TestThinCellsAreSeparable:
    """Defect 2: an n=3 mean must be distinguishable from an n=150 one."""

    def test_mean_sem_reports_n_so_thin_cells_can_be_caught(self):
        mean, sem, n = mean_sem([-9.0, -8.0, -10.0])
        assert n == 3
        assert mean == pytest.approx(-9.0)
        assert sem == pytest.approx(0.5774, abs=1e-3)

    def test_sem_is_undefined_for_a_single_sample(self):
        _, sem, n = mean_sem([-9.0])
        assert n == 1 and math.isnan(sem)

    def test_sem_shrinks_with_n_while_spread_does_not(self):
        """The distinction CLAUDE.md §7 turns on: same spread, 100x the samples,
        ~10x tighter SEM. Plotting one and labelling it the other is the trap."""
        small = mean_sem([-1.0, 1.0] * 5)
        large = mean_sem([-1.0, 1.0] * 500)
        assert large[1] < small[1] / 9

    def test_thin_and_thick_cells_are_visible_in_the_grouping(self, tmp_path):
        """The real shape of the bug: one level with 150 samples, the next with 3."""
        rows = (
            [_row(f"sc{i}", "s0", "gated_deflation", -5.0, m=5, depth=5,
                  n_accepted_before=0) for i in range(150)]
            + [_row(f"sc{i}", "s1", "gated_deflation", -8.89, m=5, depth=5,
                    n_accepted_before=4) for i in range(3)]
        )
        path = _write_csv(tmp_path / "scores.csv", rows)

        grouped = group_values(load_score_rows([path]), "n_accepted_before", depth=5)
        counts = {x: mean_sem(v)[2] for x, v in grouped["gated_deflation"].items()}

        assert counts == {0: 150, 4: 3}
        thin = [x for x, n in counts.items() if n < 25]
        assert thin == [4], "the n=3 level must be identifiable as thin"


class TestControlFlatnessIsChecked:
    """Defect 3: the precondition must be computed, not remembered."""

    def _sweep(self, tmp_path, control_by_m):
        rows = []
        for m, value in control_by_m.items():
            for i in range(30):
                rows.append(_row(f"sc{m}_{i}", "s0", "no_recursion", value, m=m, depth=2))
        return load_score_rows([_write_csv(tmp_path / "sweep.csv", rows)])

    def test_flat_control_reports_a_small_slope(self, tmp_path):
        rows = self._sweep(tmp_path, {3: 1.60, 4: 1.60, 5: 1.76})

        drift, lo, hi = control_slope(rows, "m", depth=2)

        assert (lo, hi) == (3, 5)
        assert drift == pytest.approx(0.16, abs=1e-6), "the 2026-08-04 value"

    def test_sloped_control_is_detected(self, tmp_path):
        """The 2026-08-09 checkpoint. Nothing flagged this at the time."""
        rows = self._sweep(tmp_path, {3: 1.60, 4: 1.33, 5: 0.65})

        drift, _, _ = control_slope(rows, "m", depth=2)

        assert drift == pytest.approx(-0.95, abs=1e-6)
        assert abs(drift) > 0.3, "must exceed the tolerance the plot warns on"

    def test_no_slope_when_the_control_spans_one_point(self, tmp_path):
        assert control_slope(self._sweep(tmp_path, {3: 1.6}), "m", depth=2) is None


class TestPairingRemovesTheNeedForAFlatControl:
    def test_paired_difference_cancels_a_per_scene_offset_exactly(self):
        """Two scenes of wildly different difficulty; the system is a constant
        -2 dB behind the control in both. The paired mean recovers exactly -2,
        where subtracting two grand means would only approximate it."""
        rows = [
            _row("easy", "s0", "no_recursion", 20.0),
            _row("easy", "s0", "ungated_deflation", 18.0),
            _row("hard", "s0", "no_recursion", -30.0),
            _row("hard", "s0", "ungated_deflation", -32.0),
        ]
        for row in rows:
            row["source"] = "fixture.csv"

        diffs = paired_differences(rows, "ungated_deflation")

        assert diffs == [-2.0, -2.0]

    def test_unmatched_rows_are_dropped_rather_than_compared(self):
        rows = [
            _row("sc0", "s0", "no_recursion", 1.0),
            _row("sc0", "s0", "ungated_deflation", 0.0),
            _row("sc1", "s0", "ungated_deflation", -100.0),  # no control partner
        ]
        for row in rows:
            row["source"] = "fixture.csv"

        assert paired_differences(rows, "ungated_deflation") == [-1.0]


class TestTerminalStepIsMarked:
    """The one-and-rest endpoint is a special case, not a trend point."""

    def test_last_level_of_the_chain_is_terminal(self, tmp_path):
        rows = [
            _row(f"sc{i}", "s0", "ungated_deflation", -5.0, m=5, depth=5,
                 n_accepted_before=level)
            for level in range(5) for i in range(5)
        ]
        path = _write_csv(tmp_path / "scores.csv", rows)

        terminal = terminal_x_values(load_score_rows([path]), "n_accepted_before")

        assert terminal["ungated_deflation"] == {4}, "only x == m-1"

    def test_mixed_buckets_are_not_marked_terminal(self, tmp_path):
        """Level 3 is terminal for m=4 scenes but mid-chain for m=5 ones, so it
        must stay an ordinary point rather than being dashed on a guess."""
        rows = [
            _row("a", "s0", "ungated_deflation", -5.0, m=4, depth=4, n_accepted_before=3),
            _row("b", "s0", "ungated_deflation", -5.0, m=5, depth=5, n_accepted_before=3),
        ]
        path = _write_csv(tmp_path / "scores.csv", rows)

        terminal = terminal_x_values(load_score_rows([path]), "n_accepted_before")

        assert terminal["ungated_deflation"] == set()

    def test_other_axes_have_no_terminal_concept(self, tmp_path):
        path = _write_csv(tmp_path / "scores.csv", [_row("a", "s0", "no_recursion", 1.0)])
        assert terminal_x_values(load_score_rows([path]), "m") == {}


class TestStaleCsvsAreRejected:
    def test_missing_accumulation_column_exits_loudly(self, tmp_path):
        path = tmp_path / "old.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["scene", "speaker", "system", "depth", "si_sdr"])
            writer.writeheader()
            writer.writerow({"scene": "sc0", "speaker": "s0", "system": "no_recursion",
                             "depth": 2, "si_sdr": 1.0})

        with pytest.raises(SystemExit, match="predates the accumulation instrumentation"):
            load_score_rows([path], required_columns=("m",))
