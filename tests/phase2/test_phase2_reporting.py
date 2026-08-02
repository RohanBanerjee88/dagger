"""Tests for Phase 2's reporting/aggregation layer (CLAUDE.md Phase 2 "1c").

These cover the sections that turn the instrumented CSVs into evidence:
per-system spread, SI-SDR by accumulation count, paired per-row differences,
gate accept rates, and the cross-run `m` sweep in scripts/aggregate_phase2.py.

All of them are pure functions over lists of dicts, so nothing here needs a
corpus, a GPU, an encoder, or an extractor. The scripts are loaded the same way
tests/phase2/test_train_phase1_curriculum.py loads its script under test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", repo_root / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_phase2 = _load_script("run_phase2")
aggregate_phase2 = _load_script("aggregate_phase2")


def _row(scene, speaker, system, depth, si_sdr, *, m=3, n_accepted_before=None):
    return {
        "scene": scene, "speaker": speaker, "system": system, "m": m,
        "depth": depth, "si_sdr": si_sdr,
        "deflation_index": n_accepted_before, "n_accepted_before": n_accepted_before,
        "refine_rounds": 0,
    }


class TestPairedSection:
    def test_win_rate_counts_rows_not_scenes(self):
        """Three rows where A beats B once by a lot and loses twice by a little:
        the mean is positive while the win rate is 33%. Reporting only the mean
        would call this a broad win -- the Phase 1 failure this table exists to
        surface."""
        rows = [
            _row("sc0", "s0", "coarse_to_fine", 2, 10.0),
            _row("sc1", "s1", "coarse_to_fine", 2, -1.0),
            _row("sc2", "s2", "coarse_to_fine", 2, -1.0),
            _row("sc0", "s0", "ungated_deflation", 2, 0.0),
            _row("sc1", "s1", "ungated_deflation", 2, 0.0),
            _row("sc2", "s2", "ungated_deflation", 2, 0.0),
        ]

        text = "\n".join(
            run_phase2._paired_section(rows, [2], "coarse_to_fine", "ungated_deflation")
        )

        assert "| 2 | 3 |" in text  # depth 2, three paired rows
        assert "33.3%" in text
        assert "2.67" in text  # mean of +10, -1, -1

    def test_nan_rows_drop_out_of_the_pairing(self):
        """A nan on either side means that (scene, speaker, depth) has nothing
        to compare, so it must not be counted in the denominator of the win
        rate."""
        rows = [
            _row("sc0", "s0", "coarse_to_fine", 2, 4.0),
            _row("sc1", "s1", "coarse_to_fine", 2, float("nan")),
            _row("sc0", "s0", "ungated_deflation", 2, 1.0),
            _row("sc1", "s1", "ungated_deflation", 2, 1.0),
        ]

        text = "\n".join(
            run_phase2._paired_section(rows, [2], "coarse_to_fine", "ungated_deflation")
        )

        assert "| 2 | 1 |" in text  # only the one fully-scoreable pair
        assert "100.0%" in text

    def test_infinities_are_clipped_not_dropped(self):
        """si_sdr() returns +inf for a perfect estimate -- informative, not
        undefined. Clipping keeps the row in the pairing; dropping it was a real
        Phase 2 bug."""
        rows = [
            _row("sc0", "s0", "coarse_to_fine", 1, float("inf")),
            _row("sc0", "s0", "ungated_deflation", 1, 0.0),
        ]

        text = "\n".join(
            run_phase2._paired_section(rows, [1], "coarse_to_fine", "ungated_deflation")
        )

        assert "| 1 | 1 |" in text
        assert f"{run_phase2.SI_SDR_CAP_DB:.2f}" in text


class TestAccumulationSection:
    def test_groups_by_accumulation_count_at_fixed_depth(self):
        rows = [
            _row("sc0", "s0", "ungated_deflation", 2, 0.0, n_accepted_before=0),
            _row("sc0", "s1", "ungated_deflation", 2, -4.0, n_accepted_before=1),
            _row("sc0", "s2", "ungated_deflation", 2, -8.0, n_accepted_before=2),
        ]

        lines = run_phase2._accumulation_section(rows, [2])
        text = "\n".join(lines)

        assert "| ungated_deflation | 0 | 0.00 (n=1) |" in text
        assert "| ungated_deflation | 1 | -4.00 (n=1) |" in text
        assert "| ungated_deflation | 2 | -8.00 (n=1) |" in text

    def test_accumulation_free_systems_appear_as_a_reference_row(self):
        """no_recursion/coarse_to_fine have no accumulation axis at all, but
        omitting them would leave the reader without the level to compare the
        deflation rows against."""
        rows = [
            _row("sc0", "s0", "no_recursion", 2, 3.0),
            _row("sc0", "s0", "coarse_to_fine", 2, 2.0),
        ]

        text = "\n".join(run_phase2._accumulation_section(rows, [2]))

        assert "| no_recursion | n/a | 3.00 (n=1) |" in text
        assert "| coarse_to_fine | n/a | 2.00 (n=1) |" in text


class TestGateSection:
    def test_accept_rate_is_per_system_and_round(self):
        gate_rows = [
            {"scene": "sc0", "speaker": "s0", "system": "coarse_to_fine", "round": 0,
             "accepted": True, "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0,
             "reason": "accepted"},
            {"scene": "sc0", "speaker": "s1", "system": "coarse_to_fine", "round": 0,
             "accepted": False, "margin": -1.0, "vad_coverage": 1.0, "artifact_score": 0.0,
             "reason": "margin"},
            {"scene": "sc0", "speaker": "s0", "system": "coarse_to_fine", "round": 1,
             "accepted": True, "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0,
             "reason": "accepted"},
            {"scene": "sc0", "speaker": "s1", "system": "coarse_to_fine", "round": 1,
             "accepted": True, "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0,
             "reason": "accepted"},
        ]

        text = "\n".join(run_phase2._gate_section(gate_rows))

        assert "| coarse_to_fine | 0 | 2 | 1 | 50.0% | 0 | margin=1 |" in text
        assert "| coarse_to_fine | 1 | 2 | 2 | 100.0% | 0 | -- |" in text

    def test_missing_clip_is_not_counted_as_a_rejection(self):
        """"No overlap-only region to re-embed from" is not a gate decision.
        Counting it as a rejection would understate the accept rate and hide a
        rubber-stamping gate."""
        gate_rows = [
            {"scene": "sc0", "speaker": "s0", "system": "coarse_to_fine", "round": 0,
             "accepted": True, "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0,
             "reason": "accepted"},
            {"scene": "sc0", "speaker": "s1", "system": "coarse_to_fine", "round": 0,
             "accepted": None, "margin": None, "vad_coverage": None,
             "artifact_score": None, "reason": "no_overlap_clip"},
        ]

        text = "\n".join(run_phase2._gate_section(gate_rows))

        # One decision, one acceptance, 100% -- and the skipped speaker counted
        # separately in the "no clip" column.
        assert "| coarse_to_fine | 0 | 1 | 1 | 100.0% | 1 | -- |" in text


class TestSpreadSection:
    def test_reports_percentiles_alongside_the_mean(self):
        # 0.0 .. 10.0 in 0.1 steps, all comfortably inside the +-50 dB clip.
        rows = [_row("sc0", f"s{i}", "no_recursion", 2, i / 10.0) for i in range(101)]

        text = "\n".join(run_phase2._spread_section(rows, [2]))

        # mean 5, p5 0.5, p95 9.5, min 0, max 10.
        assert "| no_recursion | 2 | 101 | 5.00 | 0.50 | 9.50 | 0.00 | 10.00 |" in text

    def test_values_beyond_the_clip_saturate_rather_than_vanish(self):
        """A perfect (+inf) row is clipped to the cap, so both the mean and the
        upper percentiles saturate there. That saturation is expected at depth 1
        and is called out in the section's own preamble."""
        rows = [_row("sc0", f"s{i}", "no_recursion", 1, float("inf")) for i in range(10)]

        text = "\n".join(run_phase2._spread_section(rows, [1]))

        cap = run_phase2.SI_SDR_CAP_DB
        assert f"| no_recursion | 1 | 10 | {cap:.2f} | {cap:.2f} | {cap:.2f} | {cap:.2f} | {cap:.2f} |" in text


class TestAggregateLoad:
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
        import csv as csv_module

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_rejects_a_csv_written_before_the_m_column_existed(self, tmp_path):
        """Silently aggregating an old CSV would collapse every run into one
        meaningless m-group -- exactly the confound this script exists to
        remove."""
        path = self._write_csv(
            tmp_path / "old.csv",
            ["scene", "speaker", "system", "depth", "si_sdr"],
            [{"scene": "sc0", "speaker": "s0", "system": "no_recursion",
              "depth": 2, "si_sdr": 1.0}],
        )

        with pytest.raises(SystemExit, match="no 'm' column"):
            aggregate_phase2._load([path])

    def test_concatenates_runs_and_keeps_m_distinct(self, tmp_path):
        fields = run_phase2.SCORE_FIELDS
        paths = [
            self._write_csv(
                tmp_path / f"{m}spk.csv", fields,
                [_row("sc0", "s0", "no_recursion", 2, float(m), m=m)],
            )
            for m in (3, 4, 5)
        ]

        rows = aggregate_phase2._load(paths)

        assert sorted({r["m"] for r in rows}) == [3, 4, 5]
        assert aggregate_phase2._mean(rows, "no_recursion", 4, 2) == 4.0

    def test_drops_nan_but_clips_infinity(self, tmp_path):
        fields = run_phase2.SCORE_FIELDS
        path = self._write_csv(
            tmp_path / "3spk.csv", fields,
            [
                _row("sc0", "s0", "no_recursion", 2, float("nan")),
                _row("sc1", "s1", "no_recursion", 2, float("inf")),
            ],
        )

        rows = aggregate_phase2._load([path])

        assert len(rows) == 1
        assert rows[0]["si_sdr"] == aggregate_phase2.SI_SDR_CAP_DB


class TestExcessDegradation:
    def test_control_subtraction_isolates_accumulation(self):
        """The eval sets are different audio, so part of any m-dependence is
        set difficulty. Here every system loses 2 dB from m=3 to m=5 for that
        reason, and ungated_deflation loses a further 3 dB. Only the 3 dB is
        accumulation, and only that should survive the correction."""
        rows = []
        for m, difficulty in ((3, 0.0), (5, -2.0)):
            rows.append({"source": "f", "system": "no_recursion", "m": m, "depth": 2,
                         "si_sdr": 0.0 + difficulty})
            rows.append({"source": "f", "system": "coarse_to_fine", "m": m, "depth": 2,
                         "si_sdr": 0.0 + difficulty})
            rows.append({"source": "f", "system": "gated_deflation", "m": m, "depth": 2,
                         "si_sdr": 0.0 + difficulty - (1.0 if m == 5 else 0.0)})
            rows.append({"source": "f", "system": "ungated_deflation", "m": m, "depth": 2,
                         "si_sdr": 0.0 + difficulty - (3.0 if m == 5 else 0.0)})

        text = "\n".join(aggregate_phase2._excess_section(rows, [3, 5], [2]))

        assert "| 2 | 3->5 | no_recursion | -2.00 | control |" in text
        assert "| 2 | 3->5 | coarse_to_fine | -2.00 | +0.00 |" in text
        assert "| 2 | 3->5 | gated_deflation | -3.00 | -1.00 |" in text
        assert "| 2 | 3->5 | ungated_deflation | -5.00 | -3.00 |" in text
