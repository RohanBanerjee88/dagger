"""The un-stratified, whole-output SI-SDR (CLAUDE.md §7).

Stage B's dilation sweep produced +2.19 dB at depth 2 against -29.85 dB at
depth 1 and left "is dilating net better?" unanswerable, because a gain and its
cost landed in different rows with no exchange rate between them. This metric is
the exchange rate.

The tests below pin the two things that make it safe rather than dangerous:

1. **It lives at its own grain, in its own file.** One row per
   (scene, speaker, system) -- not per depth. Every per-depth table in this
   project groups by ``depth``, and a whole-output row carrying a depth value
   would be silently absorbed as "another depth". That exact class of defect has
   shipped three times here (the ``+-inf`` drop, the ``dilate_ms`` sweep, the
   ceiling's objective), so the separation is enforced structurally.

2. **It genuinely reflects more than one depth**, rather than duplicating one
   of them -- otherwise it would answer nothing the per-depth rows do not
   already answer.

CORRECTION (2026-08-23): this file used to claim the whole-track number "pools
the depths". It does not, and reading it that way is what let a defect ship. It
fits ONE scale over the whole track, so the bit-exact solo copy pins that scale
near 1 and a pure LEVEL error in the overlap region is charged at full price --
while every per-depth row fits its own scale and discounts the same error. On
the verification run the whole-track number therefore landed BELOW every depth
it appeared to summarise in 271 of 288 rows. The pooled exchange rate that
genuinely is bounded by its constituents is ``si_sdr_pooled``; see
``tests/phase3/test_pooled_depth_metric.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.oracle import OracleDiarizer  # noqa: E402
from dagger.diarize.regions import scene_regions  # noqa: E402
from dagger.enroll.encoder import SpeakerEncoder  # noqa: E402
from dagger.eval.systems import (  # noqa: E402
    OVERALL_FIELDS, SCORE_FIELDS, score_scene,
)
from dagger.extract.base import Extractor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_run_phase3():
    spec = importlib.util.spec_from_file_location(
        "run_phase3_overall", ROOT / "scripts" / "run_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_phase3_overall"] = module
    spec.loader.exec_module(module)
    return module


run_phase3 = _load_run_phase3()


class _FakeEncoder(SpeakerEncoder):
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float64)
        if w.size == 0:
            return np.zeros(3)
        zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0)) if w.size > 1 else 0.0
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), zcr])


class _DeterministicExtractor(Extractor):
    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        return x_O * (0.5 + 0.25 * float(np.tanh(np.sum(embedding))))


GATE_CFG = {
    "tau_margin": 0.1, "max_mean_variance": 0.05,
    "min_vad_coverage": 0.5, "max_artifact_score": 0.9,
}


def _score(scene, diarizer=None, refine_rounds=0):
    regions = scene_regions(scene, diarizer or OracleDiarizer())
    return score_scene(
        scene, 0, 3, 100.0, None, _FakeEncoder(), _DeterministicExtractor(),
        GATE_CFG, refine_rounds, regions=regions, on_unenrollable="drop",
    )


class TestGrain:
    def test_one_row_per_scene_speaker_system(self, three_speaker_scheduled_scene):
        rows, _, overall = _score(three_speaker_scheduled_scene)
        keys = [(r["scene"], r["speaker"], r["system"]) for r in overall]
        assert len(keys) == len(set(keys)), "overall rows are not unique per grain"

        # And the per-depth rows genuinely span more than one depth, so the two
        # grains are actually different here rather than coincidentally equal.
        assert len({r["depth"] for r in rows}) > 1
        assert len(overall) < len(rows)

    def test_it_carries_no_depth_column(self, three_speaker_scheduled_scene):
        """The structural guard. A `depth` key would let any existing
        depth-stratified table absorb these rows as an extra depth."""
        _, _, overall = _score(three_speaker_scheduled_scene)
        assert overall
        for row in overall:
            assert "depth" not in row
        assert "depth" not in OVERALL_FIELDS
        assert "depth" in SCORE_FIELDS

    def test_every_scored_speaker_gets_an_overall_row(self, three_speaker_scheduled_scene):
        rows, _, overall = _score(three_speaker_scheduled_scene)
        per_depth = {(r["scene"], r["speaker"], r["system"]) for r in rows}
        assert {(r["scene"], r["speaker"], r["system"]) for r in overall} == per_depth


class TestItPoolsRatherThanDuplicates:
    def test_it_pools_a_perfect_region_with_a_bad_one(
        self, three_speaker_scheduled_scene
    ):
        """The case the per-depth tables cannot express in one number.

        In this fixture depth 1 is a bit-exact solo copy (``+inf``) and depth 3
        is extracted (finite, ~-3 dB). A pooled score must land strictly
        between: better than the overlap alone, because most of the track is
        perfect -- but still finite, because the overlap is not. Measured here:
        overall ~-0.65 dB against a per-depth pair of {+inf, -3.01}.

        A metric that merely echoed one depth would sit on -3.01 or on +inf.
        """
        rows, _, overall = _score(three_speaker_scheduled_scene)
        by_key: dict[tuple, list[float]] = {}
        for r in rows:
            by_key.setdefault(
                (r["scene"], r["speaker"], r["system"]), []
            ).append(float(r["si_sdr"]))

        checked = 0
        for row in overall:
            depths = by_key[(row["scene"], row["speaker"], row["system"])]
            finite = [d for d in depths if math.isfinite(d)]
            if not (finite and any(d == math.inf for d in depths)):
                continue
            value = float(row["si_sdr"])
            assert math.isfinite(value), (
                "a perfect solo region made the whole track score +inf -- the "
                "overlap error is being lost"
            )
            assert value > max(finite), (
                f"pooled {value:.3f} is no better than the worst depth "
                f"{max(finite):.3f} -- the perfect region is not counted"
            )
            checked += 1

        assert checked > 0, "fixture produced no perfect-plus-imperfect speaker"

    def test_a_uniformly_worse_estimate_scores_worse_overall(
        self, three_speaker_scheduled_scene
    ):
        """Sanity on direction: real diarization should not score better."""
        _, _, clean = _score(three_speaker_scheduled_scene)
        _, _, noisy = _score(three_speaker_scheduled_scene, FakeDiarizer(jitter=0.25))

        def keyed(rowset):
            return {
                (r["speaker"], r["system"]): float(r["si_sdr"])
                for r in rowset if math.isfinite(float(r["si_sdr"]))
            }

        a, b = keyed(clean), keyed(noisy)
        shared = set(a) & set(b)
        assert shared, "no comparable overall rows across the two region sets"
        assert sum(b[k] < a[k] for k in shared) > len(shared) // 2


class TestItReachesTheReport:
    def test_a_separate_overall_csv_is_written(self, tmp_path):
        def row(system, si_sdr, dilate_ms=0.0):
            return {
                "diarization": "oracle", "dilate_ms": dilate_ms, "scene": "s",
                "speaker": "s1", "system": system, "m": 3, "si_sdr": si_sdr,
                "deflation_index": None, "n_accepted_before": None,
                "refine_rounds": 0, "cluster": "s1", "n_clusters": 3,
            }

        overall = [row("no_recursion", 4.0), row("no_recursion", 9.0, 200.0)]
        run_phase3._write_results(
            [], [], [], overall, tmp_path, "stem", ["oracle"], n_scenes=1,
        )

        written = (tmp_path / "stem_overall.csv").read_text()
        assert "si_sdr" in written and "4.0" in written
        assert "depth" not in written.splitlines()[0], "overall CSV grew a depth column"

        # The sweep comparison the metric exists for must be legible in the .md.
        text = (tmp_path / "stem.md").read_text()
        assert "Overall SI-SDR" in text
        assert "4.00" in text and "9.00" in text, (
            "sweep points are not compared side by side -- this is the ONLY table "
            "where they can be, since every other one is baseline-only"
        )

    def test_the_md_warns_against_optimizing_it(self, tmp_path):
        """The caveat is load-bearing, not decoration: optimizing a
        scale-anchored whole-output number is what voided the Stage B ceiling."""
        overall = [{
            "diarization": "oracle", "dilate_ms": 0.0, "scene": "s", "speaker": "s1",
            "system": "no_recursion", "m": 3, "si_sdr": 1.0, "deflation_index": None,
            "n_accepted_before": None, "refine_rounds": 0, "cluster": "s1",
            "n_clusters": 3,
        }]
        run_phase3._write_results(
            [], [], [], overall, tmp_path, "stem", ["oracle"], n_scenes=1,
        )
        text = (tmp_path / "stem.md").read_text().lower()
        assert "never optimize" in text or "never optimise" in text
        assert "instead of" in text


def _load_aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_phase3_overall", ROOT / "scripts" / "aggregate_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_phase3_overall"] = module
    spec.loader.exec_module(module)
    return module


aggregate_phase3 = _load_aggregate()


def _overall_csv(path: Path, records, legacy: bool = False):
    """``legacy=True`` writes the pre-2026-08-23 schema (no pooled column), so
    the back-compat path is exercised by the same helper rather than by a
    hand-rolled second one that could drift from it."""
    fields = ["diarization", "dilate_ms", "scene", "speaker", "system", "m",
              "si_sdr", "deflation_index", "n_accepted_before", "refine_rounds",
              "cluster", "n_clusters"]
    if not legacy:
        fields[7:7] = ["si_sdr_pooled", "level_error_db"]
    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {**{f: "" for f in fields}, **r}
            w.writerow({f: row[f] for f in fields})


def _rec(arm, scene, speaker, si_sdr, dilate_ms=0.0, system="no_recursion"):
    # The pooled value is what the gap table reads by default; give it the same
    # number so these pairing tests stay about pairing.
    return {"diarization": arm, "dilate_ms": dilate_ms, "scene": scene,
            "speaker": speaker, "system": system, "m": 3, "si_sdr": si_sdr,
            "si_sdr_pooled": si_sdr, "level_error_db": 0.0, "refine_rounds": 0}


class TestAggregationReadsTheSiblingFile:
    def test_a_missing_overall_csv_degrades_gracefully(self, tmp_path):
        """Every Phase 3 CSV written before 2026-08-20 lacks this file.

        Those must still aggregate -- a hard failure would make the committed
        Stage A results unreadable by the current tooling.
        """
        rows = aggregate_phase3._load_overall([tmp_path / "nothing.csv"])
        assert rows == []
        rendered = "\n".join(aggregate_phase3._overall_table([], "real", "oracle"))
        assert "no `_overall.csv`" in rendered

    def test_it_pairs_arms_on_the_sibling_file(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        csv_path.write_text("placeholder\n")
        _overall_csv(tmp_path / "run_overall.csv", [
            _rec("oracle", "s1", "a", 5.0), _rec("real", "s1", "a", 2.0),
            _rec("oracle", "s2", "b", 7.0), _rec("real", "s2", "b", 4.0),
        ])
        rows = aggregate_phase3._load_overall([csv_path])
        assert len(rows) == 4

        rendered = "\n".join(aggregate_phase3._overall_table(rows, "real", "oracle"))
        assert "| no_recursion | 2 | -3.00 |" in rendered, rendered

    def test_it_never_pairs_across_dilation_values(self, tmp_path):
        """A swept file holds several pipelines. Pairing `real` at 400 ms against
        `oracle` at 0 ms would measure the knob and the diarizer together and
        report the sum as the diarizer's cost."""
        csv_path = tmp_path / "run.csv"
        csv_path.write_text("placeholder\n")
        _overall_csv(tmp_path / "run_overall.csv", [
            _rec("oracle", "s1", "a", 5.0, dilate_ms=0.0),
            _rec("real", "s1", "a", 2.0, dilate_ms=400.0),
        ])
        rows = aggregate_phase3._load_overall([csv_path])
        rendered = "\n".join(aggregate_phase3._overall_table(rows, "real", "oracle"))
        assert "no paired overall rows" in rendered, rendered

    def test_matched_dilations_do_pair(self, tmp_path):
        """The complement, so the previous test cannot pass by pairing nothing."""
        csv_path = tmp_path / "run.csv"
        csv_path.write_text("placeholder\n")
        _overall_csv(tmp_path / "run_overall.csv", [
            _rec("oracle", "s1", "a", 5.0, dilate_ms=400.0),
            _rec("real", "s1", "a", 2.0, dilate_ms=400.0),
        ])
        rows = aggregate_phase3._load_overall([csv_path])
        rendered = "\n".join(aggregate_phase3._overall_table(rows, "real", "oracle"))
        assert "| no_recursion | 1 | -3.00 |" in rendered, rendered

    def test_the_section_is_not_offered_as_a_replacement(self):
        """§6.4 forbids reporting an aggregate INSTEAD of stratification, and the
        report has to say so where a reader will actually see it -- in the
        rendered markdown, not only in a source comment."""
        source = (ROOT / "scripts" / "aggregate_phase3.py").read_text()
        i = source.index("### overall (un-stratified")
        section = source[i:i + 600]
        assert "INSTEAD of them" in section
        assert "never optimize against it" in section


class TestLegacyRunsWithoutTheNewColumns:
    """Every CSV written before 2026-08-23 has `si_sdr` and nothing else. Those
    must still aggregate -- and must say so, rather than rendering an empty
    table that reads as "the metric found nothing"."""

    def test_the_pooled_gap_table_says_the_run_predates_it(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        csv_path.write_text("placeholder\n")
        _overall_csv(tmp_path / "run_overall.csv", [
            _rec("oracle", "s1", "a", 5.0), _rec("real", "s1", "a", 2.0),
        ], legacy=True)
        rows = aggregate_phase3._load_overall([csv_path])
        assert rows and "si_sdr_pooled" not in rows[0], (
            "the key must be DROPPED, not defaulted -- a default makes 'never "
            "measured' look like 'measured as zero'"
        )
        rendered = "\n".join(
            aggregate_phase3._overall_table(rows, "real", "oracle", "si_sdr_pooled")
        )
        assert "predates it" in rendered, rendered

    def test_the_whole_track_gap_table_still_pairs(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        csv_path.write_text("placeholder\n")
        _overall_csv(tmp_path / "run_overall.csv", [
            _rec("oracle", "s1", "a", 5.0), _rec("real", "s1", "a", 2.0),
        ], legacy=True)
        rows = aggregate_phase3._load_overall([csv_path])
        rendered = "\n".join(
            aggregate_phase3._overall_table(rows, "real", "oracle", "si_sdr")
        )
        assert "| no_recursion | 1 | -3.00 |" in rendered, rendered

    def test_the_md_section_names_the_missing_column(self, tmp_path):
        legacy_row = {
            "diarization": "oracle", "dilate_ms": 0.0, "scene": "s",
            "speaker": "s1", "system": "no_recursion", "m": 3, "si_sdr": 1.0,
            "deflation_index": None, "n_accepted_before": None,
            "refine_rounds": 0, "cluster": "s1", "n_clusters": 3,
        }
        run_phase3._write_results(
            [], [], [], [legacy_row], tmp_path, "stem", ["oracle"], n_scenes=1,
        )
        text = (tmp_path / "stem.md").read_text()
        assert "no `si_sdr_pooled` in this run" in text


class TestTheLevelColumnIsNotClipped:
    """`clip_score` caps at ±50 dB to tame `±inf` SI-SDR. A level DISAGREEMENT
    is not an SI-SDR and has no such pathology, so capping it would report a
    60 dB level error as 50 -- a silently truncated diagnostic, in the column
    that exists precisely to stop level error hiding."""

    def _md(self, tmp_path, level_error_db):
        run_phase3._write_results(
            [], [], [], [{
                "diarization": "oracle", "dilate_ms": 0.0, "scene": "s",
                "speaker": "s1", "system": "no_recursion", "m": 3, "si_sdr": 1.0,
                "si_sdr_pooled": 1.0, "level_error_db": level_error_db,
                "deflation_index": None, "n_accepted_before": None,
                "refine_rounds": 0, "cluster": "s1", "n_clusters": 3,
            }], tmp_path, "stem", ["oracle"], n_scenes=1,
        )
        return (tmp_path / "stem.md").read_text()

    def test_a_large_level_error_is_reported_in_full(self, tmp_path):
        text = self._md(tmp_path, 60.0)
        assert "60.00" in text, "level error was clipped to the SI-SDR cap"

    def test_si_sdr_itself_is_still_clipped(self, tmp_path):
        """The complement, so the test above cannot pass by disabling clipping
        everywhere -- ±inf must still be tamed in the SI-SDR tables."""
        run_phase3._write_results(
            [], [], [], [{
                "diarization": "oracle", "dilate_ms": 0.0, "scene": "s",
                "speaker": "s1", "system": "no_recursion", "m": 3,
                "si_sdr": float("inf"), "si_sdr_pooled": float("inf"),
                "level_error_db": 0.0, "deflation_index": None,
                "n_accepted_before": None, "refine_rounds": 0,
                "cluster": "s1", "n_clusters": 3,
            }], tmp_path, "stem", ["oracle"], n_scenes=1,
        )
        assert "50.00" in (tmp_path / "stem.md").read_text()

    def test_an_unmeasurable_level_error_is_dropped_not_zeroed(self, tmp_path):
        """`nan` means fewer than two depths were scoreable, i.e. nothing could
        disagree. Rendering that as 0.00 would claim a measurement was made."""
        text = self._md(tmp_path, float("nan"))
        i = text.index("### Level disagreement")
        assert "0.00" not in text[i:i + 400], text[i:i + 400]
