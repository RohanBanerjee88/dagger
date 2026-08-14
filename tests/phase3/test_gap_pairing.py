"""The oracle-vs-real gap must actually pair. This is a shipped-bug regression test.

During implementation, score rows keyed ``speaker`` on the *cluster id*, so real
rows read ``SPEAKER_00`` while oracle rows read ``s1``. Every paired lookup
missed, and `aggregate_phase3.py` would have written a gap table that was
**empty** — with nothing raising, nothing warning, and the run reporting success.

That is the exact shape of every reporting defect this project has shipped: the
number was wrong and nothing failed. So the property gets a test rather than a
comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.oracle import OracleDiarizer  # noqa: E402
from dagger.diarize.regions import scene_regions  # noqa: E402
from dagger.eval.systems import _score_targets  # noqa: E402
from dagger.metrics.phase2_scores import paired_by_field  # noqa: E402


class TestRowsAreLabelledForPairing:
    def test_oracle_and_real_rows_share_a_speaker_vocabulary(
        self, three_speaker_scheduled_scene
    ):
        """Without this, no oracle-vs-real difference can ever be computed."""
        scene = three_speaker_scheduled_scene

        oracle = _score_targets(scene, scene_regions(scene, OracleDiarizer()))
        real = _score_targets(scene, scene_regions(scene, FakeDiarizer()))

        oracle_labels = {label for label, _ in oracle if label is not None}
        real_labels = {label for label, _ in real if label is not None}
        assert oracle_labels == real_labels == set(scene.speakers)

    def test_no_row_is_labelled_with_a_cluster_id(self, three_speaker_scheduled_scene):
        scene = three_speaker_scheduled_scene
        real = _score_targets(scene, scene_regions(scene, FakeDiarizer()))
        assert all(
            label is None or not label.startswith("SPEAKER_") for label, _ in real
        )


class TestPairedByField:
    """`paired_by_field` is what turns two arms into a gap number."""

    def _rows(self, oracle_scores, real_scores):
        rows = []
        for arm, scores in (("oracle", oracle_scores), ("real", real_scores)):
            for (speaker, depth), value in scores.items():
                rows.append({
                    "source": "f.csv", "scene": "sc", "speaker": speaker,
                    "depth": depth, "system": "no_recursion",
                    "diarization": arm, "si_sdr": value,
                })
        return rows

    def test_differences_are_taken_on_matched_rows(self):
        rows = self._rows(
            {("s1", 2): 5.0, ("s2", 2): 3.0},
            {("s1", 2): 4.0, ("s2", 2): 1.0},
        )
        diffs = paired_by_field(rows, "diarization", "real", "oracle")
        assert sorted(diffs) == [-2.0, -1.0]

    def test_unmatched_rows_are_excluded_not_zero_filled(self):
        """A speaker present in one arm only contributes nothing.

        Treating it as a 0 dB difference would quietly dilute the gap toward zero
        in exactly the scenes where the diarizer failed hardest.
        """
        rows = self._rows({("s1", 2): 5.0, ("s2", 2): 3.0}, {("s1", 2): 4.0})
        assert paired_by_field(rows, "diarization", "real", "oracle") == [-1.0]

    def test_systems_are_not_conflated(self):
        """The key includes `system`: this compares one system across arms, not
        two systems against each other."""
        rows = self._rows({("s1", 2): 5.0}, {("s1", 2): 4.0})
        rows.append({
            "source": "f.csv", "scene": "sc", "speaker": "s1", "depth": 2,
            "system": "coarse_to_fine", "diarization": "real", "si_sdr": 99.0,
        })
        assert paired_by_field(rows, "diarization", "real", "oracle") == [-1.0]

    def test_depth_is_not_conflated(self):
        rows = self._rows(
            {("s1", 2): 5.0, ("s1", 3): 1.0},
            {("s1", 2): 4.0, ("s1", 3): 0.0},
        )
        assert sorted(paired_by_field(rows, "diarization", "real", "oracle")) == [-1.0, -1.0]

    def test_empty_when_an_arm_is_missing(self):
        rows = self._rows({("s1", 2): 5.0}, {})
        assert paired_by_field(rows, "diarization", "real", "oracle") == []
