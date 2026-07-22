"""Phase 0: oracle diarization plumbing (dagger.diarize.oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import Provenance
from dagger.diarize.oracle import (
    Segment,
    activity_matrix,
    overlap_mixture,
    read_rttm,
    solo_overlap_regions,
    speaker_order,
)

RTTM_TEXT = """\
;; a comment line
SPEAKER file1 1 0.0 2.0 <NA> <NA> spk_a <NA> <NA>
SPEAKER file1 1 1.5 1.0 <NA> <NA> spk_b <NA> <NA>
# another comment
SPEAKER file1 1 3.0 0.5 <NA> <NA> spk_a <NA> <NA>
"""


class TestSegment:
    def test_end_is_start_plus_duration(self):
        seg = Segment(speaker="a", start=1.0, duration=0.5)
        assert seg.end == pytest.approx(1.5)


class TestReadRttm:
    def test_parses_speaker_lines_and_skips_comments(self, tmp_path):
        path = tmp_path / "test.rttm"
        path.write_text(RTTM_TEXT)
        segments = read_rttm(str(path))
        assert len(segments) == 3
        assert segments[0] == Segment(speaker="spk_a", start=0.0, duration=2.0)
        assert segments[1] == Segment(speaker="spk_b", start=1.5, duration=1.0)
        assert segments[2] == Segment(speaker="spk_a", start=3.0, duration=0.5)

    def test_empty_file_yields_no_segments(self, tmp_path):
        path = tmp_path / "empty.rttm"
        path.write_text("")
        assert read_rttm(str(path)) == []


class TestSpeakerOrder:
    def test_first_appearance_order_deduplicated(self):
        segments = [
            Segment("b", 0.0, 1.0),
            Segment("a", 1.0, 1.0),
            Segment("b", 2.0, 1.0),
            Segment("c", 3.0, 1.0),
        ]
        assert speaker_order(segments) == ["b", "a", "c"]


class TestActivityMatrix:
    def test_shape_and_alignment(self):
        segments = [Segment("a", 0.0, 1.0), Segment("b", 0.5, 0.5)]
        activity, speakers = activity_matrix(segments, num_samples=1000, sample_rate=1000)
        assert activity.shape == (2, 1000)
        assert speakers == ["a", "b"]
        assert activity[0, 0:1000].all()
        assert activity[1, 500:1000].all()
        assert not activity[1, 0:500].any()

    def test_explicit_speaker_order_is_honored(self):
        segments = [Segment("a", 0.0, 1.0), Segment("b", 0.0, 1.0)]
        activity, speakers = activity_matrix(
            segments, num_samples=100, sample_rate=100, speakers=["b", "a"]
        )
        assert speakers == ["b", "a"]

    def test_out_of_range_segment_is_clipped(self):
        segments = [Segment("a", 0.9, 1.0)]  # extends past num_samples=100 @ 100 Hz
        activity, _ = activity_matrix(segments, num_samples=100, sample_rate=100)
        assert activity.shape == (1, 100)
        assert activity[0, 90:100].all()

    def test_unknown_speaker_in_segments_is_ignored_when_speakers_given(self):
        segments = [Segment("a", 0.0, 1.0), Segment("ghost", 0.0, 1.0)]
        activity, speakers = activity_matrix(
            segments, num_samples=100, sample_rate=100, speakers=["a"]
        )
        assert activity.shape == (1, 100)


class TestSoloOverlapRegions:
    def test_solo_and_overlap_partition_correctly(self):
        # 3 samples: [solo-a, overlap-ab, silence]
        activity = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        solo, overlap = solo_overlap_regions(activity)
        np.testing.assert_array_equal(overlap, [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(solo[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(solo[1], [0.0, 0.0, 0.0])

    def test_no_overlap_when_never_concurrent(self):
        activity = np.array([[1.0, 0.0], [0.0, 1.0]])
        solo, overlap = solo_overlap_regions(activity)
        assert not overlap.any()
        np.testing.assert_array_equal(solo, activity)

    def test_triple_overlap_is_still_overlap_not_solo(self):
        activity = np.ones((3, 5))
        solo, overlap = solo_overlap_regions(activity)
        assert overlap.all()
        assert not solo.any()


class TestOverlapMixture:
    def test_returns_original_mixture_provenance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        overlap = np.array([0.0, 1.0, 1.0, 0.0])
        x_O = overlap_mixture(x, overlap)
        assert x_O.provenance is Provenance.ORIGINAL_MIXTURE
        np.testing.assert_array_equal(x_O.samples, [0.0, 2.0, 3.0, 0.0])

    def test_wraps_raw_ndarray_automatically(self):
        x_O = overlap_mixture(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
        assert x_O.is_original_mixture
