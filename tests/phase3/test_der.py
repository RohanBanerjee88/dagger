"""DER: hand-computed fixtures where every component is known by construction.

A metric that is only checked against itself proves nothing, so each case below
states the arithmetic in its docstring and asserts the exact sample counts, not
just that the number "looks reasonable".
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.metrics.der import diarization_error_rate


def _activity(*rows: str) -> np.ndarray:
    return np.array([[float(c) for c in row] for row in rows])


class TestPerfectAndEmpty:
    def test_identical_activity_scores_zero(self):
        ref = _activity("11110000", "00001111")
        result = diarization_error_rate(ref, ref)
        assert result.der == 0.0
        assert (result.miss, result.false_alarm, result.confusion) == (0, 0, 0)

    def test_permuted_rows_still_score_zero(self):
        """Row order is arbitrary; the optimal mapping must absorb it."""
        ref = _activity("11110000", "00001111")
        assert diarization_error_rate(ref, ref[::-1]).der == 0.0

    def test_predicting_nothing_is_all_miss(self):
        ref = _activity("11110000", "00001111")
        pred = np.zeros((1, 8))
        result = diarization_error_rate(ref, pred)
        assert result.miss == 8  # every reference speech sample
        assert result.false_alarm == 0
        assert result.der == 1.0


class TestComponents:
    def test_pure_miss(self):
        """Reference has 4 samples of speech, prediction covers only 2.

        miss = 2, FA = 0, confusion = 0, total = 4 -> DER = 0.5
        """
        ref = _activity("11110000")
        pred = _activity("11000000")
        result = diarization_error_rate(ref, pred)
        assert (result.miss, result.false_alarm, result.confusion) == (2, 0, 0)
        assert result.total_speech == 4
        assert result.der == pytest.approx(0.5)

    def test_pure_false_alarm(self):
        """Prediction speaks for 2 samples where the reference is silent.

        miss = 0, FA = 2, confusion = 0, total = 4 -> DER = 0.5
        """
        ref = _activity("11110000")
        pred = _activity("11111100")
        result = diarization_error_rate(ref, pred)
        assert (result.miss, result.false_alarm, result.confusion) == (0, 2, 0)
        assert result.der == pytest.approx(0.5)

    def test_pure_confusion(self):
        """Both agree someone speaks throughout, but attribute 2 samples wrongly.

        Reference: A speaks 0-3, B speaks 4-7.
        Prediction: the matched pair agrees on 0-1 and 6-7 but swaps 2-5.
        Counts are equal at every sample, so miss = FA = 0 and everything
        mis-attributed lands in confusion.
        """
        ref = _activity("11110000", "00001111")
        pred = _activity("11000011", "00111100")
        result = diarization_error_rate(ref, pred)
        assert result.miss == 0
        assert result.false_alarm == 0
        assert result.confusion == 4
        assert result.total_speech == 8
        assert result.der == pytest.approx(0.5)

    def test_der_can_exceed_one(self):
        """False alarms are unbounded, so DER > 1 is legal and must not clamp."""
        ref = _activity("10000000")
        pred = _activity("11111111")
        result = diarization_error_rate(ref, pred)
        assert result.der > 1.0


class TestOverlapAware:
    def test_missing_one_speaker_in_an_overlap_counts_as_a_miss(self):
        """Two speakers overlap for 4 samples; the diarizer reports only one.

        Per-frame reference count is 2, predicted 1, so miss = 4 -- a
        single-label-per-frame DER would score this frame as "correct" and hide
        exactly the failure this project cares about most.
        """
        ref = _activity("1111", "1111")
        pred = _activity("1111")
        result = diarization_error_rate(ref, pred)
        assert result.miss == 4
        assert result.total_speech == 8
        assert result.der == pytest.approx(0.5)

    def test_overlap_recall_full_when_overlap_detected(self):
        ref = _activity("1111", "0111")
        result = diarization_error_rate(ref, ref)
        assert result.overlap_recall == pytest.approx(1.0)

    def test_overlap_recall_zero_when_overlap_flattened(self):
        """The diarizer sees speech everywhere but never two people at once."""
        ref = _activity("1111", "0111")
        pred = _activity("1111")
        result = diarization_error_rate(ref, pred)
        assert result.overlap_recall == pytest.approx(0.0)

    def test_overlap_recall_is_nan_without_reference_overlap(self):
        ref = _activity("1100", "0011")
        assert np.isnan(diarization_error_rate(ref, ref).overlap_recall)


class TestSpeakerCounts:
    def test_missed_and_spurious_speakers_are_reported(self):
        ref = _activity("110000", "001100", "000011")
        pred = _activity("110000", "001100")
        result = diarization_error_rate(ref, pred)
        assert result.n_ref == 3
        assert result.n_pred == 2
        assert result.n_missed_speakers == 1
        assert result.n_spurious_clusters == 0


class TestAgainstPyannoteMetrics:
    """Cross-check the frame-level implementation against the reference library."""

    def test_matches_pyannote_metrics(self):
        pytest.importorskip("pyannote.metrics")
        from pyannote.core import Annotation
        from pyannote.core import Segment as PSegment
        from pyannote.metrics.diarization import DiarizationErrorRate

        sample_rate = 100  # 1 sample = 10 ms, so boundaries are exact

        def to_annotation(activity: np.ndarray, prefix: str) -> Annotation:
            ann = Annotation()
            for i, row in enumerate(activity):
                padded = np.concatenate([[0], row.astype(int), [0]])
                edges = np.flatnonzero(np.diff(padded))
                for start, end in zip(edges[0::2], edges[1::2]):
                    ann[PSegment(start / sample_rate, end / sample_rate)] = f"{prefix}{i}"
            return ann

        ref = _activity(
            "111111000000000000",
            "000000111111110000",
            "000000000000111111",
        )
        pred = _activity(
            "111110000000000000",
            "000000111100000000",
            "000000000011111111",
        )

        ours = diarization_error_rate(ref, pred).der
        theirs = DiarizationErrorRate()(to_annotation(ref, "R"), to_annotation(pred, "P"))
        assert ours == pytest.approx(theirs, abs=1e-6)
