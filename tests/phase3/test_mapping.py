"""Cluster->speaker mapping: the mis-attribution hazard, pinned.

Nothing in the pre-Phase-3 code would fail on a permuted activity matrix --
reconstruction indexes outputs by row and Phase 2 scored ``outputs[i]`` against
``sources[i]`` with no permutation search anywhere. It would simply report wrong
numbers that look plausible. These tests are what makes that impossible now.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.diarize.mapping import map_clusters_to_speakers


def _activity(*rows: str) -> np.ndarray:
    """Build an activity matrix from strings like ``"1100"``."""
    return np.array([[float(c) for c in row] for row in rows])


class TestRecoversPermutation:
    def test_identity_when_already_aligned(self):
        ref = _activity("1100", "0011")
        mapping = map_clusters_to_speakers(ref, ref)
        assert mapping.cluster_to_ref == (0, 1)
        assert mapping.ref_to_cluster == (0, 1)

    def test_reversed_rows_are_recovered(self):
        ref = _activity("1100", "0011")
        pred = ref[::-1]
        mapping = map_clusters_to_speakers(pred, ref)
        # Predicted row 0 is reference row 1 and vice versa.
        assert mapping.cluster_to_ref == (1, 0)
        assert mapping.unmatched_refs == ()
        assert mapping.unmatched_clusters == ()

    def test_three_way_permutation(self):
        ref = _activity("111000000", "000111000", "000000111")
        order = [2, 0, 1]
        pred = ref[order]
        mapping = map_clusters_to_speakers(pred, ref)
        assert mapping.cluster_to_ref == tuple(order)

    def test_best_overlap_wins_not_row_position(self):
        """The match is by shared frames, not by index proximity."""
        ref = _activity("11110000", "00001111")
        # Predicted row 0 mostly agrees with reference row 1.
        pred = _activity("00001110", "11100000")
        mapping = map_clusters_to_speakers(pred, ref)
        assert mapping.cluster_to_ref == (1, 0)


class TestCardinalityMismatchIsAResult:
    """k != m is a diarization failure to be PRICED, not an error to swallow."""

    def test_fewer_clusters_than_speakers_reports_a_missed_speaker(self):
        ref = _activity("110000", "001100", "000011")
        pred = _activity("110000", "001100")  # third speaker never found
        mapping = map_clusters_to_speakers(pred, ref)
        assert mapping.unmatched_refs == (2,)
        assert mapping.unmatched_clusters == ()
        assert mapping.ref_to_cluster[2] is None

    def test_more_clusters_than_speakers_reports_a_spurious_one(self):
        ref = _activity("110000", "001100")
        pred = _activity("110000", "001100", "000011")  # invented a speaker
        mapping = map_clusters_to_speakers(pred, ref)
        assert mapping.unmatched_clusters == (2,)
        assert mapping.unmatched_refs == ()
        assert mapping.cluster_to_ref[2] is None

    def test_zero_overlap_pair_is_a_miss_plus_a_spurious_cluster(self):
        """A pairing with no shared speech is not evidence of a correspondence.

        The assignment step will happily pair disjoint rows just to complete the
        matching. Reporting that as a match would claim the diarizer found a
        speaker it demonstrably did not.
        """
        ref = _activity("110000")
        pred = _activity("001100")
        mapping = map_clusters_to_speakers(pred, ref)
        assert mapping.cluster_to_ref == (None,)
        assert mapping.unmatched_refs == (0,)
        assert mapping.unmatched_clusters == (0,)


class TestValidation:
    def test_length_mismatch_raises_rather_than_broadcasting(self):
        with pytest.raises(ValueError, match="same number of"):
            map_clusters_to_speakers(_activity("110"), _activity("1100"))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match=r"\[rows, T\]"):
            map_clusters_to_speakers(np.array([1.0, 0.0]), _activity("10"))


class TestFallbackMatchesScipy:
    """The no-scipy brute force must agree with the scipy path exactly."""

    def test_brute_force_agrees_with_scipy(self, monkeypatch):
        import builtins

        ref = _activity("11110000", "00111100", "00001111")
        pred = ref[[1, 2, 0]]
        expected = map_clusters_to_speakers(pred, ref).cluster_to_ref

        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise ImportError("scipy disabled for this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        assert map_clusters_to_speakers(pred, ref).cluster_to_ref == expected
