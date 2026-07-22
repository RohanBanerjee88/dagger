"""Tests for dagger.diarize.oracle.overlap_depth (CLAUDE.md §5 Phase 2)."""

from __future__ import annotations

import numpy as np

from dagger.diarize.oracle import overlap_depth, solo_overlap_regions


def test_overlap_depth_matches_manual_counts():
    activity = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    depth = overlap_depth(activity)
    np.testing.assert_array_equal(depth, [1, 2, 3, 2, 1])


def test_overlap_depth_all_silent_is_zero():
    activity = np.zeros((3, 4))
    depth = overlap_depth(activity)
    np.testing.assert_array_equal(depth, [0, 0, 0, 0])


def test_overlap_depth_consistent_with_solo_overlap_regions():
    rng = np.random.default_rng(0)
    activity = (rng.random((4, 50)) > 0.6).astype(np.float64)
    depth = overlap_depth(activity)
    solo, overlap = solo_overlap_regions(activity)
    np.testing.assert_array_equal(overlap, (depth >= 2).astype(np.float64))
    np.testing.assert_array_equal(solo.sum(axis=0), (depth == 1).astype(np.float64))
