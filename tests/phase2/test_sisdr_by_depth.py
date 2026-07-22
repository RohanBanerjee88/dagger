"""Tests for dagger.metrics.sisdr.si_sdr_by_depth (CLAUDE.md §5 Phase 2)."""

from __future__ import annotations

import numpy as np

from dagger.metrics.sisdr import si_sdr, si_sdr_by_depth


def test_buckets_match_manual_regionwise_scores():
    rng = np.random.default_rng(0)
    target = rng.normal(size=100)
    estimate = target + 0.01 * rng.normal(size=100)
    depth = np.array([1] * 40 + [2] * 30 + [3] * 30)

    result = si_sdr_by_depth(estimate, target, depth)

    assert set(result.keys()) == {1, 2, 3}
    for k in (1, 2, 3):
        mask = depth == k
        expected = si_sdr(estimate[mask], target[mask])
        assert result[k] == expected


def test_depth_zero_is_omitted():
    target = np.array([1.0, 2.0, 0.0, 0.0])
    estimate = np.array([1.0, 2.0, 0.0, 0.0])
    depth = np.array([1, 1, 0, 0])
    result = si_sdr_by_depth(estimate, target, depth)
    assert set(result.keys()) == {1}


def test_no_scoreable_depth_returns_empty_dict():
    target = np.zeros(5)
    estimate = np.zeros(5)
    depth = np.zeros(5, dtype=int)
    assert si_sdr_by_depth(estimate, target, depth) == {}
