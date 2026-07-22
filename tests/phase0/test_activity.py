"""Phase 0: oracle activity derived from clean sources (dagger.data.activity)."""

from __future__ import annotations

import numpy as np
import pytest

from dagger.data.activity import active_mask, segments_from_placement, segments_from_sources

SAMPLE_RATE = 8000


class TestActiveMask:
    def test_silent_source_is_never_active(self):
        mask = active_mask(np.zeros(1000), SAMPLE_RATE)
        assert not mask.any()

    def test_loud_tone_is_fully_active(self):
        t = np.arange(1000) / SAMPLE_RATE
        tone = 0.8 * np.sin(2 * np.pi * 200 * t)
        mask = active_mask(tone, SAMPLE_RATE, min_dur_ms=1.0)
        # A steady tone should be active almost everywhere (edges may dip due
        # to windowed energy smoothing).
        assert mask.mean() > 0.9

    def test_isolated_clip_amid_silence_is_localized(self):
        n = 4000
        source = np.zeros(n)
        t = np.arange(1000) / SAMPLE_RATE
        source[1000:2000] = 0.8 * np.sin(2 * np.pi * 200 * t)
        mask = active_mask(source, SAMPLE_RATE, min_dur_ms=5.0)
        assert not mask[:900].any()
        assert mask[1100:1900].all()
        assert not mask[2100:].any()

    def test_short_blips_are_dropped(self):
        n = 2000
        source = np.zeros(n)
        source[1000] = 1.0  # a single-sample blip
        mask = active_mask(source, SAMPLE_RATE, min_dur_ms=50.0)
        assert not mask.any()


class TestSegmentsFromSources:
    def test_matches_number_of_speakers(self):
        n = 2000
        sources = np.zeros((2, n))
        t = np.arange(n) / SAMPLE_RATE
        sources[0] = 0.8 * np.sin(2 * np.pi * 200 * t)
        segments = segments_from_sources(sources, ["a", "b"], SAMPLE_RATE, min_dur_ms=5.0)
        speakers_seen = {seg.speaker for seg in segments}
        assert speakers_seen == {"a"}  # b is silent -> no segments

    def test_mismatched_speaker_count_raises(self):
        sources = np.zeros((2, 100))
        with pytest.raises(ValueError):
            segments_from_sources(sources, ["only_one"], SAMPLE_RATE)


class TestSegmentsFromPlacement:
    def test_exact_windows_no_leakage(self):
        segments = segments_from_placement(
            offsets=[0, 500], lengths=[1000, 800], speakers=["a", "b"], sample_rate=SAMPLE_RATE
        )
        assert len(segments) == 2
        seg_a, seg_b = segments
        assert seg_a.speaker == "a"
        assert seg_a.start == pytest.approx(0.0)
        assert seg_a.duration == pytest.approx(1000 / SAMPLE_RATE)
        assert seg_b.start == pytest.approx(500 / SAMPLE_RATE)
        assert seg_b.duration == pytest.approx(800 / SAMPLE_RATE)

    def test_zero_length_speaker_skipped(self):
        segments = segments_from_placement(
            offsets=[0, 100], lengths=[1000, 0], speakers=["a", "b"], sample_rate=SAMPLE_RATE
        )
        assert len(segments) == 1
        assert segments[0].speaker == "a"

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            segments_from_placement(offsets=[0], lengths=[1000, 2000], speakers=["a", "b"], sample_rate=SAMPLE_RATE)
