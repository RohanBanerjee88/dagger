"""Tests for the Phase 2 scene scheduler (CLAUDE.md §5 Phase 2 "heads-up"):
every speaker must get a guaranteed solo window AND the scene must reach a
genuine depth == num_speakers overlap -- something the Phase 0/1 chain-based
``stagger_offsets`` structurally cannot produce at the same time.
"""

from __future__ import annotations

import numpy as np

from dagger.data.activity import segments_from_chunks
from dagger.data.mixing import mix_scheduled_sources, schedule_solo_then_overlap
from dagger.diarize.oracle import activity_matrix, overlap_depth, solo_overlap_regions

SAMPLE_RATE = 8000


def _build_scene(lengths: list[int], min_solo: int, freqs: list[float] | None = None):
    num = len(lengths)
    speakers = [f"s{i}" for i in range(num)]
    freqs = freqs or [220.0 * (i + 1) for i in range(num)]
    chunks = schedule_solo_then_overlap(lengths, min_solo)

    sources_raw = []
    for i, length in enumerate(lengths):
        t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
        sources_raw.append(0.5 * np.sin(2.0 * np.pi * freqs[i] * t))

    sources, mixture = mix_scheduled_sources(sources_raw, chunks)
    segments = segments_from_chunks(chunks, speakers, SAMPLE_RATE)
    activity, speakers = activity_matrix(
        segments, num_samples=mixture.shape[0], sample_rate=SAMPLE_RATE, speakers=speakers
    )
    solo, overlap = solo_overlap_regions(activity)
    depth = overlap_depth(activity)
    return {
        "chunks": chunks,
        "mixture": mixture,
        "sources": sources,
        "activity": activity,
        "solo": solo,
        "overlap": overlap,
        "depth": depth,
    }


class TestScheduleSoloThenOverlap:
    def test_every_speaker_gets_its_own_exclusive_solo_slot(self):
        chunks = schedule_solo_then_overlap(lengths=[4000, 4000, 4000], min_solo=800)
        # solo slots are (i*min_solo, 0, min(min_solo, length)) and non-overlapping
        # across speakers by construction.
        for i, speaker_chunks in enumerate(chunks):
            solo_offset, src_start, solo_len = speaker_chunks[0]
            assert solo_offset == i * 800
            assert src_start == 0
            assert solo_len == 800

    def test_short_utterance_has_no_overlap_tail(self):
        chunks = schedule_solo_then_overlap(lengths=[500, 4000], min_solo=800)
        assert len(chunks[0]) == 1  # fully consumed by the solo slot
        assert chunks[0][0] == (0, 0, 500)
        assert len(chunks[1]) == 2

    def test_overlap_tails_share_the_same_absolute_offset(self):
        chunks = schedule_solo_then_overlap(lengths=[4000, 3000, 5000], min_solo=800)
        tail_offsets = {c[-1][0] for c in chunks if len(c) == 2}
        assert tail_offsets == {3 * 800}

    def test_rejects_non_positive_min_solo(self):
        import pytest

        with pytest.raises(ValueError):
            schedule_solo_then_overlap(lengths=[100], min_solo=0)

    def test_deterministic(self):
        a = schedule_solo_then_overlap(lengths=[4000, 3000, 5000], min_solo=800)
        b = schedule_solo_then_overlap(lengths=[4000, 3000, 5000], min_solo=800)
        assert a == b


class TestSceneReachesDepth3WithSoloGuaranteed:
    def test_three_speakers_each_get_solo_time(self):
        scene = _build_scene(lengths=[4000, 4000, 4000], min_solo=800)
        for i in range(3):
            solo_samples = scene["solo"][i].sum()
            assert solo_samples >= 800 - 1  # allow off-by-one from rounding-free construction

    def test_reaches_full_depth_when_all_tails_overlap(self):
        scene = _build_scene(lengths=[4000, 4000, 4000], min_solo=800)
        assert scene["depth"].max() == 3

    def test_depth_tapers_as_shorter_tails_end(self):
        # tails: s0 -> 4000-800=3200, s1 -> 3000-800=2200, s2 -> 5000-800=4200
        scene = _build_scene(lengths=[4000, 3000, 5000], min_solo=800)
        depth = scene["depth"]
        assert depth.max() == 3
        # after the two shorter tails end, only the longest-tailed speaker remains (depth 1)
        assert depth[-1] == 1

    def test_utterance_shorter_than_min_solo_has_no_tail_contribution(self):
        scene = _build_scene(lengths=[500, 4000, 4000], min_solo=800)
        # s0 never reaches the overlap zone, so max depth there is 2, not 3.
        overlap_start = 3 * 800
        assert scene["depth"][overlap_start:].max() == 2
